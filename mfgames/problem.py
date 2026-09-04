"""
Mean Field Games Problem Solvers

This module provides object-oriented solvers for Mean Field Game (MFG) systems,
which model large populations of rational agents making optimal decisions while
influenced by the aggregate behavior of the population.

The module implements:
1. MFGSolver: Single-population systems with static or dynamic goals (pursuit-evasion)
2. MFG2PopSolver: Coupled two-population competitive/cooperative systems

Mathematical Foundation
-----------------------
Each solver solves a coupled system of PDEs via Picard iteration:
- Hamilton-Jacobi-Bellman (HJB) equation: Backward in time, computes optimal control (value function u)
- Kolmogorov-Fokker-Planck (KFP) equation: Forward in time, evolves population density (m)

MFGSolver consolidates time-stepping loops for both static exit doors
and dynamic target goals into a unified execution path. The Picard iteration alternates
between solving HJB given density m, then solving KFP given value u, with under-relaxation
(thetaUM parameter) to stabilize convergence.

Goal Capacity Saturation
------------------------
For mobile landing platforms or capacity-limited exits, the cumulative density absorbed
by each goal region is integrated forward in space and time:
    C_g(t) = ∫₀ᵗ ∫_{\\Omega_g(s)} m(s, x) dx ds

When C_g(t) reaches capacity threshold C_max, the exit region deactivates (door_mask = 0),
switching boundary conditions from Dirichlet mass absorption (u=0, m=0) to interior transport.
Optionally, for soft potential goals, attraction cost weights decay proportionally as capacity fills.
"""
import time
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
from mfgames.evasion import Goal
from mfgames.numerics import (
    compute_FP_matrix_entries,
    getFnU_2D,
    compute_HJB_matrix_entries,
    compute_FP_matrix_entries_2Pop,
    getFnU_2D_2Pop,
    compute_HJB_matrix_entries_2Pop,
)


class MFGSolver:
    """
    Mean Field Game solver for single-population systems with goal-seeking behavior.

    This solver handles drone swarms navigating toward static, dynamic, or capacity-constrained
    goals in a 2D spatial domain with obstacles. It handles static exit locations,
    moving doors, reactive target goals, and finite-capacity landing platforms.

    Attributes:
        pde_mesh (PDEMeshData): Spatial mesh containing geometry, obstacles, and initial conditions
        Nx (int): Number of grid points in x-direction
        Ny (int): Number of grid points in y-direction
        Lx (float): Physical domain width in metres
        Ly (float): Physical domain height in metres
        Dx (float): Grid spacing in x-direction
        Dy (float): Grid spacing in y-direction
        Dt (float): Time step size (seconds)
        Nt (int): Number of time steps
        thetaUM (float): Under-relaxation parameter for Picard iteration (0 < theta <= 1)
        goals_are_exits (bool): If True, goals act as absorbing boundaries (Dirichlet BC)
        running_cost_weight (float): Weight for distance-to-goal running cost in HJB
        obstacle_penalty (float): Large negative penalty applied inside obstacles
        omask (ndarray): Obstacle mask (Nx, Ny) - 0 for obstacles, 1 for free space
        m0 (ndarray): Initial density distribution (Nx, Ny)
        M (ndarray): Density trajectory (Nt+1, Nx, Ny) - solution to KFP equation
        U (ndarray): Value function trajectory (Nt+1, Nx, Ny) - solution to HJB equation
        goal (Goal|None): Goal/target object managing dynamic goal trajectories and capacities
        door_mask (ndarray): Consolidated time-dependent exit/goal mask (Nt+1, Nx, Ny)
        door_mask_3d (ndarray): Property alias for backward compatibility with MFGPlotter
    """

    def __init__(
        self,
        pde_mesh_data,
        T: float = 3.0,
        Nt: int = 100,
        thetaUM: float = 0.1,
        door_mask=None,
        door_mask_3d=None,  # Fallback keyword argument for backward compatibility
        goal_configs: list | None = None,
        goals_are_exits: bool = False,
        obstacle_penalty: float | None = None,
        running_cost_weight: float = 0.01,
    ):
        """
        Initialize the Mean Field Game solver.

        Args:
            pde_mesh_data (PDEMeshData): Mesh object containing spatial grid, obstacles,
                initial conditions, and goal configurations
            T (float): Total simulation time in seconds (default: 3.0)
            Nt (int): Number of time discretization steps (default: 100)
            thetaUM (float): Under-relaxation parameter for Picard iteration.
                Smaller values (0.05-0.1) stabilize convergence but require more iterations.
                Larger values (0.3-0.5) converge faster but may oscillate. (default: 0.1)
            door_mask_3d (ndarray|None): Optional pre-computed time-dependent exit mask
                of shape (Nt+1, Nx, Ny). If None, computed from goal_configs (default: None)
            goal_configs (list|None): List of goal configuration dictionaries. If None,
                goals are extracted from pde_mesh_data (default: None)
            goals_are_exits (bool): If True, goals act as absorbing boundaries where
                density exits the domain (Dirichlet BC u=0) (default: False)
            obstacle_penalty (float): Large negative value assigned to obstacle cells.
                If None, auto-scaled to -5 * max_running_cost (default: -500.0)
            running_cost_weight (float): Coefficient multiplying squared distance to
                nearest goal in the HJB running cost (default: 0.01)
        """
        self.pde_mesh = pde_mesh_data
        self.Nx, self.Ny = pde_mesh_data.X.shape
        self.Lx, self.Ly = pde_mesh_data.Lx, pde_mesh_data.Ly
        self.Dx, self.Dy = pde_mesh_data.dx, pde_mesh_data.dy
        self.Dt, self.Nt = T / Nt, Nt
        self.thetaUM = thetaUM
        self.goals_are_exits = goals_are_exits
        self.running_cost_weight = running_cost_weight

        # Dynamically scale obstacle penalty relative to max potential drop on grid if not provided.
        # This ensures obstacles remain strongly repulsive regardless of domain size.
        if obstacle_penalty is not None:
            self.obstacle_penalty = obstacle_penalty
        else:
            max_grid_dist_sq = self.Lx**2 + self.Ly**2
            max_cost = self.running_cost_weight * max_grid_dist_sq
            self.obstacle_penalty = -5.0 * max_cost

        self.omask = pde_mesh_data.get_pde_obstacle_mask()
        self.m0 = pde_mesh_data.build_initial_density()

        # Allocate solution arrays: M for density, U for value function
        self.M = np.zeros((self.Nt + 1, self.Nx, self.Ny))
        self.U = np.zeros((self.Nt + 1, self.Nx, self.Ny))
        self.M[0] = self.m0

        initial_door_mask = door_mask if door_mask is not None else door_mask_3d

        # Automatic Goal detection from mesh data or explicit config
        raw_goals = goal_configs if goal_configs is not None else pde_mesh_data.get_goals()
        if raw_goals:
            self.goal = Goal(
                goal_configs=raw_goals,
                Nt=Nt,
                Dt=self.Dt,
                T=T,
                Lx=self.Lx,
                Ly=self.Ly,
            )
            # Alias for backward compatibility with MFGPlotter
            self.evader_swarm = self.goal
            self.door_mask = self._build_dynamic_goal_doors(self.goal.Y_trajectories)
        else:
            self.goal = None
            self.evader_swarm = None
            if initial_door_mask is not None:
                self.door_mask = initial_door_mask
            else:
                self.door_mask = np.zeros((Nt + 1, self.Nx, self.Ny))

        self.door_mask_3d = self.door_mask

    def _build_dynamic_goal_doors(self, goal_trajectories):
        """Construct time-dependent exit mask for active exit goals."""
        door_mask = np.zeros((self.Nt + 1, self.Nx, self.Ny))
        if not self.goal or goal_trajectories is None:
            return door_mask

        X, Y = self.pde_mesh.X, self.pde_mesh.Y
        for k in range(self.Nt + 1):
            for g_idx, g_info in enumerate(self.goal.goals):
                is_exit = g_info.get('is_exit', self.goals_are_exits)
                if is_exit:
                    gx, gy = goal_trajectories[k, g_idx]
                    region = (np.abs(X - gx) <= self.Dx) & (np.abs(Y - gy) <= self.Dy)
                    door_mask[k][region] = 1.0
        return door_mask

    def compute_saturated_door_mask(self, M_field, goal_trajectories):
        """
        Constructs time-dependent exit mask considering goal capacity limits.

        Integrates agent density entering each goal region over time. When cumulative
        absorbed mass reaches a goal's capacity ceiling C_max, the exit boundary is closed
        (door_mask = 0) for all subsequent timesteps.

        Args:
            M_field (ndarray): Density distribution trajectory, shape (Nt+1, Nx, Ny)
            goal_trajectories (ndarray): Goal positions, shape (Nt+1, num_goals, 2)

        Returns:
            ndarray: Updated door mask of shape (Nt+1, Nx, Ny)
        """
        door_mask = np.zeros((self.Nt + 1, self.Nx, self.Ny))
        if not self.goal or goal_trajectories is None:
            return door_mask

        X, Y = self.pde_mesh.X, self.pde_mesh.Y
        cumulative_mass = np.zeros(self.goal.num_goals)

        for k in range(self.Nt + 1):
            for g_idx, g_info in enumerate(self.goal.goals):
                is_exit = g_info.get('is_exit', self.goals_are_exits)
                if not is_exit:
                    continue

                cap = g_info.get('capacity', float('inf'))
                gx, gy = goal_trajectories[k, g_idx]
                region = (np.abs(X - gx) <= self.Dx) & (np.abs(Y - gy) <= self.Dy)

                if cumulative_mass[g_idx] < cap:
                    door_mask[k][region] = 1.0

                if k < self.Nt:
                    mass_in_region = np.sum(M_field[k][region]) * self.Dx * self.Dy
                    cumulative_mass[g_idx] += mass_in_region

        return door_mask

    def compute_running_cost(self, goal_positions_k, k=None, M_trajectory=None):
        """
        Compute distance-based running cost to nearest goal at timestep k, accounting for capacity.

        The running cost penalizes distance from goals, incentivizing the swarm to
        move toward targets. If finite capacity constraints exist, effective distance is
        scaled up as the goal fills (dist^2 / weight). Once fully saturated (weight = 0),
        the goal is assigned an infinite distance penalty and excluded from attracting agents.

        Args:
            goal_positions_k (ndarray): Goal positions at time k, shape (num_goals, 2)
            k (int|None): Current timestep index for capacity tracking
            M_trajectory (ndarray|None): Full density trajectory for capacity integration

        Returns:
            ndarray: Running cost field of shape (Nx, Ny)
        """
        if goal_positions_k is None or len(goal_positions_k) == 0:
            return np.zeros((self.Nx, self.Ny))

        X, Y = self.pde_mesh.X, self.pde_mesh.Y
        capacity_weights = np.ones(len(goal_positions_k))

        if k is not None and M_trajectory is not None and self.goal is not None and self.goal.has_capacity_limits:
            for g_idx, g_info in enumerate(self.goal.goals):
                cap = g_info.get('capacity', float('inf'))
                if np.isfinite(cap):
                    cum_mass = 0.0
                    for t_idx in range(k + 1):
                        gx, gy = self.goal.Y_trajectories[t_idx, g_idx]
                        region = (np.abs(X - gx) <= self.Dx) & (np.abs(Y - gy) <= self.Dy)
                        cum_mass += np.sum(M_trajectory[t_idx][region]) * self.Dx * self.Dy

                    capacity_weights[g_idx] = max(0.0, 1.0 - cum_mass / cap)

        # Filter active (unsaturated) goals
        active_distances = []
        for g_idx, (gx, gy) in enumerate(goal_positions_k):
            weight = capacity_weights[g_idx]
            if weight > 1e-5:
                # Effective distance increases as goal fills up
                eff_dist_sq = ((X - gx) ** 2 + (Y - gy) ** 2) / weight
                active_distances.append(eff_dist_sq)

        if active_distances:
            min_dist_sq = np.minimum.reduce(active_distances)
        else:
            # All goals are saturated -> zero running cost gradient
            min_dist_sq = np.zeros((self.Nx, self.Ny))

        return self.running_cost_weight * min_dist_sq

    def solve_forward_FP_step(self, U_trajectory, door_mask):
        """
        Solve the Fokker-Planck (KFP) equation forward in time.

        The KFP equation evolves the density distribution m from initial condition m0
        under the transport field induced by the optimal control (derived from gradient
        of value function U). Uses implicit Euler time discretization.

        Mathematical formulation:
            ∂m/∂t - ∇·(m ∇H_p(x,∇u)) = 0
        where H_p is the momentum gradient of the Hamiltonian.

        Args:
            U_trajectory (ndarray): Value function trajectory, shape (Nt+1, Nx, Ny)
            door_mask (ndarray): Time-dependent exit mask, shape (Nt+1, Nx, Ny)

        Returns:
            ndarray: Evolved density m of shape (Nt+1, Nx, Ny)
        """
        m = np.zeros_like(self.M)
        m[0] = self.m0
        N_total = self.Nx * self.Ny

        # Time-stepping loop: implicit solve at each timestep
        for k in range(1, self.Nt + 1):
            rows, cols, vals, b = compute_FP_matrix_entries(
                m[k - 1], U_trajectory[k - 1], self.omask, door_mask[k - 1],
                self.Nx, self.Ny, self.Dx, self.Dy, self.Dt
            )
            A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()
            mtmp = sparse.linalg.spsolve(A, b)
            m[k] = mtmp.reshape((self.Nx, self.Ny))
        return m

    def solve_backward_HJB_step(self, M_trajectory, goal_trajectories, door_mask):
        """
        Solve the Hamilton-Jacobi-Bellman (HJB) equation backward in time.

        The HJB equation computes optimal value function u, which represents cost-to-go
        from any position to goals. The equation is nonlinear due to the Hamiltonian,
        requiring Newton iteration at each timestep. Uses implicit Euler discretization.
        
        Mathematical formulation:
            $$-\\frac{\\partial u}{\\partial t} + H(x, m, \\nabla u) - \\nu \\Delta u = 0$$
        Args:
            M_trajectory (ndarray): Density trajectory, shape (Nt+1, Nx, Ny)
            goal_trajectories (ndarray|None): Goal positions, shape (Nt+1, num_goals, 2) or None
            door_mask (ndarray): Time-dependent exit mask, shape (Nt+1, Nx, Ny)

        Returns:
            ndarray: Value function u of shape (Nt+1, Nx, Ny)
        """
        u = np.zeros_like(self.U)

        # Terminal condition at t = T
        if goal_trajectories is not None:
            running_cost_Nt = self.compute_running_cost(goal_trajectories[self.Nt], k=self.Nt, M_trajectory=M_trajectory)
            u[self.Nt] = -running_cost_Nt
        else:
            u[self.Nt] = np.zeros((self.Nx, self.Ny))

        # Backward time-stepping loop
        for k in range(self.Nt - 1, -1, -1):
            running_cost_k = (
                self.compute_running_cost(goal_trajectories[k], k=k, M_trajectory=M_trajectory)
                if goal_trajectories is not None
                else np.zeros((self.Nx, self.Ny))
            )
            Unew_n = np.copy(u[k + 1])
            N_total = self.Nx * self.Ny

            # Newton iteration for nonlinear Hamiltonian (max 30 iterations)
            for _ in range(30):
                # Compute nonlinear residual F(U^n)
                FnU_flat = getFnU_2D(
                    u[k + 1], Unew_n, M_trajectory[k + 1], self.omask, door_mask[k],
                    running_cost_k, self.Nx, self.Ny, self.Dx, self.Dy, self.Dt,
                    obstacle_penalty=self.obstacle_penalty
                ).flatten()

                # Compute Jacobian matrix A = ∂F/∂U
                rows, cols, vals = compute_HJB_matrix_entries(
                    Unew_n, M_trajectory[k + 1], self.omask, door_mask[k],
                    self.Nx, self.Ny, self.Dx, self.Dy, self.Dt
                )
                A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()
                b = A.dot(Unew_n.flatten()) - FnU_flat

                # Apply boundary conditions: Dirichlet u=0 at exit doors, potential penalty at obstacles
                for i in range(self.Nx):
                    for j in range(self.Ny):
                        ind = i * self.Ny + j
                        if door_mask[k, i, j] == 1:
                            b[ind] = 0.0
                        elif self.omask[i, j] == 0:
                            b[ind] = self.obstacle_penalty

                # Newton update: solve A * U_new = b
                Unres = sparse.linalg.spsolve(A, b).reshape((self.Nx, self.Ny))
                l2err = np.linalg.norm(Unew_n.flatten() - Unres.flatten()) * np.sqrt(self.Dx * self.Dy)
                Unew_n = np.copy(Unres)
                if l2err < 1e-6:
                    break
            u[k] = Unew_n
        return u

    def run_picard_system(self, max_iters: int = 10, tolerance: float = 1e-5):
        """
        Execute Picard iteration to solve the coupled HJB-KFP system with capacity limits.

        Picard iteration alternates between solving HJB (given density M) and solving
        KFP (given value U), with under-relaxation to stabilize convergence. For dynamic
        goals, also updates goal positions based on density gradients at each iteration.

        Algorithm:
            1. Solve HJB backward: U^{k+1} = HJB_solve(M^k, Y^k)
            2. Solve KFP forward: M^{k+1} = KFP_solve(U^{k+1})
            3. Update goals: Y^{k+1} = Goal_update(M^{k+1})  [if dynamic goals]
            4. Apply relaxation: X^{k+1} = θ X_new + (1-θ) X^k for X ∈ {U, M, Y}
            5. Check convergence: ||U^{k+1} - U^k|| < tol and ||M^{k+1} - M^k|| < tol

        Args:
            max_iters (int): Maximum number of Picard iterations (default: 10)
            tolerance (float): L2 convergence tolerance for U and M residuals (default: 1e-5)

        Returns:
            tuple: (U, M) where
                U (ndarray): Converged value function, shape (Nt+1, Nx, Ny)
                M (ndarray): Converged density distribution, shape (Nt+1, Nx, Ny)
        """
        # Normalization factor for L2 error in space-time
        space_time_factor = np.sqrt(self.Dx * self.Dy * self.Dt)

        for iiter in range(1, max_iters + 1):
            start_time = time.time()
            print(f"\n>>> Macro Picard Loop Execution: {iiter} / {max_iters}", flush=True)

            goal_trajectories = self.goal.Y_trajectories if self.goal is not None else None

            if self.goal is not None and self.goal.has_capacity_limits:
                self.door_mask = self.compute_saturated_door_mask(self.M, goal_trajectories)
                self.door_mask_3d = self.door_mask

            # Step 1: Solve HJB backward in time
            U_temp = self.solve_backward_HJB_step(self.M, goal_trajectories, self.door_mask)
            U_new = self.thetaUM * U_temp + (1.0 - self.thetaUM) * self.U

            # Step 2: Solve FP forward in time
            M_temp = self.solve_forward_FP_step(U_new, self.door_mask)
            M_new = self.thetaUM * M_temp + (1.0 - self.thetaUM) * self.M

            # Step 3: Update dynamic goals
            if self.goal is not None:
                Y_temp = self.goal.update_positions(M_new, self.omask, self.Dx, self.Dy, self.Lx, self.Ly)
                Y_new = self.thetaUM * Y_temp + (1.0 - self.thetaUM) * self.goal.Y_trajectories
                y_err = np.linalg.norm(Y_new - self.goal.Y_trajectories)
                self.goal.Y_trajectories = np.copy(Y_new)
            else:
                y_err = 0.0

            # Compute L2 residuals for convergence check
            u_err = np.linalg.norm(U_new - self.U) * space_time_factor
            m_err = np.linalg.norm(M_new - self.M) * space_time_factor

            print(
                f"    u_residual: {u_err:.6e} | m_residual: {m_err:.6e} | "
                f"y_goal_residual: {y_err:.6e} | Time: {time.time() - start_time:.2f}s",
                flush=True
            )

            self.U = np.copy(U_new)
            self.M = np.copy(M_new)

            # Check convergence: all residuals below tolerance
            if u_err < tolerance and m_err < tolerance and y_err < tolerance:
                print(f"\n[Success] Converged at iteration {iiter}!", flush=True)
                break

        return self.U, self.M


class MFG2PopSolver:
    """
    Coupled two-population Mean Field Game solver.

    This solver handles competitive or cooperative scenarios with two distinct populations,
    each solving its own HJB-KFP system while being influenced by the other population's
    density distribution. Examples: pursuit-evasion with two swarms, competing crowds
    navigating toward different exits, or cooperative swarms with different objectives.

    The key difference from single-population MFG is that each population's Hamiltonian
    depends on both its own density and the other population's density, creating a coupled
    game-theoretic interaction.

    Attributes:
        Nx (int): Number of grid points in x-direction
        Ny (int): Number of grid points in y-direction
        Dx (float): Grid spacing in x-direction
        Dy (float): Grid spacing in y-direction
        Dt (float): Time step size (seconds)
        Nt (int): Number of time steps
        thetaUM (float): Under-relaxation parameter for Picard iteration (0 < theta <= 1)
        omask (ndarray): Obstacle mask (Nx, Ny) - 0 for obstacles, 1 for free space
        m0_1 (ndarray): Initial density for population 1, shape (Nx, Ny)
        m0_2 (ndarray): Initial density for population 2, shape (Nx, Ny)
        g_x_1 (ndarray): Terminal cost for population 1, shape (Nx, Ny)
        g_x_2 (ndarray): Terminal cost for population 2, shape (Nx, Ny)
        M1 (ndarray): Density trajectory for population 1, shape (Nt+1, Nx, Ny)
        M2 (ndarray): Density trajectory for population 2, shape (Nt+1, Nx, Ny)
        U1 (ndarray): Value function for population 1, shape (Nt+1, Nx, Ny)
        U2 (ndarray): Value function for population 2, shape (Nt+1, Nx, Ny)
    """

    def __init__(self, pde_mesh_data_1, pde_mesh_data_2, T: float = 300.0, Nt: int = 3000, thetaUM: float = 0.1):
        """
        Initialize the two-population Mean Field Game solver.

        Args:
            pde_mesh_data_1 (PDEMeshData): Mesh data for population 1 (contains initial
                condition, terminal cost, and geometry)
            pde_mesh_data_2 (PDEMeshData): Mesh data for population 2 (must have same
                spatial grid as population 1)
            T (float): Total simulation time in seconds (default: 300.0)
            Nt (int): Number of time discretization steps (default: 3000)
            thetaUM (float): Under-relaxation parameter for Picard iteration.
                Smaller values stabilize convergence. (default: 0.1)
        """
        self.Nx, self.Ny = pde_mesh_data_1.X.shape
        self.Dx, self.Dy = pde_mesh_data_1.dx, pde_mesh_data_1.dy
        self.Dt = T / Nt
        self.Nt = Nt
        self.thetaUM = thetaUM

        self.omask = pde_mesh_data_1.get_pde_obstacle_mask()
        self.m0_1 = pde_mesh_data_1.build_initial_density(sigma_multiplier=3)
        self.g_x_1 = pde_mesh_data_1.build_terminal_cost()

        self.m0_2 = pde_mesh_data_2.build_initial_density(sigma_multiplier=3)
        self.g_x_2 = pde_mesh_data_2.build_terminal_cost()

        self.M1, self.M2 = np.zeros((self.Nt + 1, self.Nx, self.Ny)), np.zeros((self.Nt + 1, self.Nx, self.Ny))
        self.U1, self.U2 = np.zeros((self.Nt + 1, self.Nx, self.Ny)), np.zeros((self.Nt + 1, self.Nx, self.Ny))

        self.M1[0], self.M2[0] = self.m0_1, self.m0_2
        self.U1[self.Nt], self.U2[self.Nt] = self.g_x_1, self.g_x_2

    def solve_forward_FP(self, U_trajectory, m0, M_other_trajectory):
        """
        Solve the Fokker-Planck equation forward in time for one population.

        Similar to single-population KFP, but the transport field now depends on the
        other population's density through the Hamiltonian coupling.

        Args:
            U_trajectory (ndarray): This population's value function, shape (Nt+1, Nx, Ny)
            m0 (ndarray): This population's initial density, shape (Nx, Ny)
            M_other_trajectory (ndarray): Other population's density, shape (Nt+1, Nx, Ny)

        Returns:
            ndarray: Evolved density m of shape (Nt+1, Nx, Ny)
        """
        m = np.zeros((self.Nt + 1, self.Nx, self.Ny))
        m[0] = m0
        N_total = self.Nx * self.Ny

        for k in range(1, self.Nt + 1):
            rows, cols, vals, b = compute_FP_matrix_entries_2Pop(
                m[k - 1], M_other_trajectory[k - 1], U_trajectory[k - 1],
                self.omask, self.Nx, self.Ny, self.Dx, self.Dy, self.Dt
            )
            A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()
            m[k] = sparse.linalg.spsolve(A, b).reshape((self.Nx, self.Ny))
        return m

    def get_u_onestep_newton(self, Uk_n, Unew_np1, Unew_n_tmp, Mk_np1, Mk_other, pop):
        """
        Perform one Newton iteration step for the HJB nonlinearity.

        Solves the linearized system A * U_new = b where A is the Jacobian and b
        is the residual, accounting for coupling with the other population's density.

        Args:
            Uk_n (ndarray): Current value at time k, shape (Nx, Ny) [not used, kept for signature]
            Unew_np1 (ndarray): Value at time k+1, shape (Nx, Ny)
            Unew_n_tmp (ndarray): Current Newton iterate at time k, shape (Nx, Ny)
            Mk_np1 (ndarray): This population's density at time k+1, shape (Nx, Ny)
            Mk_other (ndarray): Other population's density at time k+1, shape (Nx, Ny)
            pop (int): Population identifier (1 or 2) for Hamiltonian selection

        Returns:
            ndarray: Updated value function U_new after Newton step, shape (Nx, Ny)
        """
        N_total = self.Nx * self.Ny
        # Compute nonlinear residual F(U^n) with two-population Hamiltonian
        FnU_flat = getFnU_2D_2Pop(
            Unew_np1, Unew_n_tmp, Mk_np1, Mk_other,
            self.omask, self.Nx, self.Ny, self.Dx, self.Dy, self.Dt, pop
        ).flatten()

        # Compute Jacobian matrix A = ∂F/∂U
        rows, cols, vals = compute_HJB_matrix_entries_2Pop(
            Unew_n_tmp, Mk_np1, Mk_other, self.omask,
            self.Nx, self.Ny, self.Dx, self.Dy, self.Dt
        )
        A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()
        b = A.dot(Unew_n_tmp.flatten()) - FnU_flat

        # Apply obstacle penalty
        for i in range(self.Nx):
            for j in range(self.Ny):
                if self.omask[i, j] == 0:
                    b[i * self.Ny + j] = -500.0

        utmp = sparse.linalg.spsolve(A, b)
        return utmp.reshape((self.Nx, self.Ny))

    def solve_backward_HJB(self, M_trajectory, M_trajectory_other, U_temp, g_x, pop):
        """
        Solve the Hamilton-Jacobi-Bellman equation backward in time for one population.
        
        Similar to single-population HJB, but the Hamiltonian now depends on both this
        population's density and the other population's density (game-theoretic coupling).
        Uses Newton iteration at each timestep.

        Args:
            M_trajectory (ndarray): This population's density, shape (Nt+1, Nx, Ny)
            M_trajectory_other (ndarray): Other population's density, shape (Nt+1, Nx, Ny)
            U_temp (ndarray): Previous value function estimate (for initialization), shape (Nt+1, Nx, Ny)
            g_x (ndarray): Terminal cost g(x) at final time, shape (Nx, Ny)
            pop (int): Population identifier (1 or 2) for Hamiltonian selection

        Returns:
            ndarray: Value function u of shape (Nt+1, Nx, Ny)
        """
        u = np.zeros_like(U_temp)
        u[self.Nt] = g_x
        # Backward time-stepping loop
        for k in range(self.Nt - 1, -1, -1):
            Unew_n = np.copy(u[k + 1])
            # Newton iteration (max 5 iterations per timestep)
            for _ in range(5):
                Unres = self.get_u_onestep_newton(u[k], u[k + 1], Unew_n, M_trajectory[k + 1], M_trajectory_other[k + 1], pop)
                l2err = np.linalg.norm(Unew_n.flatten() - Unres.flatten()) * np.sqrt(self.Dx * self.Dy)
                Unew_n = np.copy(Unres)
                if l2err < 1e-6:
                    break
            u[k] = Unew_n
        return u

    def run_picard_system(self, max_iters: int = 25, tolerance: float = 1e-5):
        """
        Execute Picard iteration to solve the coupled two-population system.

        This method alternates between solving HJB and KFP for each population, where
        each population's equations depend on the other population's current state. The
        coupling creates a game-theoretic Nash equilibrium problem.

        Algorithm:
            1. Solve HJB for population 1: U1^{k+1} = HJB_solve(M1^k, M2^k)
            2. Solve HJB for population 2: U2^{k+1} = HJB_solve(M2^k, M1^k)
            3. Solve KFP for population 1: M1^{k+1} = KFP_solve(U1^{k+1}, M2^k)
            4. Solve KFP for population 2: M2^{k+1} = KFP_solve(U2^{k+1}, M1^k)
            5. Apply relaxation: X^{k+1} = θ X_new + (1-θ) X^k for all state variables
            6. Check convergence: max residuals < tolerance

        Args:
            max_iters (int): Maximum number of Picard iterations (default: 25)
            tolerance (float): L2 convergence tolerance for both populations (default: 1e-5)

        Returns:
            tuple: (U1, M1, U2, M2) where
                U1 (ndarray): Population 1 value function, shape (Nt+1, Nx, Ny)
                M1 (ndarray): Population 1 density, shape (Nt+1, Nx, Ny)
                U2 (ndarray): Population 2 value function, shape (Nt+1, Nx, Ny)
                M2 (ndarray): Population 2 density, shape (Nt+1, Nx, Ny)
        """
        # Normalization factor for L2 error in space-time
        space_time_factor = np.sqrt(self.Dx * self.Dy * self.Dt)

        for iiter in range(1, max_iters + 1):
            start_time = time.time()
            print(f"\n>>> 2-Pop Picard Loop Iteration: {iiter} / {max_iters}", flush=True)

            # Solve HJB for both populations (can be done in parallel conceptually)
            U1_temp = self.solve_backward_HJB(self.M1, self.M2, self.U1, self.g_x_1, pop=1)
            U2_temp = self.solve_backward_HJB(self.M2, self.M1, self.U2, self.g_x_2, pop=2)

            # Apply under-relaxation to value functions
            U1_new = self.thetaUM * U1_temp + (1.0 - self.thetaUM) * self.U1
            U2_new = self.thetaUM * U2_temp + (1.0 - self.thetaUM) * self.U2

            # Solve KFP for both populations with updated value functions
            M1_temp = self.solve_forward_FP(U1_new, self.m0_1, self.M2)
            M2_temp = self.solve_forward_FP(U2_new, self.m0_2, self.M1)

            # Apply under-relaxation to densities
            M1_new = self.thetaUM * M1_temp + (1.0 - self.thetaUM) * self.M1
            M2_new = self.thetaUM * M2_temp + (1.0 - self.thetaUM) * self.M2

            # Compute L2 residuals (use max over both populations)
            u_err = max(np.linalg.norm(U1_new - self.U1), np.linalg.norm(U2_new - self.U2)) * space_time_factor
            m_err = max(np.linalg.norm(M1_new - self.M1), np.linalg.norm(M2_new - self.M2)) * space_time_factor

            print(f"    u_residual: {u_err:.6e} | m_residual: {m_err:.6e} | Time: {time.time() - start_time:.2f}s", flush=True)

            # Update state for next iteration
            self.U1, self.M1 = np.copy(U1_new), np.copy(M1_new)
            self.U2, self.M2 = np.copy(U2_new), np.copy(M2_new)

            # Check convergence: both populations must satisfy tolerance
            if u_err < tolerance and m_err < tolerance:
                print(f"\n[Success] 2-Population system converged at iteration {iiter}!", flush=True)
                break

        return self.U1, self.M1, self.U2, self.M2
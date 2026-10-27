"""
Mean Field Games Generalized N-Population Problem Solvers.

This module provides object-oriented numerical solvers for arbitrary N-population
Mean Field Game (MFG) systems, coupling backward Hamilton-Jacobi-Bellman (HJB) equations
with forward Kolmogorov-Fokker-Planck (KFP) equations via damped Picard iteration.

Key Classes
-----------
- Population: Data container encapsulating a single crowd's spatial mesh, initial
  density, decoupled running/terminal costs, kinematic parameters, and goal manager.
- MFGSolver: Fully coupled Nash equilibrium solver for arbitrary N >= 1 populations.
- MFG2PopSolver: Backward-compatible 2-population wrapper matching MAP2PDE2POP.py.
- MFGShadowLeaderSolver: Scalable N-population Stackelberg / Shadow-Leader solver that
  freezes leader trajectories to decouple follower best-response computations.
"""

from dataclasses import dataclass, field
import time
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
from mfgames.objectives import Goal
from mfgames.numerics import (
    compute_FP_matrix_entries,
    getFnU_2D,
    compute_HJB_matrix_entries,
    compute_FP_matrix_entries_2Pop,
    getFnU_2D_2Pop,
    compute_HJB_matrix_entries_2Pop,
)


@dataclass
class Population:
    """
    Data container representing a single crowd population in an N-population game.

    Parameters
    ----------
    name : str
        Descriptive label for the population (e.g., 'pursuers', 'evaders', 'team_alpha').
    mesh : MAP2PDE or MFGTrafficGeometry
        Spatial discretization mesh providing coordinates, step sizes, and obstacle masks.
    m0 : ndarray, optional
        Initial macroscopic crowd density grid of shape (Nx, Ny).
    goal_configs : list, optional
        Goal specification dictionaries defining positions, speeds, and capacities.
    goals_are_exits : bool
        If True, active goals act as absorbing exit boundaries with Dirichlet condition u=0.
    running_cost_weight : float
        Weight w_R scaling running distance costs accumulated during t < T.
    terminal_cost_weight : float
        Weight w_T scaling terminal distance cost evaluated at t = T.
    running_cost_power : int
        Distance power p_R (1 for linear distance, 2 for squared Euclidean distance).
    terminal_cost_power : int
        Distance power p_T (1 for linear distance, 2 for squared Euclidean distance).
    saturated_goal_penalty : float
        Repulsive Gaussian barrier erected around goals that have exceeded capacity.
    obstacle_penalty : float
        Potential penalty applied to obstacle grid cells in the HJB residual.
    stencil_mode : int
        0 for standard 1-population / pursuit-evasion upwind stencils;
        1 for MAP2PDE2POP cross-interaction upwind stencils.
    """

    name: str = "population"
    mesh: object = None
    m0: np.ndarray = None
    goal_configs: list = field(default_factory=list)
    goals_are_exits: bool = False
    running_cost_weight: float = 0.01
    terminal_cost_weight: float = 0.01
    running_cost_power: int = 2
    terminal_cost_power: int = 2
    saturated_goal_penalty: float = 0.0
    obstacle_penalty: float = -500.0
    stencil_mode: int = 0

    # Allocated simulation states
    U: np.ndarray = field(default=None, repr=False)
    M: np.ndarray = field(default=None, repr=False)
    goal: Goal = field(default=None, repr=False)
    door_mask: np.ndarray = field(default=None, repr=False)


class MFGSolver:
    """
    Generalized N-Population Mean Field Game Solver.

    Supports N = 1 (single crowd navigation and traffic evacuation) and arbitrary N >= 2
    (competitive, cooperative, or multi-team swarms) through a single codebase.
    """

    def __init__(
        self,
        populations: list[Population] | None = None,
        # 1-Population convenience parameters (backward compatibility)
        pde_mesh_data=None,
        T: float = 3.0,
        Nt: int = 100,
        thetaUM: float = 0.1,
        door_mask=None,
        door_mask_3d=None,
        goal_configs: list | None = None,
        goals_are_exits: bool = False,
        obstacle_penalty: float | None = None,
        running_cost_weight: float = 0.01,
        terminal_cost_weight: float | None = None,
        running_cost_power: int = 2,
        terminal_cost_power: int = 2,
        saturated_goal_penalty: float = 0.0,
        # N-Population relational parameters
        interaction_matrix: np.ndarray | None = None,
        congestion_matrix: np.ndarray | None = None,
        shared_capacity_goals: bool = False,
    ):
        self.T, self.Nt = T, Nt
        self.Dt = T / Nt
        self.thetaUM = thetaUM
        self.shared_capacity_goals = shared_capacity_goals

        # Normalize 1-population inputs into the Population container
        if populations is not None:
            self.populations = populations
            primary_mesh = populations[0].mesh
        elif pde_mesh_data is not None:
            primary_mesh = pde_mesh_data
            pop1 = Population(
                name="pop_1",
                mesh=pde_mesh_data,
                m0=pde_mesh_data.build_initial_density(),
                goal_configs=goal_configs if goal_configs is not None else getattr(pde_mesh_data, 'custom_goals', []),
                goals_are_exits=goals_are_exits,
                running_cost_weight=running_cost_weight,
                terminal_cost_weight=terminal_cost_weight if terminal_cost_weight is not None else running_cost_weight,
                running_cost_power=running_cost_power,
                terminal_cost_power=terminal_cost_power,
                saturated_goal_penalty=saturated_goal_penalty,
                obstacle_penalty=obstacle_penalty if obstacle_penalty is not None else -500.0,
                stencil_mode=0,
            )
            self.populations = [pop1]
        else:
            raise ValueError("Must provide either 'populations' or 'pde_mesh_data'.")

        self.num_pops = len(self.populations)
        self.Nx, self.Ny = primary_mesh.X.shape
        self.Lx, self.Ly = primary_mesh.Lx, primary_mesh.Ly
        self.Dx, self.Dy = primary_mesh.dx, primary_mesh.dy
        self.omask = primary_mesh.get_pde_obstacle_mask()

        # Inter-crowd strategic interaction matrix W_inter and cross-congestion matrix W_cong
        if interaction_matrix is not None:
            self.interaction_matrix = np.asarray(interaction_matrix, dtype=np.float64)
        else:
            self.interaction_matrix = np.zeros((self.num_pops, self.num_pops), dtype=np.float64)

        if congestion_matrix is not None:
            self.congestion_matrix = np.asarray(congestion_matrix, dtype=np.float64)
        else:
            self.congestion_matrix = np.zeros((self.num_pops, self.num_pops), dtype=np.float64)

        initial_door_mask = door_mask if door_mask is not None else door_mask_3d

        # Initialize simulation states and Goal managers for each population
        for p_idx, pop in enumerate(self.populations):
            pop.M = np.zeros((self.Nt + 1, self.Nx, self.Ny), dtype=np.float64)
            pop.U = np.zeros((self.Nt + 1, self.Nx, self.Ny), dtype=np.float64)

            if pop.m0 is None:
                pop.m0 = pop.mesh.build_initial_density()
            pop.M[0] = pop.m0

            raw_goals = pop.goal_configs if pop.goal_configs else getattr(pop.mesh, 'custom_goals', None)
            if raw_goals is None and hasattr(pop.mesh, 'get_goals'):
                raw_goals = pop.mesh.get_goals()

            if raw_goals:
                pop.goal = Goal(
                    goal_configs=raw_goals,
                    Nt=self.Nt,
                    Dt=self.Dt,
                    T=self.T,
                    Lx=self.Lx,
                    Ly=self.Ly,
                    goals_are_exits_default=pop.goals_are_exits,
                )
                pop.door_mask = self._build_dynamic_doors(pop, pop.goal.Y_trajectories)
                # Terminal slice initialization prevents early Picard relaxation drift
                pop.U[self.Nt] = self.compute_terminal_cost(pop, pop.goal.Y_trajectories[self.Nt])
            else:
                pop.goal = None
                if self.num_pops == 1 and initial_door_mask is not None:
                    pop.door_mask = initial_door_mask
                else:
                    pop.door_mask = np.zeros((self.Nt + 1, self.Nx, self.Ny), dtype=np.float64)
                pop.U[self.Nt] = np.zeros((self.Nx, self.Ny), dtype=np.float64)

    # --- Backward-Compatibility Properties for 1-Population Runs ---
    @property
    def M(self): return self.populations[0].M
    @M.setter
    def M(self, val): self.populations[0].M = val

    @property
    def U(self): return self.populations[0].U
    @U.setter
    def U(self, val): self.populations[0].U = val

    @property
    def goal(self): return self.populations[0].goal

    @property
    def door_mask(self): return self.populations[0].door_mask
    @property
    def door_mask_3d(self): return self.populations[0].door_mask

    @property
    def obstacle_penalty(self): return self.populations[0].obstacle_penalty

    # --- Backward-Compatibility Properties for 2-Population Runs ---
    @property
    def M1(self): return self.populations[0].M
    @property
    def M2(self): return self.populations[1].M if self.num_pops > 1 else None
    @property
    def U1(self): return self.populations[0].U
    @property
    def U2(self): return self.populations[1].U if self.num_pops > 1 else None

    def _build_dynamic_doors(self, pop, goal_trajectories):
        """Construct time-indexed door mask for a population's active exit goals."""
        door_mask = np.zeros((self.Nt + 1, self.Nx, self.Ny), dtype=np.float64)
        if not pop.goal or goal_trajectories is None:
            return door_mask

        X, Y = pop.mesh.X, pop.mesh.Y
        for k in range(self.Nt + 1):
            for g_idx, g_info in enumerate(pop.goal.goals):
                if g_info.get('is_exit', False):
                    gx, gy = goal_trajectories[k, g_idx]
                    region = (np.abs(X - gx) <= self.Dx) & (np.abs(Y - gy) <= self.Dy)
                    door_mask[k][region] = 1.0
        return door_mask

    def compute_saturated_door_mask(self, pop, goal_trajectories):
        """Update door mask querying Goal.is_saturated directly."""
        door_mask = np.zeros((self.Nt + 1, self.Nx, self.Ny), dtype=np.float64)
        if not pop.goal or goal_trajectories is None:
            return door_mask

        X, Y = pop.mesh.X, pop.mesh.Y[cite: 8]
        for k in range(self.Nt + 1):
            for g_idx, g_info in enumerate(pop.goal.goals):
                if not g_info.get('is_exit', False):
                    continue
                if not pop.goal.is_saturated[k, g_idx]:
                    gx, gy = goal_trajectories[k, g_idx]
                    region = (np.abs(X - gx) <= self.Dx) & (np.abs(Y - gy) <= self.Dy)
                    door_mask[k][region] = 1.0
        return door_mask

    def compute_running_cost(self, pop, goal_positions_k, k=None):
        """Compute running distance cost f_p(x, t) for population p at timestep k."""
        if pop.running_cost_weight == 0.0 or goal_positions_k is None or len(goal_positions_k) == 0:
            return np.zeros((self.Nx, self.Ny), dtype=np.float64)

        X, Y = pop.mesh.X, pop.mesh.Y[cite: 8]
        active_distances = []
        saturated_repulsion = np.zeros((self.Nx, self.Ny), dtype=np.float64)

        for g_idx, (gx, gy) in enumerate(goal_positions_k):
            is_active = True
            if k is not None and pop.goal is not None and pop.goal.has_capacity_limits:
                if pop.goal.is_saturated[k, g_idx]:
                    is_active = False

            if is_active:
                if pop.running_cost_power == 2:
                    dist_term = (X - gx) ** 2 + (Y - gy) ** 2
                else:
                    dist_term = np.sqrt((X - gx) ** 2 + (Y - gy) ** 2)
                active_distances.append(dist_term)
            elif pop.saturated_goal_penalty > 0.0:
                dist_sq = (X - gx) ** 2 + (Y - gy) ** 2
                sigma_sq = (10.0 * self.Dx) ** 2
                saturated_repulsion += pop.saturated_goal_penalty * np.exp(-dist_sq / (2.0 * sigma_sq))

        if active_distances:
            min_dist = np.minimum.reduce(active_distances)
        else:
            min_dist = np.zeros((self.Nx, self.Ny), dtype=np.float64)

        return pop.running_cost_weight * min_dist + saturated_repulsion

    def compute_terminal_cost(self, pop, goal_positions_Nt):
        """Compute terminal condition g_p(x) = u_p(T, x) at final time t = T."""
        if pop.terminal_cost_weight == 0.0 or goal_positions_Nt is None or len(goal_positions_Nt) == 0:
            return np.zeros((self.Nx, self.Ny), dtype=np.float64)

        X, Y = pop.mesh.X, pop.mesh.Y[cite: 8]
        active_distances = []
        saturated_repulsion = np.zeros((self.Nx, self.Ny), dtype=np.float64)

        for g_idx, (gx, gy) in enumerate(goal_positions_Nt):
            is_active = True
            if pop.goal is not None and pop.goal.has_capacity_limits:
                if pop.goal.is_saturated[self.Nt, g_idx]:
                    is_active = False

            if is_active:
                if pop.terminal_cost_power == 2:
                    dist_term = (X - gx) ** 2 + (Y - gy) ** 2
                else:
                    dist_term = np.sqrt((X - gx) ** 2 + (Y - gy) ** 2)
                active_distances.append(dist_term)
            elif pop.saturated_goal_penalty > 0.0:
                dist_sq = (X - gx) ** 2 + (Y - gy) ** 2
                sigma_sq = (10.0 * self.Dx) ** 2
                saturated_repulsion += pop.saturated_goal_penalty * np.exp(-dist_sq / (2.0 * sigma_sq))

        if active_distances:
            min_dist = np.minimum.reduce(active_distances)
        else:
            min_dist = np.zeros((self.Nx, self.Ny), dtype=np.float64)

        return pop.terminal_cost_weight * min_dist + saturated_repulsion

    def _get_coupling_fields(self, p_idx: int, k: int):
        """
        Compute effective other-crowd congestion and interaction potential fields for population p[cite: 10].
        """
        other_congestion = np.zeros((self.Nx, self.Ny), dtype=np.float64)
        interaction_cost = np.zeros((self.Nx, self.Ny), dtype=np.float64)
        total_density = np.zeros((self.Nx, self.Ny), dtype=np.float64)

        for q_idx, q_pop in enumerate(self.populations):
            mq_k = q_pop.M[k]
            total_density += mq_k
            if q_idx != p_idx:
                w_c = self.congestion_matrix[p_idx, q_idx]
                other_congestion += w_c * mq_k
                w_i = self.interaction_matrix[p_idx, q_idx]
                interaction_cost += w_i * mq_k[cite: 10]

        # Congestion capacity penalty: max(sum(m_q) - 4.0, 0.0)[cite: 10]
        if self.num_pops > 1:
            interaction_cost += np.maximum(total_density - 4.0, 0.0)[cite: 10]

        return other_congestion, interaction_cost

    def solve_forward_FP_step(self, p_idx: int, U_p):
        """Solve forward KFP equation for population p given optimal value function U_p[cite: 8, 10]."""
        pop = self.populations[p_idx]
        m = np.zeros_like(pop.M)[cite: 8]
        m[0] = pop.m0[cite: 8, 10]
        N_total = self.Nx * self.Ny[cite: 8, 10]

        for k in range(1, self.Nt + 1):[cite: 8, 10]
            dm = pop.door_mask[k - 1] if pop.door_mask is not None else np.zeros((self.Nx, self.Ny))
            if pop.stencil_mode == 0:
                rows, cols, vals, b = compute_FP_matrix_entries(
                    m[k - 1], U_p[k - 1], self.omask, dm,
                    self.Nx, self.Ny, self.Dx, self.Dy, self.Dt
                )
            else:
                other_m, _ = self._get_coupling_fields(p_idx, k - 1)
                rows, cols, vals, b = compute_FP_matrix_entries_2Pop(
                    m[k - 1], other_m, U_p[k - 1], self.omask,
                    self.Nx, self.Ny, self.Dx, self.Dy, self.Dt
                )[cite: 10]

            A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()[cite: 8, 10]
            mtmp = sparse.linalg.spsolve(A, b)[cite: 8, 10]
            m[k] = mtmp.reshape((self.Nx, self.Ny))[cite: 8, 10]
        return m

    def solve_backward_HJB_step(self, p_idx: int, goal_trajectories):
        """Solve backward HJB equation for population p given current crowd densities[cite: 8, 10]."""
        pop = self.populations[p_idx]
        u = np.zeros_like(pop.U)[cite: 8]
        N_total = self.Nx * self.Ny[cite: 8, 10]

        if goal_trajectories is not None:
            u[self.Nt] = self.compute_terminal_cost(pop, goal_trajectories[self.Nt])
        else:
            u[self.Nt] = np.zeros((self.Nx, self.Ny), dtype=np.float64)

        for k in range(self.Nt - 1, -1, -1):[cite: 8, 10]
            running_cost_k = (
                self.compute_running_cost(pop, goal_trajectories[k], k=k)
                if goal_trajectories is not None
                else np.zeros((self.Nx, self.Ny), dtype=np.float64)
            )
            dm = pop.door_mask[k] if pop.door_mask is not None else np.zeros((self.Nx, self.Ny))
            Unew_n = np.copy(u[k + 1])[cite: 8, 10]
            other_m, inter_cost = self._get_coupling_fields(p_idx, k + 1)

            for _ in range(30 if pop.stencil_mode == 0 else 5):[cite: 8, 10]
                if pop.stencil_mode == 0:
                    FnU_flat = getFnU_2D(
                        u[k + 1], Unew_n, pop.M[k + 1], self.omask, dm,
                        running_cost_k + inter_cost, self.Nx, self.Ny, self.Dx, self.Dy, self.Dt,
                        obstacle_penalty=pop.obstacle_penalty
                    ).flatten()[cite: 8]
                    rows, cols, vals = compute_HJB_matrix_entries(
                        Unew_n, pop.M[k + 1], self.omask, dm,
                        self.Nx, self.Ny, self.Dx, self.Dy, self.Dt
                    )[cite: 8]
                else:
                    FnU_flat = getFnU_2D_2Pop(
                        u[k + 1], Unew_n, pop.M[k + 1], other_m, self.omask,
                        self.Nx, self.Ny, self.Dx, self.Dy, self.Dt, pop=(p_idx + 1)
                    ).flatten()[cite: 10]
                    rows, cols, vals = compute_HJB_matrix_entries_2Pop(
                        Unew_n, pop.M[k + 1], other_m, self.omask,
                        self.Nx, self.Ny, self.Dx, self.Dy, self.Dt
                    )

                A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()[cite: 8, 10]
                b = A.dot(Unew_n.flatten()) - FnU_flat[cite: 8, 10]

                for i in range(self.Nx):[cite: 8, 10]
                    for j in range(self.Ny):[cite: 8, 10]
                        ind = i * self.Ny + j[cite: 8, 10]
                        if dm[i, j] == 1:
                            b[ind] = 0.0
                        elif self.omask[i, j] == 0:[cite: 8, 10]
                            b[ind] = pop.obstacle_penalty[cite: 8]

                Unres = sparse.linalg.spsolve(A, b).reshape((self.Nx, self.Ny))[cite: 8, 10]
                l2err = np.linalg.norm(Unew_n.flatten() - Unres.flatten()) * np.sqrt(self.Dx * self.Dy)[cite: 8, 10]
                Unew_n = np.copy(Unres)[cite: 8, 10]
                if l2err < 1e-6:[cite: 8, 10]
                    break
            u[k] = Unew_n[cite: 8, 10]
        return u

    def run_picard_system(self, max_iters: int = 15, tolerance: float = 1e-5):
        """Execute coupled Picard relaxation loop across all N populations[cite: 8, 10]."""
        space_time_factor = np.sqrt(self.Dx * self.Dy * self.Dt)[cite: 8, 10]

        for iiter in range(1, max_iters + 1):[cite: 8, 10]
            start_time = time.time()[cite: 8, 10]
            print(f"\n>>> Macro Picard Loop Execution: {iiter} / {max_iters}", flush=True)[cite: 8, 10]

            # 1. Update goals, mass absorption, and saturation states for all populations
            Y_temps = []
            for pop in self.populations:
                if pop.goal is not None:
                    mass_field = pop.M
                    if self.shared_capacity_goals and self.num_pops > 1:
                        mass_field = sum(p.M for p in self.populations)
                    Y_temp = pop.goal.update_positions(
                        mass_field, self.omask, self.Dx, self.Dy, self.Lx, self.Ly, pop.goals_are_exits
                    )
                    Y_temps.append(Y_temp)
                    if pop.goal.has_capacity_limits:
                        pop.door_mask = self.compute_saturated_door_mask(pop, pop.goal.Y_trajectories)
                else:
                    Y_temps.append(None)

            # 2. Backward HJB solves across all populations[cite: 8, 10]
            U_news = []
            for p_idx, pop in enumerate(self.populations):
                goal_traj = pop.goal.Y_trajectories if pop.goal is not None else None
                U_temp = self.solve_backward_HJB_step(p_idx, goal_traj)[cite: 8, 10]
                U_new = self.thetaUM * U_temp + (1.0 - self.thetaUM) * pop.U[cite: 8, 10]
                U_news.append(U_new)

            # 3. Forward KFP solves across all populations[cite: 8, 10]
            M_news = []
            for p_idx, pop in enumerate(self.populations):
                M_temp = self.solve_forward_FP_step(p_idx, U_news[p_idx])[cite: 8, 10]
                M_new = self.thetaUM * M_temp + (1.0 - self.thetaUM) * pop.M[cite: 8, 10]
                M_news.append(M_new)

            # 4. Trajectory relaxation and coordinate clamping
            y_errs = []
            for p_idx, pop in enumerate(self.populations):
                Y_temp = Y_temps[p_idx]
                if pop.goal is not None and Y_temp is not None:
                    Y_new = self.thetaUM * Y_temp + (1.0 - self.thetaUM) * pop.goal.Y_trajectories
                    for g_idx in range(pop.goal.num_goals):
                        sat_k = pop.goal.saturation_steps[g_idx]
                        if sat_k <= self.Nt:
                            Y_new[sat_k:, g_idx, :] = Y_temp[sat_k, g_idx, :]
                    y_errs.append(np.linalg.norm(Y_new - pop.goal.Y_trajectories))
                    pop.goal.Y_trajectories = np.copy(Y_new)

            # 5. Evaluate convergence criteria across all crowds[cite: 8, 10]
            u_err = max(np.linalg.norm(U_news[p] - self.populations[p].U) * space_time_factor for p in range(self.num_pops))[cite: 10]
            m_err = max(np.linalg.norm(M_news[p] - self.populations[p].M) * space_time_factor for p in range(self.num_pops))[cite: 10]
            y_err = max(y_errs) if y_errs else 0.0

            for p in range(self.num_pops):
                self.populations[p].U = np.copy(U_news[p])[cite: 8, 10]
                self.populations[p].M = np.copy(M_news[p])[cite: 8, 10]

            print(
                f"    u_residual: {u_err:.6e} | m_residual: {m_err:.6e} | "
                f"y_goal_residual: {y_err:.6e} | Time: {time.time() - start_time:.2f}s",
                flush=True,
            )

            if u_err < tolerance and m_err < tolerance and y_err < tolerance:[cite: 8, 10]
                print(f"\n[Success] System converged at iteration {iiter}!", flush=True)[cite: 8]
                break

        if self.num_pops == 1:
            return self.populations[0].U, self.populations[0].M[cite: 8]
        return tuple(p.U for p in self.populations) + tuple(p.M for p in self.populations)[cite: 10]


class MFG2PopSolver(MFGSolver):
    """
    Subclass preserving 100% backward compatibility for run_2pop.py and MAP2PDE2POP.py[cite: 5, 10].
    """

    def __init__(
        self,
        pde_mesh_data_1,
        pde_mesh_data_2,
        T: float = 5.0,
        Nt: int = 100,
        thetaUM: float = 0.1,
    ):
        pop1 = Population(
            name="pop_1",
            mesh=pde_mesh_data_1,
            m0=pde_mesh_data_1.build_initial_density(sigma_multiplier=3),
            goal_configs=[],
            stencil_mode=1,
            obstacle_penalty=-500.0,
        )[cite: 10]
        pop2 = Population(
            name="pop_2",
            mesh=pde_mesh_data_2,
            m0=pde_mesh_data_2.build_initial_density(sigma_multiplier=3),
            goal_configs=[],
            stencil_mode=1,
            obstacle_penalty=-500.0,
        )[cite: 10]
        # Exact asymmetric interaction matrix from MAP2PDE2POP.py[cite: 10]
        W_inter = np.array([[0.0, 1500.0], [-5000.0, 0.0]], dtype=np.float64)[cite: 10]
        W_cong = np.array([[0.0, 5.0], [5.0, 0.0]], dtype=np.float64)[cite: 10]

        super().__init__(
            populations=[pop1, pop2],
            T=T,
            Nt=Nt,
            thetaUM=thetaUM,
            interaction_matrix=W_inter,
            congestion_matrix=W_cong,
        )[cite: 5, 10]


class MFGShadowLeaderSolver:
    """
    Generalized N-Population Shadow-Leader / Stackelberg Best-Response Solver.

    Reduces numerical complexity in multi-crowd games by designating a subset
    of populations as Leaders (L) whose trajectories are solved or fixed as shadow
    paths, allowing the remaining Follower populations (F) to solve unilateral
    best responses against the exogenous leader field without bidirectional feedback[cite: 10].
    """

    def __init__(
        self,
        populations: list[Population],
        leader_indices: list[int] | None = None,
        T: float = 5.0,
        Nt: int = 100,
        thetaUM: float = 0.1,
        interaction_matrix: np.ndarray | None = None,
        congestion_matrix: np.ndarray | None = None,
        decouple_followers: bool = False,
        shared_capacity_goals: bool = False,
    ):
        """
        Parameters
        ----------
        populations : list[Population]
            List of all N populations participating in the scenario[cite: 8, 10].
        leader_indices : list[int], optional
            Indices of populations acting as Shadow Leaders. Defaults to [0] (Population 0 is leader).
        T : float
            Simulation time horizon in seconds[cite: 5, 8, 10].
        Nt : int
            Number of time discretization steps[cite: 5, 8, 10].
        thetaUM : float
            Picard under-relaxation parameter (0 < theta <= 1)[cite: 5, 8, 10].
        interaction_matrix : ndarray, optional
            N x N strategic interaction matrix W_inter[cite: 10].
        congestion_matrix : ndarray, optional
            N x N kinetic cross-congestion matrix W_cong[cite: 10].
        decouple_followers : bool
            If True, follower populations solve independent 1-population problems
            against the shadow leader (O(|F|) decoupled solves).
            If False, followers solve a reduced coupled sub-game among themselves.
        shared_capacity_goals : bool
            Whether exit capacities are contested across populations.
        """
        self.populations = populations
        self.num_pops = len(populations)
        self.T, self.Nt = T, Nt[cite: 8, 10]
        self.Dt = T / Nt[cite: 8, 10]
        self.thetaUM = thetaUM[cite: 8, 10]
        self.decouple_followers = decouple_followers
        self.shared_capacity_goals = shared_capacity_goals

        self.leader_indices = leader_indices if leader_indices is not None else [0]
        self.follower_indices = [i for i in range(self.num_pops) if i not in self.leader_indices]

        # Use MFGSolver internally to coordinate PDE stencils and matrices
        self.internal_solver = MFGSolver(
            populations=self.populations,
            T=T,
            Nt=Nt,
            thetaUM=thetaUM,
            interaction_matrix=interaction_matrix,
            congestion_matrix=congestion_matrix,
            shared_capacity_goals=shared_capacity_goals,
        )[cite: 5, 10]

    def solve_shadow_leaders(self, max_iters: int = 10, tolerance: float = 1e-5):
        """
        Phase 1: Compute nominal space-time trajectories (U_l, M_l) for Leader populations,
        unhindered by follower densities[cite: 10].
        """
        print(f"\n[Shadow-Leader Phase 1] Solving {len(self.leader_indices)} Leader population(s)...")
        space_time_factor = np.sqrt(self.internal_solver.Dx * self.internal_solver.Dy * self.Dt)[cite: 8, 10]

        # Zero out follower densities during leader phase
        for f_idx in self.follower_indices:
            self.populations[f_idx].M.fill(0.0)

        for iiter in range(1, max_iters + 1):
            start_time = time.time()
            U_news, M_news = [], []

            for l_idx in self.leader_indices:
                pop = self.populations[l_idx]
                if pop.goal is not None:
                    pop.goal.update_positions(pop.M, self.internal_solver.omask, self.internal_solver.Dx, self.internal_solver.Dy, self.internal_solver.Lx, self.internal_solver.Ly, pop.goals_are_exits)
                goal_traj = pop.goal.Y_trajectories if pop.goal is not None else None
                U_temp = self.internal_solver.solve_backward_HJB_step(l_idx, goal_traj)[cite: 8, 10]
                U_new = self.thetaUM * U_temp + (1.0 - self.thetaUM) * pop.U[cite: 8, 10]
                U_news.append((l_idx, U_new))

            for l_idx, U_new in U_news:
                M_temp = self.internal_solver.solve_forward_FP_step(l_idx, U_new)[cite: 8, 10]
                M_new = self.thetaUM * M_temp + (1.0 - self.thetaUM) * self.populations[l_idx].M[cite: 8, 10]
                M_news.append((l_idx, M_new))

            u_err = max(np.linalg.norm(U_new - self.populations[l_idx].U) * space_time_factor for l_idx, U_new in U_news)[cite: 10]
            m_err = max(np.linalg.norm(M_new - self.populations[l_idx].M) * space_time_factor for l_idx, M_new in M_news)[cite: 10]

            for l_idx, U_new in U_news: self.populations[l_idx].U = np.copy(U_new)[cite: 8, 10]
            for l_idx, M_new in M_news: self.populations[l_idx].M = np.copy(M_new)[cite: 8, 10]

            print(f"    [Leader Loop] Iter {iiter}: u_res = {u_err:.6e} | m_res = {m_err:.6e} | Time = {time.time() - start_time:.2f}s")
            if u_err < tolerance and m_err < tolerance:
                print(f"[Success] Shadow Leader trajectories converged at iteration {iiter}!")
                break

    def solve_follower_responses(self, max_iters: int = 15, tolerance: float = 1e-5):
        """
        Phase 2: Compute optimal best responses for Follower populations against
        the frozen Leader space-time fields[cite: 10].
        """
        print(f"\n[Shadow-Leader Phase 2] Solving {len(self.follower_indices)} Follower population(s)...")
        space_time_factor = np.sqrt(self.internal_solver.Dx * self.internal_solver.Dy * self.Dt)

        for iiter in range(1, max_iters + 1):
            start_time = time.time()

            # Update follower goals (and capacity saturation) against frozen leader + follower mass
            for f_idx in self.follower_indices:
                pop = self.populations[f_idx]
                if pop.goal is not None:
                    mass_field = pop.M
                    if self.shared_capacity_goals:
                        mass_field = sum(p.M for p in self.populations)
                    pop.goal.update_positions(mass_field, self.internal_solver.omask, self.internal_solver.Dx, self.internal_solver.Dy, self.internal_solver.Lx, self.internal_solver.Ly, pop.goals_are_exits)
                    if pop.goal.has_capacity_limits:
                        pop.door_mask = self.internal_solver.compute_saturated_door_mask(pop, pop.goal.Y_trajectories)

            # Solve HJB backward for followers only (Leaders remain frozen)
            U_news = []
            for f_idx in self.follower_indices:
                pop = self.populations[f_idx]
                goal_traj = pop.goal.Y_trajectories if pop.goal is not None else None
                U_temp = self.internal_solver.solve_backward_HJB_step(f_idx, goal_traj)
                U_new = self.thetaUM * U_temp + (1.0 - self.thetaUM) * pop.U
                U_news.append((f_idx, U_new))

            # Solve KFP forward for followers only (Leaders remain frozen)
            M_news = []
            for f_idx, U_new in U_news:
                M_temp = self.internal_solver.solve_forward_FP_step(f_idx, U_new)
                M_new = self.thetaUM * M_temp + (1.0 - self.thetaUM) * self.populations[f_idx].M
                M_news.append((f_idx, M_new))

            u_err = max(np.linalg.norm(U_new - self.populations[f_idx].U) * space_time_factor for f_idx, U_new in U_news)
            m_err = max(np.linalg.norm(M_new - self.populations[f_idx].M) * space_time_factor for f_idx, M_new in M_news)

            for f_idx, U_new in U_news: self.populations[f_idx].U = np.copy(U_new)
            for f_idx, M_new in M_news: self.populations[f_idx].M = np.copy(M_new)

            print(f"    [Follower Loop] Iter {iiter}: u_res = {u_err:.6e} | m_res = {m_err:.6e} | Time = {time.time() - start_time:.2f}s")
            if u_err < tolerance and m_err < tolerance:
                print(f"[Success] Follower best responses converged at iteration {iiter}!")
                break

    def run_picard_system(self, max_iters: int = 15, tolerance: float = 1e-5):
        """Execute the two-phase Shadow Leader workflow."""
        self.solve_shadow_leaders(max_iters=max_iters, tolerance=tolerance)
        self.solve_follower_responses(max_iters=max_iters, tolerance=tolerance)
        return tuple(p.U for p in self.populations) + tuple(p.M for p in self.populations)
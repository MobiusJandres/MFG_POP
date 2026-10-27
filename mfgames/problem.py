"""
Mean Field Games Generalized N-Population Problem Solvers.

This module provides high-level object-oriented numerical solvers for arbitrary
N-population Mean Field Game (MFG) systems. The framework models the collective
behavior of large crowds of rational, interacting agents under the macroscopic
continuum limit, coupling backward Hamilton-Jacobi-Bellman (HJB) equations with
forward Kolmogorov-Fokker-Planck (KFP) equations via damped Picard fixed-point
iteration.

Mathematical Formulation
-------------------------
For an arbitrary system of N continuum populations indexed by p ∈ {1, ..., N},
each individual agent in crowd p chooses a velocity control field v_p(t, x) to
minimize a spatio-temporal cost functional:

    J_p(v) = ∫_0^T [ f_p(x(t), t) + C_inter,p(m_1, ..., m_N) + (1/2)|v_p(t)|^2 ] dt + g_p(x(T))

where:
    - f_p(x, t) is the running cost field (distance to active targets and saturation barriers).
    - C_inter,p is the strategic inter-crowd interaction potential (attraction/repulsion).
    - g_p(x) is the terminal boundary cost evaluated at t = T.

The macroscopic equilibrium corresponds to a coupled PDE system on (0, T) × Ω:

1. Backward Hamilton-Jacobi-Bellman (HJB) Equations (Optimal Control):
       -∂u_p/∂t - ν Δu_p + H_p(m_1, ..., m_N, ∇u_p) = f_p(x, t) + C_inter,p(m)
   subject to the terminal boundary condition:
       u_p(T, x) = g_p(x)
   and Dirichlet boundary condition u_p(t, x) = 0 on active exit doors Γ_door,p(t).

2. Forward Kolmogorov-Fokker-Planck (KFP) Equations (Mass Conservation):
       ∂m_p/∂t - ν Δm_p - div( m_p * c_h,p(m) * ∇u_p ) = 0
   subject to the initial density distribution:
       m_p(0, x) = m_{0,p}(x)
   with homogeneous Neumann conditions on solid obstacles Γ_wall and Dirichlet
   mass absorption (m_p = 0) on active exit doors Γ_door,p(t).

Classes
-------
- Population : Data container encapsulating a single crowd's spatial mesh, initial
  density, decoupled running and terminal costs, kinematic parameters, and goal manager.
- MFGSolver : Generalized Nash equilibrium solver for arbitrary N >= 1 populations.
- MFG2PopSolver : Backward-compatible wrapper reproducing the competitive 2-population
  pursuit-evasion scenario from MAP2PDE2POP.py.
- MFGShadowLeaderSolver : Scalable Stackelberg / Shadow-Leader best-response solver
  that freezes designated leader trajectories to decouple follower solves.
"""

from dataclasses import dataclass, field
import time
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
from mfgames.objectives import Goal
from mfgames.numerics import (
    assemble_FP_matrix,
    evaluate_HJB_residual,
    assemble_HJB_jacobian,
)


@dataclass
class Population:
    """
    Data container representing a single crowd population in an N-population game.

    Encapsulates all physical parameters, numerical stencils, spatial discretization
    grids, initial conditions, decoupled cost functional weights, and dynamic goal
    specifications for one independent or interacting crowd.

    Attributes
    ----------
    name : str
        Descriptive label for the population (e.g., 'pursuers', 'evaders', 'team_alpha').
    mesh : MAP2PDE or MFGTrafficGeometry
        Spatial discretization object providing grid coordinates (X, Y), cell step
        sizes (dx, dy), domain dimensions (Lx, Ly), and obstacle masks (omask).
    m0 : ndarray of shape (Nx, Ny), optional
        Initial macroscopic population density grid. If None, built from mesh.build_initial_density().
    goal_configs : list of dict, optional
        List of goal dictionaries specifying kinematics ('stationary', 'prescribed',
        or 'evader'), initial positions, speed limits (v_max), and capacity limits.
    goals_are_exits : bool, default=False
        If True, active goals act as absorbing exit boundaries enforcing Dirichlet conditions
        u = 0 (zero cost-to-go) and m = 0 (mass absorption).
    running_cost_weight : float, default=0.01
        Scaling weight w_R for spatial running distance costs accumulated during t < T.
    terminal_cost_weight : float, default=0.01
        Scaling weight w_T for terminal distance costs evaluated at final time t = T.
    running_cost_power : int, default=2
        Distance exponent p_R for running cost (1 for linear distance, 2 for squared Euclidean distance).
    terminal_cost_power : int, default=2
        Distance exponent p_T for terminal cost (1 for linear distance, 2 for squared Euclidean distance).
    saturated_goal_penalty : float, default=0.0
        Magnitude of the repulsive Gaussian potential barrier erected around goals that have
        exceeded their capacity limit.
    obstacle_penalty : float, default=-500.0
        Potential penalty applied to obstacle cells in the HJB residual to prevent penetration.
    alpha : float, default=8.0
        Hamiltonian kinetic energy mobility coefficient numerator.
    beta : float, default=0.75
        Congestion exponent in the mobility denominator c_h = gamma / (1 + m)^beta.
    h0 : float, default=0.1
        Baseline running cost offset in the Hamiltonian.
    gamma : float, default=16.0
        Kinetic mobility coefficient numerator for the forward Fokker-Planck drift flux.
    upwind_mode : int, default=0
        Numerical directional upwind scheme:
            0: Standard gradient advection (used in 1-population and pursuit-evasion models).
            1: Reversed gradient advection (used in legacy competitive two-population games).
    U : ndarray of shape (Nt+1, Nx, Ny), optional
        Value function space-time solution tensor, allocated by solver during initialization.
    M : ndarray of shape (Nt+1, Nx, Ny), optional
        Density space-time solution tensor, allocated by solver during initialization.
    goal : Goal, optional
        Instantiated goal kinematics and capacity saturation manager.
    door_mask : ndarray of shape (Nt+1, Nx, Ny), optional
        Time-dependent exit door mask (1 for exit, 0 for walkable interior).
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

    # Physical mobility and numerical stencil parameters
    alpha: float = 8.0
    beta: float = 0.75
    h0: float = 0.1
    gamma: float = 16.0
    upwind_mode: int = 0

    # State solution tensors allocated during solver initialization
    U: np.ndarray = field(default=None, repr=False)
    M: np.ndarray = field(default=None, repr=False)
    goal: Goal = field(default=None, repr=False)
    door_mask: np.ndarray = field(default=None, repr=False)


class MFGSolver:
    """
    Generalized N-Population Mean Field Game Solver.

    Solves coupled systems of HJB and KFP partial differential equations for arbitrary
    N >= 1 populations via damped Picard iteration with post-saturation coordinate clamping.
    Provides 100% backward compatibility for single-population scripts and competitive 2-population runs.
    """

    def __init__(
        self,
        populations: list[Population] | None = None,
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
        interaction_matrix: np.ndarray | None = None,
        congestion_matrix: np.ndarray | None = None,
        shared_capacity_goals: bool = False,
    ):
        """
        Initialize the generalized N-population Mean Field Game solver.

        Parameters
        ----------
        populations : list of Population, optional
            List of Population objects participating in the simulation. If provided,
            multi-population mode is activated.
        pde_mesh_data : MAP2PDE or MFGTrafficGeometry, optional
            Convenience spatial mesh for single-population runs (backward-compatible).
        T : float, default=3.0
            Total time horizon in seconds.
        Nt : int, default=100
            Number of discrete time intervals (giving Dt = T / Nt).
        thetaUM : float, default=0.1
            Picard under-relaxation mixing parameter (0 < thetaUM <= 1).
        door_mask : ndarray of shape (Nt+1, Nx, Ny), optional
            Static or dynamic exit door mask for single-population runs.
        door_mask_3d : ndarray, optional
            Alias for door_mask.
        goal_configs : list of dict, optional
            Goal specifications for single-population runs.
        goals_are_exits : bool, default=False
            Whether active goals enforce absorbing exit boundary conditions.
        obstacle_penalty : float, optional
            Potential penalty value enforced at obstacle cells (default: -500.0).
        running_cost_weight : float, default=0.01
            Running cost scaling weight w_R for single-population runs.
        terminal_cost_weight : float, optional
            Terminal cost scaling weight w_T. Defaults to running_cost_weight if None.
        running_cost_power : int, default=2
            Distance power p_R for single-population running cost.
        terminal_cost_power : int, default=2
            Distance power p_T for single-population terminal cost.
        saturated_goal_penalty : float, default=0.0
            Repulsive barrier magnitude around filled goals for single-population runs.
        interaction_matrix : ndarray of shape (N, N), optional
            Strategic interaction matrix W_inter defining cross-population attraction (< 0)
            or repulsion (> 0) potentials.
        congestion_matrix : ndarray of shape (N, N), optional
            Kinetic congestion matrix W_cong defining cross-population mobility friction.
        shared_capacity_goals : bool, default=False
            If True, cumulative mass from all populations is pooled together when evaluating
            goal capacity thresholds.
        """
        self.T, self.Nt = T, Nt
        self.Dt = T / Nt
        self.thetaUM = thetaUM
        self.shared_capacity_goals = shared_capacity_goals

        # Normalize single-population inputs into the unified Population container
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
                alpha=8.0,
                beta=0.75,
                h0=0.1,
                gamma=16.0,
                upwind_mode=0,
            )
            self.populations = [pop1]
        else:
            raise ValueError("Must provide either 'populations' or 'pde_mesh_data'.")

        self.num_pops = len(self.populations)
        self.Nx, self.Ny = primary_mesh.X.shape
        self.Lx, self.Ly = primary_mesh.Lx, primary_mesh.Ly
        self.Dx, self.Dy = primary_mesh.dx, primary_mesh.dy
        self.omask = primary_mesh.get_pde_obstacle_mask()

        # Strategic interaction and cross-congestion matrices
        if interaction_matrix is not None:
            self.interaction_matrix = np.asarray(interaction_matrix, dtype=np.float64)
        else:
            self.interaction_matrix = np.zeros((self.num_pops, self.num_pops), dtype=np.float64)

        if congestion_matrix is not None:
            self.congestion_matrix = np.asarray(congestion_matrix, dtype=np.float64)
        else:
            self.congestion_matrix = np.zeros((self.num_pops, self.num_pops), dtype=np.float64)

        initial_door_mask = door_mask if door_mask is not None else door_mask_3d

        # Allocate state solution arrays and instantiate Goal managers for each population
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
                # Initialize terminal slice to prevent Picard under-relaxation drift
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
    def M(self):
        """Density tensor M of the primary population, shape (Nt+1, Nx, Ny)."""
        return self.populations[0].M

    @M.setter
    def M(self, val):
        self.populations[0].M = val

    @property
    def U(self):
        """Value function tensor U of the primary population, shape (Nt+1, Nx, Ny)."""
        return self.populations[0].U

    @U.setter
    def U(self, val):
        self.populations[0].U = val

    @property
    def goal(self):
        """Goal manager instance of the primary population."""
        return self.populations[0].goal

    @property
    def door_mask(self):
        """Exit door mask of the primary population, shape (Nt+1, Nx, Ny)."""
        return self.populations[0].door_mask

    @property
    def door_mask_3d(self):
        """Alias for door_mask."""
        return self.populations[0].door_mask

    @property
    def obstacle_penalty(self):
        """Obstacle cell penalty value of the primary population."""
        return self.populations[0].obstacle_penalty

    # --- Backward-Compatibility Properties for 2-Population Runs ---
    @property
    def M1(self):
        """Population 1 density tensor, shape (Nt+1, Nx, Ny)."""
        return self.populations[0].M

    @property
    def M2(self):
        """Population 2 density tensor, shape (Nt+1, Nx, Ny)."""
        return self.populations[1].M if self.num_pops > 1 else None

    @property
    def U1(self):
        """Population 1 value function tensor, shape (Nt+1, Nx, Ny)."""
        return self.populations[0].U

    @property
    def U2(self):
        """Population 2 value function tensor, shape (Nt+1, Nx, Ny)."""
        return self.populations[1].U if self.num_pops > 1 else None

    def _build_dynamic_doors(self, pop: Population, goal_trajectories: np.ndarray) -> np.ndarray:
        """
        Construct the time-indexed binary door mask for goals configured as absorbing exits.

        For each timestep k ∈ {0, ..., Nt} and each goal g marked with is_exit=True,
        identifies spatial grid cells within one grid spacing (Dx, Dy) of the goal coordinates
        (gx, gy) and marks them as absorbing doors (door_mask[k, i, j] = 1.0).

        Parameters
        ----------
        pop : Population
            Target population owning the goal manager.
        goal_trajectories : ndarray of shape (Nt+1, G, 2)
            Interpolated space-time coordinates for all G goals across all timesteps.

        Returns
        -------
        door_mask : ndarray of shape (Nt+1, Nx, Ny), dtype float64
            Binary door mask tensor where 1.0 indicates an active exit cell.
        """
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

    def compute_saturated_door_mask(self, pop: Population, goal_trajectories: np.ndarray) -> np.ndarray:
        """
        Construct dynamic door mask reflecting goal capacity saturation states.

        Directly queries Goal.is_saturated[k, g] to deactivate exit cells for goals that
        have reached their capacity ceiling C_max. Once a goal saturates, its exit region
        reverts to standard walkable space (door_mask = 0.0), terminating Dirichlet mass absorption.

        Parameters
        ----------
        pop : Population
            Target population owning the saturated goal manager.
        goal_trajectories : ndarray of shape (Nt+1, G, 2)
            Goal coordinates across all timesteps.

        Returns
        -------
        door_mask : ndarray of shape (Nt+1, Nx, Ny), dtype float64
            Updated exit door mask reflecting deactivations.
        """
        door_mask = np.zeros((self.Nt + 1, self.Nx, self.Ny), dtype=np.float64)
        if not pop.goal or goal_trajectories is None:
            return door_mask

        X, Y = pop.mesh.X, pop.mesh.Y
        for k in range(self.Nt + 1):
            for g_idx, g_info in enumerate(pop.goal.goals):
                if not g_info.get('is_exit', False):
                    continue
                # Keep door open only if capacity limit has not been breached
                if not pop.goal.is_saturated[k, g_idx]:
                    gx, gy = goal_trajectories[k, g_idx]
                    region = (np.abs(X - gx) <= self.Dx) & (np.abs(Y - gy) <= self.Dy)
                    door_mask[k][region] = 1.0
        return door_mask

    def compute_running_cost(self, pop: Population, goal_positions_k: np.ndarray, k: int | None = None) -> np.ndarray:
        """
        Compute the spatial running cost field f_p(x, t) for population p at timestep k.

        Evaluates the distance-based running penalty and saturated goal repulsion barrier:
            f_p(x, t) = w_R * min_{g ∈ A(t)} ||x - y_g(t)||^{p_R} + S_p(x, t)

        where:
        - A(t) is the active goal set (goals that have not saturated by time step k).
        - S_p(x, t) is a repulsive Gaussian barrier erected around saturated goals:
              S_p(x, t) = w_sat * sum_{g ∉ A(t)} exp( -||x - y_g(t)||^2 / (2 * σ_sat^2) )
          with spatial radius σ_sat = 10 * Dx.

        Parameters
        ----------
        pop : Population
            Target population.
        goal_positions_k : ndarray of shape (G, 2)
            Positions of all G goals at timestep k.
        k : int, optional
            Current time step index used to query Goal.is_saturated.

        Returns
        -------
        running_cost : ndarray of shape (Nx, Ny), dtype float64
            2D running cost field. Returns zeros if running_cost_weight == 0.0.
        """
        if pop.running_cost_weight == 0.0 or goal_positions_k is None or len(goal_positions_k) == 0:
            return np.zeros((self.Nx, self.Ny), dtype=np.float64)

        X, Y = pop.mesh.X, pop.mesh.Y
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

    def compute_terminal_cost(self, pop: Population, goal_positions_Nt: np.ndarray) -> np.ndarray:
        """
        Compute terminal boundary condition g_p(x) = u_p(T, x) at final time t = T.

        Evaluates the terminal distance cost:
            g_p(x) = w_T * min_{g ∈ A(T)} ||x - y_g(T)||^{p_T} + S_p(x, T)

        Saturated goals are excluded from the attraction minimum and exert a local
        Gaussian repulsion barrier if saturated_goal_penalty > 0.

        Parameters
        ----------
        pop : Population
            Target population.
        goal_positions_Nt : ndarray of shape (G, 2)
            Positions of all G goals at the terminal time step Nt.

        Returns
        -------
        terminal_cost : ndarray of shape (Nx, Ny), dtype float64
            2D terminal cost grid. Returns zeros if terminal_cost_weight == 0.0.
        """
        if pop.terminal_cost_weight == 0.0 or goal_positions_Nt is None or len(goal_positions_Nt) == 0:
            return np.zeros((self.Nx, self.Ny), dtype=np.float64)

        X, Y = pop.mesh.X, pop.mesh.Y
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

    def _get_coupling_fields(self, p_idx: int, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute effective other-crowd congestion and strategic interaction potential fields.

        For population p at time step k, computes:
        1. Cross-congestion field entering the mobility denominator:
               m_cong(x) = sum_{q ≠ p} W_{pq}^{cong} * m_q(t_k, x)
        2. Strategic interaction cost entering the HJB potential:
               C_inter,p(x) = sum_{q ≠ p} W_{pq}^{inter} * m_q(t_k, x) + max(sum_q m_q(x) - 4.0, 0.0)

        Parameters
        ----------
        p_idx : int
            Index of the target population.
        k : int
            Current discrete time step index.

        Returns
        -------
        other_congestion : ndarray of shape (Nx, Ny), dtype float64
            Cross-population density field weighted by W_cong.
        interaction_cost : ndarray of shape (Nx, Ny), dtype float64
            Spatial potential field combining strategic interactions and physical congestion limits.
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
                interaction_cost += w_i * mq_k

        # Overcrowding penalty activated when aggregate density exceeds 4.0 agents/m^2
        if self.num_pops > 1:
            interaction_cost += np.maximum(total_density - 4.0, 0.0)

        return other_congestion, interaction_cost

    def solve_forward_FP_step(self, p_idx: int, U_p: np.ndarray) -> np.ndarray:
        """
        Solve the forward Kolmogorov-Fokker-Planck equation for population p across all timesteps.

        Solves the linear transport system forward in time from k = 1 to Nt:
            A * m^{k} = m^{k-1} / Dt
        using the unified assemble_FP_matrix stencil with JIT acceleration.

        Parameters
        ----------
        p_idx : int
            Index of the population being advanced.
        U_p : ndarray of shape (Nt+1, Nx, Ny)
            Value function trajectory prescribing the advective drift field.

        Returns
        -------
        m : ndarray of shape (Nt+1, Nx, Ny), dtype float64
            Updated macroscopic density distribution across space and time.
        """
        pop = self.populations[p_idx]
        m = np.zeros_like(pop.M)
        m[0] = pop.m0
        N_total = self.Nx * self.Ny

        for k in range(1, self.Nt + 1):
            dm = pop.door_mask[k - 1] if pop.door_mask is not None else np.zeros((self.Nx, self.Ny))
            other_cong, _ = self._get_coupling_fields(p_idx, k - 1)

            rows, cols, vals, b = assemble_FP_matrix(
                m[k - 1], other_cong, U_p[k - 1], self.omask, dm,
                self.Nx, self.Ny, self.Dx, self.Dy, self.Dt,
                gamma=pop.gamma, beta=pop.beta, upwind_mode=pop.upwind_mode
            )

            A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()
            mtmp = sparse.linalg.spsolve(A, b)
            m[k] = mtmp.reshape((self.Nx, self.Ny))
        return m

    def solve_backward_HJB_step(self, p_idx: int, goal_trajectories: np.ndarray | None) -> np.ndarray:
        """
        Solve the backward Hamilton-Jacobi-Bellman equation for population p.

        Steps backward in time from k = Nt - 1 down to 0, applying Newton-Raphson iteration
        at each timestep to resolve the nonlinear Hamiltonian:
            A(u^n) * δu = -F(u^n)
            u^{n, new} = u^n + δu
        using evaluate_HJB_residual and assemble_HJB_jacobian from the unified numerics module.

        Parameters
        ----------
        p_idx : int
            Index of the population being solved.
        goal_trajectories : ndarray of shape (Nt+1, G, 2) or None
            Space-time goal coordinates used to evaluate terminal and running costs.

        Returns
        -------
        u : ndarray of shape (Nt+1, Nx, Ny), dtype float64
            Updated value function trajectory across space and time.
        """
        pop = self.populations[p_idx]
        u = np.zeros_like(pop.U)
        N_total = self.Nx * self.Ny

        # Terminal boundary condition u(T, x) = g(x)
        if goal_trajectories is not None:
            u[self.Nt] = self.compute_terminal_cost(pop, goal_trajectories[self.Nt])
        else:
            u[self.Nt] = np.zeros((self.Nx, self.Ny), dtype=np.float64)

        # Backward time-stepping loop
        for k in range(self.Nt - 1, -1, -1):
            running_cost_k = (
                self.compute_running_cost(pop, goal_trajectories[k], k=k)
                if goal_trajectories is not None
                else np.zeros((self.Nx, self.Ny), dtype=np.float64)
            )
            dm = pop.door_mask[k] if pop.door_mask is not None else np.zeros((self.Nx, self.Ny))
            Unew_n = np.copy(u[k + 1])
            other_cong, inter_cost = self._get_coupling_fields(p_idx, k + 1)
            total_cost_k = running_cost_k + inter_cost

            # Newton-Raphson inner loop
            max_newton = 30 if pop.upwind_mode == 0 else 5
            for _ in range(max_newton):
                FnU_flat = evaluate_HJB_residual(
                    u[k + 1], Unew_n, pop.M[k + 1], other_cong, self.omask, dm,
                    total_cost_k, self.Nx, self.Ny, self.Dx, self.Dy, self.Dt,
                    obstacle_penalty=pop.obstacle_penalty,
                    alpha=pop.alpha, beta=pop.beta, h0=pop.h0, upwind_mode=pop.upwind_mode
                ).flatten()

                rows, cols, vals = assemble_HJB_jacobian(
                    Unew_n, pop.M[k + 1], other_cong, self.omask, dm,
                    self.Nx, self.Ny, self.Dx, self.Dy, self.Dt,
                    alpha=pop.alpha, beta=pop.beta, upwind_mode=pop.upwind_mode
                )

                A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()
                b = A.dot(Unew_n.flatten()) - FnU_flat

                # Enforce boundary conditions on linear RHS vector
                for i in range(self.Nx):
                    for j in range(self.Ny):
                        ind = i * self.Ny + j
                        if dm[i, j] == 1:
                            b[ind] = 0.0
                        elif self.omask[i, j] == 0:
                            b[ind] = pop.obstacle_penalty

                Unres = sparse.linalg.spsolve(A, b).reshape((self.Nx, self.Ny))
                l2err = np.linalg.norm(Unew_n.flatten() - Unres.flatten()) * np.sqrt(self.Dx * self.Dy)
                Unew_n = np.copy(Unres)
                if l2err < 1e-6:
                    break
            u[k] = Unew_n
        return u

    def run_picard_system(self, max_iters: int = 15, tolerance: float = 1e-5):
        """
        Execute the coupled Picard fixed-point under-relaxation loop across all N populations.

        Coordinates the five-step macro iteration loop:
        1. Goal Motion & Saturation: Updates reactive goal positions and closes saturated doors.
        2. Backward HJB Solves: Computes U_temp for each crowd and blends via U = θ*U_temp + (1-θ)*U.
        3. Forward KFP Solves: Computes M_temp for each crowd and blends via M = θ*M_temp + (1-θ)*M.
        4. Trajectory Relaxation & Clamping: Relaxes goal trajectories and enforces hard geometric
           coordinate clamping at saturation locations (y_g(t) = y_g(t_sat) for all t >= t_sat).
        5. Convergence Verification: Checks space-time L2 residual thresholds across all states.

        Parameters
        ----------
        max_iters : int, default=15
            Maximum allowable Picard macro iterations.
        tolerance : float, default=1e-5
            Convergence threshold for space-time L2 residual norms.

        Returns
        -------
        If num_pops == 1:
            tuple of (U, M) where U and M have shape (Nt+1, Nx, Ny).
        If num_pops > 1:
            tuple of (U_1, ..., U_N, M_1, ..., M_N).
        """
        space_time_factor = np.sqrt(self.Dx * self.Dy * self.Dt)

        for iiter in range(1, max_iters + 1):
            start_time = time.time()
            print(f"\n>>> Macro Picard Loop Execution: {iiter} / {max_iters}", flush=True)

            # Step 1: Update dynamic goals and capacity saturation
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

            # Step 2: Backward HJB solves across all populations
            U_news = []
            for p_idx, pop in enumerate(self.populations):
                goal_traj = pop.goal.Y_trajectories if pop.goal is not None else None
                U_temp = self.solve_backward_HJB_step(p_idx, goal_traj)
                U_new = self.thetaUM * U_temp + (1.0 - self.thetaUM) * pop.U
                U_news.append(U_new)

            # Step 3: Forward KFP solves across all populations
            M_news = []
            for p_idx, pop in enumerate(self.populations):
                M_temp = self.solve_forward_FP_step(p_idx, U_news[p_idx])
                M_new = self.thetaUM * M_temp + (1.0 - self.thetaUM) * pop.M
                M_news.append(M_new)

            # Step 4: Trajectory relaxation and post-saturation coordinate clamping
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

            # Step 5: Evaluate convergence criteria
            u_err = max(np.linalg.norm(U_news[p] - self.populations[p].U) * space_time_factor for p in range(self.num_pops))
            m_err = max(np.linalg.norm(M_news[p] - self.populations[p].M) * space_time_factor for p in range(self.num_pops))
            y_err = max(y_errs) if y_errs else 0.0

            for p in range(self.num_pops):
                self.populations[p].U = np.copy(U_news[p])
                self.populations[p].M = np.copy(M_news[p])

            print(
                f"    u_residual: {u_err:.6e} | m_residual: {m_err:.6e} | "
                f"y_goal_residual: {y_err:.6e} | Time: {time.time() - start_time:.2f}s",
                flush=True,
            )

            if u_err < tolerance and m_err < tolerance and y_err < tolerance:
                print(f"\n[Success] System converged at iteration {iiter}!", flush=True)
                break

        if self.num_pops == 1:
            return self.populations[0].U, self.populations[0].M
        return tuple(p.U for p in self.populations) + tuple(p.M for p in self.populations)


class MFG2PopSolver(MFGSolver):
    """
    Subclass preserving 100% backward compatibility for run_2pop.py and MAP2PDE2POP.py.

    Automatically sets up Population 1 (Pursuers) and Population 2 (Evaders) with the exact
    asymmetric interaction potential matrix W_inter and cross-congestion matrix W_cong:
        W_inter = [[0.0, +1500.0], [-5000.0, 0.0]]
        W_cong  = [[0.0,    +5.0], [   +5.0, 0.0]]
    """

    def __init__(
        self,
        pde_mesh_data_1,
        pde_mesh_data_2,
        T: float = 5.0,
        Nt: int = 100,
        thetaUM: float = 0.1,
    ):
        """
        Initialize the two-population pursuit-evasion solver.

        Parameters
        ----------
        pde_mesh_data_1 : MAP2PDE
            Mesh and scenario geometry for Population 1 (Pursuers).
        pde_mesh_data_2 : MAP2PDE
            Mesh and scenario geometry for Population 2 (Evaders).
        T : float, default=5.0
            Simulation time horizon in seconds.
        Nt : int, default=100
            Number of discrete time steps.
        thetaUM : float, default=0.1
            Picard under-relaxation parameter.
        """
        pop1 = Population(
            name="pop_1",
            mesh=pde_mesh_data_1,
            m0=pde_mesh_data_1.build_initial_density(sigma_multiplier=3),
            goal_configs=[],
            alpha=1.0,
            beta=1.0,
            h0=0.0,
            gamma=2.0,
            upwind_mode=1,
            obstacle_penalty=-500.0,
        )
        pop2 = Population(
            name="pop_2",
            mesh=pde_mesh_data_2,
            m0=pde_mesh_data_2.build_initial_density(sigma_multiplier=3),
            goal_configs=[],
            alpha=1.0,
            beta=1.0,
            h0=0.0,
            gamma=2.0,
            upwind_mode=1,
            obstacle_penalty=-500.0,
        )

        W_inter = np.array([[0.0, 1500.0], [-5000.0, 0.0]], dtype=np.float64)
        W_cong = np.array([[0.0, 5.0], [5.0, 0.0]], dtype=np.float64)

        super().__init__(
            populations=[pop1, pop2],
            T=T,
            Nt=Nt,
            thetaUM=thetaUM,
            interaction_matrix=W_inter,
            congestion_matrix=W_cong,
        )


class MFGShadowLeaderSolver:
    """
    Generalized N-Population Shadow-Leader / Stackelberg Best-Response Solver.

    Reduces computational complexity in multi-crowd games by designating a subset of
    populations as Leaders (L) whose trajectories are solved or prescribed as shadow
    paths. The remaining Follower populations (F) solve unilateral best responses
    against the frozen leader fields without bidirectional Picard feedback.
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
        shared_capacity_goals: bool = False,
    ):
        """
        Initialize the Shadow-Leader solver.

        Parameters
        ----------
        populations : list of Population
            All N populations participating in the scenario.
        leader_indices : list of int, optional
            Indices of populations acting as Shadow Leaders (default: [0]).
        T : float, default=5.0
            Simulation time horizon in seconds.
        Nt : int, default=100
            Number of time discretization steps.
        thetaUM : float, default=0.1
            Picard under-relaxation parameter.
        interaction_matrix : ndarray of shape (N, N), optional
            Strategic interaction matrix W_inter.
        congestion_matrix : ndarray of shape (N, N), optional
            Cross-congestion matrix W_cong.
        shared_capacity_goals : bool, default=False
            Whether exit capacities are contested across all swarms.
        """
        self.populations = populations
        self.num_pops = len(populations)
        self.T, self.Nt = T, Nt
        self.Dt = T / Nt
        self.thetaUM = thetaUM
        self.shared_capacity_goals = shared_capacity_goals

        self.leader_indices = leader_indices if leader_indices is not None else [0]
        self.follower_indices = [i for i in range(self.num_pops) if i not in self.leader_indices]

        self.internal_solver = MFGSolver(
            populations=self.populations,
            T=T,
            Nt=Nt,
            thetaUM=thetaUM,
            interaction_matrix=interaction_matrix,
            congestion_matrix=congestion_matrix,
            shared_capacity_goals=shared_capacity_goals,
        )

    @property
    def M1(self):
        return self.populations[0].M

    @property
    def M2(self):
        return self.populations[1].M if self.num_pops > 1 else None

    @property
    def U1(self):
        return self.populations[0].U

    @property
    def U2(self):
        return self.populations[1].U if self.num_pops > 1 else None

    @property
    def omask(self):
        return self.internal_solver.omask

    @property
    def Dx(self):
        return self.internal_solver.Dx

    @property
    def Dy(self):
        return self.internal_solver.Dy

    def solve_shadow_leaders(self, max_iters: int = 10, tolerance: float = 1e-5):
        """
        Phase 1: Solve Leader populations unhindered by follower densities.

        Follower densities are temporarily zeroed out to compute the Leader's nominal path.
        """
        print(f"\n[Shadow-Leader Phase 1] Solving {len(self.leader_indices)} Leader population(s)...")
        space_time_factor = np.sqrt(self.internal_solver.Dx * self.internal_solver.Dy * self.internal_solver.Dt)

        for f_idx in self.follower_indices:
            self.populations[f_idx].M.fill(0.0)

        for iiter in range(1, max_iters + 1):
            start_time = time.time()
            U_news, M_news = [], []

            for l_idx in self.leader_indices:
                pop = self.populations[l_idx]
                if pop.goal is not None:
                    pop.goal.update_positions(
                        pop.M, self.internal_solver.omask, self.internal_solver.Dx,
                        self.internal_solver.Dy, self.internal_solver.Lx, self.internal_solver.Ly,
                        pop.goals_are_exits
                    )
                goal_traj = pop.goal.Y_trajectories if pop.goal is not None else None
                U_temp = self.internal_solver.solve_backward_HJB_step(l_idx, goal_traj)
                U_new = self.thetaUM * U_temp + (1.0 - self.thetaUM) * pop.U
                U_news.append((l_idx, U_new))

            for l_idx, U_new in U_news:
                M_temp = self.internal_solver.solve_forward_FP_step(l_idx, U_new)
                M_new = self.thetaUM * M_temp + (1.0 - self.thetaUM) * self.populations[l_idx].M
                M_news.append((l_idx, M_new))

            u_err = max(np.linalg.norm(U_new - self.populations[l_idx].U) * space_time_factor for l_idx, U_new in U_news)
            m_err = max(np.linalg.norm(M_new - self.populations[l_idx].M) * space_time_factor for l_idx, M_new in M_news)

            for l_idx, U_new in U_news: self.populations[l_idx].U = np.copy(U_new)
            for l_idx, M_new in M_news: self.populations[l_idx].M = np.copy(M_new)

            print(f"    [Leader Loop] Iter {iiter}: u_res = {u_err:.6e} | m_res = {m_err:.6e} | Time = {time.time() - start_time:.2f}s")
            if u_err < tolerance and m_err < tolerance:
                print(f"[Success] Shadow Leader trajectories converged at iteration {iiter}!")
                break

    def solve_follower_responses(self, max_iters: int = 15, tolerance: float = 1e-5):
        """
        Phase 2: Solve Follower best responses against the frozen Leader space-time fields.

        Leader density and value fields are treated as fixed exogenous inputs.
        """
        print(f"\n[Shadow-Leader Phase 2] Solving {len(self.follower_indices)} Follower population(s)...")
        space_time_factor = np.sqrt(self.internal_solver.Dx * self.internal_solver.Dy * self.internal_solver.Dt)

        for iiter in range(1, max_iters + 1):
            start_time = time.time()

            for f_idx in self.follower_indices:
                pop = self.populations[f_idx]
                if pop.goal is not None:
                    mass_field = pop.M
                    if self.shared_capacity_goals:
                        mass_field = sum(p.M for p in self.populations)
                    pop.goal.update_positions(
                        mass_field, self.internal_solver.omask, self.internal_solver.Dx,
                        self.internal_solver.Dy, self.internal_solver.Lx, self.internal_solver.Ly,
                        pop.goals_are_exits
                    )
                    if pop.goal.has_capacity_limits:
                        pop.door_mask = self.internal_solver.compute_saturated_door_mask(pop, pop.goal.Y_trajectories)

            U_news = []
            for f_idx in self.follower_indices:
                pop = self.populations[f_idx]
                goal_traj = pop.goal.Y_trajectories if pop.goal is not None else None
                U_temp = self.internal_solver.solve_backward_HJB_step(f_idx, goal_traj)
                U_new = self.thetaUM * U_temp + (1.0 - self.thetaUM) * pop.U
                U_news.append((f_idx, U_new))

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
        """
        Execute the complete two-phase Stackelberg best-response workflow.

        Parameters
        ----------
        max_iters : int, default=15
            Maximum iterations for each phase.
        tolerance : float, default=1e-5
            Convergence threshold for space-time L2 residuals.

        Returns
        -------
        If num_pops == 2:
            tuple of (U1, M1, U2, M2).
        If num_pops != 2:
            tuple of (U_1, ..., U_N, M_1, ..., M_N).
        """
        self.solve_shadow_leaders(max_iters=max_iters, tolerance=tolerance)
        self.solve_follower_responses(max_iters=max_iters, tolerance=tolerance)
        if self.num_pops == 2:
            return self.populations[0].U, self.populations[0].M, self.populations[1].U, self.populations[1].M
        return tuple(p.U for p in self.populations) + tuple(p.M for p in self.populations)
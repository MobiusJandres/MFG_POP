"""
Goal management module supporting stationary goals, prescribed paths, evasive goals,
and capacity-constrained landing platforms / exit goals.

This module implements target/goal dynamics for Mean Field Games simulations,
supporting four primary operational behaviors:
1. Stationary: Fixed position targets (e.g., exit doors, static landing pads)
2. Prescribed: Time-dependent parametric trajectories (e.g., moving obstacles/platforms)
3. Evader: Dynamically reactive goals that move away from swarm density using
   repulsive force fields (implements evasion game dynamics)
4. Capacity-Constrained: Exit goals or landing platforms with maximum mass ceilings
   that saturate and deactivate after absorbing a specified cumulative density threshold.
   Once saturated, moving targets freeze at their saturation location.

The evasive goal dynamics use a continuous repulsive potential field computed from
the swarm density distribution M(x,y,t), enabling adversarial game scenarios where
goals actively avoid the pursuing swarm.
"""
import numpy as np


def eval_motion_expr(expr_str, t, T_final, room_size, default_val):
    """
    Evaluate time-dependent parametric motion expressions for prescribed trajectories.

    Supports mathematical expressions involving trigonometric functions and time variables.
    Used for defining analytical goal trajectories (e.g., circular paths, oscillations).

    Args:
        expr_str: String expression to evaluate (e.g., "L/2 + 100*sin(2*pi*t/T)"),
                 numeric value, or None for default
        t: Current time value (seconds)
        T_final: Final simulation time T (seconds), available as 'T' in expressions
        room_size: Domain size L (meters), available as 'L' in expressions
        default_val: Fallback value if expr_str is None

    Returns:
        float: Evaluated expression result

    Examples:
        >>> eval_motion_expr("L/2 + 50*cos(2*pi*t/T)", 1.5, 3.0, 768.0, 384.0)
        384.0  # Center + oscillation term at t=T/2
    """
    if expr_str is None:
        return float(default_val)
    if isinstance(expr_str, (int, float)):
        return float(expr_str)

    # Create restricted evaluation environment with only safe math functions
    # Disable __builtins__ to prevent code injection attacks
    eval_env = {
        'sin': np.sin,
        'cos': np.cos,
        'pi': np.pi,
        't': float(t),        # Current time
        'T': float(T_final),  # Total simulation time
        'L': float(room_size), # Domain dimension for normalization
    }
    return float(eval(str(expr_str), {"__builtins__": None}, eval_env))


class Goal:
    """
    Manages system goals (stationary positions, prescribed trajectories, evasive goals,
    and capacity-constrained exit goals / mobile landing platforms).

    This class handles all goal dynamics in the Mean Field Games framework, supporting
    four operational modes:

    1. **Stationary goals**: Fixed spatial targets (e.g., exit doors, static pads)
       - Position remains constant: Y(t) = Y₀ for all t
       - Defines Dirichlet boundary conditions for value function u

    2. **Prescribed goals**: Deterministic time-dependent trajectories
       - Position follows analytical expressions: Y(t) = [x(t), y(t)]
       - Useful for modeling moving obstacles or predictable targets

    3. **Evader goals**: Dynamically reactive targets using repulsive force fields
       - Velocity computed from swarm density M(x,y,t) via inverse-square repulsion
       - Implements evasion game: dY/dt = v_max * ∇Φ(Y, M) where Φ is repulsive potential
       - Maximum speed constraint v_max enforces bounded evasion capability

    4. **Capacity-Constrained goals**: Mobile or static landing pads with maximum mass ceilings
       - Stores finite capacity limit C_max (cumulative density units allowed)
       - Automatically saturates once total absorbed mass reaches C_max
       - Deactivates exit Dirichlet conditions or attenuates HJB attraction cost
       - Freezes target position at the saturation location upon reaching C_max

    Attributes:
        Nt (int): Number of time steps in simulation
        Dt (float): Time step size (seconds)
        T (float): Total simulation time (seconds)
        Lx (float): Domain width (meters)
        Ly (float): Domain height (meters)
        goals (list[dict]): Goal configuration dictionaries with keys:
            - 'type': 'stationary', 'prescribed', or 'evader'
            - 'position': [x, y] initial position
            - 'v_max': Maximum speed for evaders (m/s)
            - 'path_x', 'path_y': Expression strings for prescribed paths
            - 'capacity': Maximum cumulative mass ceiling (float, default inf)
            - 'is_exit': Whether goal acts as mass-absorbing exit (bool)
        num_goals (int): Total number of goals
        capacities (list[float]): Array of capacity limits per goal
        Y_trajectories (ndarray): Shape (Nt+1, num_goals, 2) position history [x, y]
    """

    def __init__(self, goal_configs: list, Nt: int, Dt: float, T: float = 3.0, Lx: float = 768.0, Ly: float = 768.0):
        """
        Initialize goal manager with configuration and temporal discretization.

        Args:
            goal_configs (list): List of goal configurations, each element either:
                - [x, y] tuple/list: Creates evader at position with default v_max=15.0
                - dict with keys:
                    - 'type': 'stationary', 'prescribed', or 'evader' (default 'evader')
                    - 'position': [x, y] initial position (required)
                    - 'v_max': Maximum evasion speed in m/s (default 15.0, evaders only)
                    - 'path_x': Expression for prescribed x(t) (prescribed goals only)
                    - 'path_y': Expression for prescribed y(t) (prescribed goals only)
                    - 'capacity': Maximum cumulative density threshold (default inf)
                    - 'is_exit': Whether goal absorbs mass via Dirichlet BC (default False)
            Nt (int): Number of temporal discretization steps
            Dt (float): Time step size in seconds
            T (float): Total simulation time in seconds (default 3.0)
            Lx (float): Domain width in meters (default 768.0)
            Ly (float): Domain height in meters (default 768.0)

        Raises:
            ValueError: If goal configuration format is invalid
        """
        self.Nt = Nt
        self.Dt = Dt
        self.T = T
        self.Lx, self.Ly = Lx, Ly
        self.goals = []

        for cfg in goal_configs:
            if isinstance(cfg, (list, tuple)):
                cfg_dict = {
                    'type': 'evader',
                    'position': [float(cfg[0]), float(cfg[1])],
                    'v_max': 15.0,
                    'capacity': float('inf'),
                    'is_exit': False
                }
            elif isinstance(cfg, dict):
                pos = cfg.get('position', [0.0, 0.0])
                cfg_dict = {
                    'type': str(cfg.get('type', 'evader')).lower(),
                    'position': [float(pos[0]), float(pos[1])],
                    'v_max': float(cfg.get('v_max', 15.0)),
                    'path_x': cfg.get('path_x', None),
                    'path_y': cfg.get('path_y', None),
                    'capacity': float(cfg.get('capacity', float('inf'))),
                    'is_exit': bool(cfg.get('is_exit', False))
                }
            else:
                raise ValueError(f"Invalid goal configuration format: {cfg}")

            self.goals.append(cfg_dict)

        self.num_goals = len(self.goals)
        self.capacities = [g['capacity'] for g in self.goals]
        self.Y_trajectories = np.zeros((Nt + 1, self.num_goals, 2))

        # Precompute trajectories for deterministic goal types (stationary and prescribed)
        # Evader trajectories are initialized but will be updated dynamically via update_positions()
        tSpace = np.linspace(0, T, Nt + 1)
        for g_idx, g_info in enumerate(self.goals):
            init_pos = g_info['position']
            self.Y_trajectories[0, g_idx, :] = init_pos

            if g_info['type'] == 'stationary':
                # Replicate initial position for all time steps
                for k in range(1, Nt + 1):
                    self.Y_trajectories[k, g_idx, :] = init_pos

            elif g_info['type'] == 'prescribed':
                # Evaluate parametric path expressions at each time point
                x_expr, y_expr = g_info['path_x'], g_info['path_y']
                for k, t_val in enumerate(tSpace):
                    px = eval_motion_expr(x_expr, t_val, T, Lx, init_pos[0])
                    py = eval_motion_expr(y_expr, t_val, T, Ly, init_pos[1])
                    self.Y_trajectories[k, g_idx, :] = [px, py]

            elif g_info['type'] == 'evader':
                # Initialize with stationary position; actual trajectory computed in update_positions()
                for k in range(1, Nt + 1):
                    self.Y_trajectories[k, g_idx, :] = init_pos

    @property
    def has_dynamic_goals(self) -> bool:
        """
        Check if any goals have time-dependent positions requiring iterative updates.

        This property determines whether the MFG solver needs to use the full iterative
        update scheme (Picard iteration with goal position updates) or can use a simpler
        fixed-target solver.

        Returns:
            bool: True if any goal is type 'evader' or 'prescribed', False if all stationary
        
        Notes:
            - Stationary goals → can solve HJB-KFP system once with fixed boundary conditions
            - Dynamic goals → require outer loop updating goal positions Y(t) and re-solving
        """
        return any(g['type'] in ('evader', 'prescribed') for g in self.goals)

    @property
    def has_capacity_limits(self) -> bool:
        """
        Check if any goals have finite capacity limits requiring mass accumulation tracking.

        Returns:
            bool: True if any goal has a finite capacity threshold, False if all infinite
        """
        return any(np.isfinite(c) for c in self.capacities)

    def update_positions(self, M_field, omask, Dx, Dy, Lx, Ly):
        """
        Update goal trajectories based on swarm density field for evader-type goals.

        Implements the evasion dynamics for reactive goals using repulsive force fields
        computed from the swarm density distribution M(x,y,t). The evader velocity at
        position Y is computed as:

            F(Y) = ∫∫ (Y - x) / |Y - x|² M(x,y) dx dy   (repulsive force field)
            v(Y) = v_max * F(Y) / |F(Y)|                (normalized to max speed)
            Y(t+Δt) = Y(t) + v(Y) Δt                    (forward Euler integration)
        
        This implements a greedy evasion strategy where goals move directly away from
        the center of mass of nearby swarm density with bounded maximum speed.

        Args:
            M_field (ndarray): Swarm density field, shape (Nt+1, Nx, Ny)
                               Mass distribution M(x,y,t) from KFP solution
            omask (ndarray): Obstacle mask, shape (Nx, Ny)
                             1 = walkable, 0 = obstacle (blocks evader motion)
            Dx (float): Spatial step size in x direction (meters)
            Dy (float): Spatial step size in y direction (meters)
            Lx (float): Domain width (meters)
            Ly (float): Domain height (meters)

        Returns:
            ndarray: Updated trajectory array, shape (Nt+1, num_goals, 2)
        
        Notes:
            - Stationary goals: Position copied from initial state
            - Prescribed goals: Position copied from precomputed trajectory
            - Evader goals: Position updated via repulsive force field integration
            - Collision detection: Goals cannot move into obstacle cells (omask == 0)
            - Regularization: Distance denominator includes +1e-3 to prevent singularities        
        """
        Nx, Ny = omask.shape
        new_trajectories = np.copy(self.Y_trajectories)

        # Create spatial coordinate grids for force field computation
        # Cell-centered grid: coordinates at bin centers, not edges
        x_coords = np.linspace(Dx / 2, Lx - Dx / 2, Nx)
        y_coords = np.linspace(Dy / 2, Ly - Dy / 2, Ny)
        X_grid, Y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')

        cumulative_mass = np.zeros(self.num_goals)

        # Time-step loop: update all goals from time k to k+1
        for k in range(self.Nt):
            M_k = M_field[k]  # Swarm density at current time step
            total_mass = np.sum(M_k) # Total mass for zero-density check

            for g, g_info in enumerate(self.goals):
                cap = g_info.get('capacity', float('inf'))
                curr_x, curr_y = new_trajectories[k, g]

                # Track cumulative mass entering goal region up to step k
                if np.isfinite(cap):
                    region = (np.abs(X_grid - curr_x) <= Dx) & (np.abs(Y_grid - curr_y) <= Dy)
                    mass_in_region = np.sum(M_k[region]) * Dx * Dy
                    cumulative_mass[g] += mass_in_region * self.Dt

                # If capacity ceiling is reached, freeze goal at current location
                if cumulative_mass[g] >= cap:
                    new_trajectories[k + 1, g] = [curr_x, curr_y]
                    continue

                if g_info['type'] == 'stationary':
                    new_trajectories[k + 1, g] = self.Y_trajectories[0, g]

                elif g_info['type'] == 'prescribed':
                    # Unsaturated prescribed goal follows analytical parametric path
                    x_expr, y_expr = g_info['path_x'], g_info['path_y']
                    t_next = (k + 1) * self.Dt
                    px = eval_motion_expr(x_expr, t_next, self.T, Lx, self.Y_trajectories[0, g, 0])
                    py = eval_motion_expr(y_expr, t_next, self.T, Ly, self.Y_trajectories[0, g, 1])
                    new_trajectories[k + 1, g] = [px, py]

                elif g_info['type'] == 'evader':
                    v_max = g_info['v_max']

                    if total_mass > 1e-6:
                        rx = curr_x - X_grid
                        ry = curr_y - Y_grid
                        dist_sq = rx**2 + ry**2 + 1e-3

                        force_x = np.sum((rx / dist_sq) * M_k)
                        force_y = np.sum((ry / dist_sq) * M_k)
                        norm_force = np.sqrt(force_x**2 + force_y**2)

                        if norm_force > 1e-8:
                            vx = (force_x / norm_force) * v_max
                            vy = (force_y / norm_force) * v_max
                        else:
                            vx, vy = 0.0, 0.0
                    else:
                        vx, vy = 0.0, 0.0

                    # Forward Euler integration: Y(t+Δt) = Y(t) + v·Δt
                    # Clip to domain boundaries [Dx, L-Dx] to stay within valid region
                    next_x = np.clip(curr_x + vx * self.Dt, Dx, Lx - Dx)
                    next_y = np.clip(curr_y + vy * self.Dt, Dy, Ly - Dy)

                    # Collision detection: check if proposed position is walkable
                    next_i = int(np.clip(next_x / Dx, 0, Nx - 1))
                    next_j = int(np.clip(next_y / Dy, 0, Ny - 1))

                    if omask[next_i, next_j] == 1:
                        # Cell is walkable → accept new position
                        new_trajectories[k + 1, g] = [next_x, next_y]
                    else:
                        # Cell is obstacle → reject motion, stay at current position
                        new_trajectories[k + 1, g] = [curr_x, curr_y]

        return new_trajectories


TargetSwarm = Goal
EvaderSwarm = Goal
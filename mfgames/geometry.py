"""
Geometry layout routines for Mean Field Games spatial domain configuration.

This module provides two primary geometry systems:
1. **MAP2PDE**: Translates MovingAI benchmark files (.map and .scen) into spatial PDE grids
   for MFG simulations with realistic pathfinding scenarios.
2. **MFGTrafficGeometry**: Synthetic corridor geometry with parallel channels and obstacles,
   replicating the original MFG_Traffic_numba.py layout.

Additionally provides utilities for:
- Dynamic exit door mask generation (time-dependent Dirichlet boundary conditions)
- Domain validation (door continuity, obstacle overlap detection)
- Motion expression parsing for time-varying boundaries

Classes:
    MAP2PDE: Parser for MovingAI benchmark files with PDE grid generation
    MFGTrafficGeometry: Synthetic 5-corridor geometry layout

Functions:
    create_moving_door_mask: Generate time-dependent exit door masks
    parse_motion_expr: Convert string formulas to callable time functions
    build_door_trajectories_from_config: Parse YAML door configurations
    validate_door_mask_exists: Check door presence at all timesteps
    validate_door_mask_no_obstacle_overlap: Check door-obstacle separation
    validate_door_mask_continuous: Check door velocity constraints
"""
import os
import math
import numpy as np


# =============================================================================
# 1. MOVINGAI BENCHMARK PARSER & MAP MESH GENERATOR
# =============================================================================

class MAP2PDE:
    """
    Translates MovingAI benchmark files (.map and .scen) into spatial PDE grids.

    This class bridges discrete grid-based pathfinding benchmarks (MovingAI format) to
    continuous PDE-based Mean Field Games simulations. It maps obstacle grids to spatial
    masks and agent start/goal positions to initial density distributions and terminal costs.

    Attributes:
        map_filepath (str|None): Path to .map file (MovingAI benchmark format)
        scen_filepath (str|None): Path to .scen file (agent start/goal positions)
        Lx (float): Physical domain width in metres (default: 768.0)
        Ly (float): Physical domain height in metres (default: 768.0)
        Nx (int): Number of PDE grid points in x-direction (default: 100)
        Ny (int): Number of PDE grid points in y-direction (default: 100)
        dx (float): Grid spacing in x-direction (Lx / Nx)
        dy (float): Grid spacing in y-direction (Ly / Ny)
        grid_shape (tuple): Shape of raw MovingAI grid (height, width)
        raw_boolean_grid (np.ndarray|None): Boolean obstacle grid from .map file
        raw_agents (list[dict]): List of agent start/goal positions from .scen file
        X (np.ndarray|None): Meshgrid x-coordinates (Nx × Ny)
        Y (np.ndarray|None): Meshgrid y-coordinates (Nx × Ny)
        custom_goals (list|None): Override goals for terminal cost calculation
        custom_m0 (np.ndarray|None): Override initial density distribution
    """

    def __init__(
        self,
        map_filepath: str|None = None,
        scen_filepath: str|None = None,
        Lx: float = 768.0,
        Ly: float = 768.0,
        Nx: int = 100,
        Ny: int = 100,
        custom_goals: list|None = None,
    ):
        self.map_filepath = map_filepath
        self.scen_filepath = scen_filepath
        self.Lx, self.Ly = Lx, Ly
        self.Nx, self.Ny = Nx, Ny
        self.grid_shape = (0, 0)
        self.raw_boolean_grid = None
        self.raw_agents = []
        self.dx, self.dy = Lx / Nx, Ly / Ny
        self.X, self.Y = None, None
        self._m0 = None
        self.custom_m0 = None
        self.custom_goals = custom_goals

    def parse_files(self, num_agents: int|None = None, start_idx: int = 0):
        """
        Parse MovingAI benchmark files to extract obstacle grid and agent scenarios.

        Reads .map file to build boolean obstacle grid (passable vs. blocked cells) and
        .scen file to extract agent start/goal positions. If no files provided, generates
        a synthetic default layout with central vertical wall obstacle.

        Args:
            num_agents (int|None): Maximum number of agents to load from .scen file.
                If None, loads all agents. Default: None.
            start_idx (int): Starting line index in .scen file (for skipping header).
                Default: 0.

        Raises:
            FileNotFoundError: If map_filepath or scen_filepath does not exist.

        Returns:
            None: Updates self.raw_boolean_grid and self.raw_agents in-place.
        """
        if self.map_filepath or self.scen_filepath:
            if not self.map_filepath or not os.path.exists(self.map_filepath):
                raise FileNotFoundError(f"[MAP2PDE Error] Map file not found at: '{self.map_filepath}'")
            if not self.scen_filepath or not os.path.exists(self.scen_filepath):
                raise FileNotFoundError(f"[MAP2PDE Error] Scenario file not found at: '{self.scen_filepath}'")

            print(f"Loading MovingAI map '{self.map_filepath}'...", flush=True)
            with open(self.map_filepath, 'r') as f:
                lines = f.readlines()
            height, width = int(lines[1].split()[1]), int(lines[2].split()[1])
            self.grid_shape = (height, width)
            self.raw_boolean_grid = np.zeros((height, width), dtype=bool)

            for r, line in enumerate(lines[4:]):
                for c, char in enumerate(line.strip()):
                    if char in ['.', 'G']:
                        self.raw_boolean_grid[r, c] = True

            with open(self.scen_filepath, 'r') as f:
                scen_lines = f.readlines()[1:]

            for i, line in enumerate(scen_lines):
                if i < start_idx:
                    continue

                parts = line.strip().split()
                if len(parts) >= 9:
                    self.raw_agents.append({
                        'start_x': int(parts[4]), 'start_y': int(parts[5]),
                        'goal_x': int(parts[6]), 'goal_y': int(parts[7])
                    })
                if num_agents and len(self.raw_agents) >= num_agents:
                    break
        else:
            self.grid_shape = (self.Ny, self.Nx)
            self.raw_boolean_grid = np.ones(self.grid_shape, dtype=bool)
            h, w = self.grid_shape
            self.raw_boolean_grid[h // 4:3 * h // 4, w // 2 - 2:w // 2 + 2] = False
            for a in range(num_agents or 10):
                self.raw_agents.append({
                    'start_x': 5 + (a % 10) * 3, 'start_y': 5 + (a // 10) * 3,
                    'goal_x': w - 10, 'goal_y': h - 10 - (a % 5) * 4
                })

    def build_spatial_mesh(self):
        """
        Construct cell-centred spatial meshgrid for PDE discretization.

        Creates meshgrid with coordinates at cell centres (not corners), which is
        standard for finite-volume and finite-difference PDE discretizations. The
        grid spans [dx/2, Lx - dx/2] × [dy/2, Ly - dy/2] to avoid boundary cells.

        Args:
            None

        Returns:
            tuple[np.ndarray, np.ndarray]: (X, Y) meshgrids of shape (Nx, Ny) containing
                x and y coordinates at each grid cell centre.
        """
        xSpace = np.linspace(self.dx / 2, self.Lx - self.dx / 2, self.Nx, endpoint=True)
        ySpace = np.linspace(self.dy / 2, self.Ly - self.dy / 2, self.Ny, endpoint=True)
        self.X, self.Y = np.meshgrid(xSpace, ySpace, indexing='ij')
        return self.X, self.Y

    def get_pde_obstacle_mask(self):
        """
        Map MovingAI boolean obstacle grid to PDE spatial discretization.

        Performs nearest-neighbour interpolation from coarse MovingAI grid to fine
        PDE grid. Each PDE cell inherits the passability status of the closest
        MovingAI cell. Coordinate transformation accounts for y-axis inversion
        (MovingAI uses top-left origin, PDE uses bottom-left origin).

        Args:
            None

        Returns:
            np.ndarray: Integer mask of shape (Nx, Ny) where:
                - 1 = passable (no obstacle)
                - 0 = blocked (obstacle present)
        """
        pde_mask = np.ones((self.Nx, self.Ny), dtype=np.int64)
        map_H, map_W = self.grid_shape
        map_dx, map_dy = self.Lx / map_W, self.Ly / map_H

        # Nearest-neighbour interpolation from MovingAI grid to PDE grid
        for i in range(self.Nx):
            for j in range(self.Ny):
                col_idx = min(max(0, int(self.X[i, j] / map_dx)), map_W - 1)
                # Y-axis inversion: MovingAI origin at top-left, PDE origin at bottom-left
                row_idx = min(max(0, int((self.Ly - self.Y[i, j]) / map_dy)), map_H - 1)
                if not self.raw_boolean_grid[row_idx, col_idx]:
                    pde_mask[i, j] = 0
        return pde_mask

    def set_custom_initial_blobs(self, blob_list, normalize_mass: bool = False):
        """
        Override scenario initial density with custom Gaussian blob distributions.

        Replaces agent-based initial density with sum of Gaussian kernels centred at
        specified locations. Useful for studying aggregate crowd behavior without
        individual agent tracking. Automatically masks out obstacle regions.

        Args:
            blob_list (list[tuple]): List of Gaussian blobs, each specified as:
                - (x, y, sigma): Centre (x, y), spread sigma, amplitude = 1.0
                - (x, y, sigma, amp): Centre, spread, and explicit amplitude
            normalize_mass (bool): If True, normalize to unit integral over domain
                (accounting for cell areas dx * dy). Default: False.

        Returns:
            np.ndarray: Initial density field m0 of shape (Nx, Ny).

        Raises:
            ValueError: If build_spatial_mesh() has not been called yet.
        """
        if self.X is None or self.Y is None:
            raise ValueError("Must call build_spatial_mesh() before setting custom density.")

        m_0 = np.zeros_like(self.X, dtype=np.float64)
        for blob in blob_list:
            if len(blob) == 3:
                cx, cy, sigma = blob
                amp = 1.0
            else:
                cx, cy, sigma, amp = blob
            m_0 += amp * np.exp(-((self.X - cx)**2 + (self.Y - cy)**2) / (2.0 * sigma**2))

        m_0 *= self.get_pde_obstacle_mask()
        if normalize_mass and np.sum(m_0) > 0:
            m_0 /= (np.sum(m_0) * self.dx * self.dy)

        self.custom_m0 = m_0
        return m_0

    def build_initial_density(self, sigma_multiplier: float = 1.5):
        """
        Generate initial density field from agent starting positions.

        Each agent from .scen file is represented as a Gaussian kernel with standard
        deviation proportional to grid spacing. This smooths discrete agent positions
        into a continuous density field suitable for PDE evolution.

        Args:
            sigma_multiplier (float): Scaling factor for Gaussian kernel spread.
                sigma = sigma_multiplier * max(dx, dy). Larger values create more
                diffuse initial distributions. Default: 1.5.

        Returns:
            np.ndarray: Initial density field m0 of shape (Nx, Ny).
        """
        if self.custom_m0 is not None:
            return self.custom_m0
        if self._m0 is not None:
            return self._m0

        m_0 = np.zeros_like(self.X)
        sigma = sigma_multiplier * max(self.dx, self.dy)
        map_H, map_W = self.grid_shape
        map_dx, map_dy = self.Lx / map_W, self.Ly / map_H

        # Convert agent start positions from MovingAI grid to PDE coordinates
        for agent in self.raw_agents:
            sx = (agent['start_x'] + 0.5) * map_dx
            sy = self.Ly - (agent['start_y'] + 0.5) * map_dy  # Y-axis inversion
            m_0 += np.exp(-((self.X - sx)**2 + (self.Y - sy)**2) / (2 * sigma**2))

        self._m0 = m_0
        return self._m0


    @property
    def m0(self):
        """
        Lazy-evaluated initial density field.

        Returns:
            np.ndarray: Initial density m0 of shape (Nx, Ny), building it if needed.
        """
        if self._m0 is None:
            self._m0 = self.build_initial_density()
        return self._m0

    @m0.setter
    def m0(self, value):
        """
        Set custom initial density field.

        Args:
            value (np.ndarray): New initial density array of shape (Nx, Ny).
        """
        self._m0 = value

    def set_custom_goals(self, goal_list):
        """
        Override scenario goal positions with custom (x, y) coordinates.

        Args:
            goal_list (list[tuple | dict]): List of goal positions as (x, y) tuples
                or dicts with 'position' key containing [x, y].

        Returns:
            None: Updates self.custom_goals in-place.
        """
        self.custom_goals = goal_list

    def get_goals(self):
        """
        Retrieve active goal positions (custom or from scenario file).

        Returns list of unique goal locations, preferring custom_goals if set,
        otherwise extracting from raw_agents parsed from .scen file.

        Returns:
            list[tuple]: List of (x, y) goal coordinates in PDE domain space.
        """
        if self.custom_goals is not None:
            goals = []
            for item in self.custom_goals:
                if isinstance(item, dict):
                    pos = item['position']
                    goals.append((float(pos[0]), float(pos[1])))
                else:
                    goals.append((float(item[0]), float(item[1])))
            return goals

        if not hasattr(self, 'grid_shape') or self.grid_shape == (0, 0):
            return []

        map_H, map_W = self.grid_shape
        map_dx, map_dy = self.Lx / map_W, self.Ly / map_H
        goals = [((a['goal_x'] + 0.5) * map_dx, self.Ly - (a['goal_y'] + 0.5) * map_dy) for a in self.raw_agents]
        return list(dict.fromkeys(goals))

    def get_goal_positions(self):
        """
        Alias for get_goals() for API compatibility.

        Returns:
            list[tuple]: List of (x, y) goal coordinates (same as get_goals()).
        """
        return self.get_goals()

    def build_terminal_cost(self, penalty_scale: float = 15.0):
        """
        Compute distance-based terminal cost function g(x, y).

        Terminal cost penalizes agents for ending far from goal locations. At each
        grid cell, computes Euclidean distance to nearest goal and scales by penalty
        factor. Used as boundary condition for HJB backward solve: g(x, T) = terminal cost.

        Args:
            penalty_scale (float): Multiplicative penalty per unit distance from goal.
                Higher values enforce stronger terminal matching. Default: 15.0.

        Returns:
            np.ndarray: Terminal cost grid g of shape (Nx, Ny), with g[i, j] =
                penalty_scale * min_goal(distance from (x[i,j], y[i,j]) to goal).
        """
        Nx, Ny = self.X.shape
        g = np.zeros((Nx, Ny), dtype=np.float64)
        goal_positions = self.get_goals()

        if not goal_positions:
            return g

        for i in range(Nx):
            for j in range(Ny):
                x_val, y_val = self.X[i, j], self.Y[i, j]
                min_dist = min(np.sqrt((x_val - gx)**2 + (y_val - gy)**2) for gx, gy in goal_positions)
                g[i, j] = penalty_scale * min_dist
        return g


# =============================================================================
# 2. PRIMITIVE TRAFFIC GEOMETRY (5 CORRIDORS & OBSTACLES)
# =============================================================================

class MFGTrafficGeometry:
    """
    Synthetic corridor geometry with parallel channels and obstacles.

    Recreates the 5-corridor layout from the original MFG_Traffic_numba.py script:
    five horizontal passageways separated by solid walls, modeling pedestrian flow
    through narrow channels with congestion effects. Used for testing MFG solvers
    without external benchmark dependencies.

    Attributes:
        Lx (float): Physical domain width in metres (default: 50.0)
        Ly (float): Physical domain height in metres (default: 50.0)
        Nx (int): Number of PDE grid points in x-direction (default: 75)
        Ny (int): Number of PDE grid points in y-direction (default: 75)
        dx (float): Grid spacing in x-direction (Lx / (Nx - 1))
        dy (float): Grid spacing in y-direction (Ly / (Ny - 1))
        X (np.ndarray|None): Meshgrid x-coordinates (Nx × Ny)
        Y (np.ndarray|None): Meshgrid y-coordinates (Nx × Ny)
        orectangles (list[tuple]): List of 5 obstacle rectangles (x1, x2, y1, y2)
        rectangles (list[tuple]): List of 5 initial density corridors (x1, x2, y1, y2)
    """

    def __init__(self, Lx: float = 50.0, Ly: float = 50.0, Nx: int = 75, Ny: int = 75):
        self.Lx, self.Ly = Lx, Ly
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = Lx / (Nx - 1), Ly / (Ny - 1)
        self.X, self.Y = None, None
        self._m0 = None

        # 5 Obstacle rectangles: horizontal walls separating corridors
        # Format: (x_min, x_max, y_min, y_max) in metres
        self.orectangles = [
            (10.0, 40.0, 10.0, 14.0),
            (10.0, 40.0, 18.0, 22.0),
            (10.0, 40.0, 26.0, 30.0),
            (10.0, 40.0, 34.0, 38.0),
            (10.0, 40.0, 42.0, 46.0)
        ]

        # 5 Initial density corridors: passable channels between walls
        # Format: (x_min, x_max, y_min, y_max) in metres
        self.rectangles = [
            (10.0, 40.0, 14.0, 18.0),
            (10.0, 40.0, 22.0, 26.0),
            (10.0, 40.0, 30.0, 34.0),
            (10.0, 40.0, 38.0, 42.0),
            (10.0, 40.0, 46.0, 50.0)
        ]

    def build_spatial_mesh(self):
        """
        Construct uniform spatial meshgrid spanning [0, Lx] × [0, Ly].

        Args:
            None

        Returns:
            tuple[np.ndarray, np.ndarray]: (X, Y) meshgrids of shape (Nx, Ny).
        """
        xSpace = np.linspace(0.0, self.Lx, self.Nx, endpoint=True)
        ySpace = np.linspace(0.0, self.Ly, self.Ny, endpoint=True)
        self.X, self.Y = np.meshgrid(xSpace, ySpace, indexing='ij')
        return self.X, self.Y

    def get_pde_obstacle_mask(self):
        """
        Generate obstacle mask from hardcoded wall rectangles.

        Returns:
            np.ndarray: Integer mask of shape (Nx, Ny) where:
                - 1 = passable corridor
                - 0 = solid wall (obstacle)
        """
        omask = np.ones((self.Nx, self.Ny), dtype=np.int64)
        for (x1, x2, y1, y2) in self.orectangles:
            region = (self.X >= x1) & (self.X <= x2) & (self.Y >= y1) & (self.Y <= y2)
            omask[region] = 0
        return omask

    def build_initial_density(self):
        """
        Generate initial density with uniform distribution in each corridor.

        Assigns constant density value (4.0) within each passable corridor region
        and zero density in obstacle regions.

        Returns:
            np.ndarray: Initial density m0 of shape (Nx, Ny).
        """
        if self._m0 is not None:
            return self._m0

        m0 = np.zeros((self.Nx, self.Ny))
        # Fill each corridor with constant density
        for (x1, x2, y1, y2) in self.rectangles:
            region = (self.X > x1) & (self.X < x2) & (self.Y > y1) & (self.Y < y2)
            m0[region] = 4.0

        # Zero out obstacle cells
        omask = self.get_pde_obstacle_mask()
        m0[omask == 0] = 0.0
        self._m0 = m0
        return self._m0

    @property
    def m0(self):
        """
        Lazy-evaluated initial density field.

        Returns:
            np.ndarray: Initial density m0 of shape (Nx, Ny).
        """
        if self._m0 is None:
            self._m0 = self.build_initial_density()
        return self._m0

    @m0.setter
    def m0(self, value):
        """
        Set custom initial density field.

        Args:
            value (np.ndarray): New initial density array of shape (Nx, Ny).
        """
        self._m0 = value

    def get_goals(self):
        """
        Return goal positions (empty for synthetic traffic geometry).

        The original MFG_Traffic_numba.py geometry has no explicit goal locations,
        using exit doors instead for Dirichlet boundary conditions.

        Returns:
            list: Empty list (no goals defined).
        """
        return []

    def get_goal_positions(self):
        """
        Alias for get_goals() for API compatibility.

        Returns:
            list: Empty list (same as get_goals()).
        """
        return self.get_goals()


# =============================================================================
# 3. DYNAMIC & STATIC EXIT DOOR MASK GENERATORS
# =============================================================================

def create_moving_door_mask(door_trajectories, Nt, Nx, Ny, X, Y, T):
    """
    Generate time-dependent 3D exit door mask for moving Dirichlet boundaries.

    Evaluates door trajectory functions at each timestep to produce a boolean indicator
    grid marking exit regions. Used to impose time-varying Dirichlet boundary conditions
    (zero value function at exit doors) in the HJB equation.

    Args:
        door_trajectories (list[dict]): List of door trajectory dictionaries, each with
            callable functions 'x1'(t), 'x2'(t), 'y1'(t), 'y2'(t) defining rectangular
            door boundaries at time t.
        Nt (int): Number of time steps in simulation.
        Nx (int): Number of spatial grid points in x-direction.
        Ny (int): Number of spatial grid points in y-direction.
        X (np.ndarray): Meshgrid of x-coordinates, shape (Nx, Ny).
        Y (np.ndarray): Meshgrid of y-coordinates, shape (Nx, Ny).
        T (float): Final simulation time.

    Returns:
        np.ndarray: Boolean mask of shape (Nt+1, Nx, Ny) where mask[k, i, j] = 1.0
            if cell (i, j) is inside a door region at timestep k, else 0.0.
    """
    door_mask_3d = np.zeros((Nt + 1, Nx, Ny))
    tSpace = np.linspace(0, T, Nt + 1)

    # Evaluate door positions at each timestep
    for k, t in enumerate(tSpace):
        for traj in door_trajectories:
            x1, x2 = traj['x1'](t), traj['x2'](t)
            y1, y2 = traj['y1'](t), traj['y2'](t)
            region = (X >= x1) & (X <= x2) & (Y >= y1) & (Y <= y2)
            door_mask_3d[k][region] = 1.0

    return door_mask_3d


def parse_motion_expr(expr_str, T_final, room_width, room_height):
    """
    Parse string expression or constant into callable time-dependent function.

    Supports mathematical formulas for dynamic door motion. String expressions can
    reference 't' (time), 'T' (final time), 'room_width', 'room_height', and standard
    math functions (sin, cos, pi). Numeric constants are converted to constant functions.

    Args:
        expr_str (str | int | float): Motion expression to parse. Examples:
            - "40.0": constant position at x=40.0
            - "room_width - 5": constant offset from room boundary
            - "20 + 10*sin(2*pi*t/T)": sinusoidal motion with period T
        T_final (float): Final simulation time (available as 'T' in expressions).
        room_width (float): Domain width (available as 'room_width' in expressions).
        room_height (float): Domain height (available as 'room_height' in expressions).

    Returns:
        callable: Function f(t) returning evaluated position at time t.
    """
    eval_env = {
        'sin': np.sin,
        'cos': np.cos,
        'pi': np.pi,
        'T': float(T_final),
        'room_width': float(room_width),
        'room_height': float(room_height),
    }

    if isinstance(expr_str, (int, float)):
        val = float(expr_str)
        return lambda t: val

    # Evaluate string expression with restricted builtins for security
    return lambda t: float(eval(str(expr_str), {"__builtins__": None}, {**eval_env, 't': float(t)}))


def build_door_trajectories_from_config(door_configs, T_final, room_width=50.0, room_height=50.0):
    """
    Convert YAML door configuration list into callable trajectory dictionaries.

    Parses door configuration from YAML/dict format into trajectory functions suitable
    for create_moving_door_mask(). Each door is defined by four boundary expressions
    (x1, x2, y1, y2) that can be constant or time-varying.

    Args:
        door_configs (list[dict]): List of door configuration dictionaries, each with keys:
            - 'x1': Left boundary expression (str or numeric)
            - 'x2': Right boundary expression (str or numeric)
            - 'y1': Bottom boundary expression (str or numeric)
            - 'y2': Top boundary expression (str or numeric)
        T_final (float): Final simulation time for expression evaluation.
        room_width (float): Domain width for expression evaluation. Default: 50.0.
        room_height (float): Domain height for expression evaluation. Default: 50.0.

    Returns:
        list[dict]: List of trajectory dictionaries, each with callable keys:
            {'x1': f(t), 'x2': f(t), 'y1': f(t), 'y2': f(t)}
    """
    if not door_configs:
        return []

    door_trajectories = []
    for cfg in door_configs:
        door_trajectories.append({
            'x1': parse_motion_expr(cfg['x1'], T_final, room_width, room_height),
            'x2': parse_motion_expr(cfg['x2'], T_final, room_width, room_height),
            'y1': parse_motion_expr(cfg['y1'], T_final, room_width, room_height),
            'y2': parse_motion_expr(cfg['y2'], T_final, room_width, room_height),
        })
    return door_trajectories


# =============================================================================
# 4. DOMAIN VALIDATION ROUTINES
# =============================================================================

def validate_door_mask_exists(door_mask_3d):
    """
    Validate that at least one door cell exists at every timestep.

    Ensures domain always has an exit boundary for Dirichlet conditions. Critical
    for HJB solver stability - without exit doors, value function has no finite
    minimum and may diverge.

    Args:
        door_mask_3d (np.ndarray): Door mask array of shape (Nt+1, Nx, Ny).

    Returns:
        bool: True if validation passes.

    Raises:
        ValueError: If any timestep has zero door cells.
    """
    for k in range(door_mask_3d.shape[0]):
        if np.sum(door_mask_3d[k]) == 0:
            raise ValueError(f"No doors exist at timestep k={k}")
    return True


def validate_door_mask_no_obstacle_overlap(door_mask_3d, omask):
    """
    Validate that doors do not overlap with obstacle cells.

    Ensures physical consistency - exit doors must be in passable regions. Overlap
    would create contradictory boundary conditions (Dirichlet at doors vs. Neumann
    at solid walls).

    Args:
        door_mask_3d (np.ndarray): Door mask array of shape (Nt+1, Nx, Ny).
        omask (np.ndarray): Obstacle mask of shape (Nx, Ny) where 0 = obstacle.

    Returns:
        bool: True if validation passes.

    Raises:
        ValueError: If door overlaps with obstacle at any timestep.
    """
    for k in range(door_mask_3d.shape[0]):
        overlap = np.logical_and(door_mask_3d[k] == 1, omask == 0)
        if np.any(overlap):
            raise ValueError(f"Door overlaps with obstacle at timestep k={k}")
    return True


def validate_door_mask_continuous(door_mask_3d, Dt, Dx, Dy, max_velocity=5.0):
    """
    Validate that doors move continuously without teleporting or exceeding max_velocity.

    Checks door centroid displacement between consecutive timesteps. Prevents
    numerical artifacts from excessively fast door motion, which can violate CFL
    condition and cause solver instability. Includes grid diagonal buffer to account
    for discretization effects.

    Args:
        door_mask_3d (np.ndarray): Door mask array of shape (Nt+1, Nx, Ny).
        Dt (float): Timestep size in seconds.
        Dx (float): Grid spacing in x-direction.
        Dy (float): Grid spacing in y-direction.
        max_velocity (float): Maximum allowable door velocity in m/s. Default: 5.0.

    Returns:
        bool: True if validation passes.

    Raises:
        ValueError: If door centroid displacement exceeds max_velocity * Dt + diagonal_buffer
            between any consecutive timesteps.
    """
    Nt = door_mask_3d.shape[0] - 1
    max_displacement = max_velocity * Dt
    grid_buffer = np.sqrt(Dx**2 + Dy**2)  # Allow one diagonal cell tolerance

    for k in range(Nt):
        door_k, door_k1 = door_mask_3d[k].astype(bool), door_mask_3d[k + 1].astype(bool)
        if np.sum(door_k) > 0 and np.sum(door_k1) > 0:
            # Compute door centroids at k and k+1
            ik, jk = np.where(door_k)
            ik1, jk1 = np.where(door_k1)
            cx_k, cy_k = np.mean(ik) * Dx, np.mean(jk) * Dy
            cx_k1, cy_k1 = np.mean(ik1) * Dx, np.mean(jk1) * Dy

            # Check centroid displacement
            displacement = np.sqrt((cx_k1 - cx_k)**2 + (cy_k1 - cy_k)**2)
            if displacement > max_displacement + grid_buffer:
                raise ValueError(f"Door moved too fast between timestep k={k} and k={k+1}.")

    return True
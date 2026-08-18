"""
Geometry layout routines for both MovingAI benchmark maps (MAP2PDE) 
and synthetic primitive layouts (MFGTrafficGeometry).
"""
import os
import numpy as np


class MAP2PDE:
    """Translates MovingAI benchmark files (.map and .scen) into spatial PDE grids."""

    def __init__(self, map_filepath: str = None, scen_filepath: str = None, Lx: float = 768.0, Ly: float = 768.0, Nx: int = 100, Ny: int = 100):
        self.map_filepath = map_filepath
        self.scen_filepath = scen_filepath
        self.Lx, self.Ly = Lx, Ly
        self.Nx, self.Ny = Nx, Ny
        self.grid_shape = (Ny, Nx)
        self.raw_boolean_grid = None
        self.raw_agents = []
        self.dx, self.dy = Lx / Nx, Ly / Ny
        self.X, self.Y = None, None
        self._m0 = None

    def parse_files(self, num_agents: int = 50):
        if self.map_filepath or self.scen_filepath:
            if not self.map_filepath or not os.path.exists(self.map_filepath):
                raise FileNotFoundError(f"[MAP2PDE Error] Map file not found at: '{self.map_filepath}'")
            if not self.scen_filepath or not os.path.exists(self.scen_filepath):
                raise FileNotFoundError(f"[MAP2PDE Error] Scenario file not found at: '{self.scen_filepath}'")

            print(f"Loading MovingAI map '{self.map_filepath}' and scenario '{self.scen_filepath}'...", flush=True)
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

            for line in scen_lines:
                parts = line.strip().split()
                if len(parts) >= 9:
                    self.raw_agents.append({
                        'start_x': int(parts[4]), 'start_y': int(parts[5]),
                        'goal_x': int(parts[6]), 'goal_y': int(parts[7])
                    })
                if num_agents and len(self.raw_agents) >= num_agents:
                    break
            print(f"Loaded grid ({height}x{width}) with {len(self.raw_agents)} agents.", flush=True)
        else:
            print("No map files specified. Building default synthetic grid...", flush=True)
            self.grid_shape = (self.Ny, self.Nx)
            self.raw_boolean_grid = np.ones(self.grid_shape, dtype=bool)
            h, w = self.grid_shape
            self.raw_boolean_grid[h // 4:3 * h // 4, w // 2 - 2:w // 2 + 2] = False
            for a in range(num_agents):
                self.raw_agents.append({
                    'start_x': 5 + (a % 10) * 3, 'start_y': 5 + (a // 10) * 3,
                    'goal_x': w - 10, 'goal_y': h - 10 - (a % 5) * 4
                })

    def build_spatial_mesh(self):
        xSpace = np.linspace(self.dx / 2, self.Lx - self.dx / 2, self.Nx, endpoint=True)
        ySpace = np.linspace(self.dy / 2, self.Ly - self.dy / 2, self.Ny, endpoint=True)
        self.X, self.Y = np.meshgrid(xSpace, ySpace, indexing='ij')
        return self.X, self.Y

    def get_pde_obstacle_mask(self):
        pde_mask = np.ones((self.Nx, self.Ny), dtype=np.int64)
        map_H, map_W = self.grid_shape
        map_dx, map_dy = self.Lx / map_W, self.Ly / map_H
        for i in range(self.Nx):
            for j in range(self.Ny):
                col_idx = min(max(0, int(self.X[i, j] / map_dx)), map_W - 1)
                row_idx = min(max(0, int((self.Ly - self.Y[i, j]) / map_dy)), map_H - 1)
                if not self.raw_boolean_grid[row_idx, col_idx]:
                    pde_mask[i, j] = 0
        return pde_mask

    def build_initial_density(self, sigma_multiplier: float = 1.5):
        if self._m0 is not None:
            return self._m0

        m_0 = np.zeros_like(self.X)
        sigma = sigma_multiplier * max(self.dx, self.dy)
        map_H, map_W = self.grid_shape
        map_dx, map_dy = self.Lx / map_W, self.Ly / map_H
        for agent in self.raw_agents:
            sx = (agent['start_x'] + 0.5) * map_dx
            sy = self.Ly - (agent['start_y'] + 0.5) * map_dy
            m_0 += np.exp(-((self.X - sx)**2 + (self.Y - sy)**2) / (2 * sigma**2))
        self._m0 = m_0
        return self._m0

    @property
    def m0(self):
        if self._m0 is None:
            self._m0 = self.build_initial_density()
        return self._m0

    @m0.setter
    def m0(self, value):
        self._m0 = value

    def get_goal_positions(self):
        """Returns order-preserved target physical coordinates."""
        map_H, map_W = self.grid_shape
        map_dx, map_dy = self.Lx / map_W, self.Ly / map_H
        goals = [((a['goal_x'] + 0.5) * map_dx, self.Ly - (a['goal_y'] + 0.5) * map_dy) for a in self.raw_agents]
        # Order-preserving deduplication
        return list(dict.fromkeys(goals))


class MFGTrafficGeometry:
    """Recreates the exact geometric layout from MFG_Traffic_numba_original.py."""

    def __init__(self, Lx: float = 50.0, Ly: float = 50.0, Nx: int = 75, Ny: int = 75):
        self.Lx, self.Ly = Lx, Ly
        self.Nx, self.Ny = Nx, Ny
        self.dx, self.dy = Lx / (Nx - 1), Ly / (Ny - 1)
        self.X, self.Y = None, None
        self._m0 = None

        # 5 Obstacle rectangles from original script
        self.orectangles = [
            (10.0, 40.0, 10.0, 14.0),
            (10.0, 40.0, 18.0, 22.0),
            (10.0, 40.0, 26.0, 30.0),
            (10.0, 40.0, 34.0, 38.0),
            (10.0, 40.0, 42.0, 46.0)
        ]

        # 5 Initial density corridors from original script
        self.rectangles = [
            (10.0, 40.0, 14.0, 18.0),
            (10.0, 40.0, 22.0, 26.0),
            (10.0, 40.0, 30.0, 34.0),
            (10.0, 40.0, 38.0, 42.0),
            (10.0, 40.0, 46.0, 50.0)
        ]

    def build_spatial_mesh(self):
        xSpace = np.linspace(0.0, self.Lx, self.Nx, endpoint=True)
        ySpace = np.linspace(0.0, self.Ly, self.Ny, endpoint=True)
        self.X, self.Y = np.meshgrid(xSpace, ySpace, indexing='ij')
        return self.X, self.Y

    def get_pde_obstacle_mask(self):
        omask = np.ones((self.Nx, self.Ny), dtype=np.int64)
        for (x1, x2, y1, y2) in self.orectangles:
            region = (self.X >= x1) & (self.X <= x2) & (self.Y >= y1) & (self.Y <= y2)
            omask[region] = 0
        return omask

    def build_initial_density(self):
        if self._m0 is not None:
            return self._m0

        m0 = np.zeros((self.Nx, self.Ny))
        for (x1, x2, y1, y2) in self.rectangles:
            region = (self.X > x1) & (self.X < x2) & (self.Y > y1) & (self.Y < y2)
            m0[region] = 4.0

        omask = self.get_pde_obstacle_mask()
        m0[omask == 0] = 0.0
        self._m0 = m0
        return self._m0

    @property
    def m0(self):
        if self._m0 is None:
            self._m0 = self.build_initial_density()
        return self._m0

    @m0.setter
    def m0(self, value):
        self._m0 = value

    def get_goal_positions(self):
        return [(5.0, 2.5), (45.0, 2.5)]


def create_moving_door_mask(door_trajectories, Nt, Nx, Ny, X, Y, T):
    door_mask_3d = np.zeros((Nt + 1, Nx, Ny))
    tSpace = np.linspace(0, T, Nt + 1)
    for k, t in enumerate(tSpace):
        for traj in door_trajectories:
            x1, x2 = traj['x1'](t), traj['x2'](t)
            y1, y2 = traj['y1'](t), traj['y2'](t)
            region = (X >= x1) & (X <= x2) & (Y >= y1) & (Y <= y2)
            door_mask_3d[k][region] = 1.0
    return door_mask_3d

def parse_motion_expr(expr_str, T_final, room_width, room_height):
    """
    Parses a string formula into a callable function of time t.
    """
    # Safe evaluation environment with standard math symbols
    eval_env = {
        'sin': np.sin,
        'cos': np.cos,
        'pi': np.pi,
        'T': float(T_final),
        'room_width': float(room_width),
        'room_height': float(room_height),
    }

    # Handle static float/int values
    if isinstance(expr_str, (int, float)):
        val = float(expr_str)
        return lambda t: val

    return lambda t: float(eval(expr_str, {"__builtins__": None}, {**eval_env, 't': float(t)}))

def build_door_trajectories_from_config(door_configs, T_final, room_width=50.0, room_height=50.0):
    """
    Converts YAML door configuration dictionaries into callable trajectory dicts.
    """
    door_trajectories = []
    for cfg in door_configs:
        door_trajectories.append({
            'x1': parse_motion_expr(cfg['x1'], T_final, room_width, room_height),
            'x2': parse_motion_expr(cfg['x2'], T_final, room_width, room_height),
            'y1': parse_motion_expr(cfg['y1'], T_final, room_width, room_height),
            'y2': parse_motion_expr(cfg['y2'], T_final, room_width, room_height),
        })
    return door_trajectories
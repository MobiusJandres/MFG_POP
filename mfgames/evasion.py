"""
Goal management module supporting stationary goals, prescribed paths, and evasive goals.
"""
import numpy as np


def eval_motion_expr(expr_str, t, T_final, room_size, default_val):
    """Evaluates time-dependent parametric motion expressions."""
    if expr_str is None:
        return float(default_val)
    if isinstance(expr_str, (int, float)):
        return float(expr_str)
    eval_env = {
        'sin': np.sin,
        'cos': np.cos,
        'pi': np.pi,
        't': float(t),
        'T': float(T_final),
        'L': float(room_size),
    }
    return float(eval(str(expr_str), {"__builtins__": None}, eval_env))


class Goal:
    """Manages system goals (stationary positions, prescribed trajectories, or evasive goals)."""

    def __init__(self, goal_configs: list, Nt: int, Dt: float, T: float = 3.0, Lx: float = 768.0, Ly: float = 768.0):
        self.Nt = Nt
        self.Dt = Dt
        self.T = T
        self.Lx, self.Ly = Lx, Ly
        self.goals = []

        for cfg in goal_configs:
            if isinstance(cfg, (list, tuple)):
                cfg_dict = {'type': 'evader', 'position': [float(cfg[0]), float(cfg[1])], 'v_max': 15.0}
            elif isinstance(cfg, dict):
                pos = cfg.get('position', [0.0, 0.0])
                cfg_dict = {
                    'type': str(cfg.get('type', 'evader')).lower(),
                    'position': [float(pos[0]), float(pos[1])],
                    'v_max': float(cfg.get('v_max', 15.0)),
                    'path_x': cfg.get('path_x', None),
                    'path_y': cfg.get('path_y', None)
                }
            else:
                raise ValueError(f"Invalid goal configuration format: {cfg}")

            self.goals.append(cfg_dict)

        self.num_goals = len(self.goals)
        self.Y_trajectories = np.zeros((Nt + 1, self.num_goals, 2))

        # Precompute trajectories for stationary and prescribed goals
        tSpace = np.linspace(0, T, Nt + 1)
        for g_idx, g_info in enumerate(self.goals):
            init_pos = g_info['position']
            self.Y_trajectories[0, g_idx, :] = init_pos

            if g_info['type'] == 'stationary':
                for k in range(1, Nt + 1):
                    self.Y_trajectories[k, g_idx, :] = init_pos

            elif g_info['type'] == 'prescribed':
                x_expr, y_expr = g_info['path_x'], g_info['path_y']
                for k, t_val in enumerate(tSpace):
                    px = eval_motion_expr(x_expr, t_val, T, Lx, init_pos[0])
                    py = eval_motion_expr(y_expr, t_val, T, Ly, init_pos[1])
                    self.Y_trajectories[k, g_idx, :] = [px, py]

            elif g_info['type'] == 'evader':
                for k in range(1, Nt + 1):
                    self.Y_trajectories[k, g_idx, :] = init_pos

    @property
    def has_dynamic_goals(self) -> bool:
        """Returns True if any goal moves via evader logic or prescribed paths."""
        return any(g['type'] in ('evader', 'prescribed') for g in self.goals)

    def update_positions(self, M_field, omask, Dx, Dy, Lx, Ly):
        """Updates goal trajectories based on drone swarm density M_field."""
        Nx, Ny = omask.shape
        new_trajectories = np.copy(self.Y_trajectories)

        x_coords = np.linspace(Dx / 2, Lx - Dx / 2, Nx)
        y_coords = np.linspace(Dy / 2, Ly - Dy / 2, Ny)
        X_grid, Y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')

        for k in range(self.Nt):
            M_k = M_field[k]
            total_mass = np.sum(M_k)

            for g, g_info in enumerate(self.goals):
                if g_info['type'] == 'stationary':
                    new_trajectories[k + 1, g] = self.Y_trajectories[0, g]

                elif g_info['type'] == 'prescribed':
                    new_trajectories[k + 1, g] = self.Y_trajectories[k + 1, g]

                elif g_info['type'] == 'evader':
                    curr_x, curr_y = new_trajectories[k, g]
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

                    next_x = np.clip(curr_x + vx * self.Dt, Dx, Lx - Dx)
                    next_y = np.clip(curr_y + vy * self.Dt, Dy, Ly - Dy)

                    next_i = int(np.clip(next_x / Dx, 0, Nx - 1))
                    next_j = int(np.clip(next_y / Dy, 0, Ny - 1))

                    if omask[next_i, next_j] == 1:
                        new_trajectories[k + 1, g] = [next_x, next_y]
                    else:
                        new_trajectories[k + 1, g] = [curr_x, curr_y]

        return new_trajectories


# Backward compatibility aliases
TargetSwarm = Goal
EvaderSwarm = Goal
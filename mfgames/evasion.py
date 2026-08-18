"""
Target Evader swarm dynamics with order-preserving goal initialization.
"""
import numpy as np


class EvaderSwarm:
    def __init__(self, initial_goals, Nt: int, Dt: float, v_max: float = 15.0):
        # Deduplicate goals while preserving scenario order
        unique_goals = list(dict.fromkeys([(float(g[0]), float(g[1])) for g in initial_goals]))
        self.num_evaders = len(unique_goals)
        self.Nt = Nt
        self.Dt = Dt
        self.v_max = v_max

        self.Y_trajectories = np.zeros((Nt + 1, self.num_evaders, 2))
        for e_idx, pos in enumerate(unique_goals):
            self.Y_trajectories[0, e_idx, :] = pos

        for k in range(1, Nt + 1):
            self.Y_trajectories[k] = np.copy(self.Y_trajectories[0])

    def update_evader_positions(self, M_field, omask, Dx, Dy, Lx, Ly):
        Nx, Ny = omask.shape
        new_trajectories = np.zeros_like(self.Y_trajectories)
        new_trajectories[0] = np.copy(self.Y_trajectories[0])

        x_coords = np.linspace(Dx / 2, Lx - Dx / 2, Nx)
        y_coords = np.linspace(Dy / 2, Ly - Dy / 2, Ny)
        X_grid, Y_grid = np.meshgrid(x_coords, y_coords, indexing='ij')

        for k in range(self.Nt):
            M_k = M_field[k]
            total_mass = np.sum(M_k)

            for e in range(self.num_evaders):
                curr_x, curr_y = new_trajectories[k, e]

                if total_mass > 1e-6:
                    rx = curr_x - X_grid
                    ry = curr_y - Y_grid
                    dist_sq = rx**2 + ry**2 + 1e-3

                    force_x = np.sum((rx / dist_sq) * M_k)
                    force_y = np.sum((ry / dist_sq) * M_k)
                    norm_force = np.sqrt(force_x**2 + force_y**2)

                    if norm_force > 1e-8:
                        vx = (force_x / norm_force) * self.v_max
                        vy = (force_y / norm_force) * self.v_max
                    else:
                        vx, vy = 0.0, 0.0
                else:
                    vx, vy = 0.0, 0.0

                next_x = np.clip(curr_x + vx * self.Dt, Dx, Lx - Dx)
                next_y = np.clip(curr_y + vy * self.Dt, Dy, Ly - Dy)

                next_i = int(np.clip(next_x / Dx, 0, Nx - 1))
                next_j = int(np.clip(next_y / Dy, 0, Ny - 1))

                if omask[next_i, next_j] == 1:
                    new_trajectories[k + 1, e] = [next_x, next_y]
                else:
                    new_trajectories[k + 1, e] = [curr_x, curr_y]

        return new_trajectories
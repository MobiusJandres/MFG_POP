"""
Unified Object-Oriented wrapper supporting both Hard Doors and Pursuit-Evasion Targets.
"""
import time
import numpy as np
from mfgames.evasion import EvaderSwarm
from mfgames.solvers import solveFP_2D, solveHJB_withM


class MFGSolver:
    def __init__(self, pde_mesh_data, T: float = 3.0, Nt: int = 100, thetaUM: float = 0.1, door_mask_3d=None, is_pursuit_evasion=False, v_max_evader=15.0):
        self.pde_mesh = pde_mesh_data
        self.Nx, self.Ny = pde_mesh_data.X.shape
        self.Lx, self.Ly = pde_mesh_data.Lx, pde_mesh_data.Ly
        self.Dx, self.Dy = pde_mesh_data.dx, pde_mesh_data.dy
        self.Dt, self.Nt = T / Nt, Nt
        self.thetaUM = thetaUM
        self.is_pursuit_evasion = is_pursuit_evasion

        self.omask = pde_mesh_data.get_pde_obstacle_mask()
        self.m0 = pde_mesh_data.build_initial_density()

        self.M = np.zeros((self.Nt + 1, self.Nx, self.Ny))
        self.U = np.zeros((self.Nt + 1, self.Nx, self.Ny))
        self.M[0] = self.m0

        if is_pursuit_evasion:
            goals = pde_mesh_data.get_goal_positions()
            self.evader_swarm = EvaderSwarm(goals, Nt, self.Dt, v_max=v_max_evader)
            self.door_mask_3d = np.zeros((Nt + 1, self.Nx, self.Ny))
        else:
            self.evader_swarm = None
            self.door_mask_3d = door_mask_3d if door_mask_3d is not None else np.zeros((Nt + 1, self.Nx, self.Ny))

    def compute_running_cost_history(self, evader_trajectories):
        rc = np.zeros((self.Nt + 1, self.Nx, self.Ny))
        X, Y = self.pde_mesh.X, self.pde_mesh.Y
        for k in range(self.Nt + 1):
            min_dist_sq = np.full((self.Nx, self.Ny), 1e6)
            for ex, ey in evader_trajectories[k]:
                min_dist_sq = np.minimum(min_dist_sq, (X - ex)**2 + (Y - ey)**2)
            rc[k] = 0.01 * min_dist_sq
        return rc

    def run_picard_system(self, max_iters: int = 10, tolerance: float = 1e-5):
        space_time_factor = np.sqrt(self.Dx * self.Dy * self.Dt)

        for iiter in range(1, max_iters + 1):
            start_time = time.time()
            print(f"\n>>> Macro Picard Loop Execution: {iiter} / {max_iters}", flush=True)

            rc_hist = self.compute_running_cost_history(self.evader_swarm.Y_trajectories) if self.is_pursuit_evasion else None

            U_temp = solveHJB_withM(
                self.U, self.M, self.door_mask_3d, self.omask, rc_hist,
                self.Nx, self.Ny, self.Nt, self.Dx, self.Dy, self.Dt
            )
            U_new = self.thetaUM * U_temp + (1.0 - self.thetaUM) * self.U

            M_temp = solveFP_2D(
                self.m0, U_new, self.door_mask_3d, self.omask,
                self.Nx, self.Ny, self.Nt, self.Dx, self.Dy, self.Dt
            )
            M_new = self.thetaUM * M_temp + (1.0 - self.thetaUM) * self.M

            if self.is_pursuit_evasion:
                Y_temp = self.evader_swarm.update_evader_positions(M_new, self.omask, self.Dx, self.Dy, self.Lx, self.Ly)
                Y_new = self.thetaUM * Y_temp + (1.0 - self.thetaUM) * self.evader_swarm.Y_trajectories
                y_err = np.linalg.norm(Y_new - self.evader_swarm.Y_trajectories)
                self.evader_swarm.Y_trajectories = np.copy(Y_new)
            else:
                y_err = 0.0

            u_err = np.linalg.norm(U_new - self.U) * space_time_factor
            m_err = np.linalg.norm(M_new - self.M) * space_time_factor

            print(f"    u_residual: {u_err:.6e} | m_residual: {m_err:.6e} | y_evader_residual: {y_err:.6e} | Time: {time.time() - start_time:.2f}s", flush=True)

            self.U = np.copy(U_new)
            self.M = np.copy(M_new)

            if u_err < tolerance and m_err < tolerance and y_err < tolerance:
                print(f"\n[Success] Converged at iteration {iiter}!", flush=True)
                break

        return self.U, self.M
"""
Unified Object-Oriented solvers for 1-Population, Pursuit-Evasion, and 2-Population systems.
"""
import time
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
from mfgames.evasion import Goal
from mfgames.solvers import solveFP_2D, solveHJB_withM
from mfgames.numerics import (
    compute_FP_matrix_entries,
    getFnU_2D,
    compute_HJB_matrix_entries,
    compute_FP_matrix_entries_2Pop,
    getFnU_2D_2Pop,
    compute_HJB_matrix_entries_2Pop,
)


class MFGSolver:
    """Mean Field Game solver for Drone Swarms heading towards Goals."""

    def __init__(
        self,
        pde_mesh_data,
        T: float = 3.0,
        Nt: int = 100,
        thetaUM: float = 0.1,
        door_mask_3d=None,
        goal_configs: list|None = None,
        goals_are_exits: bool = False,
        obstacle_penalty: float = -500.0,
        running_cost_weight: float = 0.01
    ):
        self.pde_mesh = pde_mesh_data
        self.Nx, self.Ny = pde_mesh_data.X.shape
        self.Lx, self.Ly = pde_mesh_data.Lx, pde_mesh_data.Ly
        self.Dx, self.Dy = pde_mesh_data.dx, pde_mesh_data.dy
        self.Dt, self.Nt = T / Nt, Nt
        self.thetaUM = thetaUM
        self.goals_are_exits = goals_are_exits
        self.running_cost_weight = running_cost_weight

        # Dynamically scale obstacle penalty relative to max potential drop on grid
        if obstacle_penalty is not None:
            self.obstacle_penalty = obstacle_penalty
        else:
            max_grid_dist_sq = self.Lx**2 + self.Ly**2
            max_cost = self.running_cost_weight * max_grid_dist_sq
            self.obstacle_penalty = -5.0 * max_cost        

        self.omask = pde_mesh_data.get_pde_obstacle_mask()
        self.m0 = pde_mesh_data.build_initial_density()

        self.M = np.zeros((self.Nt + 1, self.Nx, self.Ny))
        self.U = np.zeros((self.Nt + 1, self.Nx, self.Ny))
        self.M[0] = self.m0

        # Automatic Goal detection
        raw_goals = goal_configs if goal_configs is not None else pde_mesh_data.get_goals()
        if raw_goals:
            self.goal = Goal(
                goal_configs=raw_goals,
                Nt=Nt,
                Dt=self.Dt,
                T=T,
                Lx=self.Lx,
                Ly=self.Ly
            )
            # Alias for backward compatibility with MFGPlotter
            self.evader_swarm = self.goal
            self.door_mask_3d = self._build_dynamic_goal_doors(self.goal.Y_trajectories)
        else:
            self.goal = None
            self.evader_swarm = None
            if door_mask_3d is not None:
                self.door_mask_3d = door_mask_3d #if goals_are_exits else np.zeros((Nt + 1, self.Nx, self.Ny))
            else:
                self.door_mask_3d = np.zeros((Nt + 1, self.Nx, self.Ny))

    def _build_dynamic_goal_doors(self, goal_trajectories):
        """Constructs 3D door mask around moving evaders if targets_are_exits is enabled."""
        door_mask = np.zeros((self.Nt + 1, self.Nx, self.Ny))
        if not self.goal or goal_trajectories is None:
            return door_mask

        X, Y = self.pde_mesh.X, self.pde_mesh.Y
        for k in range(self.Nt + 1):
            for g_idx, g_info in enumerate(self.goal.goals):
                if g_info.get('is_exit', False):
                    gx, gy = goal_trajectories[k, g_idx]
                    region = (np.abs(X - gx) <= self.Dx) & (np.abs(Y - gy) <= self.Dy)
                    door_mask[k][region] = 1.0
        return door_mask

    def compute_running_cost(self, goal_positions_k):
        X, Y = self.pde_mesh.X, self.pde_mesh.Y
        min_dist_sq = np.full((self.Nx, self.Ny), 1e6)
        for gx, gy in goal_positions_k:
            dist_sq = (X - gx)**2 + (Y - gy)**2
            min_dist_sq = np.minimum(min_dist_sq, dist_sq)
        return self.running_cost_weight * min_dist_sq

    def solve_forward_FP_step(self, U_trajectory, door_mask_3d):
        m = np.zeros_like(self.M)
        m[0] = self.m0
        N_total = self.Nx * self.Ny

        for k in range(1, self.Nt + 1):
            rows, cols, vals, b = compute_FP_matrix_entries(
                m[k - 1], U_trajectory[k - 1], self.omask, door_mask_3d[k - 1],
                self.Nx, self.Ny, self.Dx, self.Dy, self.Dt
            )
            A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()
            mtmp = sparse.linalg.spsolve(A, b)
            m[k] = mtmp.reshape((self.Nx, self.Ny))
        return m

    def solve_backward_HJB_step(self, M_trajectory, goal_trajectories, door_mask_3d):
        u = np.zeros_like(self.U)
        running_cost_Nt = self.compute_running_cost(goal_trajectories[self.Nt])
        u[self.Nt] = -running_cost_Nt

        for k in range(self.Nt - 1, -1, -1):
            running_cost_k = self.compute_running_cost(goal_trajectories[k])
            Unew_n = np.copy(u[k + 1])
            N_total = self.Nx * self.Ny

            for _ in range(30):
                FnU_flat = getFnU_2D(
                    u[k + 1], Unew_n, M_trajectory[k + 1], self.omask, door_mask_3d[k],
                    running_cost_k, self.Nx, self.Ny, self.Dx, self.Dy, self.Dt,
                    obstacle_penalty=self.obstacle_penalty
                ).flatten()

                rows, cols, vals = compute_HJB_matrix_entries(
                    Unew_n, M_trajectory[k + 1], self.omask, door_mask_3d[k],
                    self.Nx, self.Ny, self.Dx, self.Dy, self.Dt
                )
                A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()
                b = A.dot(Unew_n.flatten()) - FnU_flat

                for i in range(self.Nx):
                    for j in range(self.Ny):
                        if self.omask[i, j] == 0:
                            b[i * self.Ny + j] = self.obstacle_penalty

                Unres = sparse.linalg.spsolve(A, b).reshape((self.Nx, self.Ny))
                l2err = np.linalg.norm(Unew_n.flatten() - Unres.flatten()) * np.sqrt(self.Dx * self.Dy)
                Unew_n = np.copy(Unres)
                if l2err < 1e-6:
                    break
            u[k] = Unew_n
        return u

    def run_picard_system(self, max_iters: int = 10, tolerance: float = 1e-5):
        space_time_factor = np.sqrt(self.Dx * self.Dy * self.Dt)

        for iiter in range(1, max_iters + 1):
            start_time = time.time()
            print(f"\n>>> Macro Picard Loop Execution: {iiter} / {max_iters}", flush=True)

            if self.goal is not None:
                current_door_mask = self._build_dynamic_goal_doors(self.goal.Y_trajectories)

                U_temp = self.solve_backward_HJB_step(self.M, self.goal.Y_trajectories, current_door_mask)
                U_new = self.thetaUM * U_temp + (1.0 - self.thetaUM) * self.U

                M_temp = self.solve_forward_FP_step(U_new, current_door_mask)
                M_new = self.thetaUM * M_temp + (1.0 - self.thetaUM) * self.M

                Y_temp = self.goal.update_positions(M_new, self.omask, self.Dx, self.Dy, self.Lx, self.Ly)
                Y_new = self.thetaUM * Y_temp + (1.0 - self.thetaUM) * self.goal.Y_trajectories

                y_err = np.linalg.norm(Y_new - self.goal.Y_trajectories)
                self.goal.Y_trajectories = np.copy(Y_new)
            else:
                U_temp = solveHJB_withM(
                    self.U, self.M, self.door_mask_3d, self.omask, None,
                    self.Nx, self.Ny, self.Nt, self.Dx, self.Dy, self.Dt
                )
                U_new = self.thetaUM * U_temp + (1.0 - self.thetaUM) * self.U

                M_temp = solveFP_2D(
                    self.m0, U_new, self.door_mask_3d, self.omask,
                    self.Nx, self.Ny, self.Nt, self.Dx, self.Dy, self.Dt
                )
                M_new = self.thetaUM * M_temp + (1.0 - self.thetaUM) * self.M
                y_err = 0.0

            u_err = np.linalg.norm(U_new - self.U) * space_time_factor
            m_err = np.linalg.norm(M_new - self.M) * space_time_factor

            print(f"    u_residual: {u_err:.6e} | m_residual: {m_err:.6e} | y_goal_residual: {y_err:.6e} | Time: {time.time() - start_time:.2f}s", flush=True)

            self.U = np.copy(U_new)
            self.M = np.copy(M_new)

            if u_err < tolerance and m_err < tolerance and y_err < tolerance:
                print(f"\n[Success] Converged at iteration {iiter}!", flush=True)
                break

        return self.U, self.M


class MFG2PopSolver:
    """Coupled 2-Population HJB-FP Mean Field Game Solver."""

    def __init__(self, pde_mesh_data_1, pde_mesh_data_2, T: float = 300.0, Nt: int = 3000, thetaUM: float = 0.1):
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
        N_total = self.Nx * self.Ny
        FnU_flat = getFnU_2D_2Pop(
            Unew_np1, Unew_n_tmp, Mk_np1, Mk_other,
            self.omask, self.Nx, self.Ny, self.Dx, self.Dy, self.Dt, pop
        ).flatten()

        rows, cols, vals = compute_HJB_matrix_entries_2Pop(
            Unew_n_tmp, Mk_np1, Mk_other, self.omask,
            self.Nx, self.Ny, self.Dx, self.Dy, self.Dt
        )
        A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()
        b = A.dot(Unew_n_tmp.flatten()) - FnU_flat

        for i in range(self.Nx):
            for j in range(self.Ny):
                if self.omask[i, j] == 0:
                    b[i * self.Ny + j] = -500.0

        utmp = sparse.linalg.spsolve(A, b)
        return utmp.reshape((self.Nx, self.Ny))

    def solve_backward_HJB(self, M_trajectory, M_trajectory_other, U_temp, g_x, pop):
        u = np.zeros_like(U_temp)
        u[self.Nt] = g_x
        for k in range(self.Nt - 1, -1, -1):
            Unew_n = np.copy(u[k + 1])
            for _ in range(5):
                Unres = self.get_u_onestep_newton(u[k], u[k + 1], Unew_n, M_trajectory[k + 1], M_trajectory_other[k + 1], pop)
                l2err = np.linalg.norm(Unew_n.flatten() - Unres.flatten()) * np.sqrt(self.Dx * self.Dy)
                Unew_n = np.copy(Unres)
                if l2err < 1e-6:
                    break
            u[k] = Unew_n
        return u

    def run_picard_system(self, max_iters: int = 25, tolerance: float = 1e-5):
        space_time_factor = np.sqrt(self.Dx * self.Dy * self.Dt)

        for iiter in range(1, max_iters + 1):
            start_time = time.time()
            print(f"\n>>> 2-Pop Picard Loop Iteration: {iiter} / {max_iters}", flush=True)

            U1_temp = self.solve_backward_HJB(self.M1, self.M2, self.U1, self.g_x_1, pop=1)
            U2_temp = self.solve_backward_HJB(self.M2, self.M1, self.U2, self.g_x_2, pop=2)

            U1_new = self.thetaUM * U1_temp + (1.0 - self.thetaUM) * self.U1
            U2_new = self.thetaUM * U2_temp + (1.0 - self.thetaUM) * self.U2

            M1_temp = self.solve_forward_FP(U1_new, self.m0_1, self.M2)
            M2_temp = self.solve_forward_FP(U2_new, self.m0_2, self.M1)

            M1_new = self.thetaUM * M1_temp + (1.0 - self.thetaUM) * self.M1
            M2_new = self.thetaUM * M2_temp + (1.0 - self.thetaUM) * self.M2

            u_err = max(np.linalg.norm(U1_new - self.U1), np.linalg.norm(U2_new - self.U2)) * space_time_factor
            m_err = max(np.linalg.norm(M1_new - self.M1), np.linalg.norm(M2_new - self.M2)) * space_time_factor

            print(f"    u_residual: {u_err:.6e} | m_residual: {m_err:.6e} | Time: {time.time() - start_time:.2f}s", flush=True)

            self.U1, self.M1 = np.copy(U1_new), np.copy(M1_new)
            self.U2, self.M2 = np.copy(U2_new), np.copy(M2_new)

            if u_err < tolerance and m_err < tolerance:
                print(f"\n[Success] 2-Population system converged at iteration {iiter}!", flush=True)
                break

        return self.U1, self.M1, self.U2, self.M2
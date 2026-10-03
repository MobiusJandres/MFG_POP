# %%
import numpy as np
import time
import scipy.sparse as sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit
import os

# %% Numba Utilities with Cross-Population Mortality
@jit(nopython=True, cache=True)
def ppart(x):
    return np.maximum(x, 0.0)

@jit(nopython=True, cache=True)
def npart(x):
    return -np.minimum(x, 0.0)

@jit(nopython=True, cache=True)
def compute_FP_matrix_entries_2Pop(m_prev, m_other, ukm1, omask_arr, Nx, Ny, Dx, Dy, Dt, gamma_mortality=0.5):
    """
    Fokker-Planck solver matrix builder with inter-population mortality sink term.
    Mass depletion: -gamma_mortality * m_prev * m_other
    """
    N_total = Nx * Ny
    max_entries = N_total * 9
    rows = np.zeros(max_entries, dtype=np.int64)
    cols = np.zeros(max_entries, dtype=np.int64)
    vals = np.zeros(max_entries, dtype=np.float64)
    b = np.zeros(N_total, dtype=np.float64)

    entry_idx = 0

    for i in range(Nx):
        for j in range(Ny):
            ind = i * Ny + j
            b[ind] = m_prev[i, j] / Dt

            if omask_arr[i, j] == 0:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, 1.0
                entry_idx += 1
                b[ind] = 0.0
                continue

            ip1 = i + 1 if (i < Nx - 1 and omask_arr[i + 1, j] == 1) else i
            im1 = i - 1 if (i > 0      and omask_arr[i - 1, j] == 1) else i
            jp1 = j + 1 if (j < Ny - 1 and omask_arr[i, j + 1] == 1) else j
            jm1 = j - 1 if (j > 0      and omask_arr[i, j - 1] == 1) else j

            p1 = (ukm1[ip1, j] - ukm1[i, j]) / Dx 
            p2 = (ukm1[i, j] - ukm1[im1, j]) / Dx
            p3 = (ukm1[i, jp1] - ukm1[i, j]) / Dy
            p4 = (ukm1[i, j] - ukm1[i, jm1]) / Dy
            
            # Diagonal term including time step, diffusion, advection, AND mortality depletion
            diag_val = (1.0 / Dt) + gamma_mortality * m_other[i, j]
            
            if ip1 != i: diag_val += 0.05 / (Dx ** 2)
            if im1 != i: diag_val += 0.05 / (Dx ** 2)
            if jp1 != j: diag_val += 0.05 / (Dy ** 2)
            if jm1 != j: diag_val += 0.05 / (Dy ** 2)

            c_h_curr = 2.0 / (1.0 + m_prev[i, j] + 5.0 * m_other[i, j])
            diag_val += c_h_curr * (npart(p1) / Dx + ppart(p2) / Dx + npart(p3) / Dy + ppart(p4) / Dy)

            rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, diag_val
            entry_idx += 1

            if ip1 != i:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, (ip1 * Ny + j), -0.05 / (Dx ** 2)
                entry_idx += 1
            if im1 != i:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, (im1 * Ny + j), -0.05 / (Dx ** 2)
                entry_idx += 1
            if jp1 != j:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, (i * Ny + jp1), -0.05 / (Dy ** 2)
                entry_idx += 1
            if jm1 != j:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, (i * Ny + jm1), -0.05 / (Dy ** 2)
                entry_idx += 1

            if i > 0 and omask_arr[i - 1, j] == 1:
                c_h_left = 2.0 / (1.0 + m_prev[i - 1, j] + 5.0 * m_other[i - 1, j])
                vals[entry_idx] = -c_h_left * npart((ukm1[i, j] - ukm1[i - 1, j]) / Dx) / Dx
                rows[entry_idx], cols[entry_idx] = ind, ((i - 1) * Ny + j)
                entry_idx += 1
            if i < Nx - 1 and omask_arr[i + 1, j] == 1:
                c_h_right = 2.0 / (1.0 + m_prev[i + 1, j] + 5.0 * m_other[i + 1, j])
                vals[entry_idx] = -c_h_right * ppart((ukm1[i + 1, j] - ukm1[i, j]) / Dx) / Dx
                rows[entry_idx], cols[entry_idx] = ind, ((i + 1) * Ny + j)
                entry_idx += 1
            if j > 0 and omask_arr[i, j - 1] == 1:
                c_h_bottom = 2.0 / (1.0 + m_prev[i, j - 1] + 5.0 * m_other[i, j - 1])
                vals[entry_idx] = -c_h_bottom * npart((ukm1[i, j] - ukm1[i, j - 1]) / Dy) / Dy
                rows[entry_idx], cols[entry_idx] = ind, (i * Ny + j - 1)
                entry_idx += 1
            if j < Ny - 1 and omask_arr[i, j + 1] == 1:
                c_h_top = 2.0 / (1.0 + m_prev[i, j + 1] + 5.0 * m_other[i, j + 1])
                vals[entry_idx] = -c_h_top * ppart((ukm1[i, j + 1] - ukm1[i, j]) / Dy) / Dy
                rows[entry_idx], cols[entry_idx] = ind, (i * Ny + j + 1)
                entry_idx += 1
            
    return rows[:entry_idx], cols[:entry_idx], vals[:entry_idx], b

@jit(nopython=True, cache=True)
def getFnU_2D_2Pop(Ukp1_np1, Ukp1_n, Mk_np1, Mk_other, V_goal_k, omask_arr, Nx, Ny, Dx, Dy, Dt, pop=1, gamma_mortality=0.5):
    """
    HJB Residual calculation including mortality discount/hazard rate.
    """
    FnU = np.zeros((Nx, Ny))

    for i in range(Nx):
        for j in range(Ny):
            if omask_arr[i, j] == 0:
                FnU[i, j] = Ukp1_n[i, j] + 500.0
                continue
            
            ip1 = i + 1 if (i < Nx - 1 and omask_arr[i + 1, j] == 1) else i
            im1 = i - 1 if (i > 0      and omask_arr[i - 1, j] == 1) else i
            jp1 = j + 1 if (j < Ny - 1 and omask_arr[i, j + 1] == 1) else j
            jm1 = j - 1 if (j > 0      and omask_arr[i, j - 1] == 1) else j

            time_deriv = -(Ukp1_np1[i, j] - Ukp1_n[i, j]) / Dt

            p1 = (Ukp1_n[ip1, j] - Ukp1_n[i, j]) / Dx
            p2 = (Ukp1_n[i, j] - Ukp1_n[im1, j]) / Dx
            p3 = (Ukp1_n[i, jp1] - Ukp1_n[i, j]) / Dy
            p4 = (Ukp1_n[i, j] - Ukp1_n[i, jm1]) / Dy

            laplacian_x = (Ukp1_n[ip1, j] - 2 * Ukp1_n[i, j] + Ukp1_n[im1, j]) / (Dx ** 2)
            laplacian_y = (Ukp1_n[i, jp1] - 2 * Ukp1_n[i, j] + Ukp1_n[i, jm1]) / (Dy ** 2)
            diffusion = -0.05 * (laplacian_x + laplacian_y)

            hamiltonian = (1 / (1 + Mk_np1[i, j] + 5 * Mk_other[i, j])) * (
                        npart(p1)**2 + ppart(p2)**2 + npart(p3)**2 + ppart(p4)**2)
            
            # Interaction cost + mortality hazard rate term
            mortality_hazard = gamma_mortality * Mk_other[i, j] * Ukp1_n[i, j]

            if pop == 1:
                interaction_cost = +1500.0 * Mk_other[i, j] + ppart(Mk_np1[i, j] + Mk_other[i, j] - 4)
            else:
                interaction_cost = -5000.0 * Mk_other[i, j] + ppart(Mk_np1[i, j] + Mk_other[i, j] - 4)

            FnU[i, j] = time_deriv + diffusion + hamiltonian + interaction_cost + V_goal_k[i, j] + mortality_hazard
    return FnU

@jit(nopython=True, cache=True)
def compute_HJB_matrix_entries(Unew_n_tmp, Mk_np1, Mk_other, omask_arr, Nx, Ny, Dx, Dy, Dt, gamma_mortality=0.5):
    """
    HJB Jacobian matrix builder accounting for mortality discount diagonal entries.
    """
    N_total = Nx * Ny
    max_entries = N_total * 16
    rows = np.zeros(max_entries, dtype=np.int64)
    cols = np.zeros(max_entries, dtype=np.int64)
    vals = np.zeros(max_entries, dtype=np.float64)

    entry_idx = 0

    for i in range(Nx):
        for j in range(Ny):
            ind = i * Ny + j
            if omask_arr[i, j] == 0:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, 1.0
                entry_idx += 1
                continue
            
            # Diagonal incorporating time step and mortality hazard rate
            diag_val = (1.0 / Dt) + gamma_mortality * Mk_other[i, j]

            ip1 = i + 1 if (i < Nx - 1 and omask_arr[i + 1, j] == 1) else i
            im1 = i - 1 if (i > 0      and omask_arr[i - 1, j] == 1) else i
            jp1 = j + 1 if (j < Ny - 1 and omask_arr[i, j + 1] == 1) else j
            jm1 = j - 1 if (j > 0      and omask_arr[i, j - 1] == 1) else j

            if ip1 != i: diag_val += 0.05 / (Dx ** 2)
            if im1 != i: diag_val += 0.05 / (Dx ** 2)
            if jp1 != j: diag_val += 0.05 / (Dy ** 2)
            if jm1 != j: diag_val += 0.05 / (Dy ** 2)

            p1 = (Unew_n_tmp[ip1, j] - Unew_n_tmp[i, j]) / Dx
            p2 = (Unew_n_tmp[i, j] - Unew_n_tmp[im1, j]) / Dx
            p3 = (Unew_n_tmp[i, jp1] - Unew_n_tmp[i, j]) / Dy
            p4 = (Unew_n_tmp[i, j] - Unew_n_tmp[i, jm1]) / Dy

            c_h = 2.0 / (1.0 + Mk_np1[i, j] + 5.0 * Mk_other[i, j])

            rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, diag_val
            entry_idx += 1

            if ip1 != i:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, (ip1 * Ny + j), -0.05 / (Dx ** 2)
                entry_idx += 1
            if im1 != i:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, (im1 * Ny + j), -0.05 / (Dx ** 2)
                entry_idx += 1
            if jp1 != j:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, (i * Ny + jp1), -0.05 / (Dy ** 2)
                entry_idx += 1
            if jm1 != j:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, (i * Ny + jm1), -0.05 / (Dy ** 2)
                entry_idx += 1

            if ip1 != i:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, (ip1 * Ny + j), -c_h * npart(p1) / Dx
                entry_idx += 1
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, c_h * npart(p1) / Dx
                entry_idx += 1
            if im1 != i:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, (im1 * Ny + j), -c_h * ppart(p2) / Dx
                entry_idx += 1
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, c_h * ppart(p2) / Dx
                entry_idx += 1
            if jp1 != j:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, (i * Ny + jp1), -c_h * npart(p3) / Dy
                entry_idx += 1
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, c_h * npart(p3) / Dy
                entry_idx += 1
            if jm1 != j:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, (i * Ny + jm1), -c_h * ppart(p4) / Dy
                entry_idx += 1
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, c_h * ppart(p4) / Dy
                entry_idx += 1
    return rows[:entry_idx], cols[:entry_idx], vals[:entry_idx]

# %% Map & Mesh Data Interface
class MAP2PDE:
    def __init__(self, Lx: float = 768.0, Ly: float = 768.0, Nx: int = 80, Ny: int = 80):
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        
        xSpace = np.linspace(self.dx / 2, self.Lx - self.dx / 2, self.Nx)
        ySpace = np.linspace(self.dy / 2, self.Ly - self.dy / 2, self.Ny)
        self.X, self.Y = np.meshgrid(xSpace, ySpace, indexing='ij')
        
        self.custom_m0 = None
        self.custom_goals = []

    def get_pde_obstacle_mask(self):
        return np.ones((self.Nx, self.Ny), dtype=np.int64)

    def set_custom_initial_blobs(self, blob_list):
        m_0 = np.zeros_like(self.X, dtype=np.float64)
        for blob in blob_list:
            cx, cy, sigma, amp = blob
            m_0 += amp * np.exp(-((self.X - cx)**2 + (self.Y - cy)**2) / (2.0 * sigma**2))
        self.custom_m0 = m_0
        return m_0

    def set_custom_goals(self, goal_list):
        self.custom_goals = goal_list

    def get_goals(self):
        return self.custom_goals

    def build_terminal_cost(self, penalty_scale: float = 15.0):
        g = np.zeros((self.Nx, self.Ny), dtype=np.float64)
        if not self.custom_goals:
            return g
        for i in range(self.Nx):
            for j in range(self.Ny):
                x_val, y_val = self.X[i, j], self.Y[i, j]
                min_dist = min(np.sqrt((x_val - gx)**2 + (y_val - gy)**2) for gx, gy in self.custom_goals)
                g[i, j] = penalty_scale * min_dist
        return g

# %% Two-Population Solver with Mortality
class MFG2PopMortalitySolver:
    def __init__(self, mesh_1, mesh_2, T: float = 5.0, Nt: int = 100, thetaUM: float = 0.1, 
                 gamma_mortality_1: float = 0.3, gamma_mortality_2: float = 0.3):
        self.Nx, self.Ny = mesh_1.X.shape
        self.Dx = mesh_1.dx  
        self.Dy = mesh_1.dy  
        self.Dt = T / Nt
        self.Nt = Nt
        self.thetaUM = thetaUM
        
        # Inter-population mortality coefficients
        self.gamma_mortality_1 = gamma_mortality_1  # Population 1 mortality rate caused by Pop 2
        self.gamma_mortality_2 = gamma_mortality_2  # Population 2 mortality rate caused by Pop 1

        self.omask = mesh_1.get_pde_obstacle_mask()
        self.m0_1 = mesh_1.custom_m0
        self.g_x_1 = mesh_1.build_terminal_cost()
        self.m0_2 = mesh_2.custom_m0
        self.g_x_2 = mesh_2.build_terminal_cost()

        self.V_goal_1 = np.zeros((self.Nt + 1, self.Nx, self.Ny), dtype=np.float64)
        self.V_goal_2 = np.zeros((self.Nt + 1, self.Nx, self.Ny), dtype=np.float64)

        # State trajectories
        self.M1, self.M2 = np.zeros((self.Nt + 1, self.Nx, self.Ny)), np.zeros((self.Nt + 1, self.Nx, self.Ny))
        self.U1, self.U2 = np.zeros((self.Nt + 1, self.Nx, self.Ny)), np.zeros((self.Nt + 1, self.Nx, self.Ny))
        
        self.M1[0], self.M2[0] = self.m0_1, self.m0_2
        self.U1[self.Nt], self.U2[self.Nt] = self.g_x_1, self.g_x_2

    def solve_forward_FP(self, U_trajectory, m0, M_other_trajectory, gamma_mortality):
        m = np.zeros((self.Nt + 1, self.Nx, self.Ny))
        m[0] = m0
        N_total = self.Nx * self.Ny

        for k in range(1, self.Nt + 1):
            rows, cols, vals, b = compute_FP_matrix_entries_2Pop(
                m[k - 1], M_other_trajectory[k-1], U_trajectory[k - 1], self.omask,
                self.Nx, self.Ny, self.Dx, self.Dy, self.Dt, gamma_mortality
            )
            A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()
            mtmp = sparse.linalg.spsolve(A, b)
            m[k] = mtmp.reshape((self.Nx, self.Ny))
        return m

    def get_u_onestep_newton(self, Uk_n, Unew_np1, Unew_n_tmp, Mk_np1, Mk_other, V_goal_k, pop, gamma_mortality):
        N_total = self.Nx * self.Ny
        FnU_flat = getFnU_2D_2Pop(Unew_np1, Unew_n_tmp, Mk_np1, Mk_other, V_goal_k, self.omask, 
                                  self.Nx, self.Ny, self.Dx, self.Dy, self.Dt, pop, gamma_mortality).flatten()

        rows, cols, vals = compute_HJB_matrix_entries(
            Unew_n_tmp, Mk_np1, Mk_other, self.omask, self.Nx, self.Ny, self.Dx, self.Dy, self.Dt, gamma_mortality
        )
        A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()
        b = A.dot(Unew_n_tmp.flatten()) - FnU_flat

        for i in range(self.Nx):
            for j in range(self.Ny):
                if self.omask[i, j] == 0:
                    b[i * self.Ny + j] = -500.0

        utmp = sparse.linalg.spsolve(A, b)
        return utmp.reshape((self.Nx, self.Ny))

    def solve_hjb_single_timestep(self, Uk_n, Unew_np1, Mk_np1, Mk_other_np1, V_goal_k, pop, gamma_mortality):
        Unew_n = np.copy(Unew_np1)
        for _ in range(5):
            Unres = self.get_u_onestep_newton(Uk_n, Unew_np1, Unew_n, Mk_np1, Mk_other_np1, V_goal_k, pop, gamma_mortality)
            l2err = np.linalg.norm(Unew_n.flatten() - Unres.flatten()) * np.sqrt(self.Dx * self.Dy)
            Unew_n = np.copy(Unres)
            if l2err < 1e-6:
                break
        return Unew_n

    def solve_backward_HJB(self, M_trajectory, M_trajectory_other, U_temp, g_x, V_goal, pop, gamma_mortality):
        u = np.zeros_like(U_temp)
        u[self.Nt] = g_x
        for k in range(self.Nt - 1, -1, -1):
            u[k] = self.solve_hjb_single_timestep(
                U_temp[k], u[k + 1], M_trajectory[k + 1], M_trajectory_other[k + 1], V_goal[k], pop, gamma_mortality
            )
        return u

    def run_picard_system(self, max_iters: int = 20, tolerance: float = 1e-5):
        space_time_factor = np.sqrt(self.Dx * self.Dy * self.Dt)
        
        for iiter in range(1, max_iters + 1):
            start_time = time.time()
            print(f"\n>>> Macro Picard Loop Execution: {iiter} / {max_iters}")

            # Backward HJB solver calls with respective mortality rates
            U1_temp = self.solve_backward_HJB(self.M1, self.M2, self.U1, self.g_x_1, self.V_goal_1, pop=1, gamma_mortality=self.gamma_mortality_1)
            U2_temp = self.solve_backward_HJB(self.M2, self.M1, self.U2, self.g_x_2, self.V_goal_2, pop=2, gamma_mortality=self.gamma_mortality_2)
            
            U1_new = self.thetaUM * U1_temp + (1.0 - self.thetaUM) * self.U1
            U2_new = self.thetaUM * U2_temp + (1.0 - self.thetaUM) * self.U2

            # Forward FP solver calls with mortality mass loss
            M1_temp = self.solve_forward_FP(U1_new, self.m0_1, self.M2, gamma_mortality=self.gamma_mortality_1) 
            M2_temp = self.solve_forward_FP(U2_new, self.m0_2, self.M1, gamma_mortality=self.gamma_mortality_2)
            
            M1_new = self.thetaUM * M1_temp + (1.0 - self.thetaUM) * self.M1
            M2_new = self.thetaUM * M2_temp + (1.0 - self.thetaUM) * self.M2
            
            u_err = max(np.linalg.norm(U1_new - self.U1), np.linalg.norm(U2_new - self.U2)) * space_time_factor
            m_err = max(np.linalg.norm(M1_new - self.M1), np.linalg.norm(M2_new - self.M2)) * space_time_factor

            self.U1, self.M1 = np.copy(U1_new), np.copy(M1_new)
            self.U2, self.M2 = np.copy(U2_new), np.copy(M2_new)

            # Mass tracking over time
            total_mass_1 = np.sum(self.M1[-1]) * self.Dx * self.Dy
            total_mass_2 = np.sum(self.M2[-1]) * self.Dx * self.Dy

            print(f"    Iter {iiter} | u_err: {u_err:.6f}, m_err: {m_err:.6f} | Pop1 End Mass: {total_mass_1:.2f}, Pop2 End Mass: {total_mass_2:.2f} | Execution: {time.time() - start_time:.2f}s")

            if u_err < tolerance and m_err < tolerance:
                print(">>> System converged.")
                break
        return self.U1, self.M1, self.U2, self.M2

# %% Visualization Plotter
class MFGPlotter:
    def __init__(self, pde_mesh_data_1, pde_mesh_data_2, solver_instance):
        self.Lx = pde_mesh_data_1.Lx
        self.Ly = pde_mesh_data_1.Ly
        self.Dt = solver_instance.Dt
        self.Nt = solver_instance.Nt
        
        self.M1 = solver_instance.M1
        self.M2 = solver_instance.M2
        self.U1 = solver_instance.U1
        self.U2 = solver_instance.U2
        
        self.wall_mask = (solver_instance.omask == 0)
        self.extent = [0, self.Lx, 0, self.Ly]

        self.goals_1 = pde_mesh_data_1.get_goals()
        self.goals_2 = pde_mesh_data_2.get_goals()

    def _draw_goals(self, ax):
        """Helper to draw bold goal markers on any given axis."""
        if self.goals_1:
            gxs, gys = zip(*self.goals_1)
            ax.scatter(gxs, gys, color='#ff2222', marker='X', s=70,
                       edgecolor='white', linewidth=1.2, label='Pop 1 Goals', zorder=5)
        if self.goals_2:
            gxs, gys = zip(*self.goals_2)
            ax.scatter(gxs, gys, color='#2288ff', marker='X', s=70,
                       edgecolor='white', linewidth=1.2, label='Pop 2 Goals', zorder=5)

    def _get_spatial_frame(self, data_array, t_index):
        shape = data_array.shape
        Nx, Ny = self.wall_mask.shape
        if shape == (self.Nt + 1, Nx, Ny):
            return data_array[t_index, :, :]
        elif shape == (Nx, Ny, self.Nt + 1):
            return data_array[:, :, t_index]
        else:
            return data_array[t_index, :, :]

    def _build_combined_rgb(self, m1_frame, m2_frame, m1_max, m2_max):
        safe_m1 = np.clip(m1_frame.T / m1_max, 0.0, 1.0) if m1_max > 0 else np.zeros_like(m1_frame.T)
        safe_m2 = np.clip(m2_frame.T / m2_max, 0.0, 1.0) if m2_max > 0 else np.zeros_like(m2_frame.T)
        
        alpha1 = safe_m1 ** 0.4
        alpha2 = safe_m2 ** 0.4
        
        Ny, Nx = alpha1.shape
        rgb = np.ones((Ny, Nx, 3)) * 0.95
        
        rgb[:, :, 1] -= alpha1 * 0.95
        rgb[:, :, 2] -= alpha1 * 0.95
        
        rgb[:, :, 0] -= alpha2 * 0.95
        rgb[:, :, 1] -= alpha2 * 0.95
        
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb[self.wall_mask.T] = [0.1725, 0.2431, 0.3137]
        return rgb

    def plot_snapshots(self):
        """Shows a 3-panel timeline (Start, Mid, End) with marked goals."""
        t0, t_mid, t_end = 0, self.Nt // 2, self.Nt
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        m1_max = np.max(self.M1) if np.max(self.M1) > 0 else 1.0
        m2_max = np.max(self.M2) if np.max(self.M2) > 0 else 1.0

        def _style_snapshot_axis(ax, title, rgb_img):
            ax.set_facecolor('#2c3e50')
            ax.set_title(title)
            ax.set_xlabel("X (meters)")
            ax.set_ylabel("Y (meters)")
            ax.imshow(rgb_img, origin='lower', extent=self.extent, interpolation='nearest')
            self._draw_goals(ax)

        rgb0 = self._build_combined_rgb(self._get_spatial_frame(self.M1, t0),
                                       self._get_spatial_frame(self.M2, t0), m1_max, m2_max)
        rgbMid = self._build_combined_rgb(self._get_spatial_frame(self.M1, t_mid),
                                         self._get_spatial_frame(self.M2, t_mid), m1_max, m2_max)
        rgbEnd = self._build_combined_rgb(self._get_spatial_frame(self.M1, t_end),
                                         self._get_spatial_frame(self.M2, t_end), m1_max, m2_max)

        _style_snapshot_axis(axes[0], "Start ($t=0$)", rgb0)
        _style_snapshot_axis(axes[1], f"Midpoint ($t={self.Dt * t_mid:.1f}s$)", rgbMid)
        _style_snapshot_axis(axes[2], f"End ($t={self.Dt * t_end:.1f}s$)", rgbEnd)

        axes[0].legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def save_density_frames(self, output_dir="mfg_simulation_output_b"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"Exporting animation PNG frames to './{output_dir}'...")

        m1_max = np.max(self.M1) if np.max(self.M1) > 0 else 1.0
        m2_max = np.max(self.M2) if np.max(self.M2) > 0 else 1.0
        
        for k in range(self.Nt + 1):
            fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
            ax.set_facecolor('#2c3e50')
            
            m1_frame = self._get_spatial_frame(self.M1, k)
            m2_frame = self._get_spatial_frame(self.M2, k)
            rgb_img = self._build_combined_rgb(m1_frame, m2_frame, m1_max, m2_max)

            ax.imshow(rgb_img, origin='lower', extent=self.extent, interpolation='nearest')
            self._draw_goals(ax)
            
            ax.set_title(f"Pop 1 (Red) & Pop 2 (Blue) - Time: {k * self.Dt:.2f}s")
            ax.set_xlabel("X (meters)")
            ax.set_ylabel("Y (meters)")
            
            fig.savefig(f"{output_dir}/frame_{k:03d}.png", dpi=150)
            plt.close(fig)

    def save_mp4(self, filename="mfg_simulation_b.mp4", fps=15):
        """Compiles the simulation directly into an MP4 file with goal markers."""
        print(f"Exporting MP4 video directly to '{filename}'...")
        
        fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
        ax.set_facecolor('#2c3e50') 
        
        m1_max = np.max(self.M1) if np.max(self.M1) > 0 else 1.0
        m2_max = np.max(self.M2) if np.max(self.M2) > 0 else 1.0
        
        rgb_img = self._build_combined_rgb(self._get_spatial_frame(self.M1, 0),
                                          self._get_spatial_frame(self.M2, 0), m1_max, m2_max)
        
        im = ax.imshow(rgb_img, origin='lower', extent=self.extent, interpolation='nearest')
        self._draw_goals(ax)
        ax.legend(loc='upper right', fontsize='small')
        
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        title = ax.set_title("Pop 1 (Red) & Pop 2 (Blue) - Time: 0.00s")

        def update(k):
            m1_frame = self._get_spatial_frame(self.M1, k)
            m2_frame = self._get_spatial_frame(self.M2, k)
            new_rgb = self._build_combined_rgb(m1_frame, m2_frame, m1_max, m2_max)
            
            im.set_data(new_rgb)
            title.set_text(f"Pop 1 (Red) & Pop 2 (Blue) - Time: {k * self.Dt:.2f}s")
            return [im, title]

        ani = animation.FuncAnimation(fig, update, frames=self.Nt + 1, blit=True)
        
        try:
            ani.save(filename, writer='ffmpeg', fps=fps, dpi=150)
            print(f"Successfully exported video to '{filename}'.")
        except Exception as e:
            print(f"Could not save MP4 via FFmpeg. Fallback to 'save_density_frames()'.")
            print(f"Error details: {e}")
        finally:
            plt.close(fig)


MortalityPlotter = MFGPlotter

# %% Main Execution
if __name__ == "__main__":
    # Domain setup
    mesh_pop1 = MAP2PDE(Lx=768.0, Ly=768.0, Nx=60, Ny=60)
    mesh_pop2 = MAP2PDE(Lx=768.0, Ly=768.0, Nx=60, Ny=60)

    # Initial overlapping spatial distributions (to trigger combat/mortality)
    mesh_pop1.set_custom_initial_blobs([(350.0, 380.0, 50.0, 8.0)])
    mesh_pop2.set_custom_initial_blobs([(410.0, 380.0, 50.0, 8.0)])

    mesh_pop1.set_custom_goals([(650.0, 380.0)])
    mesh_pop2.set_custom_goals([(110.0, 380.0)])

    # Initialize solver with inter-population mortality rates
    mfg_solver = MFG2PopMortalitySolver(
        mesh_1=mesh_pop1,
        mesh_2=mesh_pop2,
        T=10.0,
        Nt=100,
        thetaUM=0.1,
        gamma_mortality_1=0,  # Mortality rate for Pop 1 when encountering Pop 2
        gamma_mortality_2=0.4   # Mortality rate for Pop 2 when encountering Pop 1
    )

    # Execute Picard Iteration
    U1, M1, U2, M2 = mfg_solver.run_picard_system(max_iters=15)

    # Visualize results with the same plotting API as MAP2PDE2POP.py
    plotter = MFGPlotter(mesh_pop1, mesh_pop2, mfg_solver)
    plotter.plot_snapshots()
    plotter.save_mp4(filename="mfg_simulation_test3.mp4", fps=30)
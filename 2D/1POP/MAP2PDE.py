#%%
import numpy as np
import time
import scipy as sc
import scipy.sparse as sparse
import scipy.sparse.linalg
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import time
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from numba import jit, prange
import numba_scipy
import os as os

# %%
@jit(nopython=True, cache=True)
def ppart(x):
    return np.maximum(x, 0.0)

@jit(nopython=True, cache=True)
def npart(x):
    return -np.minimum(x, 0.0)

@jit(nopython=True, cache=True)
def compute_FP_matrix_entries(m_prev, ukm1, omask_arr, Nx, Ny, Dx, Dy, Dt):
    """
    Function that creates the matrix used for the solving the Fokker-Planck equation in 2D.    
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

            # Boundary handling: Treat obstacles as dead space
            if omask_arr[i, j] == 0:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, 1.0
                entry_idx += 1
                b[ind] = 0.0
                continue

            # Check neighbors with reflective Zero-Neumann bounds
            ip1 = i + 1 if (i < Nx - 1 and omask_arr[i + 1, j] == 1) else i
            im1 = i - 1 if (i > 0      and omask_arr[i - 1, j] == 1) else i
            jp1 = j + 1 if (j < Ny - 1 and omask_arr[i, j + 1] == 1) else j
            jm1 = j - 1 if (j > 0      and omask_arr[i, j - 1] == 1) else j

            p1 = (ukm1[ip1, j] - ukm1[i, j]) / Dx
            p2 = (ukm1[i, j] - ukm1[im1, j]) / Dx
            p3 = (ukm1[i, jp1] - ukm1[i, j]) / Dy
            p4 = (ukm1[i, j] - ukm1[i, jm1]) / Dy

            diag_val = 1.0 / Dt
            
            if ip1 != i: diag_val += 0.05 / (Dx ** 2)
            if im1 != i: diag_val += 0.05 / (Dx ** 2)
            if jp1 != j: diag_val += 0.05 / (Dy ** 2)
            if jm1 != j: diag_val += 0.05 / (Dy ** 2)

            c_h_curr = 16.0 / ((1.0 + m_prev[i, j]) ** 0.75)
            diag_val += c_h_curr * (ppart(p1) / Dx + npart(p2) / Dx + ppart(p3) / Dy + npart(p4) / Dy)

            rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, diag_val
            entry_idx += 1

            # Off-diagonal Diffusion elements
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

            # Off-diagonal Non-linear Transport elements
            if i > 0 and omask_arr[i - 1, j] == 1:
                c_h_left = 16.0 / ((1.0 + m_prev[i - 1, j]) ** 0.75)
                vals[entry_idx] = -c_h_left * ppart((ukm1[i, j] - ukm1[i - 1, j]) / Dx) / Dx
                rows[entry_idx], cols[entry_idx] = ind, ((i - 1) * Ny + j)
                entry_idx += 1
            if i < Nx - 1 and omask_arr[i + 1, j] == 1:
                c_h_right = 16.0 / ((1.0 + m_prev[i + 1, j]) ** 0.75)
                vals[entry_idx] = -c_h_right * npart((ukm1[i + 1, j] - ukm1[i, j]) / Dx) / Dx
                rows[entry_idx], cols[entry_idx] = ind, ((i + 1) * Ny + j)
                entry_idx += 1
            if j > 0 and omask_arr[i, j - 1] == 1:
                c_h_bottom = 16.0 / ((1.0 + m_prev[i, j - 1]) ** 0.75)
                vals[entry_idx] = -c_h_bottom * ppart((ukm1[i, j] - ukm1[i, j - 1]) / Dy) / Dy
                rows[entry_idx], cols[entry_idx] = ind, (i * Ny + j - 1)
                entry_idx += 1
            if j < Ny - 1 and omask_arr[i, j + 1] == 1:
                c_h_top = 16.0 / ((1.0 + m_prev[i, j + 1]) ** 0.75)
                vals[entry_idx] = -c_h_top * npart((ukm1[i, j + 1] - ukm1[i, j]) / Dy) / Dy
                rows[entry_idx], cols[entry_idx] = ind, (i * Ny + j + 1)
                entry_idx += 1

    return rows[:entry_idx], cols[:entry_idx], vals[:entry_idx], b

@jit(nopython=True, cache=True)
def getFnU_2D(Ukp1_np1, Ukp1_n, Mk_np1, omask_arr, Nx, Ny, Dx, Dy, Dt):
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


            
            hamiltonian = -8 * (1 / (1 + Mk_np1[i, j]) ** 0.75) * (
                ppart(p1)**2 + npart(p2)**2 + ppart(p3)**2 + npart(p4)**2
            ) + 0.1

            FnU[i, j] = time_deriv + diffusion + hamiltonian
    return FnU

@jit(nopython=True, cache=True)
def compute_HJB_matrix_entries(Unew_n_tmp, Mk_np1, omask_arr, Nx, Ny, Dx, Dy, Dt):
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

            diag_val = 1.0 / Dt
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

            c_h = 16.0 / ((1.0 + Mk_np1[i, j]) ** 0.75)

            rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, diag_val
            entry_idx += 1

            # Diffusion Stencils
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

            # Hamiltonian Linearization Additions
            if ip1 != i:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, (ip1 * Ny + j), -c_h * ppart(p1) / Dx
                entry_idx += 1
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, c_h * ppart(p1) / Dx
                entry_idx += 1
            if im1 != i:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, (im1 * Ny + j), -c_h * npart(p2) / Dx
                entry_idx += 1
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, c_h * npart(p2) / Dx
                entry_idx += 1
            if jp1 != j:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, (i * Ny + jp1), -c_h * ppart(p3) / Dy
                entry_idx += 1
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, c_h * ppart(p3) / Dy
                entry_idx += 1
            if jm1 != j:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, (i * Ny + jm1), -c_h * npart(p4) / Dy
                entry_idx += 1
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, c_h * npart(p4) / Dy
                entry_idx += 1

    return rows[:entry_idx], cols[:entry_idx], vals[:entry_idx]
# %%
class MAP2PDE:
    """
    Translates MovingAI benchmark files into continuous spatial meshes 
    and initial condition arrays for Mean Field Game PDE solvers.
    """
    
    def __init__(self, map_filepath: str, scen_filepath: str, Lx: float = 1.0, Ly: float = 1.0, Nx: int = None, Ny: int = None):
        self.map_filepath = map_filepath
        self.scen_filepath = scen_filepath
        self.Lx = Lx
        self.Ly = Ly

        self.grid_shape = (0, 0)
        self.raw_boolean_grid = None
        self.raw_agents = []

        self.Nx = Nx
        self.Ny = Ny
        self.dx = 0.0
        self.dy = 0.0
        self.X = None
        self.Y = None

    def parse_files(self, num_agents: int = None):
        """Parses the .map and .scen files and stores the raw discrete data."""
        with open(self.map_filepath, 'r') as f:
            lines = f.readlines()
        
        height = int(lines[1].split()[1]) 
        width = int(lines[2].split()[1])
        self.grid_shape = (height, width)
        self.raw_boolean_grid = np.zeros((height, width), dtype=bool)
        
        #Extract the boolean grid from the map file
        for r, line in enumerate(lines[4:]):
            for c, char in enumerate(line.strip()):
                if char in ['.', 'G']:
                    self.raw_boolean_grid[r, c] = True  

        #Extract agents from the scenario file            
        with open(self.scen_filepath, 'r') as f:
            scen_lines = f.readlines()[1:] 
            
        for line in scen_lines:
            parts = line.strip().split()
            if len(parts) >= 9:
                x_start, y_start = int(parts[4]), int(parts[5])
                x_goal, y_goal = int(parts[6]), int(parts[7])
                
                self.raw_agents.append({
                    'start_x': x_start, 'start_y': y_start,
                    'goal_x': x_goal, 'goal_y': y_goal
                })
                
            if num_agents and len(self.raw_agents) >= num_agents:
                break

    def get_pde_obstacle_mask(self):
        """
        Determines the obstacle mask by checking which original MovingAI 
        map cell each continuous PDE mesh point (X, Y) falls into.
        """
        if self.X is None or self.Y is None:
            raise ValueError("Mesh needs initialization.")
            
        Nx, Ny = self.X.shape
        pde_mask = np.ones((Nx, Ny), dtype=np.int64)
        
        map_H, map_W = self.grid_shape
        map_dx = self.Lx / map_W
        map_dy = self.Ly / map_H
        
        for i in range(Nx):
            for j in range(Ny):
                x_val = self.X[i, j]
                y_val = self.Y[i, j]
                
                col_idx = int(x_val / map_dx)
                row_idx = int((self.Ly - y_val) / map_dy)
                
                col_idx = max(0, min(col_idx, map_W - 1))
                row_idx = max(0, min(row_idx, map_H - 1))
                
                if not self.raw_boolean_grid[row_idx, col_idx]:
                    pde_mask[i, j] = 0
                    
        return pde_mask
    
    def build_spatial_mesh(self):
        if self.raw_boolean_grid is None:
            raise ValueError("Must call parse_files() before building the mesh.")
            
        H, W = self.grid_shape
        
        if self.Nx is None: self.Nx = W
        if self.Ny is None: self.Ny = H
        
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        
        xSpace = np.linspace(self.dx / 2, self.Lx - self.dx / 2, self.Nx, endpoint=True)
        ySpace = np.linspace(self.dy / 2, self.Ly - self.dy / 2, self.Ny,endpoint=True)

        self.X, self.Y = np.meshgrid(xSpace, ySpace, indexing='ij')
        return self.X, self.Y

    def build_initial_density(self, sigma_multiplier: float = 1.5):
        if self.X is None or self.Y is None:
            raise ValueError("Must call build_spatial_mesh() before building density.")
            
        m_0 = np.zeros_like(self.X)
        sigma = sigma_multiplier * max(self.dx, self.dy)
        
        
        H, W = self.grid_shape
        map_dx = self.Lx / W
        map_dy = self.Ly / H
        
        for agent in self.raw_agents:
            start_x = (agent['start_x'] + 0.5) * map_dx
            start_y = self.Ly - (agent['start_y'] + 0.5) * map_dy
            
            m_0 += np.exp(-((self.X - start_x)**2 + (self.Y - start_y)**2) / (2 * sigma**2))
            
        return m_0

    def build_terminal_cost(self, penalty_scale: float = 1.0):
        if self.X is None or self.Y is None:
            raise ValueError("Mesh needs initialization.")
        Nx, Ny = self.X.shape
        g = np.zeros((Nx, Ny), dtype=np.float64)
        
        H, W = self.grid_shape
        map_dx = self.Lx / W
        map_dy = self.Ly / H
        
        goal_positions = []
        for agent in self.raw_agents:
            goal_x = (agent['goal_x'] + 0.5) * map_dx
            goal_y = self.Ly - (agent['goal_y'] + 0.5) * map_dy
            goal_positions.append((goal_x, goal_y))
            
        if not goal_positions:
            return g
            
        for i in range(Nx):
            for j in range(Ny):
                x_val, y_val = self.X[i, j], self.Y[i, j]
                min_dist = min(np.sqrt((x_val - gx)**2 + (y_val - gy)**2) for gx, gy in goal_positions)
                g[i, j] = -penalty_scale * (min_dist ** 2)
        return g
        
# %%
class MFGSolver:
    def __init__(self, pde_mesh_data, T: float = 5.0, Nt: int = 100, thetaUM: float = 0.1):
        self.Nx, self.Ny = pde_mesh_data.X.shape
        self.Dx = pde_mesh_data.dx  
        self.Dy = pde_mesh_data.dy  
        self.Dt = T / Nt
        self.Nt = Nt
        self.thetaUM = thetaUM
        
        self.omask = pde_mesh_data.get_pde_obstacle_mask()
        self.m0 = pde_mesh_data.build_initial_density()
        self.g_x = pde_mesh_data.build_terminal_cost()
        
        # State histories
        self.M = np.zeros((self.Nt + 1, self.Nx, self.Ny))
        self.U = np.zeros((self.Nt + 1, self.Nx, self.Ny))
        
        self.M[0] = self.m0
        self.U[self.Nt] = self.g_x

    def solve_forward_FP(self, U_trajectory):
        m = np.zeros_like(self.M)
        m[0] = self.m0
        N_total = self.Nx * self.Ny

        for k in range(1, self.Nt + 1):
            rows, cols, vals, b = compute_FP_matrix_entries(
                m[k - 1], U_trajectory[k - 1], self.omask,
                self.Nx, self.Ny, self.Dx, self.Dy, self.Dt
            )
            A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()
            mtmp = sparse.linalg.spsolve(A, b)
            m[k] = mtmp.reshape((self.Nx, self.Ny))
        return m

    def get_u_onestep_newton(self, Uk_n, Unew_np1, Unew_n_tmp, Mk_np1):
        N_total = self.Nx * self.Ny
        FnU_flat = getFnU_2D(Unew_np1, Unew_n_tmp, Mk_np1, self.omask, 
                             self.Nx, self.Ny, self.Dx, self.Dy, self.Dt).flatten()

        rows, cols, vals = compute_HJB_matrix_entries(
            Unew_n_tmp, Mk_np1, self.omask, self.Nx, self.Ny, self.Dx, self.Dy, self.Dt
        )
        A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()
        b = A.dot(Unew_n_tmp.flatten()) - FnU_flat

        for i in range(self.Nx):
            for j in range(self.Ny):
                if self.omask[i, j] == 0:
                    b[i * self.Ny + j] = -500.0

        utmp = sparse.linalg.spsolve(A, b)
        return utmp.reshape((self.Nx, self.Ny))

    def solve_hjb_single_timestep(self, Uk_n, Unew_np1, Mk_np1):
        Unew_n = np.copy(Unew_np1)
        for _ in range(30):
            Unres = self.get_u_onestep_newton(Uk_n, Unew_np1, Unew_n, Mk_np1)
            l2err = np.linalg.norm(Unew_n.flatten() - Unres.flatten()) * np.sqrt(self.Dx * self.Dy)
            Unew_n = np.copy(Unres)
            if l2err < 1e-6:
                break
        return Unew_n

    def solve_backward_HJB(self, M_trajectory):
        u = np.zeros_like(self.U)
        u[self.Nt] = self.g_x
        for k in range(self.Nt - 1, -1, -1):
            u[k] = self.solve_hjb_single_timestep(self.U[k], u[k + 1], M_trajectory[k + 1])
        return u

    def run_picard_system(self, max_iters: int = 25, tolerance: float = 1e-5):
        space_time_factor = np.sqrt(self.Dx * self.Dy * self.Dt)
        
        for iiter in range(1, max_iters + 1):
            start_time = time.time()
            print(f"\n>>> Macro Picard Loop Execution: {iiter} / {max_iters}")

            # Backward HJB update step
            U_temp = self.solve_backward_HJB(self.M)
            U_new = self.thetaUM * U_temp + (1.0 - self.thetaUM) * self.U

            # Forward FP update step
            M_temp = self.solve_forward_FP(U_new)
            M_new = self.thetaUM * M_temp + (1.0 - self.thetaUM) * self.M

            # Convergence Check
            u_err = np.linalg.norm(U_new - self.U) * space_time_factor
            m_err = np.linalg.norm(M_new - self.M) * space_time_factor
            print(f"    u_residual: {u_err:.6f} | m_residual: {m_err:.6f} | Time: {time.time() - start_time:.2f}s")

            self.U = np.copy(U_new)
            self.M = np.copy(M_new)

            if u_err < tolerance and m_err < tolerance:
                print(f"\n[Success] Converged at loop index {iiter}!")
                break
        return self.U, self.M
    

# %%
# Made with the help of AI

class MFGPlotter:
    def __init__(self, pde_mesh_data, solver_instance):
        """
        Bases all spatial dimensions, extents, and obstacle masks 
        directly on your parsed mesh and solver state.
        """
        self.Lx = pde_mesh_data.Lx
        self.Ly = pde_mesh_data.Ly
        self.Dt = solver_instance.Dt
        self.Nt = solver_instance.Nt
        
        # Pull state matrices
        self.M = solver_instance.M
        self.U = solver_instance.U
        
        # Create the matplotlib mask (True where walls exist)
        self.wall_mask = (solver_instance.omask == 0)
        self.extent = [0, self.Lx, 0, self.Ly]

        H, W = pde_mesh_data.grid_shape
        map_dx = self.Lx / W
        map_dy = self.Ly / H
        
        self.goals = []
        for agent in pde_mesh_data.raw_agents:
            gx = (agent['goal_x'] + 0.5) * map_dx
            gy = self.Ly - (agent['goal_y'] + 0.5) * map_dy
            self.goals.append((gx, gy))

    def _get_spatial_frame(self, data_array, t_index):
        """Extracts the (Nx, Ny) spatial slice matching the time index."""
        shape = data_array.shape
        Nx, Ny = self.wall_mask.shape
        if shape == (self.Nt + 1, Nx, Ny):
            return data_array[t_index, :, :]
        elif shape == (Nx, Ny, self.Nt + 1):
            return data_array[:, :, t_index]
        else:
            return data_array[t_index, :, :]

    def plot_snapshots(self):
        """Generates a 2x2 summary dashboard of the MFG simulation with marked goals."""
        t0, t_mid, t_end = 0, self.Nt // 2, self.Nt
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Helper internal function to avoid repetitive code
        def _style_snapshot_axis(ax, title, im, show_goals=True):
            ax.set_facecolor('#2c3e50')  
            ax.set_title(title)
            ax.set_xlabel("X (meters)")
            ax.set_ylabel("Y (meters)")
            fig.colorbar(im, ax=ax, label='Value')
            # Plot the target goals as bright cyan 'X' markers
            if show_goals and self.goals:
                gxs, gys = zip(*self.goals)
                ax.scatter(gxs, gys, color='#00f2fe', marker='X', s=50, 
                           edgecolor='black', linewidth=0.5, label='Goals', zorder=10)

        # Panel 1: Initial Density
        m0_frame = self._get_spatial_frame(self.M, t0)
        m0_masked = np.ma.masked_where(self.wall_mask, m0_frame)
        im1 = axes[0, 0].imshow(m0_masked.T, origin='lower', extent=self.extent, cmap='YlOrRd')
        _style_snapshot_axis(axes[0, 0], "Initial Density $M$ ($t=0$)", im1)
        
        # Panel 2: Midpoint Density
        mmid_frame = self._get_spatial_frame(self.M, t_mid)
        mmid_masked = np.ma.masked_where(self.wall_mask, mmid_frame)
        im2 = axes[0, 1].imshow(mmid_masked.T, origin='lower', extent=self.extent, cmap='YlOrRd')
        _style_snapshot_axis(axes[0, 1], f"Midpoint Density $M$ ($t={self.Dt * t_mid:.1f}s$)", im2)

        # Panel 3: Final Density
        mend_frame = self._get_spatial_frame(self.M, t_end)
        mend_masked = np.ma.masked_where(self.wall_mask, mend_frame)
        im3 = axes[1, 0].imshow(mend_masked.T, origin='lower', extent=self.extent, cmap='YlOrRd')
        _style_snapshot_axis(axes[1, 0], f"Final Density $M$ ($t={self.Dt * t_end:.1f}s$)", im3)

        # Panel 4: Initial Value Function (Pathfinding Map)
        u0_frame = self._get_spatial_frame(self.U, t0)
        u0_masked = np.ma.masked_where(self.wall_mask, u0_frame)
        im4 = axes[1, 1].imshow(u0_masked.T, origin='lower', extent=self.extent, cmap='viridis_r')
        _style_snapshot_axis(axes[1, 1], "Value Function $U$ ($t=0$)", im4)

        axes[0, 0].legend(loc='upper right') # Show legend once on the first panel
        plt.tight_layout()
        plt.show()

    def save_density_frames(self, output_dir="mfg_simulation_output"):
        """Saves individual image frames of the crowd density over time with a stable color scale."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        print(f"Exporting animation frames to './{output_dir}'...")

        m0_frame = self._get_spatial_frame(self.M, 0)
        m0_masked = np.ma.masked_where(self.wall_mask, m0_frame)
        stable_vmax = np.max(m0_masked) if np.max(m0_masked) > 0 else 1.0
        
        for k in range(self.Nt + 1):
            fig, ax = plt.subplots(figsize=(6, 5), layout='constrained') 
            
            mk_frame = self._get_spatial_frame(self.M, k)
            m_masked = np.ma.masked_where(self.wall_mask, mk_frame)
            
            im = ax.imshow(m_masked.T, origin='lower', extent=self.extent, 
                           cmap='YlOrRd', vmin=0, vmax=stable_vmax)

            if self.goals:
                gxs, gys = zip(*self.goals)
                ax.scatter(gxs, gys, color='#00f2fe', marker='X', s=45, 
                           edgecolor='black', linewidth=0.5, zorder=10)

            ax.set_facecolor('#2c3e50')
            ax.set_title(f"Crowd Density - Time: {k * self.Dt:.2f}s")
            ax.set_xlabel("X (meters)")
            ax.set_ylabel("Y (meters)")
            fig.colorbar(im, ax=ax, label='Density')
            
            # FIX 2: Remove bbox_inches='tight'. 
            # A 6x5 inch figure at 150 DPI will now always be exactly 900x750 pixels (both even numbers).
            fig.savefig(f"{output_dir}/frame_{k:03d}.png", dpi=150)
            plt.close(fig)
            
        print(f"Successfully exported {self.Nt + 1} frames to folder '{output_dir}'.")
# %%
if __name__ == "__main__":
    # Specify the target files and room sizing constants
    MAP_FILE ="Maps/AcrosstheCape.map"
    SCEN_FILE = "Scenarios/AcrosstheCape_1g.map.scen" #All densities have same goal.
    
    #This changes depending on the map
    ROOM_WIDTH = 768.0   # Physical meters (Lx) 
    ROOM_HEIGHT = 768.0  # Physical meters (Ly)
    
    # Run Parser & Initialize Spatial Objects
    print("Initializing environment layout...")
    pde_mesh = MAP2PDE(MAP_FILE, SCEN_FILE, Lx=ROOM_WIDTH, Ly=ROOM_HEIGHT, Nx=200, Ny=200)
    
    # Read first agents from the benchmark file files
    pde_mesh.parse_files(num_agents=50) 
    pde_mesh.build_spatial_mesh()
    
    # run simulation
    print("Configuring MFG system matrices...")
    mfg_solver = MFGSolver(pde_mesh_data=pde_mesh, T=3.0, Nt=100, thetaUM=0.1)
    
    print("Launching numerical solver loop...")
    U_final, M_final = mfg_solver.run_picard_system(max_iters=10)
    
    #%%  Visualization 

    # Initialize the plotter object using your completed simulation state
    plotter = MFGPlotter(pde_mesh_data=pde_mesh, solver_instance=mfg_solver)
    
    # Displays the 4-panel summary dashboard
    plotter.plot_snapshots()
    
    # Exports individual PNG images of the crowd's trajectory frame-by-frame
    plotter.save_density_frames(output_dir="mfg_simulation_1POP")

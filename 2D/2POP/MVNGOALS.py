# %%
import numpy as np
import time
import scipy as sc
import scipy.sparse as sparse
import scipy.sparse.linalg
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
from numba import jit, prange
import os as os

# %%
@jit(nopython=True, cache=True)
def ppart(x):
    return np.maximum(x, 0.0)

@jit(nopython=True, cache=True)
def npart(x):
    return -np.minimum(x, 0.0)

@jit(nopython=True, cache=True)
def compute_FP_matrix_entries_2Pop(m_prev, m_other, ukm1, omask_arr, Nx, Ny, Dx, Dy, Dt):
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
            
            diag_val = 1.0 / Dt
            
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
                c_h_right= 2.0 / (1.0 + m_prev[i + 1, j] + 5.0 * m_other[i + 1, j])
                vals[entry_idx] = -c_h_right * ppart((ukm1[i + 1, j] - ukm1[i, j]) / Dx) / Dx 
                rows[entry_idx], cols[entry_idx] = ind, ((i + 1) * Ny + j)
                entry_idx += 1
            if j > 0 and omask_arr[i, j - 1] == 1:
                c_h_bottom= 2.0 / (1.0 + m_prev[i, j-1] + 5.0 * m_other[i, j-1])
                vals[entry_idx] = -c_h_bottom * npart((ukm1[i, j] - ukm1[i, j - 1]) / Dy) / Dy 
                rows[entry_idx], cols[entry_idx] = ind, (i * Ny + j - 1)
                entry_idx += 1
            if j < Ny - 1 and omask_arr[i, j + 1] == 1:
                c_h_top= 2.0 / (1.0 + m_prev[i, j+1] + 5.0 * m_other[i, j+1])
                vals[entry_idx] = -c_h_top * ppart((ukm1[i, j + 1] - ukm1[i, j]) / Dy) / Dy
                rows[entry_idx], cols[entry_idx] = ind, (i * Ny + j + 1)
                entry_idx += 1
            
    return rows[:entry_idx], cols[:entry_idx], vals[:entry_idx], b

@jit(nopython=True, cache=True)
def getFnU_2D_2Pop(Ukp1_np1, Ukp1_n, Mk_np1, Mk_other, V_goal_k, omask_arr, Nx, Ny, Dx, Dy, Dt, pop=1):
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

            hamiltonian = (1/(1+Mk_np1[i,j]+5*Mk_other[i,j]))*(
                        npart(p1)**2 + ppart(p2)**2 + npart(p3)**2 + ppart(p4)**2)
            
            if pop == 1:
                interaction_cost = +1500.0 * Mk_other[i, j]+ppart(Mk_np1[i,j]+Mk_other[i,j]-4)
            else:
                interaction_cost = -5000.0 * Mk_other[i, j]+ppart(Mk_np1[i,j]+Mk_other[i,j]-4)

            # This residual uses -u_t + H = running_cost, so the cost enters
            # with a negative sign on the left-hand side.
            goal_cost = V_goal_k[i, j]
            FnU[i, j] = time_deriv + diffusion + hamiltonian + interaction_cost - goal_cost
    return FnU

@jit(nopython=True, cache=True)
def compute_HJB_matrix_entries(Unew_n_tmp, Mk_np1,Mk_other, omask_arr, Nx, Ny, Dx, Dy, Dt):
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

# %%
class MAP2PDE:
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

    def parse_files(self, num_agents: int = None, start_idx: int = 0):
        with open(self.map_filepath, 'r') as f:
            lines = f.readlines()
        
        height = int(lines[1].split()[1]) 
        width = int(lines[2].split()[1])
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
                x_start, y_start = int(parts[4]), int(parts[5])
                x_goal, y_goal = int(parts[6]), int(parts[7])
                
                self.raw_agents.append({
                    'start_x': x_start, 'start_y': y_start,
                    'goal_x': x_goal, 'goal_y': y_goal
                })
                
            if num_agents and len(self.raw_agents) >= num_agents:
                break
    
    def set_custom_initial_blobs(self, blob_list, normalize_mass: bool = False):
        if self.X is None or self.Y is None:
            raise ValueError("Must call build_spatial_mesh() before setting custom initial density.")

        m_0 = np.zeros_like(self.X, dtype=np.float64)

        for blob in blob_list:
            if len(blob) == 3:
                cx, cy, sigma = blob
                amp = 1.0
            else:
                cx, cy, sigma, amp = blob
            m_0 += amp * np.exp(-((self.X - cx)**2 + (self.Y - cy)**2) / (2.0 * sigma**2))

        pde_mask = self.get_pde_obstacle_mask()
        m_0 *= pde_mask

        if normalize_mass and np.sum(m_0) > 0:
            m_0 /= (np.sum(m_0) * self.dx * self.dy)

        self.custom_m0 = m_0
        return m_0
    
    def get_pde_obstacle_mask(self):
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
        if hasattr(self, 'custom_m0') and self.custom_m0 is not None:
            return self.custom_m0

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
            
        total_mass = np.sum(m_0) * self.dx * self.dy
        if total_mass > 0:
            m_0 /= total_mass

        return m_0

# %%
class MFG2PopSolver:
    def __init__(self, pde_mesh_data_1, pde_mesh_data_2, T=5.0, Nt=100, thetaUM=0.1,
                 goal_config=None):
        self.Nx, self.Ny = pde_mesh_data_1.X.shape
        self.Dx = pde_mesh_data_1.dx
        self.Dy = pde_mesh_data_1.dy
        self.T = T
        self.Dt = T / Nt
        self.Nt = Nt
        self.time_grid = np.linspace(0.0, T, Nt + 1)
        self.thetaUM = thetaUM

        # Goals come exclusively from the caller; no population gets an implicit goal.
        self.goal_config = dict(goal_config) if goal_config is not None else {1: None, 2: None}

        self.omask = pde_mesh_data_1.get_pde_obstacle_mask()
        self.m0_1 = pde_mesh_data_1.build_initial_density(sigma_multiplier=3)
        self.m0_2 = pde_mesh_data_2.build_initial_density(sigma_multiplier=3)

        self.M1, self.M2 = np.zeros((self.Nt + 1, self.Nx, self.Ny)), np.zeros((self.Nt + 1, self.Nx, self.Ny))
        self.U1, self.U2 = np.zeros((self.Nt + 1, self.Nx, self.Ny)), np.zeros((self.Nt + 1, self.Nx, self.Ny))

        self.M1[0], self.M2[0] = self.m0_1, self.m0_2

        self.switch_time = T / 2.0
        self.goal_switch_step = int(self.switch_time / self.Dt)

        self.goal_trajectories = {
            1: self._build_goal_trajectory(self.goal_config.get(1)),
            2: self._build_goal_trajectory(self.goal_config.get(2)),
        }
        self.V_goal_1 = self._build_goal_potential_for_population(
            pde_mesh_data_1, self.goal_trajectories[1]
        )
        self.V_goal_2 = self._build_goal_potential_for_population(
            pde_mesh_data_2, self.goal_trajectories[2]
        )

        self.U1[self.Nt] = np.zeros((self.Nx, self.Ny))
        self.U2[self.Nt] = np.zeros((self.Nx, self.Ny))

    def _build_goal_trajectory(self, goal_spec):
        """Sample timed waypoints, or preserve the legacy midpoint-switch pair."""
        if goal_spec is None:
            return None

        if isinstance(goal_spec, dict):
            if "times" not in goal_spec or "positions" not in goal_spec:
                raise ValueError("Goal trajectories require 'times' and 'positions'.")
            waypoint_times = np.asarray(goal_spec["times"], dtype=np.float64)
            waypoint_positions = np.asarray(goal_spec["positions"], dtype=np.float64)
            if (waypoint_times.ndim != 1 or waypoint_times.size == 0
                    or waypoint_positions.shape != (waypoint_times.size, 2)):
                raise ValueError("Goal trajectory times and positions must have shapes (N,) and (N, 2).")
            if (not np.all(np.isfinite(waypoint_times))
                    or not np.all(np.isfinite(waypoint_positions))
                    or np.any(np.diff(waypoint_times) <= 0)):
                raise ValueError("Goal trajectory values must be finite and times strictly increasing.")

            return np.column_stack((
                np.interp(self.time_grid, waypoint_times, waypoint_positions[:, 0]),
                np.interp(self.time_grid, waypoint_times, waypoint_positions[:, 1]),
            ))

        goal_pair = np.asarray(goal_spec, dtype=np.float64)
        if goal_pair.shape != (2, 2):
            raise ValueError(
                "A goal must be None, a pair of points, or a dict with timed waypoints."
            )
        switch_step = max(0, min(int(self.switch_time / self.Dt), self.Nt))
        trajectory = np.empty((self.Nt + 1, 2), dtype=np.float64)
        trajectory[:switch_step] = goal_pair[0]
        trajectory[switch_step:] = goal_pair[1]
        return trajectory

    def _build_goal_potential_for_population(self, mesh_data, goal_trajectory, penalty_scale=15.0):
        V_goal = np.zeros((self.Nt + 1, self.Nx, self.Ny), dtype=np.float64)

        if goal_trajectory is None:
            return V_goal

        for k in range(self.Nt + 1):
            active_x, active_y = goal_trajectory[k]
            for i in range(self.Nx):
                for j in range(self.Ny):
                    x_val, y_val = mesh_data.X[i, j], mesh_data.Y[i, j]
                    dist = np.sqrt((x_val - active_x) ** 2 + (y_val - active_y) ** 2)
                    V_goal[k, i, j] = penalty_scale * dist
        return V_goal

    def solve_forward_FP(self, U_trajectory, m0, M_other_trajectory):
        m = np.zeros((self.Nt + 1, self.Nx, self.Ny))
        m[0] = m0
        N_total = self.Nx * self.Ny

        for k in range(1, self.Nt + 1):
            rows, cols, vals, b = compute_FP_matrix_entries_2Pop(
                m[k - 1], M_other_trajectory[k-1], U_trajectory[k - 1], self.omask,
                self.Nx, self.Ny, self.Dx, self.Dy, self.Dt
            )
            A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()
            mtmp = sparse.linalg.spsolve(A, b)
            m[k] = mtmp.reshape((self.Nx, self.Ny))
        return m

    def get_u_onestep_newton(self, Uk_n, Unew_np1, Unew_n_tmp, Mk_np1, Mk_other, V_goal_k, pop):
        N_total = self.Nx * self.Ny
        
        # Pass the spatial goal slice into the HJB solver
        FnU_flat = getFnU_2D_2Pop(Unew_np1, Unew_n_tmp, Mk_np1, Mk_other, V_goal_k, self.omask, 
                             self.Nx, self.Ny, self.Dx, self.Dy, self.Dt, pop).flatten()

        rows, cols, vals = compute_HJB_matrix_entries(
            Unew_n_tmp, Mk_np1, Mk_other, self.omask, self.Nx, self.Ny, self.Dx, self.Dy, self.Dt
        )
        A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()
        b = A.dot(Unew_n_tmp.flatten()) - FnU_flat

        for i in range(self.Nx):
            for j in range(self.Ny):
                if self.omask[i, j] == 0:
                    b[i * self.Ny + j] = -500.0

        utmp = sparse.linalg.spsolve(A, b)
        return utmp.reshape((self.Nx, self.Ny))

    def solve_hjb_single_timestep(self, Uk_n, Unew_np1, Mk_np1, Mk_other_np1, V_goal_k, pop):
        Unew_n = np.copy(Unew_np1)
        for _ in range(5):
            Unres = self.get_u_onestep_newton(Uk_n, Unew_np1, Unew_n, Mk_np1, Mk_other_np1, V_goal_k, pop)
            l2err = np.linalg.norm(Unew_n.flatten() - Unres.flatten()) * np.sqrt(self.Dx * self.Dy)
            Unew_n = np.copy(Unres)
            if l2err < 1e-6:
                break
        return Unew_n

    def solve_backward_HJB(self, M_trajectory, M_trajectory_other, U_temp, g_x, V_goal_trajectory, pop):
        u = np.zeros_like(U_temp)
        u[self.Nt] = g_x
        for k in range(self.Nt - 1, -1, -1):
            # Extract spatial potential slice V_goal_trajectory[k] for this exact timestep
            u[k] = self.solve_hjb_single_timestep(
                U_temp[k], u[k + 1], M_trajectory[k + 1], M_trajectory_other[k + 1], V_goal_trajectory[k], pop
            )
        return u

    def run_picard_system(self, max_iters: int = 25, tolerance: float = 1e-5):
        space_time_factor = np.sqrt(self.Dx * self.Dy * self.Dt)
        
        for iiter in range(1, max_iters + 1):
            start_time = time.time()
            print(f"\n>>> Macro Picard Loop Execution: {iiter} / {max_iters}")

            # Note: We pass self.U1[self.Nt] as the final cost (which is just an array of zeros)
            U1_temp = self.solve_backward_HJB(self.M1, self.M2, self.U1, self.U1[self.Nt], self.V_goal_1, pop=1)
            U2_temp = self.solve_backward_HJB(self.M2, self.M1, self.U2, self.U2[self.Nt], self.V_goal_2, pop=2)
            
            U1_new = self.thetaUM * U1_temp + (1.0 - self.thetaUM) * self.U1
            U2_new = self.thetaUM * U2_temp + (1.0 - self.thetaUM) * self.U2

            M1_temp = self.solve_forward_FP(U1_new, self.m0_1, self.M2) 
            M2_temp = self.solve_forward_FP(U2_new, self.m0_2, self.M1)
            
            M1_new = self.thetaUM * M1_temp + (1.0 - self.thetaUM) * self.M1
            M2_new = self.thetaUM * M2_temp + (1.0 - self.thetaUM) * self.M2
            
            u_err = max(np.linalg.norm(U1_new - self.U1), np.linalg.norm(U2_new - self.U2)) * space_time_factor
            m_err = max(np.linalg.norm(M1_new - self.M1), np.linalg.norm(M2_new - self.M2)) * space_time_factor

            self.U1, self.M1 = np.copy(U1_new), np.copy(M1_new) 
            self.U2, self.M2 = np.copy(U2_new), np.copy(M2_new) 

            if u_err < tolerance and m_err < tolerance:
                break
        return self.U1, self.M1, self.U2, self.M2


# %%
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

        self.goal_1_trajectory = solver_instance.goal_trajectories[1]
        self.goal_2_trajectory = solver_instance.goal_trajectories[2]

        self.wall_mask = (solver_instance.omask == 0)
        self.extent = [0, self.Lx, 0, self.Ly]

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

    def _active_goal(self, goal_trajectory, t_index):
        if goal_trajectory is None:
            return None
        return goal_trajectory[t_index]
    
    def plot_snapshots(self):
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

        for ax, t_idx in zip(axes, [t0, t_mid, t_end]):
            g1 = self._active_goal(self.goal_1_trajectory, t_idx)
            g2 = self._active_goal(self.goal_2_trajectory, t_idx)
            if g1 is not None:
                ax.scatter([g1[0]], [g1[1]], s=80, c='#ff5a5a',
                           edgecolors='black', linewidths=1.0, zorder=10)
            if g2 is not None:
                ax.scatter([g2[0]], [g2[1]], s=80, c='#5aa9ff',
                           edgecolors='black', linewidths=1.0, zorder=10)

        rgb0 = self._build_combined_rgb(self._get_spatial_frame(self.M1, t0),
                                        self._get_spatial_frame(self.M2, t0), m1_max, m2_max)
        rgbMid = self._build_combined_rgb(self._get_spatial_frame(self.M1, t_mid),
                                          self._get_spatial_frame(self.M2, t_mid), m1_max, m2_max)
        rgbEnd = self._build_combined_rgb(self._get_spatial_frame(self.M1, t_end),
                                          self._get_spatial_frame(self.M2, t_end), m1_max, m2_max)

        _style_snapshot_axis(axes[0], "Start ($t=0$)", rgb0)
        _style_snapshot_axis(axes[1], f"Midpoint ($t={self.Dt * t_mid:.1f}s$)", rgbMid)
        _style_snapshot_axis(axes[2], f"End ($t={self.Dt * t_end:.1f}s$)", rgbEnd)

        plt.tight_layout()
        plt.show()

    def save_mp4(self, filename="mfg_simulation_b.mp4", fps=15):
        print(f"Exporting MP4 video directly to '{filename}'...")

        fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
        ax.set_facecolor('#2c3e50')

        m1_max = np.max(self.M1) if np.max(self.M1) > 0 else 1.0
        m2_max = np.max(self.M2) if np.max(self.M2) > 0 else 1.0

        rgb_img = self._build_combined_rgb(
            self._get_spatial_frame(self.M1, 0),
            self._get_spatial_frame(self.M2, 0),
            m1_max, m2_max
        )

        im = ax.imshow(rgb_img, origin='lower', extent=self.extent, interpolation='nearest')
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        title = ax.set_title("Pop 1 (Red) & Pop 2 (Blue) - Time: 0.00s")

        goal1_marker = ax.scatter([], [], s=90, c='#ff5a5a',
                                 edgecolors='black', linewidths=0.8, zorder=10)
        goal2_marker = ax.scatter([], [], s=90, c='#5aa9ff',
                                 edgecolors='black', linewidths=0.8, zorder=10)

        def update(k):
            m1_frame = self._get_spatial_frame(self.M1, k)
            m2_frame = self._get_spatial_frame(self.M2, k)
            new_rgb = self._build_combined_rgb(m1_frame, m2_frame, m1_max, m2_max)

            im.set_data(new_rgb)

            g1 = self._active_goal(self.goal_1_trajectory, k)
            g2 = self._active_goal(self.goal_2_trajectory, k)

            if g1 is None:
                goal1_marker.set_offsets(np.empty((0, 2)))
                goal1_marker.set_visible(False)
            else:
                goal1_marker.set_visible(True)
                goal1_marker.set_offsets(np.array([[g1[0], g1[1]]]))

            if g2 is None:
                goal2_marker.set_offsets(np.empty((0, 2)))
                goal2_marker.set_visible(False)
            else:
                goal2_marker.set_visible(True)
                goal2_marker.set_offsets(np.array([[g2[0], g2[1]]]))

            title.set_text(f"Pop 1 (Red) & Pop 2 (Blue) - Time: {k * self.Dt:.2f}s")
            return [im, title, goal1_marker, goal2_marker]

        ani = animation.FuncAnimation(fig, update, frames=self.Nt + 1, blit=True)

        try:
            ani.save(filename, writer='ffmpeg', fps=fps, dpi=150)
            print(f"Successfully exported video to '{filename}'.")
        except Exception as e:
            print(f"Could not save MP4 via FFmpeg. Error details: {e}")
        finally:
            plt.close(fig)

# %%   
if __name__ == "__main__":
    MAP_FILE = "Maps/AcrosstheCape.map"
    SCEN_FILE = "Scenarios/AcrosstheCape.map.scen"
    
    ROOM_WIDTH = 768.0   
    ROOM_HEIGHT = 768.0  
  
    pde_mesh_pop1 = MAP2PDE(MAP_FILE, SCEN_FILE, Lx=ROOM_WIDTH, Ly=ROOM_HEIGHT, Nx=100, Ny=100)
    pde_mesh_pop2 = MAP2PDE(MAP_FILE, SCEN_FILE, Lx=ROOM_WIDTH, Ly=ROOM_HEIGHT, Nx=100, Ny=100)

    pde_mesh_pop1.parse_files(start_idx=0, num_agents=1) 
    pde_mesh_pop2.parse_files(start_idx=0, num_agents=1)

    pde_mesh_pop1.build_spatial_mesh()
    pde_mesh_pop2.build_spatial_mesh()

    pde_mesh_pop1.set_custom_initial_blobs([
        (400.0, 575.0, 40, 5) 
    ])

    pde_mesh_pop2.set_custom_initial_blobs([
        (400.0, 175.0, 40.0, 5)
    ])

    goal_config = {
    1: None,  # no goal for population 1
    2: {
        "times": [0.0,25.0, 50.0,75.0, 100.0],
        "positions": [(100.0, 700.0), (100.0, 650.0), (100.0, 500.0),(200,400),(100,300)],
    },
        }
    # Initialize solver
    mfg_solver = MFG2PopSolver(
        pde_mesh_data_1=pde_mesh_pop1, 
        pde_mesh_data_2=pde_mesh_pop2, 
        T=100.0, 
        Nt=1000, 
        thetaUM=0.1,
        goal_config=goal_config
    )
    
    # Solve and plot
    U1, M1, U2, M2 = mfg_solver.run_picard_system(max_iters=10)
    
    plotter = MFGPlotter(pde_mesh_data_1=pde_mesh_pop1, pde_mesh_data_2=pde_mesh_pop2, solver_instance=mfg_solver)
    plotter.plot_snapshots()
    plotter.save_mp4(filename="mfg_simulation_test_jumping_goal_4.mp4", fps=30)
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
import matplotlib.animation as animation
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
def compute_FP_matrix_entries_2Pop(m_prev, m_other, ukm1, omask_arr, Nx, Ny, Dx, Dy, Dt):
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

            #Linearization of the Hamiltonian terms
            p1 = (ukm1[ip1, j] - ukm1[i, j]) / Dx 
            p2 = (ukm1[i, j] - ukm1[im1, j]) / Dx
            p3 = (ukm1[i, jp1] - ukm1[i, j]) / Dy
            p4 = (ukm1[i, j] - ukm1[i, jm1]) / Dy
            
            # Diagonal entry for the current grid point
            diag_val = 1.0 / Dt
            
            # Diffusion contributions to the diagonal
            if ip1 != i: diag_val += 0.05 / (Dx ** 2)
            if im1 != i: diag_val += 0.05 / (Dx ** 2)
            if jp1 != j: diag_val += 0.05 / (Dy ** 2)
            if jm1 != j: diag_val += 0.05 / (Dy ** 2)

            #This changes depending on the choice of Hamiltonian
            c_h_curr = 2.0 / (1.0 + m_prev[i, j] + 5.0 * m_other[i, j])
            #diag_val += c_h_curr * (ppart(p1) / Dx + npart(p2) / Dx + ppart(p3) / Dy + npart(p4) / Dy)
            diag_val += c_h_curr * (npart(p1) / Dx + ppart(p2) / Dx + npart(p3) / Dy + ppart(p4) / Dy)

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
          # if i > 0 and omask_arr[i - 1, j] == 1:
                #c_h_left = 2.0 / (1.0 + m_prev[i - 1, j] + 5.0 * m_other[i - 1, j])
                #vals[entry_idx] = -c_h_left * ppart((ukm1[i, j] - ukm1[i - 1, j]) / Dx) / Dx
                #rows[entry_idx], cols[entry_idx] = ind, ((i - 1) * Ny + j)
                #entry_idx += 1
            #if i < Nx - 1 and omask_arr[i + 1, j] == 1:
                #c_h_right= 2.0 / (1.0 + m_prev[i + 1, j] + 5.0 * m_other[i + 1, j])
                #vals[entry_idx] = -c_h_right * npart((ukm1[i + 1, j] - ukm1[i, j]) / Dx) / Dx
                #rows[entry_idx], cols[entry_idx] = ind, ((i + 1) * Ny + j)
                #entry_idx += 1
            #if j > 0 and omask_arr[i, j - 1] == 1:
                #c_h_bottom= 2.0 / (1.0 + m_prev[i, j-1] + 5.0 * m_other[i, j-1])
                #vals[entry_idx] = -c_h_bottom * ppart((ukm1[i, j] - ukm1[i, j - 1]) / Dy) / Dy
                #rows[entry_idx], cols[entry_idx] = ind, (i * Ny + j - 1)
                #entry_idx += 1
            #if j < Ny - 1 and omask_arr[i, j + 1] == 1:
                #c_h_top= 2.0 / (1.0 + m_prev[i, j+1] + 5.0 * m_other[i, j+1])
                #vals[entry_idx] = -c_h_top * npart((ukm1[i, j + 1] - ukm1[i, j]) / Dy) / Dy
                #rows[entry_idx], cols[entry_idx] = ind, (i * Ny + j + 1)
                #entry_idx += 1
            if i > 0 and omask_arr[i - 1, j] == 1:
                c_h_left = 2.0 / (1.0 + m_prev[i - 1, j] + 5.0 * m_other[i - 1, j])
                vals[entry_idx] = -c_h_left * npart((ukm1[i, j] - ukm1[i - 1, j]) / Dx) / Dx  # Changed to npart
                rows[entry_idx], cols[entry_idx] = ind, ((i - 1) * Ny + j)
                entry_idx += 1
            if i < Nx - 1 and omask_arr[i + 1, j] == 1:
                c_h_right= 2.0 / (1.0 + m_prev[i + 1, j] + 5.0 * m_other[i + 1, j])
                vals[entry_idx] = -c_h_right * ppart((ukm1[i + 1, j] - ukm1[i, j]) / Dx) / Dx # Changed to ppart
                rows[entry_idx], cols[entry_idx] = ind, ((i + 1) * Ny + j)
                entry_idx += 1
            if j > 0 and omask_arr[i, j - 1] == 1:
                c_h_bottom= 2.0 / (1.0 + m_prev[i, j-1] + 5.0 * m_other[i, j-1])
                vals[entry_idx] = -c_h_bottom * npart((ukm1[i, j] - ukm1[i, j - 1]) / Dy) / Dy # Changed to npart
                rows[entry_idx], cols[entry_idx] = ind, (i * Ny + j - 1)
                entry_idx += 1
            if j < Ny - 1 and omask_arr[i, j + 1] == 1:
                c_h_top= 2.0 / (1.0 + m_prev[i, j+1] + 5.0 * m_other[i, j+1])
                vals[entry_idx] = -c_h_top * ppart((ukm1[i, j + 1] - ukm1[i, j]) / Dy) / Dy
                rows[entry_idx], cols[entry_idx] = ind, (i * Ny + j + 1)
                entry_idx += 1
            
    return rows[:entry_idx], cols[:entry_idx], vals[:entry_idx], b

@jit(nopython=True, cache=True)
def getFnU_2D_2Pop(Ukp1_np1, Ukp1_n, Mk_np1,Mk_other, omask_arr, Nx, Ny, Dx, Dy, Dt,pop=1):
    FnU = np.zeros((Nx, Ny))
    

    for i in range(Nx):
        for j in range(Ny):
            if omask_arr[i, j] == 0:
                FnU[i, j] = Ukp1_n[i, j] + 500.0
                continue
            
            # Compute indices of neighboring grid points with reflective boundary conditions
            ip1 = i + 1 if (i < Nx - 1 and omask_arr[i + 1, j] == 1) else i
            im1 = i - 1 if (i > 0      and omask_arr[i - 1, j] == 1) else i
            jp1 = j + 1 if (j < Ny - 1 and omask_arr[i, j + 1] == 1) else j
            jm1 = j - 1 if (j > 0      and omask_arr[i, j - 1] == 1) else j

            # Compute the time derivative term
            time_deriv = -(Ukp1_np1[i, j] - Ukp1_n[i, j]) / Dt

            # Compute the spatial derivatives for the Hamiltonian and diffusion terms
            p1 = (Ukp1_n[ip1, j] - Ukp1_n[i, j]) / Dx
            p2 = (Ukp1_n[i, j] - Ukp1_n[im1, j]) / Dx
            p3 = (Ukp1_n[i, jp1] - Ukp1_n[i, j]) / Dy
            p4 = (Ukp1_n[i, j] - Ukp1_n[i, jm1]) / Dy

            # Compute the diffusion term using a standard 5-point stencil
            laplacian_x = (Ukp1_n[ip1, j] - 2 * Ukp1_n[i, j] + Ukp1_n[im1, j]) / (Dx ** 2)
            laplacian_y = (Ukp1_n[i, jp1] - 2 * Ukp1_n[i, j] + Ukp1_n[i, jm1]) / (Dy ** 2)
            diffusion = -0.05 * (laplacian_x + laplacian_y)


            #Hamiltonian changes the scenario 

            #hamiltonian = (1/(1+Mk_np1[i,j]+5*Mk_other[i,j]))*(
               # ppart(p1)**2 + npart(p2)**2 + ppart(p3)**2 + npart(p4)**2)
            hamiltonian = (1/(1+Mk_np1[i,j]+5*Mk_other[i,j]))*(
                        npart(p1)**2 + ppart(p2)**2 + npart(p3)**2 + ppart(p4)**2)
            
            if pop == 1:
                # POP 1: Attracted to Pop 2 (+2500) AND attracted to their own swarm (+800)
                interaction_cost = +1500.0 * Mk_other[i, j]+ppart(Mk_np1[i,j]+Mk_other[i,j]-4)#+ 200.0 * Mk_np1[i, j]
            else:
                # POP 2: Phobic of Pop 1 (-5000) AND attracted to their own swarm (+800)
                interaction_cost = -5000.0 * Mk_other[i, j]+ppart(Mk_np1[i,j]+Mk_other[i,j]-4)#+  200.0 * Mk_np1[i, j]

            FnU[i, j] = time_deriv + diffusion + hamiltonian+interaction_cost
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
            
            # Compute the diagonal entry for the current grid point
            diag_val = 1.0 / Dt
            # Check neighbors with reflective Zero-Neumann bounds
            ip1 = i + 1 if (i < Nx - 1 and omask_arr[i + 1, j] == 1) else i
            im1 = i - 1 if (i > 0      and omask_arr[i - 1, j] == 1) else i
            jp1 = j + 1 if (j < Ny - 1 and omask_arr[i, j + 1] == 1) else j
            jm1 = j - 1 if (j > 0      and omask_arr[i, j - 1] == 1) else j

            # Diffusion contributions to the diagonal
            if ip1 != i: diag_val += 0.05 / (Dx ** 2)
            if im1 != i: diag_val += 0.05 / (Dx ** 2)
            if jp1 != j: diag_val += 0.05 / (Dy ** 2)
            if jm1 != j: diag_val += 0.05 / (Dy ** 2)

            #Linearization of the Hamiltonian terms
            p1 = (Unew_n_tmp[ip1, j] - Unew_n_tmp[i, j]) / Dx
            p2 = (Unew_n_tmp[i, j] - Unew_n_tmp[im1, j]) / Dx
            p3 = (Unew_n_tmp[i, jp1] - Unew_n_tmp[i, j]) / Dy
            p4 = (Unew_n_tmp[i, j] - Unew_n_tmp[i, jm1]) / Dy

            #Change depending on the Hamiltonian 
            c_h = 2.0 / (1.0 + Mk_np1[i, j] + 5.0 * Mk_other[i, j])

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


            if ip1 != i: # p1 (Forward X)
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, (ip1 * Ny + j), -c_h * npart(p1) / Dx
                entry_idx += 1
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, c_h * npart(p1) / Dx
                entry_idx += 1
            if im1 != i: # p2 (Backward X)
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, (im1 * Ny + j), -c_h * ppart(p2) / Dx
                entry_idx += 1
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, c_h * ppart(p2) / Dx
                entry_idx += 1
            if jp1 != j: # p3 (Forward Y)
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, (i * Ny + jp1), -c_h * npart(p3) / Dy
                entry_idx += 1
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, c_h * npart(p3) / Dy
                entry_idx += 1
            if jm1 != j: # p4 (Backward Y)
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, (i * Ny + jm1), -c_h * ppart(p4) / Dy
                entry_idx += 1
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, c_h * ppart(p4) / Dy
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

    def parse_files(self, num_agents: int = None, start_idx: int = 0):
        """Parses the .map and .scen files and stores the raw discrete data."""
        with open(self.map_filepath, 'r') as f:
            lines = f.readlines()
        
        height = int(lines[1].split()[1]) 
        width = int(lines[2].split()[1])
        self.grid_shape = (height, width)
        self.raw_boolean_grid = np.zeros((height, width), dtype=bool)
        
        # Extract the boolean grid from the map file
        for r, line in enumerate(lines[4:]):
            for c, char in enumerate(line.strip()):
                if char in ['.', 'G']:
                    self.raw_boolean_grid[r, c] = True  

        # Extract agents from the scenario file            
        with open(self.scen_filepath, 'r') as f:
            scen_lines = f.readlines()[1:] 
            
        for i, line in enumerate(scen_lines):
            # Skip agents until we reach the assigned starting index
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
        """
        Overrides scenario starting points with custom Gaussian density blobs.

        Parameters:
        -----------
        blob_list : list of tuples
            Format: [(x_center, y_center, sigma), ...]
            or [(x_center, y_center, sigma, amplitude), ...]
        normalize_mass : bool
            If True, normalizes total population mass grid sum to 1.0.
        """
        if self.X is None or self.Y is None:
            raise ValueError("Must call build_spatial_mesh() before setting custom initial density.")

        m_0 = np.zeros_like(self.X, dtype=np.float64)

        for blob in blob_list:
            if len(blob) == 3:
                cx, cy, sigma = blob
                amp = 1.0
            else:
                cx, cy, sigma, amp = blob

            # Add multi-variate Gaussian blob centered at (cx, cy)
            m_0 += amp * np.exp(-((self.X - cx)**2 + (self.Y - cy)**2) / (2.0 * sigma**2))

        # Zero out density values inside wall obstacles
        pde_mask = self.get_pde_obstacle_mask()
        m_0 *= pde_mask

        if normalize_mass and np.sum(m_0) > 0:
            m_0 /= (np.sum(m_0) * self.dx * self.dy)

        self.custom_m0 = m_0
        return m_0
    
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

    def set_custom_goals(self, goal_list):
        """
        Overrides scenario goals with an explicit list of (x, y) coordinates.
        """
        self.custom_goals = goal_list

    def get_goals(self):
        """Returns active goal positions (custom goals if defined, else scenario goals)."""
        if hasattr(self, 'custom_goals') and self.custom_goals is not None:
            return self.custom_goals

        if not hasattr(self, 'grid_shape') or self.grid_shape == (0, 0):
            return []

        H, W = self.grid_shape
        map_dx = self.Lx / W
        map_dy = self.Ly / H
        goal_positions = []
        for agent in self.raw_agents:
            goal_x = (agent['goal_x'] + 0.5) * map_dx
            goal_y = self.Ly - (agent['goal_y'] + 0.5) * map_dy
            goal_positions.append((goal_x, goal_y))
            
        return goal_positions
    
    def build_terminal_cost(self, penalty_scale: float = 15.0):
        if self.X is None or self.Y is None:
            raise ValueError("Mesh needs initialization.")
        Nx, Ny = self.X.shape
        g = np.zeros((Nx, Ny), dtype=np.float64)
        
        # 1. Check if custom goals were set via set_custom_goals()
        # Check 'is not None' so that an empty list [] is accepted as a valid override
        if hasattr(self, 'custom_goals') and self.custom_goals is not None:
            goal_positions = self.custom_goals
            
        # 2. Otherwise, fall back to the raw scenario file goals
        else:
            H, W = self.grid_shape
            map_dx = self.Lx / W
            map_dy = self.Ly / H
            goal_positions = []
            for agent in self.raw_agents:
                goal_x = (agent['goal_x'] + 0.5) * map_dx
                goal_y = self.Ly - (agent['goal_y'] + 0.5) * map_dy
                goal_positions.append((goal_x, goal_y))
            
        # If the goal_positions list is empty, return the flat zero array
        if not goal_positions:
            return g
            
        #Build the actual spatial PDE cost grid
        for i in range(Nx):
            for j in range(Ny):
                x_val, y_val = self.X[i, j], self.Y[i, j]
                min_dist = min(np.sqrt((x_val - gx)**2 + (y_val - gy)**2) for gx, gy in goal_positions)
                g[i, j] = penalty_scale * min_dist
                
        return g
# %%
class MFG2PopSolver:
    def __init__(self, pde_mesh_data_1,pde_mesh_data_2, T: float = 5.0, Nt: int = 100, thetaUM: float = 0.1):
        self.Nx, self.Ny = pde_mesh_data_1.X.shape
        self.Dx = pde_mesh_data_1.dx  
        self.Dy = pde_mesh_data_1.dy  
        self.Dt = T / Nt
        self.Nt = Nt
        self.thetaUM = thetaUM
        
        self.omask = pde_mesh_data_1.get_pde_obstacle_mask()
        self.m0_1 = pde_mesh_data_1.build_initial_density(sigma_multiplier=3)
        self.g_x_1 = pde_mesh_data_1.build_terminal_cost()
        self.m0_2 = pde_mesh_data_2.build_initial_density(sigma_multiplier=3)
        self.g_x_2 = pde_mesh_data_2.build_terminal_cost()

        # State histories
        self.M1,self.M2 = np.zeros((self.Nt + 1, self.Nx, self.Ny)),np.zeros((self.Nt + 1, self.Nx, self.Ny))
        self.U1,self.U2 = np.zeros((self.Nt + 1, self.Nx, self.Ny)),np.zeros((self.Nt + 1, self.Nx, self.Ny))
        
        #Initial densities and final costs
        self.M1[0], self.M2[0] = self.m0_1, self.m0_2
        self.U1[self.Nt], self.U2[self.Nt] = self.g_x_1,self.g_x_2

    def solve_forward_FP(self, U_trajectory,m0,M_other_trajectory):
        m = np.zeros((self.Nt + 1, self.Nx, self.Ny))
        m[0] = m0
        N_total = self.Nx * self.Ny

        for k in range(1, self.Nt + 1):
            rows, cols, vals, b = compute_FP_matrix_entries_2Pop(
                m[k - 1],M_other_trajectory[k-1], U_trajectory[k - 1], self.omask,
                self.Nx, self.Ny, self.Dx, self.Dy, self.Dt
            )
            A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()
            mtmp = sparse.linalg.spsolve(A, b)
            m[k] = mtmp.reshape((self.Nx, self.Ny))
        return m

    def get_u_onestep_newton(self, Uk_n, Unew_np1, Unew_n_tmp, Mk_np1,Mk_other,pop):
        N_total = self.Nx * self.Ny
        FnU_flat = getFnU_2D_2Pop(Unew_np1, Unew_n_tmp, Mk_np1,Mk_other, self.omask, 
                             self.Nx, self.Ny, self.Dx, self.Dy, self.Dt,pop).flatten()

        #No need to add the other population as jacobian with respect to the other bvalue function is zero.
        rows, cols, vals = compute_HJB_matrix_entries(
            Unew_n_tmp, Mk_np1,Mk_other, self.omask, self.Nx, self.Ny, self.Dx, self.Dy, self.Dt
        )
        A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()
        b = A.dot(Unew_n_tmp.flatten()) - FnU_flat

        for i in range(self.Nx):
            for j in range(self.Ny):
                if self.omask[i, j] == 0:
                    b[i * self.Ny + j] = -500.0

        utmp = sparse.linalg.spsolve(A, b)
        return utmp.reshape((self.Nx, self.Ny))

    def solve_hjb_single_timestep(self, Uk_n, Unew_np1, Mk_np1, Mk_other_np1,pop):
        Unew_n = np.copy(Unew_np1)
        for _ in range(5):
            Unres = self.get_u_onestep_newton(Uk_n, Unew_np1, Unew_n, Mk_np1, Mk_other_np1,pop)
            l2err = np.linalg.norm(Unew_n.flatten() - Unres.flatten()) * np.sqrt(self.Dx * self.Dy)
            Unew_n = np.copy(Unres)
            if l2err < 1e-6:
                break
        return Unew_n

    def solve_backward_HJB(self, M_trajectory,M_trajectory_other,U_temp,g_x,pop):
        u = np.zeros_like(U_temp)
        u[self.Nt] = g_x
        for k in range(self.Nt - 1, -1, -1):
            u[k] = self.solve_hjb_single_timestep(U_temp[k], u[k + 1], M_trajectory[k + 1], M_trajectory_other[k + 1],pop)
        return u

    def run_picard_system(self, max_iters: int = 25, tolerance: float = 1e-5):
        space_time_factor = np.sqrt(self.Dx * self.Dy * self.Dt)
        
        for iiter in range(1, max_iters + 1):
            start_time = time.time()
            print(f"\n>>> Macro Picard Loop Execution: {iiter} / {max_iters}")

            # Backward HJB update step
            U1_temp = self.solve_backward_HJB(self.M1, self.M2, self.U1, self.g_x_1,pop=1)
            U2_temp = self.solve_backward_HJB(self.M2, self.M1, self.U2, self.g_x_2,pop=2)
            
            U1_new = self.thetaUM * U1_temp + (1.0 - self.thetaUM) * self.U1
            U2_new = self.thetaUM * U2_temp + (1.0 - self.thetaUM) * self.U2

            # Forward FP update step
            M1_temp = self.solve_forward_FP(U1_new, self.m0_1,self.M2) 
            M2_temp = self.solve_forward_FP(U2_new, self.m0_2,self.M1)
            
            M1_new = self.thetaUM * M1_temp + (1.0 - self.thetaUM) * self.M1
            M2_new = self.thetaUM * M2_temp + (1.0 - self.thetaUM) * self.M2
            # Convergence Check
            u_err = max(np.linalg.norm(U1_new - self.U1), np.linalg.norm(U2_new - self.U2)) * space_time_factor
            m_err = max(np.linalg.norm(M1_new - self.M1), np.linalg.norm(M2_new - self.M2)) * space_time_factor

            self.U1, self.M1 = np.copy(U1_new), np.copy(M1_new) #Pop1 
            self.U2, self.M2 = np.copy(U2_new), np.copy(M2_new) #Pop2

            if u_err < tolerance and m_err < tolerance:
                break
        return self.U1, self.M1, self.U2, self.M2


# %%

# Made mostly with AI
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

        # Retrieve active goal locations (handles custom goals automatically)
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
        # Safely clip to 0.0 BEFORE applying the fractional power to prevent NaNs
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
            # Add interpolation='nearest' here
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


   
          
        

 # %%   
if __name__ == "__main__":
    MAP_FILE = "Maps/AcrosstheCape.map"
    SCEN_FILE = "Scenarios/AcrosstheCape.map.scen"
    
    ROOM_WIDTH = 768.0   
    ROOM_HEIGHT = 768.0  
    
    pde_mesh_pop1 = MAP2PDE(MAP_FILE, SCEN_FILE, Lx=ROOM_WIDTH, Ly=ROOM_HEIGHT, Nx=100, Ny=100)
    pde_mesh_pop2 = MAP2PDE(MAP_FILE, SCEN_FILE, Lx=ROOM_WIDTH, Ly=ROOM_HEIGHT, Nx=100, Ny=100)

    # Must parse map walls first
    pde_mesh_pop1.parse_files(start_idx=0, num_agents=1) #Not necessary for the custom initial density
    pde_mesh_pop2.parse_files(start_idx=0, num_agents=1)

    # Build spatial coordinate grids
    pde_mesh_pop1.build_spatial_mesh()
    pde_mesh_pop2.build_spatial_mesh()

    # --- POPULATION 1: Large blob centered at (x, y) ---
    pde_mesh_pop1.set_custom_initial_blobs([
        (400.0, 575.0, 40, 5)  # (x_center, y_center, sigma, amplitude)
    ])

    # --- POPULATION 2: Large blob centered at (x, y) ---
    
    pde_mesh_pop2.set_custom_initial_blobs([
        (400.0, 175.0, 40.0, 5)
    ])

    #pop1_goals = [(400.0, 500.0)]
    pop1_goals=[]
    pde_mesh_pop1.set_custom_goals(pop1_goals) #No goals for Pop 1

    #pop2_goals = [(12.0, 24.0), (12.0, 42.0), (12.0, 60.0)] #Designated goals for Pop 2
    #pop2_goals = [(400.0, 700.0),(100,600),(700,600)]
    pop2_goals = [(400.0, 700.0)]
    pde_mesh_pop2.set_custom_goals(pop2_goals)

    # Initialize solver
    mfg_solver = MFG2PopSolver(
        pde_mesh_data_1=pde_mesh_pop1, 
        pde_mesh_data_2=pde_mesh_pop2, 
        T=300.0, #Finite time horizon 
        Nt=3000, #Time Steps
        thetaUM=0.1 #Mixing parameter for Picard iterations
    )
    
    # Solve and plot
    
    U1, M1, U2, M2 = mfg_solver.run_picard_system(max_iters=10)
    
    plotter = MFGPlotter(pde_mesh_data_1=pde_mesh_pop1, pde_mesh_data_2=pde_mesh_pop2, solver_instance=mfg_solver)
    plotter.plot_snapshots()
    plotter.save_mp4(filename="mfg_simulation_test3.mp4", fps=30)
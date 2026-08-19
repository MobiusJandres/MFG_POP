"""
Low-level Numba JIT-compiled finite difference operators and matrix assembly.
Handles both Hard Dirichlet exits (door_mask_n) and Soft attraction costs (running_cost_k).
Supports 1-Population (Traffic & Pursuit-Evasion) and 2-Population systems.
"""
import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def ppart(x):
    return np.maximum(x, 0.0)


@jit(nopython=True, cache=True)
def npart(x):
    return -np.minimum(x, 0.0)


@jit(nopython=True, cache=True)
def H_withM(m_val, p1, p2, p3, p4, scale=8.0, power=0.75, offset=0.1):
    grad_norm_sq = ppart(p1) ** 2 + npart(p2) ** 2 + ppart(p3) ** 2 + npart(p4) ** 2
    return -scale * (1.0 / (1.0 + m_val) ** power) * grad_norm_sq + offset


# =============================================================================
# 1-POPULATION NUMERICAL OPERATORS
# =============================================================================

@jit(nopython=True, cache=True)
def compute_FP_matrix_entries(m_prev, ukm1, omask_arr, door_mask_arr, Nx, Ny, Dx, Dy, Dt):
    """Assembles sparse COO matrix for forward Fokker-Planck transport."""
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

            if door_mask_arr[i, j] == 1 or omask_arr[i, j] == 0:
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

            c_h_curr = 16.0 / ((1.0 + m_prev[i, j]) ** 0.75)
            diag_val += c_h_curr * (ppart(p1) / Dx + npart(p2) / Dx + ppart(p3) / Dy + npart(p4) / Dy)

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

            if i > 0 and omask_arr[i - 1, j] == 1 and door_mask_arr[i - 1, j] == 0:
                c_h_left = 16.0 / ((1.0 + m_prev[i - 1, j]) ** 0.75)
                vals[entry_idx] = -c_h_left * ppart((ukm1[i, j] - ukm1[i - 1, j]) / Dx) / Dx
                rows[entry_idx], cols[entry_idx] = ind, ((i - 1) * Ny + j)
                entry_idx += 1
            if i < Nx - 1 and omask_arr[i + 1, j] == 1 and door_mask_arr[i + 1, j] == 0:
                c_h_right = 16.0 / ((1.0 + m_prev[i + 1, j]) ** 0.75)
                vals[entry_idx] = -c_h_right * npart((ukm1[i + 1, j] - ukm1[i, j]) / Dx) / Dx
                rows[entry_idx], cols[entry_idx] = ind, ((i + 1) * Ny + j)
                entry_idx += 1
            if j > 0 and omask_arr[i, j - 1] == 1 and door_mask_arr[i, j - 1] == 0:
                c_h_bottom = 16.0 / ((1.0 + m_prev[i, j - 1]) ** 0.75)
                vals[entry_idx] = -c_h_bottom * ppart((ukm1[i, j] - ukm1[i, j - 1]) / Dy) / Dy
                rows[entry_idx], cols[entry_idx] = ind, (i * Ny + j - 1)
                entry_idx += 1
            if j < Ny - 1 and omask_arr[i, j + 1] == 1 and door_mask_arr[i, j + 1] == 0:
                c_h_top = 16.0 / ((1.0 + m_prev[i, j + 1]) ** 0.75)
                vals[entry_idx] = -c_h_top * npart((ukm1[i, j + 1] - ukm1[i, j]) / Dy) / Dy
                rows[entry_idx], cols[entry_idx] = ind, (i * Ny + j + 1)
                entry_idx += 1

    return rows[:entry_idx], cols[:entry_idx], vals[:entry_idx], b


@jit(nopython=True, cache=True)
def getFnU_2D(Unew_np1, Unew_n, Mk_np1, omask_arr, door_mask_n, running_cost_k, Nx, Ny, Dx, Dy, Dt):
    """Computes HJB residual with door Dirichlet conditions and soft running costs."""
    FnU = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            if door_mask_n[i, j] == 1:
                FnU[i, j] = Unew_n[i, j] - 0.0
                continue

            if omask_arr[i, j] == 0:
                FnU[i, j] = Unew_n[i, j] - 500.0
                continue

            ip1 = i + 1 if (i < Nx - 1 and omask_arr[i + 1, j] == 1) else i
            im1 = i - 1 if (i > 0      and omask_arr[i - 1, j] == 1) else i
            jp1 = j + 1 if (j < Ny - 1 and omask_arr[i, j + 1] == 1) else j
            jm1 = j - 1 if (j > 0      and omask_arr[i, j - 1] == 1) else j

            time_deriv = -(Unew_np1[i, j] - Unew_n[i, j]) / Dt

            p1 = (Unew_n[ip1, j] - Unew_n[i, j]) / Dx
            p2 = (Unew_n[i, j] - Unew_n[im1, j]) / Dx
            p3 = (Unew_n[i, jp1] - Unew_n[i, j]) / Dy
            p4 = (Unew_n[i, j] - Unew_n[i, jm1]) / Dy

            laplacian_x = (Unew_n[ip1, j] - 2 * Unew_n[i, j] + Unew_n[im1, j]) / (Dx ** 2)
            laplacian_y = (Unew_n[i, jp1] - 2 * Unew_n[i, j] + Unew_n[i, jm1]) / (Dy ** 2)
            diffusion = -0.05 * (laplacian_x + laplacian_y)

            hamiltonian = H_withM(Mk_np1[i, j], p1, p2, p3, p4)
            FnU[i, j] = time_deriv + diffusion + hamiltonian + running_cost_k[i, j]

    return FnU


@jit(nopython=True, cache=True)
def compute_HJB_matrix_entries(Unew_n_tmp, Mk_np1, omask_arr, door_mask_arr, Nx, Ny, Dx, Dy, Dt):
    """Assembles sparse COO matrix for backward HJB linearization."""
    N_total = Nx * Ny
    max_entries = N_total * 16
    rows = np.zeros(max_entries, dtype=np.int64)
    cols = np.zeros(max_entries, dtype=np.int64)
    vals = np.zeros(max_entries, dtype=np.float64)

    entry_idx = 0

    for i in range(Nx):
        for j in range(Ny):
            ind = i * Ny + j
            if door_mask_arr[i, j] == 1 or omask_arr[i, j] == 0:
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


# =============================================================================
# 2-POPULATION NUMERICAL OPERATORS
# =============================================================================

@jit(nopython=True, cache=True)
def compute_FP_matrix_entries_2Pop(m_prev, m_other, ukm1, omask_arr, Nx, Ny, Dx, Dy, Dt):
    """Fokker-Planck matrix assembly for 2-population coupled transport."""
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
def getFnU_2D_2Pop(Ukp1_np1, Ukp1_n, Mk_np1, Mk_other, omask_arr, Nx, Ny, Dx, Dy, Dt, pop=1):
    """Computes 2-population HJB residual including interaction costs."""
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

            hamiltonian = (1.0 / (1.0 + Mk_np1[i, j] + 5.0 * Mk_other[i, j])) * (
                npart(p1)**2 + ppart(p2)**2 + npart(p3)**2 + ppart(p4)**2
            )

            if pop == 1:
                interaction_cost = +1500.0 * Mk_other[i, j] + ppart(Mk_np1[i, j] + Mk_other[i, j] - 4.0)
            else:
                interaction_cost = -5000.0 * Mk_other[i, j] + ppart(Mk_np1[i, j] + Mk_other[i, j] - 4.0)

            FnU[i, j] = time_deriv + diffusion + hamiltonian + interaction_cost

    return FnU


@jit(nopython=True, cache=True)
def compute_HJB_matrix_entries_2Pop(Unew_n_tmp, Mk_np1, Mk_other, omask_arr, Nx, Ny, Dx, Dy, Dt):
    """Assembles sparse COO matrix for 2-population HJB linearization."""
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
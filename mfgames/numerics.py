"""
Low-level Numba JIT-compiled finite difference operators and matrix assembly.

This module implements the numerical discretization for Mean Field Game (MFG) systems,
including both Hamilton-Jacobi-Bellman (HJB) and Kolmogorov-Fokker-Planck (KFP) equations.
All functions are JIT-compiled with Numba for performance-critical numerical operations.

The module supports:
- 1-Population systems: Traffic flow and pursuit-evasion scenarios
- 2-Population systems: Competitive multi-agent interactions
- Hard Dirichlet boundary conditions: Exit doors with zero value (door_mask)
- Soft attraction costs: Spatially-varying running costs (running_cost_k)
- Obstacle handling: Both static and dynamic obstacles via omask

Key Mathematical Components:
- Upwind finite difference scheme for first-order derivatives
- Central difference scheme for diffusion (Laplacian)
- Semi-implicit time discretization (implicit for spatial operators)
- Sparse matrix assembly in COO format for efficient linear solves

Reference: See notes.tex for mathematical derivation of discretization schemes.
"""
import numpy as np
from numba import jit


@jit(nopython=True, cache=True)
def ppart(x):
    """
    Positive part operator for upwind scheme.

    Extracts the positive component of a scalar value, used in upwind finite
    difference discretization to select the appropriate directional derivative
    based on the characteristic direction (flow direction).

    Args:
        x: Scalar value (typically a finite difference approximation)

    Returns:
        float: max(x, 0.0) - the positive part of x

    Note:
        This implements the "positive part" in the upwind discretization:
        max(p, 0)^2 corresponds to forward differences when flow is rightward/upward.
    """
    return np.maximum(x, 0.0)


@jit(nopython=True, cache=True)
def npart(x):
    """
    Negative part operator for upwind scheme.

    Extracts the negative component of a scalar value (as positive magnitude),
    used in upwind finite difference discretization to select backward differences
    when characteristics point in the negative coordinate direction.

    Args:
        x: Scalar value (typically a finite difference approximation)

    Returns:
        float: -min(x, 0.0) - the absolute value of the negative part of x

    Note:
        This implements the "negative part" in the upwind discretization:
        min(p, 0)^2 = npart(p)^2 corresponds to backward differences when flow
        is leftward/downward.
    """
    return -np.minimum(x, 0.0)


@jit(nopython=True, cache=True)
def H_withM(m_val, p1, p2, p3, p4, scale=8.0, power=0.75, offset=0.1):
    """
    Congestion-dependent Hamiltonian for traffic MFG.

    Computes the discrete Hamiltonian H(m, ∇u) with upwind gradient norm:
        H = -scale * (1 + m)^(-power) * ||∇u||²_upwind + offset

    The congestion term (1 + m)^(-power) models crowd avoidance: higher density m
    reduces agent speed/mobility. The upwind gradient norm uses positive/negative
    parts to ensure proper directionality in the characteristic flow.

    Args:
        m_val: Population density at the current grid point (float)
        p1: Forward x-difference (u[i+1,j] - u[i,j])/Dx
        p2: Backward x-difference (u[i,j] - u[i-1,j])/Dx
        p3: Forward y-difference (u[i,j+1] - u[i,j])/Dy
        p4: Backward y-difference (u[i,j] - u[i,j-1])/Dy
        scale: Mobility coefficient (default 8.0)
        power: Congestion exponent (default 0.75)
        offset: Running cost baseline (default 0.1)

    Returns:
        float: Hamiltonian value H(m, p)

    Note:
        The upwind norm ||∇u||²_upwind = max(p1,0)² + max(-p2,0)² + max(p3,0)² + max(-p4,0)²
        ensures numerical stability by respecting characteristic directions.
    """
    grad_norm_sq = ppart(p1) ** 2 + npart(p2) ** 2 + ppart(p3) ** 2 + npart(p4) ** 2
    return -scale * (1.0 / (1.0 + m_val) ** power) * grad_norm_sq + offset


# =============================================================================
# 1-POPULATION NUMERICAL OPERATORS
# =============================================================================

@jit(nopython=True, cache=True)
def compute_FP_matrix_entries(m_prev, ukm1, omask_arr, door_mask_arr, Nx, Ny, Dx, Dy, Dt):
    """
    Assembles sparse COO matrix for forward-in-time Fokker-Planck (KFP) equation.

    Discretizes the KFP continuity equation:
        ∂m/∂t = div(m * ∇H_p) + ε * Δm
    where the drift term div(m * ∇H_p) uses upwind transport based on the optimal
    velocity field derived from the value function u, and ε * Δm is diffusion.

    The method constructs a sparse linear system A * m^{n+1} = b in COO format,
    where m^{n+1} is the unknown density at the next time step.

    Boundary Conditions:
        - Exit doors (door_mask == 1): Homogeneous Dirichlet (m = 0)
        - Obstacles (omask == 0): Homogeneous Dirichlet (m = 0)
        - Domain boundaries: Neumann (zero flux)

    Args:
        m_prev: Population density at current time step, shape (Nx, Ny)
        ukm1: Value function from previous Picard iteration, shape (Nx, Ny)
        omask_arr: Obstacle mask (1 = walkable, 0 = obstacle), shape (Nx, Ny)
        door_mask_arr: Exit door mask (1 = exit, 0 = interior), shape (Nx, Ny)
        Nx: Number of grid points in x-direction (int)
        Ny: Number of grid points in y-direction (int)
        Dx: Grid spacing in x-direction (float, metres)
        Dy: Grid spacing in y-direction (float, metres)
        Dt: Time step size (float, seconds)

    Returns:
        tuple: (rows, cols, vals, b) where:
            - rows: Row indices for sparse COO matrix (int64 array)
            - cols: Column indices for sparse COO matrix (int64 array)
            - vals: Non-zero values for sparse COO matrix (float64 array)
            - b: Right-hand side vector (float64 array, length Nx*Ny)

    Note:
        The diffusion coefficient ε = 0.05 is hardcoded (line 64-67, 76-85).
        The congestion coupling c_h = 16/(1+m)^0.75 appears in the drift term (line 69).
        The matrix is row-major indexed: grid point (i,j) → linear index i*Ny + j.
    """
    # Pre-allocate COO sparse matrix storage (max 9 entries per grid point)
    N_total = Nx * Ny
    max_entries = N_total * 9
    rows = np.zeros(max_entries, dtype=np.int64)
    cols = np.zeros(max_entries, dtype=np.int64)
    vals = np.zeros(max_entries, dtype=np.float64)
    b = np.zeros(N_total, dtype=np.float64)

    entry_idx = 0

    for i in range(Nx):
        for j in range(Ny):
            ind = i * Ny + j  # Row-major linear index
            b[ind] = m_prev[i, j] / Dt  # RHS from implicit Euler time stepping

            # Boundary conditions: exits and obstacles enforce m = 0
            if door_mask_arr[i, j] == 1 or omask_arr[i, j] == 0:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, 1.0
                entry_idx += 1
                b[ind] = 0.0
                continue

            # Ghost nodes for Neumann boundaries: clamp to boundary if neighbor is obstacle/outside
            ip1 = i + 1 if (i < Nx - 1 and omask_arr[i + 1, j] == 1) else i
            im1 = i - 1 if (i > 0      and omask_arr[i - 1, j] == 1) else i
            jp1 = j + 1 if (j < Ny - 1 and omask_arr[i, j + 1] == 1) else j
            jm1 = j - 1 if (j > 0      and omask_arr[i, j - 1] == 1) else j

            # Compute finite differences for optimal velocity field ∇H_p(u)
            p1 = (ukm1[ip1, j] - ukm1[i, j]) / Dx
            p2 = (ukm1[i, j] - ukm1[im1, j]) / Dx
            p3 = (ukm1[i, jp1] - ukm1[i, j]) / Dy
            p4 = (ukm1[i, j] - ukm1[i, jm1]) / Dy

            # Diagonal accumulator: starts with time derivative term
            # Diagonal accumulator: starts with time derivative term
            diag_val = 1.0 / Dt
            # Add diffusion contributions (ε = 0.05): diagonal gets +ε/Dx² for each valid neighbor
            if ip1 != i: diag_val += 0.05 / (Dx ** 2)
            if im1 != i: diag_val += 0.05 / (Dx ** 2)
            if jp1 != j: diag_val += 0.05 / (Dy ** 2)
            if jm1 != j: diag_val += 0.05 / (Dy ** 2)

            # Add upwind transport contribution: c_h * div(∇H_p) on diagonal
            c_h_curr = 16.0 / ((1.0 + m_prev[i, j]) ** 0.75)  # Congestion mobility
            diag_val += c_h_curr * (ppart(p1) / Dx + npart(p2) / Dx + ppart(p3) / Dy + npart(p4) / Dy)

            rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, diag_val
            entry_idx += 1

            # Diffusion off-diagonal entries: -ε/Dx² or -ε/Dy² for each valid neighbor
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

            # Upwind transport off-diagonal entries: -c_h * flux from neighbor cells
            # These implement the discretized divergence of the drift term m * ∇H_p
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
def getFnU_2D(Unew_np1, Unew_n, Mk_np1, omask_arr, door_mask_n, running_cost_k, Nx, Ny, Dx, Dy, Dt, obstacle_penalty=500.0):
    """
    Computes the HJB residual F(u) for Newton iteration in the value function solver.

    Evaluates the discrete HJB equation residual:
        F(u) = -∂u/∂t + H(m, ∇u) - ε * Δu + L(x)

    Args:
        Unew_np1: Value at next time step n+1, shape (Nx, Ny)
        Unew_n: Value at current time step n, shape (Nx, Ny)
        Mk_np1: Density at time n+1, shape (Nx, Ny)
        omask_arr: Obstacle mask (1=walkable, 0=obstacle), shape (Nx, Ny)
        door_mask_n: Exit door mask (1=exit, 0=interior), shape (Nx, Ny)
        running_cost_k: Spatially-varying cost field, shape (Nx, Ny)
        Nx, Ny: Grid dimensions (int)
        Dx, Dy: Grid spacing (float, metres)
        Dt: Time step (float, seconds)
        obstacle_penalty: Obstacle potential (float, default 500.0)

    Returns:
        ndarray: Residual F(u), shape (Nx, Ny)
    """
    FnU = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            # Exit doors: enforce hard Dirichlet condition u = 0
            if door_mask_n[i, j] == 1:
                FnU[i, j] = Unew_n[i, j] - 0.0
                continue

            # Obstacles: large positive potential barrier to discourage entry
            if omask_arr[i, j] == 0:
                FnU[i, j] = Unew_n[i, j] - obstacle_penalty
                continue

            # Ghost nodes for Neumann boundaries
            ip1 = i + 1 if (i < Nx - 1 and omask_arr[i + 1, j] == 1) else i
            im1 = i - 1 if (i > 0      and omask_arr[i - 1, j] == 1) else i
            jp1 = j + 1 if (j < Ny - 1 and omask_arr[i, j + 1] == 1) else j
            jm1 = j - 1 if (j > 0      and omask_arr[i, j - 1] == 1) else j

            # Backward time derivative: -∂u/∂t ≈ -(u^{n+1} - u^n)/Dt
            time_deriv = -(Unew_np1[i, j] - Unew_n[i, j]) / Dt

            # Finite differences for gradient (used in Hamiltonian)
            p1 = (Unew_n[ip1, j] - Unew_n[i, j]) / Dx
            p2 = (Unew_n[i, j] - Unew_n[im1, j]) / Dx
            p3 = (Unew_n[i, jp1] - Unew_n[i, j]) / Dy
            p4 = (Unew_n[i, j] - Unew_n[i, jm1]) / Dy

            # Diffusion term: -ε * Δu (viscosity, ε = 0.05)
            laplacian_x = (Unew_n[ip1, j] - 2 * Unew_n[i, j] + Unew_n[im1, j]) / (Dx ** 2)
            laplacian_y = (Unew_n[i, jp1] - 2 * Unew_n[i, j] + Unew_n[i, jm1]) / (Dy ** 2)
            diffusion = -0.05 * (laplacian_x + laplacian_y)

            # Hamiltonian: H = -8*(1+m)^(-0.75)*||∇u||² + 0.1
            hamiltonian = -8.0 * (1.0 / (1.0 + Mk_np1[i, j]) ** 0.75) * (
                ppart(p1)**2 + npart(p2)**2 + ppart(p3)**2 + npart(p4)**2
            ) + 0.1

            # HJB residual: F(u) = -∂u/∂t - ε*Δu + H + L(x)
            FnU[i, j] = time_deriv + diffusion + hamiltonian + running_cost_k[i, j]

    return FnU


@jit(nopython=True, cache=True)
def compute_HJB_matrix_entries(Unew_n_tmp, Mk_np1, omask_arr, door_mask_arr, Nx, Ny, Dx, Dy, Dt):
    """
    Assembles sparse COO matrix for backward-in-time HJB equation linearization.

    Constructs the Jacobian matrix A for Newton's method applied to the nonlinear HJB:
        -∂u/∂t + H(m, ∇u) - ε * Δu = 0
    Linearizing around current iterate gives A * δu = -F(u), where A encodes
    the implicit time discretization, diffusion, and linearized Hamiltonian transport.

    Args:
        Unew_n_tmp: Current Newton iterate at time n, shape (Nx, Ny)
        Mk_np1: Density at time n+1 (fixed during HJB solve), shape (Nx, Ny)
        omask_arr: Obstacle mask (1=walkable, 0=obstacle), shape (Nx, Ny)
        door_mask_arr: Exit door mask (1=exit, 0=interior), shape (Nx, Ny)
        Nx, Ny: Grid dimensions (int)
        Dx, Dy: Grid spacing (float, metres)
        Dt: Time step (float, seconds)

    Returns:
        tuple: (rows, cols, vals) - COO sparse matrix triplets
            rows: Row indices (int64 array)
            cols: Column indices (int64 array)
            vals: Matrix entries (float64 array)

    Note:
        Diffusion coefficient ε = 0.05 hardcoded. Congestion coupling c_h = 16/(1+m)^0.75.
        Matrix uses row-major indexing: (i,j) → i*Ny + j.
    """
    # Pre-allocate COO storage (max 16 entries per point: 1 diagonal + 4 diffusion + up to 11 Hamiltonian terms)
    N_total = Nx * Ny
    max_entries = N_total * 16
    rows = np.zeros(max_entries, dtype=np.int64)
    cols = np.zeros(max_entries, dtype=np.int64)
    vals = np.zeros(max_entries, dtype=np.float64)

    entry_idx = 0

    for i in range(Nx):
        for j in range(Ny):
            ind = i * Ny + j  # Row-major linear index
            # Boundary conditions: exits and obstacles enforce identity row (RHS = 0)
            if door_mask_arr[i, j] == 1 or omask_arr[i, j] == 0:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, 1.0
                entry_idx += 1
                continue

            # Start diagonal accumulator with time derivative term
            diag_val = 1.0 / Dt
            # Ghost nodes for Neumann boundaries
            ip1 = i + 1 if (i < Nx - 1 and omask_arr[i + 1, j] == 1) else i
            im1 = i - 1 if (i > 0      and omask_arr[i - 1, j] == 1) else i
            jp1 = j + 1 if (j < Ny - 1 and omask_arr[i, j + 1] == 1) else j
            jm1 = j - 1 if (j > 0      and omask_arr[i, j - 1] == 1) else j

            # Add diffusion diagonal contributions (ε = 0.05)
            if ip1 != i: diag_val += 0.05 / (Dx ** 2)
            if im1 != i: diag_val += 0.05 / (Dx ** 2)
            if jp1 != j: diag_val += 0.05 / (Dy ** 2)
            if jm1 != j: diag_val += 0.05 / (Dy ** 2)

            # Compute finite differences for Hamiltonian linearization
            p1 = (Unew_n_tmp[ip1, j] - Unew_n_tmp[i, j]) / Dx
            p2 = (Unew_n_tmp[i, j] - Unew_n_tmp[im1, j]) / Dx
            p3 = (Unew_n_tmp[i, jp1] - Unew_n_tmp[i, j]) / Dy
            p4 = (Unew_n_tmp[i, j] - Unew_n_tmp[i, jm1]) / Dy

            # Congestion mobility coefficient
            c_h = 16.0 / ((1.0 + Mk_np1[i, j]) ** 0.75)

            # Write diagonal entry (will be augmented by Hamiltonian terms below)
            rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, diag_val
            entry_idx += 1

            # Diffusion off-diagonal entries: -ε/h² for each valid neighbor
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

            # Hamiltonian linearization: ∂H/∂u adds off-diagonal and diagonal terms
            # For each direction, upwind discretization contributes both neighbor and self entries
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
    """
    Assembles sparse COO matrix for 2-population Fokker-Planck equation.

    Discretizes the coupled KFP system for pursuit-evasion games:
        ∂m/∂t = div(m * ∇H_p(m_self, m_other, ∇u)) + ε * Δm
    where the Hamiltonian depends on both own population density (m_self) and
    the opponent's density (m_other) via asymmetric interaction weights.

    The congestion term c_h = 2/(1 + m_self + 5*m_other) models stronger
    avoidance of the opponent population (5x weight) than own population.

    Args:
        m_prev: Own population density at current time, shape (Nx, Ny)
        m_other: Opponent density at current time, shape (Nx, Ny)
        ukm1: Own value function from previous Picard iteration, shape (Nx, Ny)
        omask_arr: Obstacle mask (1=walkable, 0=obstacle), shape (Nx, Ny)
        Nx, Ny: Grid dimensions (int)
        Dx, Dy: Grid spacing (float, metres)
        Dt: Time step (float, seconds)

    Returns:
        tuple: (rows, cols, vals, b) - COO matrix and RHS vector
            rows, cols: Sparse matrix indices (int64 arrays)
            vals: Matrix entries (float64 array)
            b: Right-hand side vector (float64 array, length Nx*Ny)

    Note:
        Unlike 1-population case, no door_mask - boundary conditions handled by omask only.
        Diffusion coefficient ε = 0.05 hardcoded.
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
    """
    Computes 2-population HJB residual with asymmetric interaction costs.

    Evaluates the discrete HJB equation for pursuit-evasion games:
        F(u) = -∂u/∂t + H(m_self, m_other, ∇u) - ε * Δu + C_interaction(m_self, m_other)
    where the interaction cost C_interaction differs by population:
        - Pursuers (pop=1): +1500*m_other (attracted to evaders)
        - Evaders (pop=2): -5000*m_other (repelled by pursuers)
    Both populations also include congestion penalty: max(m_self + m_other - 4, 0)

    Args:
        Ukp1_np1: Value at next time step n+1, shape (Nx, Ny)
        Ukp1_n: Value at current time step n, shape (Nx, Ny)
        Mk_np1: Own population density at time n+1, shape (Nx, Ny)
        Mk_other: Opponent density at time n+1, shape (Nx, Ny)
        omask_arr: Obstacle mask (1=walkable, 0=obstacle), shape (Nx, Ny)
        Nx, Ny: Grid dimensions (int)
        Dx, Dy: Grid spacing (float, metres)
        Dt: Time step (float, seconds)
        pop: Population identifier (1=pursuer, 2=evader, int)

    Returns:
        ndarray: Residual F(u), shape (Nx, Ny)

    Note:
        Obstacle penalty = -500.0 (negative sign on line 322 differs from 1-pop).
        Diffusion coefficient ε = 0.05 hardcoded.
        Hamiltonian uses c_h = 1/(1 + m_self + 5*m_other).
    """
    FnU = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            # Obstacles: negative potential (note sign difference from 1-pop case)
            if omask_arr[i, j] == 0:
                FnU[i, j] = Ukp1_n[i, j] + 500.0
                continue

            # Ghost nodes for Neumann boundaries
            ip1 = i + 1 if (i < Nx - 1 and omask_arr[i + 1, j] == 1) else i
            im1 = i - 1 if (i > 0      and omask_arr[i - 1, j] == 1) else i
            jp1 = j + 1 if (j < Ny - 1 and omask_arr[i, j + 1] == 1) else j
            jm1 = j - 1 if (j > 0      and omask_arr[i, j - 1] == 1) else j

            # Backward time derivative
            time_deriv = -(Ukp1_np1[i, j] - Ukp1_n[i, j]) / Dt
            # Finite differences for gradient
            p1 = (Ukp1_n[ip1, j] - Ukp1_n[i, j]) / Dx
            p2 = (Ukp1_n[i, j] - Ukp1_n[im1, j]) / Dx
            p3 = (Ukp1_n[i, jp1] - Ukp1_n[i, j]) / Dy
            p4 = (Ukp1_n[i, j] - Ukp1_n[i, jm1]) / Dy

            # Diffusion term: -ε * Δu
            laplacian_x = (Ukp1_n[ip1, j] - 2 * Ukp1_n[i, j] + Ukp1_n[im1, j]) / (Dx ** 2)
            laplacian_y = (Ukp1_n[i, jp1] - 2 * Ukp1_n[i, j] + Ukp1_n[i, jm1]) / (Dy ** 2)
            diffusion = -0.05 * (laplacian_x + laplacian_y)

            # 2-pop Hamiltonian: congestion depends on both populations (5x weight for opponent)
            hamiltonian = (1.0 / (1.0 + Mk_np1[i, j] + 5.0 * Mk_other[i, j])) * (
                npart(p1)**2 + ppart(p2)**2 + npart(p3)**2 + ppart(p4)**2
            )

            # Asymmetric interaction costs: pursuers attracted (+), evaders repelled (-)
            if pop == 1:
                # Pursuers: attracted to evaders, avoid overcrowding
                interaction_cost = +1500.0 * Mk_other[i, j] + ppart(Mk_np1[i, j] + Mk_other[i, j] - 4.0)
            else:
                # Evaders: repelled by pursuers, avoid overcrowding
                interaction_cost = -5000.0 * Mk_other[i, j] + ppart(Mk_np1[i, j] + Mk_other[i, j] - 4.0)

            FnU[i, j] = time_deriv + diffusion + hamiltonian + interaction_cost

    return FnU


@jit(nopython=True, cache=True)
def compute_HJB_matrix_entries_2Pop(Unew_n_tmp, Mk_np1, Mk_other, omask_arr, Nx, Ny, Dx, Dy, Dt):
    """
    Assembles sparse COO matrix for 2-population HJB linearization.

    Constructs the Jacobian matrix A for Newton's method in the 2-population HJB:
        -∂u/∂t + H(m_self, m_other, ∇u) - ε * Δu = 0
    The linearization accounts for coupled congestion effects via c_h = 2/(1 + m_self + 5*m_other).

    Args:
        Unew_n_tmp: Current Newton iterate at time n, shape (Nx, Ny)
        Mk_np1: Own population density at time n+1, shape (Nx, Ny)
        Mk_other: Opponent density at time n+1, shape (Nx, Ny)
        omask_arr: Obstacle mask (1=walkable, 0=obstacle), shape (Nx, Ny)
        Nx, Ny: Grid dimensions (int)
        Dx, Dy: Grid spacing (float, metres)
        Dt: Time step (float, seconds)

    Returns:
        tuple: (rows, cols, vals) - COO sparse matrix triplets
            rows: Row indices (int64 array)
            cols: Column indices (int64 array)
            vals: Matrix entries (float64 array)

    Note:
        No door_mask parameter - exits handled differently in 2-population games.
        Diffusion coefficient ε = 0.05 hardcoded.
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
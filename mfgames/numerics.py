"""
Low-Level Numba JIT-Compiled Finite Difference Operators and Matrix Assembly.

This module implements the unified numerical discretization for Mean Field Game (MFG)
systems, providing high-performance stencils for coupled Hamilton-Jacobi-Bellman (HJB)
and Kolmogorov-Fokker-Planck (KFP) equations. All core operators are compiled using Numba's
`@jit(nopython=True, cache=True)` for near-C execution speeds.

Architecture & Generalization:
-----------------------------
This module provides a unified numerical foundation that generalizes both single-population
crowd navigation and arbitrary N-population multi-swarm interaction games:
- 1-Population Systems: Single crowd navigation, evacuation, and major-minor pursuit-evasion.
- N-Population Systems: Multi-team games with arbitrary cross-congestion and strategic interactions.
- Generalized Mobility: Parameterized mobility coefficients c_h = gamma / (1 + m_self + m_cong)^beta,
  unifying standard congestion (gamma=16.0, beta=0.75) and multi-crowd friction (gamma=2.0, beta=1.0).
- Boundary Conditions:
    * Hard Dirichlet Exit Boundaries: Door masks enforce u = 0 (zero cost-to-go) and m = 0 (absorption).
    * Reflective Neumann Boundaries: Zero-flux wall boundaries via obstacle masks (omask).
    * Soft Running Costs & Potentials: Pre-aggregated spatial potential arrays combining distance costs,
      goal capacity repulsion barriers, and inter-crowd attraction/repulsion.

Mathematical Discretization Schemes:
-----------------------------------
1. Hamilton-Jacobi-Bellman (HJB) Equation:
       -∂u_p/∂t - ν Δu_p + H_p(m_1,...,m_N, ∇u_p) = L_p(x, t)
   - Discretization: Backward-in-time semi-implicit Euler scheme.
   - Viscosity / Diffusion: 5-point central difference Laplacian (-ν Δu with ν = 0.05).
   - Upwind Hamiltonian: Rouy-Tourin monotone upwind numerical Hamiltonian selecting directional
     derivatives based on characteristic flow direction.
   - Nonlinear Solve: Damped Newton-Raphson iteration with sparse Jacobian matrix assembly in COO format.

2. Kolmogorov-Fokker-Planck (KFP) Equation:
       ∂m_p/∂t - ν Δm_p - div( m_p * ∇_p H_p ) = 0
   - Discretization: Forward-in-time implicit Euler scheme.
   - Transport / Advection: Conservative upwind finite volume flux matching the optimal drift field.
   - Matrix Assembly: COO sparse matrix format inverted via scipy.sparse.linalg.spsolve.

Coordinate Indexing Convention:
------------------------------
All 2D spatial fields are indexed as (Nx, Ny) corresponding to spatial coordinates (X, Y).
Sparse matrices use row-major linear indexing:
    linear_index = i * Ny + j,  where i ∈ [0, Nx-1], j ∈ [0, Ny-1].
"""

import numpy as np
from numba import jit


# =============================================================================
# UPWIND DIRECTIONAL OPERATORS
# =============================================================================

@jit(nopython=True, cache=True)
def ppart(x: float) -> float:
    """
    Positive part operator for directional upwind finite differences.

    Extracts the positive component of a scalar difference approximation:
        ppart(x) = max(x, 0.0)

    In the Rouy-Tourin upwind discretization of Hamilton-Jacobi equations,
    this operator isolates forward differences when characteristic trajectories
    propagate along the positive coordinate axis.

    Parameters
    ----------
    x : float
        Scalar finite difference approximation (e.g., (u[i+1, j] - u[i, j]) / Dx).

    Returns
    -------
    float
        The positive part max(x, 0.0). Returns 0.0 if x <= 0.0.
    """
    return np.maximum(x, 0.0)


@jit(nopython=True, cache=True)
def npart(x: float) -> float:
    """
    Negative part operator for directional upwind finite differences.

    Extracts the negative component of a scalar difference approximation as a positive magnitude:
        npart(x) = -min(x, 0.0) = max(-x, 0.0)

    In the Rouy-Tourin upwind discretization of Hamilton-Jacobi equations,
    this operator isolates backward differences when characteristic trajectories
    propagate along the negative coordinate axis.

    Parameters
    ----------
    x : float
        Scalar finite difference approximation (e.g., (u[i, j] - u[i-1, j]) / Dx).

    Returns
    -------
    float
        The absolute magnitude of the negative component -min(x, 0.0). Returns 0.0 if x >= 0.0.
    """
    return -np.minimum(x, 0.0)


# =============================================================================
# GENERALIZED FOKKER-PLANCK (KFP) MATRIX ASSEMBLY
# =============================================================================

@jit(nopython=True, cache=True)
def assemble_FP_matrix(
    m_prev,
    m_cong,
    ukm1,
    omask_arr,
    door_mask_arr,
    Nx: int,
    Ny: int,
    Dx: float,
    Dy: float,
    Dt: float,
    gamma: float = 16.0,
    beta: float = 0.75,
    upwind_mode: int = 0,
):
    """
    Assemble the sparse COO matrix and RHS vector for the forward Kolmogorov-Fokker-Planck equation.

    Discretizes the continuum mass conservation PDE forward in time:
        ∂m/∂t - ν Δm - div( m * c_h(m, m_cong) * ∇u ) = 0

    where ν = 0.05 is the kinetic diffusion coefficient, and the mobility scaling
    c_h(m, m_cong) models macroscopic crowding and cross-population friction:
        c_h(m, m_cong) = gamma / (1.0 + m + m_cong)^beta

    The resulting implicit Euler system:
        A * m^{k} = b
    is assembled in Coordinate (COO) format with a pre-allocated capacity of 9 entries
    per spatial grid node (1 diagonal + 4 diffusion off-diagonals + 4 advective flux off-diagonals).

    Boundary Conditions:
    --------------------
    - Absorbing Exits (door_mask_arr == 1):
        Enforces Dirichlet boundary m(t, x) = 0. Row is set to the identity (ind, ind, 1.0)
        and b[ind] = 0.0, causing exiting agents to be removed from the domain.
    - Obstacles (omask_arr == 0):
        Enforces Dirichlet dead-space m(t, x) = 0. Row is set to the identity (ind, ind, 1.0)
        and b[ind] = 0.0.
    - Walkable Domain Boundary & Obstacle Perimeters:
        Enforces reflective zero-flux Neumann boundary conditions (∇m · n = 0) by clamping
        neighbor stencils to the current node whenever adjacent cells are non-walkable.

    Parameters
    ----------
    m_prev : ndarray of shape (Nx, Ny), dtype float64
        Population density distribution from the previous time step k - 1.
    m_cong : ndarray of shape (Nx, Ny), dtype float64
        Pre-aggregated cross-population congestion field:
            m_cong = sum_{q ≠ p} W_{pq}^{cong} * m_q.
        Pass a zero array of shape (Nx, Ny) for single-population scenarios.
    ukm1 : ndarray of shape (Nx, Ny), dtype float64
        Value function driving the advective drift velocity at time step k - 1.
    omask_arr : ndarray of shape (Nx, Ny), dtype int64
        Binary obstacle mask where 1 indicates walkable space and 0 indicates solid wall.
    door_mask_arr : ndarray of shape (Nx, Ny), dtype float64
        Time-dependent exit mask where 1 indicates active absorbing doors and 0 indicates interior.
    Nx : int
        Number of grid discretization points along the X spatial axis.
    Ny : int
        Number of grid discretization points along the Y spatial axis.
    Dx : float
        Grid cell spatial width in meters (Lx / Nx).
    Dy : float
        Grid cell spatial height in meters (Ly / Ny).
    Dt : float
        Time step increment in seconds (T / Nt).
    gamma : float, optional
        Kinetic mobility coefficient numerator (default: 16.0 for single-crowd/traffic,
        2.0 for competitive two-population games).
    beta : float, optional
        Kinetic mobility congestion exponent denominator (default: 0.75 for single-crowd,
        1.0 for competitive two-population games).
    upwind_mode : int, optional
        Directional convention for advective flux:
            0: Standard gradient advection (used in single-population and pursuit-evasion models).
            1: Reversed gradient advection (used in legacy competitive two-population games).

    Returns
    -------
    rows : ndarray of shape (K,), dtype int64
        Row indices for non-zero entries in the sparse COO matrix.
    cols : ndarray of shape (K,), dtype int64
        Column indices for non-zero entries in the sparse COO matrix.
    vals : ndarray of shape (K,), dtype float64
        Numerical values corresponding to (rows, cols) sparse matrix triplets.
    b : ndarray of shape (Nx * Ny,), dtype float64
        Right-hand side load vector representing m_prev / Dt subject to boundary conditions.

    Notes
    -----
    - Viscosity coefficient ν = 0.05 is hardcoded for numerical stability across grid resolutions.
    - Grid point (i, j) maps to the 1D linear index: ind = i * Ny + j.
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

            # Boundary conditions: absorbing exit doors and solid obstacles enforce m = 0
            if omask_arr[i, j] == 0 or door_mask_arr[i, j] == 1:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, 1.0
                entry_idx += 1
                b[ind] = 0.0
                continue

            # Neighbor stencil indexing with reflective Neumann boundary conditions
            ip1 = i + 1 if (i < Nx - 1 and omask_arr[i + 1, j] == 1 and door_mask_arr[i + 1, j] == 0) else i
            im1 = i - 1 if (i > 0      and omask_arr[i - 1, j] == 1 and door_mask_arr[i - 1, j] == 0) else i
            jp1 = j + 1 if (j < Ny - 1 and omask_arr[i, j + 1] == 1 and door_mask_arr[i, j + 1] == 0) else j
            jm1 = j - 1 if (j > 0      and omask_arr[i, j - 1] == 1 and door_mask_arr[i, j - 1] == 0) else j

            # Upwind spatial derivatives of value function u
            p1 = (ukm1[ip1, j] - ukm1[i, j]) / Dx
            p2 = (ukm1[i, j] - ukm1[im1, j]) / Dx
            p3 = (ukm1[i, jp1] - ukm1[i, j]) / Dy
            p4 = (ukm1[i, j] - ukm1[i, jm1]) / Dy

            # Diagonal accumulator initialized with time derivative contribution
            diag_val = 1.0 / Dt

            # Central difference diffusion contributions (ν = 0.05)
            if ip1 != i: diag_val += 0.05 / (Dx ** 2)
            if im1 != i: diag_val += 0.05 / (Dx ** 2)
            if jp1 != j: diag_val += 0.05 / (Dy ** 2)
            if jm1 != j: diag_val += 0.05 / (Dy ** 2)

            # Congestion-dependent mobility coefficient
            denom = 1.0 + m_prev[i, j] + m_cong[i, j]
            c_h = gamma / (denom ** beta) if denom > 0.0 else gamma

            # Upwind advection contribution to the diagonal
            if upwind_mode == 0:
                diag_val += c_h * (ppart(p1) / Dx + npart(p2) / Dx + ppart(p3) / Dy + npart(p4) / Dy)
            else:
                diag_val += c_h * (npart(p1) / Dx + ppart(p2) / Dx + npart(p3) / Dy + ppart(p4) / Dy)

            rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, diag_val
            entry_idx += 1

            # Diffusion off-diagonal entries (-ν / h^2)
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

            # Upwind transport off-diagonal entries representing flux divergence from neighbors
            if upwind_mode == 0:
                if i > 0 and omask_arr[i - 1, j] == 1 and door_mask_arr[i - 1, j] == 0:
                    denom_l = 1.0 + m_prev[i - 1, j] + m_cong[i - 1, j]
                    c_l = gamma / (denom_l ** beta) if denom_l > 0.0 else gamma
                    vals[entry_idx] = -c_l * ppart((ukm1[i, j] - ukm1[i - 1, j]) / Dx) / Dx
                    rows[entry_idx], cols[entry_idx] = ind, ((i - 1) * Ny + j)
                    entry_idx += 1
                if i < Nx - 1 and omask_arr[i + 1, j] == 1 and door_mask_arr[i + 1, j] == 0:
                    denom_r = 1.0 + m_prev[i + 1, j] + m_cong[i + 1, j]
                    c_r = gamma / (denom_r ** beta) if denom_r > 0.0 else gamma
                    vals[entry_idx] = -c_r * npart((ukm1[i + 1, j] - ukm1[i, j]) / Dx) / Dx
                    rows[entry_idx], cols[entry_idx] = ind, ((i + 1) * Ny + j)
                    entry_idx += 1
                if j > 0 and omask_arr[i, j - 1] == 1 and door_mask_arr[i, j - 1] == 0:
                    denom_b = 1.0 + m_prev[i, j - 1] + m_cong[i, j - 1]
                    c_b = gamma / (denom_b ** beta) if denom_b > 0.0 else gamma
                    vals[entry_idx] = -c_b * ppart((ukm1[i, j] - ukm1[i, j - 1]) / Dy) / Dy
                    rows[entry_idx], cols[entry_idx] = ind, (i * Ny + j - 1)
                    entry_idx += 1
                if j < Ny - 1 and omask_arr[i, j + 1] == 1 and door_mask_arr[i, j + 1] == 0:
                    denom_t = 1.0 + m_prev[i, j + 1] + m_cong[i, j + 1]
                    c_t = gamma / (denom_t ** beta) if denom_t > 0.0 else gamma
                    vals[entry_idx] = -c_t * npart((ukm1[i, j + 1] - ukm1[i, j]) / Dy) / Dy
                    rows[entry_idx], cols[entry_idx] = ind, (i * Ny + j + 1)
                    entry_idx += 1
            else:
                if i > 0 and omask_arr[i - 1, j] == 1 and door_mask_arr[i - 1, j] == 0:
                    denom_l = 1.0 + m_prev[i - 1, j] + m_cong[i - 1, j]
                    c_l = gamma / (denom_l ** beta) if denom_l > 0.0 else gamma
                    vals[entry_idx] = -c_l * npart((ukm1[i, j] - ukm1[i - 1, j]) / Dx) / Dx
                    rows[entry_idx], cols[entry_idx] = ind, ((i - 1) * Ny + j)
                    entry_idx += 1
                if i < Nx - 1 and omask_arr[i + 1, j] == 1 and door_mask_arr[i + 1, j] == 0:
                    denom_r = 1.0 + m_prev[i + 1, j] + m_cong[i + 1, j]
                    c_r = gamma / (denom_r ** beta) if denom_r > 0.0 else gamma
                    vals[entry_idx] = -c_r * ppart((ukm1[i + 1, j] - ukm1[i, j]) / Dx) / Dx
                    rows[entry_idx], cols[entry_idx] = ind, ((i + 1) * Ny + j)
                    entry_idx += 1
                if j > 0 and omask_arr[i, j - 1] == 1 and door_mask_arr[i, j - 1] == 0:
                    denom_b = 1.0 + m_prev[i, j - 1] + m_cong[i, j - 1]
                    c_b = gamma / (denom_b ** beta) if denom_b > 0.0 else gamma
                    vals[entry_idx] = -c_b * npart((ukm1[i, j] - ukm1[i, j - 1]) / Dy) / Dy
                    rows[entry_idx], cols[entry_idx] = ind, (i * Ny + j - 1)
                    entry_idx += 1
                if j < Ny - 1 and omask_arr[i, j + 1] == 1 and door_mask_arr[i, j + 1] == 0:
                    denom_t = 1.0 + m_prev[i, j + 1] + m_cong[i, j + 1]
                    c_t = gamma / (denom_t ** beta) if denom_t > 0.0 else gamma
                    vals[entry_idx] = -c_t * ppart((ukm1[i, j + 1] - ukm1[i, j]) / Dy) / Dy
                    rows[entry_idx], cols[entry_idx] = ind, (i * Ny + j + 1)
                    entry_idx += 1

    return rows[:entry_idx], cols[:entry_idx], vals[:entry_idx], b


# =============================================================================
# GENERALIZED HJB RESIDUAL EVALUATION
# =============================================================================

@jit(nopython=True, cache=True)
def evaluate_HJB_residual(
    Unew_np1,
    Unew_n,
    m_self,
    m_cong,
    omask_arr,
    door_mask_arr,
    total_cost_arr,
    Nx: int,
    Ny: int,
    Dx: float,
    Dy: float,
    Dt: float,
    obstacle_penalty: float = -500.0,
    alpha: float = 8.0,
    beta: float = 0.75,
    h0: float = 0.1,
    upwind_mode: int = 0,
):
    """
    Evaluate the discrete nonlinear HJB residual matrix F(u) across all spatial grid nodes.

    Discretizes the backward-in-time Hamilton-Jacobi-Bellman equation:
        F(u) = -∂u/∂t - ν Δu + H(m, m_cong, ∇u) + L_total(x) = 0

    where:
    - Time derivative: Backward difference -∂u/∂t ≈ -(u^{n+1} - u^n) / Dt.
    - Viscosity / Diffusion: -ν Δu discretized using a 5-point central stencil with ν = 0.05.
    - Hamiltonian H: Congestion-dependent kinetic energy modeled via directional upwind differences:
        Mode 0 (standard): H = -alpha / (1 + m_self + m_cong)^beta * ||∇u||_upwind^2 + h0
        Mode 1 (reversed): H = +alpha / (1 + m_self + m_cong)^beta * ||∇u||_upwind^2 + h0
    - Total Cost L_total(x): Pre-accumulated spatial field incorporating running distance costs,
      repulsive barriers around saturated goals, and cross-crowd strategic interactions.

    Boundary Conditions:
    --------------------
    - Absorbing Exits (door_mask_arr == 1):
        Enforces Dirichlet condition u(t, x) = 0. Residual evaluates to:
            F(u)[i, j] = Unew_n[i, j] - 0.0.
    - Solid Obstacles (omask_arr == 0):
        Enforces potential barrier to prevent crowd entry. Residual evaluates to:
            F(u)[i, j] = Unew_n[i, j] - obstacle_penalty.
    - Walkable Domain Boundary:
        Reflective zero-flux Neumann boundary conditions implemented via ghost-node clamping.

    Parameters
    ----------
    Unew_np1 : ndarray of shape (Nx, Ny), dtype float64
        Value function array at future time step n + 1 (known from backward stepping).
    Unew_n : ndarray of shape (Nx, Ny), dtype float64
        Current Newton iterate for the value function at time step n.
    m_self : ndarray of shape (Nx, Ny), dtype float64
        Own crowd population density at time step n + 1.
    m_cong : ndarray of shape (Nx, Ny), dtype float64
        Pre-aggregated other-crowd congestion field at time step n + 1:
            m_cong = sum_{q ≠ p} W_{pq}^{cong} * m_q.
        Pass zeros for single-population scenarios.
    omask_arr : ndarray of shape (Nx, Ny), dtype int64
        Binary obstacle mask (1 for walkable, 0 for obstacle).
    door_mask_arr : ndarray of shape (Nx, Ny), dtype float64
        Exit door mask (1 for absorbing exit, 0 for interior).
    total_cost_arr : ndarray of shape (Nx, Ny), dtype float64
        Unified 2D spatial cost field pre-summing running distance penalties, saturated goal
        repulsion barriers, and inter-population strategic interaction potentials.
    Nx : int
        Grid resolution along the X-axis.
    Ny : int
        Grid resolution along the Y-axis.
    Dx : float
        Grid cell spatial width in meters.
    Dy : float
        Grid cell spatial height in meters.
    Dt : float
        Time step increment in seconds.
    obstacle_penalty : float, optional
        Potential value enforced at obstacle nodes (default: -500.0).
    alpha : float, optional
        Kinetic energy mobility coefficient (default: 8.0 for 1-pop, 1.0 for competitive 2-pop).
    beta : float, optional
        Crowd congestion exponent (default: 0.75 for 1-pop, 1.0 for competitive 2-pop).
    h0 : float, optional
        Baseline control expenditure offset (default: 0.1 for 1-pop, 0.0 for 2-pop).
    upwind_mode : int, optional
        Upwind directional selection mode (0 for standard gradient flow, 1 for reversed).

    Returns
    -------
    FnU : ndarray of shape (Nx, Ny), dtype float64
        Nonlinear HJB residual matrix. A converged Newton step satisfies ||FnU|| < tolerance.
    """
    FnU = np.zeros((Nx, Ny), dtype=np.float64)

    for i in range(Nx):
        for j in range(Ny):
            # Exit doors enforce Dirichlet condition u = 0
            if door_mask_arr[i, j] == 1:
                FnU[i, j] = Unew_n[i, j] - 0.0
                continue

            # Obstacles enforce potential barrier
            if omask_arr[i, j] == 0:
                FnU[i, j] = Unew_n[i, j] - obstacle_penalty
                continue

            # Ghost node clamping for reflective Neumann boundaries
            ip1 = i + 1 if (i < Nx - 1 and omask_arr[i + 1, j] == 1 and door_mask_arr[i + 1, j] == 0) else i
            im1 = i - 1 if (i > 0      and omask_arr[i - 1, j] == 1 and door_mask_arr[i - 1, j] == 0) else i
            jp1 = j + 1 if (j < Ny - 1 and omask_arr[i, j + 1] == 1 and door_mask_arr[i, j + 1] == 0) else j
            jm1 = j - 1 if (j > 0      and omask_arr[i, j - 1] == 1 and door_mask_arr[i, j - 1] == 0) else j

            # Backward temporal derivative: -∂u/∂t ≈ -(u^{n+1} - u^n) / Dt
            time_deriv = -(Unew_np1[i, j] - Unew_n[i, j]) / Dt

            # Directional spatial finite differences
            p1 = (Unew_n[ip1, j] - Unew_n[i, j]) / Dx
            p2 = (Unew_n[i, j] - Unew_n[im1, j]) / Dx
            p3 = (Unew_n[i, jp1] - Unew_n[i, j]) / Dy
            p4 = (Unew_n[i, j] - Unew_n[i, jm1]) / Dy

            # Viscous diffusion term: -ν Δu (5-point Laplacian stencil with ν = 0.05)
            laplacian_x = (Unew_n[ip1, j] - 2.0 * Unew_n[i, j] + Unew_n[im1, j]) / (Dx ** 2)
            laplacian_y = (Unew_n[i, jp1] - 2.0 * Unew_n[i, j] + Unew_n[i, jm1]) / (Dy ** 2)
            diffusion = -0.05 * (laplacian_x + laplacian_y)

            # Congestion-dependent mobility denominator
            denom = 1.0 + m_self[i, j] + m_cong[i, j]
            coeff = alpha / (denom ** beta) if denom > 0.0 else alpha

            # Monotone numerical Hamiltonian evaluation
            if upwind_mode == 0:
                grad_norm_sq = ppart(p1)**2 + npart(p2)**2 + ppart(p3)**2 + npart(p4)**2
                hamiltonian = -coeff * grad_norm_sq + h0
            else:
                grad_norm_sq = npart(p1)**2 + ppart(p2)**2 + npart(p3)**2 + ppart(p4)**2
                hamiltonian = coeff * grad_norm_sq + h0

            # Full HJB residual: F(u) = -∂u/∂t - ν Δu + H(m, ∇u) + L_total(x)
            FnU[i, j] = time_deriv + diffusion + hamiltonian + total_cost_arr[i, j]

    return FnU


# =============================================================================
# GENERALIZED HJB NEWTON JACOBIAN MATRIX ASSEMBLY
# =============================================================================

@jit(nopython=True, cache=True)
def assemble_HJB_jacobian(
    Unew_n_tmp,
    m_self,
    m_cong,
    omask_arr,
    door_mask_arr,
    Nx: int,
    Ny: int,
    Dx: float,
    Dy: float,
    Dt: float,
    alpha: float = 8.0,
    beta: float = 0.75,
    upwind_mode: int = 0,
):
    """
    Assemble the sparse COO Newton Jacobian matrix A = ∂F/∂u for the backward HJB equation.

    Linearizes the nonlinear discrete HJB residual operator F(u) around the current Newton
    iterate u^n to construct the linear correction system:
        A * δu = -F(u)
        u^{n, new} = u^n + δu

    Matrix Structure and Pre-allocation:
    -----------------------------------
    The Jacobian is assembled in Coordinate (COO) format with a pre-allocated capacity of
    16 entries per grid node (1 time derivative diagonal + 4 diffusion off-diagonals +
    up to 8 directional advective derivative entries + self-node advection diagonals).

    Linearization Breakdown:
    ------------------------
    1. Diagonal Entry (ind, ind):
       - Time derivative: +1.0 / Dt
       - Viscous diffusion: +2*ν/Dx^2 + 2*ν/Dy^2 for each walkable neighbor
       - Upwind Hamiltonian: +c_h * [ |p_x| / Dx + |p_y| / Dy ]
    2. Off-Diagonal Entries:
       - Viscous diffusion: -ν/Dx^2 and -ν/Dy^2
       - Upwind Hamiltonian: Directional derivatives -c_h * ppart(p) / Dx, etc.
    3. Boundary Rows:
       - Exit doors and obstacle cells enforce identity rows: A[ind, ind] = 1.0, all off-diagonals 0.

    Parameters
    ----------
    Unew_n_tmp : ndarray of shape (Nx, Ny), dtype float64
        Current Newton iterate for the value function u at time step n.
    m_self : ndarray of shape (Nx, Ny), dtype float64
        Own population density at time step n + 1 (fixed during the backward HJB sweep).
    m_cong : ndarray of shape (Nx, Ny), dtype float64
        Pre-aggregated cross-population congestion field from other crowds.
    omask_arr : ndarray of shape (Nx, Ny), dtype int64
        Binary obstacle mask (1 for walkable, 0 for obstacle).
    door_mask_arr : ndarray of shape (Nx, Ny), dtype float64
        Exit door mask (1 for absorbing exit, 0 for interior).
    Nx : int
        Grid resolution along the X-axis.
    Ny : int
        Grid resolution along the Y-axis.
    Dx : float
        Spatial step size along the X-axis in meters.
    Dy : float
        Spatial step size along the Y-axis in meters.
    Dt : float
        Time step size in seconds.
    alpha : float, optional
        Kinetic energy mobility coefficient (default: 8.0 for 1-pop, 1.0 for competitive 2-pop).
    beta : float, optional
        Crowd congestion exponent (default: 0.75 for 1-pop, 1.0 for competitive 2-pop).
    upwind_mode : int, optional
        Directional selection mode matching evaluate_HJB_residual.

    Returns
    -------
    rows : ndarray of shape (K,), dtype int64
        Row indices for non-zero entries in the sparse Jacobian matrix.
    cols : ndarray of shape (K,), dtype int64
        Column indices for non-zero entries in the sparse Jacobian matrix.
    vals : ndarray of shape (K,), dtype float64
        Numerical values corresponding to (rows, cols) sparse Jacobian matrix triplets.
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

            # Boundary conditions: exit doors and obstacles enforce identity rows
            if door_mask_arr[i, j] == 1 or omask_arr[i, j] == 0:
                rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, 1.0
                entry_idx += 1
                continue

            # Diagonal accumulator initialized with time derivative contribution
            diag_val = 1.0 / Dt

            # Ghost node clamping for reflective Neumann boundaries
            ip1 = i + 1 if (i < Nx - 1 and omask_arr[i + 1, j] == 1 and door_mask_arr[i + 1, j] == 0) else i
            im1 = i - 1 if (i > 0      and omask_arr[i - 1, j] == 1 and door_mask_arr[i - 1, j] == 0) else i
            jp1 = j + 1 if (j < Ny - 1 and omask_arr[i, j + 1] == 1 and door_mask_arr[i, j + 1] == 0) else j
            jm1 = j - 1 if (j > 0      and omask_arr[i, j - 1] == 1 and door_mask_arr[i, j - 1] == 0) else j

            # Diffusion diagonal contributions (ν = 0.05)
            if ip1 != i: diag_val += 0.05 / (Dx ** 2)
            if im1 != i: diag_val += 0.05 / (Dx ** 2)
            if jp1 != j: diag_val += 0.05 / (Dy ** 2)
            if jm1 != j: diag_val += 0.05 / (Dy ** 2)

            # Spatial finite differences of current value function iterate
            p1 = (Unew_n_tmp[ip1, j] - Unew_n_tmp[i, j]) / Dx
            p2 = (Unew_n_tmp[i, j] - Unew_n_tmp[im1, j]) / Dx
            p3 = (Unew_n_tmp[i, jp1] - Unew_n_tmp[i, j]) / Dy
            p4 = (Unew_n_tmp[i, j] - Unew_n_tmp[i, jm1]) / Dy

            # Linearized Hamiltonian derivative coefficient: c_h = 2 * alpha / (1 + m)^beta
            denom = 1.0 + m_self[i, j] + m_cong[i, j]
            c_h = (2.0 * alpha) / (denom ** beta) if denom > 0.0 else (2.0 * alpha)

            # Write baseline diagonal entry (augmented by Hamiltonian contributions below)
            rows[entry_idx], cols[entry_idx], vals[entry_idx] = ind, ind, diag_val
            entry_idx += 1

            # Diffusion off-diagonal entries (-ν / h^2)
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

            # Linearized Hamiltonian transport entries: ∂H/∂u contributes both neighbor and self entries
            if upwind_mode == 0:
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
            else:
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


# =============================================================================
# BACKWARD COMPATIBILITY FUNCTION ALIASES
# =============================================================================

@jit(nopython=True, cache=True)
def compute_FP_matrix_entries(m_prev, ukm1, omask_arr, door_mask_arr, Nx, Ny, Dx, Dy, Dt):
    """Legacy alias routing single-population KFP assembly to assemble_FP_matrix."""
    m_cong_zeros = np.zeros((Nx, Ny), dtype=np.float64)
    return assemble_FP_matrix(
        m_prev, m_cong_zeros, ukm1, omask_arr, door_mask_arr,
        Nx, Ny, Dx, Dy, Dt, gamma=16.0, beta=0.75, upwind_mode=0
    )


@jit(nopython=True, cache=True)
def getFnU_2D(Unew_np1, Unew_n, Mk_np1, omask_arr, door_mask_n, running_cost_k, Nx, Ny, Dx, Dy, Dt, obstacle_penalty=-500.0):
    """Legacy alias routing single-population HJB residual evaluation to evaluate_HJB_residual."""
    m_cong_zeros = np.zeros((Nx, Ny), dtype=np.float64)
    return evaluate_HJB_residual(
        Unew_np1, Unew_n, Mk_np1, m_cong_zeros, omask_arr, door_mask_n, running_cost_k,
        Nx, Ny, Dx, Dy, Dt, obstacle_penalty=obstacle_penalty, alpha=8.0, beta=0.75, h0=0.1, upwind_mode=0
    )


@jit(nopython=True, cache=True)
def compute_HJB_matrix_entries(Unew_n_tmp, Mk_np1, omask_arr, door_mask_arr, Nx, Ny, Dx, Dy, Dt):
    """Legacy alias routing single-population HJB Jacobian assembly to assemble_HJB_jacobian."""
    m_cong_zeros = np.zeros((Nx, Ny), dtype=np.float64)
    return assemble_HJB_jacobian(
        Unew_n_tmp, Mk_np1, m_cong_zeros, omask_arr, door_mask_arr,
        Nx, Ny, Dx, Dy, Dt, alpha=8.0, beta=0.75, upwind_mode=0
    )


@jit(nopython=True, cache=True)
def compute_FP_matrix_entries_2Pop(m_prev, m_other, ukm1, omask_arr, Nx, Ny, Dx, Dy, Dt):
    """Legacy alias routing 2-population KFP assembly to assemble_FP_matrix with 5x opponent congestion."""
    no_door_mask = np.zeros((Nx, Ny), dtype=np.float64)
    m_cong = 5.0 * m_other
    return assemble_FP_matrix(
        m_prev, m_cong, ukm1, omask_arr, no_door_mask,
        Nx, Ny, Dx, Dy, Dt, gamma=2.0, beta=1.0, upwind_mode=1
    )


@jit(nopython=True, cache=True)
def getFnU_2D_2Pop(Ukp1_np1, Ukp1_n, Mk_np1, Mk_other, omask_arr, Nx, Ny, Dx, Dy, Dt, pop=1):
    """Legacy alias routing 2-population HJB residual evaluation to evaluate_HJB_residual."""
    no_door_mask = np.zeros((Nx, Ny), dtype=np.float64)
    m_cong = 5.0 * Mk_other
    if pop == 1:
        interaction_cost = 1500.0 * Mk_other + ppart(Mk_np1 + Mk_other - 4.0)
    else:
        interaction_cost = -5000.0 * Mk_other + ppart(Mk_np1 + Mk_other - 4.0)

    return evaluate_HJB_residual(
        Ukp1_np1, Ukp1_n, Mk_np1, m_cong, omask_arr, no_door_mask, interaction_cost,
        Nx, Ny, Dx, Dy, Dt, obstacle_penalty=-500.0, alpha=1.0, beta=1.0, h0=0.0, upwind_mode=1
    )


@jit(nopython=True, cache=True)
def compute_HJB_matrix_entries_2Pop(Unew_n_tmp, Mk_np1, Mk_other, omask_arr, Nx, Ny, Dx, Dy, Dt):
    """Legacy alias routing 2-population HJB Jacobian assembly to assemble_HJB_jacobian."""
    no_door_mask = np.zeros((Nx, Ny), dtype=np.float64)
    m_cong = 5.0 * Mk_other
    return assemble_HJB_jacobian(
        Unew_n_tmp, Mk_np1, m_cong, omask_arr, no_door_mask,
        Nx, Ny, Dx, Dy, Dt, alpha=1.0, beta=1.0, upwind_mode=1
    )
"""
PDE solvers for Fokker-Planck (forward) and HJB (backward) coupled system.

This module provides high-level solver routines for the mean field game (MFG)
partial differential equation system. The MFG consists of two coupled equations:

1. Hamilton-Jacobi-Bellman (HJB) equation: Solved backward in time for the value
   function u(t,x), representing optimal cost-to-go for agents at position x.

2. Kolmogorov-Fokker-Planck (KFP) equation: Solved forward in time for the
   population density m(t,x), describing how the agent distribution evolves.

These equations are coupled through:
- The density m appears in the HJB Hamiltonian (congestion effects)
- The value gradient ∇u drives the transport in the KFP equation (optimal control)

Boundary conditions:
- Door masks: Hard Dirichlet conditions (u = 0 at exits, m = 0 at exits)
- Obstacles: Neumann conditions (zero normal derivative) or penalty potentials
- Running costs: Soft attraction/repulsion fields added to the Hamiltonian

Numerical method:
- Implicit finite difference discretization for both equations
- Newton iteration for HJB nonlinearity (Hamiltonian is nonlinear in ∇u)
- Sparse linear system solvers (scipy.sparse.linalg.spsolve)

See notes.tex for full mathematical formulation and discretization details.
"""
import time
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
from .numerics import (
    compute_FP_matrix_entries,
    compute_HJB_matrix_entries,
    getFnU_2D,
)


def solveFP_2D(m0, u, door_mask_3d, omask, Nx, Ny, Nt, Dx, Dy, Dt):
    """
    Solve the forward Kolmogorov-Fokker-Planck (KFP) equation for population density evolution.

    The KFP equation describes how the agent population density m(t,x) evolves forward in time
    under optimal control guided by the value function u(t,x). The discretized system is:

        (1/Δt)(m^{k+1} - m^k) = ∇·(m^{k+1} H_p(∇u^k, m^k)) + ν Δm^{k+1}

    where H_p is the Hamiltonian gradient with respect to momentum (optimal velocity field)
    and ν is the diffusion coefficient. The implicit scheme ensures stability for large Δt.

    Boundary conditions:
    - Door cells: m = 0 (agents exit and are removed from the system)
    - Obstacles: m = 0 (impenetrable barriers)
    - Domain edges: Neumann conditions (handled via one-sided differences in matrix assembly)

    Args:
        m0 (np.ndarray): Initial density distribution, shape (Nx, Ny). Should integrate to
            total population mass (typically normalized to 1.0).
        u (np.ndarray): Value function trajectory, shape (Nt+1, Nx, Ny). Provides the
            gradient field ∇u that determines optimal agent velocities.
        door_mask_3d (np.ndarray): Time-dependent door indicator, shape (Nt+1, Nx, Ny).
            door_mask[k,i,j] = 1 indicates an exit cell (enforces m = 0).
        omask (np.ndarray): Obstacle mask, shape (Nx, Ny). omask[i,j] = 0 for obstacles,
            omask[i,j] = 1 for walkable cells.
        Nx (int): Number of grid points in x-direction.
        Ny (int): Number of grid points in y-direction.
        Nt (int): Number of time steps.
        Dx (float): Spatial step size in x-direction (metres).
        Dy (float): Spatial step size in y-direction (metres).
        Dt (float): Time step size (seconds).

    Returns:
        np.ndarray: Population density trajectory m, shape (Nt+1, Nx, Ny). m[k,i,j]
            is the density at time step k and spatial location (i,j). Total mass
            may decrease over time as agents exit through doors.

    Notes:
        - The transport operator is upwinded based on the sign of ∇u to maintain
          stability and prevent spurious oscillations.
        - Congestion effects enter through the density-dependent coefficient in H_p,
          implemented as 16/(1+m)^{0.75} in the current Hamiltonian formulation.
        - The sparse linear system A·m^{k+1} = b is solved via direct LU factorization
          (spsolve), which is efficient for 2D grids up to ~100×100 resolution.
    """
    N_total = Nx * Ny  # Total number of spatial grid points (flattened)
    m = np.zeros((Nt + 1, Nx, Ny))
    m[0] = m0  # Set initial condition

    # Forward march in time: solve for m^{k+1} given m^k and u^k
    for k in range(1, Nt + 1):
        # Assemble sparse matrix A and RHS vector b for implicit time step
        rows, cols, vals, b = compute_FP_matrix_entries(
            m[k - 1], u[k - 1], omask, door_mask_3d[k], Nx, Ny, Dx, Dy, Dt
        )
        # Build sparse matrix in COO format (efficient for construction), convert to CSR (efficient for solve)
        A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()
        # Solve linear system A·m^{k+1} = b and reshape to 2D grid
        m[k] = sparse.linalg.spsolve(A, b).reshape((Nx, Ny))

    return m


def solveHJB_withM(Uk, Mk, door_mask_3d, omask, running_cost_history, Nx, Ny, Nt, Dx, Dy, Dt, NiterNewton=30, l2errBoundNewton=1e-6):
    """
    Solve the backward Hamilton-Jacobi-Bellman (HJB) equation for the value function.

    The HJB equation describes the optimal cost-to-go u(t,x) for an agent at position x
    and time t. It is solved backward in time from terminal condition u(T,x) = 0 at exits
    (or -running_cost(T,x) if soft target costs are present). The discretized system is:

        -(u^n - u^{n+1})/Δt + H(x, m^{n+1}, ∇u^n) - ν Δu^n + L(x) = 0

    where:
    - H is the congestive Hamiltonian: H(x,m,p) = -(8/(1+m)^{0.75})|p|^2 + c_0
    - ν Δu is the viscosity/diffusion term (small positive coefficient)
    - L(x) is the running cost (soft attraction to targets)

    The Hamiltonian is nonlinear in ∇u, requiring Newton iteration at each time step.
    The linearization is derived via the discrete Hamiltonian gradient formulation.

    Boundary conditions:
    - Door cells: u = 0 (Dirichlet - zero cost-to-go at exits)
    - Obstacles: u = +500 (large penalty to prevent entry)
    - Domain edges: Neumann conditions (handled via one-sided differences in matrix assembly)

    Args:
        Uk (np.ndarray): Previous value function iterate, shape (Nt+1, Nx, Ny). Used as
            initial guess; typically passed from previous Picard iteration.
        Mk (np.ndarray): Population density trajectory, shape (Nt+1, Nx, Ny). The density
            m^{n+1} at time n+1 enters the Hamiltonian at time n (backward scheme).
        door_mask_3d (np.ndarray): Time-dependent door indicator, shape (Nt+1, Nx, Ny).
            door_mask[n,i,j] = 1 enforces u[n,i,j] = 0 (exit cell).
        omask (np.ndarray): Obstacle mask, shape (Nx, Ny). omask[i,j] = 0 enforces
            u[i,j] = 500 (obstacle penalty).
        running_cost_history (np.ndarray or None): Time-dependent spatial cost field,
            shape (Nt+1, Nx, Ny). Positive values attract agents (reduce cost), negative
            values repel. If None, no running costs are applied.
        Nx (int): Number of grid points in x-direction.
        Ny (int): Number of grid points in y-direction.
        Nt (int): Number of time steps.
        Dx (float): Spatial step size in x-direction (metres).
        Dy (float): Spatial step size in y-direction (metres).
        Dt (float): Time step size (seconds).
        NiterNewton (int, optional): Maximum Newton iterations per time step. Default 30.
        l2errBoundNewton (float, optional): L2 convergence tolerance for Newton iteration.
            Default 1e-6. Iteration stops when ||u^{k+1} - u^k||_L2 < l2errBoundNewton.

    Returns:
        np.ndarray: Value function trajectory Unew, shape (Nt+1, Nx, Ny). Unew[n,i,j]
            is the optimal cost-to-go from position (i,j) at time step n to the terminal
            condition (typically reaching an exit).

    Notes:
        - Newton iteration solves the nonlinear system F(u^n) = 0, where F is the discrete
          HJB residual. The Jacobian matrix A = ∂F/∂u is assembled via compute_HJB_matrix_entries.
        - The upwind discretization of the Hamiltonian (using ppart/npart operators) ensures
          that the transport operator is properly oriented based on gradient sign.
        - Obstacle penalty of +500 creates a steep potential barrier, effectively making
          obstacles impenetrable in the optimal control synthesis.
        - Convergence is measured in discrete L2 norm scaled by √(Dx·Dy) to approximate
          the continuous L2 norm.
    """
    N_total = Nx * Ny  # Total number of spatial grid points (flattened)
    Unew = np.zeros_like(Uk)

    # Terminal condition at t = T (final time step)
    if running_cost_history is not None:
        # If running costs present, terminal cost is -L(T,x) (accumulate soft target attraction)
        Unew[Nt] = -running_cost_history[Nt]
    else:
        # Otherwise, u(T,x) = 0 everywhere except at exits (handled by door_mask)
        Unew[Nt] = np.zeros((Nx, Ny))

    # Backward march in time: solve for u^n given u^{n+1} and m^{n+1}
    for n in range(Nt - 1, -1, -1):
        # Extract running cost at current time step (or zero field if not provided)
        rc_n = running_cost_history[n] if running_cost_history is not None else np.zeros((Nx, Ny))
        # Initialize Newton iteration with value from next time step (good initial guess)
        Unew_n = np.copy(Unew[n + 1])

        # Newton iteration to resolve Hamiltonian nonlinearity
        for _ in range(NiterNewton):
            # Compute discrete HJB residual F(u^n) at current iterate
            FnU_flat = getFnU_2D(
                Unew[n + 1],  # u^{n+1} (known from previous backward step)
                Unew_n,       # u^n (current iterate)
                Mk[n + 1],    # m^{n+1} (density from coupled KFP equation)
                omask,
                door_mask_3d[n],
                rc_n,
                Nx, Ny, Dx, Dy, Dt
            ).flatten()

            # Assemble Jacobian matrix A = ∂F/∂u^n (linearization of HJB operator)
            rows, cols, vals = compute_HJB_matrix_entries(
                Unew_n, Mk[n + 1], omask, door_mask_3d[n], Nx, Ny, Dx, Dy, Dt
            )
            A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()

            # Newton step: solve A·(u^{k+1} - u^k) = -F(u^k), rearranged as A·u^{k+1} = A·u^k - F
            b = A.dot(Unew_n.flatten()) - FnU_flat

            # Apply boundary conditions by modifying RHS vector
            for i in range(Nx):
                for j in range(Ny):
                    ind = i * Ny + j
                    if door_mask_3d[n, i, j] == 1:
                        # Door cell: u = 0 (zero cost-to-go at exit)
                        b[ind] = 0.0
                    elif omask[i, j] == 0:
                        # Obstacle cell: u = +500 (large penalty to deter entry)
                        b[ind] = 500.0

            # Solve linear system A·u^{k+1} = b for next Newton iterate
            Unres = sparse.linalg.spsolve(A, b).reshape((Nx, Ny))

            # Check convergence: discrete L2 error between successive iterates
            l2err = np.linalg.norm(Unew_n.flatten() - Unres.flatten()) * np.sqrt(Dx * Dy)
            Unew_n = np.copy(Unres)

            # Exit Newton loop if converged
            if l2err < l2errBoundNewton:
                break

        # Store converged solution for time step n
        Unew[n] = Unew_n

    return Unew
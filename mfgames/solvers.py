"""
PDE solvers for Fokker-Planck (forward) and HJB (backward) coupled system.
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
    N_total = Nx * Ny
    m = np.zeros((Nt + 1, Nx, Ny))
    m[0] = m0

    for k in range(1, Nt + 1):
        rows, cols, vals, b = compute_FP_matrix_entries(
            m[k - 1], u[k - 1], omask, door_mask_3d[k], Nx, Ny, Dx, Dy, Dt
        )
        A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()
        m[k] = sparse.linalg.spsolve(A, b).reshape((Nx, Ny))

    return m


def solveHJB_withM(Uk, Mk, door_mask_3d, omask, running_cost_history, Nx, Ny, Nt, Dx, Dy, Dt, NiterNewton=30, l2errBoundNewton=1e-6):
    N_total = Nx * Ny
    Unew = np.zeros_like(Uk)
    
    if running_cost_history is not None:
        Unew[Nt] = -running_cost_history[Nt]
    else:
        Unew[Nt] = np.zeros((Nx, Ny))

    for n in range(Nt - 1, -1, -1):
        rc_n = running_cost_history[n] if running_cost_history is not None else np.zeros((Nx, Ny))
        Unew_n = np.copy(Unew[n + 1])

        for _ in range(NiterNewton):
            FnU_flat = getFnU_2D(
                Unew[n + 1],
                Unew_n,
                Mk[n + 1],
                omask,
                door_mask_3d[n],
                rc_n,
                Nx, Ny, Dx, Dy, Dt
            ).flatten()

            rows, cols, vals = compute_HJB_matrix_entries(
                Unew_n, Mk[n + 1], omask, door_mask_3d[n], Nx, Ny, Dx, Dy, Dt
            )
            A = sparse.coo_matrix((vals, (rows, cols)), shape=(N_total, N_total)).tocsr()
            b = A.dot(Unew_n.flatten()) - FnU_flat

            for i in range(Nx):
                for j in range(Ny):
                    ind = i * Ny + j
                    if door_mask_3d[n, i, j] == 1:
                        b[ind] = 0.0
                    elif omask[i, j] == 0:
                        b[ind] = 500.0  # Correct obstacle cost (+500.0)

            Unres = sparse.linalg.spsolve(A, b).reshape((Nx, Ny))
            l2err = np.linalg.norm(Unew_n.flatten() - Unres.flatten()) * np.sqrt(Dx * Dy)
            Unew_n = np.copy(Unres)
            if l2err < l2errBoundNewton:
                break

        Unew[n] = Unew_n

    return Unew
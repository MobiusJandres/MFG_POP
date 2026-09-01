"""
PDE Solvers Procedural Interface

This module provides legacy procedural wrapper functions for the Mean Field Game (MFG)
system. These functions act as lightweight delegates
that instantiate an internal adapter mesh and call unified MFGSolver instance methods[cite: 9, 10].

This ensures full backward compatibility with legacy procedural scripts while avoiding
code duplication.
"""
import numpy as np


class _DummyMesh:
    """
    Adapter mesh translating raw procedural arrays into a PDEMeshData interface[cite: 9, 10].

    Attributes:
        Lx (float): Physical domain width in meters[cite: 9]
        Ly (float): Physical domain height in meters[cite: 9]
        dx (float): Grid spacing along x-axis[cite: 9]
        dy (float): Grid spacing along y-axis[cite: 9]
        Nx (int): Grid point count along x-axis[cite: 9]
        Ny (int): Grid point count along y-axis[cite: 9]
        omask (ndarray): Obstacle mask array[cite: 9]
        m0 (ndarray): Initial density field[cite: 9]
        X (ndarray): 2D spatial coordinate grid X[cite: 9]
        Y (ndarray): 2D spatial coordinate grid Y[cite: 9]
    """

    def __init__(self, m0, omask, Dx, Dy, Nx, Ny):
        """
        Initialize the adapter mesh[cite: 9, 10].

        Args:
            m0 (ndarray): Initial density array, shape (Nx, Ny)[cite: 10]
            omask (ndarray): Obstacle mask array, shape (Nx, Ny)[cite: 10]
            Dx (float): Spatial step along x-axis[cite: 10]
            Dy (float): Spatial step along y-axis[cite: 10]
            Nx (int): Grid point count in x-direction[cite: 10]
            Ny (int): Grid point count in y-direction[cite: 10]
        """
        self.Lx, self.Ly = Dx * Nx, Dy * Ny
        self.dx, self.dy = Dx, Dy
        self.Nx, self.Ny = Nx, Ny
        self.omask = omask
        self.m0 = m0

        xSpace = np.linspace(Dx / 2, self.Lx - Dx / 2, Nx)
        ySpace = np.linspace(Dy / 2, self.Ly - Dy / 2, Ny)
        self.X, self.Y = np.meshgrid(xSpace, ySpace, indexing='ij')

    def get_pde_obstacle_mask(self):
        """Returns the obstacle mask array[cite: 9]."""
        return self.omask

    def build_initial_density(self):
        """Returns the initial density array[cite: 9]."""
        return self.m0

    def get_goals(self):
        """Returns an empty goal list for procedural calls[cite: 9]."""
        return []


def solveFP_2D(m0, u, door_mask_3d, omask, Nx, Ny, Nt, Dx, Dy, Dt):
    """
    Solve the forward Kolmogorov-Fokker-Planck (KFP) equation for population density evolution[cite: 10].

    This procedural function acts as a wrapper delegating execution to
    MFGSolver.solve_forward_FP_step[cite: 9, 10].

    Args:
        m0 (ndarray): Initial density distribution, shape (Nx, Ny)[cite: 10]
        u (ndarray): Value function trajectory, shape (Nt+1, Nx, Ny)[cite: 10]
        door_mask_3d (ndarray): Time-dependent exit mask, shape (Nt+1, Nx, Ny)[cite: 10]
        omask (ndarray): Obstacle mask, shape (Nx, Ny)[cite: 10]
        Nx (int): Number of grid points in x-direction[cite: 10]
        Ny (int): Number of grid points in y-direction[cite: 10]
        Nt (int): Number of time steps[cite: 10]
        Dx (float): Spatial step size in x-direction (meters)[cite: 10]
        Dy (float): Spatial step size in y-direction (meters)[cite: 10]
        Dt (float): Time step size (seconds)[cite: 10]

    Returns:
        ndarray: Population density trajectory m, shape (Nt+1, Nx, Ny)[cite: 10]
    """
    from mfgames.problem import MFGSolver

    dummy_mesh = _DummyMesh(m0, omask, Dx, Dy, Nx, Ny)
    solver = MFGSolver(
        pde_mesh_data=dummy_mesh,
        T=Dt * Nt,
        Nt=Nt,
        door_mask_3d=door_mask_3d,
    )
    return solver.solve_forward_FP_step(u, door_mask_3d)


def solveHJB_withM(Uk, Mk, door_mask_3d, omask, running_cost_history, Nx, Ny, Nt, Dx, Dy, Dt, NiterNewton=30, l2errBoundNewton=1e-6, obstacle_penalty=500.0):
    """
    Solve the backward Hamilton-Jacobi-Bellman (HJB) equation for value function[cite: 10].

    This procedural function acts as a wrapper delegating execution to
    MFGSolver.solve_backward_HJB_step[cite: 9, 10].

    Args:
        Uk (ndarray): Value function initial guess, shape (Nt+1, Nx, Ny)[cite: 10]
        Mk (ndarray): Population density trajectory, shape (Nt+1, Nx, Ny)[cite: 10]
        door_mask_3d (ndarray): Time-dependent exit mask, shape (Nt+1, Nx, Ny)[cite: 10]
        omask (ndarray): Obstacle mask, shape (Nx, Ny)[cite: 10]
        running_cost_history (ndarray|None): Time-dependent cost field, shape (Nt+1, Nx, Ny) or None[cite: 10]
        Nx (int): Number of grid points in x-direction[cite: 10]
        Ny (int): Number of grid points in y-direction[cite: 10]
        Nt (int): Number of time steps[cite: 10]
        Dx (float): Spatial step size in x-direction (meters)[cite: 10]
        Dy (float): Spatial step size in y-direction (meters)[cite: 10]
        Dt (float): Time step size (seconds)[cite: 10]
        NiterNewton (int, optional): Max Newton iterations. Default 30[cite: 10].
        l2errBoundNewton (float, optional): Newton convergence tolerance. Default 1e-6[cite: 10].
        obstacle_penalty (float, optional): Obstacle cell potential penalty. Default 500.0[cite: 10].

    Returns:
        ndarray: Value function trajectory Unew, shape (Nt+1, Nx, Ny)[cite: 10]
    """
    from mfgames.problem import MFGSolver

    dummy_mesh = _DummyMesh(Mk[0], omask, Dx, Dy, Nx, Ny)
    solver = MFGSolver(
        pde_mesh_data=dummy_mesh,
        T=Dt * Nt,
        Nt=Nt,
        door_mask_3d=door_mask_3d,
        obstacle_penalty=obstacle_penalty,
    )
    return solver.solve_backward_HJB_step(Mk, None, door_mask_3d)
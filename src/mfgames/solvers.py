"""
PDE Solvers Procedural Interface.

This module provides legacy procedural wrapper functions for the Mean Field Game (MFG)
system. These functions act as lightweight delegates that instantiate an internal
adapter mesh and call unified MFGSolver instance methods.

This ensures full backward compatibility with legacy procedural scripts while avoiding
code duplication.
"""
import numpy as np


class _DummyMesh:
    """
    Adapter mesh translating raw procedural arrays into a PDEMeshData interface.

    Attributes:
        Lx (float): Physical domain width in meters
        Ly (float): Physical domain height in meters
        dx (float): Grid spacing along x-axis
        dy (float): Grid spacing along y-axis
        Nx (int): Grid point count along x-axis
        Ny (int): Grid point count along y-axis
        omask (ndarray): Obstacle mask array
        m0 (ndarray): Initial density field
        X (ndarray): 2D spatial coordinate grid X
        Y (ndarray): 2D spatial coordinate grid Y
    """

    def __init__(self, m0, omask, Dx, Dy, Nx, Ny):
        """
        Initialize the adapter mesh.

        Args:
            m0 (ndarray): Initial density array, shape (Nx, Ny)
            omask (ndarray): Obstacle mask array, shape (Nx, Ny)
            Dx (float): Spatial step along x-axis
            Dy (float): Spatial step along y-axis
            Nx (int): Grid point count in x-direction
            Ny (int): Grid point count in y-direction
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
        """Returns the obstacle mask array."""
        return self.omask

    def build_initial_density(self):
        """Returns the initial density array."""
        return self.m0

    def get_goals(self):
        """Returns an empty goal list for procedural calls."""
        return []


def solveFP_2D(m0, u, door_mask=None, omask=None, Nx=None, Ny=None, Nt=None, Dx=None, Dy=None, Dt=None, door_mask_3d=None):
    """
    Solve the forward Kolmogorov-Fokker-Planck (KFP) equation for population density evolution.

    This procedural function acts as a wrapper delegating execution to
    MFGSolver.solve_forward_FP_step.

    Args:
        m0 (ndarray): Initial density distribution, shape (Nx, Ny)
        u (ndarray): Value function trajectory, shape (Nt+1, Nx, Ny)
        door_mask (ndarray): Time-dependent exit mask, shape (Nt+1, Nx, Ny)
        omask (ndarray): Obstacle mask, shape (Nx, Ny)
        Nx (int): Number of grid points in x-direction
        Ny (int): Number of grid points in y-direction
        Nt (int): Number of time steps
        Dx (float): Spatial step size in x-direction (meters)
        Dy (float): Spatial step size in y-direction (meters)
        Dt (float): Time step size (seconds)
        door_mask_3d (ndarray, optional): Backward compatibility alias for door_mask

    Returns:
        ndarray: Population density trajectory m, shape (Nt+1, Nx, Ny)
    """
    from mfgames.problem import MFGSolver

    active_door_mask = door_mask if door_mask is not None else door_mask_3d
    dummy_mesh = _DummyMesh(m0, omask, Dx, Dy, Nx, Ny)
    solver = MFGSolver(
        pde_mesh_data=dummy_mesh,
        T=Dt * Nt,
        Nt=Nt,
        door_mask=active_door_mask,
    )
    return solver.solve_forward_FP_step(u, active_door_mask)


def solveHJB_withM(Uk, Mk, door_mask=None, omask=None, running_cost_history=None, Nx=None, Ny=None, Nt=None, Dx=None, Dy=None, Dt=None, NiterNewton=30, l2errBoundNewton=1e-6, obstacle_penalty=500.0, door_mask_3d=None):
    """
    Solve the backward Hamilton-Jacobi-Bellman (HJB) equation for value function.

    This procedural function acts as a wrapper delegating execution to
    MFGSolver.solve_backward_HJB_step.

    Args:
        Uk (ndarray): Value function initial guess, shape (Nt+1, Nx, Ny)
        Mk (ndarray): Population density trajectory, shape (Nt+1, Nx, Ny)
        door_mask (ndarray): Time-dependent exit mask, shape (Nt+1, Nx, Ny)
        omask (ndarray): Obstacle mask, shape (Nx, Ny)
        running_cost_history (ndarray|None): Time-dependent cost field, shape (Nt+1, Nx, Ny) or None
        Nx (int): Number of grid points in x-direction
        Ny (int): Number of grid points in y-direction
        Nt (int): Number of time steps
        Dx (float): Spatial step size in x-direction (meters)
        Dy (float): Spatial step size in y-direction (meters)
        Dt (float): Time step size (seconds)
        NiterNewton (int, optional): Max Newton iterations. Default 30.
        l2errBoundNewton (float, optional): Newton convergence tolerance. Default 1e-6.
        obstacle_penalty (float, optional): Obstacle cell potential penalty. Default 500.0.
        door_mask_3d (ndarray, optional): Backward compatibility alias for door_mask

    Returns:
        ndarray: Value function trajectory Unew, shape (Nt+1, Nx, Ny)
    """
    from mfgames.problem import MFGSolver

    active_door_mask = door_mask if door_mask is not None else door_mask_3d
    dummy_mesh = _DummyMesh(Mk[0], omask, Dx, Dy, Nx, Ny)
    solver = MFGSolver(
        pde_mesh_data=dummy_mesh,
        T=Dt * Nt,
        Nt=Nt,
        door_mask=active_door_mask,
        obstacle_penalty=obstacle_penalty,
    )
    return solver.solve_backward_HJB_step(Mk, None, active_door_mask)
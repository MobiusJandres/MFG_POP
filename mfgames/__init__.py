"""
Mean Field Games Library: Mean Field Games PDE Solvers for Traffic Flow and Pursuit-Evasion.
"""

from .geometry import MAP2PDE, create_moving_door_mask
from .evasion import EvaderSwarm
from .solvers import solveFP_2D, solveHJB_withM
from .plotting import MFGPlotter
from .problem import MFGSolver

__all__ = [
    "MAP2PDE",
    "create_moving_door_mask",
    "EvaderSwarm",
    "solveFP_2D",
    "solveHJB_withM",
    "MFGPlotter",
    "MFGSolver",
]
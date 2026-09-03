"""
Mean Field Games Library: Coupled PDE Solvers for Crowd Dynamics and Pursuit-Evasion Games.

This package implements numerical solvers for Mean Field Games (MFG) systems, solving
coupled Hamilton-Jacobi-Bellman (HJB) and Kolmogorov-Fokker-Planck (KFP) partial
differential equations using finite difference methods and Picard iteration.

Mathematical Framework
----------------------
The library solves the coupled MFG system:

    -∂u/∂t - ν Δu + H(x, m, ∇u) = f(x, m)     (Hamilton-Jacobi-Bellman)
     ∂m/∂t - ν Δm - div(m * DₚH(x, m, ∇u)) = 0  (Kolmogorov-Fokker-Planck)

where:
    - u(x, t): value function (cost-to-go from position x at time t)
    - m(x, t): population density distribution
    - H(x, m, p): congestion-dependent Hamiltonian
    - ν: diffusion coefficient (0.05 in current implementation)
    - f(x, m): running cost (distance to goals, interaction penalties)

The Hamiltonian incorporates congestion effects via:
    H(x, m, p) = -α / (1 + m)^β * |p|²
where higher density m reduces mobility.

Core Modules
------------
geometry : Spatial domain and obstacle configuration
    - MAP2PDE: Parses MovingAI benchmark .map/.scen files into PDE grids
    - MFGTrafficGeometry: Generates primitive corridor geometries
    - create_moving_door_mask: Dynamic exit door trajectory generation

evasion : Goal and evader trajectory management
    - Goal: Handles stationary goals, prescribed paths, and evasive targets
    - EvaderSwarm: Alias for Goal (backward compatibility)

solvers : Low-level PDE time-stepping routines
    - solveFP_2D: Forward-in-time Fokker-Planck density evolution
    - solveHJB_withM: Backward-in-time HJB value function solver

numerics : Numba-JIT compiled finite difference operators
    - compute_FP_matrix_entries: Sparse matrix assembly for KFP transport
    - compute_HJB_matrix_entries: Sparse matrix assembly for HJB linearization
    - getFnU_2D: Residual computation for Newton iteration

problem : High-level object-oriented solver interfaces
    - MFGSolver: 1-population solver (traffic flow, pursuit-evasion)
    - MFG2PopSolver: 2-population competitive interaction solver

plotting : Visualization and animation utilities
    - MFGPlotter: Snapshot dashboards, time-lapse PNGs, GIF/MP4 export
    - plot_snapshot: Single timestep density + value heatmap
    - plot_progression: Multi-snapshot time evolution visualization

Typical Usage
-------------
1-Population Pursuit-Evasion Scenario:
    >>> from mfgames import MAP2PDE, MFGSolver, MFGPlotter
    >>>
    >>> # Load MovingAI benchmark map and scenario
    >>> mesh = MAP2PDE(map_filepath="arena.map", scen_filepath="arena.scen",
    ...                Lx=768.0, Ly=768.0, Nx=150, Ny=150)
    >>> mesh.parse_files(num_agents=50)
    >>> mesh.build_spatial_mesh()
    >>>
    >>> # Configure evader goals (moving targets)
    >>> goals = [{'type': 'evader', 'position': [400, 400], 'v_max': 15.0}]
    >>>
    >>> # Solve coupled MFG system
    >>> solver = MFGSolver(mesh, T=300.0, Nt=3000, goal_configs=goals)
    >>> U, M = solver.run_picard_system(max_iters=25, tolerance=1e-5)
    >>>
    >>> # Export visualization
    >>> plotter = MFGPlotter(pde_mesh_data=mesh, solver_instance=solver)
    >>> plotter.plot_snapshots(output_file="results.png")
    >>> plotter.save_mp4("simulation.mp4", fps=30)

2-Population Competitive Scenario:
    >>> from mfgames import MAP2PDE, MFG2PopSolver, MFGPlotter
    >>>
    >>> mesh1 = MAP2PDE("map.map", "team1.scen", Nx=100, Ny=100)
    >>> mesh2 = MAP2PDE("map.map", "team2.scen", Nx=100, Ny=100)
    >>> mesh1.parse_files(num_agents=30)
    >>> mesh2.parse_files(num_agents=30)
    >>>
    >>> solver = MFG2PopSolver(mesh1, mesh2, T=300.0, Nt=3000)
    >>> U1, M1, U2, M2 = solver.run_picard_system(max_iters=25)
    >>>
    >>> plotter = MFGPlotter(pde_mesh_data_1=mesh1, pde_mesh_data_2=mesh2,
    ...                      solver_instance=solver)
    >>> plotter.save_density_frames()

Numerical Methods
-----------------
- **Discretization**: Upwind finite differences for Hamiltonian gradient terms
- **Time Integration**: Implicit Euler backward (HJB), implicit forward (KFP)
- **Nonlinear Solver**: Newton iteration for HJB, Picard iteration for coupling
- **Linear Algebra**: Scipy sparse CSR matrices with UMFPACK direct solve
- **Performance**: Numba JIT compilation for matrix assembly hot loops

Boundary Conditions
-------------------
- **Obstacles**: Neumann (zero flux) via ghost cell clamping in stencil
- **Exit Doors**: Dirichlet u = 0 (hard zero-cost exits)
- **Domain Edges**: Neumann (reflective walls)

Key Parameters
--------------
Spatial Resolution:
    Nx, Ny : Grid points (75-200 typical, affects accuracy and runtime)
    Lx, Ly : Physical domain size in meters

Temporal Resolution:
    Nt : Number of timesteps (100-3000 typical)
    T  : Final time in seconds

Solver Tuning:
    thetaUM : Picard under-relaxation (0.1 typical, reduces oscillations)
    NiterNewton : Newton iterations per timestep (5-30)
    l2errBoundNewton : Newton convergence tolerance (1e-6)

Physics Parameters (hardcoded in numerics.py):
    ν = 0.05           : Diffusion coefficient
    α = 8.0, β = 0.75  : Congestion scaling in Hamiltonian
    obstacle_penalty   : High cost for obstacle cells (-500 typical)

Performance Considerations
---------------------------
- Runtime scales as O(Nx × Ny × Nt × Niter_Picard × Niter_Newton)
- Memory scales as O(Nx × Ny × Nt) for storing full trajectories
- Typical 150×150×3000 simulation: ~5-15 minutes on modern CPU
- Numba compilation adds ~5s startup overhead on first run

References
----------
Mathematical formulation detailed in:
    - MFG_Traffic_numba.md (repository documentation)
    - notes.tex (LaTeX mathematical derivations)

Package Structure
-----------------
mfgames/
├── __init__.py       : Package exports and high-level API
├── geometry.py       : Spatial domain and obstacle handling
├── evasion.py        : Goal/evader trajectory management
├── solvers.py        : Time-stepping routines for HJB/KFP
├── numerics.py       : JIT-compiled finite difference operators
├── problem.py        : Object-oriented MFGSolver/MFG2PopSolver
└── plotting.py       : Visualization and animation tools

See Also
--------
- MovingAI benchmark: https://movingai.com/benchmarks/
- Mean Field Games theory: Lions & Lasry (2006-2007 Collège de France lectures)
"""

from .geometry import MAP2PDE, create_moving_door_mask
from .evasion import EvaderSwarm, Goal
from .solvers import solveFP_2D, solveHJB_withM
from .plotting import MFGPlotter
from .problem import MFGSolver

__all__ = [
    "MAP2PDE",
    "create_moving_door_mask",
    "EvaderSwarm",
    "Goal",
    "solveFP_2D",
    "solveHJB_withM",
    "MFGPlotter",
    "MFGSolver",
]
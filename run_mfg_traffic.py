"""
Traffic Evacuation Mean Field Game Simulation.

This script demonstrates a crowd evacuation scenario modeled as a Mean Field Game (MFG),
where a large population of agents navigates through a confined space with obstacles
toward exit doors, subject to congestion effects.

Key MFG Concepts Illustrated:
-----------------------------
- **Congestion-Dependent Dynamics**: Agent movement costs increase in crowded regions
  via a congestion Hamiltonian H(x, m, p) that depends on local density m(x,t).
- **Coupled PDE System**: Solves Hamilton-Jacobi-Bellman (HJB) equation backward in time
  for optimal cost-to-go u(x,t), and Kolmogorov-Fokker-Planck (KFP) equation forward
  in time for population density m(x,t).
- **Picard Iteration**: Iteratively couples HJB and KFP solutions until convergence,
  producing Nash equilibrium strategies for the mean field game.
- **Moving Targets**: Supports time-varying exit door positions (e.g., closing/opening doors).
- **Obstacle Avoidance**: Positive obstacle penalty (default +500) creates repulsive
  potential that guides agents around walls and barriers.

Problem Variants:
-----------------
This script is configured for traffic evacuation. Related scenarios:
- Pursuit-evasion: Use negative obstacle_penalty (see run_mfg_evasion.py)
- Two-population: Distinct evader/pursuer populations (see run_mfg_2pop.py)

Expected Outputs:
-----------------
Generated in the directory specified by --save_dir (default: results/):
- snapshots_dashboard.png: Multi-panel dashboard showing density and value function
  evolution at key time snapshots
- frames/: Directory containing individual density heatmap frames for each timestep
- traffic_simulation.gif: Animated GIF of density evolution over time

Usage:
------
    python run_mfg_traffic.py [OPTIONS]

Examples:
    # Run with default config (configs/mfg_traffic.yml)
    python run_mfg_traffic.py

    # Use custom config and increase Picard iterations
    python run_mfg_traffic.py --config configs/custom.yml --max_iters 100

    # Higher resolution grid and custom obstacle penalty
    python run_mfg_traffic.py --Nx 200 --Ny 200 --obstacle_penalty 800.0

Configuration:
--------------
Key parameters loaded from config file (configs/mfg_traffic.yml):
- room_width, room_height: Physical domain size (meters)
- Nx, Ny: Spatial grid resolution
- T, Nt: Simulation duration (seconds) and number of timesteps
- relaxation_theta: Under-relaxation parameter for Picard iteration (0 < theta <= 1)
- doors: List of exit door specifications (position, timing, movement)
- max_iters: Maximum Picard iterations (typically 50-100 for convergence)

See Also:
---------
- mfgames.geometry.MFGTrafficGeometry: Spatial mesh and obstacle setup
- mfgames.problem.MFGSolver: Core Picard iteration solver for HJB-KFP system
- mfgames.plotting.MFGPlotter: Visualization and animation utilities
"""

# === Imports ===

# Command-line argument parsing and configuration management
from args.Options import Options

# Spatial discretization: mesh construction, obstacle masks, and door boundary conditions
from mfgames.geometry import (
    MFGTrafficGeometry,              # 2D spatial mesh with obstacle support
    create_moving_door_mask,         # Time-varying exit door masks for moving targets
    build_door_trajectories_from_config,  # Parse door configs into trajectories
)

# Core MFG solver: Picard iteration for coupled HJB-KFP system
from mfgames.problem import MFGSolver

# Visualization: snapshot dashboards, frame generation, and GIF animations
from mfgames.plotting import MFGPlotter


# === Default Configuration ===

# Default static exit doors at bottom-left and bottom-right corners.
# These are fallback door positions used if the config file does not specify 'doors'.
# Door coordinates support expressions (e.g., "room_width - 10.0") evaluated at runtime.
DEFAULT_STATIC_DOORS = [
    {'x1': "2.0", 'x2': "10.0", 'y1': "0.0", 'y2': "5.0"},
    {'x1': "room_width - 10.0", 'x2': "room_width - 2.0", 'y1': "0.0", 'y2': "5.0"}
]


def main():
    """
    Execute traffic evacuation Mean Field Game simulation.

    Workflow:
    ---------
    1. **Argument Parsing**: Load configuration from YAML file (default: configs/mfg_traffic.yml)
       and parse command-line overrides.
    2. **Geometry Setup**: Initialize 2D spatial mesh with obstacles and walls.
    3. **Door Configuration**: Build time-varying exit door masks from config specifications
       (supports static or moving doors).
    4. **Solver Initialization**: Instantiate MFGSolver with positive obstacle penalty (+500)
       to create repulsive barrier around obstacles (agents avoid walls).
    5. **Picard Iteration**: Solve coupled HJB-KFP system iteratively until convergence.
       - HJB solved backward in time for optimal cost-to-go u(x,t)
       - KFP solved forward in time for population density m(x,t)
    6. **Visualization**: Generate snapshot dashboard, individual frames, and animated GIF.

    Configuration Parameters:
    -------------------------
    Loaded from config file and command-line arguments:
    - room_width, room_height (float): Physical domain dimensions in meters
    - Nx, Ny (int): Spatial grid resolution (higher = more accurate, slower)
    - T (float): Total simulation time in seconds
    - Nt (int): Number of discrete timesteps
    - relaxation_theta (float): Under-relaxation parameter for Picard iteration (0 < theta <= 1)
      Lower values improve stability but require more iterations
    - doors (list[dict]): Exit door specifications with position and timing
    - max_iters (int): Maximum Picard iterations (50-100 typical for convergence)
    - obstacle_penalty (float): Penalty for entering obstacle cells
      * Positive (+500): Repulsive barrier (traffic evacuation)
      * Negative (-500): Attractive barrier (pursuit-evasion, see run_mfg_evasion.py)

    Outputs:
    --------
    All outputs saved to directory specified by --save_dir (default: results/):
    - snapshots_dashboard.png: Multi-panel visualization at key time snapshots
    - frames/: Individual density heatmap images for each timestep
    - traffic_simulation.gif: Animated visualization of density evolution

    Returns:
    --------
    None
        Results are saved to disk and success message printed to stdout.

    Raises:
    -------
    ValueError
        If configuration parameters are invalid (e.g., negative grid resolution)
    ConvergenceError
        If Picard iteration fails to converge within max_iters iterations
    """
    # === Step 1: Parse command-line arguments and load configuration ===
    options = Options()
    options.parser.set_defaults(config='configs/mfg_traffic.yml')
    # Add obstacle_penalty argument: positive for traffic (repulsive), negative for evasion (attractive)
    options.parser.add_argument('--obstacle_penalty', type=float, default=500.0, help='Obstacle cell potential penalty (+500 for traffic)')

    args = options.parseArgs()
    print(f"Results will be saved to: {args.save_dir}", flush=True)

    # === Step 2: Initialize spatial geometry and mesh ===
    # Create 2D Cartesian mesh discretizing the [0, Lx] × [0, Ly] domain
    # MFGTrafficGeometry handles obstacle mask generation and boundary conditions
    mesh = MFGTrafficGeometry(
        Lx=args.room_width,
        Ly=args.room_height,
        Nx=args.Nx,
        Ny=args.Ny
    )
    mesh.build_spatial_mesh()

    # === Step 3: Configure exit doors (static or time-varying) ===
    # Extract door specifications from config file, or use default static doors if not specified
    doors_cfg = getattr(args, 'doors', None) or DEFAULT_STATIC_DOORS

    # Parse door configs into time-indexed trajectories (supports moving/opening/closing doors)
    # Each door trajectory specifies position bounds [x1, x2] × [y1, y2] at each timestep
    door_trajectories = build_door_trajectories_from_config(
        doors_cfg, args.T, args.room_width, args.room_height
    )
    # Create 3D boolean mask (Nt+1, Nx, Ny) indicating door cells at each timestep
    # Door cells have Dirichlet boundary condition u = 0 (zero cost-to-go at exits)
    door_mask_3d = create_moving_door_mask(
        door_trajectories, args.Nt, args.Nx, args.Ny, mesh.X, mesh.Y, args.T
    )

    # Determine if goals are exits (True for traffic) or safe zones (False for evasion)
    goals_are_exits = getattr(args, 'goals_are_exits', True)

    # === Step 4: Instantiate MFG solver ===
    # Positive obstacle_penalty (+500) creates repulsive barrier around walls,
    # guiding agents to avoid obstacles and move toward exits
    solver = MFGSolver(
        pde_mesh_data=mesh,             # Spatial mesh with coordinates and obstacle mask
        T=args.T,                       # Total simulation time (seconds)
        Nt=args.Nt,                     # Number of discrete timesteps
        thetaUM=args.relaxation_theta,  # Under-relaxation parameter for Picard iteration
        door_mask_3d=door_mask_3d,      # Time-varying exit door mask
        goals_are_exits=goals_are_exits, # True: agents seek exits (traffic scenario)
        obstacle_penalty=args.obstacle_penalty  # Positive: repulsive barrier around walls
    )

    # === Step 5: Solve coupled HJB-KFP system via Picard iteration ===
    # Iteratively solves HJB (backward) and KFP (forward) until convergence:
    # - HJB: ∂u/∂t + H(x, m, ∇u) = 0 (optimal control for value function)
    # - KFP: ∂m/∂t - div(m ∇H_p) - ε Δm = 0 (density evolution under optimal policy)
    # Convergence checked via L2 norm of successive iterates
    solver.run_picard_system(max_iters=args.max_iters)

    # === Step 6: Generate visualizations ===
    plotter = MFGPlotter(pde_mesh_data=mesh, solver_instance=solver)

    # Generate snapshot dashboard: multi-panel view of density m(x,t) and value function u(x,t)
    # at key time snapshots (t=0, T/4, T/2, 3T/4, T)
    dashboard_path = args.save_dir / "snapshots_dashboard.png"
    plotter.plot_snapshots(output_file=str(dashboard_path))

    # Save individual density heatmap frames for each timestep
    # These frames are used to create the animated GIF
    frames_dir = args.save_dir / "frames"
    plotter.save_density_frames(output_dir=str(frames_dir))

    # Create animated GIF showing density evolution over time
    # fps=15 provides smooth playback (can increase for faster animation)
    animation_path = args.save_dir / "traffic_simulation.gif"
    plotter.create_movie(frame_dir=str(frames_dir), output_file=str(animation_path), fps=15)

    print("Traffic run completed successfully!", flush=True)


# === Script Entry Point ===
# Execute main() when script is run directly (not imported as module)
if __name__ == "__main__":
    main()
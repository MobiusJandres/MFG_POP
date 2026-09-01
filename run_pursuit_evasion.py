"""
Pursuit-Evasion Mean Field Game Simulation Entry Point

This script demonstrates a two-population Mean Field Game (MFG) where pursuers
and evaders interact with opposing objectives. It showcases heterogeneous agent
modeling, dynamic target tracking, and congestion-aware crowd dynamics.

Key MFG Concepts Illustrated:
    - Pursuit-evasion dynamics: Pursuit swarm minimize distance to evaders, evaders maximize
    - Heterogeneous targets: Each population type has different v_max (movement capability)
    - Coupled HJB-KFP system: Value functions and densities evolve simultaneously
    - Obstacle-aware navigation: Negative obstacle penalties create repulsion barriers
    - Congestion effects: Agent density influences optimal control via Hamiltonian

Configuration:
    Primary config file: configs/pursuit_evasion.yml
    Defines: spatial domain, time horizon, Picard iteration parameters, populations

Expected Outputs:
    - snapshots_dashboard.png: Multi-timestep visualization grid (density + value)
    - frames/*.png: Individual frames for each timestep
    - pursuit_evasion.gif: Animated simulation showing population evolution
    All saved to args.save_dir (default: runs/pursuit_evasion_YYYYMMDD_HHMMSS/)

Usage:
    python run_pursuit_evasion.py
    python run_pursuit_evasion.py --config configs/custom_pursuit.yml
    python run_pursuit_evasion.py --obstacle_penalty -1000.0 --running_cost_weight 0.05
"""

# Command-line argument parsing and configuration management
from args.Options import Options

# Geometry and mesh generation (MAP2PDE converts spatial maps to PDE discretization)
from mfgames.geometry import MAP2PDE

# Core MFG solver (Picard iteration for coupled HJB-KFP system)
from mfgames.problem import MFGSolver

# Visualization tools (heatmaps, animations, dashboards)
from mfgames.plotting import MFGPlotter


def main():
    """
    Execute pursuit-evasion MFG simulation with heterogeneous populations.

    This function orchestrates the complete workflow:
    1. Parse command-line arguments and load configuration file
    2. Initialize spatial geometry and mesh discretization
    3. Configure MFG solver with population-specific parameters
    4. Run Picard iteration to solve coupled HJB-KFP system
    5. Generate visualizations (dashboard, frames, animation)

    Key Configuration Parameters:
        obstacle_penalty (float): Negative value creates repulsion field around obstacles.
            Default -500.0 for pursuit-evasion (stronger than traffic scenarios which
            use positive penalties). More negative = stronger avoidance.

        running_cost_weight (float): Scales the congestion/running cost term in the
            Hamiltonian. Smaller values (0.01-0.05) prioritize reaching targets quickly;
            larger values emphasize avoiding crowded regions.

        goals (list): Heterogeneous target dictionaries with 'type' (pursuer/evader),
            'position' ([x, y]), and 'v_max' (max velocity). Pursuers track evaders;
            evaders maximize distance to pursuers.

        goals_are_exits (bool): If True, populations are absorbed at goal locations
            (Dirichlet BC: u=0). If False, goals create attractive potential wells.

    Workflow Steps:
        1. Argument Parsing: Load YAML config and override with command-line flags
        2. Geometry Setup: Convert map/scenario files to PDE mesh (MAP2PDE)
        3. Solver Configuration: Initialize MFGSolver with goal dictionaries
        4. Picard Iteration: Solve HJB backward, KFP forward until convergence
        5. Visualization: Generate dashboard, individual frames, and GIF animation

    Returns:
        None. Outputs are saved to args.save_dir.

    Raises:
        FileNotFoundError: If config file or map/scenario files are missing.
        ValueError: If goal dictionaries have invalid structure.
    """
    # ============================================================================
    # STEP 1: Argument Parsing and Configuration
    # ============================================================================
    options = Options()
    options.parser.set_defaults(config='configs/pursuit_evasion.yml')

    # Add pursuit-evasion specific arguments
    options.parser.add_argument('--goals', default=None, nargs='*', help='Target goal dictionaries or coordinates')
    options.parser.add_argument('--goals_are_exits', action='store_true', default=False, help='If True, goals act as exits')

    # Negative obstacle penalty creates repulsion barriers (contrast with positive penalties in traffic scenarios)
    options.parser.add_argument('--obstacle_penalty', type=float, default=-500.0, help='Obstacle cell potential penalty')

    # Running cost balances target-seeking vs congestion-avoidance behavior
    options.parser.add_argument('--running_cost_weight', type=float, default=0.01, help='Running cost scaling weight')

    args = options.parseArgs()
    print(f"Results will be saved to: {args.save_dir}", flush=True)

    # ============================================================================
    # STEP 2: Geometry and Mesh Initialization
    # ============================================================================
    # Extract map and scenario file paths if provided (optional for simple domains)
    map_str = str(args.map_file) if getattr(args, 'map_file', None) else None
    scen_str = str(args.scen_file) if getattr(args, 'scen_file', None) else None

    # Parse heterogeneous targets (dictionaries or legacy coordinate lists)
    # Goal dictionaries must contain: {'type': 'pursuer'|'evader', 'position': [x,y], 'v_max': float}
    raw_goals = getattr(args, 'goals', None)
    goal_configs = []

    if raw_goals:
        for g in raw_goals:
            if isinstance(g, dict):
                # Full goal dictionary with type, position, v_max
                goal_configs.append(g)
            elif isinstance(g, (list, tuple)):
                # Legacy format: assume [x, y] coordinates, default to evader with v_max=15.0
                goal_configs.append({'type': 'evader', 'position': [float(g[0]), float(g[1])], 'v_max': 15.0})

    # Build PDE mesh from spatial domain specification
    # MAP2PDE handles: obstacle masks, door/exit locations, initial agent placement
    pde_mesh = MAP2PDE(
        map_filepath=map_str,           # .map file defining obstacles/free space
        scen_filepath=scen_str,         # .scen file with agent start positions
        Lx=args.room_width,             # Physical domain width (meters)
        Ly=args.room_height,            # Physical domain height (meters)
        Nx=args.Nx,                     # Spatial grid resolution (x-direction)
        Ny=args.Ny,                     # Spatial grid resolution (y-direction)
        custom_goals=goal_configs       # Heterogeneous population targets
    )
    pde_mesh.parse_files(num_agents=args.num_agents)  # Load map/scenario data
    pde_mesh.build_spatial_mesh()                      # Discretize continuous domain

    # ============================================================================
    # STEP 3: Solver Configuration
    # ============================================================================
    # Initialize MFG solver with mesh, time discretization, and population parameters
    mfg_solver = MFGSolver(
        pde_mesh_data=pde_mesh,                       # Spatial mesh with obstacles/doors
        T=args.T,                                     # Time horizon (seconds)
        Nt=args.Nt,                                   # Number of time steps
        thetaUM=args.relaxation_theta,                # Picard relaxation (0 < theta < 1)
        goal_configs=goal_configs,                    # Heterogeneous population targets
        goals_are_exits=args.goals_are_exits,         # Exit mode vs potential well mode
        obstacle_penalty=args.obstacle_penalty,       # Negative for repulsion barriers
        running_cost_weight=args.running_cost_weight  # Congestion sensitivity scaling
    )

    # ============================================================================
    # STEP 4: Picard Iteration (Core MFG Solver)
    # ============================================================================
    # Solve coupled HJB-KFP system:
    #   - HJB: Backward in time for value functions (optimal cost-to-go)
    #   - KFP: Forward in time for densities (population evolution)
    # Iterates until L2 error between successive solutions < tolerance or max_iters reached
    mfg_solver.run_picard_system(max_iters=args.max_iters)

    # ============================================================================
    # STEP 5: Visualization Generation
    # ============================================================================
    plotter = MFGPlotter(pde_mesh_data=pde_mesh, solver_instance=mfg_solver)

    # Generate multi-timestep dashboard (density + value function heatmaps in grid layout)
    dashboard_path = args.save_dir / "snapshots_dashboard.png"
    plotter.plot_snapshots(output_file=str(dashboard_path))

    # Save individual frames for each timestep (used for animation creation)
    frames_dir = args.save_dir / "frames"
    plotter.save_density_frames(output_dir=str(frames_dir))

    # Create GIF animation showing temporal evolution of population densities
    animation_path = args.save_dir / "pursuit_evasion.gif"
    plotter.create_movie(frame_dir=str(frames_dir), output_file=str(animation_path), fps=15)

    print("Pursuit-Evasion run completed successfully!", flush=True)



# Script entry point: execute main() when run directly (not imported as module)
if __name__ == "__main__":
    main()
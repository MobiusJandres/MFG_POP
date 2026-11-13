"""
Map-based Mean Field Game simulation entry point.

This runner supports two related workflows on a MovingAI-derived spatial domain:
1. a legacy single-population solve with heterogeneous goals via top-level `goals`,
2. a generalized multi-population solve via `populations` plus numeric
   `interaction_matrix` and `congestion_matrix` inputs.

The generalized path is intended for scenarios with multiple interacting
populations on a shared map. The interaction matrix controls how each
population contributes to the running potential seen by the others, while the
congestion matrix controls how strongly one population slows another.

Default configuration:
    `configs/map_simulation.yml`

Generalized multi-population example:
    `configs/map_simulation_3populations.yml`

Outputs:
    - `snapshots_dashboard.png`
    - `frames/*.png`
    - `dynamic_goals.gif`

All outputs are written beneath the derived `save_dir`, which is constructed from
`results_dir`, `exp_name`, and `run_name`.
"""

# Command-line argument parsing and configuration management
from mfgames.config import Options

# Geometry and mesh generation (MAP2PDE converts spatial maps to PDE discretization)
from mfgames.geometry import MAP2PDE

# Core MFG solver (Picard iteration for coupled HJB-KFP system)
from mfgames.problem import MFGSolver, Population

# Visualization tools (heatmaps, animations, dashboards)
from mfgames.plotting import MFGPlotter


def main():
    """
    Execute a map-based MFG simulation.

    This function orchestrates the complete workflow:
    1. Parse command-line arguments and load configuration file
    2. Initialize spatial geometry and mesh discretization
    3. Configure either a single-population or generalized multi-population solver
    4. Run Picard iteration to solve coupled HJB-KFP system
    5. Generate visualizations (dashboard, frames, animation)

    Key Configuration Parameters:
        goals (list): Top-level shared goals. In generalized multi-population mode,
            these are inherited by every population automatically.

        populations (list): Generalized multi-population configuration. Each entry may
            specify its own initial blobs, goal set, mobility coefficients, and costs.
            Population-specific goals should be provided under `goals` and are
            appended to the shared top-level `goals` list.

        interaction_matrix (list[list[float]]): Numeric interaction coupling matrix.
            Entry `(p, q)` weights how strongly density from population `q`
            contributes to the running potential seen by population `p`.
            Positive values are repulsive and negative values are attractive.

        congestion_matrix (list[list[float]]): Numeric mobility-friction matrix.
            Larger positive values make one population slow another more strongly.

    Workflow Steps:
        1. Argument Parsing: Load YAML config and override with command-line flags
        2. Geometry Setup: Convert map/scenario files to PDE mesh (MAP2PDE)
        3. Solver Configuration: Build `Population` objects when `populations` is provided
        4. Picard Iteration: Solve HJB backward, KFP forward until convergence
        5. Visualization: Generate dashboard, individual frames, and GIF animation

    Returns:
        None. Outputs are saved to args.save_dir.

    Raises:
        FileNotFoundError: If config file or map/scenario files are missing.
        ValueError: If goal/population dictionaries or coupling matrices are invalid.
    """
    # ============================================================================
    # STEP 1: Argument Parsing and Configuration
    # ============================================================================
    options = Options()
    options.parser.set_defaults(config='configs/map_simulation.yml')

    # Scenario-specific arguments
    options.parser.add_argument('--goals', default=None, nargs='*', help='Goal dictionaries or coordinates.')
    options.parser.set_defaults(
        goals_are_exits=False,
        obstacle_penalty=-500.0,
        running_cost_weight=0.01,
        terminal_cost_weight=None,
        running_cost_power=2,
        terminal_cost_power=2,
    )

    args = options.parseArgs()
    print(f"Results will be saved to: {args.save_dir}", flush=True)

    # ============================================================================
    # STEP 2: Geometry and Mesh Initialization
    # ============================================================================
    # Extract map and scenario file paths if provided (optional for simple domains)
    map_str = str(args.map_file) if getattr(args, 'map_file', None) else None
    scen_str = str(args.scen_file) if getattr(args, 'scen_file', None) else None

    # Parse top-level shared goals (dictionaries or legacy coordinate lists)
    # Goal dictionaries must contain: {'type': 'stationary'|'prescribed'|'evader', ...}
    raw_goals = getattr(args, 'goals', None)
    shared_goal_configs = []

    if raw_goals:
        for g in raw_goals:
            if isinstance(g, dict):
                shared_goal_configs.append(g)
            elif isinstance(g, (list, tuple)):
                shared_goal_configs.append({'type': 'evader', 'position': [float(g[0]), float(g[1])], 'v_max': args.v_max_evader})

    population_configs = getattr(args, 'populations', None)

    # Build PDE mesh from spatial domain specification
    # MAP2PDE handles: obstacle masks, door/exit locations, initial agent placement
    pde_mesh = MAP2PDE(
        map_filepath=map_str,
        scen_filepath=scen_str,
        Lx=args.room_width,
        Ly=args.room_height,
        Nx=args.Nx,
        Ny=args.Ny,
        custom_goals=shared_goal_configs,
    )
    pde_mesh.parse_files(num_agents=args.num_agents)
    pde_mesh.build_spatial_mesh()

    populations = None
    interaction_matrix = getattr(args, 'interaction_matrix', None)
    congestion_matrix = getattr(args, 'congestion_matrix', None)
    if population_configs:
        populations = []

        for idx, pop_cfg in enumerate(population_configs):
            population_goals = list(pop_cfg.get('goals', []))
            combined_goals = list(shared_goal_configs) + population_goals

            mesh = MAP2PDE(
                map_filepath=map_str,
                scen_filepath=scen_str,
                Lx=args.room_width,
                Ly=args.room_height,
                Nx=args.Nx,
                Ny=args.Ny,
                custom_goals=combined_goals,
            )
            mesh.parse_files(
                num_agents=pop_cfg.get('num_agents', args.num_agents),
                start_idx=pop_cfg.get('start_idx', 0),
            )
            mesh.build_spatial_mesh()

            if pop_cfg.get('initial_blobs'):
                mesh.set_custom_initial_blobs(
                    pop_cfg['initial_blobs'],
                    normalize_mass=pop_cfg.get('normalize_mass', False),
                )

            populations.append(
                Population(
                    name=pop_cfg.get('name', f'population_{idx + 1}'),
                    mesh=mesh,
                    m0=mesh.build_initial_density(
                        sigma_multiplier=pop_cfg.get('sigma_multiplier', 1.5)
                    ),
                    goal_configs=combined_goals,
                    goals_are_exits=pop_cfg.get('goals_are_exits', args.goals_are_exits),
                    running_cost_weight=pop_cfg.get('running_cost_weight', args.running_cost_weight),
                    terminal_cost_weight=pop_cfg.get('terminal_cost_weight', args.terminal_cost_weight),
                    running_cost_power=pop_cfg.get('running_cost_power', args.running_cost_power),
                    terminal_cost_power=pop_cfg.get('terminal_cost_power', args.terminal_cost_power),
                    saturated_goal_penalty=pop_cfg.get('saturated_goal_penalty', args.saturated_goal_penalty),
                    obstacle_penalty=pop_cfg.get('obstacle_penalty', args.obstacle_penalty),
                    alpha=pop_cfg.get('alpha', 8.0),
                    beta=pop_cfg.get('beta', 0.75),
                    h0=pop_cfg.get('h0', 0.1),
                    gamma=pop_cfg.get('gamma', 16.0),
                    upwind_mode=pop_cfg.get('upwind_mode', 0),
                )
            )

    # ============================================================================
    # STEP 3: Solver Configuration
    # ============================================================================
    # Initialize MFG solver with mesh, time discretization, and population parameters
    if populations is not None:
        mfg_solver = MFGSolver(
            populations=populations,
            T=args.T,
            Nt=args.Nt,
            thetaUM=args.relaxation_theta,
            interaction_matrix=interaction_matrix,
            congestion_matrix=congestion_matrix,
            shared_capacity_goals=getattr(args, 'shared_capacity_goals', False),
        )
    else:
        mfg_solver = MFGSolver(
            pde_mesh_data=pde_mesh,
            T=args.T,
            Nt=args.Nt,
            thetaUM=args.relaxation_theta,
            goal_configs=shared_goal_configs,
            goals_are_exits=args.goals_are_exits,
            obstacle_penalty=args.obstacle_penalty,
            running_cost_weight=args.running_cost_weight,
            terminal_cost_weight=args.terminal_cost_weight,
            running_cost_power=args.running_cost_power,
            terminal_cost_power=args.terminal_cost_power,
            saturated_goal_penalty=args.saturated_goal_penalty,
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
    animation_path = args.save_dir / "population_dynamics.gif"
    plotter.create_movie(frame_dir=str(frames_dir), output_file=str(animation_path), fps=15)

    print("Map simulation run completed successfully!", flush=True)

# Script entry point: execute main() when run directly (not imported as module)
if __name__ == "__main__":
    main()

"""
Pursuit-Evasion entry point driven by configuration files and Goal dictionaries.
"""
from args.Options import Options
from mfgames.geometry import MAP2PDE
from mfgames.problem import MFGSolver
from mfgames.plotting import MFGPlotter


def main():
    options = Options()
    options.parser.set_defaults(config='configs/pursuit_evasion.yml')
    options.parser.add_argument('--goals', default=None, nargs='*', help='Target goal dictionaries or coordinates')
    options.parser.add_argument('--goals_are_exits', action='store_true', default=False, help='If True, goals act as exits')
    options.parser.add_argument('--obstacle_penalty', type=float, default=-500.0, help='Obstacle cell potential penalty')
    options.parser.add_argument('--running_cost_weight', type=float, default=0.01, help='Running cost scaling weight')

    args = options.parseArgs()
    print(f"Results will be saved to: {args.save_dir}", flush=True)

    # Initialize environment
    map_str = str(args.map_file) if getattr(args, 'map_file', None) else None
    scen_str = str(args.scen_file) if getattr(args, 'scen_file', None) else None

    # Parse heterogeneous targets (dictionaries or legacy coordinate lists)
    raw_goals = getattr(args, 'goals', None)
    goal_configs = []

    if raw_goals:
        for g in raw_goals:
            if isinstance(g, dict):
                goal_configs.append(g)
            elif isinstance(g, (list, tuple)):
                goal_configs.append({'type': 'evader', 'position': [float(g[0]), float(g[1])], 'v_max': 15.0})

    pde_mesh = MAP2PDE(
        map_filepath=map_str,
        scen_filepath=scen_str,
        Lx=args.room_width,
        Ly=args.room_height,
        Nx=args.Nx,
        Ny=args.Ny,
        custom_goals=goal_configs
    )
    pde_mesh.parse_files(num_agents=args.num_agents)
    pde_mesh.build_spatial_mesh()

    # Configure solver
    mfg_solver = MFGSolver(
        pde_mesh_data=pde_mesh,
        T=args.T,
        Nt=args.Nt,
        thetaUM=args.relaxation_theta,
        goal_configs=goal_configs,
        goals_are_exits=args.goals_are_exits,
        obstacle_penalty=args.obstacle_penalty,
        running_cost_weight=args.running_cost_weight
    )

    # Execute 3-way Picard solver
    mfg_solver.run_picard_system(max_iters=args.max_iters)

    plotter = MFGPlotter(pde_mesh_data=pde_mesh, solver_instance=mfg_solver)

    dashboard_path = args.save_dir / "snapshots_dashboard.png"
    plotter.plot_snapshots(output_file=str(dashboard_path))

    frames_dir = args.save_dir / "frames"
    plotter.save_density_frames(output_dir=str(frames_dir))

    animation_path = args.save_dir / "pursuit_evasion.gif"
    plotter.create_movie(frame_dir=str(frames_dir), output_file=str(animation_path), fps=15)

    print("Pursuit-Evasion run completed successfully!", flush=True)


if __name__ == "__main__":
    main()
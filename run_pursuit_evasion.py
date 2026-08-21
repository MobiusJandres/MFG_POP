"""
Entry point for Pursuit-Evasion scenario using MAP2PDE benchmark/synthetic map.
"""
from args.Options import Options
from mfgames.geometry import MAP2PDE
from mfgames.problem import MFGSolver
from mfgames.plotting import MFGPlotter


def main():
    options = Options()
    options.parser.set_defaults(config='configs/pursuit_evasion.yml')
    options.parser.add_argument('--targets', default=None, nargs='*', help='Custom target coordinates [[x1, y1], [x2, y2], ...]')
    options.parser.add_argument(
        '--targets_are_exits', 
        action='store_true', 
        default=False, 
        help='If True, targets act as exits and absorb pursuer density upon capture.'
    )
    args = options.parseArgs()
    print(f"Results will be saved to: {args.save_dir}", flush=True)

    # Initialize environment
    map_str = str(args.map_file) if getattr(args, 'map_file', None) else None
    scen_str = str(args.scen_file) if getattr(args, 'scen_file', None) else None

    # Parse target coordinates (handles nested YAML lists or flat CLI nargs)
    raw_targets = getattr(args, 'targets', None)
    custom_targets = None

    if raw_targets:
        if isinstance(raw_targets[0], (int, float, str)) and not isinstance(raw_targets[0], (list, tuple)):
            # Handle flat CLI arguments: --targets 100 700 700 700 384 100
            coords = [float(x) for x in raw_targets]
            custom_targets = [coords[i:i + 2] for i in range(0, len(coords), 2)]
        else:
            # Handle nested YAML list
            custom_targets = [[float(c) for c in t] for t in raw_targets]

    # Instantiate MovingAI map with target overrides
    pde_mesh = MAP2PDE(
        map_filepath=map_str,
        scen_filepath=scen_str,
        Lx=args.room_width,
        Ly=args.room_height,
        Nx=args.Nx,
        Ny=args.Ny,
        custom_targets=custom_targets  # Passes parsed targets to MAP2PDE
    )
    pde_mesh.parse_files(num_agents=args.num_agents)
    pde_mesh.build_spatial_mesh()

    # Configure solver
    mfg_solver = MFGSolver(
        pde_mesh_data=pde_mesh,
        T=args.T,
        Nt=args.Nt,
        thetaUM=args.relaxation_theta,
        is_pursuit_evasion=True,
        v_max_evader=args.v_max_evader,
        targets_are_exits=args.targets_are_exits
    )

    # Execute 3-way Picard solver
    mfg_solver.run_picard_system(max_iters=args.max_iters)

    # Generate output artifacts
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
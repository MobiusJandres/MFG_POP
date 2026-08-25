"""
Traffic Evacuation Mean Field Game entry point.
"""
from args.Options import Options
from mfgames.geometry import (
    MFGTrafficGeometry,
    create_moving_door_mask,
    build_door_trajectories_from_config,
)
from mfgames.problem import MFGSolver
from mfgames.plotting import MFGPlotter

# Default static exit doors at bottom-left and bottom-right corners
DEFAULT_STATIC_DOORS = [
    {'x1': "2.0", 'x2': "10.0", 'y1': "0.0", 'y2': "5.0"},
    {'x1': "room_width - 10.0", 'x2': "room_width - 2.0", 'y1': "0.0", 'y2': "5.0"}
]


def main():
    options = Options()
    options.parser.set_defaults(config='configs/mfg_traffic.yml')
    options.parser.add_argument('--obstacle_penalty', type=float, default=500.0, help='Obstacle cell potential penalty (+500 for traffic)')

    args = options.parseArgs()
    print(f"Results will be saved to: {args.save_dir}", flush=True)

    # Initialize traffic geometry
    mesh = MFGTrafficGeometry(
        Lx=args.room_width,
        Ly=args.room_height,
        Nx=args.Nx,
        Ny=args.Ny
    )
    mesh.build_spatial_mesh()

    # Extract doors from config or fall back to DEFAULT_STATIC_DOORS
    doors_cfg = getattr(args, 'doors', None) or DEFAULT_STATIC_DOORS

    # Build door trajectories from config
    door_trajectories = build_door_trajectories_from_config(
        doors_cfg, args.T, args.room_width, args.room_height
    )
    door_mask_3d = create_moving_door_mask(
        door_trajectories, args.Nt, args.Nx, args.Ny, mesh.X, mesh.Y, args.T
    )

    goals_are_exits = getattr(args, 'goals_are_exits', True)

    # Instantiate solver with positive obstacle penalty
    solver = MFGSolver(
        pde_mesh_data=mesh,
        T=args.T,
        Nt=args.Nt,
        thetaUM=args.relaxation_theta,
        door_mask_3d=door_mask_3d,
        goals_are_exits=goals_are_exits,
        obstacle_penalty=args.obstacle_penalty
    )

    solver.run_picard_system(max_iters=args.max_iters)

    # Generate visualizations
    plotter = MFGPlotter(pde_mesh_data=mesh, solver_instance=solver)

    dashboard_path = args.save_dir / "snapshots_dashboard.png"
    plotter.plot_snapshots(output_file=str(dashboard_path))

    frames_dir = args.save_dir / "frames"
    plotter.save_density_frames(output_dir=str(frames_dir))

    animation_path = args.save_dir / "traffic_simulation.gif"
    plotter.create_movie(frame_dir=str(frames_dir), output_file=str(animation_path), fps=15)

    print("Traffic run completed successfully!", flush=True)


if __name__ == "__main__":
    main()
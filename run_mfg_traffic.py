"""
Entry point for original MFG Traffic scenario using primitive shapes and moving doors.
"""
import numpy as np
from args.Options import Options
from mfgames.geometry import (
    MFGTrafficGeometry,
    create_moving_door_mask,
    build_door_trajectories_from_config,
)
from mfgames.problem import MFGSolver
from mfgames.plotting import MFGPlotter


def main():
    options = Options()
    options.parser.set_defaults(config='configs/mfg_traffic.yml')
    args = options.parseArgs()

    print(f"Results will be saved to: {args.save_dir}", flush=True)

    # 1. Initialize geometry and spatial mesh
    pde_mesh = MFGTrafficGeometry(Lx=args.room_width, Ly=args.room_height, Nx=args.Nx, Ny=args.Ny)
    pde_mesh.build_spatial_mesh()

    # 2. Build door trajectories from YAML config or fallback defaults
    if hasattr(args, 'doors') and args.doors:
        door_trajectories = build_door_trajectories_from_config(
            args.doors,
            T_final=args.T,
            room_width=args.room_width,
            room_height=args.room_height
        )
    else:
        # Default STATIC door trajectories fallback:
        # Left door: x in [0, 10], y in [0, 5]
        # Right door: x in [40, 50], y in [0, 5]
        door_trajectories = [
            {
                'x1': lambda t: 0.0,
                'x2': lambda t: 10.0,
                'y1': lambda t: 0.0,
                'y2': lambda t: 5.0,
            },
            {
                'x1': lambda t: args.room_width - 10.0,
                'x2': lambda t: args.room_width,
                'y1': lambda t: 0.0,
                'y2': lambda t: 5.0,
            }
        ]

    door_mask_3d = create_moving_door_mask(
        door_trajectories,
        Nt=args.Nt,
        Nx=args.Nx,
        Ny=args.Ny,
        X=pde_mesh.X,
        Y=pde_mesh.Y,
        T=args.T
    )

    # 3. Solve HJB-FP coupled system
    mfg_solver = MFGSolver(
        pde_mesh_data=pde_mesh,
        T=args.T,
        Nt=args.Nt,
        thetaUM=args.relaxation_theta,
        door_mask_3d=door_mask_3d,
        is_pursuit_evasion=False
    )
    mfg_solver.run_picard_system(max_iters=args.max_iters)

    # 4. Generate visualizations and animations
    plotter = MFGPlotter(pde_mesh_data=pde_mesh, solver_instance=mfg_solver)

    dashboard_path = args.save_dir / "snapshots_dashboard.png"
    plotter.plot_snapshots(output_file=str(dashboard_path))

    frames_dir = args.save_dir / "frames"
    plotter.save_density_frames(output_dir=str(frames_dir))

    animation_path = args.save_dir / "traffic_animation.gif"
    plotter.create_movie(frame_dir=str(frames_dir), output_file=str(animation_path), fps=15)

    print("MFG Traffic run completed successfully!", flush=True)


if __name__ == "__main__":
    main()
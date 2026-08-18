"""
Entry point for Pursuit-Evasion scenario using MAP2PDE benchmark/synthetic map.
"""
from args.Options import Options
from mfgames.geometry import MAP2PDE
from mfgames.problem import MFGSolver
from mfgames.plotting import MFGPlotter


def main():
    options = Options()
    args = options.parseArgs()

    print(f"Results will be saved to: {args.save_dir}", flush=True)

    # Initialize environment
    map_str = str(args.map_file) if args.map_file else None
    scen_str = str(args.scen_file) if args.scen_file else None

    pde_mesh = MAP2PDE(
        map_filepath=map_str,
        scen_filepath=scen_str,
        Lx=args.room_width,
        Ly=args.room_height,
        Nx=args.Nx,
        Ny=args.Ny
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
        v_max_evader=args.v_max_evader
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
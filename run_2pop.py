"""
Example runner for 2-Population Mean Field Game simulation.
"""
from args.Options import Options
from mfgames.geometry import MAP2PDE
from mfgames.problem import MFG2PopSolver
from mfgames.plotting import MFGPlotter


def main():
    options = Options()

    # 1. Add population-specific arguments to the parser
    options.parser.set_defaults(config='configs/mfg_2pop.yml')
    options.parser.add_argument('--pop1_blobs', default=None, nargs='*', help='Population 1 blobs [[x, y, sigma, amp], ...]')
    options.parser.add_argument('--pop2_blobs', default=None, nargs='*', help='Population 2 blobs [[x, y, sigma, amp], ...]')
    options.parser.add_argument('--pop1_goals', default=None, nargs='*', help='Population 1 custom goals [[x, y], ...]')
    options.parser.add_argument('--pop2_goals', default=None, nargs='*', help='Population 2 custom goals [[x, y], ...]')

    args = options.parseArgs()
    print(f"Results will be saved to: {args.save_dir}", flush=True)

    map_str = str(args.map_file) if getattr(args, 'map_file', None) else None
    scen_str = str(args.scen_file) if getattr(args, 'scen_file', None) else None

    # 2. Initialize spatial grids for both populations
    mesh_pop1 = MAP2PDE(
        map_filepath=map_str,
        scen_filepath=scen_str,
        Lx=args.room_width,
        Ly=args.room_height,
        Nx=args.Nx,
        Ny=args.Ny
    )
    mesh_pop2 = MAP2PDE(
        map_filepath=map_str,
        scen_filepath=scen_str,
        Lx=args.room_width,
        Ly=args.room_height,
        Nx=args.Nx,
        Ny=args.Ny
    )

    mesh_pop1.parse_files(num_agents=getattr(args, 'num_agents', 1))
    mesh_pop2.parse_files(num_agents=getattr(args, 'num_agents', 1))

    mesh_pop1.build_spatial_mesh()
    mesh_pop2.build_spatial_mesh()

    # 3. Apply custom initial Gaussian density blobs from config/CLI
    pop1_raw = getattr(args, 'pop1_blobs', None) or [[400.0, 575.0, 40.0, 5.0]]
    pop2_raw = getattr(args, 'pop2_blobs', None) or [[400.0, 175.0, 40.0, 5.0]]

    pop1_blobs = [tuple(float(c) for c in b) for b in pop1_raw]
    pop2_blobs = [tuple(float(c) for c in b) for b in pop2_raw]

    mesh_pop1.set_custom_initial_blobs(pop1_blobs, normalize_mass=False)
    mesh_pop2.set_custom_initial_blobs(pop2_blobs, normalize_mass=False)

    # 4. Apply custom goals from config/CLI
    pop1_goals = getattr(args, 'pop1_goals', []) or []
    pop2_goals = getattr(args, 'pop2_goals', None) or [(400.0, 700.0)]

    mesh_pop1.set_custom_goals(pop1_goals)
    mesh_pop2.set_custom_goals(pop2_goals)

    # 5. Solve coupled 2-population system
    solver = MFG2PopSolver(
        pde_mesh_data_1=mesh_pop1,
        pde_mesh_data_2=mesh_pop2,
        T=args.T,
        Nt=args.Nt,
        thetaUM=args.relaxation_theta
    )
    solver.run_picard_system(max_iters=args.max_iters)

    # 6. Export outputs and visualizations
    plotter = MFGPlotter(mesh_pop1, solver, mesh_pop2)

    dashboard_path = args.save_dir / "snapshots_dashboard.png"
    plotter.plot_snapshots(output_file=str(dashboard_path))

    frames_dir = args.save_dir / "frames"
    plotter.save_density_frames(output_dir=str(frames_dir))

    animation_path = args.save_dir / "mfg_2pop_simulation.mp4"
    plotter.save_mp4(filename=str(animation_path), fps=30)

    print("2-Population run completed successfully!", flush=True)


if __name__ == "__main__":
    main()
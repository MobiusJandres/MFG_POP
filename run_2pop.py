"""
2-Population Mean Field Game Simulation Runner.

This script demonstrates a coupled two-population MFG system, typically used for
modeling pursuit-evasion or competitive crowd dynamics scenarios. Unlike the
single-population traffic/navigation MFG (run_mfg.py), this solver handles
interactions between two distinct agent populations with potentially conflicting
objectives.

Key MFG Concepts Illustrated
-----------------------------
- **Coupled HJB-KFP Systems**: Each population solves its own Hamilton-Jacobi-Bellman
  (value function) and Kolmogorov-Fokker-Planck (density evolution) equations, but
  the systems are coupled through interaction terms in the Hamiltonians.

- **Congestion/Interaction Coupling**: Each population's optimal control depends on
  both its own density distribution and the opposing population's density. This is
  typically implemented via congestion penalties or repulsion terms.

- **Pursuit-Evasion Dynamics**: Common use case where one population (pursuers)
  seeks to intercept/capture the other population (evaders), while evaders try to
  reach goals while avoiding pursuers.

- **Symmetric or Asymmetric Objectives**: Populations can have different goal
  configurations, movement capabilities (diffusion coefficients), and interaction
  strengths.

Expected Outputs
----------------
The script generates three types of visualization outputs in the specified save
directory (default: `output/2pop/`):

1. **snapshots_dashboard.png**: Multi-panel static plot showing density and value
   function evolution for both populations at key time snapshots (t=0, T/4, T/2, 3T/4, T).

2. **frames/**: Directory containing individual PNG frames for each time step, useful
   for detailed inspection or custom animation generation.

3. **mfg_2pop_simulation.mp4**: Animated video (30 fps) showing the full temporal
   evolution of both population densities and value functions.

Configuration
-------------
Simulation parameters are controlled via:
- **YAML config file**: `configs/mfg_2pop.yml` (default, can be overridden with --config)
- **Command-line arguments**: Override config values or specify custom initial
  conditions and goal locations

Key configurable parameters include:
- Spatial domain size (`room_width`, `room_height`)
- Grid resolution (`Nx`, `Ny`)
- Time horizon and discretization (`T`, `Nt`)
- Picard iteration settings (`max_iters`, `relaxation_theta`)
- Population-specific initial density blobs (`pop1_blobs`, `pop2_blobs`)
- Population-specific goal locations (`pop1_goals`, `pop2_goals`)

Usage Example
-------------
Basic run with default configuration:
    $ python run_2pop.py

Custom initial conditions via CLI:
    $ python run_2pop.py --pop1_blobs 300 500 50 8.0 --pop2_blobs 500 300 50 8.0

Custom goals for both populations:
    $ python run_2pop.py --pop1_goals 100 100 --pop2_goals 700 700

Override config file:
    $ python run_2pop.py --config configs/pursuit_evasion_custom.yml

See Also
--------
- run_mfg.py: Single-population traffic/navigation MFG
- configs/mfg_2pop.yml: Default configuration file
- mfgames/problem.py: MFG2PopSolver implementation
- mfgames/plotting.py: Visualization utilities
"""
# Import section
from args.Options import Options  # Command-line argument parsing and config file loading
from mfgames.geometry import MAP2PDE  # Spatial mesh construction and geometry handling
from mfgames.problem import MFG2PopSolver  # Coupled 2-population MFG solver (Picard iteration)
from mfgames.plotting import MFGPlotter  # Visualization tools for density/value function output


def main():
    """
    Execute a full 2-population Mean Field Game simulation workflow.

    This function orchestrates the complete simulation pipeline from configuration
    loading through solver execution to final visualization output. It handles both
    populations symmetrically, allowing independent specification of initial
    conditions and goal locations.

    Workflow Steps
    --------------
    1. Parse command-line arguments and load YAML configuration
    2. Initialize spatial grids (MAP2PDE meshes) for both populations
    3. Apply custom initial density distributions (Gaussian blobs)
    4. Configure goal locations for each population
    5. Solve the coupled 2-population system via Picard iteration
    6. Generate and save visualization outputs (snapshots, frames, animation)

    Configuration Parameters Used
    -----------------------------
    Spatial Domain:
        - room_width, room_height: Physical dimensions in meters
        - Nx, Ny: Grid resolution (number of cells in x and y directions)
        - map_file, scen_file: Optional obstacle/scenario definitions

    Temporal Discretization:
        - T: Time horizon for the simulation (seconds)
        - Nt: Number of time steps

    Solver Settings:
        - max_iters: Maximum number of Picard iterations for convergence
        - relaxation_theta: Under-relaxation parameter (0 < theta <= 1)
          Lower values improve stability but slow convergence

    Population-Specific Configuration:
        - pop1_blobs, pop2_blobs: Initial density as Gaussian blobs
          Format: [[x, y, sigma, amplitude], ...]
          Default: Pop1 at (400, 575), Pop2 at (400, 175)

        - pop1_goals, pop2_goals: Target locations for each population
          Format: [[x, y], ...]
          Default: Pop1 has no explicit goals (free movement), Pop2 targets (400, 700)

    Output Files:
        - save_dir: Base directory for all outputs (default: output/2pop/)
        - snapshots_dashboard.png: Multi-panel static visualization
        - frames/: Individual frame images for each time step
        - mfg_2pop_simulation.mp4: Animated video of full simulation

    Why Choices Matter
    ------------------
    - **Negative obstacle_penalty**: For pursuit-evasion scenarios, negative penalties
      in the Hamiltonian repel agents from obstacles more strongly than in pure
      navigation tasks, preventing cornering/trapping scenarios.

    - **Separate MAP2PDE instances**: Each population maintains its own spatial mesh
      to allow for population-specific boundary conditions, diffusion coefficients,
      or goal configurations.

    - **Non-normalized blob amplitudes**: Unlike single-population traffic models,
      pursuit-evasion scenarios often require precise density ratios between
      populations (e.g., 2:1 pursuer-to-evader ratio), so we disable automatic
      normalization and specify amplitudes explicitly.

    - **Pop1 default no goals**: In pursuit-evasion, pursuers often have no fixed
      spatial goal; their objective is purely to maximize overlap with evader density,
      which is encoded in the interaction coupling term rather than terminal cost.

    Returns
    -------
    None
        All outputs are written to disk in the configured save_dir.

    Raises
    ------
    ValueError
        If blob or goal specifications have incorrect format/dimensions.
    FileNotFoundError
        If specified config file, map file, or scenario file does not exist.
    ConvergenceError
        If Picard iteration fails to converge within max_iters (logged, not raised).
    """
    options = Options()

    # ========================================================================
    # STEP 1: Argument Parsing and Configuration Loading
    # ========================================================================
    # Extend the base Options parser with 2-population-specific command-line
    # arguments. This allows overriding YAML config values for initial conditions
    # and goals on a per-run basis without editing config files.
    options.parser.set_defaults(config='configs/mfg_2pop.yml')
    options.parser.add_argument('--pop1_blobs', default=None, nargs='*', help='Population 1 blobs [[x, y, sigma, amp], ...]')
    options.parser.add_argument('--pop2_blobs', default=None, nargs='*', help='Population 2 blobs [[x, y, sigma, amp], ...]')
    options.parser.add_argument('--pop1_goals', default=None, nargs='*', help='Population 1 custom goals [[x, y], ...]')
    options.parser.add_argument('--pop2_goals', default=None, nargs='*', help='Population 2 custom goals [[x, y], ...]')

    args = options.parseArgs()
    print(f"Results will be saved to: {args.save_dir}", flush=True)

    # Extract optional map/scenario file paths for obstacle geometry
    map_str = str(args.map_file) if getattr(args, 'map_file', None) else None
    scen_str = str(args.scen_file) if getattr(args, 'scen_file', None) else None

    # ========================================================================
    # STEP 2: Spatial Grid Initialization for Both Populations
    # ========================================================================
    # Create separate MAP2PDE mesh objects for each population. While they share
    # the same physical domain (Lx, Ly) and discretization (Nx, Ny), maintaining
    # separate instances allows for:
    # - Population-specific boundary conditions (e.g., different goal locations)
    # - Independent initial density distributions
    # - Potential future extensions: different diffusion coefficients or obstacle maps
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

    # Parse geometry files (if provided) and build coordinate arrays, cell centers,
    # and obstacle masks for the finite difference discretization
    mesh_pop1.parse_files(num_agents=getattr(args, 'num_agents', 1))
    mesh_pop2.parse_files(num_agents=getattr(args, 'num_agents', 1))

    mesh_pop1.build_spatial_mesh()
    mesh_pop2.build_spatial_mesh()

    # ========================================================================
    # STEP 3: Initial Density Configuration
    # ========================================================================
    # Set up initial population distributions as Gaussian blobs. Each blob is
    # specified as [x_center, y_center, sigma, amplitude].
    # Default configuration: Pop1 starts at (400, 575), Pop2 at (400, 175)
    # Both populations have sigma=40 (spread) and amplitude=5 (peak density).
    pop1_raw = getattr(args, 'pop1_blobs', None) or [[400.0, 575.0, 40.0, 5.0]]
    pop2_raw = getattr(args, 'pop2_blobs', None) or [[400.0, 175.0, 40.0, 5.0]]

    # Convert to tuples of floats for geometry module API
    pop1_blobs = [tuple(float(c) for c in b) for b in pop1_raw]
    pop2_blobs = [tuple(float(c) for c in b) for b in pop2_raw]

    # Apply blobs without normalization. Unlike single-population traffic models,
    # pursuit-evasion scenarios often require specific density ratios between
    # populations (e.g., more pursuers than evaders), so we preserve absolute
    # amplitude values rather than normalizing to unit mass.
    mesh_pop1.set_custom_initial_blobs(pop1_blobs, normalize_mass=False)
    mesh_pop2.set_custom_initial_blobs(pop2_blobs, normalize_mass=False)

    # ========================================================================
    # STEP 4: Goal Location Configuration
    # ========================================================================
    # Define target locations for each population. Goals affect the terminal cost
    # in the value function (HJB equation) and bias agent movement trajectories.
    #
    # Default configuration for pursuit-evasion:
    # - Pop1 (pursuers): No explicit spatial goals (empty list)
    #   Their objective is encoded via interaction coupling term (maximize overlap
    #   with Pop2 density) rather than a fixed terminal cost.
    # - Pop2 (evaders): Single goal at (400, 700)
    #   Evaders have a clear spatial objective (escape to exit/safe zone) while
    #   avoiding pursuers.
    pop1_goals = getattr(args, 'pop1_goals', []) or []
    pop2_goals = getattr(args, 'pop2_goals', None) or [(400.0, 700.0)]

    mesh_pop1.set_custom_goals(pop1_goals)
    mesh_pop2.set_custom_goals(pop2_goals)

    # ========================================================================
    # STEP 5: Coupled System Solution via Picard Iteration
    # ========================================================================
    # Initialize the 2-population MFG solver with configured mesh data and temporal
    # parameters. The solver will iterate between solving:
    # - HJB equations (backward in time): Compute optimal value functions u1, u2
    #   given current density estimates m1, m2
    # - KFP equations (forward in time): Compute evolved densities m1, m2 given
    #   optimal controls derived from u1, u2
    # The systems are coupled through interaction terms in the Hamiltonians
    # (e.g., congestion penalties, pursuit-evasion forces).
    #
    # relaxation_theta (thetaUM): Under-relaxation parameter for Picard iteration
    # - theta=1.0: No relaxation (full update each iteration, may diverge)
    # - theta<1.0: Blend old and new solutions for stability
    # Typical range: 0.5-0.9 for pursuit-evasion, 0.9-1.0 for simple traffic
    solver = MFG2PopSolver(
        pde_mesh_data_1=mesh_pop1,
        pde_mesh_data_2=mesh_pop2,
        T=args.T,
        Nt=args.Nt,
        thetaUM=args.relaxation_theta
    )
    # Run Picard iteration until convergence or max_iters reached. Convergence
    # diagnostics (L2 error norms) are printed to stdout during execution.
    solver.run_picard_system(max_iters=args.max_iters)

    # ========================================================================
    # STEP 6: Visualization Generation and Export
    # ========================================================================
    # Create a plotter instance with access to both population mesh data and the
    # solved system. The plotter generates multi-population visualizations showing
    # density and value function evolution for both populations simultaneously.
    plotter = MFGPlotter(mesh_pop1, solver, mesh_pop2)

    # Generate static dashboard: 5 time snapshots (t=0, T/4, T/2, 3T/4, T) with
    # 4 subplots each (Pop1 density, Pop1 value, Pop2 density, Pop2 value)
    dashboard_path = args.save_dir / "snapshots_dashboard.png"
    plotter.plot_snapshots(output_file=str(dashboard_path))

    # Export individual frames for each time step. Useful for detailed inspection
    # or custom animation generation with external tools.
    frames_dir = args.save_dir / "frames"
    plotter.save_density_frames(output_dir=str(frames_dir))

    # Create animated MP4 video (30 fps) showing full temporal evolution of both
    # population densities and value functions. Requires ffmpeg to be available.
    animation_path = args.save_dir / "mfg_2pop_simulation.mp4"
    plotter.save_mp4(filename=str(animation_path), fps=30)

    print("2-Population run completed successfully!", flush=True)


# Script entry point: Execute main() when run directly (not imported as module)
if __name__ == "__main__":
    main()
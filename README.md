# mfgames

A Python library for solving **Mean Field Games (MFG)** with applications to crowd traffic flow, pursuit-evasion dynamics, and multi-population interactions. Built with Numba-accelerated finite difference methods for fast, scalable simulations of crowd dynamics and optimal control problems.

## Overview

**mfgames** implements numerical solvers for coupled Hamilton-Jacobi-Bellman (HJB) and Kolmogorov-Fokker-Planck (KFP) partial differential equation systems using Picard iteration. The library models large-scale agent populations as continuous density fields evolving under optimal control strategies with congestion effects.

### What Problems Does It Solve?

- **Crowd Traffic Flow**: Model evacuation dynamics through moving doors with congestion penalties
- **Multi-Population Interactions**: Couple multiple agent populations with distinct objectives and interaction costs
- **Crowd Dynamics with Obstacles**: Handle complex geometries imported from MovingAI benchmark maps

## Features

- **Three Simulation Modes**:
  - Traffic evacuation with static or moving exits
  - Pursuit-evasion with reactive goals
  - Two-population games with coupled dynamics

- **Flexible Goal Types**:
  - Stationary goals
  - Prescribed trajectories (user-defined parametric paths)
  - Evader (reactive goals using heuristic repulsive force dynamics)

- **Numerics**:
  - Numba JIT compilation for sparse matrix assembly
  - Implicit finite difference schemes with upwind discretization
  - Picard iteration for nonlinear coupling with under-relaxation

- **YAML-Driven Configuration**: Declarative experiment setup with support for mathematical expressions in goal trajectories

- **Comprehensive Visualization**:
  - Density and value function heatmaps
  - Time-evolution animations (GIF/MP4)
  - Evader trajectory overlays

## Mathematical Background

Mean Field Games model large populations of rational agents making optimal decisions in a congested environment. The system is described by:

**Hamilton-Jacobi-Bellman (HJB) Equation** (optimal control):
```
∂u/∂t + ν Δu - H(x, m, ∇u) = -g(x)
```

**Kolmogorov-Fokker-Planck (KFP) Equation** (density evolution):
```
∂m/∂t - ν Δm - div(m ∇_p H(x, m, ∇u)) = 0
```

Where:
- `u(x,t)`: Value function (cost-to-go to reach goals)
- `m(x,t)`: Agent density distribution
- `H(x,m,p)`: Hamiltonian encoding control cost and congestion effects
- `g(x)`: Running cost (distance to goals, obstacle penalties)
- `ν`: Diffusion coefficient (agent randomness)

The **congestive Hamiltonian** used in this library:
```
H(x, m, p) = -8|p|²/(1+m)^(3/4) + 1/3200
```

This models agents moving optimally toward low-cost regions while slowing down in congested areas.

### Numerical Scheme

- **Spatial Discretization**: 5-point finite difference stencil on Cartesian grids
- **Time Discretization**: Implicit backward Euler for HJB, implicit forward Euler for KFP
- **Coupling**: Picard iteration alternates solving HJB (backward in time) and KFP (forward in time) until convergence
- **Boundary Conditions**:
  - Dirichlet at exits/doors (u = 0)
  - Neumann at solid walls (zero normal derivative)

## Installation

### Requirements

- Python ≥ 3.11
- Core dependencies: `numpy`, `scipy`, `numba`, `matplotlib`, `pyyaml`, `pillow`
- Optional: `imageio` for MP4 video export

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/c-JAndres/MFG_POP.git
cd MFG_POP

# Install with uv (includes all dependencies)
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Using pip

```bash
pip install -e .

# Optional: Install video export support
pip install -e ".[video]"

# Optional: Install development tools
pip install -e ".[dev]"
```

## Quick Start

### Basic Traffic Evacuation

```python
from mfgames import MAP2PDE, MFGSolver, MFGPlotter

# Define spatial domain
mesh = MAP2PDE(Lx=50.0, Ly=50.0, Nx=75, Ny=75)
mesh.build_spatial_mesh()

# Configure solver
solver = MFGSolver(
    pde_mesh_data=mesh,
    T=100.0,            # Simulation time (seconds)
    Nt=600,             # Time steps
    thetaUM=0.1,        # Relaxation parameter
    goals_are_exits=True,
    obstacle_penalty=500.0
)

# Run Picard iteration
solver.run_picard_system(max_iters=25)

# Visualize results
plotter = MFGPlotter(mesh, solver)
plotter.plot_snapshots(output_file="results.png")
plotter.create_movie(frame_dir="frames", output_file="simulation.gif", fps=15)
```

### Running Example Scripts

The repository includes three ready-to-run examples:

```bash
# 1. Traffic evacuation (50m × 50m room with exit doors)
python run_mfg_traffic.py

# 2. Pursuit-evasion (multiple moving goals on MovingAI map)
python run_pursuit_evasion.py

# 3. Two-population game (coupled crowds with distinct goals)
python run_2pop.py
```

All scripts read configuration from `configs/*.yml` and save outputs to `results/<timestamp>/`.

## Examples

### 1. Traffic Evacuation (`run_mfg_traffic.py`)

Models crowd evacuation through exit doors with congestion effects.

**Key Features**:
- Static or moving door trajectories (configurable via YAML)
- Positive obstacle penalty (repulsive potential)
- Exits absorb mass (`goals_are_exits=True`)

**Configuration** (`configs/mfg_traffic.yml`):
```yaml
room_width: 50.0
room_height: 50.0
Nx: 75
Ny: 75
T: 100.0
Nt: 600
max_iters: 25
obstacle_penalty: 500.0
goals_are_exits: true
```

**Outputs**: Density evolution animation, value function snapshots, dashboard plots.

### 2. Pursuit-Evasion (`run_pursuit_evasion.py`)

Simulates a crowd chasing moving goals that optimize their escape routes.

**Key Features**:
- Moving goals react to crowd density gradient
- Heterogeneous goal types (stationary, prescribed paths, evaders)
- MovingAI map import for realistic obstacle layouts
- Negative obstacle penalty (attractive potential for pursuers)

**Configuration** (`configs/pursuit_evasion.yml`):
```yaml
map_file: "MAP2PDE/Maps/AcrosstheCape.map"
scen_file: "MAP2PDE/Scenarios/AcrosstheCape_1g.map.scen"
room_width: 768.0
room_height: 768.0
Nx: 100
Ny: 100
T: 5.0
Nt: 150
goals_are_exits: false
obstacle_penalty: -50000.0

goals:
  - type: "evader"
    position: [100.0, 700.0]
    v_max: 25.0
  - type: "evader"
    position: [700.0, 700.0]
    v_max: 25.0
```

**Outputs**: Evader trajectory overlays, pursuer density heatmaps, animated pursuit dynamics.

#### Pursuit-Evasion Dynamics: Heuristic Evader Model

The pursuit-evasion mode uses a **simplified heuristic approach** for evader dynamics that provides intuitive, computationally efficient behavior while maintaining the full MFG framework.

**Evader Dynamics Implementation** (`mfgames/evasion.py`):

Rather than solving a full game-theoretic optimal control problem, evaders use a **greedy repulsive force model**:

1. **Repulsive Force Calculation**:
   ```python
   # Compute inverse-square Coulomb-like repulsion from density field
   F_x = Σ_grid [(x_evader - x_i) / (dist² + ε)] * m(x_i)
   F_y = Σ_grid [(y_evader - y_i) / (dist² + ε)] * m(x_i)
   ```
   where m(x_i) is the pursuer density at grid point i, and ε = 1e-3 prevents singularities.

2. **Velocity Normalization**:
   ```python
   # Normalize direction and scale by maximum velocity
   v = (F / ||F||) * v_max
   y_new = y_old + v * Δt
   ```

3. **Integration with MFG Solver**:
   - Each Picard iteration: HJB → KFP → **Evader Update** → repeat
   - Pursuers solve full HJB-KFP system with dynamic goal y(t)
   - Evaders respond myopically to current pursuer distribution
   - Under-relaxation applied to trajectories: y^(k) = θ·y_new + (1-θ)·y^(k-1)

**Physical Interpretation**:

The evader moves **directly away from the weighted center of mass** of nearby pursuers. Close pursuers contribute more strongly (inverse-square law), creating a "flee to open space" behavior similar to electrostatic repulsion. This produces realistic evasion without solving nested optimal control problems.

**Comparison to Full Game-Theoretic Framework**:

A rigorous **Major-Minor Mean Field Game** formulation (not currently implemented) would include:

- **Evader cost functional**: Minimize cumulative exposure to crowd dyanics + control effort penalty
- **Pontryagin's Maximum Principle**: Derive optimal velocity via costate (adjoint) equations
- **Backward-forward coupling**: Costate integrated backward from terminal cost, trajectory forward from initial position
- **Nash Equilibrium**: Both pursuers and evaders optimally respond to each other

**Current Heuristic Benefits**:
- ✅ **Fast**: One force summation per evader per timestep (~30 lines of code)
- ✅ **Intuitive**: Clear physical interpretation (repulsive force)
- ✅ **Stable**: No nested optimization or adjoint solvers required
- ✅ **Tunable**: Single parameter v_max controls evasion speed

**Full Optimal Control Trade-offs**:
- ⚙️ **Optimal**: Game-theoretically provable best response
- ⚙️ **Foresight**: Plans over full time horizon with terminal objectives
- ⚠️ **Complex**: Requires costate solver, terminal cost specification, control penalty tuning
- ⚠️ **Expensive**: Backward adjoint integration at each Picard iteration

**Practical Use Cases**:

The heuristic model is well-suited for:
- Qualitative pursuit-evasion scenario exploration
- Real-time or interactive applications
- Fast prototyping of crowd dynamics

For safety-critical systems requiring provable optimality or formal game-theoretic guarantees, implementing the full Major-Minor framework (costate equations, terminal costs, Pontryagin's Principle) would be the recommended extension.

**Configuration**:
```yaml
goals:
  - type: "evader"
    position: [x, y]      # Initial evader position
    v_max: 25.0           # Maximum evasion velocity (m/s)
```

### 3. Two-Population Game (`run_2pop.py`)

Couples two distinct populations with separate goals and interaction costs.

**Key Features**:
- Independent initial Gaussian blobs for each population
- Separate goal configurations
- Coupled dynamics via interaction Hamiltonian
- Synchronized visualization of both populations

**Configuration** (`configs/mfg_2pop.yml`):
```yaml
pop1_blobs: [[400.0, 575.0, 40.0, 5.0]]  # [x, y, sigma, amplitude]
pop2_blobs: [[400.0, 175.0, 40.0, 5.0]]
pop1_goals: []
pop2_goals: [[400.0, 700.0]]
T: 5.0
Nt: 150
max_iters: 15
```

**Outputs**: Side-by-side density evolution, interaction metrics, MP4 animation.

## Package Structure

```
mfgames/
├── __init__.py          # Package exports
├── problem.py           # MFGSolver, MFG2PopSolver (high-level OOP interface)
├── solvers.py           # solveFP_2D, solveHJB_withM (low-level PDE solvers)
├── numerics.py          # Numba-compiled matrix assembly and FD operators
├── geometry.py          # MAP2PDE, MFGTrafficGeometry (spatial grids, obstacles)
├── evasion.py           # Goal (dynamic goal management)
├── plotting.py          # MFGPlotter (visualization and animation)
└── time.py              # Timestamped directory utilities
```

### Module Overview

- **`problem.py`**: Object-oriented solvers wrapping the full Picard iteration workflow
  - `MFGSolver`: Single-population traffic and pursuit-evasion
  - `MFG2PopSolver`: Two-population coupled system

- **`solvers.py`**: Low-level PDE solution routines called by `MFGSolver`
  - `solveFP_2D()`: Forward Fokker-Planck density evolution
  - `solveHJB_withM()`: Backward Hamilton-Jacobi value function with Newton iteration

- **`numerics.py`**: Numba-accelerated finite difference operators
  - `compute_FP_matrix_entries()`: Sparse COO assembly for KFP transport
  - `compute_HJB_matrix_entries()`: Sparse COO assembly for HJB system
  - `getFnU_2D()`: Nonlinear residual for Newton solver

- **`geometry.py`**: Spatial domain setup
  - `MAP2PDE`: Import MovingAI maps, set custom initial densities, define goals
  - `MFGTrafficGeometry`: Simple rectangular domains for evacuation scenarios
  - `create_moving_door_mask()`: Time-varying exit boundary conditions

- **`evasion.py`**: Dynamic goal management
  - `Goal`: Heterogeneous goal handler (stationary, prescribed, evader)

- **`plotting.py`**: Visualization toolkit
  - `MFGPlotter`: Generate heatmaps, snapshots, GIF/MP4 animations
  - Supports overlaying evader trajectories on density fields

## Configuration

All example scripts use YAML configuration files for declarative experiment setup.

### Common Configuration Parameters

```yaml
# Experiment metadata
exp_name: "my_experiment"
run_name: "test_run"
results_dir: "results"

# Spatial domain
room_width: 50.0
room_height: 50.0
Nx: 75                    # Grid points in x
Ny: 75                    # Grid points in y

# Time discretization
T: 100.0                  # Simulation duration (seconds)
Nt: 600                   # Number of timesteps

# Solver parameters
max_iters: 25             # Picard iterations
relaxation_theta: 0.1     # Under-relaxation (0 < θ < 1)

# Physical parameters
obstacle_penalty: 500.0   # Obstacle repulsion strength
goals_are_exits: true     # Whether goals absorb mass
```

### Dynamic Goal Trajectories

Goals can follow parametric paths using mathematical expressions:

```yaml
goals:
  - type: "prescribed"
    position: [100.0, 100.0]
    path_x: "100.0 + 50.0 * sin(2 * pi * t / T)"
    path_y: "100.0 + 50.0 * cos(2 * pi * t / T)"
```

Expressions support:
- `t`: Current time
- `T`: Total simulation time
- `room_width`, `room_height`: Domain dimensions
- Standard math functions: `sin`, `cos`, `exp`, `sqrt`, `pi`

## API Overview

### Key Classes

#### `MFGSolver`

Main solver for single-population systems.

```python
solver = MFGSolver(
    pde_mesh_data,           # Geometry object (MAP2PDE or MFGTrafficGeometry)
    T=100.0,                 # Simulation time
    Nt=600,                  # Time steps
    thetaUM=0.1,             # Relaxation parameter
    door_mask_3d=None,       # 3D array for time-varying exits
    goal_configs=None,       # List of goal dictionaries
    goals_are_exits=False,   # Whether to absorb mass at goals
    obstacle_penalty=-500.0, # Obstacle potential strength
    running_cost_weight=0.01 # Distance cost scaling
)

solver.run_picard_system(max_iters=25)  # Execute solver
```

**Attributes**:
- `solver.M`: Density array `(Nt+1, Nx, Ny)`
- `solver.U`: Value function array `(Nt+1, Nx, Ny)`
- `solver.goal`: Goal manager (if goals configured)

#### `MAP2PDE`

Geometry handler for MovingAI maps and custom initial conditions.

```python
mesh = MAP2PDE(
    map_filepath="path/to/map.map",
    scen_filepath="path/to/scenario.scen",
    Lx=768.0,
    Ly=768.0,
    Nx=100,
    Ny=100,
    custom_goals=[...]
)

mesh.parse_files(num_agents=50)
mesh.build_spatial_mesh()

# Customize initial density
mesh.set_custom_initial_blobs([(x, y, sigma, amplitude), ...])
```

#### `MFGPlotter`

Visualization toolkit for generating plots and animations.

```python
plotter = MFGPlotter(pde_mesh_data=mesh, solver_instance=solver)

# Generate dashboard with multiple time snapshots
plotter.plot_snapshots(output_file="dashboard.png")

# Export density frames
plotter.save_density_frames(output_dir="frames/")

# Create animation
plotter.create_movie(frame_dir="frames/", output_file="anim.gif", fps=15)
plotter.save_mp4(filename="anim.mp4", fps=30)  # Requires imageio
```

### Key Functions

#### Low-Level PDE Solvers

```python
from mfgames.solvers import solveFP_2D, solveHJB_withM

# Solve Fokker-Planck equation forward in time
M_new = solveFP_2D(M_prev, U_field, omask, door_mask, Nx, Ny, Dx, Dy, Dt)

# Solve Hamilton-Jacobi-Bellman backward in time with Newton iteration
U_new = solveHJB_withM(
    U_prev, M_field, omask, door_mask, running_cost,
    Nx, Ny, Dx, Dy, Dt, tol=1e-8, max_newton_iters=10
)
```

## Performance

### Numba JIT Compilation

All performance-critical functions are decorated with `@jit(nopython=True, cache=True)`:

- **First run**: ~30 seconds compilation time (cached for subsequent runs)
- **Subsequent runs**: Near-C performance for sparse matrix assembly and FD operators
- **Sparse solvers**: `scipy.sparse.linalg.spsolve` for large linear systems

### Typical Performance

On a modern CPU (example: Intel i7-9700K):

| Scenario | Grid Size | Time Steps | Picard Iters | Runtime |
|----------|-----------|------------|--------------|---------|
| Traffic | 75×75 | 600 | 25 | ~2 min |
| Pursuit-Evasion | 100×100 | 150 | 15 | ~3 min |
| 2-Population | 75×75 | 150 | 15 | ~4 min |

**Optimization Tips**:
- Increase `relaxation_theta` (0.1 → 0.3) for faster convergence at risk of instability
- Reduce grid resolution for prototyping (Nx=50 is ~4× faster than Nx=100)
- Use fewer Picard iterations for debugging (10 iters sufficient for qualitative behavior)

## Dependencies

### Core Dependencies

```
numpy >= 1.26.4        # Array operations and linear algebra
scipy >= 1.10.1        # Sparse matrix solvers
numba >= 0.66.0        # JIT compilation for performance
matplotlib >= 3.11.1   # Visualization
pyyaml >= 6.0.1        # Configuration file parsing
pillow >= 9.3.0        # Image processing
```

### Optional Dependencies

```
imageio >= 2.22.0      # MP4 video export
pytest >= 7.0          # Testing framework (dev)
black                  # Code formatting (dev)
```

Install optional dependencies:
```bash
pip install mfgames[video]  # Add MP4 support
pip install mfgames[dev]    # Add development tools
```

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## Citation

If you use this library in academic work, please cite:

```bibtex
@software{mfgames2024,
  title = {mfgames: Mean Field Games Library for Crowd Dynamics},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourrepo/mfgames}
}
```

## Acknowledgments

- Mean Field Games formulation based on [Lasry-Lions 2007] and [Achdou-Capuzzo-Dolcetta 2010]
- MovingAI map format from [Nathan Sturtevant's pathfinding benchmarks](https://movingai.com/benchmarks/)
- Numerical schemes adapted from [Achdou et al. 2012] finite difference methods for MFG

## Contact

For questions, bug reports, or feature requests, please open an issue on the repository or contact the maintainers at `your.email@example.com`.

---

**Project Status**: Beta (v0.1.0) — Active development, API subject to change

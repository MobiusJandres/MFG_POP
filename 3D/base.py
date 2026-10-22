# %%

import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh as gmshio
import ufl

# ==========================================
# 1. Mesh Generation in Gmsh
# ==========================================
gmsh.initialize()
gmsh.model.add("skyscraper_mfg")
occ = gmsh.model.occ

# Bounding Air Box & 4 Skyscrapers
air_box = occ.addBox(0, 0, 0, 100.0, 100.0, 50.0)
b1 = occ.addBox(15, 15, 0, 20, 20, 35)
b2 = occ.addBox(60, 15, 0, 25, 15, 40)
b3 = occ.addBox(15, 60, 0, 15, 25, 30)
b4 = occ.addBox(55, 55, 0, 25, 25, 45)

occ.cut([(3, air_box)], [(3, b1), (3, b2), (3, b3), (3, b4)])
occ.synchronize()

gmsh.option.setNumber("Mesh.MeshSizeMin", 3.0)
gmsh.option.setNumber("Mesh.MeshSizeMax", 10.0)

# Physical Groups
vol_entities = gmsh.model.getEntities(dim=3)
gmsh.model.addPhysicalGroup(3, [e[1] for e in vol_entities], tag=1, name="air_domain")
surf_entities = gmsh.model.getEntities(dim=2)
gmsh.model.addPhysicalGroup(2, [s[1] for s in surf_entities], tag=2, name="boundaries")

gmsh.model.mesh.generate(3)

# Pipeline directly into FEniCSx
mesh_data = gmshio.model_to_mesh(
    gmsh.model, comm=MPI.COMM_WORLD, rank=0, gdim=3
)
domain = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags
gmsh.finalize()

# ==========================================
# 2. Time & Problem Parameters
# ==========================================
T_final = 10.0        # Time horizon [0, T]
N_steps = 100         # Number of time steps
dt = T_final / N_steps
nu = 0.5              # Agent diffusion coefficient
congestion_coef = 0.5 # Coupling weight for density penalty
max_mfg_iter = 30     # Outer fixed-point loop iterations
alpha_relax = 0.3     # Picard relaxation parameter
mfg_tolerance = 1e-4

V = fem.functionspace(domain, ("Lagrange", 1))
W = fem.functionspace(domain, ("DG", 0))

# Storage across time horizon
u_traj = [fem.Function(V) for _ in range(N_steps + 1)]
m_traj = [fem.Function(W) for _ in range(N_steps + 1)]

# Start and goal are in the free-space domain.
start_position = (10.0, 10.0, 10.0)
goal_position = (85.0, 85.0, 25.0)
initial_spread = 5.0
goal_cost_weight = 0.1
x_coord = ufl.SpatialCoordinate(domain)

# Initial density is a unit-mass Gaussian centered at the start.
m0_expr = ufl.exp(
    -sum((x_coord[i] - start_position[i]) ** 2 for i in range(3))
    / (2.0 * initial_spread**2)
)
m0_expression = fem.Expression(m0_expr, W.element.interpolation_points)
m0_func = fem.Function(W)
m0_func.name = "initial_density"
m0_func.interpolate(m0_expression)
initial_mass_local = fem.assemble_scalar(fem.form(m0_func * ufl.dx))
normalization_mass = domain.comm.allreduce(initial_mass_local, op=MPI.SUM)
if not np.isfinite(normalization_mass) or normalization_mass <= 0.0:
    raise ValueError(f"Initial Gaussian has invalid mass: {normalization_mass}")
m0_func.x.array[:] /= normalization_mass
m0_func.x.scatter_forward()
initial_mass = 1.0

# Use the initial profile as the fixed-point solver's starting guess.
for density in m_traj:
    density.x.array[:] = m0_func.x.array[:]
    density.x.scatter_forward()

# Terminal cost makes the destination the preferred location at t = T.
uT_expr = goal_cost_weight * sum(
    (x_coord[i] - goal_position[i]) ** 2 for i in range(3)
)
uT_func = fem.Expression(uT_expr, V.element.interpolation_points)


def require_finite(function, name):
    local_finite = bool(np.isfinite(function.x.array).all())
    globally_finite = domain.comm.allreduce(local_finite, op=MPI.LAND)
    if not globally_finite:
        raise FloatingPointError(f"{name} contains non-finite values")


# ==========================================
# 3. Solvers Setup
# ==========================================

# --- HJB backward step (implicit viscous HJB) ---
def solve_hjb_step(u_next, m_curr, dt_val, nu_val):
    v_test = ufl.TestFunction(V)
    u_curr = fem.Function(V)
    u_curr.x.array[:] = u_next.x.array[:]
    residual = (
        (u_curr - u_next) / dt_val * v_test
        + nu_val * ufl.inner(ufl.grad(u_curr), ufl.grad(v_test))
        + 0.5 * ufl.inner(ufl.grad(u_curr), ufl.grad(u_curr)) * v_test
        - congestion_coef * m_curr * v_test
    ) * ufl.dx
    problem = fem.petsc.NonlinearProblem(
        residual,
        u_curr,
        petsc_options_prefix="hjb_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_rtol": 1e-8,
            "snes_max_it": 30,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    u_curr = problem.solve()
    require_finite(u_curr, "HJB value function")
    return u_curr


# --- Fokker-Planck forward step (upwind finite-volume transport) ---
def solve_fp_step(m_prev, u_curr, dt_val, nu_val):
    m_trial = ufl.TrialFunction(W)
    v_test = ufl.TestFunction(W)
    # The optimal drift for this Hamiltonian is b = -grad(u).
    v_drift = -ufl.grad(u_curr)
    normal = ufl.FacetNormal(domain)
    cell_size = 2.0 * ufl.avg(ufl.CellDiameter(domain))
    normal_drift = ufl.dot(ufl.avg(v_drift), normal("+"))
    upwind_density = ufl.conditional(
        ufl.gt(normal_drift, 0.0), m_trial("+"), m_trial("-")
    )

    # DG0 upwind fluxes conserve mass; the face penalty approximates diffusion.
    # Omitting exterior fluxes imposes impermeable, zero-flux building walls.
    a = (1.0 / dt_val) * m_trial * v_test * ufl.dx
    a += normal_drift * upwind_density * ufl.jump(v_test) * ufl.dS
    a += (nu_val / cell_size) * ufl.jump(m_trial) * ufl.jump(v_test) * ufl.dS
    L = (1.0 / dt_val) * m_prev * v_test * ufl.dx

    problem = LinearProblem(
        a,
        L,
        petsc_options_prefix="fp_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    m_next = problem.solve()
    require_finite(m_next, "Fokker-Planck density")
    return m_next


# ==========================================
# 4. Outer MFG Coupling Loop
# ==========================================
print("Starting Mean Field Game Coupled Forward-Backward Iterations...")

converged = False
for it in range(max_mfg_iter):
    # --- Step A: Backward HJB Pass (from t=T down to t=0) ---
    u_traj[N_steps].interpolate(uT_func)
    
    for n in range(N_steps - 1, -1, -1):
        u_traj[n] = solve_hjb_step(u_traj[n + 1], m_traj[n + 1], dt, nu)
        
    # --- Step B: Forward Fokker-Planck Pass (from t=0 up to t=T) ---
    m_new_traj = [fem.Function(W) for _ in range(N_steps + 1)]
    m_new_traj[0].x.array[:] = m0_func.x.array[:]
    m_new_traj[0].x.scatter_forward()
    
    for n in range(0, N_steps):
        m_new_traj[n + 1] = solve_fp_step(m_new_traj[n], u_traj[n + 1], dt, nu)

    for n, density in enumerate(m_new_traj):
        mass_local = fem.assemble_scalar(fem.form(density * ufl.dx))
        mass = domain.comm.allreduce(mass_local, op=MPI.SUM)
        if not np.isclose(mass, initial_mass, rtol=1e-7, atol=1e-10):
            raise RuntimeError(
                f"Fokker-Planck mass changed at step {n}: "
                f"expected {initial_mass}, got {mass}"
            )

    # Check the full space-time trajectory in the global finite-element L2 norm.
    difference_squared_local = 0.0
    density_squared_local = 0.0
    for n in range(1, N_steps + 1):
        difference_squared_local += fem.assemble_scalar(
            fem.form((m_new_traj[n] - m_traj[n]) ** 2 * ufl.dx)
        )
        density_squared_local += fem.assemble_scalar(
            fem.form(m_new_traj[n] ** 2 * ufl.dx)
        )
    difference_squared = domain.comm.allreduce(difference_squared_local, op=MPI.SUM)
    density_squared = domain.comm.allreduce(density_squared_local, op=MPI.SUM)
    relative_diff = np.sqrt(
        difference_squared / max(density_squared, np.finfo(float).tiny)
    )
    print(
        f"Iteration {it+1:02d}/{max_mfg_iter:02d} | "
        f"Relative trajectory change L2: {relative_diff:.6e}"
    )
    
    # Relax update to stabilize Picard iterations
    for n in range(1, N_steps + 1):
        m_traj[n].x.array[:] = (1.0 - alpha_relax) * m_traj[n].x.array[:] + alpha_relax * m_new_traj[n].x.array[:]
        m_traj[n].x.scatter_forward()

    if relative_diff < mfg_tolerance:
        print("MFG System Converged Successfully!")
        converged = True
        break

if not converged:
    print(
        f"WARNING: MFG fixed-point iteration did not reach "
        f"{mfg_tolerance:.1e} within {max_mfg_iter} iterations."
    )

minimum_density_local = min(float(np.min(density.x.array)) for density in m_traj)
minimum_density = domain.comm.allreduce(minimum_density_local, op=MPI.MIN)
if minimum_density < -1e-10:
    raise RuntimeError(
        f"Upwind Fokker-Planck solve produced negative density "
        f"values (minimum {minimum_density:.3e})"
    )

# ==========================================
# 5. Export Time-Series Results to VTX / ParaView
# ==========================================
m_out = fem.Function(W)
m_out.name = "density"
with io.VTXWriter(
    domain.comm, "mfg_swarm_density.bp", m_out, engine="BP4"
) as vtx_m:
    for n in range(N_steps + 1):
        m_out.x.array[:] = m_traj[n].x.array[:]
        m_out.x.scatter_forward()
        vtx_m.write(n * dt)

m_out = fem.Function(V)
m_out.name = "value_function"
with io.VTXWriter(
    domain.comm, "mfg_value_function.bp", m_out, engine="BP4"
) as vtx_u:
    for n in range(N_steps + 1):
        m_out.x.array[:] = u_traj[n].x.array[:]
        m_out.x.scatter_forward()
        vtx_u.write(n * dt)

print("Exported full temporal MFG trajectories (mfg_swarm_density.bp & mfg_value_function.bp) for ParaView visualization!")
# %%

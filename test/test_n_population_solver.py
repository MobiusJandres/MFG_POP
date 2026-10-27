import numpy as np

from mfgames.problem import MFGSolver, Population


class DummyMesh:
    def __init__(self, Nx=6, Ny=5, dx=1.0, dy=1.0):
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.Lx = Nx * dx
        self.Ly = Ny * dy
        self.omask = np.zeros((Nx, Ny), dtype=np.float64)
        x = np.linspace(dx / 2.0, self.Lx - dx / 2.0, Nx)
        y = np.linspace(dy / 2.0, self.Ly - dy / 2.0, Ny)
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")
        self._m0 = np.zeros((Nx, Ny), dtype=np.float64)
        self._goals = []

    def get_pde_obstacle_mask(self):
        return self.omask

    def build_initial_density(self, *args, **kwargs):
        return self._m0.copy()

    def get_goals(self):
        return list(self._goals)


def make_population(name, center, amplitude):
    mesh = DummyMesh()
    cx, cy = center
    density = amplitude * np.exp(-((mesh.X - cx) ** 2 + (mesh.Y - cy) ** 2) / 2.0)
    return Population(
        name=name,
        mesh=mesh,
        m0=density,
        goal_configs=[{"type": "stationary", "position": [cx, cy]}],
        running_cost_weight=0.01,
        terminal_cost_weight=0.02,
        obstacle_penalty=-10.0,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        h0=0.0,
    )


def test_three_population_solver_runs_and_preserves_shapes():
    populations = [
        make_population("alpha", (1.0, 1.0), 0.8),
        make_population("beta", (3.0, 2.0), 0.6),
        make_population("gamma", (5.0, 4.0), 0.4),
    ]
    interaction = np.array(
        [
            [0.0, 0.2, -0.1],
            [-0.3, 0.0, 0.15],
            [0.05, -0.2, 0.0],
        ],
        dtype=np.float64,
    )
    congestion = np.array(
        [
            [0.0, 0.4, 0.1],
            [0.2, 0.0, 0.3],
            [0.1, 0.25, 0.0],
        ],
        dtype=np.float64,
    )

    solver = MFGSolver(
        populations=populations,
        T=0.2,
        Nt=2,
        thetaUM=0.5,
        interaction_matrix=interaction,
        congestion_matrix=congestion,
    )

    results = solver.run_picard_system(max_iters=1, tolerance=0.0)

    assert isinstance(results, tuple)
    assert len(results) == 6

    for pop in solver.populations:
        assert pop.U.shape == (solver.Nt + 1, solver.Nx, solver.Ny)
        assert pop.M.shape == (solver.Nt + 1, solver.Nx, solver.Ny)
        assert np.all(np.isfinite(pop.U))
        assert np.all(np.isfinite(pop.M))
        assert np.all(pop.M >= -1e-10)

    other_cong, interaction_cost = solver._get_coupling_fields(0, 0)
    expected_cong = congestion[0, 1] * solver.populations[1].M[0] + congestion[0, 2] * solver.populations[2].M[0]
    expected_interaction = (
        interaction[0, 1] * solver.populations[1].M[0]
        + interaction[0, 2] * solver.populations[2].M[0]
        + np.maximum(
            solver.populations[0].M[0] + solver.populations[1].M[0] + solver.populations[2].M[0] - 4.0,
            0.0,
        )
    )

    assert np.allclose(other_cong, expected_cong)
    assert np.allclose(interaction_cost, expected_interaction)

"""Regression tests for the generalized multi-population solver.

This module preserves and modernizes the lightweight solver smoke test that
previously lived in ``test/test_n_population_solver.py``. The test uses a
small in-memory mesh stub so it can exercise the generalized ``MFGSolver``
without depending on external map files, YAML configs, or expensive scenario
setups.

Why keep this test
------------------
The existing test covers a useful slice of functionality that is not fully
overlapped by the larger script-style checks in ``tests/test_consistency.py``
and ``tests/test_shadow_leader.py``:

- construction of a solver with three explicit ``Population`` objects
- one Picard iteration of the generalized N-population solve path
- shape and finiteness checks on the value and density fields
- verification of the population-coupling helper against analytically built
  expectations from the configured interaction and congestion matrices

Because it is deterministic, fast, and self-contained, it is worth preserving
as a true pytest regression test.
"""

from __future__ import annotations

import numpy as np

from mfgames.problem import MFGSolver, Population


class DummyMesh:
    """Minimal structured-grid mesh stub for solver regression tests.

    The production solver only relies on a small subset of the geometry API for
    this test path: domain dimensions, Cartesian grid coordinates, obstacle
    mask access, initial-density construction, and goal enumeration. This stub
    implements just enough of that interface to let the test focus on solver
    behavior rather than file-backed geometry setup.
    """

    def __init__(self, Nx: int = 6, Ny: int = 5, dx: float = 1.0, dy: float = 1.0) -> None:
        """Build a small obstacle-free Cartesian mesh.

        Parameters
        ----------
        Nx, Ny
            Number of grid cells along the x- and y-directions.
        dx, dy
            Uniform cell widths in each direction.
        """

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
        self._goals: list[dict] = []

    def get_pde_obstacle_mask(self) -> np.ndarray:
        """Return the obstacle mask expected by the solver."""

        return self.omask

    def build_initial_density(self, *args, **kwargs) -> np.ndarray:
        """Return a copy of the cached initial density field.

        The signature is intentionally permissive because the production mesh
        interface accepts additional optional parameters.
        """

        return self._m0.copy()

    def get_goals(self) -> list[dict]:
        """Return configured goal dictionaries for compatibility."""

        return list(self._goals)


def make_population(name: str, center: tuple[float, float], amplitude: float) -> Population:
    """Create a simple Gaussian-initialized population for testing.

    Parameters
    ----------
    name
        Human-readable population label.
    center
        Goal and Gaussian center in mesh coordinates.
    amplitude
        Peak scale factor for the initial density field.
    """

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
        upwind_mode=0,
    )


def test_three_population_solver_runs_and_preserves_shapes() -> None:
    """Exercise one generalized three-population solve and verify outputs.

    The test intentionally uses a tiny grid and a single Picard iteration so it
    remains cheap while still touching the generalized coupling logic.
    """

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
    expected_cong = (
        congestion[0, 1] * solver.populations[1].M[0]
        + congestion[0, 2] * solver.populations[2].M[0]
    )
    expected_interaction = (
        interaction[0, 1] * solver.populations[1].M[0]
        + interaction[0, 2] * solver.populations[2].M[0]
        + np.maximum(
            solver.populations[0].M[0]
            + solver.populations[1].M[0]
            + solver.populations[2].M[0]
            - 4.0,
            0.0,
        )
    )

    assert np.allclose(other_cong, expected_cong)
    assert np.allclose(interaction_cost, expected_interaction)

"""
N-population Mean Field Game example runner.

This script demonstrates the generalized `mfgames.problem.MFGSolver` interface
with three populations sharing one spatial geometry and using explicit
interaction and congestion matrices.

The example is intentionally compact: it builds a synthetic corridor-style
geometry, seeds three Gaussian density blobs, defines one stationary goal per
population, and performs a short Picard solve.
"""

import numpy as np

from mfgames.geometry import MFGTrafficGeometry
from mfgames.problem import MFGSolver, Population


def _gaussian_blob(mesh, center_x, center_y, sigma, amplitude):
    return amplitude * np.exp(
        -((mesh.X - center_x) ** 2 + (mesh.Y - center_y) ** 2) / (2.0 * sigma**2)
    )


def main():
    mesh = MFGTrafficGeometry(Lx=120.0, Ly=80.0, Nx=40, Ny=30)
    mesh.build_spatial_mesh()

    pop1 = Population(
        name="alpha",
        mesh=mesh,
        m0=_gaussian_blob(mesh, 20.0, 20.0, sigma=8.0, amplitude=2.0),
        goal_configs=[{"type": "stationary", "position": [95.0, 20.0]}],
    )
    pop2 = Population(
        name="beta",
        mesh=mesh,
        m0=_gaussian_blob(mesh, 25.0, 55.0, sigma=8.0, amplitude=1.5),
        goal_configs=[{"type": "stationary", "position": [95.0, 55.0]}],
    )
    pop3 = Population(
        name="gamma",
        mesh=mesh,
        m0=_gaussian_blob(mesh, 55.0, 40.0, sigma=10.0, amplitude=1.0),
        goal_configs=[{"type": "stationary", "position": [20.0, 40.0]}],
    )

    interaction = np.array(
        [
            [0.0, 0.4, -0.2],
            [-0.2, 0.0, 0.3],
            [0.1, -0.1, 0.0],
        ],
        dtype=np.float64,
    )
    congestion = np.array(
        [
            [0.0, 0.8, 0.4],
            [0.6, 0.0, 0.5],
            [0.3, 0.4, 0.0],
        ],
        dtype=np.float64,
    )

    solver = MFGSolver(
        populations=[pop1, pop2, pop3],
        T=1.0,
        Nt=20,
        thetaUM=0.2,
        interaction_matrix=interaction,
        congestion_matrix=congestion,
    )

    results = solver.run_picard_system(max_iters=3, tolerance=1e-4)
    num_pops = len(solver.populations)
    U_list = results[:num_pops]
    M_list = results[num_pops:]

    print("Completed N-population solve")
    for idx, pop in enumerate(solver.populations):
        print(
            f"{pop.name}: U shape={U_list[idx].shape}, M shape={M_list[idx].shape}, final mass={M_list[idx][-1].sum():.6f}"
        )


if __name__ == "__main__":
    main()

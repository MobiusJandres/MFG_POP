#!/usr/bin/env python3
"""
Consistency Verification Script: MAP2PDE.py vs. mfgames MFGSolver

This test script validates that the refactored mfgames library acts as an exact
mathematical generalization of MAP2PDE.py. When configured with:
    - running_cost_weight = 0.0 (no spatial running cost accumulated during t < T)
    - terminal_cost_weight = -1.0 (negative squared-distance potential well)
    - terminal_cost_power = 2 (squared Euclidean distance)
    - obstacle_penalty = -500.0 (matching legacy obstacle potential)
    - goals_are_exits = False (no absorbing Dirichlet door masks)

the state trajectories U(t, x, y) and M(t, x, y) computed by MFGSolver must match
those produced by MAP2PDE.py to within floating-point tolerance across every timestep.
"""

import sys
import argparse
import importlib.util
from pathlib import Path
import numpy as np

# Import refactored mfgames library components
from mfgames.geometry import MAP2PDE as ModernMAP2PDE
from mfgames.problem import MFGSolver as ModernMFGSolver


def load_legacy_map2pde(legacy_path: Path):
    """
    Dynamically load MAP2PDE.py as an isolated module to avoid namespace collisions.

    Args:
        legacy_path (Path): Path to the legacy MAP2PDE.py script.

    Returns:
        module: Loaded legacy module object.
    """
    if not legacy_path.exists():
        raise FileNotFoundError(f"Legacy script not found at: '{legacy_path}'")

    spec = importlib.util.spec_from_file_location("legacy_map2pde", str(legacy_path))
    legacy_module = importlib.util.module_from_spec(spec)
    sys.modules["legacy_map2pde"] = legacy_module
    spec.loader.exec_module(legacy_module)
    return legacy_module


def run_verification(
    map_file: str,
    scen_file: str,
    legacy_script_path: str = "2D/1POP/MAP2PDE.py",
    room_width: float = 768.0,
    room_height: float = 768.0,
    Nx: int = 100,
    Ny: int = 100,
    num_agents: int = 50,
    T: float = 3.0,
    Nt: int = 50,
    max_iters: int = 5,
    relaxation_theta: float = 0.1,
    tolerance: float = 1e-5,
):
    """
    Execute legacy and modern solvers side-by-side and evaluate residual norms.
    """
    print("=" * 80)
    print("MFGAMES vs. MAP2PDE NUMERICAL CONSISTENCY VERIFICATION")
    print("=" * 80)
    print(f"Map File:       {map_file}")
    print(f"Scenario File:  {scen_file}")
    print(f"Domain Size:    {room_width}m x {room_height}m | Grid: {Nx} x {Ny}")
    print(f"Time Horizon:   T = {T}s | Nt = {Nt} | dt = {T / Nt:.4f}s")
    print(f"Picard Loops:   max_iters = {max_iters} | theta = {relaxation_theta}")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # 1. Run Legacy MAP2PDE.py
    # -------------------------------------------------------------------------
    print("\n[1/3] Executing Legacy MAP2PDE.py Solver...")
    legacy_mod = load_legacy_map2pde(Path(legacy_script_path))

    legacy_mesh = legacy_mod.MAP2PDE(
        map_filepath=map_file,
        scen_filepath=scen_file,
        Lx=room_width,
        Ly=room_height,
        Nx=Nx,
        Ny=Ny,
    )
    legacy_mesh.parse_files(num_agents=num_agents)
    legacy_mesh.build_spatial_mesh()

    legacy_solver = legacy_mod.MFGSolver(
        pde_mesh_data=legacy_mesh,
        T=T,
        Nt=Nt,
        thetaUM=relaxation_theta,
    )
    U_legacy, M_legacy = legacy_solver.run_picard_system(max_iters=max_iters)

    # -------------------------------------------------------------------------
    # 2. Run Modern mfgames MFGSolver with Generalization Parameters
    # -------------------------------------------------------------------------
    print("\n[2/3] Executing Refactored mfgames MFGSolver...")
    modern_mesh = ModernMAP2PDE(
        map_filepath=map_file,
        scen_filepath=scen_file,
        Lx=room_width,
        Ly=room_height,
        Nx=Nx,
        Ny=Ny,
    )
    modern_mesh.parse_files(num_agents=num_agents)
    modern_mesh.build_spatial_mesh()

    # Configure goals as stationary targets matching the benchmark scenario
    goal_positions = modern_mesh.get_goals()
    goal_configs = [
        {
            "type": "stationary",
            "position": [gx, gy],
            "capacity": float("inf"),
            "is_exit": False,
        }
        for gx, gy in goal_positions
    ]

    modern_solver = ModernMFGSolver(
        pde_mesh_data=modern_mesh,
        T=T,
        Nt=Nt,
        thetaUM=relaxation_theta,
        goal_configs=goal_configs,
        goals_are_exits=False,             # Disable absorbing Dirichlet exit boundaries
        obstacle_penalty=-500.0,           # Exact obstacle penalty from MAP2PDE.py
        running_cost_weight=0.0,           # Disable running cost accumulation (f(x,t) = 0)
        terminal_cost_weight=-1.0,         # Exact negative quadratic terminal cost
        running_cost_power=2,
        terminal_cost_power=2,
        saturated_goal_penalty=0.0,
    )
    U_modern, M_modern = modern_solver.run_picard_system(max_iters=max_iters)

    # -------------------------------------------------------------------------
    # 3. Compute Trajectory Discrepancies and Residual Norms
    # -------------------------------------------------------------------------
    print("\n[3/3] Evaluating Numerical Consistency...")

    # Spatial-temporal cell volume factor for L2 integration
    dx, dy, dt = room_width / Nx, room_height / Ny, T / Nt
    dV = np.sqrt(dx * dy * dt)

    diff_U = U_modern - U_legacy
    diff_M = M_modern - M_legacy

    max_abs_U = np.max(np.abs(diff_U))
    max_abs_M = np.max(np.abs(diff_M))

    l2_err_U = np.linalg.norm(diff_U) * dV
    l2_err_M = np.linalg.norm(diff_M) * dV

    rel_l2_U = np.linalg.norm(diff_U) / (np.linalg.norm(U_legacy) + 1e-12)
    rel_l2_M = np.linalg.norm(diff_M) / (np.linalg.norm(M_legacy) + 1e-12)

    # Initial condition check
    m0_diff = np.max(np.abs(modern_solver.M[0] - legacy_solver.M[0]))
    gT_diff = np.max(
        np.abs(
            modern_solver.compute_terminal_cost(modern_solver.populations[0], goal_positions)
            - legacy_solver.g_x
        )
    )

    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS TABLE")
    print("=" * 80)
    print(f"{'Metric':<35} | {'Value':<18} | {'Status'}")
    print("-" * 80)
    print(f"{'Initial Density m0 Diff':<35} | {m0_diff:<18.6e} | {'PASS' if m0_diff < 1e-10 else 'FAIL'}")
    print(f"{'Terminal Cost g(x) Diff':<35} | {gT_diff:<18.6e} | {'PASS' if gT_diff < 1e-10 else 'FAIL'}")
    print(f"{'Max Absolute Diff in U':<35} | {max_abs_U:<18.6e} | {'PASS' if max_abs_U < tolerance else 'FAIL'}")
    print(f"{'Relative L2 Error in U':<35} | {rel_l2_U:<18.6e} | {'PASS' if rel_l2_U < tolerance else 'FAIL'}")
    print(f"{'Max Absolute Diff in M':<35} | {max_abs_M:<18.6e} | {'PASS' if max_abs_M < tolerance else 'FAIL'}")
    print(f"{'Relative L2 Error in M':<35} | {rel_l2_M:<18.6e} | {'PASS' if rel_l2_M < tolerance else 'FAIL'}")
    print("=" * 80)

    is_consistent = (max_abs_U < tolerance) and (max_abs_M < tolerance)
    if is_consistent:
        print("[SUCCESS] mfgames results align with MAP2PDE.py within tolerance.")
    else:
        print("[FAILURE] Trajectory discrepancy exceeds numerical tolerance threshold.")

    return is_consistent


def main():
    parser = argparse.ArgumentParser(description="Verify consistency between MAP2PDE.py and mfgames")
    parser.add_argument("--map_file", default="Maps/AcrosstheCape.map", type=str, help="MovingAI map file path")
    parser.add_argument("--scen_file", default="Scenarios/AcrosstheCape_1g.map.scen", type=str, help="MovingAI scenario file path")
    parser.add_argument("--legacy_script", default="2D/1POP/MAP2PDE.py", type=str, help="Path to original MAP2PDE.py script")
    parser.add_argument("--Nx", default=200, type=int, help="Grid size Nx")
    parser.add_argument("--Ny", default=200, type=int, help="Grid size Ny")
    parser.add_argument("--Nt", default=100, type=int, help="Timestep count Nt")
    parser.add_argument("--max_iters", default=10, type=int, help="Picard loop iterations")
    parser.add_argument("--tol", default=1e-5, type=float, help="Residual check tolerance")
    args = parser.parse_args()

    success = run_verification(
        map_file=args.map_file,
        scen_file=args.scen_file,
        legacy_script_path=args.legacy_script,
        Nx=args.Nx,
        Ny=args.Ny,
        Nt=args.Nt,
        max_iters=args.max_iters,
        tolerance=args.tol,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

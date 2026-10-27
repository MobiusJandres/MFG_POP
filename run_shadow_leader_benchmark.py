#!/usr/bin/env python3
"""
Benchmark: N-Population Coupled Nash vs. N-Population Shadow-Leader Solver.

Demonstrates complexity reduction in an N = 4 population simulation:
- Pop 0 (Leader): Primary escort/adversary moving from North to South.
- Pop 1, 2, 3 (Followers): Three separate crowds evading the Leader and racing to exits.
"""

import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from mfgames.geometry import MAP2PDE
from mfgames.problem import MFGSolver, MFGShadowLeaderSolver, Population


def create_n_pop_setup(Nx=80, Ny=80, room_size=768.0):
    map_file = "Maps/AcrosstheCape.map"[cite: 10]
    scen_file = "Scenarios/AcrosstheCape.map.scen"[cite: 10]

    # Shared mesh geometry
    base_mesh = MAP2PDE(map_file, scen_file, Lx=room_size, Ly=room_size, Nx=Nx, Ny=Ny)[cite: 10]
    base_mesh.parse_files(start_idx=0, num_agents=1)[cite: 10]
    base_mesh.build_spatial_mesh()[cite: 10]

    # Define 4 distinct populations
    # Pop 0: Leader Swarm (moves South toward (400, 100))
    pop0 = Population(
        name="leader_swarm",
        mesh=base_mesh,
        m0=base_mesh.set_custom_initial_blobs([(400.0, 650.0, 35.0, 6.0)]),
        goal_configs=[{"type": "stationary", "position": [400.0, 100.0]}],
        running_cost_weight=0.01,
        stencil_mode=0,
    )

    # Pop 1, 2, 3: Follower Swarms positioned across the domain
    pop1 = Population(
        name="follower_west",
        mesh=base_mesh,
        m0=base_mesh.set_custom_initial_blobs([(150.0, 384.0, 30.0, 4.0)]),
        goal_configs=[{"type": "stationary", "position": [150.0, 700.0], "capacity": 20.0, "is_exit": True}],
        running_cost_weight=0.01,
        stencil_mode=0,
    )
    pop2 = Population(
        name="follower_center",
        mesh=base_mesh,
        m0=base_mesh.set_custom_initial_blobs([(400.0, 384.0, 30.0, 4.0)]),
        goal_configs=[{"type": "stationary", "position": [400.0, 700.0], "capacity": 20.0, "is_exit": True}],
        running_cost_weight=0.01,
        stencil_mode=0,
    )
    pop3 = Population(
        name="follower_east",
        mesh=base_mesh,
        m0=base_mesh.set_custom_initial_blobs([(650.0, 384.0, 30.0, 4.0)]),
        goal_configs=[{"type": "stationary", "position": [650.0, 700.0], "capacity": 20.0, "is_exit": True}],
        running_cost_weight=0.01,
        stencil_mode=0,
    )

    pops = [pop0, pop1, pop2, pop3]
    N = len(pops)

    # W_inter: Followers are strongly repelled by Pop 0 (Leader)
    W_inter = np.zeros((N, N), dtype=np.float64)
    for f in [1, 2, 3]:
        W_inter[f, 0] = 3000.0  # Follower f flees Leader 0
        W_inter[0, f] = 0.0     # Leader ignores follower f

    # W_cong: High counter-flow friction between leader and followers
    W_cong = np.ones((N, N), dtype=np.float64)
    for f in [1, 2, 3]:
        W_cong[f, 0] = 4.0
        W_cong[0, f] = 4.0

    return pops, W_inter, W_cong


def main():
    Nx, Ny = 80, 80
    T, Nt = 80.0, 80
    max_iters = 6

    print("=" * 80)
    print("N = 4 POPULATION SOLVER BENCHMARK: FULL NASH vs. SHADOW LEADER")
    print("=" * 80)

    # 1. Benchmark Method A: Full Coupled Nash Solver
    pops_nash, W_inter_n, W_cong_n = create_n_pop_setup(Nx=Nx, Ny=Ny)
    solver_nash = MFGSolver(
        populations=pops_nash,
        T=T,
        Nt=Nt,
        thetaUM=0.1,
        interaction_matrix=W_inter_n,
        congestion_matrix=W_cong_n,
        shared_capacity_goals=True,
    )

    print("\n[Method A] Running Full Coupled N=4 MFGSolver...")
    t0_nash = time.time()
    results_nash = solver_nash.run_picard_system(max_iters=max_iters)
    time_nash = time.time() - t0_nash
    print(f">> Full Coupled N=4 Solver Time: {time_nash:.2f}s")

    # 2. Benchmark Method B: N-Population Shadow Leader Solver
    pops_shadow, W_inter_s, W_cong_s = create_n_pop_setup(Nx=Nx, Ny=Ny)
    solver_shadow = MFGShadowLeaderSolver(
        populations=pops_shadow,
        leader_indices=[0],      # Pop 0 is the shadow leader
        T=T,
        Nt=Nt,
        thetaUM=0.1,
        interaction_matrix=W_inter_s,
        congestion_matrix=W_cong_s,
        shared_capacity_goals=True,
    )

    print("\n[Method B] Running N=4 MFGShadowLeaderSolver (Leader Fixed, 3 Followers Best-Response)...")
    t0_shadow = time.time()
    results_shadow = solver_shadow.run_picard_system(max_iters=max_iters)
    time_shadow = time.time() - t0_shadow
    print(f">> N=4 Shadow-Leader Solver Time: {time_shadow:.2f}s")

    # 3. Performance Summary
    speedup = time_nash / time_shadow if time_shadow > 0 else 1.0
    print("\n" + "=" * 80)
    print("SCALABILITY SUMMARY")
    print("=" * 80)
    print(f"{'Method':<32} | {'Time (s)':<12} | {'Status'}")
    print("-" * 80)
    print(f"{'Full Coupled Nash (N=4)':<32} | {time_nash:<12.2f} | Baseline")
    print(f"{'Shadow-Leader Best-Response (N=4)':<32} | {time_shadow:<12.2f} | Speedup: {speedup:.2f}x")
    print("=" * 80)


if __name__ == "__main__":
    main()
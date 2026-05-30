import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from algorithms.physarum import PhysarumSolver
from algorithms.dijkstra import DijkstraSolver
from topology.mesh_topo import get_networkx_graph as mesh_graph
from evaluation.metrics import enrich_graph_with_sim_attributes
from evaluation.runner import _apply_cost_strategy

def run_stability_benchmark(steps=100, noise_std=0.2, seed=42):
    rng = np.random.default_rng(seed)
    
    # Generate parallel path (diamond) topology
    # Path A: 0 -> 1 -> 2 -> 3 (base weight 9.0)
    # Path B: 0 -> 4 -> 5 -> 3 (base weight 9.15)
    G_base = nx.Graph()
    G_base.add_edge(0, 1, latency_ms=3.0)
    G_base.add_edge(1, 2, latency_ms=3.0)
    G_base.add_edge(2, 3, latency_ms=3.0)

    G_base.add_edge(0, 4, latency_ms=3.05)
    G_base.add_edge(4, 5, latency_ms=3.05)
    G_base.add_edge(5, 3, latency_ms=3.05)
    
    src, dst = 0, 3

    # Track paths and flaps
    dj_paths = []
    phy_paths = []
    
    dj_costs = []
    phy_costs = []
    
    # Physarum memory state
    D_memory = None
    
    print(f"Running stability benchmark for {steps} steps (noise std={noise_std})...")
    
    for t in range(steps):
        # Perturb edge weights with transient noise
        G_noisy = G_base.copy()
        for u, v in G_noisy.edges():
            base_w = G_base[u][v].get("latency_ms", 1.0)
            noise = rng.normal(0, noise_std)
            # Ensure weight stays positive
            G_noisy[u][v]["weight"] = max(0.1, base_w + noise)
            
        # 1. Dijkstra Solver (no memory)
        dj_res = DijkstraSolver(G_noisy, src, dst).solve()
        dj_paths.append(dj_res.path)
        dj_costs.append(dj_res.cost)
        
        # 2. Physarum Solver (warm started from previous step, run for 10 iterations to preserve memory)
        solver = PhysarumSolver(G_noisy, src, dst, decay=0.1, dt=0.05, max_iter=10, patience=999)
        phy_res = solver.solve(init_D=D_memory)
        D_memory = solver.get_all_conductivities()
        phy_paths.append(phy_res.path)
        
        # Compute true cost of physarum path on current noisy graph
        p_cost = 0.0
        if phy_res.path:
            for u_p, v_p in zip(phy_res.path[:-1], phy_res.path[1:]):
                p_cost += G_noisy[u_p][v_p]["weight"]
        else:
            p_cost = float("inf")
        phy_costs.append(p_cost)

    # Calculate flaps (changes in path sequence)
    dj_flaps = sum(1 for i in range(1, len(dj_paths)) if dj_paths[i] != dj_paths[i-1])
    phy_flaps = sum(1 for i in range(1, len(phy_paths)) if phy_paths[i] != phy_paths[i-1])
    
    print("\n--- Benchmark Results ---")
    print(f"Dijkstra Route Flaps: {dj_flaps}")
    print(f"Physarum Route Flaps: {phy_flaps}")
    print(f"Dijkstra Avg Path Cost: {np.mean(dj_costs):.3f}")
    print(f"Physarum Avg Path Cost: {np.mean(phy_costs):.3f}")
    
    # Plotting path changes over time
    plt.figure(figsize=(12, 6))
    
    # We can plot path IDs (hash of path tuple) to visualize flapping
    dj_ids = [0 if p == [0, 1, 2, 3] else 1 for p in dj_paths]
    phy_ids = [0 if p == [0, 1, 2, 3] else 1 for p in phy_paths]
    
    plt.step(range(steps), dj_ids, where="post", label=f"Dijkstra (Flaps: {dj_flaps})", alpha=0.6, color="orange")
    plt.step(range(steps), phy_ids, where="post", label=f"Physarum (Flaps: {phy_flaps})", alpha=0.9, color="blue", linewidth=2.5)
    
    plt.title(f"Route Flapping Comparison under Transient Weight Noise")
    plt.xlabel("Time Step")
    plt.ylabel("Chosen Route (0 = Path A, 1 = Path B)")
    plt.yticks([0, 1], ["Path A ([0, 1, 2, 3])", "Path B ([0, 4, 5, 3])"])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = "results/figures/route_stability.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved stability plot to {output_path}")

if __name__ == "__main__":
    import os
    os.makedirs("results/figures", exist_ok=True)
    run_stability_benchmark()

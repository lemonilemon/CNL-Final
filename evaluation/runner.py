"""
Experiment runner for routing algorithm benchmarks.

Usage (macOS — standalone):
  $ python -m evaluation.runner
  $ python -m evaluation.runner --algorithms physarum_classic,dijkstra --node-counts 6,10 --trials 3
"""

from __future__ import annotations
import argparse, csv, os, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algorithms.physarum import PhysarumSolver
from algorithms.dijkstra import DijkstraSolver
from algorithms.aco import ACOSolver
from controller.base_controller import LinkCostStrategy
from controller.physarum_controller import compute_route
from topology.mesh_topo import get_networkx_graph as mesh_graph
from topology.dynamic_topo import DynamicGraph
from evaluation.metrics import (
    RoutingMetrics, compute_transmission_delay,
    compute_bottleneck_throughput, enrich_graph_with_sim_attributes,
)

DEFAULT_ALGORITHMS = [
    "physarum_classic_dijkstra",
    "physarum_classic_greedy",
    "physarum_improved_dijkstra",
    "physarum_improved_greedy",
    "dijkstra",
    "aco",
]
DEFAULT_NODE_COUNTS = [6, 10, 20, 50]
DEFAULT_COST_STRATEGIES = LinkCostStrategy.ALL
DEFAULT_TRIALS = 5


def _apply_cost_strategy(G, strategy: str) -> None:
    for u, v in G.edges():
        d = G[u][v]
        G[u][v]["weight"] = LinkCostStrategy.compute(
            strategy, latency_ms=d.get("latency_ms", 1.0),
            bandwidth_bps=d.get("bandwidth_mbps", 100.0) * 1e6)


def _dijkstra_optimal_cost(G, src, dst) -> float:
    try:
        return DijkstraSolver(G, src, dst).solve().cost
    except (ValueError, Exception):
        return float("inf")


def run_all(algorithms, node_counts, cost_strategies, trials, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    # We will count total runs for progress printing
    total_single = len(cost_strategies) * len(node_counts) * trials * len(algorithms)
    total_adapt = len(cost_strategies) * len(node_counts) * len(algorithms)
    total = total_single + total_adapt
    cnt = 0

    # 1. Run single trials on the same random graph per condition and trial
    for strat in cost_strategies:
        for n in node_counts:
            for t in range(trials):
                seed = n * 1000 + t
                rng = np.random.default_rng(seed)

                G_base = enrich_graph_with_sim_attributes(mesh_graph(n=n, connectivity=0.3, seed=seed), rng=rng)
                _apply_cost_strategy(G_base, strat)
                nodes = sorted(G_base.nodes())
                src, dst = nodes[0], nodes[-1]
                
                optimal_cost = _dijkstra_optimal_cost(G_base, src, dst)
                
                for algo in algorithms:
                    cnt += 1
                    print(f"[{cnt}/{total}] {algo} n={n} {strat} trial={t+1}", flush=True)
                    try:
                        G = G_base.copy()
                        rr = compute_route(algo, G, src, dst)
                        path, cost, elapsed = rr.path, rr.cost, rr.elapsed_seconds
                        
                        delay = compute_transmission_delay(G, path) if path else float("inf")
                        tp = compute_bottleneck_throughput(G, path) if path else 0.0
                        
                        is_optimal = abs(cost - optimal_cost) < 1e-6 if cost < float("inf") else False
                        
                        results.append(RoutingMetrics(
                            algorithm=algo, n_nodes=n, cost_strategy=strat,
                            execution_time_s=elapsed, path_cost=cost,
                            path_length=len(path) if path else 0,
                            transmission_delay_ms=delay, throughput_mbps=tp,
                            converged=rr.converged, iterations=rr.iterations,
                            is_optimal=is_optimal,
                        ))
                    except Exception as e:
                        print(f"  ERROR: {e}")

    # 2. Run adaptiveness trials on the same random graph per condition
    print("\n--- Adaptiveness ---")
    for strat in cost_strategies:
        for n in node_counts:
            seed = n * 1000
            rng = np.random.default_rng(seed)
            dg = DynamicGraph(n=n, connectivity=0.4, seed=seed)
            G_base = enrich_graph_with_sim_attributes(dg.get_graph(), rng=rng)
            _apply_cost_strategy(G_base, strat)
            nodes = sorted(G_base.nodes())
            src, dst = nodes[0], nodes[-1]
            
            # Find the failed link based on Dijkstra's path to ensure all algos fail the same link
            try:
                dj_res = DijkstraSolver(G_base, src, dst).solve()
                path_dj = dj_res.path
            except Exception:
                path_dj = []
                
            if not path_dj or len(path_dj) < 3:
                # Skip this topology if there's no route or it's too short
                for algo in algorithms:
                    cnt += 1
                continue
                
            mid = len(path_dj) // 2
            fail_u, fail_v = path_dj[mid], path_dj[mid+1]
            
            G2_base = G_base.copy()
            if G2_base.has_edge(fail_u, fail_v):
                G2_base.remove_edge(fail_u, fail_v)
                
            optimal_cost_deg = _dijkstra_optimal_cost(G2_base, src, dst)
            
            physarum_solvers = {}
            for algo in algorithms:
                if "physarum" in algo:
                    variant = "classic" if "classic" in algo else "improved"
                    policy = "greedy" if "greedy" in algo else "dijkstra"
                    solver = PhysarumSolver(G_base.copy(), src, dst, variant=variant, extraction_policy=policy)
                    solver.solve()
                    physarum_solvers[algo] = solver

            for algo in algorithms:
                cnt += 1
                print(f"[{cnt}/{total}] Adaptiveness: {algo} n={n} {strat}", flush=True)
                try:
                    G2 = G2_base.copy()
                    t0 = time.perf_counter()
                    if "physarum" in algo and algo in physarum_solvers:
                        solver = physarum_solvers[algo]
                        D_old = solver.D.copy()
                        u_idx = solver.node_to_idx[fail_u]
                        v_idx = solver.node_to_idx[fail_v]
                        D_old[u_idx, v_idx] = 0.0
                        D_old[v_idx, u_idx] = 0.0
                        
                        solver2 = PhysarumSolver(G2, src, dst, variant=solver.variant, extraction_policy=solver.extraction_policy)
                        rr2 = solver2.solve(init_D=D_old)
                    else:
                        rr2 = compute_route(algo, G2, src, dst)
                    rt = time.perf_counter() - t0
                    
                    p2, c2 = rr2.path, rr2.cost
                    delay = compute_transmission_delay(G2, p2) if p2 else float("inf")
                    tp = compute_bottleneck_throughput(G2, p2) if p2 else 0.0
                    
                    is_optimal = abs(c2 - optimal_cost_deg) < 1e-6 if c2 < float("inf") else False
                    
                    results.append(RoutingMetrics(
                        algorithm=algo, n_nodes=n, cost_strategy=strat,
                        execution_time_s=rt, path_cost=c2,
                        path_length=len(p2) if p2 else 0,
                        transmission_delay_ms=delay, throughput_mbps=tp,
                        adaptiveness_s=rt, converged=rr2.converged, iterations=rr2.iterations,
                        is_optimal=is_optimal,
                    ))
                except Exception as e:
                    print(f"  ERROR: {e}")
                    
    csv_path = os.path.join(output_dir, "results.csv")
    fields = ["algorithm","n_nodes","cost_strategy","execution_time_s","path_cost",
              "path_length","transmission_delay_ms","throughput_mbps","adaptiveness_s",
              "converged","iterations","is_optimal"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in results:
            w.writerow({k: getattr(r, k, "") for k in fields})
    print(f"\nSaved {len(results)} results to {csv_path}")
    return results


def main():
    p = argparse.ArgumentParser(description="SDN routing benchmarks")
    p.add_argument("--algorithms", default=",".join(DEFAULT_ALGORITHMS))
    p.add_argument("--node-counts", default=",".join(str(n) for n in DEFAULT_NODE_COUNTS))
    p.add_argument("--cost-strategies", default=",".join(DEFAULT_COST_STRATEGIES))
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    p.add_argument("--output", default="results")
    a = p.parse_args()
    run_all(a.algorithms.split(","), [int(x) for x in a.node_counts.split(",")],
            a.cost_strategies.split(","), a.trials, a.output)

if __name__ == "__main__":
    main()

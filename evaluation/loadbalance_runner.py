"""
Multipath load-balancing experiment runner (Phase 1).

Sweeps {topology} x {offered-load level} x {method} x {trial} through the
flow-level simulator and writes results/loadbalance.csv.

Usage:
  $ python -m evaluation.loadbalance_runner
  $ python -m evaluation.loadbalance_runner --trials 5 --output results
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from topology.datacenter_topo import (
    get_fat_tree_graph,
    get_leaf_spine_graph,
    cross_block_pairs,
)
from evaluation.loadbalance_sim import simulate, METHODS

# Per-flow demand levels (Mbps) — swept from under-subscribed into congestion.
DEFAULT_DEMANDS = [25, 50, 100, 150, 200, 300, 400, 600]
N_FLOWS = 8


def _build_topology(kind: str, seed: int):
    if kind == "leaf_spine":
        return get_leaf_spine_graph(n_leaf=4, n_spine=4, hosts_per_leaf=4, seed=seed)
    elif kind == "fat_tree":
        # Symmetric: structure is seed-independent.
        return get_fat_tree_graph(k=4)
    raise ValueError(kind)


def run_all(topologies, demands, trials, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    methods = list(METHODS.keys())
    total = len(topologies) * trials * len(demands) * len(methods)
    cnt = 0

    for topo in topologies:
        for t in range(trials):
            G, meta = _build_topology(topo, seed=100 + t)
            flows = cross_block_pairs(meta, N_FLOWS, np.random.default_rng(t))
            for demand in demands:
                for method in methods:
                    cnt += 1
                    print(f"[{cnt}/{total}] {topo} {method} demand={demand} trial={t+1}",
                          flush=True)
                    r = simulate(G, method, flows, float(demand))
                    rows.append({
                        "topology": topo,
                        "method": method,
                        "demand_mbps": demand,
                        "n_flows": len(flows),
                        "offered_load": r.offered_load,
                        "aggregate_throughput": round(r.aggregate_throughput, 3),
                        "packet_loss": round(r.packet_loss, 5),
                        "max_link_util": round(r.max_link_util, 4),
                        "util_imbalance": round(r.util_imbalance, 5),
                        "trial": t,
                    })

    csv_path = os.path.join(output_dir, "loadbalance.csv")
    fields = ["topology", "method", "demand_mbps", "n_flows", "offered_load",
              "aggregate_throughput", "packet_loss", "max_link_util",
              "util_imbalance", "trial"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} results to {csv_path}")
    return rows


def main():
    p = argparse.ArgumentParser(description="Multipath load-balancing benchmark")
    p.add_argument("--topologies", default="leaf_spine,fat_tree")
    p.add_argument("--demands", default=",".join(str(d) for d in DEFAULT_DEMANDS))
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--output", default="results")
    a = p.parse_args()
    run_all(a.topologies.split(","),
            [int(x) for x in a.demands.split(",")],
            a.trials, a.output)


if __name__ == "__main__":
    main()

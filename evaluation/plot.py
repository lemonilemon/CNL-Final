"""
Visualization of benchmark results.

Reads results.csv and generates comparison charts:
  - Execution time vs. node count (grouped by algorithm)
  - Transmission delay comparison
  - Throughput comparison
  - Adaptiveness (re-route time) comparison
  - Convergence iterations for iterative algorithms
  - Optimality rate comparison
  - Adaptation iterations after link failure comparison

Usage:
  $ python -m evaluation.plot --input results/results.csv --output results/figures/
"""

from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    HAS_PLOT_DEPS = True
except ImportError:
    HAS_PLOT_DEPS = False


# Color palette for algorithms
COLORS = {
    "physarum_classic_dijkstra": "#2ecc71",
    "physarum_classic_greedy": "#27ae60",
    "physarum_improved_dijkstra": "#e74c3c",
    "physarum_improved_greedy": "#c0392b",
    "physarum_classic": "#2ecc71",
    "physarum_improved": "#e74c3c",
    "dijkstra": "#3498db",
    "aco": "#f39c12",
}

LABELS = {
    "physarum_classic_dijkstra": "Physarum Classic (Dijkstra)",
    "physarum_classic_greedy": "Physarum Classic (Greedy)",
    "physarum_improved_dijkstra": "Physarum Improved (Dijkstra)",
    "physarum_improved_greedy": "Physarum Improved (Greedy)",
    "physarum_classic": "Physarum Classic",
    "physarum_improved": "Physarum Improved",
    "dijkstra": "Dijkstra",
    "aco": "ACO",
}


def load_results(csv_path: str) -> "pd.DataFrame":
    if not HAS_PLOT_DEPS:
        raise ImportError("matplotlib and pandas required: pip install matplotlib pandas")
    return pd.read_csv(csv_path)


def plot_execution_time(df: "pd.DataFrame", output_dir: str, strategy: str = "hop_count"):
    """Bar chart: execution time vs. node count, grouped by algorithm."""
    sub = df[(df["cost_strategy"] == strategy) & (df["adaptiveness_s"].isna())]
    fig, ax = plt.subplots(figsize=(10, 6))
    algorithms = sub["algorithm"].unique()
    node_counts = sorted(sub["n_nodes"].unique())
    x = np.arange(len(node_counts))
    width = 0.8 / len(algorithms)

    for i, algo in enumerate(algorithms):
        means = [sub[(sub["algorithm"] == algo) & (sub["n_nodes"] == n)]["execution_time_s"].mean()
                 for n in node_counts]
        ax.bar(x + i * width, means, width, label=LABELS.get(algo, algo),
               color=COLORS.get(algo, f"C{i}"), alpha=0.85)

    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Execution Time (s)")
    ax.set_title(f"Routing Algorithm Execution Time ({strategy})")
    ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels(node_counts)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"execution_time_{strategy}.png"), dpi=150)
    plt.close()


def plot_delay(df: "pd.DataFrame", output_dir: str, strategy: str = "hop_count"):
    """Line chart: transmission delay vs. node count."""
    sub = df[(df["cost_strategy"] == strategy) & (df["adaptiveness_s"].isna())]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, algo in enumerate(sub["algorithm"].unique()):
        data = sub[sub["algorithm"] == algo]
        means = data.groupby("n_nodes")["transmission_delay_ms"].mean()
        ax.plot(means.index, means.values, "o-", label=LABELS.get(algo, algo),
                color=COLORS.get(algo, f"C{i}"), linewidth=2, markersize=6)

    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Transmission Delay (ms)")
    ax.set_title(f"End-to-End Transmission Delay ({strategy})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"delay_{strategy}.png"), dpi=150)
    plt.close()


def plot_throughput(df: "pd.DataFrame", output_dir: str, strategy: str = "hop_count"):
    """Box plot: throughput distribution per algorithm."""
    sub = df[(df["cost_strategy"] == strategy) & (df["adaptiveness_s"].isna())]
    fig, ax = plt.subplots(figsize=(10, 6))
    algorithms = sorted(sub["algorithm"].unique())
    data = [sub[sub["algorithm"] == a]["throughput_mbps"].dropna().values for a in algorithms]
    
    # Avoid DeprecationWarning for labels in matplotlib >= 3.9
    try:
        bp = ax.boxplot(data, tick_labels=[LABELS.get(a, a) for a in algorithms], patch_artist=True)
    except TypeError:
        bp = ax.boxplot(data, labels=[LABELS.get(a, a) for a in algorithms], patch_artist=True)
        
    for patch, algo in zip(bp["boxes"], algorithms):
        patch.set_facecolor(COLORS.get(algo, "gray"))
        patch.set_alpha(0.7)

    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title(f"End-to-End Throughput Distribution ({strategy})")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"throughput_{strategy}.png"), dpi=150)
    plt.close()


def plot_adaptiveness(df: "pd.DataFrame", output_dir: str, strategy: str = "hop_count"):
    """Bar chart: re-route time after link failure."""
    sub = df[(df["cost_strategy"] == strategy) & (df["adaptiveness_s"].notna())]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    algorithms = sub["algorithm"].unique()
    node_counts = sorted(sub["n_nodes"].unique())
    x = np.arange(len(node_counts))
    width = 0.8 / len(algorithms)

    for i, algo in enumerate(algorithms):
        means = [sub[(sub["algorithm"] == algo) & (sub["n_nodes"] == n)]["adaptiveness_s"].mean()
                 for n in node_counts]
        ax.bar(x + i * width, means, width, label=LABELS.get(algo, algo),
               color=COLORS.get(algo, f"C{i}"), alpha=0.85)

    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Re-route Time (s)")
    ax.set_title(f"Adaptiveness: Re-routing After Link Failure ({strategy})")
    ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels(node_counts)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"adaptiveness_{strategy}.png"), dpi=150)
    plt.close()


def plot_iterations(df: "pd.DataFrame", output_dir: str, strategy: str = "hop_count"):
    """Line chart: average iterations to converge vs. node count for iterative algorithms."""
    sub = df[(df["cost_strategy"] == strategy) & (df["adaptiveness_s"].isna())]
    iterative_algos = [a for a in sub["algorithm"].unique() if a not in ("dijkstra",)]
    if not iterative_algos:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, algo in enumerate(iterative_algos):
        data = sub[sub["algorithm"] == algo]
        data = data.dropna(subset=["iterations"])
        if data.empty:
            continue
        means = data.groupby("n_nodes")["iterations"].mean()
        ax.plot(means.index, means.values, "o-", label=LABELS.get(algo, algo),
                color=COLORS.get(algo, f"C{i}"), linewidth=2, markersize=6)

    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Average Iterations")
    ax.set_title(f"Average Convergence Iterations ({strategy})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"iterations_{strategy}.png"), dpi=150)
    plt.close()


def plot_optimality(df: "pd.DataFrame", output_dir: str, strategy: str = "hop_count"):
    """Bar chart: optimality rate (percentage of trials finding optimal path) vs. node count."""
    sub = df[(df["cost_strategy"] == strategy) & (df["adaptiveness_s"].isna())]
    fig, ax = plt.subplots(figsize=(10, 6))
    algorithms = sub["algorithm"].unique()
    node_counts = sorted(sub["n_nodes"].unique())
    x = np.arange(len(node_counts))
    width = 0.8 / len(algorithms)

    for i, algo in enumerate(algorithms):
        rates = []
        for n in node_counts:
            subset = sub[(sub["algorithm"] == algo) & (sub["n_nodes"] == n)]
            if subset.empty:
                rates.append(0.0)
            else:
                rate = subset["is_optimal"].astype(bool).mean() * 100.0
                rates.append(rate)
                
        ax.bar(x + i * width, rates, width, label=LABELS.get(algo, algo),
               color=COLORS.get(algo, f"C{i}"), alpha=0.85)

    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Optimality Rate (%)")
    ax.set_title(f"Optimality Rate by Algorithm ({strategy})")
    ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels(node_counts)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"optimality_{strategy}.png"), dpi=150)
    plt.close()


def plot_adaptiveness_iterations(df: "pd.DataFrame", output_dir: str, strategy: str = "hop_count"):
    """Bar chart: iterations to re-route/adapt after link failure for iterative algorithms."""
    sub = df[(df["cost_strategy"] == strategy) & (df["adaptiveness_s"].notna())]
    iterative_algos = [a for a in sub["algorithm"].unique() if a not in ("dijkstra",)]
    if not iterative_algos:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    algorithms = sub["algorithm"].unique()
    node_counts = sorted(sub["n_nodes"].unique())
    x = np.arange(len(node_counts))
    width = 0.8 / len(algorithms)

    for i, algo in enumerate(algorithms):
        means = []
        for n in node_counts:
            subset = sub[(sub["algorithm"] == algo) & (sub["n_nodes"] == n)]
            if subset.empty:
                means.append(0.0)
            else:
                # Average iterations on the degraded graph
                means.append(subset["iterations"].mean())
        ax.bar(x + i * width, means, width, label=LABELS.get(algo, algo),
               color=COLORS.get(algo, f"C{i}"), alpha=0.85)

    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Adaptation Iterations")
    ax.set_title(f"Adaptation Iterations After Link Failure ({strategy})")
    ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels(node_counts)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"adaptiveness_iterations_{strategy}.png"), dpi=150)
    plt.close()


def plot_warm_start_speedup(df: "pd.DataFrame", output_dir: str, strategy: str = "hop_count"):
    """Bar chart: time difference (speedup) between cold start and warm start in seconds."""
    sub = df[(df["cost_strategy"] == strategy) & (df["warm_start_speedup_s"].notna())]
    if sub.empty:
        return
        
    fig, ax = plt.subplots(figsize=(10, 6))
    algorithms = sub["algorithm"].unique()
    node_counts = sorted(sub["n_nodes"].unique())
    x = np.arange(len(node_counts))
    width = 0.8 / len(algorithms)

    for i, algo in enumerate(algorithms):
        means = []
        for n in node_counts:
            subset = sub[(sub["algorithm"] == algo) & (sub["n_nodes"] == n)]
            if subset.empty:
                means.append(0.0)
            else:
                means.append(subset["warm_start_speedup_s"].mean())
        ax.bar(x + i * width, means, width, label=LABELS.get(algo, algo),
               color=COLORS.get(algo, f"C{i}"), alpha=0.85)

    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Speedup (Seconds)")
    ax.set_title(f"Warm Start Time Savings ({strategy})")
    ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels(node_counts)
    
    # Add a horizontal line at 0 for reference
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"warm_start_speedup_{strategy}.png"), dpi=150)
    plt.close()

def plot_route_flaps(df: "pd.DataFrame", output_dir: str, strategy: str = "hop_count"):
    """Bar chart: average route flaps under transient link noise."""
    sub = df[(df["cost_strategy"] == strategy) & (df["route_flaps"].notna())]
    if sub.empty:
        return
        
    fig, ax = plt.subplots(figsize=(10, 6))
    algorithms = sub["algorithm"].unique()
    node_counts = sorted(sub["n_nodes"].unique())
    x = np.arange(len(node_counts))
    width = 0.8 / len(algorithms)

    for i, algo in enumerate(algorithms):
        means = []
        for n in node_counts:
            subset = sub[(sub["algorithm"] == algo) & (sub["n_nodes"] == n)]
            if subset.empty:
                means.append(0.0)
            else:
                means.append(subset["route_flaps"].mean())
        ax.bar(x + i * width, means, width, label=LABELS.get(algo, algo),
               color=COLORS.get(algo, f"C{i}"), alpha=0.85)

    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Average Route Flaps (over 50 noise steps)")
    ax.set_title(f"Route Stability / Flapping under Noise ({strategy})")
    ax.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax.set_xticklabels(node_counts)
    
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"route_flaps_{strategy}.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Multipath load-balancing plots (reads results/loadbalance.csv)
# ---------------------------------------------------------------------------

LB_COLORS = {
    "dijkstra_single": "#3498db",
    "ecmp": "#9b59b6",
    "physarum": "#2ecc71",
    "aco": "#f39c12",
}
LB_LABELS = {
    "dijkstra_single": "Dijkstra (single-path)",
    "ecmp": "ECMP (equal-split)",
    "physarum": "Physarum (WCMP)",
    "aco": "ACO (pheromone-weighted)",
}
LB_ORDER = ["dijkstra_single", "ecmp", "physarum", "aco"]


def _lb_line(df, output_dir, topology, ycol, ylabel, title, fname, pct=False, ideal=False):
    sub = df[df["topology"] == topology]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in [a for a in LB_ORDER if a in sub["method"].unique()]:
        data = sub[sub["method"] == algo]
        g = data.groupby("offered_load")[ycol].mean()
        y = g.values * (100.0 if pct else 1.0)
        ax.plot(g.index, y, "o-", label=LB_LABELS.get(algo, algo),
                color=LB_COLORS.get(algo), linewidth=2, markersize=6)
    if ideal:
        lim = sub["offered_load"].max()
        ax.plot([0, lim], [0, lim], "k--", alpha=0.5, linewidth=1,
                label="Ideal (no loss)")
    ax.set_xlabel("Offered Load (Mbps, total across flows)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} — {topology}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()


def generate_loadbalance_plots(csv_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = load_results(csv_path)
    for topo in df["topology"].unique():
        print(f"Generating load-balancing plots for topology: {topo}")
        _lb_line(df, output_dir, topo, "aggregate_throughput",
                 "Aggregate Throughput (Mbps)", "Aggregate Throughput vs Offered Load",
                 f"lb_throughput_{topo}.png", ideal=True)
        _lb_line(df, output_dir, topo, "packet_loss",
                 "Packet Loss (%)", "Packet Loss vs Offered Load",
                 f"lb_loss_{topo}.png", pct=True)
        _lb_line(df, output_dir, topo, "util_imbalance",
                 "Link-Utilization Imbalance (stddev)", "Load Imbalance vs Offered Load",
                 f"lb_imbalance_{topo}.png")
    print(f"Load-balancing plots saved to {output_dir}")


# ---------------------------------------------------------------------------
# Phase-2 testbed plots (Mininet+iperf): results/phase2_udp.csv, phase2_tcp.csv
# Overlays the measured points on the Phase-1 flow-level predictions.
# ---------------------------------------------------------------------------

def _phase2_overlay(udp, lb, output_dir, topology, ycol_p2, ycol_p1,
                    ylabel, title, fname, pct=False):
    sub = udp[udp["topology"] == topology]
    if sub.empty:
        return
    lb_sub = lb[lb["topology"] == topology] if lb is not None else None
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in [a for a in LB_ORDER if a in sub["method"].unique()]:
        color = LB_COLORS.get(algo)
        d = sub[sub["method"] == algo].groupby("offered_total_mbps")[ycol_p2].mean()
        y = d.values * (100.0 if pct else 1.0)
        ax.plot(d.index, y, "o-", color=color, linewidth=2, markersize=6,
                label=f"{LB_LABELS.get(algo, algo)} (measured)")
        if lb_sub is not None and not lb_sub.empty:
            p = lb_sub[lb_sub["method"] == algo].groupby("offered_load")[ycol_p1].mean()
            yp = p.values * (100.0 if pct else 1.0)
            ax.plot(p.index, yp, "--", color=color, alpha=0.5, linewidth=1.5,
                    label=f"{LB_LABELS.get(algo, algo)} (predicted)")
    ax.set_xlabel("Offered Load (Mbps, total across flows)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} — {topology} (Phase 2 vs Phase 1)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()


def _phase2_tcp_bars(tcp, output_dir, topology):
    sub = tcp[tcp["topology"] == topology]
    if sub.empty:
        return
    algos = [a for a in LB_ORDER if a in sub["method"].unique()]
    means = [sub[sub["method"] == a]["throughput_mbps"].mean() for a in algos]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar([LB_LABELS.get(a, a) for a in algos], means,
           color=[LB_COLORS.get(a) for a in algos])
    ax.set_ylabel("Aggregate TCP Throughput (Mbps)")
    ax.set_title(f"Saturating TCP Throughput — {topology}")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"phase2_tcp_{topology}.png"), dpi=150)
    plt.close()


def generate_phase2_plots(udp_csv: str, tcp_csv: str, lb_csv: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    udp = load_results(udp_csv)
    tcp = load_results(tcp_csv) if os.path.exists(tcp_csv) else None
    lb = load_results(lb_csv) if os.path.exists(lb_csv) else None
    for topo in udp["topology"].unique():
        print(f"Generating Phase-2 plots for topology: {topo}")
        _phase2_overlay(udp, lb, output_dir, topo,
                        "throughput_mbps", "aggregate_throughput",
                        "Aggregate Throughput (Mbps)", "UDP Aggregate Throughput",
                        f"phase2_throughput_{topo}.png")
        _phase2_overlay(udp, lb, output_dir, topo,
                        "packet_loss", "packet_loss",
                        "Packet Loss (%)", "UDP Packet Loss",
                        f"phase2_loss_{topo}.png", pct=True)
        if tcp is not None:
            _phase2_tcp_bars(tcp, output_dir, topo)
    print(f"Phase-2 plots saved to {output_dir}")


def generate_all_plots(csv_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = load_results(csv_path)
    strategies = df["cost_strategy"].unique()
    for s in strategies:
        print(f"Generating plots for strategy: {s}")
        plot_execution_time(df, output_dir, s)
        plot_delay(df, output_dir, s)
        plot_throughput(df, output_dir, s)
        plot_adaptiveness(df, output_dir, s)
        plot_iterations(df, output_dir, s)
        plot_optimality(df, output_dir, s)
        plot_adaptiveness_iterations(df, output_dir, s)
        plot_warm_start_speedup(df, output_dir, s)
        plot_route_flaps(df, output_dir, s)
    print(f"All plots saved to {output_dir}")


def main():
    p = argparse.ArgumentParser(description="Plot benchmark results")
    p.add_argument("--input", default="results/results.csv")
    p.add_argument("--output", default="results/figures")
    p.add_argument("--loadbalance", action="store_true",
                   help="plot load-balancing results (input defaults to results/loadbalance.csv)")
    p.add_argument("--phase2", action="store_true",
                   help="plot Phase-2 Mininet results (results/phase2_udp.csv + phase2_tcp.csv), "
                        "overlaid on Phase-1 predictions (results/loadbalance.csv)")
    a = p.parse_args()
    if a.phase2:
        generate_phase2_plots("results/phase2_udp.csv", "results/phase2_tcp.csv",
                              "results/loadbalance.csv", a.output)
    elif a.loadbalance:
        inp = a.input if a.input != "results/results.csv" else "results/loadbalance.csv"
        generate_loadbalance_plots(inp, a.output)
    else:
        generate_all_plots(a.input, a.output)

if __name__ == "__main__":
    main()

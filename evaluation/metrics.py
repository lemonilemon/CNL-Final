"""
Metric collection for routing algorithm evaluation.

Provides functions to measure the four metrics from the proposal:
  1. Execution time   — wall-clock time of the routing algorithm
  2. Transmission delay — simulated / measured RTT
  3. End-to-end throughput — simulated bandwidth along the computed path
  4. Adaptiveness     — time to re-route after a link failure
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class RoutingMetrics:
    """Container for a single experiment measurement."""

    algorithm: str
    n_nodes: int
    cost_strategy: str # Defined in base_controller; [HOP_COUNT, LATENCY, INV_BANDWIDTH, COMPOSITE]
    execution_time_s: float
    path_cost: float
    path_length: int
    transmission_delay_ms: float
    throughput_mbps: float
    adaptiveness_s: float | None = None
    warm_start_speedup_s: float | None = None
    converged: bool | None = None
    iterations: int | None = None
    is_optimal: bool | None = None

def measure_execution_time(
    algorithm_fn,
    graph: nx.Graph,
    source: int,
    dest: int,
) -> tuple[list[int], float, float]:
    t0 = time.perf_counter()
    path, cost, _ = algorithm_fn(graph, source, dest)
    elapsed = time.perf_counter() - t0
    return path, cost, elapsed


def compute_transmission_delay(
    graph: nx.Graph,
    path: list[int],
) -> float:
    if len(path) < 2:
        return 0.0

    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        data = graph.edges[u, v]
        total += data.get("latency_ms", 1.0)
    return total


def compute_bottleneck_throughput(
    graph: nx.Graph,
    path: list[int],
) -> float:
    if len(path) < 2:
        return 0.0

    bandwidths = []
    for u, v in zip(path[:-1], path[1:]):
        data = graph.edges[u, v]
        bandwidths.append(data.get("bandwidth_mbps", 100.0))
    return min(bandwidths)


def measure_adaptiveness(
    algorithm_fn,
    graph: nx.Graph,
    source: int,
    dest: int,
    fail_u: int,
    fail_v: int,
) -> tuple[list[int], float, float]:
    G_temp = graph.copy()
    if G_temp.has_edge(fail_u, fail_v):
        G_temp.remove_edge(fail_u, fail_v)
    t0 = time.perf_counter()
    result = algorithm_fn(G_temp, source, dest)
    elapsed = time.perf_counter() - t0
    if hasattr(result, 'path'):
        return result.path, result.cost, elapsed
    else:
        return result[0], result[1], elapsed


def enrich_graph_with_sim_attributes(
    graph: nx.Graph,
    rng: np.random.Generator | None = None,
) -> nx.Graph:
    """
    Randomly add simulated latency and bandwidth attributes to graph edges.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    for u, v in graph.edges():
        if "latency_ms" not in graph[u][v]:
            graph[u][v]["latency_ms"] = float(rng.uniform(0.5, 10.0))
        if "bandwidth_mbps" not in graph[u][v]:
            graph[u][v]["bandwidth_mbps"] = float(rng.choice([10, 50, 100, 1000]))

    return graph

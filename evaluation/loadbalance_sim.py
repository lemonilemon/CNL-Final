"""
Flow-level load-balancing simulator (Phase 1 — no Mininet).

Given a topology with link capacities and a traffic matrix of concurrent flows,
each routing *method* produces a per-node split table
    {node: {next_hop: fraction}}
describing how it spreads a flow toward the destination. We push every flow's
demand through its split DAG to get per-link load, then resolve contention with
**max-min fair-share** (progressive filling) to obtain each flow's achieved rate.

This is a fluid / fair-share model, not a packet simulator: it predicts
aggregate throughput, loss (throttled demand), and link-utilization imbalance.
Phase 2 validates these predictions against real Mininet + iperf.

Three methods (see docs/multipath_loadbalancing_plan.md):
  - dijkstra_single : 100% on one hop-count shortest path (the hotspot baseline)
  - ecmp            : equal split across all hop-count shortest paths
  - physarum        : capacity-weighted split from the mu<1 flux (WCMP-style)

Link capacity is modeled per-direction (full-duplex), so a directed edge (u, v)
has capacity equal to the link's bandwidth_mbps.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
import numpy as np

from algorithms.physarum import PhysarumSolver
from algorithms.aco import ACOSolver

Split = dict[int, dict[int, float]]
EPS = 1e-9


# ---------------------------------------------------------------------------
# Split-table builders (one per method)
# ---------------------------------------------------------------------------

def _hop_distance_to(G: nx.Graph, dst: int) -> dict[int, int]:
    return nx.single_source_shortest_path_length(G, dst)


def dijkstra_single_split(G: nx.Graph, src: int, dst: int) -> Split:
    """100% of the flow onto a single hop-count shortest path."""
    path = nx.shortest_path(G, src, dst)  # unweighted = hop count
    split: Split = {}
    for u, v in zip(path[:-1], path[1:]):
        split[u] = {v: 1.0}
    return split


def ecmp_split(G: nx.Graph, src: int, dst: int) -> Split:
    """
    Equal-split ECMP: at each node, distribute equally among neighbors that lie
    one hop closer to the destination (the standard hop-count ECMP next-hop set).
    Ignores link capacity by construction.
    """
    dist = _hop_distance_to(G, dst)
    split: Split = {}
    # Only nodes reachable toward dst matter; restrict to the shortest-path DAG.
    for node in G.nodes():
        if node == dst or node not in dist:
            continue
        nexthops = [w for w in G.neighbors(node)
                    if dist.get(w, 1 << 30) == dist[node] - 1]
        if nexthops:
            f = 1.0 / len(nexthops)
            split[node] = {w: f for w in nexthops}
    return split


def physarum_split(
    G: nx.Graph,
    src: int,
    dst: int,
    mu: float = 0.8,
    max_iter: int = 500,
    patience: int = 20,
) -> Split:
    """
    Capacity-weighted multipath split from the Physarum flux. Link *length* is
    set to inverse bandwidth so conductance favors high-capacity paths; mu < 1
    keeps redundant tubes alive (network-preserving regime). The split is read at
    the solver's natural adaptive operating point, not full collapse.
    """
    H = G.copy()
    for u, v, d in H.edges(data=True):
        d["weight"] = 1000.0 / d.get("bandwidth_mbps", 1000.0)
    solver = PhysarumSolver(H, src, dst, mu=mu, variant="classic",
                            max_iter=max_iter, patience=patience)
    solver.solve()
    return solver.extract_multipath_split()


def aco_split(G: nx.Graph, src: int, dst: int) -> Split:
    """
    Pheromone-weighted multipath split (AntNet-style). ACO has no pressure
    potential to orient flow, so we restrict next-hops to the hop-count
    shortest-path DAG (as ECMP does) but weight them by learned pheromone tau
    instead of splitting equally. With inverse-bandwidth cost, pheromone
    accumulates on high-capacity paths, giving a capacity-aware split.
    """
    H = G.copy()
    for u, v, d in H.edges(data=True):
        d["weight"] = 1000.0 / d.get("bandwidth_mbps", 1000.0)
    solver = ACOSolver(H, src, dst, seed=42)
    solver.solve()
    tau = solver.get_pheromone_matrix()

    dist = _hop_distance_to(G, dst)
    split: Split = {}
    for node in G.nodes():
        if node == dst or node not in dist:
            continue
        i = solver.node_to_idx[node]
        outs: dict[int, float] = {}
        total = 0.0
        for w in G.neighbors(node):
            if dist.get(w, 1 << 30) == dist[node] - 1:
                p = float(tau[i, solver.node_to_idx[w]])
                if p > 0:
                    outs[w] = p
                    total += p
        if total > 0:
            split[node] = {w: p / total for w, p in outs.items()}
    return split


METHODS = {
    "dijkstra_single": dijkstra_single_split,
    "ecmp": ecmp_split,
    "physarum": physarum_split,
    "aco": aco_split,
}


# ---------------------------------------------------------------------------
# Flow propagation: split DAG + unit demand -> fraction of demand per directed edge
# ---------------------------------------------------------------------------

def edge_fractions(src: int, dst: int, split: Split) -> dict[tuple[int, int], float]:
    """
    Propagate a unit of demand from src through the split DAG and return the
    fraction of that demand traversing each directed edge (u, v).
    """
    # Build the directed split graph and topologically sort it.
    dg = nx.DiGraph()
    dg.add_node(src)
    for u, outs in split.items():
        for v in outs:
            dg.add_edge(u, v)
    if not nx.is_directed_acyclic_graph(dg):
        raise ValueError("split table is not a DAG — flux extraction broken")

    order = list(nx.topological_sort(dg))
    inflow: dict[int, float] = defaultdict(float)
    inflow[src] = 1.0
    frac: dict[tuple[int, int], float] = defaultdict(float)
    for node in order:
        f = inflow[node]
        if f <= EPS or node not in split:
            continue
        for nxt, share in split[node].items():
            moved = f * share
            frac[(node, nxt)] += moved
            inflow[nxt] += moved
    return dict(frac)


# ---------------------------------------------------------------------------
# Max-min fair-share allocation
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    aggregate_throughput: float
    offered_load: float
    packet_loss: float            # fraction of offered demand not carried
    max_link_util: float          # worst directed-edge utilization
    util_imbalance: float         # stddev of utilization over used edges


def _max_min_rates(
    flow_fracs: list[dict[tuple[int, int], float]],
    demands: list[float],
    cap: dict[tuple[int, int], float],
) -> list[float]:
    """
    Progressive-filling max-min fairness. Each flow f contributes
    rate[f] * frac_f(e) of load to directed edge e; total per edge <= cap[e].
    Flows grow equally until an edge saturates or a flow meets its demand.
    """
    n = len(demands)
    rate = [0.0] * n
    frozen = [False] * n
    rem = dict(cap)

    while True:
        active = [f for f in range(n) if not frozen[f] and rate[f] < demands[f] - EPS]
        if not active:
            break

        # Per-unit load each edge would receive if all active flows grow by 1.
        edge_rate: dict[tuple[int, int], float] = defaultdict(float)
        for f in active:
            for e, fr in flow_fracs[f].items():
                edge_rate[e] += fr

        # Largest equal increment before some edge saturates...
        delta = float("inf")
        for e, er in edge_rate.items():
            if er > EPS:
                delta = min(delta, rem[e] / er)
        # ...or some active flow reaches its demand.
        for f in active:
            delta = min(delta, demands[f] - rate[f])
        if not np.isfinite(delta) or delta <= EPS:
            delta = max(delta, 0.0)

        for f in active:
            rate[f] += delta
        for e, er in edge_rate.items():
            rem[e] -= delta * er

        # Freeze flows that met demand or now traverse a saturated edge.
        for f in active:
            if rate[f] >= demands[f] - EPS:
                frozen[f] = True
            elif any(rem[e] <= EPS for e in flow_fracs[f]):
                frozen[f] = True
        if delta <= EPS:
            break
    return rate


# ---------------------------------------------------------------------------
# Top-level simulation
# ---------------------------------------------------------------------------

def simulate(
    G: nx.Graph,
    method: str,
    flows: list[tuple[int, int]],
    demand_mbps: float,
    physarum_kwargs: dict | None = None,
) -> SimResult:
    """
    Run one method on one topology + traffic matrix at a given per-flow demand.
    """
    builder = METHODS[method]
    pk = physarum_kwargs or {}

    # Directed full-duplex capacities.
    cap: dict[tuple[int, int], float] = {}
    for u, v, d in G.edges(data=True):
        bw = float(d.get("bandwidth_mbps", 1000.0))
        cap[(u, v)] = bw
        cap[(v, u)] = bw

    flow_fracs: list[dict[tuple[int, int], float]] = []
    demands: list[float] = []
    for src, dst in flows:
        split = builder(G, src, dst, **pk) if method == "physarum" else builder(G, src, dst)
        flow_fracs.append(edge_fractions(src, dst, split))
        demands.append(demand_mbps)

    rates = _max_min_rates(flow_fracs, demands, cap)

    # Realized per-edge load and utilization.
    load: dict[tuple[int, int], float] = defaultdict(float)
    for f, ff in enumerate(flow_fracs):
        for e, fr in ff.items():
            load[e] += rates[f] * fr
    utils = [load[e] / cap[e] for e in load if cap[e] > 0]

    offered = float(sum(demands))
    agg = float(sum(rates))
    return SimResult(
        aggregate_throughput=agg,
        offered_load=offered,
        packet_loss=(1.0 - agg / offered) if offered > 0 else 0.0,
        max_link_util=max(utils) if utils else 0.0,
        util_imbalance=float(np.std(utils)) if utils else 0.0,
    )

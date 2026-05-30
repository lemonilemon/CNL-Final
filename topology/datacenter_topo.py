"""
Data-center topologies for the multipath load-balancing experiment.

Unlike the random Erdos-Renyi mesh (`mesh_topo.py`), these fabrics are built for
*equal-cost multi-path* (ECMP) redundancy: between two hosts on different
leaves/pods there are many equal-hop paths through the fabric core.

Two builders are provided (NetworkX, cross-platform):

  - `get_fat_tree_graph(k)`        — k-ary fat-tree. Uniform capacity, fully
    symmetric. All cross-pod paths are equal cost AND equal capacity, so plain
    ECMP balances it perfectly. Used to show Physarum does not *regress* on the
    easy case.

  - `get_leaf_spine_graph(...)`    — 2-tier folded-Clos with *heterogeneous*
    leaf<->spine link capacities. All spine paths are still equal *hop count*
    (so ECMP uses them all, equally), but they differ in *capacity*. Equal-split
    ECMP therefore overloads the thin spines. This is the WCMP motivation that
    Physarum's capacity-proportional split exploits. The headline case.

Modeling convention (see docs/multipath_loadbalancing_plan.md):
  - Every edge carries `bandwidth_mbps` (link capacity) and `latency_ms`.
  - Routing *cost* is chosen per method by the simulator, NOT stored here:
      * ECMP / Dijkstra use hop-count  -> all fabric paths are equal-cost.
      * Physarum uses inverse-bandwidth -> conductance favors fat pipes (WCMP).

Each builder returns `(G, meta)` where `meta` carries the host list and the
"block" (leaf id / pod id) each host belongs to, so the traffic generator can
pick host pairs that actually cross the fabric core.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np


@dataclass
class TopoMeta:
    """Side information about a data-center graph."""

    kind: str                                   # "fat_tree" | "leaf_spine"
    hosts: list[int] = field(default_factory=list)
    host_block: dict[int, int] = field(default_factory=dict)  # host -> pod/leaf id
    host_access: dict[int, int] = field(default_factory=dict)  # host -> attached switch


def _add_link(G: nx.Graph, u: int, v: int, bw_mbps: float, latency_ms: float = 1.0) -> None:
    G.add_edge(u, v, bandwidth_mbps=float(bw_mbps), latency_ms=float(latency_ms))


# ---------------------------------------------------------------------------
# Fat-Tree (symmetric, uniform capacity)
# ---------------------------------------------------------------------------

def get_fat_tree_graph(k: int = 4, link_bw: float = 1000.0, host_bw: float = 10000.0):
    """
    Build a standard k-ary fat-tree (k must be even).

    Structure:
      - (k/2)^2 core switches
      - k pods, each with k/2 aggregation + k/2 edge switches
      - k/2 hosts per edge switch  (total k^3/4 hosts)

    All fabric links share `link_bw`; host access links use `host_bw` (kept high
    so the fabric, not the access link, is the bottleneck). Every cross-pod
    host pair has (k/2)^2 equal-cost, equal-capacity paths.
    """
    if k % 2 != 0:
        raise ValueError("k must be even")
    half = k // 2
    G = nx.Graph()
    meta = TopoMeta(kind="fat_tree")

    nid = 0

    def new() -> int:
        nonlocal nid
        n = nid
        nid += 1
        return n

    # Core switches: core[j][i], j in [0,half), i in [0,half)
    core = [[new() for _ in range(half)] for _ in range(half)]
    for j in range(half):
        for i in range(half):
            G.add_node(core[j][i], kind="core")

    for p in range(k):  # pods
        agg = [new() for _ in range(half)]
        edge = [new() for _ in range(half)]
        for a in agg:
            G.add_node(a, kind="agg", pod=p)
        for e in edge:
            G.add_node(e, kind="edge", pod=p)

        # agg <-> core: agg index j connects to core group j (core[j][*])
        for j in range(half):
            for i in range(half):
                _add_link(G, agg[j], core[j][i], link_bw)

        # edge <-> agg: full mesh within the pod
        for e in edge:
            for a in agg:
                _add_link(G, e, a, link_bw)

        # hosts under each edge switch
        for e in edge:
            for _ in range(half):
                h = new()
                G.add_node(h, kind="host", pod=p)
                _add_link(G, h, e, host_bw)
                meta.hosts.append(h)
                meta.host_block[h] = p
                meta.host_access[h] = e

    return G, meta


# ---------------------------------------------------------------------------
# Leaf-Spine (heterogeneous capacity) — the headline case
# ---------------------------------------------------------------------------

def get_leaf_spine_graph(
    n_leaf: int = 4,
    n_spine: int = 4,
    hosts_per_leaf: int = 4,
    host_bw: float = 10000.0,
    bw_choices: tuple[float, ...] = (100.0, 400.0, 1000.0),
    seed: int = 42,
):
    """
    Build a 2-tier leaf-spine (folded Clos) with full leaf<->spine bipartite
    connectivity and *heterogeneous* leaf<->spine link capacities.

    Between two hosts on different leaves there are `n_spine` equal-hop paths
    (leaf -> spine -> leaf), one per spine. Their *capacities* differ, drawn from
    `bw_choices`. Equal-split ECMP ignores this and overloads the thin spines;
    Physarum (inverse-bandwidth conductance) splits proportionally to capacity.
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    meta = TopoMeta(kind="leaf_spine")

    spines = list(range(n_spine))
    leaves = list(range(n_spine, n_spine + n_leaf))
    for s in spines:
        G.add_node(s, kind="spine")
    for lf in leaves:
        G.add_node(lf, kind="leaf")

    # Heterogeneous leaf<->spine capacities
    for lf in leaves:
        for s in spines:
            bw = float(rng.choice(bw_choices))
            _add_link(G, lf, s, bw)

    # Hosts under each leaf
    nid = n_spine + n_leaf
    for lf in leaves:
        for _ in range(hosts_per_leaf):
            h = nid
            nid += 1
            G.add_node(h, kind="host", leaf=lf)
            _add_link(G, h, lf, host_bw)
            meta.hosts.append(h)
            meta.host_block[h] = lf
            meta.host_access[h] = lf

    return G, meta


# ---------------------------------------------------------------------------
# Traffic matrix
# ---------------------------------------------------------------------------

def cross_block_pairs(
    meta: TopoMeta,
    n_pairs: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """
    Sample `n_pairs` distinct (src_host, dst_host) pairs that live in *different*
    blocks (pods / leaves), so every flow must cross the fabric core. Each host
    is used as a source at most once (models concurrent elephant flows from
    distinct senders).
    """
    hosts = list(meta.hosts)
    rng.shuffle(hosts)
    pairs: list[tuple[int, int]] = []
    used_src: set[int] = set()
    for src in hosts:
        if len(pairs) >= n_pairs:
            break
        if src in used_src:
            continue
        candidates = [h for h in meta.hosts
                      if meta.host_block[h] != meta.host_block[src]]
        if not candidates:
            continue
        dst = candidates[int(rng.integers(0, len(candidates)))]
        pairs.append((src, dst))
        used_src.add(src)
    return pairs

from __future__ import annotations

import time
from dataclasses import dataclass

import networkx as nx

@dataclass
class DijkstraResult:
    path: list[int] # Node sequence from source to destination.
    cost: float # Sum of edge weight
    elapsed_seconds: float

class DijkstraSolver:
    def __init__(self, graph: nx.Graph, source: int, dest: int):
        if source == dest:
            raise ValueError("source and dest must differ")
        if source not in graph or dest not in graph:
            raise ValueError("source or dest not in graph")
        self.graph = graph
        self.source = source
        self.dest = dest

    def solve(self) -> DijkstraResult:
        t0 = time.perf_counter()

        try:
            path = nx.dijkstra_path(
                self.graph, self.source, self.dest, weight="weight"
            )
            cost = nx.dijkstra_path_length(
                self.graph, self.source, self.dest, weight="weight"
            )
        except nx.NetworkXNoPath:
            path = []
            cost = float("inf")

        elapsed = time.perf_counter() - t0
        return DijkstraResult(path=path, cost=cost, elapsed_seconds=elapsed)

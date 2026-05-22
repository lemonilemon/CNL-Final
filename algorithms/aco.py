"""
Reference:
  - Dorigo, M. & Stützle, T. (2004). "Ant Colony Optimization." MIT Press.
I have checked that the implementation is correct, or at least the same as Wikipedia
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

@dataclass
class ACOResult:
    path: list[int] # Node sequence form source to destination
    cost: float # Sum of edge weight of best path
    iterations: int
    converged: bool
    elapsed_seconds: float
    best_cost_history: list[float] = field(default_factory=list)

class ACOSolver:
    def __init__(
        self,
        graph: nx.Graph,
        source: int,
        dest: int,
        n_ants: int = 20,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation: float = 0.5,
        q: float = 100.0,
        max_iter: int = 200,
        patience: int = 20, # For convergence
        seed: int | None = None,
    ):
        if source == dest:
            raise ValueError("source and dest must differ")
        if source not in graph or dest not in graph:
            raise ValueError("source or dest not in graph")

        self.graph = graph
        self.source = source
        self.dest = dest
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.q = q
        self.max_iter = max_iter
        self.patience = patience
        self.rng = np.random.default_rng(seed)

        # Map nodes to contiguous indices (sorted for cross-instance consistency)
        self.nodes = sorted(list(graph.nodes()))
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.N = len(self.nodes)

        self._build_matrices()

    def _build_matrices(self) -> None: # Initialize parameters and graph
        N = self.N
        self.dist = np.full((N, N), np.inf)
        self.eta = np.zeros((N, N))      # heuristic = 1 / distance
        self.tau = np.ones((N, N)) * 0.1  # initial pheromone

        for u, v, data in self.graph.edges(data=True):
            i, j = self.node_to_idx[u], self.node_to_idx[v]
            w = data.get("weight", 1.0)
            self.dist[i, j] = w
            self.dist[j, i] = w
            self.eta[i, j] = 1.0 / w
            self.eta[j, i] = 1.0 / w

        # Adjacency list
        self.adj: list[list[int]] = [[] for _ in range(N)]
        for u, v in self.graph.edges():
            i, j = self.node_to_idx[u], self.node_to_idx[v]
            self.adj[i].append(j)
            self.adj[j].append(i)

    def _construct_path(self) -> tuple[list[int], float]:
        """
        Construct a single ant's path from source to dest. (Edge selection in Wikipedia)

        Returns (path_indices, cost) or ([], inf) if the ant gets stuck.
        """
        src = self.node_to_idx[self.source]
        dst = self.node_to_idx[self.dest]

        visited = {src}
        path = [src]
        cost = 0.0

        current = src
        while current != dst:
            neighbors = [n for n in self.adj[current] if n not in visited]
            if not neighbors:
                return [], float("inf")  # stuck

            # Compute selection probabilities
            probs = np.zeros(len(neighbors))
            for k, n in enumerate(neighbors):
                probs[k] = (
                    self.tau[current, n] ** self.alpha
                    * self.eta[current, n] ** self.beta
                )

            total = probs.sum()
            if total == 0: # Seems impossible
                return [], float("inf")
            probs /= total

            # Roulette wheel selection
            chosen_idx = self.rng.choice(len(neighbors), p=probs)
            next_node = neighbors[chosen_idx]

            path.append(next_node)
            visited.add(next_node)
            cost += self.dist[current, next_node]
            current = next_node

        return path, cost

    def _update_pheromone(
        self, all_paths: list[tuple[list[int], float]]
    ) -> None:
        # Evaporation
        self.tau *= 1.0 - self.evaporation

        # Deposit
        for path, cost in all_paths:
            if cost == float("inf") or len(path) == 0:
                continue
            deposit = self.q / cost
            for a, b in zip(path[:-1], path[1:]):
                self.tau[a, b] += deposit
                self.tau[b, a] += deposit

    def solve(self, init_tau: np.ndarray | None = None) -> ACOResult:
        t0 = time.perf_counter()

        if init_tau is not None:
            self.tau = init_tau.copy()

        best_path: list[int] = []
        best_cost = float("inf")
        best_cost_history: list[float] = []
        stable_count = 0 # For convergence
        converged = False
        iterations = 0

        for it in range(1, self.max_iter + 1):
            iterations = it
            all_paths: list[tuple[list[int], float]] = []

            for _ in range(self.n_ants):
                path, cost = self._construct_path()
                all_paths.append((path, cost))

                if cost < best_cost:
                    best_cost = cost
                    best_path = path

            self._update_pheromone(all_paths)
            best_cost_history.append(best_cost)

            # Check convergence
            if it > 1 and best_cost_history[-1] == best_cost_history[-2]:
                stable_count += 1
                if stable_count >= self.patience:
                    converged = True
                    break
            else:
                stable_count = 0

        # Convert indices back to node ids
        result_path = [self.nodes[i] for i in best_path] if best_path else []
        elapsed = time.perf_counter() - t0

        return ACOResult(
            path=result_path,
            cost=best_cost,
            iterations=iterations,
            converged=converged,
            elapsed_seconds=elapsed,
            best_cost_history=best_cost_history,
        )

    def get_pheromone_matrix(self) -> np.ndarray:
        return self.tau.copy()

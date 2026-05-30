"""
Reference papers:
  - Tero, A. et al. (2007). "A mathematical model for adaptive transport
    network in path finding by true slime mold."
  - Zhang, X. et al. (2014). "An Improved Physarum polycephalum Algorithm
    for the Shortest Path Problem." The Scientific World Journal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


@dataclass
class PhysarumResult:
    path: list[int]  # Node sequence from source to destination
    cost: float  # Sum of edge weights
    iterations: int
    converged: bool
    elapsed_seconds: float
    conductivity_history: list[np.ndarray] = field(default_factory=list)
    variant: str = "classic"  # Either classic or improved


class PhysarumSolver:

    def __init__(
        self,
        graph: nx.Graph,
        source: int,
        dest: int,
        mu: float = 1.0,  # Make f(x) = |x|
        decay: float = 0.1,
        dt: float = 0.05,
        max_iter: int = 500,
        epsilon: float = 1e-6,
        patience: int = 20,
        variant: str = "improved",
        record_history: bool = False,
        extraction_policy: str = "dijkstra",  # Either dijkstra or greedy
    ):
        if source == dest:
            raise ValueError("source and dest must differ")
        if source not in graph or dest not in graph:
            raise ValueError("source or dest not in graph")

        self.graph = graph
        self.source = source
        self.dest = dest
        self.mu = mu
        self.decay = decay
        self.dt = dt
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.patience = patience
        self.variant = variant
        self.record_history = record_history
        self.extraction_policy = extraction_policy

        self.nodes = sorted(list(graph.nodes()))
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.N = len(self.nodes)

        self.src_idx = self.node_to_idx[source]
        self.dst_idx = self.node_to_idx[dest]

        self._build_matrices()

    def _build_matrices(self) -> None:  # Initialize parameter and construct graph
        N = self.N
        self.L = np.zeros((N, N))  # length / weight
        self.D = np.zeros((N, N))  # conductivity

        for u, v, data in self.graph.edges(data=True):
            i, j = self.node_to_idx[u], self.node_to_idx[v]
            w = data.get("weight", 1.0)
            self.L[i, j] = w
            self.L[j, i] = w
            # Initialize conductivity uniformly (> 0)
            self.D[i, j] = 1.0
            self.D[j, i] = 1.0

    def _solve_pressure(self) -> np.ndarray:
        """
        Solve for node pressures using Kirchhoff's current law.

        Following Tero et al., we fix reference pressure at the sink (dest)
        node to 0, and inject a constant current I_0 = 1.0 at the source.
        """
        N = self.N
        # Conductance matrix G_ij = D_ij / L_ij (only where L > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            G = np.where(self.L > 0, self.D / self.L, 0.0)

        # Vectorized Laplacian construction
        A = -G.copy()
        # Sum of rows for diagonal
        np.fill_diagonal(A, G.sum(axis=1))

        # Apply boundary conditions (p_dest = 0)
        A[self.dst_idx, :] = 0.0
        A[self.dst_idx, self.dst_idx] = 1.0

        b = np.zeros(self.N)
        b[self.src_idx] = 1.0  # Inflow at source is 1.0
        # b[self.dst_idx] is already 0.0

        A_sparse = csr_matrix(A)
        p = spsolve(A_sparse, b)
        return p

    def _compute_flux(self, p: np.ndarray) -> np.ndarray:
        N = self.N
        # Vectorized flux computation
        p_diff = p[:, None] - p[None, :]
        mask = (self.L > 0) & (self.D > 0)
        Q = np.zeros((N, N))
        Q[mask] = self.D[mask] * p_diff[mask] / self.L[mask]
        return Q

    def _update_conductivity_classic(self, Q: np.ndarray) -> np.ndarray:
        # Note that Q here is Q/L in our slide
        abs_Q = np.abs(Q)
        f_Q = abs_Q**self.mu  # mu = 1, then f(x) = |x|, as in our slide and paper
        dD = f_Q - self.decay * self.D
        D_new = self.D + self.dt * dD
        D_new = np.maximum(D_new, 0.0)
        return D_new

    def _update_conductivity_improved(self, Q: np.ndarray, p: np.ndarray) -> np.ndarray:
        # Note that Q here is Q/L in our slide
        N = self.N
        abs_Q = np.abs(Q)
        f_Q = abs_Q ** self.mu # mu = 1, then f(x) = |x|, as in our slide and paper

        p_s = p[self.src_idx]
        p_e = p[self.dst_idx]
        pressure_diff = max(abs(p_s - p_e), 1e-12)

        p_diff_local = np.abs(p[:, None] - p[None, :])
        energy_ratio = p_diff_local / pressure_diff

        mask = self.L > 0
        dD = np.zeros((N, N))
        dD[mask] = f_Q[mask] * (1.0 + energy_ratio[mask]) - self.decay * self.D[mask]

        D_new = self.D + self.dt * dD
        D_new = np.maximum(D_new, 0.0)
        return D_new

    def _extract_path_greedy(
        self,
    ) -> tuple[list[int], float]:  # Optimal if solver converges
        """
        Greedily trace a path from source to dest by selecting the neighbor
        with the highest conductivity at each step.
        """
        current = self.source
        visited = {current}
        path = [current]

        while current != self.dest:
            curr_idx = self.node_to_idx[current]
            neighbors = list(self.graph.neighbors(current))

            best_neighbor = None
            best_conductivity = -1.0

            for nbr in neighbors:
                if nbr not in visited:
                    nbr_idx = self.node_to_idx[nbr]
                    cond = self.D[curr_idx, nbr_idx]
                    if cond > best_conductivity:
                        best_conductivity = cond
                        best_neighbor = nbr

            if best_neighbor is None or best_conductivity < 1e-5:
                return [], float("inf")

            current = best_neighbor
            visited.add(current)
            path.append(current)

        cost = 0.0
        for a, b in zip(path[:-1], path[1:]):
            i, j = self.node_to_idx[a], self.node_to_idx[b]
            cost += self.L[i, j]
        return path, cost

    def _extract_path(self) -> tuple[list[int], float]:
        if self.extraction_policy == "greedy":
            return self._extract_path_greedy()

        G_path = nx.Graph()
        for i in range(self.N):
            for j in range(i + 1, self.N):
                if self.D[i, j] > 1e-12:
                    # Weight = length / conductivity (prefer high-D, short-L)
                    w = self.L[i, j] / self.D[i, j]
                    G_path.add_edge(self.nodes[i], self.nodes[j], weight=w)

        try:
            path = nx.dijkstra_path(G_path, self.source, self.dest, weight="weight")
            # Compute true cost using original lengths
            cost = 0.0
            for a, b in zip(path[:-1], path[1:]):
                i, j = self.node_to_idx[a], self.node_to_idx[b]
                cost += self.L[i, j]
            return path, cost
        except nx.NetworkXNoPath:
            return [], float("inf")

    def solve(self, init_D: np.ndarray | None = None) -> PhysarumResult:
        t0 = time.perf_counter()

        if init_D is not None:
            self.D = init_D.copy()

        history: list[np.ndarray] = []
        prev_path: list[int] = []
        path_stable_count = 0

        converged = False
        iterations = 0

        for it in range(1, self.max_iter + 1):
            iterations = it

            p = self._solve_pressure()
            Q = self._compute_flux(p)

            if self.variant == "classic":
                D_new = self._update_conductivity_classic(Q)
            else:
                D_new = self._update_conductivity_improved(Q, p)

            if self.record_history:
                history.append(D_new.copy())

            max_change = np.max(np.abs(D_new - self.D))
            self.D = D_new
            if max_change < self.epsilon:
                converged = True
                break

            # Check convergence
            path, _ = self._extract_path()
            if path == prev_path:
                path_stable_count += 1
                if path_stable_count >= self.patience:
                    converged = True
                    break
            else:
                path_stable_count = 0
                prev_path = path

        final_path, final_cost = self._extract_path()
        elapsed = time.perf_counter() - t0

        return PhysarumResult(
            path=final_path,
            cost=final_cost,
            iterations=iterations,
            converged=converged,
            elapsed_seconds=elapsed,
            conductivity_history=history,
            variant=self.variant,
        )

    def get_all_conductivities(self) -> np.ndarray:
        return self.D.copy()

    def extract_multipath_split(
        self, threshold: float = 1e-4
    ) -> dict[int, dict[int, float]]:
        """
        Export the steady-state flow as a per-node multipath split table.

        Instead of collapsing the conductivity field to a single path, we read
        the steady-state flux Q_ij = D_ij * (p_i - p_j) / L_ij (the current on
        each edge) and, at every node, split outgoing flow across neighbors in
        proportion to the *positive outgoing* flux (current flowing toward the
        sink).

        Because pressure is a scalar potential (p_src highest, p_dest = 0),
        current always flows down the gradient, so the result is a loop-free
        flow DAG that conserves flow at every node (Kirchhoff). This only yields
        genuine multipath when run in the sublinear regime (mu < 1); with mu = 1
        and full convergence the flux concentrates on a single tube.

        Returns
        -------
        dict[node, dict[next_hop, fraction]]
            Fractions at each node sum to 1.0. The destination is absent.
        """
        p = self._solve_pressure()
        Q = self._compute_flux(p)  # Q[i, j] > 0 means current flows i -> j

        split: dict[int, dict[int, float]] = {}
        for node in self.nodes:
            if node == self.dest:
                continue
            i = self.node_to_idx[node]
            outs: dict[int, float] = {}
            total = 0.0
            for nbr in self.graph.neighbors(node):
                j = self.node_to_idx[nbr]
                q = Q[i, j]
                if q > threshold:
                    outs[nbr] = q
                    total += q
            if total > 0:
                split[node] = {nb: q / total for nb, q in outs.items()}
        return split

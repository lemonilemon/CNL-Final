import networkx as nx
import pytest

from algorithms.aco import ACOSolver


def diamond_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_edge(0, 1, weight=5.0)
    G.add_edge(0, 2, weight=2.0)
    G.add_edge(1, 3, weight=4.0)
    G.add_edge(2, 3, weight=1.0)
    return G


def parallel_paths_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_edge(0, 1, weight=5.0)
    G.add_edge(1, 3, weight=5.0)
    G.add_edge(0, 2, weight=1.0)
    G.add_edge(2, 3, weight=2.0)
    return G


class TestACO:
    def test_diamond_finds_shortest_path(self):
        """ACO should find the optimal path on a simple graph."""
        G = diamond_graph()
        solver = ACOSolver(G, 0, 3, n_ants=30, max_iter=100, seed=42)
        result = solver.solve()
        assert result.path == [0, 2, 3]
        assert result.cost == pytest.approx(3.0)

    def test_parallel_paths(self):
        G = parallel_paths_graph()
        solver = ACOSolver(G, 0, 3, n_ants=30, max_iter=100, seed=42)
        result = solver.solve()
        assert result.path == [0, 2, 3]
        assert result.cost == pytest.approx(3.0)

    def test_statistical_reliability(self):
        """Over 50 runs, ACO should find the optimal path ≥ 80% of the time."""
        G = diamond_graph()
        successes = 0
        for seed in range(50):
            solver = ACOSolver(G, 0, 3, n_ants=20, max_iter=100, seed=seed)
            result = solver.solve()
            if result.cost == pytest.approx(3.0):
                successes += 1
        assert successes / 50 >= 0.80

    def test_convergence_flag(self):
        G = diamond_graph()
        solver = ACOSolver(G, 0, 3, n_ants=30, max_iter=500, patience=20, seed=42)
        result = solver.solve()
        assert result.converged is True
        assert result.iterations < 500

    def test_cost_history_decreasing(self):
        """Best cost should be non-increasing over iterations."""
        G = parallel_paths_graph()
        solver = ACOSolver(G, 0, 3, n_ants=20, max_iter=100, seed=42)
        result = solver.solve()
        for i in range(1, len(result.best_cost_history)):
            assert result.best_cost_history[i] <= result.best_cost_history[i - 1]

    def test_source_equals_dest_raises(self):
        G = diamond_graph()
        with pytest.raises(ValueError):
            ACOSolver(G, 0, 0)

import networkx as nx
import pytest

from algorithms.physarum import PhysarumSolver

def diamond_graph() -> nx.Graph:
    """
    Diamond graph with known shortest path 0→2→3 (cost=3).

        0 --5-- 1
        |       |
        2       4
        |       |
        2 --1-- 3
    """
    G = nx.Graph()
    G.add_edge(0, 1, weight=5.0)
    G.add_edge(0, 2, weight=2.0)
    G.add_edge(1, 3, weight=4.0)
    G.add_edge(2, 3, weight=1.0)
    return G


def linear_graph() -> nx.Graph:
    """Linear graph: 0 --1-- 1 --2-- 2 --3-- 3.  Shortest = only path."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2, weight=2.0)
    G.add_edge(2, 3, weight=3.0)
    return G


def parallel_paths_graph() -> nx.Graph:
    """
    Two parallel paths from 0 to 3:
      - Upper: 0 → 1 → 3  cost = 10
      - Lower: 0 → 2 → 3  cost = 3
    """
    G = nx.Graph()
    G.add_edge(0, 1, weight=5.0)
    G.add_edge(1, 3, weight=5.0)
    G.add_edge(0, 2, weight=1.0)
    G.add_edge(2, 3, weight=2.0)
    return G


def grid_graph() -> nx.Graph:
    """
    3x3 grid graph with uniform weights.

    0 - 1 - 2
    |   |   |
    3 - 4 - 5
    |   |   |
    6 - 7 - 8
    """
    G = nx.Graph()
    for i in range(3):
        for j in range(3):
            node = i * 3 + j
            if j < 2:
                G.add_edge(node, node + 1, weight=1.0)
            if i < 2:
                G.add_edge(node, node + 3, weight=1.0)
    return G

class TestPhysarumClassic:
    """Tests for the classic Physarum variant."""

    def test_diamond_finds_shortest_path(self):
        G = diamond_graph()
        solver = PhysarumSolver(G, 0, 3, variant="classic", max_iter=300)
        result = solver.solve()
        assert result.path == [0, 2, 3]
        assert result.cost == pytest.approx(3.0)

    def test_linear_graph(self):
        G = linear_graph()
        solver = PhysarumSolver(G, 0, 3, variant="classic", max_iter=300)
        result = solver.solve()
        assert result.path == [0, 1, 2, 3]
        assert result.cost == pytest.approx(6.0)

    def test_parallel_paths(self):
        G = parallel_paths_graph()
        solver = PhysarumSolver(G, 0, 3, variant="classic", max_iter=300)
        result = solver.solve()
        assert result.path == [0, 2, 3]
        assert result.cost == pytest.approx(3.0)

    def test_grid_graph(self):
        G = grid_graph()
        solver = PhysarumSolver(G, 0, 8, variant="classic", max_iter=500)
        result = solver.solve()
        # Shortest path cost in a 3x3 grid from corner to corner = 4
        assert result.cost == pytest.approx(4.0)
        assert len(result.path) == 5  # 5 nodes in a 4-hop path

    def test_convergence_flag(self):
        G = diamond_graph()
        solver = PhysarumSolver(G, 0, 3, variant="classic", max_iter=1000, patience=5)
        result = solver.solve()
        assert result.converged is True
        assert result.iterations < 1000

class TestPhysarumImproved:
    """Tests for the improved (energy-based) Physarum variant."""

    def test_diamond_finds_shortest_path(self):
        G = diamond_graph()
        solver = PhysarumSolver(G, 0, 3, variant="improved", max_iter=300)
        result = solver.solve()
        assert result.path == [0, 2, 3]
        assert result.cost == pytest.approx(3.0)

    def test_parallel_paths(self):
        G = parallel_paths_graph()
        solver = PhysarumSolver(G, 0, 3, variant="improved", max_iter=300)
        result = solver.solve()
        assert result.path == [0, 2, 3]
        assert result.cost == pytest.approx(3.0)

    def test_grid_graph(self):
        G = grid_graph()
        solver = PhysarumSolver(G, 0, 8, variant="improved", max_iter=500)
        result = solver.solve()
        assert result.cost == pytest.approx(4.0)

    def test_converges_faster_than_classic(self):
        """The improved variant should generally converge in fewer iterations."""
        G = parallel_paths_graph()

        classic = PhysarumSolver(G, 0, 3, variant="classic", max_iter=500, patience=10)
        improved = PhysarumSolver(G, 0, 3, variant="improved", max_iter=500, patience=10)

        r_classic = classic.solve()
        r_improved = improved.solve()

        # Both should find the same path
        assert r_classic.cost == pytest.approx(r_improved.cost)
        # Improved should converge in fewer (or equal) iterations
        assert r_improved.iterations <= r_classic.iterations + 5  # small tolerance

class TestPhysarumEdgeCases:
    def test_source_equals_dest_raises(self):
        G = diamond_graph()
        with pytest.raises(ValueError, match="source and dest must differ"):
            PhysarumSolver(G, 0, 0, variant="classic")

    def test_missing_node_raises(self):
        G = diamond_graph()
        with pytest.raises(ValueError, match="source or dest not in graph"):
            PhysarumSolver(G, 0, 99, variant="classic")

    def test_single_edge_graph(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=7.0)
        solver = PhysarumSolver(G, 0, 1, variant="improved", max_iter=100)
        result = solver.solve()
        assert result.path == [0, 1]
        assert result.cost == pytest.approx(7.0)

    def test_record_history(self):
        G = diamond_graph()
        solver = PhysarumSolver(
            G, 0, 3, variant="classic", max_iter=50, record_history=True
        )
        result = solver.solve()
        assert len(result.conductivity_history) == result.iterations

    def test_warm_start_fewer_iterations(self):
        G = grid_graph()
        # Cold start on original graph
        solver_orig = PhysarumSolver(G, 0, 8, variant="improved", max_iter=500)
        res_orig = solver_orig.solve()
        
        path = res_orig.path
        assert len(path) >= 3
        # Fail the second link in the path
        fail_u, fail_v = path[1], path[2]
        
        # Create degraded graph G_deg
        G_deg = G.copy()
        G_deg.remove_edge(fail_u, fail_v)
        
        # 1. Cold start on degraded graph
        solver_cold = PhysarumSolver(G_deg, 0, 8, variant="improved", max_iter=500)
        res_cold = solver_cold.solve()
        
        # 2. Warm start on degraded graph
        # Copy D from original run, set the failed edge conductivity to 0.0
        D_warm = solver_orig.D.copy()
        u_idx = solver_orig.node_to_idx[fail_u]
        v_idx = solver_orig.node_to_idx[fail_v]
        D_warm[u_idx, v_idx] = 0.0
        D_warm[v_idx, u_idx] = 0.0
        
        solver_warm = PhysarumSolver(G_deg, 0, 8, variant="improved", max_iter=500)
        res_warm = solver_warm.solve(init_D=D_warm)
        
        # Warm start should converge in fewer iterations
        assert res_warm.iterations < res_cold.iterations
        assert len(res_warm.path) > 0
        assert len(res_cold.path) > 0

class TestPhysarumGreedy:
    """Tests for the greedy path extraction policy."""

    def test_greedy_diamond_finds_path(self):
        G = diamond_graph()
        solver = PhysarumSolver(
            G, 0, 3, variant="improved", max_iter=300, extraction_policy="greedy"
        )
        result = solver.solve()
        assert result.path == [0, 2, 3]
        assert result.cost == pytest.approx(3.0)

    def test_greedy_parallel_paths(self):
        G = parallel_paths_graph()
        solver = PhysarumSolver(
            G, 0, 3, variant="improved", max_iter=300, extraction_policy="greedy"
        )
        result = solver.solve()
        assert result.path == [0, 2, 3]
        assert result.cost == pytest.approx(3.0)


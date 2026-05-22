import networkx as nx
import pytest

from algorithms.dijkstra import DijkstraSolver


def diamond_graph() -> nx.Graph:
    G = nx.Graph()
    G.add_edge(0, 1, weight=5.0)
    G.add_edge(0, 2, weight=2.0)
    G.add_edge(1, 3, weight=4.0)
    G.add_edge(2, 3, weight=1.0)
    return G


class TestDijkstra:
    def test_diamond_shortest_path(self):
        G = diamond_graph()
        solver = DijkstraSolver(G, 0, 3)
        result = solver.solve()
        assert result.path == [0, 2, 3]
        assert result.cost == pytest.approx(3.0)

    def test_single_edge(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=5.0)
        solver = DijkstraSolver(G, 0, 1)
        result = solver.solve()
        assert result.path == [0, 1]
        assert result.cost == pytest.approx(5.0)

    def test_disconnected_graph(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_node(2)
        solver = DijkstraSolver(G, 0, 2)
        result = solver.solve()
        assert result.path == []
        assert result.cost == float("inf")

    def test_source_equals_dest_raises(self):
        G = diamond_graph()
        with pytest.raises(ValueError):
            DijkstraSolver(G, 0, 0)

    def test_elapsed_time_is_positive(self):
        G = diamond_graph()
        result = DijkstraSolver(G, 0, 3).solve()
        assert result.elapsed_seconds >= 0

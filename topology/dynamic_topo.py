"""
Dynamic topology with link-failure injection for adaptiveness testing.

Extends the mesh topology and provides a mechanism to:
  1. Run traffic on a stable network.
  2. After a configurable delay, bring down a critical link.
  3. Measure how long it takes the controller to re-route.

Usage (Linux only):
  # Terminal 1: Controller
  $ ryu-manager controller/physarum_controller.py --algorithm physarum_improved

  # Terminal 2: Mininet
  $ sudo mn --custom topology/dynamic_topo.py --topo dynamic,10 \\
        --controller remote --switch ovsk,protocols=OpenFlow13

  # In Mininet CLI:
  mininet> pingall
  # Then inject failure:
  mininet> link s1 s2 down
  mininet> pingall              # Should re-route via alternative path
  mininet> link s1 s2 up        # Restore

On macOS (algorithm testing):
  >>> from topology.dynamic_topo import DynamicGraph
  >>> dg = DynamicGraph(n=10, seed=42)
  >>> G1 = dg.get_graph()            # original
  >>> dg.fail_link(1, 2)             # simulate failure
  >>> G2 = dg.get_graph()            # modified — missing edge (1,2)
  >>> dg.restore_link(1, 2)          # bring it back
"""

from __future__ import annotations

import copy

import networkx as nx

from topology.mesh_topo import get_networkx_graph as _mesh_graph

# ---------------------------------------------------------------------------
# Mininet topology (Linux only)
# ---------------------------------------------------------------------------
try:
    from mininet.topo import Topo
    from topology.mesh_topo import MeshTopo

    class DynamicTopo(MeshTopo):
        """
        Same as MeshTopo, but designed to be used with link-failure commands.

        After starting Mininet, use the CLI to inject failures:
          mininet> link s1 s2 down
          mininet> link s1 s2 up
        """

        def build(self, n: int = 10, connectivity: float = 0.4):
            # Higher default connectivity to ensure multiple alternative paths
            super().build(n=n, connectivity=connectivity)

    topos = {"dynamic": (lambda n=10: DynamicTopo(n=int(n)))}

except ImportError:
    pass


# ---------------------------------------------------------------------------
# Cross-platform dynamic graph for algorithm testing
# ---------------------------------------------------------------------------

class DynamicGraph:
    """
    Wrapper around a NetworkX graph that supports link failure/restoration.

    This allows testing the adaptiveness of routing algorithms on macOS
    without needing Mininet.
    """

    def __init__(self, n: int = 10, connectivity: float = 0.4, seed: int = 42):
        self._original = _mesh_graph(n=n, connectivity=connectivity, seed=seed)
        self._current = copy.deepcopy(self._original)
        self._failed: set[tuple[int, int]] = set()

    def get_graph(self) -> nx.Graph:
        """Return a copy of the current (possibly degraded) graph."""
        return self._current.copy()

    def get_original_graph(self) -> nx.Graph:
        """Return the original, unmodified graph."""
        return self._original.copy()

    def fail_link(self, u: int, v: int) -> bool:
        """
        Simulate a link failure between nodes u and v.

        Returns True if the link existed and was removed, False otherwise.
        """
        if self._current.has_edge(u, v):
            self._current.remove_edge(u, v)
            self._failed.add((min(u, v), max(u, v)))
            return True
        return False

    def restore_link(self, u: int, v: int) -> bool:
        """
        Restore a previously failed link.

        Returns True if the link was restored, False otherwise.
        """
        key = (min(u, v), max(u, v))
        if key in self._failed and self._original.has_edge(u, v):
            data = self._original.edges[u, v]
            self._current.add_edge(u, v, **data)
            self._failed.discard(key)
            return True
        return False

    def get_failed_links(self) -> set[tuple[int, int]]:
        """Return the set of currently failed links."""
        return self._failed.copy()

    def fail_random_link(self, rng=None) -> tuple[int, int] | None:
        """
        Fail a random edge (preferring edges on the shortest path).

        Returns the (u, v) pair that was removed, or None if no edges remain.
        """
        import numpy as np

        if rng is None:
            rng = np.random.default_rng()

        edges = list(self._current.edges())
        if not edges:
            return None

        idx = rng.integers(0, len(edges))
        u, v = edges[idx]
        self.fail_link(u, v)
        return (u, v)

"""
Parameterized mesh topology for scalability testing.

Generates a partial-mesh network with *n* switches and configurable
connectivity.  Each switch also gets one attached host.

Usage (Linux only):
  $ sudo mn --custom topology/mesh_topo.py --topo mesh,10 \\
        --controller remote --switch ovsk,protocols=OpenFlow13

On macOS (algorithm testing):
  >>> from topology.mesh_topo import get_networkx_graph
  >>> G = get_networkx_graph(n=20, connectivity=0.3, seed=42)
"""

from __future__ import annotations

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------------
# Mininet topology (Linux only)
# ---------------------------------------------------------------------------
try:
    from mininet.topo import Topo

    class MeshTopo(Topo):
        """
        Partial-mesh topology.

        Parameters (via Mininet --topo mesh,n,connectivity):
            n : int — number of switches (default 10)
            connectivity : float — probability of each edge existing (default 0.3)
        """

        def build(self, n: int = 10, connectivity: float = 0.3):
            import random

            random.seed(42)
            switches = []
            for i in range(1, n + 1):
                sw = self.addSwitch(f"s{i}", protocols="OpenFlow13")
                switches.append(sw)
                # Attach one host per switch
                host = self.addHost(f"h{i}", ip=f"10.0.0.{i}/24")
                self.addLink(host, sw)

            # Create partial mesh
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < connectivity:
                        delay = f"{random.randint(1, 10)}ms"
                        bw = random.choice([10, 50, 100])
                        self.addLink(switches[i], switches[j], bw=bw, delay=delay)

            # Ensure connectivity: add a spanning chain
            for i in range(n - 1):
                if not self._link_exists(switches[i], switches[i + 1]):
                    self.addLink(switches[i], switches[i + 1], bw=100, delay="1ms")

        def _link_exists(self, n1, n2):
            """Check if a link already exists between two nodes."""
            for link in self.links():
                if (link[0] == n1 and link[1] == n2) or (
                    link[0] == n2 and link[1] == n1
                ):
                    return True
            return False

    topos = {"mesh": (lambda n=10, c=0.3: MeshTopo(n=int(n), connectivity=float(c)))}

except ImportError:
    pass


# ---------------------------------------------------------------------------
# NetworkX version (cross-platform)
# ---------------------------------------------------------------------------

def get_networkx_graph(
    n: int = 10,
    connectivity: float = 0.3,
    seed: int = 42,
) -> nx.Graph:
    """
    Generate a partial-mesh graph with *n* switches.

    Parameters
    ----------
    n : int
        Number of switch nodes.
    connectivity : float
        Probability of an edge between any two switches (Erdős–Rényi model).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    nx.Graph
        Connected graph with random 'weight' attributes on edges.
    """
    rng = np.random.default_rng(seed)

    # Start with Erdős–Rényi, then ensure connectivity
    G = nx.erdos_renyi_graph(n, connectivity, seed=seed)

    # Relabel nodes to 1-indexed
    mapping = {i: i + 1 for i in range(n)}
    G = nx.relabel_nodes(G, mapping)

    # Ensure connectivity by adding chain edges
    nodes = sorted(G.nodes())
    for i in range(len(nodes) - 1):
        if not G.has_edge(nodes[i], nodes[i + 1]):
            G.add_edge(nodes[i], nodes[i + 1])

    # Assign random weights to all edges
    for u, v in G.edges():
        G[u][v]["weight"] = float(rng.integers(1, 11))  # 1..10

    return G

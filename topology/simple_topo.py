"""
Simple 6-node topology for correctness verification.

Topology:
    h1 --- s1 --- s2 --- h2
            |      |
           s3 --- s4
            |
           h3

This small graph has a known shortest path structure for validating
that the controller correctly computes and installs flow rules.

Usage (Linux only):
  # Terminal 1: Start controller
  $ ryu-manager controller/physarum_controller.py --algorithm physarum_improved

  # Terminal 2: Start Mininet with this topology
  $ sudo mn --custom topology/simple_topo.py --topo simple \\
        --controller remote,ip=127.0.0.1,port=6633 \\
        --switch ovsk,protocols=OpenFlow13

  # In Mininet CLI:
  mininet> pingall
  mininet> iperf h1 h2

On macOS, this topology can be used as a NetworkX graph for algorithm testing:
  >>> from topology.simple_topo import get_networkx_graph
  >>> G = get_networkx_graph()
"""

from __future__ import annotations

import networkx as nx

# ---------------------------------------------------------------------------
# Mininet topology (Linux only)
# ---------------------------------------------------------------------------
try:
    from mininet.topo import Topo

    class SimpleTopo(Topo):
        """Simple 4-switch, 3-host topology for verification."""

        def build(self):
            # Switches
            s1 = self.addSwitch("s1", protocols="OpenFlow13")
            s2 = self.addSwitch("s2", protocols="OpenFlow13")
            s3 = self.addSwitch("s3", protocols="OpenFlow13")
            s4 = self.addSwitch("s4", protocols="OpenFlow13")

            # Hosts
            h1 = self.addHost("h1", ip="10.0.0.1/24")
            h2 = self.addHost("h2", ip="10.0.0.2/24")
            h3 = self.addHost("h3", ip="10.0.0.3/24")

            # Host-switch links
            self.addLink(h1, s1)
            self.addLink(h2, s2)
            self.addLink(h3, s3)

            # Switch-switch links (with bandwidth/delay for metrics)
            self.addLink(s1, s2, bw=100, delay="2ms")
            self.addLink(s1, s3, bw=100, delay="1ms")
            self.addLink(s2, s4, bw=100, delay="3ms")
            self.addLink(s3, s4, bw=100, delay="1ms")

    # Register for Mininet CLI
    topos = {"simple": (lambda: SimpleTopo())}

except ImportError:
    pass  # Mininet not available (macOS)


# ---------------------------------------------------------------------------
# NetworkX version (cross-platform, for algorithm testing)
# ---------------------------------------------------------------------------

def get_networkx_graph() -> nx.Graph:
    """
    Return the simple topology as a NetworkX graph.

    Nodes are switch IDs (1–4), edges have a 'weight' attribute.
    Hosts are not included — they attach to switches at the edge.
    """
    G = nx.Graph()
    G.add_edge(1, 2, weight=2.0)   # s1-s2, 2ms latency
    G.add_edge(1, 3, weight=1.0)   # s1-s3, 1ms latency
    G.add_edge(2, 4, weight=3.0)   # s2-s4, 3ms latency
    G.add_edge(3, 4, weight=1.0)   # s3-s4, 1ms latency
    return G

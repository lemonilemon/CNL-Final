"""
Ryu SDN controller with pluggable routing algorithm.

Supports three algorithms selected via command-line:
  --algorithm physarum_classic
  --algorithm physarum_improved
  --algorithm dijkstra
  --algorithm aco

And four link-cost strategies:
  --cost-strategy hop_count
  --cost-strategy latency
  --cost-strategy inv_bandwidth
  --cost-strategy composite

Usage (Linux only — requires Ryu, Mininet, Open vSwitch):
  # Terminal 1: Start the controller
  $ ryu-manager controller/physarum_controller.py \\
        --algorithm physarum_improved \\
        --cost-strategy hop_count

  # Terminal 2: Start Mininet
  $ sudo mn --custom topology/simple_topo.py --topo simple \\
        --controller remote,ip=127.0.0.1,port=6633 \\
        --switch ovsk,protocols=OpenFlow13

  # In Mininet CLI:
  mininet> pingall

On macOS, you can run the algorithm benchmarks directly without Ryu:
  $ python -m evaluation.runner --no-sdn
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Optional

import networkx as nx

import numpy as np
from dataclasses import dataclass

# Algorithms (always available)
sys.path.insert(0, ".")
from algorithms.physarum import PhysarumSolver
from algorithms.dijkstra import DijkstraSolver
from algorithms.aco import ACOSolver
from controller.base_controller import (
    LinkCostStrategy,
    TopologyManager,
    install_path_flows,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ryu imports — guarded for macOS compatibility
# ---------------------------------------------------------------------------
try:
    from ryu.base import app_manager
    from ryu.controller import ofp_event
    from ryu.controller.handler import (
        CONFIG_DISPATCHER,
        MAIN_DISPATCHER,
        DEAD_DISPATCHER,
        set_ev_cls,
    )
    from ryu.lib import hub
    from ryu.lib.packet import arp, ethernet, ipv4, packet
    from ryu.ofproto import ofproto_v1_3
    from ryu.topology import event as topo_event
    from ryu.topology.api import get_link, get_switch

    RYU_AVAILABLE = True
except ImportError:
    RYU_AVAILABLE = False


ALGORITHM_CHOICES = [
    "physarum_classic_dijkstra",
    "physarum_classic_greedy",
    "physarum_improved_dijkstra",
    "physarum_improved_greedy",
    "physarum_classic",
    "physarum_improved",
    "dijkstra",
    "aco",
]

COST_STRATEGY_CHOICES = LinkCostStrategy.ALL


@dataclass
class RouteResult:
    path: list[int]
    cost: float
    elapsed_seconds: float
    converged: bool | None = None
    iterations: int | None = None


def compute_route(
    algorithm: str,
    graph: nx.Graph,
    source: int,
    dest: int,
    init_D: np.ndarray | None = None,
    init_tau: np.ndarray | None = None,
) -> RouteResult:
    if algorithm in ("physarum_classic", "physarum_classic_dijkstra"):
        solver = PhysarumSolver(graph, source, dest, variant="classic", extraction_policy="dijkstra")
        result = solver.solve(init_D=init_D)
        return RouteResult(path=result.path, cost=result.cost, elapsed_seconds=result.elapsed_seconds,
                           converged=result.converged, iterations=result.iterations)

    elif algorithm == "physarum_classic_greedy":
        solver = PhysarumSolver(graph, source, dest, variant="classic", extraction_policy="greedy")
        result = solver.solve(init_D=init_D)
        return RouteResult(path=result.path, cost=result.cost, elapsed_seconds=result.elapsed_seconds,
                           converged=result.converged, iterations=result.iterations)

    elif algorithm in ("physarum_improved", "physarum_improved_dijkstra"):
        solver = PhysarumSolver(graph, source, dest, variant="improved", extraction_policy="dijkstra")
        result = solver.solve(init_D=init_D)
        return RouteResult(path=result.path, cost=result.cost, elapsed_seconds=result.elapsed_seconds,
                           converged=result.converged, iterations=result.iterations)

    elif algorithm == "physarum_improved_greedy":
        solver = PhysarumSolver(graph, source, dest, variant="improved", extraction_policy="greedy")
        result = solver.solve(init_D=init_D)
        return RouteResult(path=result.path, cost=result.cost, elapsed_seconds=result.elapsed_seconds,
                           converged=result.converged, iterations=result.iterations)

    elif algorithm == "dijkstra":
        solver = DijkstraSolver(graph, source, dest)
        result = solver.solve()
        return RouteResult(path=result.path, cost=result.cost, elapsed_seconds=result.elapsed_seconds)

    elif algorithm == "aco":
        solver = ACOSolver(graph, source, dest, seed=42)
        result = solver.solve(init_tau=init_tau)
        return RouteResult(path=result.path, cost=result.cost, elapsed_seconds=result.elapsed_seconds,
                           converged=result.converged, iterations=result.iterations)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

# ---------------------------------------------------------------------------
# Ryu Controller Application
# ---------------------------------------------------------------------------

if RYU_AVAILABLE:

    class PhysarumRoutingApp(app_manager.RyuApp):
        """
        Ryu application that uses a pluggable routing algorithm
        (Physarum, Dijkstra, or ACO) to compute paths and install
        OpenFlow flow rules.
        """

        OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Configuration — override via Ryu's --user-flags or env vars
            self.algorithm = kwargs.get("algorithm", "physarum_improved")
            self.cost_strategy = kwargs.get("cost_strategy", "hop_count")

            self.topo = TopologyManager(cost_strategy=self.cost_strategy)
            self.datapaths: dict[int, object] = {}

            # MAC → (dpid, port) learning table
            self.mac_to_port: dict[str, tuple[int, int]] = {}

            # Cache of installed paths: (src_mac, dst_mac) → path
            self.path_cache: dict[tuple[str, str], list[int]] = {}

            self.logger.info(
                "PhysarumRoutingApp started: algorithm=%s, cost=%s",
                self.algorithm,
                self.cost_strategy,
            )

        # ----------------------------------------------------------
        # Switch feature negotiation
        # ----------------------------------------------------------
        @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
        def switch_features_handler(self, ev):
            """Install table-miss flow entry (send to controller)."""
            dp = ev.msg.datapath
            ofproto = dp.ofproto
            parser = dp.ofproto_parser

            self.datapaths[dp.id] = dp

            # Table-miss: send to controller
            match = parser.OFPMatch()
            actions = [
                parser.OFPActionOutput(
                    ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER
                )
            ]
            inst = [
                parser.OFPInstructionActions(
                    ofproto.OFPIT_APPLY_ACTIONS, actions
                )
            ]
            mod = parser.OFPFlowMod(
                datapath=dp,
                priority=0,
                match=match,
                instructions=inst,
            )
            dp.send_msg(mod)

        # ----------------------------------------------------------
        # Topology discovery
        # ----------------------------------------------------------
        @set_ev_cls(topo_event.EventSwitchEnter)
        def switch_enter_handler(self, ev):
            self.topo.add_switch(ev.switch.dp.id)

        @set_ev_cls(topo_event.EventSwitchLeave)
        def switch_leave_handler(self, ev):
            self.topo.remove_switch(ev.switch.dp.id)
            # Invalidate path cache
            self.path_cache.clear()

        @set_ev_cls(topo_event.EventLinkAdd)
        def link_add_handler(self, ev):
            link = ev.link
            self.topo.add_link(
                link.src.dpid,
                link.dst.dpid,
                link.src.port_no,
                link.dst.port_no,
            )

        @set_ev_cls(topo_event.EventLinkDelete)
        def link_delete_handler(self, ev):
            link = ev.link
            self.topo.remove_link(link.src.dpid, link.dst.dpid)
            # Invalidate path cache — forces reroute on next packet
            self.path_cache.clear()
            self.logger.info(
                "Link down detected: %s <-> %s — cache cleared for reroute",
                link.src.dpid,
                link.dst.dpid,
            )

        # ----------------------------------------------------------
        # Packet-In handler — main routing logic
        # ----------------------------------------------------------
        @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
        def packet_in_handler(self, ev):
            """
            Handle a packet-in event:
              1. Learn the source MAC → (dpid, port).
              2. If destination MAC is known, compute a route and install flows.
              3. Otherwise flood.
            """
            msg = ev.msg
            dp = msg.datapath
            ofproto = dp.ofproto
            parser = dp.ofproto_parser
            in_port = msg.match["in_port"]

            pkt = packet.Packet(msg.data)
            eth = pkt.get_protocol(ethernet.ethernet)
            if eth is None:
                return

            src_mac = eth.src
            dst_mac = eth.dst

            # Learn source
            self.mac_to_port[src_mac] = (dp.id, in_port)

            # If we don't know the destination, flood
            if dst_mac not in self.mac_to_port:
                actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
                out = parser.OFPPacketOut(
                    datapath=dp,
                    buffer_id=msg.buffer_id,
                    in_port=in_port,
                    actions=actions,
                    data=msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None,
                )
                dp.send_msg(out)
                return

            # Destination is known
            dst_dpid, dst_port = self.mac_to_port[dst_mac]
            src_dpid = dp.id

            if src_dpid == dst_dpid:
                # Same switch — just output to destination port
                actions = [parser.OFPActionOutput(dst_port)]
                out = parser.OFPPacketOut(
                    datapath=dp,
                    buffer_id=msg.buffer_id,
                    in_port=in_port,
                    actions=actions,
                    data=msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None,
                )
                dp.send_msg(out)
                return

            # Check cache
            cache_key = (src_mac, dst_mac)
            if cache_key in self.path_cache:
                path = self.path_cache[cache_key]
            else:
                # Compute route
                graph = self.topo.get_graph()
                if src_dpid not in graph or dst_dpid not in graph:
                    self.logger.warning("Source or dest switch not in graph")
                    return

                try:
                    rr = compute_route(
                        self.algorithm, graph, src_dpid, dst_dpid
                    )
                    path, cost, elapsed = rr.path, rr.cost, rr.elapsed_seconds
                    self.logger.info(
                        "Route computed [%s]: %s → %s  path=%s cost=%.3f time=%.4fs",
                        self.algorithm, src_dpid, dst_dpid, path, cost, elapsed,
                    )
                except Exception as e:
                    self.logger.error("Route computation failed: %s", e)
                    return

                if not path:
                    self.logger.warning("No path found from %s to %s", src_dpid, dst_dpid)
                    return

                self.path_cache[cache_key] = path

            # Install flows along path
            install_path_flows(
                self.datapaths,
                path,
                self.topo,
                match_fields={"eth_dst": dst_mac},
            )

            # Forward the current packet
            next_dpid = path[1] if len(path) > 1 else dst_dpid
            out_port, _ = self.topo.get_port(src_dpid, next_dpid)
            actions = [parser.OFPActionOutput(out_port)]
            out = parser.OFPPacketOut(
                datapath=dp,
                buffer_id=msg.buffer_id,
                in_port=in_port,
                actions=actions,
                data=msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None,
            )
            dp.send_msg(out)

else:
    # Stub for macOS — allows importing the module for algorithm testing
    class PhysarumRoutingApp:  # type: ignore[no-redef]
        """Stub: Ryu not available on this platform."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PhysarumRoutingApp requires Ryu (Linux only). "
                "Use compute_route() directly for algorithm benchmarks."
            )

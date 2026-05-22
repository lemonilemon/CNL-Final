"""
Base SDN controller utilities.

Provides shared functionality used by all routing controller variants:
  - Topology discovery (switch/link event handlers)
  - NetworkX graph construction and maintenance
  - OpenFlow flow-rule installation helpers
  - Link-cost update from port statistics

This module is designed to work with the Ryu SDN framework and OpenFlow 1.3.

Usage on Linux (Ryu required):
  $ pip install ryu
  $ ryu-manager controller/physarum_controller.py

Note: Ryu only runs on Linux.  On macOS, you can import this module for
algorithm development, but the controller event loop requires Linux + OVS.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

# ---------------------------------------------------------------------------
# Ryu imports — guarded so the module can be imported on macOS for testing
# ---------------------------------------------------------------------------
try:
    from ryu.base import app_manager
    from ryu.controller import ofp_event
    from ryu.controller.handler import (
        CONFIG_DISPATCHER,
        MAIN_DISPATCHER,
        set_ev_cls,
    )
    from ryu.lib.packet import ethernet, packet
    from ryu.ofproto import ofproto_v1_3
    from ryu.topology import event as topo_event
    from ryu.topology.api import get_link, get_switch

    RYU_AVAILABLE = True
except ImportError:
    RYU_AVAILABLE = False

logger = logging.getLogger(__name__)

class LinkCostStrategy:
    """
    Computes the edge weight (L_ij) used by routing algorithms.

    Semantics:
      (a) hop_count   — every link has weight 1
      (b) latency     — weight = estimated one-way delay (ms)
      (c) inv_bw      — weight = 1 / available_bandwidth
      (d) composite   — weighted combination of the above
    """

    HOP_COUNT = "hop_count"
    LATENCY = "latency"
    INV_BANDWIDTH = "inv_bandwidth"
    COMPOSITE = "composite"

    ALL = [HOP_COUNT, LATENCY, INV_BANDWIDTH, COMPOSITE]

    @staticmethod
    def compute(
        strategy: str,
        latency_ms: float = 1.0,
        bandwidth_bps: float = 1e9,
        hop: int = 1,
        w_lat: float = 0.4,
        w_bw: float = 0.4,
        w_hop: float = 0.2,
    ) -> float:
        if strategy == LinkCostStrategy.HOP_COUNT:
            return float(hop)
        elif strategy == LinkCostStrategy.LATENCY:
            return max(latency_ms, 0.01)
        elif strategy == LinkCostStrategy.INV_BANDWIDTH:
            return 1.0 / max(bandwidth_bps, 1.0)
        elif strategy == LinkCostStrategy.COMPOSITE:
            assert(abs(w_lat + w_bw + w_hop - 1.0) < 1e-6)
            norm_lat = min(latency_ms / 100.0, 1.0)
            norm_bw = min(1.0 / (bandwidth_bps / 1e10), 1.0)
            norm_hop = float(hop)
            return w_lat * norm_lat + w_bw * norm_bw + w_hop * norm_hop
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


class TopologyManager:
    """
    Maintains a NetworkX graph representing the current network topology.

    Can be used standalone (for testing) or driven by Ryu events.
    """

    def __init__(self, cost_strategy: str = LinkCostStrategy.HOP_COUNT):
        self.net = nx.Graph()
        self.cost_strategy = cost_strategy
        # dpid → {port_no: {mac, ...}}
        self.switch_ports: dict[int, dict[int, dict[str, Any]]] = {}

    def add_switch(self, dpid: int) -> None:
        if dpid not in self.net:
            self.net.add_node(dpid)
            self.switch_ports.setdefault(dpid, {})
            logger.info("Switch added: dpid=%s", dpid)

    def remove_switch(self, dpid: int) -> None:
        if dpid in self.net:
            self.net.remove_node(dpid)
            self.switch_ports.pop(dpid, None)
            logger.info("Switch removed: dpid=%s", dpid)

    def add_link(
        self,
        src_dpid: int,
        dst_dpid: int,
        src_port: int,
        dst_port: int,
        latency_ms: float = 1.0,
        bandwidth_bps: float = 1e9,
    ) -> None:
        weight = LinkCostStrategy.compute(
            self.cost_strategy,
            latency_ms=latency_ms,
            bandwidth_bps=bandwidth_bps,
        )
        self.net.add_edge(
            src_dpid,
            dst_dpid,
            src_port=src_port,
            dst_port=dst_port,
            weight=weight,
            latency_ms=latency_ms,
            bandwidth_bps=bandwidth_bps,
        )
        logger.info(
            "Link added: %s:%s <-> %s:%s  weight=%.4f",
            src_dpid, src_port, dst_dpid, dst_port, weight,
        )

    def remove_link(self, src_dpid: int, dst_dpid: int) -> None:
        if self.net.has_edge(src_dpid, dst_dpid):
            self.net.remove_edge(src_dpid, dst_dpid)
            logger.info("Link removed: %s <-> %s", src_dpid, dst_dpid)

    def get_graph(self) -> nx.Graph:
        return self.net.copy()

    def get_port(self, src_dpid: int, dst_dpid: int) -> tuple[int, int]:
        data = self.net.edges[src_dpid, dst_dpid]
        return data["src_port"], data["dst_port"]


def install_path_flows(
    datapaths: dict,
    path: list[int],
    topo: TopologyManager,
    match_fields: dict,
    priority: int = 10,
    idle_timeout: int = 30,
) -> None:
    """
    Install OpenFlow flow rules along a computed path.

    For each switch in the path, determine the output port toward the next
    switch and install a flow entry.

    Parameters
    ----------
    datapaths : dict
        Mapping of dpid → datapath object (from Ryu).
    path : list[int]
        Ordered list of switch dpids from source to destination.
    topo : TopologyManager
        Current topology (to look up port mappings).
    match_fields : dict
        Fields for the OFPMatch (e.g., eth_dst, ipv4_dst).
    priority : int
        Flow rule priority.
    idle_timeout : int
        Seconds before the flow expires if unused.

    Note: This function requires Ryu to be installed (Linux only).
    # On Linux:
    #   ryu-manager controller/physarum_controller.py
    """
    if not RYU_AVAILABLE:
        logger.warning("Ryu not available — skipping flow installation")
        return

    for i, dpid in enumerate(path[:-1]):
        next_dpid = path[i + 1]
        src_port, _ = topo.get_port(dpid, next_dpid)

        dp = datapaths.get(dpid)
        if dp is None:
            logger.warning("No datapath for dpid=%s", dpid)
            continue

        ofproto = dp.ofproto
        parser = dp.ofproto_parser

        match = parser.OFPMatch(**match_fields)
        actions = [parser.OFPActionOutput(src_port)]
        inst = [
            parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)
        ]

        mod = parser.OFPFlowMod(
            datapath=dp,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
        )
        dp.send_msg(mod)
        logger.debug("Flow installed on dpid=%s out_port=%s", dpid, src_port)

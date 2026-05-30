"""
Mininet testbed builder (Phase 2 — Linux only).

Translates a NetworkX data-center graph (`topology/datacenter_topo.py`) into a
*structurally identical* Mininet network so the real-emulation results line up
1:1 with the Phase-1 flow-level predictions:

  - every non-host node becomes an OVS switch  's<node_id>'
  - every host node becomes a host            'h<node_id>'  with a fixed IP
  - every edge becomes a `TCLink` shaped to the edge's bandwidth_mbps / latency_ms

Bandwidths are scaled by `bw_scale` (default 0.1) because tc/HTB + a single host
CPU cannot push the nominal 10G access / 1G fabric of the abstract model; the
*ratios* between links (the only thing that drives WCMP behavior) are preserved.

The network is brought up controller-less: switches are forced to OpenFlow13 +
fail-mode=secure (so OVS installs NO default NORMAL rule — no fabric loops), and
all forwarding is programmed externally via `ovs-ofctl` (see mininet_groups.py).
Static ARP is pre-loaded so no broadcast/controller is needed for L2 discovery.

This module imports Mininet lazily; importing it on macOS/Windows (teammates)
raises a clear error instead of breaking the whole evaluation package.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from topology.datacenter_topo import TopoMeta

try:
    from mininet.net import Mininet
    from mininet.node import OVSSwitch
    from mininet.link import TCLink, TCIntf
    MININET_AVAILABLE = True
except ImportError:  # teammates on macOS / Windows
    MININET_AVAILABLE = False


if MININET_AVAILABLE:
    # Mininet shapes links with HTB, whose per-class quantum = rate / r2q
    # (r2q defaults to 10). For any link faster than ~16 Mbit/s that quantum
    # exceeds the kernel's 200000-byte limit, so `tc` emits a (harmless,
    # auto-clamped) "quantum ... is big" warning — once per link, very noisy.
    # Raising r2q on the HTB qdisc keeps the quantum in the kernel's
    # [1000, 200000] window for our 10–1000 Mbit links and silences it. We
    # inject r2q into Mininet's qdisc-add command via a thin tc() wrapper.
    _HTB_R2Q = 1000
    _orig_tcintf_tc = TCIntf.tc

    def _tc_with_r2q(self, cmd, tc="tc"):
        if "handle 5:0 htb default 1" in cmd and "r2q" not in cmd:
            cmd = cmd.replace("htb default 1", f"htb default 1 r2q {_HTB_R2Q}")
        return _orig_tcintf_tc(self, cmd, tc)

    TCIntf.tc = _tc_with_r2q


def sw_name(node: int) -> str:
    return f"s{node}"


def host_name(node: int) -> str:
    return f"h{node}"


def _host_ip(idx: int) -> str:
    """Deterministic 10.0.x.y address for the idx-th host (idx starts at 1)."""
    return f"10.0.{idx // 254}.{idx % 254 + 1}"


@dataclass
class Lab:
    """A live Mininet network plus the node<->name<->port bookkeeping."""

    net: "Mininet"
    meta: TopoMeta
    host_ip: dict[int, str] = field(default_factory=dict)        # host node -> IP
    # port_of[(switch_node, neighbor_node)] = OF output port on switch_node
    port_of: dict[tuple[int, int], int] = field(default_factory=dict)
    switches: list[int] = field(default_factory=list)            # switch node ids


def build_lab(
    G: nx.Graph,
    meta: TopoMeta,
    bw_scale: float = 0.1,
    max_bw_mbit: float = 1000.0,
) -> "Lab":
    """
    Construct (but do not start) a Mininet net mirroring G.

    Returns a `Lab`. Call `start_lab(lab)` to bring it up and fill the port map.
    """
    if not MININET_AVAILABLE:
        raise RuntimeError(
            "Mininet is unavailable. Enter the Nix shell (`nix develop`) on Linux "
            "and run under sudo; this is a Phase-2 (Linux-only) module."
        )

    net = Mininet(controller=None, switch=OVSSwitch, link=TCLink, autoSetMacs=True)
    lab = Lab(net=net, meta=meta)

    host_set = set(meta.hosts)

    # Switches: every non-host node.
    for node in sorted(n for n in G.nodes() if n not in host_set):
        net.addSwitch(sw_name(node), failMode="secure", protocols="OpenFlow13")
        lab.switches.append(node)

    # Hosts: fixed, deterministic IPs.
    for i, node in enumerate(sorted(host_set), start=1):
        ip = _host_ip(i)
        net.addHost(host_name(node), ip=f"{ip}/8")
        lab.host_ip[node] = ip

    # Links, tc-shaped. Host nodes use host_name, switch nodes sw_name.
    def nm(n: int) -> str:
        return host_name(n) if n in host_set else sw_name(n)

    for u, v, d in G.edges(data=True):
        bw = float(d.get("bandwidth_mbps", 1000.0)) * bw_scale
        bw = min(bw, max_bw_mbit)
        delay = f"{float(d.get('latency_ms', 1.0))}ms"
        net.addLink(nm(u), nm(v), bw=bw, delay=delay, use_htb=True)

    return lab


def start_lab(lab: "Lab") -> None:
    """Start the net, load static ARP, and build the switch port map."""
    net = lab.net
    net.build()
    net.start()
    net.staticArp()  # no ARP broadcast / no controller needed

    host_set = set(lab.meta.hosts)

    # Build port_of[(switch_node, neighbor_node)] from live interfaces.
    for sw_node in lab.switches:
        sw = net.get(sw_name(sw_node))
        for intf in sw.intfList():
            if intf.name == "lo" or intf.link is None:
                continue
            # The interface at the *other* end of this link.
            other = intf.link.intf2 if intf.link.intf1 == intf else intf.link.intf1
            peer = other.node
            peer_node = _node_from_name(peer.name, host_set)
            if peer_node is not None:
                lab.port_of[(sw_node, peer_node)] = sw.ports[intf]


def _node_from_name(name: str, host_set: set[int]) -> int | None:
    """Recover the integer node id from a Mininet 's<id>' / 'h<id>' name."""
    if not name or name[0] not in ("s", "h"):
        return None
    try:
        return int(name[1:])
    except ValueError:
        return None

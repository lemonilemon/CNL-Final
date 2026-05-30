"""
Controller-less OpenFlow 1.3 programming for the Phase-2 testbed (Linux only).

Takes the *same* per-node split tables the Phase-1 simulator uses
(`{node: {next_hop: fraction}}`, from `evaluation.loadbalance_sim`) and installs
them into the live OVS switches with `ovs-ofctl` — no Ryu, no controller:

  - a node with a single next-hop  -> a plain `output:<port>` flow
  - a node with multiple next-hops -> an `OFPGT_SELECT` group whose bucket
    weights are the split fractions (rounded to integers); the flow points at it

Matching is per ordered host pair (`nw_src`, `nw_dst`) so a fixed traffic matrix
gets exactly the split each method intends. Because OVS `SELECT` hashes the
5-tuple, the weighted balancing only materializes across *many* concurrent flows
(iperf `-P`), which the runner provides — a single TCP flow pins to one bucket.

Group ids are allocated per switch; each ordered pair that needs a real split at
a switch gets its own group there.
"""

from __future__ import annotations

import subprocess

from evaluation.mininet_lab import Lab, sw_name

Split = dict[int, dict[int, float]]
_OF = ["-O", "OpenFlow13"]


def _ofctl(bridge: str, *args: str) -> None:
    subprocess.run(["ovs-ofctl", *_OF, *args, bridge], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def _ofctl_bridge_first(bridge: str, verb: str, spec: str) -> None:
    # add-group / add-flow take the bridge BEFORE the spec.
    subprocess.run(["ovs-ofctl", *_OF, verb, bridge, spec], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def _int_weights(shares: dict[int, float], scale: int = 100) -> dict[int, int]:
    """Turn fractional shares into positive integer bucket weights."""
    w = {nh: max(1, round(f * scale)) for nh, f in shares.items()}
    return w


def reset_switches(lab: Lab) -> None:
    """Wipe all flows and groups so a fresh method can be installed cleanly."""
    for node in lab.switches:
        br = sw_name(node)
        _ofctl(br, "del-flows")
        _ofctl(br, "del-groups")


def install_splits(
    lab: Lab,
    ordered_splits: list[tuple[int, int, Split]],
) -> None:
    """
    Program every (src_host, dst_host, split) onto the switches.

    `ordered_splits` should contain BOTH directions of every flow (src->dst and
    dst->src) so TCP return traffic is routed too.
    """
    reset_switches(lab)
    host_set = set(lab.meta.hosts)
    next_gid: dict[int, int] = {n: 1 for n in lab.switches}

    for src_host, dst_host, split in ordered_splits:
        src_ip = lab.host_ip[src_host]
        dst_ip = lab.host_ip[dst_host]
        match = f"ip,nw_src={src_ip},nw_dst={dst_ip}"

        for node, shares in split.items():
            if node in host_set:
                continue  # hosts just send; nothing to program on an end host
            br = sw_name(node)

            # Resolve next-hops to local output ports.
            ports = {}
            for nh, frac in shares.items():
                port = lab.port_of.get((node, nh))
                if port is None:
                    raise RuntimeError(
                        f"no port on switch {node} toward {nh}; port map incomplete"
                    )
                ports[nh] = port

            if len(ports) == 1:
                (only_nh,) = ports
                _ofctl_bridge_first(
                    br, "add-flow",
                    f"priority=100,{match},actions=output:{ports[only_nh]}",
                )
            else:
                gid = next_gid[node]
                next_gid[node] += 1
                weights = _int_weights(shares)
                buckets = ",".join(
                    f"bucket=weight={weights[nh]},output:{ports[nh]}"
                    for nh in ports
                )
                _ofctl_bridge_first(
                    br, "add-group", f"group_id={gid},type=select,{buckets}",
                )
                _ofctl_bridge_first(
                    br, "add-flow",
                    f"priority=100,{match},actions=group:{gid}",
                )

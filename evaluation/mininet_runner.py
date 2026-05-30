"""
Phase-2 Mininet + iperf runner (Linux only, needs root).

Validates the Phase-1 flow-level predictions on a real OVS data plane:

  1. build a Mininet net mirroring a data-center graph (tc-shaped links),
  2. for each method, compute the SAME per-node split tables as Phase 1 and
     install them controller-less as OF1.3 SELECT groups (mininet_groups),
  3. drive concurrent iperf elephant flows across cross-block host pairs:
       - UDP at a swept offered load  -> aggregate throughput + packet loss
       - TCP saturating               -> realistic aggregate throughput
  4. write results/phase2_udp.csv and results/phase2_tcp.csv for plotting against
     the Phase-1 curves.

Run (inside `nix develop`, with project-local OVS up via `sudo -E ovs-start`):

    uv run sudo -E python -m evaluation.mininet_runner \
        --topologies leaf_spine,fat_tree --duration 10 --parallel 8

NOTE: bandwidths are scaled by --bw-scale (default 0.1); the swept demands are
scaled identically so the comparison to Phase 1 stays apples-to-apples.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from pathlib import Path

import numpy as np

from topology.datacenter_topo import (
    get_fat_tree_graph,
    get_leaf_spine_graph,
    cross_block_pairs,
)
from evaluation.loadbalance_sim import METHODS
from evaluation.mininet_lab import build_lab, start_lab, host_name, MININET_AVAILABLE
from evaluation.mininet_groups import install_splits

METHOD_ORDER = ["dijkstra_single", "ecmp", "physarum", "aco"]
DEFAULT_DEMANDS = [25.0, 50.0, 100.0, 200.0, 400.0]  # per-flow Mbps (pre-scale)
N_FLOWS = 8

# iperf2 report line parsers ------------------------------------------------
# Per-stream UDP server line (has the jitter "<x> ms" field, then Lost/Total):
#   [  2] 0.0-3.0 sec  2.39 MBytes  10.0 Mbits/sec   0.002 ms 1/1703 (0.059%)
# The [SUM] line lacks the "ms" field, so this regex matches ONLY per-stream
# lines — summing them gives the true aggregate with no double counting and
# works whether iperf ran with -P 1 or -P N.
_UDP_RE = re.compile(
    r"([\d.]+)\s+([GMK])bits/sec\s+[\d.]+\s+ms\s+(\d+)/\s*(\d+)\s+\([-\d.naN]+%\)"
)
_BW_RE = re.compile(r"([\d.]+)\s+([GMK]?)bits/sec")
_UNIT = {"G": 1000.0, "M": 1.0, "K": 1e-3, "": 1e-6}


def _topo(name: str):
    if name == "fat_tree":
        return get_fat_tree_graph(k=4)
    if name == "leaf_spine":
        return get_leaf_spine_graph()
    raise ValueError(f"unknown topology {name!r}")


def _both_direction_splits(G, flows, method, physarum_kwargs):
    """Per-method split tables for src->dst AND dst->src of every flow."""
    builder = METHODS[method]
    pk = physarum_kwargs or {}
    out = []
    for s, d in flows:
        for a, b in ((s, d), (d, s)):
            split = builder(G, a, b, **pk) if method == "physarum" else builder(G, a, b)
            out.append((a, b, split))
    return out


def _parse_udp_server(text: str) -> tuple[float, int, int]:
    """
    Sum every per-stream UDP report into (throughput_mbps, lost, total). With
    -P N the server prints N per-stream lines; summing them is the aggregate.
    """
    bw = 0.0
    lost = total = 0
    for m in _UDP_RE.finditer(text):
        bw += float(m.group(1)) * _UNIT[m.group(2)]
        lost += int(m.group(3))
        total += int(m.group(4))
    return bw, lost, total


def _parse_bw(text: str) -> float:
    """Return throughput in Mbps from the LAST bandwidth line (TCP client/SUM)."""
    last = None
    for m in _BW_RE.finditer(text):
        last = m
    if last is None:
        return 0.0
    return float(last.group(1)) * _UNIT[last.group(2)]


def _run_udp(lab, flows, per_flow_mbps: float, duration: int, parallel: int):
    """Concurrent one-way UDP flows; aggregate throughput + loss from servers."""
    net = lab.net
    base = 5200
    servers, clients = [], []
    per_stream = max(per_flow_mbps / parallel, 0.01)

    # Start one UDP server per flow on a unique port (stdout captured via PIPE).
    for i, (s, d) in enumerate(flows):
        port = base + i
        dst = net.get(host_name(d))
        servers.append(dst.popen("iperf2", "-s", "-u", "-p", str(port)))
    time.sleep(0.5)

    for i, (s, d) in enumerate(flows):
        port = base + i
        src = net.get(host_name(s))
        dst_ip = lab.host_ip[d]
        clients.append(src.popen(
            "iperf2", "-c", dst_ip, "-u", "-p", str(port),
            "-b", f"{per_stream}m", "-t", str(duration), "-P", str(parallel)))

    for c in clients:
        c.wait()
    # The client's end-of-test FIN makes the server emit its report; read it.
    time.sleep(0.5)

    agg_bw, tot_lost, tot_pkts = 0.0, 0, 0
    for srv in servers:
        srv.terminate()
        try:
            out, _ = srv.communicate(timeout=5)
        except Exception:
            out = b""
        text = out.decode() if isinstance(out, bytes) else (out or "")
        bw, lost, total = _parse_udp_server(text)
        agg_bw += bw
        tot_lost += lost
        tot_pkts += total
    loss = (tot_lost / tot_pkts) if tot_pkts else 0.0
    return agg_bw, loss


def _run_tcp(lab, flows, duration: int, parallel: int) -> float:
    """Concurrent saturating TCP flows; aggregate client-side throughput."""
    net = lab.net
    base = 5400
    servers, procs = [], []
    for i, (s, d) in enumerate(flows):
        port = base + i
        dst = net.get(host_name(d))
        servers.append(dst.popen(f"iperf2 -s -p {port}", shell=True))
    time.sleep(0.5)

    for i, (s, d) in enumerate(flows):
        port = base + i
        src = net.get(host_name(s))
        dst_ip = lab.host_ip[d]
        procs.append(src.popen(
            f"iperf2 -c {dst_ip} -p {port} -t {duration} -P {parallel}",
            shell=True))

    agg = 0.0
    for p in procs:
        out, _ = p.communicate()
        agg += _parse_bw(out.decode() if isinstance(out, bytes) else (out or ""))
    for srv in servers:
        srv.terminate()
    return agg


def run(topologies, demands, duration, parallel, bw_scale, trials, outdir, seed):
    if not MININET_AVAILABLE:
        raise SystemExit(
            "Mininet unavailable — run on Linux inside `nix develop`, under sudo.")
    if os.geteuid() != 0:
        raise SystemExit("Phase 2 needs root. Re-run with: sudo -E python -m ...")

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    udp_rows, tcp_rows = [], []
    rng = np.random.default_rng(seed)
    physarum_kwargs = {"mu": 0.8, "max_iter": 500, "patience": 20}

    for topo in topologies:
        G, meta = _topo(topo)
        flows = cross_block_pairs(meta, N_FLOWS, rng)

        for trial in range(trials):
            lab = build_lab(G, meta, bw_scale=bw_scale)
            start_lab(lab)
            try:
                for method in METHOD_ORDER:
                    splits = _both_direction_splits(G, flows, method, physarum_kwargs)
                    install_splits(lab, splits)

                    for demand in demands:
                        scaled = demand * bw_scale
                        bw, loss = _run_udp(lab, flows, scaled, duration, parallel)
                        # report throughput back at full (unscaled) units
                        udp_rows.append({
                            "topology": topo, "method": method, "trial": trial,
                            "offered_per_flow_mbps": demand,
                            "offered_total_mbps": demand * len(flows),
                            "throughput_mbps": bw / bw_scale,
                            "packet_loss": loss,
                        })
                        print(f"[UDP] {topo:10s} {method:16s} d={demand:6.1f} "
                              f"-> thr={bw/bw_scale:8.1f} loss={loss:6.2%}", flush=True)

                    tcp_bw = _run_tcp(lab, flows, duration, parallel)
                    tcp_rows.append({
                        "topology": topo, "method": method, "trial": trial,
                        "throughput_mbps": tcp_bw / bw_scale,
                    })
                    print(f"[TCP] {topo:10s} {method:16s} "
                          f"-> thr={tcp_bw/bw_scale:8.1f}")
            finally:
                lab.net.stop()

    _write_csv(out / "phase2_udp.csv", udp_rows,
               ["topology", "method", "trial", "offered_per_flow_mbps",
                "offered_total_mbps", "throughput_mbps", "packet_loss"])
    _write_csv(out / "phase2_tcp.csv", tcp_rows,
               ["topology", "method", "trial", "throughput_mbps"])
    print(f"\nWrote {out/'phase2_udp.csv'} and {out/'phase2_tcp.csv'}")


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-2 Mininet+iperf load-balancing test")
    ap.add_argument("--topologies", default="leaf_spine,fat_tree")
    ap.add_argument("--demands", default=None,
                    help="comma list of per-flow Mbps (pre-scale); default sweep")
    ap.add_argument("--duration", type=int, default=10)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--bw-scale", type=float, default=0.1)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--output", default="results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]
    demands = ([float(x) for x in args.demands.split(",")]
               if args.demands else DEFAULT_DEMANDS)
    run(topologies, demands, args.duration, args.parallel, args.bw_scale,
        args.trials, args.output, args.seed)


if __name__ == "__main__":
    main()

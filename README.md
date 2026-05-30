
# SDN Routing with Physarum Algorithm

**Computer Networking Laboratory — Final Project (Group 6)**

Bio-inspired routing for Software-Defined Networks using the *Physarum polycephalum* (slime mold) algorithm, benchmarked against Dijkstra and Ant Colony Optimization.

## Project Structure

```
CNL-Final/
├── algorithms/              # Core routing algorithms
│   ├── physarum.py           #   Physarum solver (+ extract_multipath_split)
│   ├── dijkstra.py           #   Dijkstra baseline
│   └── aco.py                #   Ant Colony Optimization
├── controller/              # Ryu SDN controller (documented design; Phase 2 is controller-less)
│   ├── base_controller.py    #   Topology discovery & flow helpers
│   └── physarum_controller.py#   Ryu app with pluggable algorithms
├── topology/                # Network topologies
│   ├── simple_topo.py        #   Small graph for verification
│   ├── mesh_topo.py          #   Parameterized mesh for scalability
│   ├── dynamic_topo.py       #   Link-failure injection for adaptiveness
│   └── datacenter_topo.py    #   Fat-Tree + heterogeneous Leaf-Spine (multipath)
├── evaluation/              # Benchmarking framework
│   ├── metrics.py            #   Metric collection
│   ├── runner.py             #   Single-path benchmark orchestrator
│   ├── loadbalance_sim.py    #   Phase 1: flow-level max-min throughput/loss sim
│   ├── loadbalance_runner.py #   Phase 1: sweep -> results/loadbalance.csv
│   ├── mininet_lab.py        #   Phase 2: build Mininet net from a graph (tc links)
│   ├── mininet_groups.py     #   Phase 2: install splits as OF1.3 SELECT groups
│   ├── mininet_runner.py     #   Phase 2: testbed driver (iperf) -> results/phase2_*.csv
│   └── plot.py               #   Visualization (--loadbalance, --phase2)
├── tests/                   # Unit tests
├── Makefile                 # Task runner for all evaluation phases
└── requirements.txt
```

## Installation

### Python environment — all phases, any OS

Phase 0/1 needs only Python, managed with [`uv`](https://docs.astral.sh/uv/).
Install `uv`, then:

```bash
make deps          # create .venv and install requirements.txt
```

### Phase 2 system tools — Linux only

The real testbed additionally needs **Mininet**, **Open vSwitch**, and **iperf**
(plus `iproute2`/`tc` for link shaping), and must run with `sudo` (Mininet sets up
network namespaces and a kernel datapath). Install them with your package manager:

```bash
# Debian / Ubuntu
sudo apt-get install mininet openvswitch-switch iperf iperf3 iproute2

# Fedora
sudo dnf install mininet openvswitch iperf iperf3 iproute
```

Alternatively, with Nix (flakes enabled) the bundled `flake.nix` provides the whole
toolchain without installing anything system-wide — enter the shell, then run the
Phase-2 targets from inside it:

```bash
nix develop        # shell with mininet/OVS/iperf on PATH
make phase2        # ...then run as usual
```

> `make phase2` expects the Phase-2 tools on your `PATH`. Install them system-wide
> (apt/dnf), or enter a shell that provides them (`nix develop`) before running it.

## Quick Start

Everything runs through the `Makefile`. Run `make help` for the full list.

### Phase 0 / 1 — algorithm benchmark + flow-level multipath (any OS)

```bash
make deps          # uv venv + install requirements.txt
make test          # unit tests
make benchmark     # single-path benchmark -> results/results.csv + figures
make loadbalance   # multipath flow-level sim -> results/loadbalance.csv + lb_*.png
make all           # test + benchmark + loadbalance
```

### Phase 2 — real testbed (Linux only)

Validates the multipath results on a real data plane with **Mininet + Open vSwitch + iperf**.
Requires Linux and `sudo` (Mininet creates network namespaces and a kernel datapath), with the
tools on your `PATH` (see [Installation](#installation)); the Python side still runs on `uv`.

```bash
make phase2        # OVS up -> build topo -> install OF1.3 groups -> iperf -> OVS down -> plots
make phase2-plots  # regenerate Phase-2 plots from existing CSVs (uv only)
make ovs-stop      # force-stop the project-local OVS if a run was interrupted
```

Knobs are overridable, e.g. `make phase2 DURATION=10 DEMANDS=100,200,400` or
`make benchmark TRIALS=3 NODE_COUNTS=6,10,20`.

Phase 2 programs OpenFlow 1.3 `SELECT` group tables directly with `ovs-ofctl`
(**controller-less** — no Ryu), reusing the exact per-node split tables computed in Phase 1.

## Link-Cost Strategies

The system supports four link-weight semantics:
- **hop_count** — uniform weight (1 per link)
- **latency** — edge weight = measured latency (ms)
- **inv_bandwidth** — edge weight = 1 / available bandwidth
- **composite** — weighted combination of all three

## Evaluation Metrics

### Single-path benchmark (`make benchmark`)

Compares Physarum / Dijkstra / ACO as shortest-path solvers:

1. **Execution time** — algorithm computation time
2. **Transmission delay** — end-to-end latency along the path
3. **Throughput** — bottleneck bandwidth on the path
4. **Adaptiveness** — re-routing time after link failure
5. **Optimality** — path cost relative to the Dijkstra optimum
6. **Route stability** — number of route flaps under churn (Physarum's one structural win)
7. **Iterations / warm-start speedup** — convergence cost, and the gain from re-seeding the solver

### Multipath load-balancing (`make loadbalance`, `make phase2`)

Reframes the comparison from *"who finds the shortest path"* to *"who uses the network's
aggregate capacity under load."* Physarum exports its steady-state flux as a capacity-proportional
traffic split (WCMP), compared against single-path Dijkstra, equal-split ECMP, and pheromone-weighted
ACO on data-center fabrics:

1. **Aggregate throughput** — total Mbps carried across all concurrent flows
2. **Packet loss** — fraction of offered demand dropped under congestion
3. **Link-utilization imbalance** — stddev of per-link utilization (quantifies the single-path hotspot)
4. **Max link utilization** — worst-case bottleneck loading

Phase 1 predicts these with a flow-level max-min fair-share model; Phase 2 validates them on a real
Mininet + OVS + iperf data plane. On the heterogeneous Leaf-Spine fabric, Physarum's capacity-weighted
split holds ~0% loss far longer than equal-split ECMP/ACO and roughly doubles aggregate throughput
over single-path Dijkstra; on the symmetric Fat-Tree it matches ECMP (no regression).

## References

- Zhang et al. (2014). "An Improved Physarum polycephalum Algorithm for the Shortest Path Problem." *The Scientific World Journal*.
- Tero et al. (2007). "A mathematical model for adaptive transport network in path finding by true slime mold."
- Dorigo & Stützle (2004). "Ant Colony Optimization." MIT Press.

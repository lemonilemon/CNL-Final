# Multipath Load Balancing with Physarum — Implementation Plan

**Group 6 — CNL Final, extension to the routing benchmark**

## 1. Motivation

The existing benchmark compares Physarum / Dijkstra / ACO as **single-path shortest-path
solvers**. On that contest Dijkstra wins decisively (speed, optimality); Physarum only wins on
route-flap stability. This is structural: there is no single-path metric a slime-mold model can
beat Dijkstra on.

Physarum's `D_ij` (conductivity) is natively a **flow distribution** — the model solves
Kirchhoff's current law over a network of conductances. Collapsing it to one path via
`_extract_path` throws away the one thing Dijkstra cannot do: spread a flow across multiple paths
in proportion to capacity. This extension reframes the comparison from *"who finds the shortest
path"* to *"who uses the network's aggregate capacity under load"* — a contest single-path routing
cannot win.

## 2. The non-negotiable design decision: μ < 1

The Physarum feedback law is `dD/dt = f(Q) − decay·D` with `f(Q) = |Q|^μ`. The current code
hard-codes **μ = 1** and runs to convergence — the **shortest-path-selecting regime**:

| Regime | Behavior at convergence |
|---|---|
| μ ≥ 1, **unequal**-cost paths | All flow collapses onto the single cheapest tube; other tubes decay `D→0`. **Identical to Dijkstra — no split.** |
| μ ≥ 1, exactly **equal**-cost paths | Symmetric even split *can* survive, but is numerically fragile. |
| **μ < 1 (sublinear)** | **Multiple tubes survive** — reproduces Tero et al.'s redundant-network results (the Tokyo-rail experiment). This is the regime we need. |

Because the showcase topology (heterogeneous Leaf-Spine) has **unequal**-cost paths, μ = 1 would
produce a degenerate single-path "multipath" table — nothing to load-balance. We therefore add a
**multipath solver mode with μ < 1**, alongside (not replacing) the existing μ = 1 single-path mode
used by the current metrics.

## 3. The forwarding object: flux Q, not raw D

We do **not** export raw `D_ij` directly. The correct Physarum-native split is the **steady-state
flux** `Q_ij = D_ij·(p_i − p_j)/L_ij` (the current on each edge). At each node, outgoing flow is
split across neighbors **proportional to the positive outgoing flux** (current flowing toward the
sink). This:

- conserves flow (Kirchhoff) — fractions at each node sum to 1,
- is loop-free in the source→sink direction (a flow DAG),
- maps directly onto OpenFlow `OFPGT_SELECT` group-table bucket weights in Phase 2.

Output of the solver's multipath mode = a **per-node split table**: `{node: {next_hop: fraction}}`.

## 4. Methods compared

| Method | Split rule |
|---|---|
| `dijkstra_single` | 100% onto the single shortest path (the status quo / hotspot baseline). |
| `ecmp` | **Plain equal-split ECMP** — enumerate equal-cost shortest paths, split evenly. The real baseline to beat. |
| `physarum_multipath` | Flux-proportional split from the μ<1 solver (weighted / WCMP-style). |
| `aco` | Pheromone-weighted split: hop-count next-hop set (as ECMP) weighted by learned pheromone τ (AntNet-style). ACO has no potential to orient flow, so we borrow the shortest-path DAG for direction. |

Including plain ECMP is essential: comparing only against single-path Dijkstra would prove the
trivial "multipath beats single-path," not that the *slime-mold proportional split* adds anything.

## 5. Topologies (both)

New file `topology/datacenter_topo.py`:

- **Symmetric Fat-Tree** (k-ary) — equal-cost paths. Expectation: Physarum ≈ plain ECMP (parity on
  the easy case). Shows we don't *lose* where ECMP already wins.
- **Heterogeneous Leaf-Spine** — leaves × spines with **varied link bandwidths**, `inv_bandwidth`
  cost. Expectation: Physarum's weighted split **beats** equal-hash ECMP (which over-uses thin
  links and bottlenecks). This is the headline result.

Each gets a cross-platform NetworkX builder now, and Mininet `Topo` classes in Phase 2.

## 6. Flow-level throughput / loss simulator (Phase 1)

New `evaluation/loadbalance_sim.py`:

1. Input: a graph with link capacities, a **traffic matrix** (concurrent host-pair demands, e.g.
   all-to-all elephant flows), and a method's per-node split tables.
2. Push each flow's demand through its split DAG → accumulate **per-link load**.
3. Resolve contention with **iterative max-min fair-share** allocation (standard, defensible):
   shared bottleneck links scale competing flows down proportionally.
4. Metrics:
   - **Aggregate throughput** (Σ achieved Mbps)
   - **Packet loss rate** = 1 − achieved/offered
   - **Max link utilization** and **load imbalance** (stddev of utilization) — quantifies the
     Dijkstra core-hotspot directly.

## 7. Runner & plots (Phase 1)

- `evaluation/loadbalance_runner.py` (separate from `runner.py` — existing metrics untouched):
  sweep {topology} × {offered-load level} × {method} × trials → `results/loadbalance.csv`.
- New plot functions: throughput-vs-offered-load, loss-vs-offered-load, link-utilization imbalance.

## 8. Phase 2 — real Mininet + iperf (after Phase 1 validates & tunes μ)

- Extend `controller/physarum_controller.py`: install **OF1.3 `OFPGT_SELECT` group tables**, with
  bucket weights = the per-node split fractions; flows point at the group.
  - Note: OVS SELECT hashes the 5-tuple → balancing is **per-flow**, so we need *many* concurrent
    flows (all-to-all iperf) for the weights to show; a single TCP flow stays on one bucket.
- Mininet Fat-Tree / Leaf-Spine topos with link `bw`/`delay`.
- iperf harness: launch concurrent elephant flows across host pairs; collect throughput + loss.
- Compare real numbers against the Phase-1 simulation predictions.

## 9. File-change summary

| File | Change | Phase |
|---|---|---|
| `algorithms/physarum.py` | Add μ<1 multipath mode + `extract_multipath_split()` (flux-based). Single-path mode untouched. | 1 |
| `topology/datacenter_topo.py` | **New.** NetworkX Fat-Tree + heterogeneous Leaf-Spine builders. | 1 |
| `evaluation/loadbalance_sim.py` | **New.** Traffic matrix + max-min fair-share throughput/loss simulator. | 1 |
| `evaluation/loadbalance_runner.py` | **New.** Sweep + `results/loadbalance.csv`. | 1 |
| `evaluation/plot.py` | Add load-balancing plots (throughput/loss/imbalance). | 1 |
| `topology/datacenter_topo.py` | Add Mininet `Topo` classes. | 2 |
| `controller/physarum_controller.py` | Add `OFPGT_SELECT` group-table install from split tables. | 2 |

## 10. Expected narrative for the report

- **Fat-Tree (symmetric):** Physarum ≈ ECMP — both balance, both beat single-path Dijkstra. Shows
  no regression on the easy case.
- **Leaf-Spine (heterogeneous):** equal-hash ECMP over-loads thin links → loss; Physarum's
  capacity-proportional split keeps utilization balanced → higher aggregate throughput, lower loss.
  Dijkstra bottlenecks a single core link the whole time.
- **Honest caveat:** computing the split is still ~2 orders slower than Dijkstra (carried over from
  the existing results); the win is in *capacity utilization under load*, not compute time.

## 11. Phase 1 — results (flow-level simulation)

Run: `python -m evaluation.loadbalance_runner --trials 5` → `results/loadbalance.csv`;
plots: `python -m evaluation.plot --loadbalance` → `results/figures/lb_*.png`.

**Heterogeneous Leaf-Spine (the headline):** Physarum tracks the no-loss ideal far longer and
saturates ~2700 Mbps vs ECMP ~1400 and single-path Dijkstra ~870; its packet loss is dramatically
lower across the whole congestion range, with the lowest link-utilization imbalance. ACO lands
slightly above ECMP but well below Physarum.

**Symmetric Fat-Tree (the control):** Physarum ≡ ECMP *exactly* (identical throughput and
imbalance) — no regression on the easy case — and both beat single-path Dijkstra once the fabric
saturates.

**Two findings worth stating honestly in the report:**
- *Why ACO ≈ ECMP, not ≈ Physarum:* ACO deposits pheromone `q/cost` over whole (short, equal-hop)
  paths, so on this fabric its pheromone barely differentiates link capacity → near-uniform split.
  Physarum's flux is *physically* capacity-proportional (current divides by conductance), which is
  why it alone achieves true WCMP behavior.
- *Operating point matters:* at **full convergence**, Physarum (any μ) collapses toward the
  minimum-cost equal-cost paths and drops thin links — the shortest-path regime. The
  capacity-*proportional* multipath that wins here is read at the solver's natural **adaptive
  operating point** (bounded iterations, μ < 1), not the converged fixed point.

## 12. Phase 2 — real testbed (Mininet + OVS + iperf), DONE

**Built and run on Linux (NixOS).** The original plan said to extend
`controller/physarum_controller.py` with a Ryu app; we instead went **controller-less** (Ryu is
unmaintained and breaks on modern Python). The Phase-1 split tables are installed directly as
OpenFlow 1.3 group tables with `ovs-ofctl`, so Phases 1 and 2 share the *exact same* split logic.

**Toolchain — `flake.nix`** (Linux-gated devShell): `mininet`, `openvswitch`, `iperf2`, `iperf3`,
`iproute2`. Python stays on **uv**; the flake just adds the Nix-built (pure-Python) Mininet module
to `PYTHONPATH` so the uv venv interpreter has numpy/scipy/networkx **and** mininet in one process.
Helpers: `ovs-start` / `sudo ovs-stop` (project-local OVS under `./.ovs`, no system config touched)
and `phase2-run` (one command: resolves the uv venv python, re-injects `PATH`/`PYTHONPATH`/`OVS_*`
through sudo, runs the harness as root).

**New files (Phase 2):**

| File | Role |
|---|---|
| `evaluation/mininet_lab.py` | Build a Mininet net structurally identical to the NetworkX graph; tc-shaped links (`bw_scale=0.1`), fail-secure OVS switches, static ARP, switch port map. |
| `evaluation/mininet_groups.py` | Install Phase-1 splits as OF1.3 `SELECT` groups (multi-next-hop) / direct `output` (single), per ordered host pair, via `ovs-ofctl`. |
| `evaluation/mininet_runner.py` | Build → install → concurrent iperf (UDP loss sweep + saturating TCP) → `results/phase2_udp.csv`, `phase2_tcp.csv`. |
| `evaluation/plot.py` | `--phase2`: measured-vs-predicted overlay + TCP bars (`phase2_*_*.png`). |

Run: `nix develop` → `sudo -E ovs-start` → `phase2-run --topologies leaf_spine,fat_tree
--duration 5 --demands 50,100,200,400,800` → `python -m evaluation.plot --phase2` → `sudo ovs-stop`.

**Results — the testbed confirms the Phase-1 predictions.** Heterogeneous Leaf-Spine, UDP, at
per-flow demand 200 (1600 Mbps offered):

| Method | Throughput (Mbps) | Loss |
|---|---|---|
| **Physarum (WCMP)** | **1600** | **0.08 %** |
| ACO | 1310 | 11.6 % |
| ECMP (equal-split) | 1263 | 13.9 % |
| Dijkstra (single-path) | 560 | 55.7 % |

Physarum holds ~0 % loss through demand 400 while ECMP/ACO already lose ~14–19 % at demand 200;
single-path Dijkstra saturates ~660 Mbps with loss rising to 88 %. On the **symmetric Fat-Tree**
control, Physarum ≈ ECMP (at saturation 5238 vs 5388 Mbps, 16 % vs 14 % loss — parity, no
regression), both beating single-path Dijkstra once the core saturates. The measured curves track
the flow-level predictions; Physarum does *slightly better* than predicted because the max-min
fluid model is more pessimistic than the real data plane.

**Two real bugs found and fixed during bring-up (worth a sentence in the report):**
- The `iperf3` package ships an `iperf` → `iperf3` symlink that can shadow iperf2 on `PATH`;
  iperf3's server rejects UDP `-u`, silently zeroing loss measurements. Pinned to `iperf2`.
- `sudo` strips `PYTHONPATH`/`PATH`/`OVS_*`; `phase2-run` re-injects them explicitly via `env`.

**Caveat:** link bandwidths are scaled ×0.1 (tc/HTB + one host CPU can't push the nominal 10G/1G);
the demand sweep is scaled identically, so ratios — the only thing WCMP cares about — are preserved.

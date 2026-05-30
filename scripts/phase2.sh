#!/usr/bin/env bash
# Phase-2 testbed orchestration.
#
# Run INSIDE the Nix dev shell so the tools and PYTHONPATH (Mininet module) are
# present — normally via `make phase2`, or directly:
#   nix develop --command bash scripts/phase2.sh --topologies leaf_spine,fat_tree
#
# Brings up a project-local Open vSwitch under ./.ovs (no system service, no
# changes to configuration.nix), runs the Mininet + iperf harness as root, then
# tears OVS and Mininet down. All extra args are forwarded to mininet_runner.
set -euo pipefail

ovsdir="${OVS_PROJECT_DIR:-$PWD/.ovs}"
ovs_root="$(dirname "$(dirname "$(readlink -f "$(command -v ovsdb-tool)")")")"
schema="$ovs_root/share/openvswitch/vswitch.ovsschema"
venvpy="$(uv run python -c 'import sys; print(sys.executable)')"

# OVS datapath + Mininet namespaces need root; sudo strips the environment, so
# re-inject PATH (Nix tools), PYTHONPATH (Mininet module) and the OVS dirs.
exec sudo env \
  PATH="$PATH" \
  PYTHONPATH="${PYTHONPATH:-}" \
  OVS_RUNDIR="$ovsdir/run" OVS_DBDIR="$ovsdir/db" OVS_LOGDIR="$ovsdir/log" \
  OVS_SCHEMA="$schema" VENVPY="$venvpy" \
  bash -seuo pipefail -- "$@" <<'ROOT'
mkdir -p "$OVS_RUNDIR" "$OVS_DBDIR" "$OVS_LOGDIR"

stop_ovs() {
  for p in ovs-vswitchd ovsdb-server; do
    if [ -f "$OVS_RUNDIR/$p.pid" ]; then
      kill "$(cat "$OVS_RUNDIR/$p.pid")" 2>/dev/null || true
      rm -f "$OVS_RUNDIR/$p.pid"
    fi
  done
}
cleanup() { stop_ovs; mn -c >/dev/null 2>&1 || true; }
trap cleanup EXIT

[ -f "$OVS_DBDIR/conf.db" ] || ovsdb-tool create "$OVS_DBDIR/conf.db" "$OVS_SCHEMA"
ovsdb-server "$OVS_DBDIR/conf.db" --remote=punix:"$OVS_RUNDIR/db.sock" \
  --pidfile="$OVS_RUNDIR/ovsdb-server.pid" \
  --log-file="$OVS_LOGDIR/ovsdb-server.log" --detach --no-chdir
ovs-vsctl --db=unix:"$OVS_RUNDIR/db.sock" --no-wait init
ovs-vswitchd unix:"$OVS_RUNDIR/db.sock" \
  --pidfile="$OVS_RUNDIR/ovs-vswitchd.pid" \
  --log-file="$OVS_LOGDIR/ovs-vswitchd.log" --detach --no-chdir

"$VENVPY" -m evaluation.mininet_runner "$@"
ROOT

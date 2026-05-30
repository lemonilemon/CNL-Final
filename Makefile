# CNL Final — Group 6: Physarum multipath routing evaluation
# ---------------------------------------------------------------------------
# Python packages come from uv (cross-platform). So:
#
#   * Phase 0/1 targets (test, benchmark, loadbalance, plots) need ONLY uv —
#     they run as-is on macOS / Windows / Linux.
#   * Phase 2 targets (phase2*) additionally need Linux + sudo and the system
#     tools (Mininet / OVS / iperf) on your PATH. Install them however you like
#     (see README: Installation).
#
# Run `make` or `make help` for the target list.
# ---------------------------------------------------------------------------

UV       ?= uv
OUT      ?= results

# Phase 0 (routing benchmark) knobs
TRIALS       ?= 5
NODE_COUNTS  ?= 6,10,20

# Phase 2 (testbed) knobs
TOPOS    ?= leaf_spine,fat_tree
DURATION ?= 5
DEMANDS  ?= 50,100,200,400,800

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@echo "CNL Final — evaluation targets"
	@echo "  Phase 0/1 (uv only, any OS):  deps test benchmark loadbalance"
	@echo "  Phase 2  (Linux+sudo):        phase2 phase2-plots ovs-stop"
	@echo
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup (uv)
# ---------------------------------------------------------------------------

.PHONY: deps
deps: ## Create .venv and install Python deps via uv (cross-platform)
	$(UV) venv
	$(UV) pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Phase 0/1 — pure Python, runs anywhere with uv
# ---------------------------------------------------------------------------

.PHONY: test
test: ## Run the unit test suite
	$(UV) run python -m pytest tests/ -q

.PHONY: benchmark
benchmark: ## Phase-0 routing benchmark -> results/results.csv + figures
	$(UV) run python -m evaluation.runner --trials $(TRIALS) --node-counts $(NODE_COUNTS)
	$(UV) run python -m evaluation.plot --input $(OUT)/results.csv

.PHONY: loadbalance
loadbalance: ## Phase-1 flow-level multipath sim -> results/loadbalance.csv + lb_*.png
	$(UV) run python -m evaluation.loadbalance_runner --trials $(TRIALS) --output $(OUT)
	$(UV) run python -m evaluation.plot --loadbalance

# ---------------------------------------------------------------------------
# Phase 2 — real Mininet+OVS+iperf testbed (Linux + sudo)
# Requires the system tools (mn, ovs-vsctl, iperf2, tc) on your PATH.
# ---------------------------------------------------------------------------

.PHONY: tools-check
tools-check: ## Verify the Phase-2 system tools are on PATH
	@command -v mn >/dev/null 2>&1 && command -v ovs-vsctl >/dev/null 2>&1 || { \
		echo "ERROR: Mininet/OVS not found on PATH. Install them (see README:"; \
		echo "Installation) or enter a shell that provides them, then retry."; \
		exit 1; }

.PHONY: phase2
phase2: tools-check ## Run the testbed (build->install->iperf) then plot. Brings OVS up/down.
	bash scripts/phase2.sh \
		--topologies $(TOPOS) --duration $(DURATION) --demands $(DEMANDS) --output $(OUT)
	$(MAKE) phase2-plots

.PHONY: phase2-plots
phase2-plots: ## Regenerate Phase-2 plots from existing CSVs (uv only)
	$(UV) run python -m evaluation.plot --phase2

.PHONY: ovs-stop
ovs-stop: ## Recovery: kill stray project OVS daemons and clean Mininet state
	-sudo pkill -x ovs-vswitchd 2>/dev/null
	-sudo pkill -x ovsdb-server 2>/dev/null
	-sudo mn -c >/dev/null 2>&1

# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

.PHONY: all
all: test benchmark loadbalance ## Phase 0/1 end-to-end

.PHONY: clean
clean: ## Remove generated CSVs and figures (keeps source)
	rm -f $(OUT)/results.csv $(OUT)/loadbalance.csv $(OUT)/phase2_*.csv
	rm -rf $(OUT)/figures

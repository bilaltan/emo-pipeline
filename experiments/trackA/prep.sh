#!/usr/bin/env bash
# Track A step 3 (WSL): Phases 0-2 on one dataset with a held-out link split, then the numpy cache.
#   bash experiments/trackA/prep.sh reddit
#   bash experiments/trackA/prep.sh ogbn-products [--reuse-phase1]
# Tiny communities go to shared misc bins (Phase 2's EMR default), so every vertex belongs to a unit.
# 4 cores and an 8g driver leave room in WSL for the other projects.
set -o pipefail
DS=$1; shift
cd "$(dirname "$0")/../.."
LOG=$HOME/lakegrl/trackA_prep_$DS.log
mkdir -p "$HOME/lakegrl"
echo "start $(date)" > "$LOG"
TIME=""; [ -x /usr/bin/time ] && TIME="/usr/bin/time -v"
$TIME timeout 5400 bash scripts/wsl_run.sh python scripts/run_local_pipeline.py --dataset "$DS" --alg louvain \
  --link-split 50000 --link-seed 0 --stop-after 2 --cores 4 --driver-mem 8g --tiny-comm misc "$@" \
  --json-out "$HOME/lakegrl/trackA_prep.jsonl" >> "$LOG" 2>&1
echo "pipeline rc=$?" >> "$LOG"
timeout 1800 bash scripts/wsl_run.sh python experiments/trackA/export_cache.py --dataset "$DS-lp" --alg louvain >> "$LOG" 2>&1
echo "export rc=$? end $(date)" >> "$LOG"
tr '\r' '\n' < "$LOG" | grep -a -v "Stage " | grep -a -E "Phase|Link split|communit|rc=|Maximum resident|Elapsed|Error|Traceback|n_units|n_halo|n_members|export_seconds|^start|end " | tail -40

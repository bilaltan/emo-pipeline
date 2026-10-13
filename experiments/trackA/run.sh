#!/usr/bin/env bash
# Track A harness on the GPU, through the GPU lock shared with LakeReplay / LakeFresh (DATA_ROOT/gpu.lock).
#   bash experiments/trackA/run.sh reddit-lp M0,M1a,M1b,M2 0,1,2 [extra harness args]
# Output: results/trackA/runs.jsonl (one JSON record per method x seed); log in $HOME/lakegrl/.
set -o pipefail
DS=$1; METHODS=${2:-M0,M1a,M1b,M2}; SEEDS=${3:-0}; shift 3 2>/dev/null
cd "$(dirname "$0")/../.."
LOCK=/mnt/c/Users/btan4/Desktop/GNN/idea3/idea3/scripts/gpu_lock.py
LOG=$HOME/lakegrl/trackA_${DS}_$(date +%Y%m%d-%H%M%S).log
bash scripts/wsl_run.sh python "$LOCK" run --owner lakegrl --wait -1 -- \
  python experiments/trackA/harness.py --dataset "$DS" --methods "$METHODS" --seeds "$SEEDS" "$@" > "$LOG" 2>&1
RC=$?; echo "rc=$RC log=$LOG"
grep -v -E "Warning|warnings.warn|sparse_coo_tensor|run_backward" "$LOG" | tail -12
exit $RC

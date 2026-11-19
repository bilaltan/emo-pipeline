#!/usr/bin/env bash
# Backbones (advisor 2026-11-19, model agnosticism): GCN or GATv2 trained on the stored Phase 2 view of Table B row 5
# (leiden_c10000, written once for GraphSAGE; Stage 3b alone, no Phase 1-2), against the full-graph baseline with the
# same layers (utils/backbones.py). As Table B: official split, 2 layers, 256 hidden, 100 epochs, seeds 0-2, GPU,
# 8 workers, peak GPU by peak.sh. A baseline that fills the 24 GB card is stopped by peak.sh and recorded as
# gpu_full = 1 (OOM; the model is not made smaller), and the LakeGRL runs follow.
#   bash experiments/tableB/backbones.sh ogbn-arxiv gcn        -> results/backbones/gcn
set -eo pipefail
ds=$1; m=$2
cd "$(dirname "$0")/../.."
O=results/backbones/$m; mkdir -p $O
store=${LAKEGRL_STORE:-$HOME/lakegrl/store}
bash experiments/tableB/peak.sh $O/peak.jsonl "$ds full" bash scripts/wsl_run.sh python \
  experiments/baselines/node_baselines.py --dataset "$ds" --method full --model $m --cache $store/$ds/tableB_full \
  --device cuda --out $O/baselines.jsonl || echo "backbones.sh: full-graph $m on $ds ended with rc $? ($O/peak.jsonl)"
TB_OUT=$O TB_ROWS=5 TB_READS=0 TB_PIPELINE_ONLY=1 TB_ARGS="--model $m --stage3-only" bash experiments/tableB/run.sh "$ds"

#!/usr/bin/env bash
# Advisor 2026-11-20 reruns on d658119 (branch backbones-cmax, no code change): experiments/tableB/backbones.sh with
# another output folder.
#   gcn  -> results/backbones/gcn_pre: every GCN cell again (full graph + LakeGRL) on the D^-1/2 (A + I) D^-1/2
#           computed once (8a6e1e7), so all GCN numbers are one code version
#   sage -> results/backbones/sage: LakeGRL alone, Stage 3b on the stored leiden_c10000 view (--stage3-only) as GCN and
#           GATv2; the full-graph GraphSAGE stays Table 4 (node_baselines' SAGE path is unchanged since fe671e8)
#   bash results/backbones/rerun_1120.sh <dataset> <gcn|sage>
set -eo pipefail
ds=$1; m=$2
cd "$(dirname "$0")/../.."
O=results/backbones/$m; [ "$m" = gcn ] && O=results/backbones/gcn_pre
mkdir -p $O
store=${LAKEGRL_STORE:-$HOME/lakegrl/store}
if [ "$m" = gcn ]; then
  bash experiments/tableB/peak.sh $O/peak.jsonl "$ds full" bash scripts/wsl_run.sh python \
    experiments/baselines/node_baselines.py --dataset "$ds" --method full --model $m --cache $store/$ds/tableB_full \
    --device cuda --out $O/baselines.jsonl || echo "rerun_1120.sh: full-graph $m on $ds ended with rc $? ($O/peak.jsonl)"
fi
TB_OUT=$O TB_ROWS=5 TB_READS=0 TB_PIPELINE_ONLY=1 TB_ARGS="--model $m --stage3-only" bash experiments/tableB/run.sh "$ds"

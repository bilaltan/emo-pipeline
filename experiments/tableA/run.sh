#!/usr/bin/env bash
# Table A for one dataset (plan: "LakeGRL Table A plan"): the eight layouts written 3 times each, then Stage 3b's reads
# of every work item from each, 3 cold runs through the S3 emulator, then the table. Needs Phase 0 and Leiden
# communities in the store (run_local_pipeline.py --alg leiden). All hardware threads, no throttling.
#   bash experiments/tableA/run.sh ogbn-arxiv [extra write_layouts.py args]
set -eo pipefail
ds=$1; shift
cd "$(dirname "$0")/../.."
bash scripts/wsl_run.sh python experiments/tableA/write_layouts.py --dataset "$ds" --alg leiden --repeats 3 "$@"
bash scripts/wsl_run.sh python experiments/tableA/read_layouts.py --dataset "$ds" --runs 3 \
  --layouts join mv hive merged merged_rg merged_blk
# Z-order / Liquid: OPTIMIZE puts every community in each ~40M-row edge row group, so a reader peaks at 7.2 GB on
# Reddit (mv 3.3 GB) and 20 readers crashed the 26 GB WSL VM 3 times; 3 readers fit.
bash scripts/wsl_run.sh python experiments/tableA/read_layouts.py --dataset "$ds" --runs 3 --layouts zorder liquid --procs 3
bash scripts/wsl_run.sh python experiments/tableA/summary.py --datasets "$ds" --md "results/tableA/tableA_$ds.md"

#!/usr/bin/env bash
# Table A for one dataset into one run set (results/tableA/README.md): the eight layouts written 3 times each, then
# Stage 3b's reads of every work item from each, 3 cold runs through the S3 emulator, then the table. Needs Phase 0
# and the run set's Phase 1 units in the store (run_local_pipeline.py --stop-after 2). All hardware threads.
#   RS=rleiden_c10000 bash experiments/tableA/run.sh ogbn-arxiv [extra write_layouts.py args, e.g. --cap 5000]
# ZL_SAMPLE=10 reads every 10th bin for Z-order / Liquid (products, as in idmod_c10000).
set -eo pipefail
ds=$1; shift
cd "$(dirname "$0")/../.."
rs=${RS:?RS = run set, e.g. rleiden_c10000}
res=results/tableA/$rs; tab=$HOME/lakegrl/tableA/$rs   # never the idmod_c10000 tables in ~/lakegrl/tableA/<ds>
mkdir -p "$res"
bash scripts/wsl_run.sh python experiments/tableA/write_layouts.py --dataset "$ds" --alg leiden --repeats 3 \
  --layouts mv join hive zorder liquid merged merged_rg merged_blk blk_128 --out-dir "$tab" --out "$res/write.jsonl" "$@"
bash scripts/wsl_run.sh python experiments/tableA/read_layouts.py --dataset "$ds" --runs 3 --out-dir "$tab" \
  --out "$res/read.jsonl" --layouts join mv hive merged merged_rg merged_blk blk_128
# Z-order / Liquid: OPTIMIZE puts every community in each ~40M-row edge row group, so a reader peaks at 7.2 GB on
# Reddit (mv 3.3 GB) and 20 readers crashed the 26 GB WSL VM 3 times; 3 readers fit.
bash scripts/wsl_run.sh python experiments/tableA/read_layouts.py --dataset "$ds" --runs 3 --out-dir "$tab" \
  --out "$res/read.jsonl" --layouts zorder liquid --procs 3 --bin-sample "${ZL_SAMPLE:-1}"
bash scripts/wsl_run.sh python experiments/tableA/summary.py --datasets "$ds" --writes "$res/write.jsonl" \
  --reads "$res/read.jsonl" --out-dir "$tab" --md "$res/tableA_$ds.md"

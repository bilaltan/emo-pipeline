#!/usr/bin/env bash
# Table B for one dataset (advisor 2026-11-02, Section 4.2): node classification on the official split, 2-layer SAGE
# with experiment_config.TRAIN (100 epochs, lr 0.01), seeds 0-2, trained on the GPU. Storage reads are timed apart from
# training, cold, through the S3 emulator (20 ms + 80 MB/s per GET) with 20 readers, as Table A; training runs count
# GETs only. Run it alone (gpu_lock): the reads need the machine.
#   row 1 full-graph SAGE, row 2 Cluster-GCN: experiments/baselines/node_baselines.py on a cache of the full graph;
#         read = read_full.py (the whole raw node, edge and mask tables); METIS time = partition_seconds
#   row 3 local units only: Stage 3a (no context), tiny 'keep', one model per coalesced work item (--pack-train)
#   row 4 id mod n: as row 5 but plain Leiden communities cut by id mod n (--unit-split idmod)
#   row 5 LakeGRL: recursive Leiden C_max 10k + CaaN macro-GNN; minor and isolated nodes scored by its global model
#   rows 3-5 store Phase 2 as Table A row 8b (hive_merged, bounded sqrt row group with a 2 MB floor, one file per
#   directory); read = read_units.py on the run's own bins, right after its seed 0; rows 4-5 add read_full.py on
#   what Stage 3b reads besides the bins: the macro-graph Phase 2 stores (advisor 2026-11-04; until then the raw
#   nodes, masks and edges and the whole p2_nodes, to rebuild it), the communities table, the unit plan (rows per
#   community; Phase 2 stores it with the macro-graph since 11-04 evening, before that p2_nodes' community_id column)
# Peak GPU / host memory: peak.sh around every training run (also the records' peak_rss_tree_mb / peak_gpu_mb).
#   bash experiments/tableB/run.sh ogbn-arxiv
set -eo pipefail
ds=$1
cd "$(dirname "$0")/../.."
O=${TB_OUT:-results/tableB}; mkdir -p $O/work
PY="bash scripts/wsl_run.sh python"
PEAK="bash experiments/tableB/peak.sh $O/peak.jsonl"
store=${LAKEGRL_STORE:-$HOME/lakegrl/store}; cache=$store/$ds/tableB_full
# TB_CORES: Spark task slots = unit models trained at once on the GPU. Reddit's dense units overflow the 24 GB card
# at 8 and the driver spills into Windows memory (11-03). TB_PIPELINE_ONLY=1: rows 3-5 only (rows 1-2 already in).
# The 11-03 memory reruns (advisor: 1- vs 8-worker GPU peak, host without the JVM) run seed 0 only into their own
# TB_OUT: TB_SEEDS (default 0 1 2), TB_REUSE_P1=1 keeps Phase 1 for seed 0 too, TB_READS=0 skips the cold reads.
# TB_ROWS (default 5 4 3) and TB_ARGS (more run_local_pipeline.py arguments): the runtime breakdown run (row 5,
# --p2-time-split).
cores=${TB_CORES:-8}

if [ -z "$TB_PIPELINE_ONLY" ]; then
  $PY experiments/tableB/read_full.py --dataset "$ds" --store "$store" --out $O/read_full.jsonl
  [ -f "$cache/meta.json" ] || $PY experiments/trackA/export_cache.py --dataset "$ds" --root "$store/$ds" --out "$cache"
  for m in full cluster; do
    $PEAK "$ds $m" $PY experiments/baselines/node_baselines.py --dataset "$ds" --method $m --cache "$cache" \
      --device cuda --out $O/baselines.jsonl
  done
fi

declare -A ROW=([5]="--unit-split leiden --tiny-comm drop" [4]="--unit-split idmod --tiny-comm drop"
                [3]="--unit-split leiden --tiny-comm keep --stage3a --pack-train --stop-after 3")
for r in ${TB_ROWS:-5 4 3}; do
  for s in ${TB_SEEDS:-0 1 2}; do
    reuse=""; [ $s -gt 0 ] || [ -n "$TB_REUSE_P1" ] && reuse=--reuse-phase1   # Phase 1 is seeded (42): the same units for every seed
    $PEAK "$ds row$r seed$s" $PY scripts/run_local_pipeline.py --dataset "$ds" --alg leiden ${ROW[$r]} --c-max 10000 \
      --p2-layout hive_merged --p2-rg-floor-mb 2 --p2-one-file --s3-emu count --device cuda --cores $cores \
      --driver-mem 16g --seed $s $reuse $TB_ARGS --work-out $O/work/${ds}_row$r.json --json-out $O/pipeline.jsonl
    if [ $s = 0 ] && [ "${TB_READS:-1}" = 1 ]; then
      $PY experiments/tableB/read_units.py --work $O/work/${ds}_row$r.json --row $r --dataset "$ds" \
        --out $O/read_units.jsonl
      tag=leiden_c10000; [ $r = 4 ] && tag=leiden   # idmod units: Phase 1 without the recursive split
      [ $r = 3 ] || $PY experiments/tableB/read_full.py --dataset "$ds" --store "$store" --row "$r global" \
        --tables p2_macro_nodes/$tag p2_macro_edges/$tag communities/$tag p2_unit_plan/$tag \
        --out $O/read_full.jsonl
    fi
  done
done

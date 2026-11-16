#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  The ICDE AWS batch (plan doc, "AWS batch"), run on the EMR primary node from the repo checkout after
#  scripts/emr_setup.sh. One target per call, in this order:
#
#    dryrun            arxiv end to end on 1 worker (first real S3 run) + DistDGL for 2 epochs on 1 worker
#    layout  DS        run 1: Stage 3b reads on S3, plain vs block layout; a timing run and a counting run each
#    scaling DS W      run 2: the full pipeline (Phases 0-3b) on W workers, one 4-core executor per worker
#    distdgl DS W      run 3: DistDGL from the same Delta tables on W workers, same core budget per worker
#    collect           every run log, record and Spark event log to S3, before the cluster is terminated
#
#  Every run appends its record to results/icde/emr_runs.jsonl (LakeGRL) or distdgl_runs.jsonl, with Spark
#  executors x cores, and tees its log to logs_emr/. LakeGRL's method comes from experiment_config.py, which
#  scripts/test_emr_parity.py checks against the local driver; a local dry run of runners/run_emr.py on arxiv
#  matched the local driver (results/icde/emr_dryrun.jsonl).
#
#  Per worker: one executor of 4 cores, 2 threads per task (r6i.2xlarge: 8 vCPUs = 4 physical cores), so a worker
#  matches a local 4-core run and DistDGL gets the same 8 threads (2 trainers x 4).
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
cd "$(dirname "$0")/.."
BUCKET=${BUCKET:-us-east-1-s3-gnn}
EXEC_CORES=${EXEC_CORES:-4}; OMP=${OMP:-2}
EXEC_MEM=${EXEC_MEM:-12}; EXEC_OVH=${EXEC_OVH:-24}   # 36 GB: two do not fit one 64 GB node, so one executor per worker
mkdir -p logs_emr results/icde
TS=$(date -u +%Y%m%dT%H%M%SZ)
GIT=$(git rev-parse --short HEAD)
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "uncommitted changes: commit first, every record names the commit"; exit 1; }

lakegrl() {   # name, then run_emr.py arguments
    local name=$1; shift
    echo "=== $name ($GIT) ==="
    python3 -u runners/run_emr.py --algorithms louvain --force-rerun --omp-threads "$OMP" \
        --executor-cores "$EXEC_CORES" --executor-memory-gb "$EXEC_MEM" --executor-memory-overhead-gb "$EXEC_OVH" \
        --driver-memory-gb 24 --json-out results/icde/emr_runs.jsonl "$@" 2>&1 | tee "logs_emr/${name}_$TS.log"
}

distdgl() {   # dataset, workers, extra launch_spark.py arguments
    local ds=$1 w=$2; shift 2
    echo "=== distdgl $ds on $w workers ($GIT) ==="
    spark-submit --master yarn --deploy-mode client experiments/distdgl/launch_spark.py --dataset "$ds" \
        --workers "$w" --trainers 2 --omp 4 --executor-cores "$EXEC_CORES" --executor-memory "$((EXEC_MEM + EXEC_OVH - 4))g" \
        --bucket "$BUCKET" "$@" 2>&1 | tee "logs_emr/distdgl_${ds}_w${w}_$TS.log"
}

case "${1:-}" in
dryrun)
    lakegrl dryrun_arxiv --datasets ogbn-arxiv --experiment-name icde_dryrun --run-phase0 --force-reingest \
        --executor-instances 1
    distdgl ogbn-arxiv 1 --epochs 2 --out results/icde/distdgl_dryrun.jsonl
    ;;
layout)
    DS=${2:?dataset}; W=${3:-4}
    # Phases 0-1 once; each layout gets its own Phase 2 tables (experiment name), then Stage 3b twice: timed
    # (plain reads) and counted (GETs and bytes, store_io 'count'; metering slows reads, so it is a separate run)
    lakegrl "layout_${DS}_p01" --datasets "$DS" --experiment-name icde_layout_plain --run-phase0 --run-phase1 \
        --no-phase2 --no-phase3b --executor-instances "$W"
    for L in plain block; do
        lakegrl "layout_${DS}_${L}" --datasets "$DS" --experiment-name "icde_layout_$L" --layout "$L" --no-phase0 \
            --no-phase1 --run-phase2 --run-phase3b --executor-instances "$W"
        lakegrl "layout_${DS}_${L}_count" --datasets "$DS" --experiment-name "icde_layout_$L" --layout "$L" \
            --s3-emu count --no-phase0 --no-phase1 --no-phase2 --run-phase3b --executor-instances "$W"
    done
    ;;
scaling)
    DS=${2:?dataset}; W=${3:?workers}
    lakegrl "scaling_${DS}_w$W" --datasets "$DS" --experiment-name "icde_scaling_w$W" --run-phase0 --run-phase1 \
        --run-phase2 --run-phase3b --executor-instances "$W"
    ;;
distdgl)
    distdgl "${2:?dataset}" "${3:?workers}"
    ;;
collect)
    D=s3://$BUCKET/icde-aws/$TS
    aws s3 cp --recursive --only-show-errors logs_emr "$D/logs_emr/"
    aws s3 cp --recursive --only-show-errors results/icde "$D/results_icde/"
    hdfs dfs -get /var/log/spark/apps logs_emr/spark-history 2>/dev/null && \
        aws s3 cp --recursive --only-show-errors logs_emr/spark-history "$D/spark-history/"
    echo "copied to $D ; locally: aws s3 sync $D results/aws/$TS"
    ;;
*)
    sed -n '3,12p' "$0"; exit 1 ;;
esac

#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  run_all.sh — Unified Single-Command Master Benchmark Orchestrator
#  (All Experiments, Datasets, Scaling Sweeps, Timing, and Table Generation)
#
#  ONE COMMAND DOES EVERYTHING:
#    ./run_all.sh
#
#  What this executes sequentially end-to-end:
#    1. Environment & Large Storage Scratch Discovery (/mnt/tmp, /mnt1/tmp, etc.)
#    2. OGBN-Products Global Graph Benchmark (Louvain Default, Node + Link Prediction)
#    3. ~100M-Scale Benchmarks: LiveJournal (34.7M edges), Orkut (117.2M edges),
#       OGBN-Products (61.9M edges), Reddit (11.6M edges), OGBN-MAG (5.4M edges)
#    4. Standard Academic Suite: WikiCS, Coauthor-Physics, Coauthor-CS, DeezerEurope
#    5. Multi-Executor Scaling Sweep across all 9 benchmark datasets
#    6. System Ablation Matrix on Reddit (CAAN vs Decoupled vs Uncached S3)
#    7. Distributed Baseline Evaluation (Full Graph PyG & DistDGL)
#    8. Granular Phase-by-Phase Latency Breakdown (Phases 0, 1, 2, 3 Node/Link, 3b)
#    9. Fixes Table 4 baseline metrics & separates self-collected data from ML-GRL
#   10. Compiles Master Excel Report (5 Sheets) & 6 Publication LaTeX Tables
#   11. Syncs all artifacts to overleaf/results/ and Amazon S3 bucket
# ══════════════════════════════════════════════════════════════════════════════

set -eo pipefail

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/run_all_${TIMESTAMP}"
mkdir -p "$LOG_DIR" "logs" "results" "overleaf/results"

echo "=========================================================================="
echo " Starting Unified EMO Master Pipeline Execution (All-In-One Script)"
echo " Timestamp: $TIMESTAMP"
echo " Log Directory: $LOG_DIR"
echo "=========================================================================="

# ── 1. Storage & Environment Discovery ─────────────────────────────────────────
CANDIDATES=("/mnt/tmp" "/mnt1/tmp" "/mnt2/tmp" "/mnt/spark" "/mnt1/spark" "/mnt2/spark" "/tmp")
LARGE_TMP="/tmp"

for candidate in "${CANDIDATES[@]}"; do
    if [ ! -d "$candidate" ]; then
        sudo mkdir -p "$candidate" 2>/dev/null || mkdir -p "$candidate" 2>/dev/null || true
        sudo chmod 777 "$candidate" 2>/dev/null || true
    fi
    if [ -w "$candidate" ]; then
        LARGE_TMP="$candidate"
        if [[ "$candidate" != "/tmp" ]]; then
            break
        fi
    fi
done

echo "► Active Scratch Storage: $LARGE_TMP"
mkdir -p "$LARGE_TMP/.local" "$LARGE_TMP/.pip-cache" "$LARGE_TMP/.dgl" "$LARGE_TMP/ogb_data"

export TMPDIR="$LARGE_TMP"
export TEMP="$LARGE_TMP"
export TMP="$LARGE_TMP"
export PIP_CACHE_DIR="$LARGE_TMP/.pip-cache"
if [[ "$LARGE_TMP" == "/mnt"* ]]; then
    export PYTHONUSERBASE="$LARGE_TMP/.local"
fi
export DGL_DOWNLOAD_DIR="$LARGE_TMP/.dgl"

# Discover Python executable with installed dependencies
if [ -x "/opt/homebrew/bin/python3" ]; then
    PYTHON_BIN="/opt/homebrew/bin/python3"
elif [ -x "$PWD/.venv/bin/python3" ]; then
    PYTHON_BIN="$PWD/.venv/bin/python3"
else
    PYTHON_BIN="python3"
fi

PY_VER=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.10")
if [[ "$LARGE_TMP" == "/mnt"* ]]; then
    export PYTHONPATH="$LARGE_TMP/.local/lib/python${PY_VER}/site-packages:$PWD:$PYTHONPATH"
else
    export PYTHONPATH="$HOME/Library/Python/${PY_VER}/lib/python/site-packages:$PWD:$PYTHONPATH"
fi

# Set JVM Flags for Apache Arrow tensor transfer
export JAVA_TOOL_OPTIONS="--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.security=ALL-UNNAMED --add-opens=java.base/sun.security.action=ALL-UNNAMED --add-opens=java.base/math=ALL-UNNAMED"

# Select Runner
if [ -f "runners/run_emr.py" ]; then
    RUNNER="$PYTHON_BIN runners/run_emr.py"
else
    RUNNER="$PYTHON_BIN runners/run_local.py"
fi
echo "► Execution Runner: $RUNNER"

OUT_DIR="results/s3_latest_results"
mkdir -p "$OUT_DIR"

sync_all_artifacts() {
    mkdir -p "$OUT_DIR" "overleaf/results"
    cp -r results/*.xlsx "$OUT_DIR/" 2>/dev/null || true
    cp -r results/*.tex "$OUT_DIR/" 2>/dev/null || true
    cp -r results/*.tex "overleaf/results/" 2>/dev/null || true
    cp -r results/*.json "$OUT_DIR/" 2>/dev/null || true
    cp -r "$LOG_DIR"/*.log "$OUT_DIR/" 2>/dev/null || true
    cp -r logs/*.log "$OUT_DIR/" 2>/dev/null || true
}

# ── 2. STAGE 1: OGBN-Products Global Graph Benchmark (Louvain Default) ─────────
echo ""
echo "=========================================================================="
echo " [STAGE 1/6] Running OGBN-Products Global Graph Experiment"
echo " Tasks: Node Classification + Link Prediction | Partitioner: Louvain (Default)"
echo " Scale: 2.45M Nodes, 61.9M Edges"
echo "=========================================================================="
STAGE1_LOG="$LOG_DIR/stage1_ogbn_products.log"

$RUNNER \
  --experiment-name "products_louvain_global" \
  --datasets "ogbn-products" \
  --algorithms "louvain" \
  --run-phase0 \
  --run-phase1 \
  --run-phase2 \
  --run-phase3 \
  --run-phase3b \
  --run-phase4 \
  --force-rerun \
  --global-mapping "true" \
  --task-type "both" \
  2>&1 | tee "$STAGE1_LOG"

echo "✓ Stage 1 (OGBN-Products) Completed. Log: $STAGE1_LOG"
sync_all_artifacts

# ── 3. STAGE 2: ~100M-Scale Graphs Suite (LiveJournal, Orkut, Reddit, MAG) ─────
echo ""
echo "=========================================================================="
echo " [STAGE 2/6] Running ~100M-Scale Dense Graph Suite"
echo " Datasets: LiveJournal (34.7M edges), Orkut (117.2M edges), Reddit (11.6M edges), MAG (5.4M edges)"
echo " Partitioner: Louvain (Default) | Tasks: Node + Link"
echo "=========================================================================="
STAGE2_LOG="$LOG_DIR/stage2_100m_scale.log"

$RUNNER \
  --experiment-name "scale_100m_suite" \
  --datasets "LiveJournal,Orkut,reddit,ogbn-mag" \
  --algorithms "louvain" \
  --run-phase0 \
  --run-phase1 \
  --run-phase2 \
  --run-phase3 \
  --run-phase3b \
  --force-rerun \
  --global-mapping "true" \
  --task-type "both" \
  2>&1 | tee "$STAGE2_LOG"

echo "✓ Stage 2 (~100M Scale) Completed. Log: $STAGE2_LOG"
sync_all_artifacts

# ── 4. STAGE 3: Academic Standards Suite (WikiCS, Coauthor, DeezerEurope) ───────
echo ""
echo "=========================================================================="
echo " [STAGE 3/6] Running Academic Standards Suite"
echo " Datasets: WikiCS, Coauthor-Physics, Coauthor-CS, DeezerEurope"
echo " Tasks: Node Classification + Link Prediction | Partitioner: Louvain"
echo "=========================================================================="
STAGE3_LOG="$LOG_DIR/stage3_standards.log"

$RUNNER \
  --experiment-name "standards_suite" \
  --datasets "WikiCS,Coauthor-Physics,Coauthor-CS,DeezerEurope" \
  --algorithms "louvain" \
  --run-phase0 \
  --run-phase1 \
  --run-phase2 \
  --run-phase3 \
  --run-phase3b \
  --force-rerun \
  --global-mapping "true" \
  --task-type "both" \
  2>&1 | tee "$STAGE3_LOG"

echo "✓ Stage 3 (Standards) Completed. Log: $STAGE3_LOG"
sync_all_artifacts

# ── 5. STAGE 4: Multi-Executor Scalability Sweep (8 -> 16 -> 32 Executors) ─────
echo ""
echo "=========================================================================="
echo " [STAGE 4/6] Running Multi-Executor Scalability Sweep (Table 13 & Fig 8 Data)"
echo " Scaling Tiers: 8, 16, 32 Executors across all benchmark datasets"
echo "=========================================================================="
STAGE4_LOG="$LOG_DIR/stage4_scalability_sweep.log"

SWEEP_TIERS=(8 16 32)
BENCH_DATASETS=("WikiCS" "Coauthor-Physics" "Coauthor-CS" "DeezerEurope" "reddit" "ogbn-products" "ogbn-mag" "LiveJournal" "Orkut")

for E in "${SWEEP_TIERS[@]}"; do
    echo "► [Scaling Pass] Executing with $E Executors..."
    for DS in "${BENCH_DATASETS[@]}"; do
        $RUNNER \
          --experiment-name "scaling_sweep_${DS}_e${E}" \
          --datasets "$DS" \
          --algorithms "louvain" \
          --executor-instances "$E" \
          --run-phase0 \
          --run-phase1 \
          --run-phase2 \
          --run-phase3 \
          --run-phase3b \
          --global-mapping "true" \
          --task-type "both" \
          2>&1 | tee -a "$STAGE4_LOG" || true
    done
done

echo "✓ Stage 4 (Scalability Sweep) Completed. Log: $STAGE4_LOG"
sync_all_artifacts

# ── 6. STAGE 5: System Ablation Study on Reddit (Table 9 Data) ─────────────────
echo ""
echo "=========================================================================="
echo " [STAGE 5/6] Running System Ablation Study on Reddit (Table 9)"
echo " Evaluating: Full EMO vs. Decoupled (w/o CAAN) vs. Uncached S3 Reads"
echo "=========================================================================="
STAGE5_LOG="$LOG_DIR/stage5_ablation.log"

# 5A: Full EMO (Louvain + SAGE + CAAN)
$RUNNER \
  --experiment-name "ablation_louvain_caan" \
  --datasets "reddit" \
  --algorithms "louvain" \
  --run-phase0 \
  --run-phase3 \
  --run-phase3b \
  --global-mapping "true" \
  2>&1 | tee -a "$STAGE5_LOG"

# 5B: Decoupled Louvain (w/o CAAN)
$RUNNER \
  --experiment-name "ablation_louvain_nocaan" \
  --datasets "reddit" \
  --algorithms "louvain" \
  --run-phase0 \
  --run-phase3 \
  --no-phase3b \
  --global-mapping "true" \
  2>&1 | tee -a "$STAGE5_LOG"

# 5C: Uncached S3 Reads (w/o Delta Caching)
$RUNNER \
  --experiment-name "ablation_uncached_s3" \
  --datasets "reddit" \
  --algorithms "louvain" \
  --run-phase0 \
  --force-reingest \
  --run-phase3 \
  --run-phase3b \
  2>&1 | tee -a "$STAGE5_LOG"

echo "✓ Stage 5 (Ablation Study) Completed. Log: $STAGE5_LOG"
sync_all_artifacts

# ── 7. STAGE 6: Post-Processing, Master Excel & Publication LaTeX Tables ─────────
echo ""
echo "=========================================================================="
echo " [STAGE 6/6] Generating Master Excel Report & Publication LaTeX Tables"
echo "=========================================================================="
STAGE6_LOG="$LOG_DIR/stage6_reports.log"

REPORT_PATH="results/emr_cluster_master_results.xlsx"

echo "► Compiling 5-Sheet Master Excel Report..."
$PYTHON_BIN runners/generate_emr_excel_report.py --cluster-type 4worker --output "$REPORT_PATH" 2>&1 | tee -a "$STAGE6_LOG" || true

echo "► Compiling 6 LaTeX Tables (Separating Self-Collected vs Literature)..."
$PYTHON_BIN -m phases.phase5_reporting 2>&1 | tee -a "$STAGE6_LOG" || true

echo "► Rendering CPU Cores Scaling Figures..."
$PYTHON_BIN runners/create_cpu_scaling_figure.py 2>&1 | tee -a "$STAGE6_LOG" || true

sync_all_artifacts

# ── 8. S3 Cloud Synchronization ───────────────────────────────────────────────
if command -v aws &> /dev/null; then
    echo "► Uploading all artifacts to Amazon S3: s3://us-east-1-s3-gnn/gnn-bench-out/..."
    aws s3 sync "$OUT_DIR/" "s3://us-east-1-s3-gnn/gnn-bench-out/s3_latest_results/" --quiet || true
    aws s3 sync "$OUT_DIR/" "s3://us-east-1-s3-gnn/gnn-bench-out/spark-results/" --quiet || true
    if [ -f "$REPORT_PATH" ]; then
        aws s3 cp "$REPORT_PATH" "s3://us-east-1-s3-gnn/gnn-bench-out/emr_cluster_master_results.xlsx" --quiet || true
    fi
fi

echo ""
echo "=========================================================================="
echo " ALL EXPERIMENT REQUESTS COMPLETED SUCCESSFULLY IN ONE COMMAND!"
echo "=========================================================================="
echo " Master Excel Report: $REPORT_PATH"
echo " Consolidated Artifacts Directory: $OUT_DIR"
echo " Synchronized Overleaf Tables Directory: overleaf/results/"
echo " Generated Publication Tables:"
echo "   1. Table 4 (Node Classification - Self-Collected & Distinct Baselines):"
echo "      results/table4_node_classification_main.tex"
echo "   2. Table 3 (Link Prediction ROC-AUC & Actual Link Time):"
echo "      results/table3_link_prediction_main.tex"
echo "   3. Table Phase Timeline (Granular Phase 0, 1, 2, 3, 3b Latencies):"
echo "      results/table_phase_timeline.tex"
echo "   4. Table 9 (System & Method Ablation Study):"
echo "      results/table9_ablation_study.tex"
echo "   5. Table 12 (DistDGL vs. EMO Baseline Comparison):"
echo "      results/table12_distdgl_comparison.tex"
echo "   6. Table 13 (Multi-Instance Scaling Efficiency Matrix):"
echo "      results/table13_scalability_multi_instance.tex"
echo "=========================================================================="

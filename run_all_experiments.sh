#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  run_all_experiments.sh
#  Master Execution Script for EMO Paper Experiments (IEEE Big Data 2026)
#
#  Usage:
#    ./run_all_experiments.sh        # Runs all suites (1 to 5) sequentially
#    ./run_all_experiments.sh 1      # Runs only Suite 1 (Approach 1 GNNs)
#    ./run_all_experiments.sh 2      # Runs only Suite 2 (Approach 2 Scaling)
#    ./run_all_experiments.sh 3      # Runs only Suite 3 (Ablation Study)
#    ./run_all_experiments.sh 4      # Runs only Suite 4 (DistDGL Baselines)
#    ./run_all_experiments.sh 5      # Runs only Suite 5 (LaTeX Tables & Fig 8)
# ══════════════════════════════════════════════════════════════════════════════

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/experiments_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

TARGET_SUITE="${1:-all}"

echo "=========================================================================="
echo " Starting EMO Pipeline Experiment Runner"
echo " Target Suite: $TARGET_SUITE"
echo " Timestamp: $TIMESTAMP"
echo " Log Directory: $LOG_DIR"
echo "=========================================================================="

# ── 1. Discover Environment & Storage Paths ────────────────────────────────────
CANDIDATES=("/mnt/tmp" "/mnt1/tmp" "/mnt2/tmp" "/mnt/spark" "/mnt1/spark" "/mnt2/spark" "/tmp")
LARGE_TMP="/tmp"

for candidate in "${CANDIDATES[@]}"; do
    if [ ! -d "$candidate" ]; then
        mkdir -p "$candidate" 2>/dev/null || true
    fi
    if [ -w "$candidate" ]; then
        LARGE_TMP="$candidate"
        if [[ "$candidate" != "/tmp" ]]; then
            break
        fi
    fi
done

echo "► Using large storage path: $LARGE_TMP"
mkdir -p "$LARGE_TMP/.local" "$LARGE_TMP/.pip-cache" "$LARGE_TMP/.dgl"

export TMPDIR="$LARGE_TMP"
export TEMP="$LARGE_TMP"
export TMP="$LARGE_TMP"
export PIP_CACHE_DIR="$LARGE_TMP/.pip-cache"
export PYTHONUSERBASE="$LARGE_TMP/.local"
export DGL_DOWNLOAD_DIR="$LARGE_TMP/.dgl"

PY_VER=$(python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH="$LARGE_TMP/.local/lib/$PY_VER/site-packages:$PYTHONPATH"

# Determine execution engine runner (local vs EMR cluster)
if [ -f "runners/run_emr.py" ]; then
    RUNNER="python3 runners/run_emr.py"
else
    RUNNER="python3 runners/run_local.py"
fi
echo "► Selected Runner: $RUNNER"

if [[ "$TARGET_SUITE" == "fast" || "$TARGET_SUITE" == "quick" ]]; then
    exec ./run_single_fast_run.sh
fi

# ── 2. SUITE 1: Approach 1 - Core Partition-Centric GNN Execution ─────────────
if [[ "$TARGET_SUITE" == "all" || "$TARGET_SUITE" == "1" ]]; then
    echo ""
    echo "=========================================================================="
    echo " [SUITE 1/5] Running Approach 1: Core Partition-Centric GNNs (Phases 0-3b)"
    echo "=========================================================================="

    SUITE1_LOG="$LOG_DIR/suite1_approach1_community_gnns.log"
    echo "Running Community Partitioning (LPA / Louvain) + Decoupled PyTorch Training + CAAN..."

    $RUNNER \
      --experiment-name "paper_approach1_suite" \
      --datasets "WikiCS,ogbn-products,reddit,ogbn-arxiv" \
      --algorithms "lpa,louvain" \
      --run-phase0 \
      --run-phase3 \
      --run-phase3b \
      --global-mapping "true" \
      2>&1 | tee "$SUITE1_LOG"

    echo "✓ Suite 1 completed. Log saved to $SUITE1_LOG"
fi

# ── 3. SUITE 2: Approach 2 - Big-Dataset Distributed Scaling (Phases 3.7 & 3.8) ─
if [[ "$TARGET_SUITE" == "all" || "$TARGET_SUITE" == "2" ]]; then
    echo ""
    echo "=========================================================================="
    echo " [SUITE 2/5] Running Approach 2: Distributed 2-Hop Scaling (Phases 3.7 & 3.8)"
    echo "=========================================================================="

    SUITE2_LOG="$LOG_DIR/suite2_approach2_scaling.log"
    echo "Executing 2-hop SIGN-style feature propagation + global Spark ML probe..."

    $RUNNER \
      --experiment-name "paper_approach2_scaling" \
      --datasets "WikiCS,ogbn-products,ogbn-papers100M" \
      --run-phase0 \
      --no-phase3 \
      --no-phase3b \
      2>&1 | tee "$SUITE2_LOG"

    echo "✓ Suite 2 completed. Log saved to $SUITE2_LOG"
fi

# ── 4. SUITE 3: Component-wise System Ablation Study (Data for Table 9) ───────
if [[ "$TARGET_SUITE" == "all" || "$TARGET_SUITE" == "3" ]]; then
    echo ""
    echo "=========================================================================="
    echo " [SUITE 3/5] Running System & Method Ablation Study (Table 9)"
    echo "=========================================================================="

    SUITE3_LOG="$LOG_DIR/suite3_ablation_study.log"
    echo "Running component-wise ablations on reddit (CAAN vs No-CAAN, Delta Lake Caching)..."

    # Variant 3A: Full EMO (Louvain + SAGE + CAAN)
    $RUNNER \
      --experiment-name "ablation_louvain_caan" \
      --datasets "reddit" \
      --algorithms "louvain" \
      --run-phase0 \
      --run-phase3 \
      --run-phase3b \
      --global-mapping "true" \
      2>&1 | tee -a "$SUITE3_LOG"

    # Variant 3B: Full EMO (LPA + SAGE + CAAN)
    $RUNNER \
      --experiment-name "ablation_lpa_caan" \
      --datasets "reddit" \
      --algorithms "lpa" \
      --run-phase0 \
      --run-phase3 \
      --run-phase3b \
      --global-mapping "true" \
      2>&1 | tee -a "$SUITE3_LOG"

    # Variant 3C: Decoupled Louvain (w/o CAAN global graph)
    $RUNNER \
      --experiment-name "ablation_louvain_nocaan" \
      --datasets "reddit" \
      --algorithms "louvain" \
      --run-phase0 \
      --run-phase3 \
      --no-phase3b \
      --global-mapping "true" \
      2>&1 | tee -a "$SUITE3_LOG"

    # Variant 3D: Decoupled LPA (w/o CAAN global graph)
    $RUNNER \
      --experiment-name "ablation_lpa_nocaan" \
      --datasets "reddit" \
      --algorithms "lpa" \
      --run-phase0 \
      --run-phase3 \
      --no-phase3b \
      --global-mapping "true" \
      2>&1 | tee -a "$SUITE3_LOG"

    # Variant 3E: Uncached S3 Parquet Reads (w/o Delta transaction log caching)
    $RUNNER \
      --experiment-name "ablation_uncached_s3" \
      --datasets "reddit" \
      --algorithms "louvain" \
      --run-phase0 \
      --force-reingest \
      --run-phase3 \
      --run-phase3b \
      2>&1 | tee -a "$SUITE3_LOG"

    echo "✓ Suite 3 completed. Log saved to $SUITE3_LOG"
fi

# ── 5. SUITE 4: Distributed Baseline Evaluation (DistDGL vs EMO - Table 12) ────
if [[ "$TARGET_SUITE" == "all" || "$TARGET_SUITE" == "4" ]]; then
    echo ""
    echo "=========================================================================="
    echo " [SUITE 4/5] Running DistDGL & Full-Graph Baselines (Table 12)"
    echo "=========================================================================="

    SUITE4_LOG="$LOG_DIR/suite4_distdgl_baselines.log"
    echo "Evaluating DistDGL and PyG full-graph baselines..."

    $RUNNER \
      --experiment-name "distdgl_comparison" \
      --datasets "WikiCS,ogbn-products,reddit,ogbn-arxiv" \
      --run-phase0 \
      --run-phase4 \
      2>&1 | tee "$SUITE4_LOG"

    echo "✓ Suite 4 completed. Log saved to $SUITE4_LOG"
fi

# ── 6. SUITE 5: Post-Processing & Figure/Table Compilation ────────────────────
if [[ "$TARGET_SUITE" == "all" || "$TARGET_SUITE" == "5" ]]; then
    echo ""
    echo "=========================================================================="
    echo " [SUITE 5/5] Generating LaTeX Tables & Publication Figures"
    echo "=========================================================================="

    echo "► 1. Exporting summary metrics to Excel and LaTeX tables..."
    python3 -m phases.phase5_reporting 2>&1 | tee -a "$LOG_DIR/phase5_reporting.log"

    echo "► 2. Rendering ML-GRL Figure 8 CPU Cores Scaling plot..."
    python3 runners/create_cpu_scaling_figure.py 2>&1 | tee -a "$LOG_DIR/figure8_generation.log"

    echo "=========================================================================="
    echo " Generated LaTeX Tables & Figures:"
    echo "   - Table 3 (Link Prediction): results/table3_link_prediction_main.tex"
    echo "   - Table 4 (Node Classification): results/table4_node_classification_main.tex"
    echo "   - Table 9 (Ablation Study): results/table9_ablation_study.tex"
    echo "   - Table 12 (DistDGL Comparison): results/table12_distdgl_comparison.tex"
    echo "   - Table 13 (Multi-Instance Scaling): results/table13_scalability_multi_instance.tex"
    echo "   - Figure 8 PDF: results/figures/fig8_cpu_cores_scaling.pdf"
    echo "=========================================================================="
fi

echo "=========================================================================="
echo " EXPERIMENT RUN FINISHED FOR SUITE: $TARGET_SUITE"
echo "=========================================================================="

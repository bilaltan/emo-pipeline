#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  EMR Cluster Setup
#  ► Run on the EMR primary node after SSH.
#  ► Installs the driver's Python packages, then provisions EVERY worker node
#    and refuses to finish until a preflight check passes on all of them.
#  ► Caches and local packages go to the large EBS/NVMe volume (e.g. /mnt) to
#    keep the 15 GB root disk from filling.
#
#  Why the worker step exists
#  --------------------------
#  This script used to set up the driver only. Workers could be missing numpy
#  and pandas entirely, and an Arrow-to-pandas conversion scheduled on such a
#  node dies inside Spark's pandas-UDF machinery before any user code runs, so
#  the in-UDF pip fallback the phases use for torch cannot rescue it. On
#  2026-09-11 one unprovisioned node out of four aborted two Reddit runs, at 48
#  and 27 minutes, both times in Phase 3b and both times after Phase 3 had
#  already produced good results.
# ══════════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== 1. Discovering Large Storage Volume ==="
CANDIDATES=("/mnt/tmp" "/mnt1/tmp" "/mnt2/tmp" "/mnt/spark" "/mnt1/spark" "/mnt2/spark" "/tmp")
LARGE_TMP="/tmp"

for candidate in "${CANDIDATES[@]}"; do
    if [ ! -d "$candidate" ]; then
        sudo mkdir -p "$candidate" 2>/dev/null || true
        sudo chmod 777 "$candidate" 2>/dev/null || true
    fi

    if [ -w "$candidate" ]; then
        LARGE_TMP="$candidate"
        if [[ "$candidate" != "/tmp" ]]; then
            break
        fi
    fi
done

echo "Using large storage path: $LARGE_TMP"
mkdir -p "$LARGE_TMP/.local" "$LARGE_TMP/.pip-cache" "$LARGE_TMP/.dgl"

# Redirect pip and build temp files to the large disk rather than the root volume.
export TMPDIR="$LARGE_TMP"
export TEMP="$LARGE_TMP"
export TMP="$LARGE_TMP"
export PIP_CACHE_DIR="$LARGE_TMP/.pip-cache"
export PYTHONUSERBASE="$LARGE_TMP/.local"
export DGL_DOWNLOAD_DIR="$LARGE_TMP/.dgl"

PY_VERSION=$(python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH="$LARGE_TMP/.local/lib/$PY_VERSION/site-packages:$PYTHONPATH"

echo "=== 2. Installing System Packages & Development Tools ==="
# EMR 7.x uses Amazon Linux 2023, EMR 6.x uses Amazon Linux 2.
if command -v dnf &> /dev/null; then
    sudo dnf groupinstall -y "Development Tools"
    sudo dnf install -y gcc-c++ make python3-devel git tmux
elif command -v yum &> /dev/null; then
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y gcc-c++ make python3-devel git tmux
else
    echo "Warning: no dnf or yum found. Ensure development tools, git and tmux are installed."
fi

echo "=== 3. Upgrading Pip ==="
python3 -m pip install --user --upgrade pip

echo "=== 4. Installing Core Python Packages (driver) ==="
# pandas and scikit-learn are required, not incidental: the Phase 3/3b UDFs call
# sklearn.metrics.roc_auc_score, and every Arrow exchange goes through pandas.
python3 -m pip install --user "numpy<2.0.0" pandas pyarrow scipy scikit-learn ogb boto3

# Community detection (driver only — Phase 1 runs on the driver).
python3 -m pip install --user igraph leidenalg

echo "=== 5. Installing ML Frameworks (driver) ==="
python3 -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --user torch-geometric

echo "=== 6. Installing DGL (optional; the pipeline degrades gracefully without it) ==="
python3 -m pip install --user dgl==1.1.3 -f https://data.dgl.ai/wheels/repo.html || \
    echo "  DGL install failed — continuing. The node baseline that needs DGL will be skipped."

echo "=== 7. Installing Reporting / Utility Packages (driver only) ==="
python3 -m pip install --user xlsxwriter openpyxl matplotlib seaborn

# ──────────────────────────────────────────────────────────────────────────────
#  Worker nodes
# ──────────────────────────────────────────────────────────────────────────────
echo "=== 8. Provisioning ALL worker nodes ==="
export JAVA_HOME="${JAVA_HOME:-$(dirname "$(dirname "$(readlink -f "$(which java)")")")}"
export PYSPARK_PYTHON="$(which python3)"
export PYSPARK_DRIVER_PYTHON="$(which python3)"

if ! command -v spark-submit &> /dev/null; then
    echo "  spark-submit not found; skipping worker provisioning."
    echo "  Run it by hand once Spark is available:"
    echo "    spark-submit --master yarn --deploy-mode client $SCRIPT_DIR/emr_provision_workers.py"
else
    # A mapPartitions job with default-sized executors gets packed onto whatever
    # executors are free and can miss nodes entirely. Few very large executors
    # force YARN to spread them. Retry, because placement is not deterministic.
    ATTEMPTS=3
    PASSED=0
    for i in $(seq 1 $ATTEMPTS); do
        echo ""
        echo "  --- provisioning attempt $i/$ATTEMPTS ---"
        spark-submit --master yarn --deploy-mode client \
            "$SCRIPT_DIR/emr_provision_workers.py" || true

        echo ""
        echo "  --- preflight verification ---"
        if spark-submit --master yarn --deploy-mode client \
                "$SCRIPT_DIR/emr_preflight.py"; then
            PASSED=1
            break
        fi
        echo "  preflight did not pass; retrying."
    done

    if [ "$PASSED" -ne 1 ]; then
        echo ""
        echo "========================================================="
        echo " WORKER PREFLIGHT FAILED after $ATTEMPTS attempts."
        echo " Do NOT start a pipeline run: a single unprovisioned node"
        echo " aborts the whole job from inside a pandas UDF, typically"
        echo " deep into Phase 3b after Phase 3 has already succeeded."
        echo "========================================================="
        exit 1
    fi
fi

echo ""
echo "========================================================="
echo " EMR SETUP COMPLETED SUCCESSFULLY"
echo "========================================================="
echo "Driver packages installed to: $LARGE_TMP/.local"
echo "All worker nodes provisioned and verified."
echo ""
echo "To run the pipeline:"
echo "  1. Edit experiment_config.py to configure the experiment."
echo "  2. python3 runners/run_emr.py --datasets reddit"
echo ""
echo "To re-verify the cluster at any time (cheap, ~1 min):"
echo "  spark-submit --master yarn --deploy-mode client scripts/emr_preflight.py"
echo "========================================================="

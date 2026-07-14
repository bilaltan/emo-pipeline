#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  EMR Driver Node Setup Script
#  ► Run this script on the EMR Driver node after SSH connection.
#  ► This script installs required system libraries and Python dependencies.
#  ► Caches and local packages are redirected to large EBS volumes (e.g., /mnt)
#    to prevent root disk (15GB) OOM.
# ══════════════════════════════════════════════════════════════════════════════

set -e

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

# Export variables so that pip and setup tools write to the large disk instead of /home/hadoop (15GB root)
export TMPDIR="$LARGE_TMP"
export TEMP="$LARGE_TMP"
export TMP="$LARGE_TMP"
export PIP_CACHE_DIR="$LARGE_TMP/.pip-cache"
export PYTHONUSERBASE="$LARGE_TMP/.local"
export DGL_DOWNLOAD_DIR="$LARGE_TMP/.dgl"

# Dynamically construct local site-packages path to let python check packages instantly
PY_VERSION=$(python3 -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')")
export PYTHONPATH="$LARGE_TMP/.local/lib/$PY_VERSION/site-packages:$PYTHONPATH"

echo "=== 2. Installing System Packages & Development Tools ==="
# EMR 7.x uses Amazon Linux 2023, while EMR 6.x uses Amazon Linux 2.
if command -v dnf &> /dev/null; then
    sudo dnf groupinstall -y "Development Tools"
    sudo dnf install -y gcc-c++ make python3-devel git tmux
elif command -v yum &> /dev/null; then
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y gcc-c++ make python3-devel git tmux
else
    echo "Warning: Could not find dnf or yum packager. Please make sure development tools, git, and tmux are installed."
fi

echo "=== 3. Upgrading Pip ==="
python3 -m pip install --user --upgrade pip

echo "=== 4. Installing Core Python Packages ==="
# Install core numpy, pyarrow, and OGB
python3 -m pip install --user numpy pyarrow ogb boto3

# Install graph community detection algorithms (igraph, leidenalg)
python3 -m pip install --user igraph leidenalg

# Install ML frameworks (PyTorch and PyTorch Geometric)
python3 -m pip install --user torch torch-geometric

echo "=== 5. Installing DGL (Deep Graph Library) ==="
# EMO Pipeline is configured to use DGL 1.1.3 (CPU version by default on EMR worker/driver machines)
python3 -m pip install --user dgl==1.1.3 -f https://data.dgl.ai/wheels/repo.html

echo "=== 6. Installing Reporting / Utility Packages ==="
python3 -m pip install --user xlsxwriter openpyxl matplotlib seaborn

echo "========================================================="
echo " EMR DRIVER SETUP COMPLETED SUCCESSFULLY!"
echo "========================================================="
echo "All packages installed to large disk: $LARGE_TMP/.local"
echo "To run the pipeline:"
echo "  1. Edit experiment_config.py to configure your settings."
echo "  2. Run: python3 runners/run_emr.py"
echo "========================================================="

#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  EMR Driver Node Setup Script
#  ► Run this script on the EMR Driver node after SSH connection.
#  ► This script installs required system libraries and Python dependencies.
# ══════════════════════════════════════════════════════════════════════════════

set -e

echo "=== 1. Installing System Packages & Development Tools ==="
# EMR 7.x uses Amazon Linux 2023, while EMR 6.x uses Amazon Linux 2.
# Install build tools and system compilers required to compile C extensions (e.g., igraph / leidenalg)
if command -v dnf &> /dev/null; then
    sudo dnf groupinstall -y "Development Tools"
    sudo dnf install -y gcc-c++ make python3-devel git
elif command -v yum &> /dev/null; then
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y gcc-c++ make python3-devel git
else
    echo "Warning: Could not find dnf or yum packager. Please make sure gcc, g++, make, python3-dev, and git are installed."
fi

echo "=== 2. Upgrading Pip ==="
python3 -m pip install --user --upgrade pip

echo "=== 3. Installing Core Python Packages ==="
# Install core numpy, pyarrow, and OGB
python3 -m pip install --user numpy pyarrow ogb boto3

# Install graph community detection algorithms (igraph, leidenalg)
python3 -m pip install --user igraph leidenalg

# Install ML frameworks (PyTorch and PyTorch Geometric)
python3 -m pip install --user torch torch-geometric

echo "=== 4. Installing DGL (Deep Graph Library) ==="
# EMO Pipeline is configured to use DGL 1.1.3 (CPU version by default on EMR worker/driver machines)
python3 -m pip install --user dgl==1.1.3 -f https://data.dgl.ai/wheels/repo.html

echo "=== 5. Installing Reporting / Utility Packages ==="
python3 -m pip install --user xlsxwriter openpyxl matplotlib seaborn

echo "========================================================="
echo " EMR DRIVER SETUP COMPLETED SUCCESSFULLY!"
echo "========================================================="
echo "To run the pipeline:"
echo "  1. Edit experiment_config.py to configure your settings."
echo "  2. Run: python3 runners/run_emr.py"
echo "========================================================="

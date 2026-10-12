#!/usr/bin/env bash
# Run a command in the local LakeGRL Spark env (WSL, micromamba env `lakegrl`) from the repo root:
#   wsl.exe -d Ubuntu-24.04 -- bash scripts/wsl_run.sh python scripts/run_local_pipeline.py --dataset reddit
#
# The phases import the repo as the package `pipeline` (a symlink in git, which a Windows checkout turns into a
# one-line text file), so a real symlink is kept under $HOME/lakegrl/pkg and put on PYTHONPATH.
# Delta tables go to the WSL file system ($LAKEGRL_STORE): /mnt/c is several times slower for many small files.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
ENV=$HOME/mamba/envs/lakegrl
PKG=$HOME/lakegrl/pkg
mkdir -p "$PKG"
[ -L "$PKG/pipeline" ] || ln -sfn "$REPO" "$PKG/pipeline"
export JAVA_HOME=$ENV/lib/jvm
[ -x "$JAVA_HOME/bin/java" ] || export JAVA_HOME=$ENV
export PATH=$ENV/bin:$PATH
export PYSPARK_PYTHON=$ENV/bin/python
export PYSPARK_DRIVER_PYTHON=$ENV/bin/python
export PYTHONPATH=$PKG:$REPO${PYTHONPATH:+:$PYTHONPATH}
export LAKEGRL_STORE=${LAKEGRL_STORE:-$HOME/lakegrl/store}
export DATA_ROOT=${DATA_ROOT:-/mnt/c/data}
export OUTDATED_IGNORE=1   # ogb's version-check thread can deadlock imports
export PYTHONUNBUFFERED=1  # progress lines reach redirected logs as they happen
mkdir -p "$LAKEGRL_STORE"
exec "$@"

#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  Set up a single EC2 machine to run Phase 4, the full-graph baseline.
#
#  Deliberately NO Java, NO Spark, NO delta-spark. Phase 4 touches Spark in only
#  four places, all of them reads, and scripts/run_phase4_ec2.py satisfies them
#  with pyarrow. That keeps this box to a plain Python environment and lets the
#  baseline training code run unmodified, with no second copy to drift from what
#  the paper reports.
#
#  Sizing (Reddit: 232,965 nodes, 114,615,892 edges, 602 features)
#  ---------------------------------------------------------------
#  Measured, not estimated: RandomLinkSplit costs a flat 64.0 bytes per edge
#  across its train/val/test outputs (checked at 200k, 500k and 1M edges, and
#  it is exactly linear), so for Reddit:
#
#    features float32, twice (pandas + np.stack)   1.12 GB
#    raw edge_index int64 [2, 114.6M]              1.83 GB
#    RandomLinkSplit train+val+test                7.34 GB
#    pandas edges frame                            1.83 GB
#    ------------------------------------------------------
#    data subtotal                                12.13 GB
#
#  Add GATv2 attention coefficients, their gradients and optimiser state over
#  the training edges -- full-batch, no neighbour sampling -- and peak lands
#  around 19-36 GB.
#
#  So 64 GB is comfortable and 32 GB is too tight. m6i.4xlarge (16 vCPU/64 GB)
#  is the sweet spot; r6i.4xlarge (16 vCPU/128 GB) buys margin if you would
#  rather not think about it. Favour vCPUs over further RAM: this is full-batch
#  CPU training, so cores set the epoch time once the graph fits.
#
#  Use CPU, not a GPU: 114.6M edges will not fit comfortably in typical VRAM,
#  and the claim being measured is "one machine", not "one GPU".
#
#  Usage:  bash scripts/ec2_phase4_setup.sh
# ══════════════════════════════════════════════════════════════════════════════

set -e

echo "=== 1. System packages ==="
if command -v dnf &> /dev/null; then
    sudo dnf install -y gcc-c++ make python3-devel git
elif command -v yum &> /dev/null; then
    sudo yum install -y gcc-c++ make python3-devel git
elif command -v apt-get &> /dev/null; then
    sudo apt-get update -y && sudo apt-get install -y build-essential python3-dev git
else
    echo "  Unknown package manager; ensure a C++ toolchain and python3-dev exist."
fi

echo "=== 2. Pip ==="
python3 -m pip install --user --upgrade pip

echo "=== 3. Python packages (no Spark, no JVM) ==="
python3 -m pip install --user "numpy<2.0.0" pandas pyarrow scipy scikit-learn boto3
python3 -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install --user torch-geometric

echo "=== 4. Verifying ==="
python3 - <<'PY'
import sys
mods = ["numpy", "pandas", "pyarrow", "sklearn", "torch", "torch_geometric", "boto3"]
bad = []
for m in mods:
    try:
        mod = __import__(m)
        print("  OK   %-16s %s" % (m, getattr(mod, "__version__", "")))
    except Exception as e:
        bad.append(m)
        print("  FAIL %-16s %s" % (m, type(e).__name__))
try:
    import torch
    from torch_geometric.nn import GATv2Conv
    conv = GATv2Conv(4, 2, heads=1)
    x = torch.randn(3, 4)
    ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    conv(x, ei)
    print("  OK   GATv2 forward pass")
except Exception as e:
    bad.append("gatv2")
    print("  FAIL GATv2 forward pass: %s" % e)
sys.exit(1 if bad else 0)
PY

echo ""
echo "=== 5. AWS credentials ==="
python3 - <<'PY'
try:
    import boto3
    s3 = boto3.client("s3", region_name="us-east-1")
    r = s3.list_objects_v2(Bucket="us-east-1-s3-gnn",
                           Prefix="delta-data/reddit/nodes/", MaxKeys=1)
    print("  OK   S3 reachable (%d key(s) listed)" % r.get("KeyCount", 0))
except Exception as e:
    print("  FAIL S3 not reachable: %s: %s" % (type(e).__name__, str(e)[:120]))
    print("       Attach an instance role with read access to the bucket,")
    print("       or configure credentials, before running Phase 4.")
PY

echo ""
echo "========================================================="
echo " EC2 PHASE 4 SETUP COMPLETE"
echo "========================================================="
echo "Prove the path works first (10 epochs, minutes not hours):"
echo "  python3 scripts/run_phase4_ec2.py --dataset reddit --quick"
echo ""
echo "Then the real baseline:"
echo "  python3 scripts/run_phase4_ec2.py --dataset reddit"
echo ""
echo "Watch for this line - without it, retention compares different tasks:"
echo "  [partition metric] loaded N communities from louvain"
echo "========================================================="

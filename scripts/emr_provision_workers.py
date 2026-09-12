#!/usr/bin/env python3
"""
Install the pipeline's Python dependencies on every YARN worker node.

Why this exists
---------------
`emr_setup.sh` used to provision the driver only. Worker nodes could be missing
numpy and pandas entirely, and an Arrow-to-pandas conversion scheduled on such a
node dies inside Spark's pandas-UDF machinery *before* any user code runs — so
the in-UDF pip fallback the phases use for torch cannot rescue it. On 2026-09-11
one node of four was unprovisioned and aborted two Reddit runs, at 48 and 27
minutes, each time in Phase 3b and each time after Phase 3 had already produced
good results. The Spark error names the host of the last retry, so it cites
several different hosts and hides the single bad node.

Two details matter and are easy to get wrong:

1. Reaching every node. A mapPartitions probe with default-sized executors gets
   packed onto whatever executors are free — attempts on 2026-09-11 reached 1,
   then 2, then 4 of 4 nodes on successive runs. Requesting a few *very large*
   executors forces YARN to spread them, because two will not fit on one node.

2. Detecting what is installed. `import numpy` depends on
   spark.executorEnv.PYTHONPATH being set, so an import check reports failure on
   healthy nodes and masks the real state. Check the filesystem instead.

Usage
-----
    spark-submit --master yarn --deploy-mode client emr_provision_workers.py
    spark-submit ... emr_provision_workers.py --packages "numpy<2.0.0" pandas
"""
import argparse
import os
import socket
import subprocess
import sys

# Everything a Python worker imports inside a UDF. Driver-only packages
# (igraph, leidenalg, ogb, matplotlib, seaborn, xlsxwriter, openpyxl) are
# deliberately absent: they are installed by emr_setup.sh on the driver.
WORKER_PACKAGES = [
    "numpy<2.0.0",
    "pandas",
    "pyarrow",
    "scikit-learn",   # sklearn.metrics.roc_auc_score, called inside the UDFs
    "scipy",
    "boto3",
]

# Installed separately from the CPU wheel index, which is not on PyPI.
TORCH_PACKAGES = ["torch"]
TORCH_INDEX = "https://download.pytorch.org/whl/cpu"

# torch_geometric must land after torch.
PYG_PACKAGES = ["torch-geometric"]


def target_dir():
    return "/mnt/tmp/.local/lib/python%d.%d/site-packages" % (
        sys.version_info.major, sys.version_info.minor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packages", nargs="*", default=None,
                    help="override the worker package list")
    ap.add_argument("--executors", type=int, default=8,
                    help="few and large, so YARN must spread them across nodes")
    ap.add_argument("--executor-memory", default="100g",
                    help="large enough that two will not fit on one node")
    ap.add_argument("--skip-torch", action="store_true",
                    help="skip torch/PyG (much faster when only fixing numpy)")
    args = ap.parse_args()

    pkgs = args.packages if args.packages else WORKER_PACKAGES
    want_torch = not args.skip_torch

    from pyspark.sql import SparkSession
    spark = (SparkSession.builder
             .appName("provision-workers")
             .config("spark.executor.instances", str(args.executors))
             .config("spark.executor.cores", "2")
             .config("spark.executor.memory", args.executor_memory)
             .config("spark.executor.memoryOverhead", "8g")
             .config("spark.dynamicAllocation.enabled", "false")
             .config("spark.locality.wait", "0")
             .getOrCreate())
    sc = spark.sparkContext

    def provision(_):
        host = socket.gethostname()
        tgt = target_dir()
        # Filesystem check, not import: import success depends on PYTHONPATH.
        needed = [p for p in ("numpy", "pandas", "pyarrow", "sklearn", "scipy")
                  if not os.path.isdir(os.path.join(tgt, p))]
        if want_torch and not os.path.isdir(os.path.join(tgt, "torch")):
            needed.append("torch")
        if not needed:
            return [(host, 1, "already provisioned")]

        # One installer per host; peers on the same node stand down. The lock
        # lives on the node's own disk, so it is naturally per-host.
        lock = "/mnt/tmp/.provision.lock"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            return [(host, 0, "peer is installing")]
        except OSError as e:
            return [(host, 3, "LOCK FAILED: %s" % e)]

        try:
            os.makedirs(tgt, exist_ok=True)
            base = [sys.executable, "-m", "pip", "install", "--no-cache-dir",
                    "--target", tgt, "--upgrade"]
            steps = [("core", base + list(pkgs))]
            if want_torch:
                steps.append(("torch", base + TORCH_PACKAGES +
                              ["--index-url", TORCH_INDEX]))
                steps.append(("pyg", base + PYG_PACKAGES))
            for label, cmd in steps:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
                if r.returncode != 0:
                    return [(host, 3, "PIP FAILED (%s) rc=%d: %s"
                             % (label, r.returncode, (r.stderr or "")[-200:]))]
            ok = os.path.isdir(os.path.join(tgt, "numpy"))
            return [(host, 3, "INSTALLED %s (numpy=%s)" % (",".join(needed), ok))]
        except Exception as e:
            return [(host, 3, "ERROR %s: %s" % (type(e).__name__, str(e)[:160]))]
        finally:
            try:
                os.remove(lock)
            except OSError:
                pass

    rows = sc.parallelize(range(256), 256).mapPartitions(provision).collect()

    # Keep the most informative message per host: an INSTALLED/FAILED result
    # must not be overwritten by a peer's "standing down" note.
    best = {}
    for host, prio, msg in rows:
        if host not in best or prio > best[host][0]:
            best[host] = (prio, msg)

    print("\n=== WORKER PROVISIONING (%d hosts reached) ===" % len(best))
    failed = 0
    for host in sorted(best):
        msg = best[host][1]
        if "FAILED" in msg or "ERROR" in msg:
            failed += 1
        print("  %-34s %s" % (host, msg))
    spark.stop()

    if failed:
        print("\n%d host(s) failed to provision." % failed)
        return 1
    print("\nAll reached hosts are provisioned.")
    print("Run scripts/emr_preflight.py to verify before starting a pipeline run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

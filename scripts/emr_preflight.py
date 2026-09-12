#!/usr/bin/env python3
"""
Verify every YARN worker can actually run the pipeline's pandas UDFs.

Run this before any long pipeline run. It exits non-zero if a single node is
bad, which is far cheaper than discovering it 27 minutes into Phase 3b.

What it checks, and why each check is the one that matters
---------------------------------------------------------
* `to_pandas()` on a real Arrow table, not just `import numpy`. The failure that
  killed two Reddit runs on 2026-09-11 happened inside pyarrow's pandas shim,
  which is invoked by Spark's own machinery before user code runs. A bare import
  check can pass while this still fails.
* torch, torch_geometric and sklearn, all imported inside the Phase 3 / 3b UDFs.
  sklearn is easy to miss: it appears only as `roc_auc_score`.
* The resolved file path of each module, so a node loading the *system* package
  instead of the one under /mnt/tmp/.local is visible rather than silent.

Usage
-----
    spark-submit --master yarn --deploy-mode client emr_preflight.py
    spark-submit ... emr_preflight.py --require-dgl
"""
import argparse
import os
import socket
import sys

PY = "python%d.%d" % (sys.version_info.major, sys.version_info.minor)
EXECUTOR_PYTHONPATH = ":".join([
    "/mnt/tmp/.local/lib/%s/site-packages" % PY,
    "/mnt1/tmp/.local/lib/%s/site-packages" % PY,
    "/mnt2/tmp/.local/lib/%s/site-packages" % PY,
    "/tmp/.local/lib/%s/site-packages" % PY,
])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-dgl", action="store_true",
                    help="treat a missing dgl as a failure (it is optional by default)")
    ap.add_argument("--executors", type=int, default=8)
    ap.add_argument("--executor-memory", default="100g",
                    help="large executors force YARN to spread across all nodes")
    args = ap.parse_args()
    require_dgl = args.require_dgl

    from pyspark.sql import SparkSession
    spark = (SparkSession.builder
             .appName("emr-preflight")
             .config("spark.executor.instances", str(args.executors))
             .config("spark.executor.cores", "2")
             .config("spark.executor.memory", args.executor_memory)
             .config("spark.executor.memoryOverhead", "8g")
             .config("spark.dynamicAllocation.enabled", "false")
             .config("spark.locality.wait", "0")
             .config("spark.pyspark.python", "python3")
             .config("spark.executorEnv.HOME", "/tmp")
             .config("spark.executorEnv.PYTHONUSERBASE", "/tmp/.local")
             .config("spark.executorEnv.PYTHONPATH", EXECUTOR_PYTHONPATH)
             .config("spark.executorEnv.DGLBACKEND", "pytorch")
             .getOrCreate())
    sc = spark.sparkContext

    def check(_):
        host = socket.gethostname()
        report = {}

        def probe(name, importer):
            try:
                mod = importer()
                report[name] = getattr(mod, "__file__", "builtin") or "builtin"
            except Exception as e:
                report[name] = "MISSING:%s" % type(e).__name__

        probe("numpy", lambda: __import__("numpy"))
        probe("pandas", lambda: __import__("pandas"))
        probe("pyarrow", lambda: __import__("pyarrow"))
        probe("sklearn", lambda: __import__("sklearn.metrics", fromlist=["roc_auc_score"]))
        probe("torch", lambda: __import__("torch"))
        probe("torch_geometric", lambda: __import__("torch_geometric"))
        probe("dgl", lambda: __import__("dgl"))

        # The check that actually reproduces the historical failure.
        try:
            import pyarrow as pa
            pa.table({"a": [1, 2, 3]}).to_pandas()
            report["to_pandas"] = "OK"
        except Exception as e:
            report["to_pandas"] = "FAIL: %s" % str(e).replace("\n", " ")[:90]

        # A GATv2 forward pass is what Phase 3 actually does on a worker.
        try:
            import torch
            from torch_geometric.nn import GATv2Conv
            conv = GATv2Conv(4, 2, heads=1)
            x = torch.randn(3, 4)
            ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
            conv(x, ei)
            report["gatv2_forward"] = "OK"
        except Exception as e:
            report["gatv2_forward"] = "FAIL: %s" % str(e).replace("\n", " ")[:90]

        return [(host, report)]

    rows = sc.parallelize(range(512), 512).mapPartitions(check).collect()
    seen = {}
    for host, rep in rows:
        seen.setdefault(host, rep)
    spark.stop()

    required = ["numpy", "pandas", "pyarrow", "sklearn", "torch", "torch_geometric"]
    if require_dgl:
        required.append("dgl")

    print("\n=== PREFLIGHT: %d worker host(s) ===" % len(seen))
    bad_hosts = []
    for host in sorted(seen):
        rep = seen[host]
        problems = [k for k in required if rep.get(k, "").startswith("MISSING")]
        if rep.get("to_pandas") != "OK":
            problems.append("to_pandas")
        if rep.get("gatv2_forward") != "OK":
            problems.append("gatv2_forward")
        status = "OK" if not problems else "PROBLEM: " + ", ".join(problems)
        print("\n  %s  [%s]" % (host, status))
        for k in required + ["dgl", "to_pandas", "gatv2_forward"]:
            v = rep.get(k, "?")
            print("     %-16s %s" % (k, v[:78]))
        if problems:
            bad_hosts.append(host)

    print("\n" + "=" * 62)
    if not seen:
        print("NO HOSTS REACHED — cannot certify the cluster. Re-run with more")
        print("executors, or check that YARN has running node managers.")
        return 2
    if bad_hosts:
        print("PREFLIGHT FAILED on %d of %d host(s):" % (len(bad_hosts), len(seen)))
        for h in bad_hosts:
            print("   - %s" % h)
        print("\nFix with:")
        print("   spark-submit --master yarn --deploy-mode client \\")
        print("       scripts/emr_provision_workers.py")
        return 1
    print("PREFLIGHT PASSED on all %d host(s). Safe to start a pipeline run." % len(seen))
    return 0


if __name__ == "__main__":
    sys.exit(main())

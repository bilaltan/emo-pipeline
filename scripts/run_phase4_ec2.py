#!/usr/bin/env python3
"""
Run Phase 4 (the full-graph baseline) on a single EC2 machine, without Spark.

Why a separate machine
----------------------
Phase 4 exists to answer "what does one machine get on the whole graph?".
Running it on the EMR driver undercuts that claim and puts cluster runs at risk:
it trains full-batch with no neighbour sampling, and its cost on a 114.6M-edge
graph is unmeasured. On the cluster it is now disabled (RUN_PHASE4 = False).

Why no Spark here
-----------------
run_phase4h touches Spark in exactly four places -- read.format('delta').load()
on nodes/edges/masks, .orderBy(), .select() and .toPandas() -- and never uses
the SparkContext at all. That surface is small enough to satisfy with pyarrow,
so this box needs no JVM, no Spark and no delta-spark. More importantly, the
training code runs *unmodified*: there is no second copy of the baseline to
drift away from what the paper reports.

Delta correctness
-----------------
A Delta table is parquet plus a _delta_log, and Phase 0 writes with
mode("overwrite"), which leaves superseded parquet files in place until vacuum.
Globbing *.parquet would therefore mix stale rows with current ones and produce
silently wrong results. This resolves the log properly: the newest checkpoint
plus every later JSON commit, taking add minus remove.

Usage
-----
    python3 scripts/run_phase4_ec2.py --dataset reddit
    python3 scripts/run_phase4_ec2.py --dataset reddit --models sage,gatv2
    python3 scripts/run_phase4_ec2.py --dataset reddit --quick   # smoke test
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs


# ── Delta log resolution ──────────────────────────────────────────────────────

def _split_uri(uri):
    """('s3://bucket/a/b/', ...) -> (filesystem, 'bucket/a/b')."""
    if uri.startswith("s3://"):
        return pafs.S3FileSystem(region=os.environ.get("AWS_REGION", "us-east-1")), \
               uri[len("s3://"):].rstrip("/")
    return pafs.LocalFileSystem(), uri.replace("file://", "").rstrip("/")


def _read_text(fs, path):
    with fs.open_input_stream(path) as f:
        return f.readall().decode("utf-8")


def _active_files(fs, root):
    """Parquet files the Delta log says are live, newest checkpoint + commits."""
    log_dir = root + "/_delta_log"
    try:
        entries = fs.get_file_info(pafs.FileSelector(log_dir, recursive=False))
    except Exception as e:
        print("  ! no _delta_log at %s (%s); falling back to every parquet file."
              % (log_dir, type(e).__name__))
        return None

    commits, checkpoints = {}, {}
    for e in entries:
        base = os.path.basename(e.path)
        if base.endswith(".json") and base[:20].isdigit():
            commits[int(base[:20])] = e.path
        elif ".checkpoint." in base and base.endswith(".parquet"):
            checkpoints[int(base.split(".")[0])] = e.path

    adds, removes = set(), set()
    start = -1
    if checkpoints:
        cv = max(checkpoints)
        try:
            tbl = ds.dataset(checkpoints[cv], filesystem=fs, format="parquet").to_table()
            if "add" in tbl.column_names:
                for rec in tbl.column("add").to_pylist():
                    if rec and rec.get("path"):
                        adds.add(rec["path"])
            start = cv
        except Exception as e:
            print("  ! checkpoint %d unreadable (%s); replaying commits from 0."
                  % (cv, type(e).__name__))
            adds.clear()
            start = -1

    for v in sorted(k for k in commits if k > start):
        for line in _read_text(fs, commits[v]).splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if "add" in obj and obj["add"].get("path"):
                adds.add(obj["add"]["path"])
            if "remove" in obj and obj["remove"].get("path"):
                removes.add(obj["remove"]["path"])

    live = sorted(adds - removes)
    if not live:
        print("  ! Delta log resolved to zero files; falling back to every parquet file.")
        return None
    return [root + "/" + p for p in live]


# ── Minimal Spark-shaped reader ───────────────────────────────────────────────

class _Table:
    def __init__(self, fs, files, root, columns=None, order_by=None):
        self._fs, self._files, self._root = fs, files, root
        self._columns, self._order_by = columns, order_by

    def select(self, *cols):
        flat = []
        for c in cols:
            flat.extend(c if isinstance(c, (list, tuple)) else [c])
        return _Table(self._fs, self._files, self._root, flat, self._order_by)

    def orderBy(self, col):
        return _Table(self._fs, self._files, self._root, self._columns, col)

    def toPandas(self):
        if self._files is not None:
            dataset = ds.dataset(self._files, filesystem=self._fs, format="parquet")
        else:
            dataset = ds.dataset(self._root, filesystem=self._fs, format="parquet",
                                 ignore_prefixes=["_", "."])
        # Column projection matters: the communities read needs 2 of the node
        # table's columns, and one of the others is a 602-float vector.
        tbl = dataset.to_table(columns=self._columns, use_threads=True)
        pdf = tbl.to_pandas()
        if self._order_by:
            pdf = pdf.sort_values(self._order_by, kind="mergesort").reset_index(drop=True)
        return pdf


class _Reader:
    def __init__(self):
        self._fmt = "delta"

    def format(self, fmt):
        self._fmt = fmt
        return self

    def load(self, path):
        fs, root = _split_uri(path)
        t0 = time.time()
        files = _active_files(fs, root)
        n = len(files) if files is not None else -1
        print("  [delta] %s -> %s file(s) in %.1fs"
              % (path, n if n >= 0 else "all (fallback)", time.time() - t0))
        return _Table(fs, files, root)


class SparkShim:
    """Satisfies the four calls run_phase4/run_phase4h make. Nothing more."""
    @property
    def read(self):
        return _Reader()


# ── Runner ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="reddit")
    ap.add_argument("--s3-bucket", default="us-east-1-s3-gnn")
    ap.add_argument("--experiment-name", default="gatv2_reddit_edgecap")
    ap.add_argument("--algorithms", default="louvain",
                    help="used only to locate the communities table for the partition metric")
    ap.add_argument("--models", default="gatv2",
                    help="comma separated: gatv2, sage, or both")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--quick", action="store_true",
                    help="10 epochs, for proving the path works before a real run")
    ap.add_argument("--out", default="phase4_ec2_results.json")
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo)
    # On EMR the tree ships as pipeline.zip; the phases import pipeline.utils.*
    pkg = types.ModuleType("pipeline")
    pkg.__path__ = [repo]
    sys.modules["pipeline"] = pkg

    import importlib.util
    spec = importlib.util.spec_from_file_location("cfg", os.path.join(repo, "experiment_config.py"))
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)

    from pipeline.utils.paths import get_paths
    from pipeline.phases.phase4_baselines import run_phase4, run_phase4h

    baseline_cfg = dict(cfg_mod.BASELINE_CFG)
    if args.quick:
        baseline_cfg.update({"epochs": 10, "node_epochs": 10,
                             "link_epochs": 10, "link_patience": 10})
        print("  [quick] epochs capped at 10 - for proving the path, not for results")

    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    def get_paths_fn(dataset, alg=None):
        return get_paths(dataset, alg,
                         experiment_name=args.experiment_name,
                         s3_bucket=args.s3_bucket)

    spark = SparkShim()
    timing, results = {}, {}

    print("=" * 66)
    print("  PHASE 4 on a single machine (no Spark)")
    print("  dataset=%s  models=%s  algorithms=%s" % (args.dataset, models, algorithms))
    print("=" * 66)

    t0 = time.time()
    for model in models:
        fn = {"gatv2": run_phase4h, "sage": run_phase4}.get(model)
        if fn is None:
            print("  ! unknown model %r, skipping" % model)
            continue
        print("\n--- %s baseline ---" % model)
        try:
            fn(spark, None,
               datasets=[args.dataset],
               dataset_cfg=cfg_mod.DATASET_CFG,
               baseline_cfg=baseline_cfg,
               get_paths_fn=get_paths_fn,
               timing=timing,
               results=results,
               task_type=getattr(cfg_mod, "TASK_TYPE", "both"),
               algorithms=algorithms,
               n_baseline_runs=args.runs)
        except Exception as e:
            import traceback
            print("  [%s] FAILED: %s: %s" % (model, type(e).__name__, str(e)[:300]))
            traceback.print_exc()

    elapsed = time.time() - t0
    out = {
        "dataset": args.dataset,
        "experiment_name": args.experiment_name,
        "models": models,
        "quick": args.quick,
        "wall_seconds": round(elapsed, 1),
        "results": {k: {kk: vv for kk, vv in v.items()
                        if isinstance(vv, (int, float, str, type(None), list))}
                    for k, v in results.items()},
        "timing": {str(k): v for k, v in timing.items()},
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 66)
    print("  wall clock: %.1fs (%dm %ds)" % (elapsed, elapsed // 60, elapsed % 60))
    for dsname, r in results.items():
        print("  %s:" % dsname)
        print("     node acc            : %s" % r.get("test_acc"))
        print("     link AUC (global)   : %s" % r.get("link_auc_global", r.get("link_auc")))
        print("     link AUC (partition): %s" % r.get("link_auc_partition"))
        if r.get("link_auc_partition") is None:
            print("     ^ partition metric did NOT run: the communities table was")
            print("       unreadable, so retention would compare different tasks.")
    print("  wrote %s" % args.out)
    print("=" * 66)
    return 0


if __name__ == "__main__":
    sys.exit(main())

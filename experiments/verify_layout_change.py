#!/usr/bin/env python3
"""
End-to-end check of the repartitionByRange change in Phase 2.

Runs Phase 1 and Phase 2 with the CURRENT code, then answers two questions
against the tables the SUBMITTED code produced for the same dataset:

    1. Is the content identical?  Per-community row counts must match exactly;
       a layout change that alters data is a bug, not an optimisation.
    2. Did skipping improve?  Bytes that must be read per training unit, from
       parquet row-group statistics, for the tables as each version wrote them.

Usage
-----
    python experiments/verify_layout_change.py --dataset ogbn-mag --alg louvain
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zorder_bench import build_spark, dir_stats, log_stats  # noqa: E402
from community_layout_bench import rowgroup_index, bytes_for, real_load  # noqa: E402

SUBMITTED = "/Users/bilaltan/Desktop/emo-submitted"


def measure(spark, path, comms, sample):
    idx = rowgroup_index(path)
    per = [bytes_for(idx, c) for c in comms]
    whole = sum(s for _f, groups in idx for _lo, _hi, s in groups)
    files, _disk = dir_stats(path)
    files_active, bytes_active = log_stats(spark, path)
    times = [real_load(path, c)[0] for c in sample]
    return {
        "files": files_active,
        "mb": round(bytes_active / 1e6, 2),
        "read_bytes_median": int(statistics.median([b for b, _ in per])),
        "read_bytes_total": sum(b for b, _ in per),
        "skip_fraction": round(1 - statistics.median([b for b, _ in per]) / max(1, whole), 4),
        "load_seconds_median": round(statistics.median(times), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ogbn-mag")
    ap.add_argument("--alg", default="louvain")
    ap.add_argument("--table", default="p2_nodes", choices=["p2_nodes", "p2_edges"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_data_dir = os.path.join(repo, "local_data")
    os.makedirs(local_data_dir, exist_ok=True)
    sys.path.insert(0, repo)

    spark = build_spark(repo, local_data_dir, "local[*]")
    from pipeline.utils.paths import get_paths
    import experiment_config as config

    gp = lambda d, a=None: get_paths(d, a, local_data_dir=local_data_dir)
    p_alg = gp(args.dataset, args.alg)

    # The raw delta tables are version-independent (the phase 0 diff since the
    # submission only touches Deezer paths), so reuse the ones already ingested
    # rather than paying for a second ingest.
    submitted_data = os.path.join(SUBMITTED, "local_data")
    if not os.path.exists(gp(args.dataset)["nodes"]):
        # Link ONLY the raw tables. Linking the dataset root would share the
        # phase 2 outputs as well, and the comparison would silently be a table
        # against itself.
        root = os.path.join(local_data_dir, "delta-data", args.dataset)
        os.makedirs(root, exist_ok=True)
        for sub in ("nodes", "edges", "masks", "original_nodes", "original_edges"):
            src = os.path.join(submitted_data, "delta-data", args.dataset, sub)
            dst = os.path.join(root, sub)
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(src, dst)

    if not os.path.exists(p_alg[args.table]):
        from pipeline.phases import run_phase1, run_phase2
        print("  running phase 1 and 2 with the CURRENT code ...")
        run_phase1(spark, spark.sparkContext,
                   datasets=[args.dataset], algorithms=[args.alg],
                   lpa_max_iter=config.LPA_MAX_ITER,
                   resolution=getattr(config, "RESOLUTION", 1.0),
                   random_seed=config.RANDOM_SEED,
                   min_size=config.MIN_COMMUNITY_SIZE,
                   dataset_cfg=config.DATASET_CFG,
                   get_paths_fn=gp, timing={}, results={},
                   metis_k=getattr(config, "METIS_K", 100), force_rerun=False)
        run_phase2(spark, spark.sparkContext,
                   datasets=[args.dataset], algorithms=[args.alg],
                   use_global_mapping=config.USE_GLOBAL_MAPPING,
                   min_size=config.MIN_COMMUNITY_SIZE,
                   get_paths_fn=gp, timing={}, results={},
                   tiny_comm_handling=getattr(config, "TINY_COMM_HANDLING", "misc"),
                   expand_boundary_nodes=getattr(config, "EXPAND_BOUNDARY_NODES", True),
                   force_rerun=False)

    old_path = get_paths(args.dataset, args.alg,
                         local_data_dir=submitted_data)[args.table]
    new_path = p_alg[args.table]

    def counts(path):
        df = spark.read.format("delta").load(path)
        return {int(r["community_id"]): int(r["count"]) for r in
                df.groupBy("community_id").count().collect()}

    old_counts, new_counts = counts(old_path), counts(new_path)
    same = old_counts == new_counts
    print("\n  per-community row counts identical: %s  (%d vs %d communities)"
          % (same, len(old_counts), len(new_counts)))
    if not same:
        only_old = set(old_counts) - set(new_counts)
        only_new = set(new_counts) - set(old_counts)
        diff = {c: (old_counts[c], new_counts[c]) for c in
                set(old_counts) & set(new_counts) if old_counts[c] != new_counts[c]}
        print("    only in submitted: %s" % sorted(only_old)[:5])
        print("    only in current:   %s" % sorted(only_new)[:5])
        print("    differing counts:  %s" % list(diff.items())[:5])

    comms = sorted(new_counts)
    sample = comms[:: max(1, len(comms) // 34)][:34]
    before = measure(spark, old_path, comms, sample)
    after = measure(spark, new_path, comms, sample)

    print("\n  %-22s %18s %18s" % ("", "submitted (hash)", "current (range)"))
    for k in ("files", "mb", "read_bytes_median", "read_bytes_total",
              "skip_fraction", "load_seconds_median"):
        print("  %-22s %18s %18s" % (k, f"{before[k]:,}", f"{after[k]:,}"))
    if after["read_bytes_median"]:
        print("\n  per-unit read reduced %.1fx"
              % (before["read_bytes_median"] / after["read_bytes_median"]))

    doc = {"dataset": args.dataset, "algorithm": args.alg, "table": args.table,
           "communities": len(comms), "content_identical": same,
           "submitted_hash": before, "current_range": after}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(doc, f, indent=2)
        print("\nwrote %s" % args.out)
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

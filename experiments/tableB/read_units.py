#!/usr/bin/env python3
"""
Table B, rows 3-5: the storage read of a pipeline run, cold, through the S3 emulator (advisor 2026-11-02). Reads the
exact bins the run's Stage 3a/3b trained (run_local_pipeline.py --work-out) from its Phase 2 tables with Table A's
reader (experiments/tableA/read_layouts.py: Stage 3b's loader, --procs reader processes, page cache dropped, one
batched read per bin, or one read per (community, block) in a bin with a split community). Reads only.
Run it right after the pipeline run: the next run on the same store rewrites the Phase 2 tables.

    bash scripts/wsl_run.sh python experiments/tableB/read_units.py --work results/tableB/work/ogbn-arxiv_row5.json --row 5
"""
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "tableA"))

from read_layouts import drop_caches, run, spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="bins saved by run_local_pipeline.py --work-out")
    ap.add_argument("--row", required=True, help="Table B row label")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--procs", type=int, default=20, help="reader processes (Table A: Stage 3b's 20 task slots)")
    ap.add_argument("--emu", default="20,80", help="utils.store_io.parse_emu: 'LAT_MS,MBPS' per GET, or 'count'")
    ap.add_argument("--out", default="results/tableB/read_units.jsonl")
    args = ap.parse_args()
    from pipeline.utils.store_io import parse_emu, to_path
    emu = parse_emu(args.emu)
    w = json.load(open(args.work))
    lay = {"nodes": spec(to_path(w["p2_nodes"]), emu), "edges": spec(to_path(w["p2_edges"]), emu),
           "comm_to_part": {int(k): v for k, v in w["comm_to_part"].items()} or None}
    lay["layout"] = {None: "range", "community_id": "hive", "storage_part": "hive_merged"}[lay["nodes"]["part_col"]]
    bins = [[tuple(r) for r in b] for b in w["bins"]]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"== {args.work}: {len(bins):,} bins, {sum(map(len, bins)):,} work items, {args.procs} readers, "
          f"GET = {args.emu}", flush=True)
    for seed in range(args.runs):
        drop_caches()
        wall, gets, nbytes, check = run(lay, None, bins, args.procs)
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "dataset": args.dataset, "row": args.row,
               "work": args.work, "layout": lay["layout"], "seed": seed, "cold": True, "emu": args.emu,
               "procs": args.procs, "wall_s": wall, "bins": len(bins), "gets": gets, "bytes": nbytes,
               "items": len(check), "node_rows": sum(v[0] for _, v in check),
               "edge_rows": sum(v[2] for _, v in check),
               "check_md5": hashlib.md5(json.dumps(check).encode()).hexdigest()}
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  run {seed}: {wall:8.1f} s {gets:>9,} GETs {nbytes / 1e9:8.2f} GB | {rec['node_rows']:,} node rows, "
              f"{rec['edge_rows']:,} edges | {rec['check_md5'][:8]}", flush=True)


if __name__ == "__main__":
    main()

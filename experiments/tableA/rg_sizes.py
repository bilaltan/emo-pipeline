#!/usr/bin/env python3
"""
Row-group sizes of written layouts (e.g. the actual row groups of the Z-order and Liquid layouts).
Reads only the parquet footers of each table's current Delta snapshot (no data pages, nothing written to the tables).
One record per (layout, table): files, row groups, rows and compressed MB per row group (min / median / mean / max).

    bash scripts/wsl_run.sh python experiments/tableA/rg_sizes.py --dirs ~/lakegrl/tableA/rleiden_c10000_v2/ogbn-products \
        --layouts zorder liquid merged_blk --out results/layout_checks/rg_sizes.jsonl
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def rg_stats(table):
    import pyarrow.parquet as pq
    from pipeline.utils.store_io import snapshot
    fs, files, version = snapshot(table)
    rows, mb = [], []
    for f in files:
        md = pq.ParquetFile(f, filesystem=fs).metadata
        for i in range(md.num_row_groups):
            rg = md.row_group(i)
            rows.append(rg.num_rows)
            mb.append(sum(rg.column(j).total_compressed_size for j in range(rg.num_columns)) / (1 << 20))
    q = lambda v: {"min": float(np.min(v)), "median": float(np.median(v)), "mean": float(np.mean(v)),  # noqa: E731
                   "max": float(np.max(v))} if v else {}
    return {"delta_version": version, "files": len(files), "row_groups": len(rows), "rows_per_rg": q(rows),
            "compressed_mb_per_rg": q(mb), "compressed_mb_total": float(np.sum(mb))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True, help="<out-dir>/<dataset> directories holding the layouts")
    ap.add_argument("--layouts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for d in args.dirs:
        for layout in args.layouts:
            for t in ("p2_nodes", "p2_edges"):
                table = os.path.join(d, layout, t)
                if not os.path.isdir(table):
                    continue
                rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "dir": d, "layout": layout, "table": t,
                       **rg_stats(table)}
                with open(args.out, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                m, r = rec["compressed_mb_per_rg"], rec["rows_per_rg"]
                print(f"{layout:12s} {t:9s} {rec['files']:5,} files {rec['row_groups']:6,} RGs | MB/RG median "
                      f"{m['median']:8.2f} (min {m['min']:.2f}, max {m['max']:.2f}) | rows/RG median "
                      f"{r['median']:12,.0f} (max {r['max']:,.0f})", flush=True)


if __name__ == "__main__":
    main()

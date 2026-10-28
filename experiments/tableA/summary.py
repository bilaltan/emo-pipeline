#!/usr/bin/env python3
"""
Table A from write_layouts.py and read_layouts.py records: one 8 x 4 markdown table per dataset (latest write run,
latest --runs reads per layout). Times are mean +/- sample std over the runs. Read amplification = bytes read /
minimum bytes, minimum = node rows read x compressed bytes per row of mv's p2_nodes (all columns, as Stage 3b reads
them) + edge rows read x the same for p2_edges. Fails if the per-item checksums differ across layouts.

    bash scripts/wsl_run.sh python experiments/tableA/summary.py --datasets ogbn-arxiv reddit ogbn-products
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
ROWS = [("join", "1. Global join"), ("mv", "2. Default MV"), ("hive", "3. Naive partitionBy"), ("zorder", "4. Z-order"),
        ("liquid", "5. Liquid clustering"), ("merged", "6. Ours: merging"), ("merged_rg", "7. Ours: + sqrt row group"),
        ("merged_blk", "8. Ours: + block order")]


def ms(v, fmt="{:.1f}"):
    v = np.asarray(v, dtype=float)
    return (fmt + " ± " + fmt).format(v.mean(), v.std(ddof=1)) if len(v) > 1 else fmt.format(v.mean())


def bytes_per_row(table):
    import pyarrow.parquet as pq
    from pipeline.utils.store_io import snapshot
    fs, files, _ = snapshot(table)
    tot = rows = 0
    for f in files:
        md = pq.ParquetFile(f, filesystem=fs).metadata
        rows += md.num_rows
        tot += sum(md.row_group(i).column(j).total_compressed_size
                   for i in range(md.num_row_groups) for j in range(md.num_columns))
    return tot / max(1, rows)


def table(ds, writes, reads, out_dir, runs):
    w = [r for r in writes if r["dataset"] == ds and r["kind"] == "layout"]
    p2 = [r for r in writes if r["dataset"] == ds and r["kind"] == "phase2"]
    if not w:
        return f"**{ds}**: no write records\n"
    rid = max(r["run_id"] for r in w)
    w = {r["layout"]: r for r in w if r["run_id"] == rid}
    p2 = next((r for r in p2 if r["run_id"] == rid), {})
    rd = {}
    for r in reads:
        if r["dataset"] == ds:
            rd.setdefault(r["layout"], []).append(r)
    rd = {k: v[-runs:] for k, v in rd.items()}
    mv = os.path.join(out_dir, ds, "mv")
    bpr = (bytes_per_row(f"{mv}/p2_nodes"), bytes_per_row(f"{mv}/p2_edges")) if os.path.isdir(mv) else None
    md5s = {r["check_md5"] for v in rd.values() for r in v}
    assert len(md5s) <= 1, f"{ds}: work items read different rows across layouts: {md5s}"
    lines = [f"**{ds}** ({p2.get('communities', '?'):,} communities, {p2.get('minor', '?'):,} minor; "
             f"{p2.get('work_items', '?'):,} work items at cap {p2.get('cap', '?'):,}; Phase 2 "
             f"{p2.get('phase2_s') or 0:.0f} s; {p2.get('spark_executors', '?')} executor x "
             f"{p2.get('spark_executor_cores', '?')} cores)", "",
             "| Layout | Write time (s) | Storage (GB), files (< 1 MB) | Bytes read (GB), amplification | "
             "Stage 3b read time (s) |", "| --- | --- | --- | --- | --- |"]
    for key, name in ROWS:
        cells = [name]
        if key in w:
            st = w[key]["storage"].values()
            cells += [ms(w[key]["write_s"]),
                      f"{sum(s['bytes'] for s in st) / 1e9:.2f}, {sum(s['files'] for s in st):,} "
                      f"({sum(s['files_lt_1mb'] for s in st):,})"]
        else:
            cells += ["", ""]
        if key in rd:
            v = rd[key]
            assert len({r["bytes"] for r in v}) == 1, f"{ds}/{key}: bytes differ across runs"
            amp = ""
            if bpr:
                need = v[0]["node_rows"] * bpr[0] + v[0]["edge_rows"] * bpr[1]
                amp = f", {v[0]['bytes'] / need:.1f}x"
            cells += [f"{v[0]['bytes'] / 1e9:.2f}{amp}", ms([r["wall_s"] for r in v]) + f" (n={len(v)}, {v[-1]['procs']} readers)"]
        else:
            cells += ["", ""]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["ogbn-arxiv", "reddit", "ogbn-products"])
    ap.add_argument("--writes", default="results/tableA/write.jsonl")
    ap.add_argument("--reads", default="results/tableA/read.jsonl")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out-dir", default=os.path.expanduser("~/lakegrl/tableA"))
    ap.add_argument("--md", default="results/tableA/tableA.md")
    args = ap.parse_args()
    load = lambda p: [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []  # noqa: E731
    writes, reads = load(args.writes), load(args.reads)
    md = "\n".join(table(ds, writes, reads, args.out_dir, args.runs) for ds in args.datasets)
    open(args.md, "w").write(md)
    print(md)


if __name__ == "__main__":
    main()

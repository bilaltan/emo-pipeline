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
        ("merged_blk", "8. Ours: + block order"),
        ("blk_128", "Diagnostic: 8 with 128 MB row groups")]


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


def table(ds, writes, reads, out_dir, runs, out=None):
    """One dataset's markdown table; with out (a list), also appends one flat record per layout (--csv)."""
    w = [r for r in writes if r["dataset"] == ds and r["kind"] == "layout"]
    p2 = [r for r in writes if r["dataset"] == ds and r["kind"] == "phase2"]
    if not w:
        return f"**{ds}**: no write records\n"
    w = {r["layout"]: r for r in w}     # latest write per layout: a resumed run rewrites only the missing layouts
    p2 = p2[-1] if p2 else {}
    rd = {}
    for r in reads:
        if r["dataset"] == ds:
            rd.setdefault(r["layout"], []).append(r)
    rd = {k: v[-runs:] for k, v in rd.items()}
    mv = os.path.join(out_dir, ds, "mv")
    bpr = (bytes_per_row(f"{mv}/p2_nodes"), bytes_per_row(f"{mv}/p2_edges")) if os.path.isdir(mv) else None
    md5s = {r["check_md5"] for v in rd.values() for r in v if r.get("bin_sample", 1) == 1}
    assert len(md5s) <= 1, f"{ds}: work items read different rows across layouts: {md5s}"
    lines = [f"**{ds}** ({p2.get('communities', '?'):,} communities, {p2.get('minor', '?'):,} minor; "
             f"{p2.get('work_items', '?'):,} work items at cap {p2.get('cap', '?'):,}; Phase 2 "
             f"{p2.get('phase2_s') or 0:.0f} s; {p2.get('spark_executors', '?')} executor x "
             f"{p2.get('spark_executor_cores', '?')} cores)", "",
             "| Layout | Write time (s) | Storage (GB), files (< 1 MB) | Bytes read (GB), amplification | "
             "Stage 3b read time (s) |", "| --- | --- | --- | --- | --- |"]
    for key, name in ROWS:
        if key == "blk_128" and key not in w and key not in rd:   # diagnostic row: arxiv and Reddit only
            continue
        cells = [name]
        rec = {"dataset": ds, "row": name, "layout": key, "cap": p2.get("cap"), "communities": p2.get("communities"),
               "work_items": p2.get("work_items")}
        if key in w:
            st = w[key]["storage"].values()
            cells += [ms(w[key]["write_s"]),
                      f"{sum(s['bytes'] for s in st) / 1e9:.2f}, {sum(s['files'] for s in st):,} "
                      f"({sum(s['files_lt_1mb'] for s in st):,})"]
            ws = np.asarray(w[key]["write_s"], dtype=float)
            rec.update(write_s_mean=ws.mean(), write_s_std=ws.std(ddof=1) if len(ws) > 1 else 0.0,
                       storage_gb=sum(s["bytes"] for s in st) / 1e9, files=sum(s["files"] for s in st),
                       files_lt_1mb=sum(s["files_lt_1mb"] for s in st), rg_mb=w[key].get("rg_mb"))
        else:
            cells += ["", ""]
        if key in rd:
            v = rd[key]
            k = v[-1].get("bin_sample", 1)   # every k-th bin read: bytes and time scaled up by k, marked est.
            nb = [r["bytes"] for r in v]
            # which reader gets which bin changes per run, and with it a few footer GETs (Reddit: < 0.01 %)
            assert max(nb) - min(nb) <= 1e-3 * max(nb), f"{ds}/{key}: bytes differ across runs: {nb}"
            nb = float(np.mean(nb))
            amp = ""
            if bpr:
                need = v[0]["node_rows"] * bpr[0] + v[0]["edge_rows"] * bpr[1]
                amp = f", {nb / need:.1f}x"
            est = f", est. from 1/{k} of the bins" if k > 1 else ""
            cells += [f"{nb * k / 1e9:.2f}{amp}{' est.' if k > 1 else ''}",
                      ms([r["wall_s"] * k for r in v]) + f" (n={len(v)}, {v[-1]['procs']} readers{est})"]
            rt = np.asarray([r["wall_s"] * k for r in v])
            rec.update(bytes_gb=nb * k / 1e9, amplification=nb / need if bpr else None, read_s_mean=rt.mean(),
                       read_s_std=rt.std(ddof=1) if len(rt) > 1 else 0.0, read_runs=len(v), readers=v[-1]["procs"],
                       bin_sample=k, gets=float(np.mean([r["gets"] for r in v])) * k, node_rows=v[0]["node_rows"],
                       edge_rows=v[0]["edge_rows"], check_md5=v[0]["check_md5"], emu=v[0]["emu"])
        else:
            cells += ["", ""]
        lines.append("| " + " | ".join(cells) + " |")
        if out is not None:
            out.append(rec)
    return "\n".join(lines) + "\n"


def halo_table(halo, auc):
    """Halo sub-table: halo_table.py's storage and cold read (latest per dataset and variant) with the link-prediction harness's M2
    cross-community link AUC over seeds (harness.py records, matched to the variant by halo_rows)."""
    lines = ["| Dataset | Halo | Units | Halo rows | Stored (GB) | Read (GB), GETs | AUC cross, global negatives | "
             "AUC cross, matched negatives | Seeds |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in {(r["dataset"], r["halo"]): r for r in halo}.values():
        a = [x for x in auc if x["dataset"] == r["dataset"] and x["halo_rows"] == r["halo_rows"]]
        a = list({x["seed"]: x for x in a}.values())
        name = "uncapped" if r["halo"] == "uncapped" else f"capped (R = {r['budget']:g})"
        lines.append(f"| {r['dataset']} | {name} | {r['units']:,} | {r['halo_rows']:,} | {r['bytes_stored'] / 1e9:.2f} | "
                     f"{r['bytes_read'] / 1e9:.2f}, {r['gets']:,} | "
                     f"{ms([x['auc_global_cross'] for x in a], '{:.4f}') if a else ''} | "
                     f"{ms([x['auc_matched_cross'] for x in a], '{:.4f}') if a else ''} | "
                     f"{','.join(str(x['seed']) for x in sorted(a, key=lambda x: x['seed']))} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["ogbn-arxiv", "reddit", "ogbn-products"])
    ap.add_argument("--writes", default="results/tableA/write.jsonl")
    ap.add_argument("--reads", default="results/tableA/read.jsonl")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out-dir", default=os.path.expanduser("~/lakegrl/tableA"))
    ap.add_argument("--md", default="results/tableA/tableA.md")
    ap.add_argument("--csv", help="also write one row per (dataset, layout), for comparing run sets")
    ap.add_argument("--halo", nargs=2, metavar=("HALO_JSONL", "AUC_JSONL"), help="also the halo sub-table")
    args = ap.parse_args()
    load = lambda p: [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []  # noqa: E731
    writes, reads = load(args.writes), load(args.reads)
    rows = []
    md = "\n".join(table(ds, writes, reads, args.out_dir, args.runs, rows) for ds in args.datasets)
    if args.halo:
        md += "\n**Halo sub-table**\n\n" + halo_table(load(args.halo[0]), load(args.halo[1]))
    open(args.md, "w").write(md)
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            cw = csv.DictWriter(fh, list(dict.fromkeys(k for r in rows for k in r)))
            cw.writeheader()
            cw.writerows(rows)
    print(md)


if __name__ == "__main__":
    main()

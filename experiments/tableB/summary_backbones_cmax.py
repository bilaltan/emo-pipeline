#!/usr/bin/env python3
"""
Tables of the two 2026-11-19 runs (advisor), from their records; mean ± sample std over seeds 0-2 (or 3 cold runs),
peak GPU = peak.sh (nvidia-smi minus idle), the largest over the seeds, as Table B (Table 4).

  backbones   results/backbones/<gcn|gatv2>/: baselines.jsonl, pipeline.jsonl, peak.jsonl (experiments/tableB/backbones.sh)
              + the GraphSAGE rows of Table 4 (results/tableB) for reference
  cmax        results/cmax_sweep/c<C>/ (run.sh TB_CMAX=C, row 5) + storage.jsonl (store_size.py); C_max = 10,000 is
              Table 4 row 5 on ogbn-products: results/tableB pipeline / peak / read_units, global read results/tableB/v4

    python experiments/tableB/summary_backbones_cmax.py      # writes results/backbones/SUMMARY.md, results/cmax_sweep/SUMMARY.md
"""
import json
import os
import statistics as st

R = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
DS = ("ogbn-arxiv", "reddit", "ogbn-products")


def load(*p):
    f = os.path.join(R, *p)
    return [json.loads(l) for l in open(f)] if os.path.exists(f) else []


def ms(v, d=1, k=1.0):
    v = [x * k for x in v]
    return "--" if not v else f"{st.mean(v):.{d}f}" if len(v) == 1 else f"{st.mean(v):.{d}f} ± {st.stdev(v):.{d}f}"


def row5(recs, ds):   # Table B row 5: recursive Leiden units, tiny communities dropped (Stage 3b)
    return [r for r in recs if r["dataset"] == ds and r["unit_split"] == "leiden" and r["tiny_comm"] == "drop"]


def s3_wall(r):
    return next(v for k, v in r["phase_seconds"].items() if k.startswith("('phase3b'"))


def peaks(d):
    return {r["label"]: r for r in load(*d, "peak.jsonl")}   # a label run twice: the last record, as Table 4


def gpu_gb(pk, ds):
    v = [pk[f"{ds} row5 seed{s}"]["peak_gpu_mb"] for s in range(3) if f"{ds} row5 seed{s}" in pk]
    return f"{max(v) / 1000:.1f}" if v else "--"


def backbones():
    L = ["| Dataset | Backbone | Full-graph Test Acc (%) | Full-graph Peak GPU (GB) | LakeGRL Test Acc (%) | "
         "LakeGRL Stage 3 Train (s) | LakeGRL Peak GPU (GB) | Seeds (full / LakeGRL) |", "|---" * 8 + "|"]
    for ds in DS:
        for m in ("sage", "gcn", "gatv2"):
            d = ("tableB",) if m == "sage" else ("backbones", m)
            pk = peaks(d)
            base = [r for r in load(*d, "baselines.jsonl") if r["dataset"] == ds and r["baseline"] == f"full_graph_{m}"]
            full = pk.get(f"{ds} full", {})
            oom = full.get("gpu_full") == 1
            # Table 4 = the fe671e8 runs (results/MANIFEST: the earlier ogbn-arxiv runs at 4e86451 are not in it)
            pipe = [r for r in row5(load(*d, "pipeline.jsonl"), ds) if r["model"] == m and (m != "sage" or r["git"] == "fe671e8")]
            if m != "sage" and not (base or full or pipe):
                continue
            killed = any(pk.get(f"{ds} row5 seed{s}", {}).get("gpu_full") == 1 for s in range(3))
            name = {"sage": "GraphSAGE (Table 4)", "gcn": "GCN", "gatv2": "GATv2"}[m]
            full_gpu = "OOM (> 23.5)" if oom else f"{full['peak_gpu_mb'] / 1000:.1f}" if full else "--"
            L.append(f"| {ds} | {name} | {'OOM' if oom else ms([r['test_acc'] for r in base], 2, 100)} | {full_gpu} | "
                     f"{'OOM' if killed else ms([r['stage3b_node_acc'] for r in pipe], 2, 100)} | "
                     f"{'--' if killed else ms([s3_wall(r) for r in pipe])} | "
                     f"{'OOM (> 23.5)' if killed else gpu_gb(pk, ds)} | {len(base)} / {len(pipe)} |")
    return L


def cmax():
    L = ["| C_max | Major units | Stored MV_unit (GB) | Cold Read (s): units + global | Stage 3 Train (s) | "
         "Peak GPU (GB) | Test Acc (%) |", "|---" * 7 + "|"]
    ds = "ogbn-products"
    size = {r["tag"]: r for r in load("cmax_sweep", "storage.jsonl")}   # last record per tag
    for c in (5000, 10000, 20000, 40000):
        d = ("tableB",) if c == 10000 else ("cmax_sweep", f"c{c}")
        pipe = [r for r in row5(load(*d, "pipeline.jsonl"), ds) if r["max_nodes"] == c and r["model"] == "sage"]
        units = [r["wall_s"] for r in load(*d, "read_units.jsonl") if r["dataset"] == ds and r["row"] == "5"]
        glob = [r["wall_s"] for r in load(*(("tableB", "v4") if c == 10000 else d), "read_full.jsonl")
                if r["dataset"] == ds and r["row"] == "5 global"]
        major = ", ".join(f"{u:,}" for u in sorted({r["n_units_stage3b"] - 1 for r in pipe})) or "--"  # + 1 global model
        sz = f"{size[f'leiden_c{c}']['mv_unit_bytes'] / 1e9:.2f}" if f"leiden_c{c}" in size else "--"
        read = f"{st.mean(units) + st.mean(glob):.1f} ({ms(units)} + {ms(glob, 2)})" if units and glob else "--"
        L.append(f"| {c:,} | {major} | {sz} | {read} | {ms([s3_wall(r) for r in pipe])} | {gpu_gb(peaks(d), ds)} | "
                 f"{ms([r['stage3b_node_acc'] for r in pipe], 2, 100)} |")
    return L


if __name__ == "__main__":
    for name, rows, note in (
            ("backbones", backbones(), "LakeGRL = Stage 3b alone on the stored leiden_c10000 view (no Phase 1-2), 8 "
             "workers; full graph = experiments/baselines/node_baselines.py --model, same layers (utils/backbones.py). "
             "OOM = peak.sh stopped the run when the 24 GB card was full (gpu_full)."),
            ("cmax_sweep", cmax(), "ogbn-products, row 5 (GraphSAGE, 8 workers, row 8b layout, 2 MB floor). Read = "
             "the run's units + the global tables (macro-graph, communities, unit plan), S3 cost model, 20 readers, "
             "3 cold runs. C_max = 10,000 = Table 4 row 5.")):
        text = "\n".join([f"# {name}", "", note, ""] + rows) + "\n"
        print(text)
        if os.path.isdir(os.path.join(R, name)):
            with open(os.path.join(R, name, "SUMMARY.md"), "w", encoding="utf-8") as fh:
                fh.write(text)

#!/usr/bin/env python3
"""
Tables of the two 2026-11-19 runs (advisor), from their records; mean ± sample std over seeds 0-2 (or 3 cold runs),
peak GPU = peak.sh (nvidia-smi minus idle), the largest over the seeds, as Table B (Table 4).

  backbones   results/backbones/<gcn_pre|gatv2|sage>/: baselines.jsonl, pipeline.jsonl, peak.jsonl (backbones.sh,
              results/backbones/rerun_1120.sh) + the full-graph GraphSAGE of Table 4 (results/tableB)
  cmax        results/cmax_sweep/c<C>/ (run.sh TB_CMAX=C, row 5) + storage.jsonl (store_size.py); C_max = 10,000:
              Stage 3 from results/backbones/sage, reads from Table 4 (results/tableB read_units, global read v4)

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


# Folders of the full graph / LakeGRL cells (advisor 11-20): GCN all again on the precomputed matrix (gcn_pre; gcn/ is
# the first code, superseded), LakeGRL GraphSAGE again by Stage 3b alone (sage); its full graph stays Table 4 (tableB)
FULL = {"sage": ("tableB",), "gcn": ("backbones", "gcn_pre"), "gatv2": ("backbones", "gatv2")}
LAKE = {"sage": ("backbones", "sage"), "gcn": ("backbones", "gcn_pre"), "gatv2": ("backbones", "gatv2")}


def backbones():
    L = ["| Dataset | Backbone | Full-graph Test Acc (%) | Full-graph Peak GPU (GB) | LakeGRL Test Acc (%) | "
         "LakeGRL Stage 3 Train (s) | LakeGRL Peak GPU (GB) | Seeds (full / LakeGRL) | Code (full / LakeGRL) |",
         "|---" * 9 + "|"]
    code = lambda rs: ", ".join(sorted({r["git"][:7] for r in rs})) or "--"  # noqa: E731
    for ds in DS:
        for m in ("sage", "gcn", "gatv2"):
            d = FULL[m]
            pk, lk = peaks(d), peaks(LAKE[m])
            base = [r for r in load(*d, "baselines.jsonl") if r["dataset"] == ds and r["baseline"] == f"full_graph_{m}"]
            full = pk.get(f"{ds} full", {})
            # OOM: the card filled and peak.sh stopped the run (gpu_full), or CUDA refused one allocation (oom record)
            oom = full.get("gpu_full") == 1 or any(r.get("oom") for r in base)
            base_code, base = code(base), [r for r in base if not r.get("oom")]
            pipe = [r for r in row5(load(*LAKE[m], "pipeline.jsonl"), ds) if r["model"] == m]
            if not (base or full or pipe):
                continue
            runs = [lk.get(f"{ds} row5 seed{s}", {}) for s in range(3)]
            killed = any(r.get("gpu_full") == 1 for r in runs)
            failed = [r["rc"] for r in runs if r.get("rc") and r.get("gpu_full") != 1]   # see the job log
            name = {"sage": "GraphSAGE", "gcn": "GCN", "gatv2": "GATv2"}[m]
            full_gpu = "OOM" if oom else f"{full['peak_gpu_mb'] / 1000:.1f}" if full else "--"
            acc = "OOM" if killed else f"failed (rc {failed[0]})" if failed and not pipe else \
                ms([r["stage3b_node_acc"] for r in pipe], 2, 100)
            L.append(f"| {ds} | {name} | {'OOM' if oom else ms([r['test_acc'] for r in base], 2, 100)} | {full_gpu} | "
                     f"{acc} | "
                     f"{'--' if killed else ms([s3_wall(r) for r in pipe])} | "
                     f"{'OOM (> 23.5)' if killed else gpu_gb(lk, ds)} | {len(base)} / {len(pipe)} | {base_code} / {code(pipe)} |")
    return L


def cmax():
    L = ["| C_max | Major units | Stored MV_unit (GB) | Cold Read (s): units + global | Stage 3 Train (s) | "
         "Peak GPU (GB) | Test Acc (%) |", "|---" * 7 + "|"]
    ds = "ogbn-products"
    size = {r["tag"]: r for r in load("cmax_sweep", "storage.jsonl")}   # last record per tag
    for c in (5000, 10000, 20000, 40000):
        d = ("tableB",) if c == 10000 else ("cmax_sweep", f"c{c}")
        s3 = ("backbones", "sage") if c == 10000 else d   # 10,000: Stage 3b alone on the stored view (advisor 11-20)
        pipe = [r for r in row5(load(*s3, "pipeline.jsonl"), ds) if r["max_nodes"] == c and r["model"] == "sage"]
        units = [r["wall_s"] for r in load(*d, "read_units.jsonl") if r["dataset"] == ds and r["row"] == "5"]
        glob = [r["wall_s"] for r in load(*(("tableB", "v4") if c == 10000 else d), "read_full.jsonl")
                if r["dataset"] == ds and r["row"] == "5 global"]
        major = ", ".join(f"{u:,}" for u in sorted({r["n_units_stage3b"] - 1 for r in pipe})) or "--"  # + 1 global model
        sz = f"{size[f'leiden_c{c}']['mv_unit_bytes'] / 1e9:.2f}" if f"leiden_c{c}" in size else "--"
        read = f"{st.mean(units) + st.mean(glob):.1f} ({ms(units)} + {ms(glob, 2)})" if units and glob else "--"
        L.append(f"| {c:,} | {major} | {sz} | {read} | {ms([s3_wall(r) for r in pipe])} | {gpu_gb(peaks(s3), ds)} | "
                 f"{ms([r['stage3b_node_acc'] for r in pipe], 2, 100)} |")
    return L


if __name__ == "__main__":
    for name, rows, note in (
            ("backbones", backbones(), "LakeGRL = Stage 3b alone on the stored leiden_c10000 view (no Phase 1-2), 8 "
             "workers, every backbone (GraphSAGE included, advisor 2026-11-20); full graph = "
             "experiments/baselines/node_baselines.py --model, same layers (utils/backbones.py); full-graph GraphSAGE = "
             "Table 4 (its code path is unchanged since). OOM = the 24 GB card was full (peak.sh stopped the run, "
             "gpu_full) or CUDA refused an allocation (oom record). GCN: every cell on D^-1/2 (A + I) D^-1/2 computed "
             "once (8a6e1e7, GCNConv(normalize=False)); the first runs (PyG normalizing in every forward, "
             "results/backbones/gcn) are superseded."),
            ("cmax_sweep", cmax(), "ogbn-products, row 5 (GraphSAGE, 8 workers, row 8b layout, 2 MB floor). Read = "
             "the run's units + the global tables (macro-graph, communities, unit plan), S3 cost model, 20 readers, "
             "3 cold runs. C_max = 10,000: units, storage and reads = Table 4 row 5; Stage 3, GPU and accuracy = the "
             "--stage3-only GraphSAGE run on the stored view (results/backbones/sage).")):
        text = "\n".join([f"# {name}", "", note, ""] + rows) + "\n"
        print(text)
        if os.path.isdir(os.path.join(R, name)):
            with open(os.path.join(R, name, "SUMMARY.md"), "w", encoding="utf-8") as fh:
                fh.write(text)

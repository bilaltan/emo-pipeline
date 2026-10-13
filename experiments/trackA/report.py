#!/usr/bin/env python3
"""
Track A table from results/trackA/runs.jsonl (generated; every number is a mean over the listed run ids).

Per dataset and method: ROC-AUC (mean ± std over seeds) for every negative scheme x stratum, the share of the
Stage 3a -> full-graph gap that the method closes, and its cost relative to Stage 3a (M1b, LakeGRL's Phase 3 today).

    python experiments/trackA/report.py            # writes results/trackA/T_A.md
"""
import json
import os
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(REPO, "results", "trackA", "runs.jsonl")
OUT = os.path.join(REPO, "results", "trackA", "T_A.md")
METHODS = ["M0", "M1a", "M1b", "M2", "M3", "M5"]
NAMES = {"M0": "full graph (upper bound)", "M1a": "units, members only", "M1b": "units + halo (Phase 3 today)",
         "M2": "units + halo + cut-edge supervision", "M3": "macro + features decoder (M5 without h_local)",
         "M5": "units + cut-edge decoder (late fusion)"}
COLS = [f"auc_{s}_{t}" for s in ("global", "matched") for t in ("all", "intra", "cross")]


def load(path=RUNS, split="test"):
    rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    by = defaultdict(list)
    for r in rows:
        if r.get("split") == split and not str(r.get("git", "")).endswith("-dirty"):
            by[(r["dataset"], json.dumps(r["cfg"], sort_keys=True), r["method"])].append(r)
    return by


def m1a_rows(path=RUNS):
    """Stage 3a rows per dataset (members only: the same for every config, seed and split)."""
    rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    return {ds: np.mean([r["subgraph_rows"] for r in rows if r["dataset"] == ds and r["method"] == "M1a"])
            for ds in {r["dataset"] for r in rows if r["method"] == "M1a" and "subgraph_rows" in r}}


def val_table(by, m1a):
    """Validation runs used to pick budgets (epochs) and to trace AUC against the halo budget (rows read)."""
    out = ["# Validation runs (budget choice and the halo-budget sweep; not test numbers)", "",
           "| dataset | epochs | halo budget | method | n | global cross | matched cross | global all | matched all | "
           "rows vs M1a | train s | runs |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (ds, cfg, m), rs in sorted(by.items(), key=lambda kv: (kv[0][0], json.loads(kv[0][1])["epochs"],
                                                               json.loads(kv[0][1]).get("halo_budget", 99), kv[0][2])):
        c = json.loads(cfg)
        mean = lambda k: f"{np.mean([r[k] for r in rs]):.4f}"
        rows = f"{np.mean([r['subgraph_rows'] for r in rs]) / m1a[ds]:.2f}x" if ds in m1a and "subgraph_rows" in rs[0] else "-"
        out.append(f"| {ds} | {c['epochs']} | {c.get('halo_budget', 'full')} | {m} | {len(rs)} | {mean('auc_global_cross')} | "
                   f"{mean('auc_matched_cross')} | {mean('auc_global_all')} | {mean('auc_matched_all')} | {rows} | "
                   f"{np.mean([r['train_seconds'] for r in rs]):.0f} | {', '.join(r['run_id'] for r in rs)} |")
    return "\n".join(out)


def table(by, m1a):
    out = []
    for ds, cfg in sorted({(k[0], k[1]) for k in by}):
        cell = {m: by.get((ds, cfg, m), []) for m in METHODS}
        if not any(cell.values()):
            continue
        n_pos = next(r["n_pos"] for rs in cell.values() for r in rs)
        n_cross = next(r["n_cross"] for rs in cell.values() for r in rs)
        out += [f"## {ds}", "", f"Config: `{cfg}`. Test positives: {n_pos:,} ({n_cross:,} cross-unit, "
                f"{n_cross / n_pos:.1%}). Mean ± std over seeds.", "",
                "| method | n | " + " | ".join(c[4:].replace("_", " ") for c in COLS) + " | train s | rows vs M1a |",
                "|---|---|" + "---|" * len(COLS) + "---|---|"]
        rows_m1a = m1a.get(ds)
        for m in METHODS:
            rs = cell[m]
            if not rs:
                continue
            vals = " | ".join(f"{np.mean([r[c] for r in rs]):.4f} ± {np.std([r[c] for r in rs]):.4f}" for c in COLS)
            rows = (f"{np.mean([r['subgraph_rows'] for r in rs]) / rows_m1a:.2f}x"
                    if rows_m1a and "subgraph_rows" in rs[0] else "-")
            out.append(f"| {m} {NAMES[m]} | {len(rs)} | {vals} | {np.mean([r['train_seconds'] for r in rs]):.0f} | {rows} |")
        mean = lambda m, c: np.mean([r[c] for r in cell[m]])
        if cell["M1b"]:
            out += ["", "Δ AUC vs M1b (Phase 3 today):", "",
                    "| method | " + " | ".join(c[4:].replace("_", " ") for c in COLS) + " |", "|---|" + "---|" * len(COLS)]
            for m in ("M0", "M1a", "M2", "M3", "M5"):
                if cell[m]:
                    out.append(f"| {m} | " + " | ".join(f"{mean(m, c) - mean('M1b', c):+.4f}" for c in COLS) + " |")
        # gap closed only where M0 is above M1b: elsewhere M0 is not an upper bound and the ratio has no meaning
        if cell["M0"] and cell["M1b"] and cell["M2"]:
            g = []
            for c in COLS:
                den = mean("M0", c) - mean("M1b", c)
                g.append(f"{(mean('M2', c) - mean('M1b', c)) / den:+.0%}" if den > 1e-3 else "M0 ≤ M1b")
            out += ["", "Share of the M1b → M0 gap that M2 closes:", "",
                    "| | " + " | ".join(c[4:].replace("_", " ") for c in COLS) + " |", "|---|" + "---|" * len(COLS),
                    "| M2 | " + " | ".join(g) + " |"]
        out += ["", "Runs: " + ", ".join(r["run_id"] for m in METHODS for r in cell[m]), ""]
    return "\n".join(out)


def _bytes(by, ds, m, f):
    """Mean estimated bytes of (dataset, method) runs. Runs that predate read_bytes_est are back-filled with the same
    formula: M1a and M2 train on exactly their subgraph edges, and M1b reads the same edges as M2 at the same halo."""
    est = []
    for r in by.get((ds, m), []):
        e = r.get("subgraph_edges")
        if e is None and m.split("@")[0] in ("M1a", "M2"):
            e = r.get("train_edges")
        elif e is None and m.startswith("M1b"):
            twin = by.get((ds, "M2" + m[3:]), [])
            e = np.mean([t["train_edges"] for t in twin if "train_edges" in t]) if twin else None
        if "read_bytes_est" in r:
            est.append(r["read_bytes_est"])
        elif f and e is not None and "subgraph_rows" in r:
            est.append(r["subgraph_rows"] * (8 + 4 * f) + e * 16)
    return float(np.mean(est)) if est else None


def gate_table(path=RUNS, epochs=100):
    """H-A at the pre-registered budget (every method `epochs` epochs, test split): share of the Stage 3a (M1a) ->
    full graph (M0) gap on the union ("all") AUC that each candidate closes, per negative scheme, and its rows vs Stage
    3a. Halo-budget variants are their own rows (method@R). H-A wants >= 50% on both schemes at <= 2x rows."""
    rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    m1a = m1a_rows(path)
    fdim = {r["dataset"]: r["feat_dim"] for r in rows if "feat_dim" in r}
    by = defaultdict(list)
    for r in rows:
        c = dict(r["cfg"])
        if r.get("split") != "test" or str(r.get("git", "")).endswith("-dirty") or c.get("epochs") != epochs:
            continue
        hb = c.pop("halo_budget", None)
        if c != {"epochs": epochs, "batch": 65536, "hidden": 256, "lr": 0.005, "dropout": 0.5}:
            continue
        by[(r["dataset"], r["method"] + (f"@R={hb:g}" if hb else ""))].append(r)
    out = [f"# Gate H-A ({epochs}-epoch protocol, test split, mean over seeds)", "",
           "Bytes vs M1a: estimated bytes read (uncompressed rows + edge pairs) relative to Stage 3a; '-' for runs that "
           "predate the estimate. H-A wants >= 50% of the gap on both schemes at <= 2x Stage 3a's bytes (rows when "
           "bytes are missing).", "",
           "| dataset | method | n | rows vs M1a | bytes vs M1a | gap closed, global all | gap closed, matched all | H-A |",
           "|---|---|---|---|---|---|---|---|"]
    for ds in sorted({k[0] for k in by}):
        mean = lambda m, c: np.mean([r[c] for r in by[(ds, m)]])
        if not by.get((ds, "M0")) or not by.get((ds, "M1a")):
            continue
        for m in sorted(k[1] for k in by if k[0] == ds and k[1] not in ("M0", "M1a")):
            closed, ok = [], True
            for c in ("auc_global_all", "auc_matched_all"):
                den = mean("M0", c) - mean("M1a", c)
                if den <= 1e-3:
                    closed.append(f"M0 ≤ M1a ({mean('M0', c):.4f} vs {mean('M1a', c):.4f})")
                    continue
                share = (mean(m, c) - mean("M1a", c)) / den
                closed.append(f"{share:+.0%}")
                ok &= share >= 0.5
            rr = np.mean([r["subgraph_rows"] for r in by[(ds, m)]]) / m1a[ds] if ds in m1a else float("nan")
            b1, bm = (_bytes(by, ds, k, fdim.get(ds)) for k in ("M1a", m))
            br = bm / b1 if b1 and bm else None
            ok &= (br if br is not None else rr) <= 2.0
            out.append(f"| {ds} | {m} | {len(by[(ds, m)])} | {rr:.2f}x | {f'{br:.2f}x' if br is not None else '-'} | "
                       f"{closed[0]} | {closed[1]} | {'meets' if ok else 'no'} |")
    return "\n".join(out)


def main():
    m1a = m1a_rows()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Track A: link prediction across unit boundaries (generated, do not edit)\n\n"
                "Protocol: docs/trackA_link_prediction.md. Runs with uncommitted code (`-dirty`) are excluded. "
                "Rows vs M1a = subgraph rows read relative to Stage 3a (members only).\n\n"
                + gate_table() + "\n\n" + table(load(), m1a) + "\n\n" + val_table(load(split="val"), m1a) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()

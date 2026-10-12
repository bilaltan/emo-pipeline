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
METHODS = ["M0", "M1a", "M1b", "M2"]
NAMES = {"M0": "full graph (upper bound)", "M1a": "units, members only", "M1b": "units + halo (Phase 3 today)",
         "M2": "units + halo + cut-edge supervision"}
COLS = [f"auc_{s}_{t}" for s in ("global", "matched") for t in ("all", "intra", "cross")]


def load(path=RUNS):
    rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    by = defaultdict(list)
    for r in rows:
        if r.get("split") == "test" and not str(r.get("git", "")).endswith("-dirty"):
            by[(r["dataset"], json.dumps(r["cfg"], sort_keys=True), r["method"])].append(r)
    return by


def table(by):
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
        rows_m1a = np.mean([r["subgraph_rows"] for r in cell["M1a"]]) if cell["M1a"] else None
        for m in METHODS:
            rs = cell[m]
            if not rs:
                continue
            vals = " | ".join(f"{np.mean([r[c] for r in rs]):.4f} ± {np.std([r[c] for r in rs]):.4f}" for c in COLS)
            rows = (f"{np.mean([r['subgraph_rows'] for r in rs]) / rows_m1a:.2f}x"
                    if rows_m1a and "subgraph_rows" in rs[0] else "-")
            out.append(f"| {m} {NAMES[m]} | {len(rs)} | {vals} | {np.mean([r['train_seconds'] for r in rs]):.0f} | {rows} |")
        # gap closed: (method - M1b) / (M0 - M1b) per column, on the mean AUCs
        if cell["M0"] and cell["M1b"]:
            out += ["", "Share of the M1b → M0 gap closed (negative = worse than M1b):", "",
                    "| method | " + " | ".join(c[4:].replace("_", " ") for c in COLS) + " |", "|---|" + "---|" * len(COLS)]
            mean = lambda m, c: np.mean([r[c] for r in cell[m]])
            for m in ("M1a", "M2"):
                if not cell[m]:
                    continue
                g = []
                for c in COLS:
                    den = mean("M0", c) - mean("M1b", c)
                    g.append(f"{(mean(m, c) - mean('M1b', c)) / den:+.0%}" if abs(den) > 1e-3 else "n/a")
                out.append(f"| {m} | " + " | ".join(g) + " |")
        out += ["", "Runs: " + ", ".join(r["run_id"] for m in METHODS for r in cell[m]), ""]
    return "\n".join(out)


def main():
    by = load()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# Track A: link prediction across unit boundaries (generated, do not edit)\n\n"
                "Protocol: docs/trackA_link_prediction.md. Runs with uncommitted code (`-dirty`) are excluded.\n\n"
                + table(by) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()

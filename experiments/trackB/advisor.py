#!/usr/bin/env python3
"""
P2 layout advisor (docs/trackB_storage_layout.md §6): choose Phase 2's row-group size before it writes, from the
table size and the unit count. The choice is scored against the measured layouts by leave-one-dataset-out regret.

Only the row-group size of Phase 2's range layout is predicted. Files, partitioning and the per-unit outputs stay the same.
Cost model: a unit read costs a seconds per byte plus b seconds per row group. With row-group size s, the units read
about the table T plus k = 2 boundary row groups each (one in nodes, one in edges), i.e. T / s + k U row groups:
    wall(s) ~ a (T + k U s) + b (T / s + k U)   ->   s* = sqrt(beta T / (k U)),  beta = b / a
beta, the per-row-group overhead in bytes, belongs to the storage environment. It is fitted on the other datasets: the
median of the per-dataset fits where both a and b come out positive.

Rules scored: default (128 MB, Phase 2 as written), B2 (largest grid size <= T / U), sqrt (s* snapped to the grid).
Regret = measured wall of the pick / best measured wall on that dataset. Wall = per-unit direct reads, median of repeats.

    python experiments/trackB/advisor.py             # prints and writes results/trackB/P2_advisor.md
"""
import json
import os
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results", "trackB")
FILES = [os.path.join(RES, f) for f in ("partition_layout.jsonl", "rg_sweep.jsonl")]
DEFAULT_MB, K = 128, 2
MB = 1 << 20
# runs written before the bench recorded rg_mb (d2f6691): the sweep jobs ran 0.25 / 1 / 4 / 16 MB in this order
SWEEP_RG = dict(zip(["20261013-193408", "20261013-193522", "20261013-193634", "20261013-193745",
                     "20261013-193855", "20261013-194025", "20261013-194155", "20261013-194326",
                     "20261013-200416", "20261013-201958", "20261013-203440", "20261013-204920"],
                    [0.25, 1, 4, 16] * 3))


def points(files=FILES):
    """{dataset: {"T": bytes, "U": units, "grid": {rg MB: [(bytes, row groups, wall s), ...]}}} for range / range_rg."""
    out = {}
    for f in files:
        if not os.path.exists(f):
            continue
        for line in open(f):
            r = json.loads(line)
            if r["kind"] != "layout" or r["layout"] not in ("range", "range_rg"):
                continue
            s = DEFAULT_MB if r["layout"] == "range" else r.get("rg_mb") or SWEEP_RG.get(r["run_id"], 1.0)
            ds = r["dataset"] + ("" if r.get("alg", "louvain") == "louvain" else "/" + r["alg"])
            d = out.setdefault(ds, {"T": r["storage"]["nodes"]["bytes"] + r["storage"]["edges"]["bytes"],
                                    "U": r["units"], "grid": {}})
            u = r["direct"]["unit"]
            d["grid"].setdefault(float(s), []).append(
                (u["bytes_selected"], u["row_groups_selected"], statistics.median(u["wall_s"])))
    return out


def fit_beta(d):
    """beta = b / a from wall ~ a bytes + b row groups on one dataset; None if the fit is not physical."""
    m = [p for ps in d["grid"].values() for p in ps]
    if len({s for s in d["grid"]}) < 3:
        return None
    (a, b), *_ = np.linalg.lstsq(np.array([[p[0], p[1]] for p in m], float), np.array([p[2] for p in m]), rcond=None)
    return b / a if a > 0 and b > 0 else None


def wall(d, s):
    return statistics.median(p[2] for p in d["grid"][s])


def sqrt_pick(T, U, beta, grid):
    s = np.sqrt(beta * T / (K * U)) / MB
    return s, min(grid, key=lambda g: abs(np.log(g / s)))


def b2_pick(T, U, grid):
    return max([g for g in grid if g <= T / U / MB] or [min(grid)])


def table(pts):
    betas = {ds: fit_beta(d) for ds, d in pts.items()}
    rows = []
    for ds, d in sorted(pts.items()):
        grid = sorted(d["grid"])
        best = min(grid, key=lambda g: wall(d, g))
        train = [b for o, b in betas.items() if o != ds and b]
        r = {"dataset": ds, "units": d["U"], "T_per_U_mb": d["T"] / d["U"] / MB, "grid": grid, "best": best,
             "walls": {g: wall(d, g) for g in grid}, "beta_own_mb": (betas[ds] or float("nan")) / MB,
             "b2": b2_pick(d["T"], d["U"], grid)}
        if train:
            beta = statistics.median(train)
            r["beta_train_mb"] = beta / MB
            r["s_star"], r["sqrt"] = sqrt_pick(d["T"], d["U"], beta, grid)
        for rule in ("b2", "sqrt"):
            if rule in r:
                r[rule + "_regret"] = wall(d, r[rule]) / wall(d, best)
        if DEFAULT_MB in grid:
            r["default_regret"] = wall(d, DEFAULT_MB) / wall(d, best)
        rows.append(r)
    return rows


def markdown(rows):
    f = lambda v, p="{:.2f}": "–" if v is None or v != v else p.format(v)  # noqa: E731
    out = ["| dataset | units | table / units (MB) | wall by row group (MB: s) | best | β own / train (MB) | "
           "default 128 MB | B2 pick, regret | √ pick (s*), regret |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        walls = ", ".join(f"{g:g}: {w:.2f}" for g, w in r["walls"].items())
        out.append(f"| {r['dataset']} | {r['units']} | {r['T_per_U_mb']:.2f} | {walls} | {r['best']:g} | "
                   f"{f(r['beta_own_mb'])} / {f(r.get('beta_train_mb'))} | {f(r.get('default_regret'), '{:.2f}×')} | "
                   f"{r['b2']:g}, {r['b2_regret']:.2f}× | "
                   + (f"{r['sqrt']:g} ({r['s_star']:.2f}), {r['sqrt_regret']:.2f}× |" if "sqrt" in r else "– |"))
    return "\n".join(out)


def main():
    md = markdown(table(points()))
    sys.stdout.reconfigure(errors="replace")       # Windows consoles are cp1252
    print(md)
    with open(os.path.join(RES, "P2_advisor.md"), "w", encoding="utf-8") as fh:
        fh.write("# P2 row-group advisor, leave-one-dataset-out (generated by experiments/trackB/advisor.py)\n\n"
                 + md + "\n")


if __name__ == "__main__":
    main()

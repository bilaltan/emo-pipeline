#!/usr/bin/env python3
"""
Checks for the P2 advisor on synthetic bench records drawn from its own cost model (seconds, no Spark):
  1. fit_beta recovers beta = b / a;
  2. leave-one-out picks the grid size nearest s* = sqrt(beta T / (k U)), and its regret is 1 when that is the best;
  3. rg_mb comes from the record, the pre-d2f6691 sweep map, or 128 MB for the range layout.

    python experiments/trackB/test_advisor.py
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import advisor as A  # noqa: E402

MB, a = 1 << 20, 1e-9
b = a * MB                                 # beta = 1 MB


def rec(ds, T, U, s, run_id="x"):
    sel, rgs = T + A.K * U * s * MB, T / (s * MB) + A.K * U
    return {"run_id": run_id, "dataset": ds, "alg": "louvain", "kind": "layout",
            "layout": "range" if s == 128 else "range_rg", "rg_mb": None if s == 128 else s, "units": U,
            "storage": {"nodes": {"bytes": T}, "edges": {"bytes": 0}},
            "direct": {"unit": {"bytes_selected": sel, "row_groups_selected": rgs, "wall_s": [a * sel + b * rgs]}}}


def main():
    grid = [0.25, 0.5, 1, 2, 4, 16, 128]
    stores = {"small": (400 * MB, 300), "mid": (2000 * MB, 1000), "big": (3000 * MB, 25)}
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "r.jsonl")
        with open(f, "w") as fh:
            for ds, (T, U) in stores.items():
                for s in grid:
                    fh.write(json.dumps(rec(ds, T, U, s)) + "\n")
        pts = A.points([f])
        for ds, d in pts.items():
            assert abs(A.fit_beta(d) / MB - 1) < 1e-6, (ds, A.fit_beta(d))
        for r in A.table(pts):
            T, U = stores[r["dataset"]]
            s = np.sqrt(MB * T / (A.K * U)) / MB     # beta = 1 MB
            assert abs(r["s_star"] - s) < 1e-6 and r["sqrt"] == min(grid, key=lambda g: abs(np.log(g / s)))
            assert r["sqrt_regret"] >= 1 and r["default_regret"] >= r["sqrt_regret"], r
            print(f"{r['dataset']}: s* {r['s_star']:.2f} MB -> {r['sqrt']:g}, best {r['best']:g}, "
                  f"regret {r['sqrt_regret']:.3f}, default {r['default_regret']:.2f}")
        # rg_mb back-fill for the sweep runs that predate the field
        old = rec("old", 400 * MB, 400, 4, run_id="20261013-194155")
        old["rg_mb"] = None
        with open(f, "w") as fh:
            fh.write(json.dumps(old) + "\n")
        assert list(A.points([f])["old"]["grid"]) == [4.0]
    print("advisor checks passed")


if __name__ == "__main__":
    main()

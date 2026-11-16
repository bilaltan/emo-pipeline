#!/usr/bin/env python3
"""
Checks for the P3 halo bench on a toy cache (seconds): every hybrid layout gives each unit the same rows, id sum and
feature sum as the full halo, and at B = 1 it reads and stores fewer bytes than the full halo.

    python experiments/trackB/test_halo_hybrid.py
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import halo_hybrid_bench as HB  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    n, f = 600, 16
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "toy", "trackA_louvain")
        os.makedirs(cache)
        comm = rng.integers(0, 6, n).astype(np.int64)
        comm[:10] = HB.NO_COMM                                  # vertices in no unit
        hv = rng.choice(np.arange(10, n), 150, replace=False)   # halo vertices, each in 1-4 other units
        pairs = {(int(v), int(u)) for v in hv for u in rng.choice(6, rng.integers(1, 5), replace=False) if u != comm[v]}
        np.save(os.path.join(cache, "comm.npy"), comm)
        np.save(os.path.join(cache, "halo.npy"), np.array(sorted(pairs), np.int64).T)
        np.save(os.path.join(cache, "x.npy"), rng.standard_normal((n, f)).astype(np.float32))
        out = os.path.join(tmp, "r.jsonl")
        sys.argv = ["x", "--store", tmp, "--dataset", "toy", "--readers", "1", "2", "--rg-mb", "0.01",
                    "--repeats", "1", "--work-dir", os.path.join(tmp, "wd"), "--out", out]
        HB.main()
        recs = [json.loads(line) for line in open(out)]
        assert len(recs) == 4 and all(r["identical"] for r in recs), recs
        full = next(r for r in recs if r["layout"] == "full")
        h1 = next(r for r in recs if r["layout"] == "hybrid" and r["readers"] == 1)
        assert h1["hot_vertices"] > 0 and h1["bytes_read"] < full["bytes_read"] and h1["rows_stored"] < full["rows_stored"]
        sys.argv += ["--latency-ms", "0", "--mbps", "1e6"]      # the same through the S3 emulator
        HB.main()
        emu = [json.loads(line) for line in open(out)][4:]
        assert len(emu) == 4 and all(r["identical"] and r["gets"] > 0 for r in emu), emu
    print("halo hybrid checks passed")


if __name__ == "__main__":
    main()

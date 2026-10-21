#!/usr/bin/env python3
"""
Check for full_graph_sage.py on a toy cache (CPU, seconds): labels are the community, features alone are
noise, so a test accuracy near 1 means the adjacency, split masks, checkpoint and output record all work.

    bash scripts/wsl_run.sh python experiments/baselines/test_full_graph_sage.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import full_graph_sage as B  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    n = 200
    y = np.repeat([0, 1], n // 2)
    # dense inside each half, no cut edges; one community-mean feature dimension buried in noise
    s, d = rng.integers(0, n // 2, (2, 1500))
    edges = np.concatenate([np.stack([s, d]), np.stack([s, d]) + n // 2], axis=1)
    edges = np.unique(np.sort(edges[:, edges[0] != edges[1]], axis=0), axis=1)
    x = rng.normal(size=(n, 16)).astype(np.float32)
    x[:, 0] += np.where(y == 1, 0.3, -0.3)
    split = rng.choice([0, 1, 2], n, p=[0.2, 0.2, 0.6]).astype(np.int8)

    root = tempfile.mkdtemp()
    try:
        for name, arr in [("x", x), ("y", y.astype(np.int64)), ("split", split), ("edges", edges)]:
            np.save(os.path.join(root, f"{name}.npy"), arr)
        out = os.path.join(root, "out.jsonl")
        B.main(["--dataset", "toy", "--cache", root, "--seeds", "0,1", "--epochs", "40",
                "--device", "cpu", "--out", out])
        recs = [json.loads(line) for line in open(out)]
        assert [r["seed"] for r in recs] == [0, 1]
        assert all(r["n_test"] == int((split == 2).sum()) for r in recs)
        assert all(r["test_acc"] > 0.95 for r in recs), [r["test_acc"] for r in recs]
    finally:
        shutil.rmtree(root)
    print("ok")


if __name__ == "__main__":
    main()

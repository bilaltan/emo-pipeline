#!/usr/bin/env python3
"""
Check for node_baselines.py (both methods) on a toy cache (CPU, seconds): labels are the community, features alone are
noise, so a test accuracy above 0.9 means the adjacency, split masks, checkpoint and output record all work.

    bash scripts/wsl_run.sh python experiments/baselines/test_node_baselines.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import node_baselines as B  # noqa: E402


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
        spark = {"spark_master": "local[2]", "spark_executors": 1, "spark_executor_cores": 2}
        with open(os.path.join(root, "meta.json"), "w") as fh:
            json.dump({"export_seconds": 1.5, "spark": spark}, fh)
        out = os.path.join(root, "out.jsonl")
        for method in ("full", "cluster"):
            B.main(["--dataset", "toy", "--method", method, "--cache", root, "--seeds", "0,1", "--epochs", "40",
                    "--part-nodes", "20", "--parts-per-batch", "3", "--device", "cpu", "--out", out])
        recs = [json.loads(line) for line in open(out)]
        assert [(r["baseline"], r["seed"]) for r in recs] == [
            ("full_graph_sage", 0), ("full_graph_sage", 1), ("cluster_gcn", 0), ("cluster_gcn", 1)]
        assert recs[2]["n_parts"] == 10
        assert all(r["n_test"] == int((split == 2).sum()) for r in recs)
        assert all(r["export_spark"] == spark and r["export_seconds"] == 1.5 for r in recs)
        # features alone give about 0.6; Cluster-GCN drops edges between parts, so it lands a little below full
        assert all(r["test_acc"] > 0.9 for r in recs), [r["test_acc"] for r in recs]

        # community-ID embedding: communities 7 and 9 are the halves, 5 is minor (< --min-size), vertex 0 has none
        cids = np.where(y == 1, 9, 7)
        cids[1:4] = 5
        rows, k = B.comm_index(np.arange(1, n), cids[1:], n, 10)
        assert k == 2 and rows[0] == rows[1] == rows[3] == 2 and rows[4] == 0 and rows[-1] == 1
        # the split layer 1 is exactly the encoder on the concatenated input
        import torch
        net = B.Net(16, 8, 2, 0.5, (torch.from_numpy(rows), k + 1)).eval()
        xt = torch.from_numpy(x)
        A = B.mean_adj(n, edges[0], edges[1], torch.device("cpu"))
        with torch.no_grad():
            ref = net.fc(net.enc(torch.cat([xt, net.emb(net.comm)], 1), A))
            assert torch.allclose(net(xt, A), ref, atol=1e-5)
        B.read_comms = lambda url: (np.arange(1, n), cids[1:], 3)
        out2 = os.path.join(root, "out2.jsonl")
        B.main(["--dataset", "toy", "--method", "full", "--cache", root, "--seeds", "0", "--epochs", "40",
                "--communities", "toy_comms", "--min-size", "10", "--device", "cpu", "--out", out2])
        r = json.loads(open(out2).readline())
        assert (r["baseline"], r["n_major"], r["n_other"], r["communities_version"]) == ("full_graph_sage_commid", 2, 4, 3)
        assert r["test_acc"] > 0.9, r["test_acc"]
    finally:
        shutil.rmtree(root)
    print("ok")


if __name__ == "__main__":
    main()

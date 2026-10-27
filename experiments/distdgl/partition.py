#!/usr/bin/env python3
"""
DistDGL step 1: the Track A cache (experiments/trackA/export_cache.py: the same Delta tables, features, labels and
split as LakeGRL and the node baselines) -> one DGL graph -> METIS partitions, one per machine.

DistDGL's own preprocessing counts toward its time: the export (meta.json) plus this partitioning. Features are
L2-normalised here, as Phase 3b and the node baselines do when they load them.

    python experiments/distdgl/partition.py --cache $LAKEGRL_STORE/ogbn-arxiv/trackA_louvain --parts 2 --out /tmp/dd_arxiv
"""
import argparse
import json
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="Track A cache directory (x.npy, y.npy, split.npy, edges.npy)")
    ap.add_argument("--parts", type=int, required=True, help="machines")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="g")
    args = ap.parse_args()

    import dgl
    import dgl.distributed
    import torch

    t0 = time.time()
    L = lambda f: np.load(os.path.join(args.cache, f))
    x, y, split, e = L("x.npy"), L("y.npy"), L("split.npy"), L("edges.npy")
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    x = (x / np.where(nrm > 0, nrm, 1.0)).astype(np.float32)
    n = len(y)
    src, dst = np.concatenate([e[0], e[1]]), np.concatenate([e[1], e[0]])   # cache holds src < dst once
    g = dgl.graph((torch.from_numpy(src), torch.from_numpy(dst)), num_nodes=n)
    g.ndata["feat"] = torch.from_numpy(x)
    g.ndata["label"] = torch.from_numpy(y.astype(np.int64))
    for k, code in (("train_mask", 0), ("val_mask", 1), ("test_mask", 2)):
        g.ndata[k] = torch.from_numpy(split == code)
    load_s = time.time() - t0

    t1 = time.time()
    # METIS with training vertices and edges balanced across machines, as DistDGL's GraphSAGE example partitions
    dgl.distributed.partition_graph(g, args.name, args.parts, args.out, num_hops=1, part_method="metis",
                                    balance_ntypes=g.ndata["train_mask"], balance_edges=True)
    part_s = time.time() - t1
    meta = {"cache": args.cache, "parts": args.parts, "n_nodes": n, "n_edges_directed": int(g.num_edges()),
            "n_train": int((split == 0).sum()), "n_val": int((split == 1).sum()), "n_test": int((split == 2).sum()),
            "n_classes": int(y.max()) + 1, "feat_dim": int(x.shape[1]), "load_seconds": round(load_s, 1),
            "partition_seconds": round(part_s, 1), "dgl": dgl.__version__, "torch": torch.__version__}
    json.dump(meta, open(os.path.join(args.out, "partition_meta.json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()

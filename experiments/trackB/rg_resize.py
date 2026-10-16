#!/usr/bin/env python3
"""
Copy a Phase 2 table pair (p2_nodes, p2_edges) at another row-group size, rows and row order unchanged (Track B P2 / P7:
the row-group sweep through the S3 emulator without rerunning Phase 2). The copy gets a one-commit _delta_log that
lists its files, which is all s3_emu_bench reads.

--block-cap N (P4b, block-ordered write): Phase 3b splits a community of more than N rows into n (a power of two)
blocks by id & (n-1). The copy adds _blk (a node's block; an edge's block if both endpoints share it, else n) and sorts
each file by (community_id, _blk, id / src, dst), so a block's rows are contiguous and its row groups can be found by
statistics. Decided before the write from the community sizes, as Phase 2 knows them. Rows are unchanged.

    python experiments/trackB/rg_resize.py ~/lakegrl/store/ogbn-arxiv-lp/p2_nodes/louvain \\
        ~/lakegrl/store/ogbn-arxiv-lp/p2_edges/louvain ~/lakegrl/rgsweep/arxiv --rg-mb 0.5
"""
import argparse
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.trackB.s3_emu_bench import delta_files, n_blocks  # noqa: E402


def block_order(t, name, comm, nb):
    """t with _blk added and sorted by (community_id, _blk, id | src, dst); comm / nb: sorted communities, blocks."""
    c = t.column("community_id").to_numpy()
    i = np.searchsorted(comm, c)
    assert (comm[np.minimum(i, len(comm) - 1)] == c).all(), "a community without node rows"
    m = nb[i] - 1
    if name == "nodes":
        blk, keys = t.column("id").to_numpy() & m, ["id"]
    else:
        bs, bd = t.column("src").to_numpy() & m, t.column("dst").to_numpy() & m
        blk, keys = np.where(bs == bd, bs, m + 1), ["src", "dst"]
    t = t.append_column("_blk", pa.array(blk, pa.int64()))
    return t.sort_by([(k, "ascending") for k in ["community_id", "_blk"] + keys])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nodes")
    ap.add_argument("edges")
    ap.add_argument("dst")
    ap.add_argument("--rg-mb", type=float, required=True)
    ap.add_argument("--block-cap", type=int, default=0, help="P4b: block-ordered copy for this Phase 3b cap")
    a = ap.parse_args()
    if a.block_cap:
        comm, cnt = np.unique(ds.dataset(delta_files(os.path.expanduser(a.nodes)), format="parquet")
                              .to_table(columns=["community_id"]).column("community_id").to_numpy(), return_counts=True)
        nb = np.array([n_blocks(k, a.block_cap) for k in cnt], np.int64)
        print(f"block-ordered: {int((nb > 1).sum())} of {len(comm)} communities split, {int(nb.sum())} work items")
    for name, src in (("nodes", a.nodes), ("edges", a.edges)):
        out = os.path.join(os.path.expanduser(a.dst), f"p2_{name}")
        os.makedirs(os.path.join(out, "_delta_log"))
        adds, rgs = [], 0
        for f in delta_files(os.path.expanduser(src)):
            rel = os.path.basename(f)
            assert "=" not in os.path.relpath(f, src), "partitioned table: range layout only"
            t = pq.read_table(f)
            if a.block_cap:
                t = block_order(t, name, comm, nb)
            rows = max(1, int(a.rg_mb * (1 << 20) * t.num_rows / max(os.path.getsize(f), 1)))  # on-disk bytes per row
            pq.write_table(t, os.path.join(out, rel), row_group_size=rows)
            md = pq.ParquetFile(os.path.join(out, rel)).metadata
            assert md.num_rows == t.num_rows
            rgs += md.num_row_groups
            adds.append(json.dumps({"add": {"path": rel}}))
        with open(os.path.join(out, "_delta_log", "00000000000000000000.json"), "w") as fh:
            fh.write("\n".join(adds) + "\n")
        size = sum(os.path.getsize(os.path.join(out, json.loads(x)["add"]["path"])) for x in adds)
        print(f"p2_{name}: {len(adds)} files, {rgs} row groups, {size / 1e6:.1f} MB "
              f"({size / max(rgs, 1) / 1e6:.2f} MB per row group, target {a.rg_mb:g})", flush=True)


if __name__ == "__main__":
    main()

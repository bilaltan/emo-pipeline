#!/usr/bin/env python3
"""
Copy a Phase 2 table pair (p2_nodes, p2_edges) at another row-group size, rows and row order unchanged (Track B P2 / P7:
the row-group sweep through the S3 emulator without rerunning Phase 2). The copy gets a one-commit _delta_log that
lists its files, which is all s3_emu_bench reads.

    python experiments/trackB/rg_resize.py ~/lakegrl/store/ogbn-arxiv-lp/p2_nodes/louvain \\
        ~/lakegrl/store/ogbn-arxiv-lp/p2_edges/louvain ~/lakegrl/rgsweep/arxiv --rg-mb 0.5
"""
import argparse
import json
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.trackB.s3_emu_bench import delta_files  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nodes")
    ap.add_argument("edges")
    ap.add_argument("dst")
    ap.add_argument("--rg-mb", type=float, required=True)
    a = ap.parse_args()
    for name, src in (("nodes", a.nodes), ("edges", a.edges)):
        out = os.path.join(os.path.expanduser(a.dst), f"p2_{name}")
        os.makedirs(os.path.join(out, "_delta_log"))
        adds, rgs = [], 0
        for f in delta_files(os.path.expanduser(src)):
            rel = os.path.basename(f)
            assert "=" not in os.path.relpath(f, src), "partitioned table: range layout only"
            t = pq.read_table(f)
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

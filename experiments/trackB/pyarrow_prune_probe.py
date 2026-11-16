#!/usr/bin/env python3
"""
Why a plain pyarrow scan over-reads the range layout (Track B; seconds, read-only).

1. Toy file: row groups kept by statistics for isin, == and a range predicate (split_by_row_group).
2. A Phase 2 nodes table: bytes and read calls of one unit's scan with isin vs an OR of ==, plain vs unit_dataset.

    bash scripts/wsl_run.sh python experiments/trackB/pyarrow_prune_probe.py [table_dir]
"""
import functools
import glob
import json
import operator
import os
import sys
import tempfile
import threading
import types

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.trackB.s3_emu_bench import emu_fs  # noqa: E402
from utils.partition_layout import unit_dataset  # noqa: E402


def snapshot(table):
    """Live files of a Delta table from its JSON log (no checkpoints needed while all commits are kept)."""
    live = {}
    for j in sorted(glob.glob(os.path.join(table, "_delta_log", "*.json"))):
        for line in open(j):
            a = json.loads(line)
            if "add" in a:
                live[a["add"]["path"]] = 1
            if "remove" in a:
                live.pop(a["remove"]["path"], None)
    return sorted(os.path.join(table, k) for k in live)


def main():
    print("pyarrow", pa.__version__)
    f = ds.field("community_id")
    with tempfile.TemporaryDirectory() as tmp:
        for name, comm in [("one community per row group", np.repeat(np.arange(40), 500)),
                           ("sorted, ranges per row group", np.sort(np.random.default_rng(0).integers(-1000, 0, 20000)))]:
            p = os.path.join(tmp, "t.parquet")
            pq.write_table(pa.table({"community_id": comm.astype(np.int64)}), p, row_group_size=500)
            c = int(comm[len(comm) // 3])
            frag = next(ds.dataset(p, format="parquet").get_fragments())
            kept = {k: len(frag.split_by_row_group(filter=e))
                    for k, e in [("isin", f.isin([c])), ("==", f == c), ("range", (f >= c) & (f <= c))]}
            print(f"toy, {name}: {frag.num_row_groups} row groups, kept by statistics {kept}")

    table = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/lakegrl/store/ogbn-arxiv-lp/p2_nodes/louvain")
    files = snapshot(table)
    cid = ds.dataset(files, format="parquet").to_table(columns=["community_id"])["community_id"].to_numpy()
    u, k = np.unique(cid, return_counts=True)
    print(table, len(files), "files,", sum(pq.ParquetFile(x).metadata.num_row_groups for x in files), "row groups,",
          len(u), "units")
    rng = np.random.default_rng(1)
    for units in ([int(u[np.argsort(k)[len(k) // 2]])], sorted(int(x) for x in rng.choice(u, 3, replace=False))):
        for name, e in [("isin", f.isin(units)), ("OR of ==", functools.reduce(operator.or_, [f == c for c in units]))]:
            rows = set()
            for pruned in (False, True):
                st = types.SimpleNamespace(lock=threading.Lock(), gets=0, bytes=0)
                d = ds.dataset(files, format="parquet", filesystem=emu_fs(0, 1e6, st))
                if pruned:
                    d = unit_dataset(d, units, e)
                    st.gets = st.bytes = 0          # footers are cached once per process; count the scan only
                t = d.to_table(filter=e)
                rows.add(t.num_rows)
                print(f"  units {units} {name:9s} {'unit_dataset' if pruned else 'plain':12s} rows={t.num_rows} "
                      f"read calls={st.gets} bytes={st.bytes:,}")
            assert len(rows) == 1, rows


if __name__ == "__main__":
    main()

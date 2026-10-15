#!/usr/bin/env python3
"""
Check for the S3 emulator bench on a toy Phase 2 table (seconds): range and hive layouts give every unit the same
rows / id sums / edges as a plain local read, every read is counted as a GET, the latency is paid, and the
stats-pruned reader (unit_dataset) needs fewer GETs than the plain one on the range layout.

    python experiments/trackB/test_s3_emu.py
"""
import json
import os
import sys
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s3_emu_bench as E  # noqa: E402


def write_table(base, tbl, part=None, n_files=6):
    os.makedirs(os.path.join(base, "_delta_log"))
    adds = []
    if part:
        for v in sorted(set(tbl.column(part).to_pylist())):
            sub = tbl.filter(pa.compute.equal(tbl.column(part), v)).drop([part])
            rel = f"{part}={v}/part-0.parquet"
            os.makedirs(os.path.join(base, f"{part}={v}"))
            pq.write_table(sub, os.path.join(base, rel), row_group_size=50)
            adds.append(rel)
    else:
        for k, idx in enumerate(np.array_split(np.arange(tbl.num_rows), n_files)):
            rel = f"part-{k}.parquet"
            pq.write_table(tbl.take(idx), os.path.join(base, rel), row_group_size=50)
            adds.append(rel)
    with open(os.path.join(base, "_delta_log", "00000000000000000000.json"), "w") as fh:
        for rel in adds:
            fh.write(json.dumps({"add": {"path": rel}}) + "\n")


def main():
    rng = np.random.default_rng(0)
    n = 1200
    comm = np.sort(rng.integers(0, 30, n)).astype(np.int64)
    nodes = pa.table({"id": np.arange(n, dtype=np.int64), "features": [rng.random(4).tolist() for _ in range(n)],
                      "community_id": comm})
    src = rng.integers(0, n, 3000)
    dst = rng.integers(0, n, 3000)
    keep = comm[src] == comm[dst]
    edges = pa.table({"src": src[keep].astype(np.int64), "dst": dst[keep].astype(np.int64),
                      "community_id": comm[src[keep]]})
    with tempfile.TemporaryDirectory() as tmp:
        for layout, part in (("range", None), ("hive", "community_id")):
            td = os.path.join(tmp, layout)
            write_table(os.path.join(td, "p2_nodes"), nodes, part)
            write_table(os.path.join(td, "p2_edges"), edges, part)
            desc, c2p, rows = E.describe(td)
            units = [[int(c)] for c in rows.index]
            _, truth = E.run_mode(td, desc, c2p, units, False, None, None, 2, 1)
            assert len(truth) == len(set(comm.tolist())), layout
            got = {}
            for prune in (False, True):
                t0 = time.perf_counter()
                passes, check = E.run_mode(td, desc, c2p, units, prune, 5.0, 1000.0, 2, 2)
                assert check == truth, (layout, prune)
                cold = passes[0]
                assert cold[1] > 0 and cold[2] > 0, (layout, prune, cold)
                assert time.perf_counter() - t0 >= cold[1] * 0.005 / 2 * 0.9, "latency not paid"
                got[prune] = cold[1]
            if layout == "range":
                assert got[True] < got[False], got
            print(f"{layout}: identical rows; cold GETs plain {got[False]}, pruned {got[True]}")


if __name__ == "__main__":
    main()

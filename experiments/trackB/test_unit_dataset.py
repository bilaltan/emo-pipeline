#!/usr/bin/env python3
"""
Check for utils.partition_layout.unit_dataset on toy parquet files (seconds): for every unit and a few unit groups it
returns the same rows as a scan of the whole dataset, and it keeps only the files whose community range holds a unit.

    python experiments/trackB/test_unit_dataset.py
"""
import os
import sys
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.partition_layout import unit_dataset  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    comm = np.sort(rng.integers(0, 40, 5000))           # range layout: sorted by community, split into files
    with tempfile.TemporaryDirectory() as tmp:
        for k, part in enumerate(np.array_split(np.arange(len(comm)), 8)):
            t = pa.table({"id": part.astype(np.int64), "community_id": comm[part].astype(np.int64)})
            pq.write_table(t, os.path.join(tmp, f"part-{k}.parquet"), row_group_size=200)
        full = ds.dataset(tmp, format="parquet")
        groups = [[c] for c in range(-1, 41)] + [[0, 39], [5, 6, 7], [3, 30, 17]]
        for g in groups:
            f = ds.field("community_id").isin(g)
            a = sorted(full.to_table(filter=f)["id"].to_pylist())
            sub = unit_dataset(full, g)
            b = sorted(sub.to_table(filter=f)["id"].to_pylist())
            assert a == b, g
            files = len(list(sub.get_fragments()))
            want = sum(1 for fr in full.get_fragments()
                       if any(lo <= c <= hi for c in g for lo, hi in [(pq.read_table(fr.path)["community_id"].to_numpy().min(),
                                                                          pq.read_table(fr.path)["community_id"].to_numpy().max())]))
            assert files == want, (g, files, want)
        single = [len(list(unit_dataset(full, [c]).get_fragments())) for c in range(40)]
        assert max(single) <= 2 < 8, single
        # row groups: a single unit keeps exactly the row groups whose community range holds it
        rgs = [(rg.statistics["community_id"]["min"], rg.statistics["community_id"]["max"])
               for fr in full.get_fragments() for rg in fr.row_groups]
        for c in range(40):
            kept = sum(fr.num_row_groups for fr in unit_dataset(full, [c]).get_fragments())
            assert kept == sum(lo <= c <= hi for lo, hi in rgs), (c, kept)
    with tempfile.TemporaryDirectory() as tmp:
        # hive: the predicate prunes directories; the files hold no community_id, so whole files are kept
        pq.write_to_dataset(pa.table({"id": np.arange(len(comm), dtype=np.int64), "community_id": comm.astype(np.int64)}),
                            tmp, partition_cols=["community_id"])
        hive = ds.dataset(tmp, format="parquet", partitioning="hive")
        for g in groups:
            f = ds.field("community_id").isin(g)
            rows = lambda d: sorted(zip(*d.to_table(filter=f).to_pydict().values()))  # noqa: E731
            sub = unit_dataset(hive, g, f)
            assert rows(sub) == rows(hive), g
            assert len(list(sub.get_fragments())) == len(set(g) & set(comm.tolist())), g
        assert unit_dataset(hive, [3]) is hive
    print("unit_dataset: identical rows for", len(groups), "unit groups (range and hive); files kept per single unit max",
          max(single), "of 8; row groups kept = row groups holding the unit")


if __name__ == "__main__":
    main()

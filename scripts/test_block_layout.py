#!/usr/bin/env python3
"""
Track B storage layout in the pipeline: Phase 2's block-ordered write (_blk, utils.partition_layout.add_blk) must give
every Stage 3b block read the same rows as Phase 2's plain order, at a read cap finer than, equal to and coarser than
the cap it was written for. Also checks blk_range against brute force and the √-rule row group.

    bash scripts/wsl_run.sh python scripts/test_block_layout.py
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_pipeline_e2e as E  # noqa: E402  (synthetic graph, paths, temp store)
from utils.partition_layout import bit_reverse, blk_range, sqrt_rg_mb  # noqa: E402

WRITE_CAP = 16


def check_ranges():
    for k in range(5):
        n_w = 1 << k
        for n_r in (1 << r for r in range(7)):
            for b in range(n_r):
                lo, hi = blk_range(b, n_r, n_w)
                ids = [x for x in range(256) if x & (n_r - 1) == b]
                via_blk = [x for x in range(256) if lo <= bit_reverse(x & (n_w - 1), k) <= hi]
                assert set(ids) == set(via_blk) if n_r <= n_w else set(ids) <= set(via_blk), (k, n_r, b)
    assert sqrt_rg_mb(2739 * 2**20, 1827) == 0.5 and sqrt_rg_mb(10, 10**6) == 0.25
    assert sqrt_rg_mb(2739 * 2**20, 1827, floor_mb=8) == 8 and sqrt_rg_mb(2739 * 2**20, 1827, floor_mb=0.25) == 0.5


def check_bins():
    """Phase 3b's bins (stage3b_bins, also Table A's): unchanged without a plan; coalesced, every work row once, the
    shared directory's unsplit units in contiguous community_id runs of at most cap rows, the rest one bin each."""
    from utils.partition_layout import stage3b_bins as bins
    items = [(c, 2 if c == 0 else 1, 1) for c in range(3000)]           # 0: split, 1-9 own dirs, 10+: shared m0/m1
    rows = [(c, b, n, w) for c, n, w in items for b in range(n)]
    assert bins(items) == [rows[i::1000] for i in range(1000)]
    assert len(bins(items, slots=500)) == 2000 and len(bins(items[1:2001])) == 2000   # 4 x slots bins; one per row
    c2p = {c: f"c{c}" if c < 10 else f"m{c // 1500}" for c in range(3000)}
    got = bins(items, c2p, {c: 1 + c % 3 for c in range(10, 3000)}, cap=100)
    assert sorted(r for b in got for r in b) == sorted(rows)
    packs = [b for b in got if c2p[b[0][0]].startswith("m")]
    assert all(len({c2p[c] for c, *_ in b}) == 1 and sum(1 + c % 3 for c, *_ in b) <= 100 and
               [c for c, *_ in b] == list(range(b[0][0], b[-1][0] + 1)) for b in packs)
    assert len(got) == len(packs) + 11   # 2 blocks of community 0 + 9 own-directory units, one bin each


def read_all(P3, spark, p_alg, read_cap, write_cap):
    """Every Stage 3b block of every community: sorted node ids and edges."""
    from pipeline.utils.common import unit_blocks
    from pipeline.utils.partition_layout import table_layout
    P3._DS_CACHE.clear()
    layout = table_layout(spark, p_alg)
    counts = spark.read.format("delta").load(p_alg["p2_nodes"]).groupBy("community_id").count().collect()
    out = {}
    for c, k in sorted((int(r[0]), int(r[1])) for r in counts):
        n_r, n_w = unit_blocks(k, read_cap), unit_blocks(k, write_cap) if write_cap else 0
        for b in range(n_r):
            nodes, edges = P3._load_communities_data_batch(p_alg["p2_nodes"], p_alg["p2_edges"], [c], block=b,
                                                           n_blocks=n_r, layout=layout, n_wblocks=n_w)
            assert "_blk" not in nodes.columns and "_blk" not in edges.columns
            out[(c, b, n_r)] = (sorted(nodes["id"].tolist()), sorted(zip(edges["src"], edges["dst"])))
    return out


def main():
    check_ranges()
    check_bins()
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession
    from phases.phase2_subgraph import run_phase2
    import phases.phase3b_caan as P3
    from pipeline.utils.partition_layout import block_cap

    builder = (SparkSession.builder.appName("lakegrl-blk").master("local[2]")
               .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
               .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
               .config("spark.driver.memory", "3g").config("spark.sql.shuffle.partitions", "4"))
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        E.build_graph(spark)
        p_alg = E.paths("synth", "louvain")
        caps = (WRITE_CAP // 2, WRITE_CAP, WRITE_CAP * 3, 10_000)   # finer, equal, coarser, one block
        kw = dict(use_global_mapping=True, min_size=10, get_paths_fn=E.paths, timing={},
                  expand_boundary_nodes=True, tiny_comm_handling="drop", force_rerun=True)
        run_phase2(spark, spark.sparkContext, ["synth"], ["louvain"], results={}, **kw)
        assert block_cap(spark, p_alg["p2_nodes"]) == 0
        plain = {c: read_all(P3, spark, p_alg, c, 0) for c in caps}
        r2 = {}
        run_phase2(spark, spark.sparkContext, ["synth"], ["louvain"], results=r2, block_cap=WRITE_CAP,
                   rg_mb="auto", **kw)
        res = r2[("synth", "louvain")]
        assert block_cap(spark, p_alg["p2_nodes"]) == WRITE_CAP and res["rg_mb"] == 0.25, res
        assert spark.sparkContext._jsc.hadoopConfiguration().get("parquet.block.size") is None
        n_units = 0
        for c in caps:
            got = read_all(P3, spark, p_alg, c, WRITE_CAP)
            assert got == plain[c], c
            n_units += len(got)
        assert any(n > 4 for (_, _, n) in plain[WRITE_CAP]), "communities must split at the write cap"
        print(f"ok: {n_units} block reads identical in both orders at read caps {caps}; units {res['n_units']}")
    finally:
        spark.stop()
        shutil.rmtree(E.ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Stage 3b's reads through utils/store_io.py on a synthetic graph: the emulated S3 store returns the same rows as the
plain one, every unit's GETs and bytes are counted (per block and per bin), each GET waits its latency, plain reads
count nothing, and emulated latency on a real S3 URL is refused.

    bash scripts/wsl_run.sh python scripts/test_store_io.py
"""
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_pipeline_e2e as E  # noqa: E402  (synthetic graph, paths, temp store)
from pipeline.utils import store_io as S  # noqa: E402

CAP = 16
LAT_MS = 5


def check_spec():
    assert S.parse_emu(None) is None and S.parse_emu("count") == (0.0, 0.0) and S.parse_emu("20,80") == (20.0, 80.0)
    for bad in ("20", "20,0", "-1,80"):
        try:
            S.parse_emu(bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    S.check("s3://bucket/t", (0.0, 0.0))
    try:
        S.check("s3a://bucket/t", (20.0, 80.0))
        raise AssertionError("latency on real S3 accepted")
    except ValueError:
        pass
    assert S.to_path("s3a://b/k/") == "b/k" and S.to_path("file:///home/x%20y") == "/home/x y"


def units(res):
    df = res[("synth", "louvain", "sage")]
    df = df[df["community_id"] >= 0]
    return df, sorted(map(tuple, df[["community_id", "n_nodes", "n_edges", "n_test"]].astype(int).values.tolist()))


def main():
    check_spec()
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession
    from phases.phase2_subgraph import run_phase2
    from phases.phase3b_caan import run_phase3b
    import phases.phase3b_caan as P3
    from pipeline.utils.partition_layout import table_layout

    builder = (SparkSession.builder.appName("lakegrl-store-io").master("local[2]")
               .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
               .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
               .config("spark.driver.memory", "3g").config("spark.sql.shuffle.partitions", "4")
               .config("spark.sql.execution.arrow.pyspark.enabled", "true"))
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        E.build_graph(spark)
        p_alg = E.paths("synth", "louvain")
        run_phase2(spark, spark.sparkContext, ["synth"], ["louvain"], use_global_mapping=True, min_size=10,
                   get_paths_fn=E.paths, timing={}, results={}, expand_boundary_nodes=True,
                   tiny_comm_handling="drop", force_rerun=True, block_cap=CAP)

        # driver-side loader: same rows, GETs counted, latency paid
        comm = int(spark.read.format("delta").load(p_alg["p2_nodes"]).first()["community_id"])

        def read(emu):
            P3._DS_CACHE.clear()
            lay = table_layout(spark, p_alg, emu=emu)
            nodes, edges = P3._load_communities_data_batch(p_alg["p2_nodes"], p_alg["p2_edges"], [comm], layout=lay)
            return sorted(nodes["id"].tolist()), sorted(zip(edges["src"], edges["dst"]))
        g0 = S.STATS["gets"]
        plain = read(None)
        assert S.STATS["gets"] == g0, "plain reads were metered"
        t0 = time.time()
        emu = read((float(LAT_MS), 1000.0))
        gets, dt = S.STATS["gets"] - g0, time.time() - t0
        assert emu == plain and plain[0], "emulated store returned other rows"
        assert gets > 0 and dt >= LAT_MS / 1e3, (gets, dt)

        # Stage 3b end to end: per block (cap) and per bin (no cap)
        cfg = {"synth": {"in_feats": E.FEAT_DIM, "num_classes": E.N_CLASSES}}
        kw = dict(use_global_mapping=True, dataset_cfg=cfg, get_paths_fn=E.paths, timing={}, models=["sage"],
                  gcn_cfg={"hidden_dim": 16, "num_epochs": 3, "lr": 0.01, "dropout": 0.3, "weight_decay": 5e-4,
                           "seed": 0, "probe_head": False},
                  task_type="node_classification", min_size=10, force_rerun=True, local_data_dir=E.ROOT,
                  context="macro", tiny_comm_handling="drop")
        runs = {}
        for name, emu, cap in (("emu", (1.0, 1000.0), CAP), ("plain", None, CAP), ("count_bins", (0.0, 0.0), 0)):
            res = {}
            run_phase3b(spark, spark.sparkContext, ["synth"], ["louvain"], results=res, s3_emu=emu,
                        max_nodes_per_community=cap, **kw)
            runs[name] = units(res) + (res[("synth", "louvain", "sage")].attrs,)
        df, rows, attrs = runs["emu"]
        assert len(df) > df["community_id"].nunique(), "no community was split into blocks"
        read_units = df[df["n_nodes"] > 0]
        assert (read_units["read_gets"] > 0).all() and (read_units["read_bytes"] > 0).all(), read_units.T
        assert attrs["read_gets"] == df["read_gets"].sum() > 0 and attrs["s3_emu"] == (1.0, 1000.0)
        assert runs["plain"][1] == rows, "the emulated store changed the units"
        assert runs["plain"][2]["read_gets"] == 0 and runs["plain"][2]["read_s"] > 0
        assert runs["count_bins"][2]["read_gets"] > 0
        print(f"ok: {len(df)} block units read the same rows from both stores; "
              f"{attrs['read_gets']:.0f} GETs, {attrs['read_bytes'] / 1e6:.1f} MB, bins counted too")
    finally:
        spark.stop()
        shutil.rmtree(E.ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()

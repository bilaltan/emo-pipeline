#!/usr/bin/env python3
"""
Phase 2 -> Stage 3b on a synthetic graph with minor communities (CaaN Eq. 1): with tiny_comm 'drop' the minor
nodes leave the units and only CaaN's global model scores them, so every test node is scored exactly once.

    bash scripts/wsl_run.sh python scripts/test_phase3b_minor.py
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_pipeline_e2e as E  # noqa: E402  (synthetic graph, paths, temp store)
from delta import configure_spark_with_delta_pip  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402

from phases.phase2_subgraph import run_phase2  # noqa: E402
from phases.phase3b_caan import run_phase3b  # noqa: E402

N_MINOR = 12   # vertices 0..11 become six 2-vertex communities, below min_size


def main():
    builder = (SparkSession.builder.appName("lakegrl-3b-minor").master("local[2]")
               .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
               .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
               .config("spark.driver.memory", "3g")
               .config("spark.sql.shuffle.partitions", "4")
               .config("spark.sql.execution.arrow.pyspark.enabled", "true"))
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        n = E.build_graph(spark)
        comm = [(i, 100 + i // 2 if i < N_MINOR else i // E.PER_COMM) for i in range(n)]
        spark.createDataFrame(comm, "id long, community_id long").write.format("delta").mode("overwrite") \
             .save(E.paths("synth", "louvain")["communities"])
        n_test = sum(1 for i in range(n) if i % 5 == 4)
        n_test_minor = sum(1 for i in range(N_MINOR) if i % 5 == 4)

        cfg = {"synth": {"in_feats": E.FEAT_DIM, "num_classes": E.N_CLASSES}}
        run_phase2(spark, spark.sparkContext, ["synth"], ["louvain"], use_global_mapping=True,
                   min_size=10, get_paths_fn=E.paths, timing={}, results={},
                   expand_boundary_nodes=True, tiny_comm_handling="drop", force_rerun=True)
        # minor vertices are not units of their own but stay as the halo of the major units they touch
        p2 = spark.read.format("delta").load(E.paths("synth", "louvain")["p2_nodes"]).filter(f"id < {N_MINOR}")
        assert p2.filter("is_member").count() == 0 and p2.select("id").distinct().count() == N_MINOR
        res = {}
        run_phase3b(spark, spark.sparkContext, ["synth"], ["louvain"], use_global_mapping=True, dataset_cfg=cfg,
                    gcn_cfg={"hidden_dim": 32, "num_epochs": 20, "lr": 0.01, "dropout": 0.3,
                             "weight_decay": 5e-4, "seed": 0, "probe_head": False},
                    get_paths_fn=E.paths, timing={}, results=res, models=["sage"],
                    task_type="node_classification", min_size=10, force_rerun=True, local_data_dir=E.ROOT,
                    context="macro", tiny_comm_handling="drop")
        df = res[("synth", "louvain", "sage")]
        glob = df[df["community_id"] == -1]
        assert len(glob) == 1, df[["community_id", "n_nodes", "n_test"]]
        assert int(glob["n_nodes"].iloc[0]) == N_MINOR, glob.T
        assert int(glob["n_test"].iloc[0]) == n_test_minor, glob.T
        assert int(df["n_test"].sum()) == n_test, (int(df["n_test"].sum()), n_test)
        assert 0.0 <= float(glob["comm_test_acc"].iloc[0]) <= 1.0
        print(f"ok: {len(df) - 1} units + global model, {int(df['n_test'].sum())} of {n_test} test nodes scored once")

        # both ways of scoring minor nodes are recorded; every minor vertex is a halo row of a trained unit
        a = df.attrs
        assert a["minor_score"] == "global" and a["acc_minor_global"] == a["weighted_comm_acc"], a
        assert a["minor_test_reached"] == 1.0 and 0.0 <= a["acc_minor_unit"] <= 1.0, a
        assert 0.0 < a["weighted_comm_val_acc"] <= 1.0, a
        held = {int(t.split(",")[0]) for s in df.loc[df["community_id"] >= 0, "minor_scores"] if s
                for t in s.split(";")}
        assert held == {i for i in range(N_MINOR) if i % 5 >= 3}, held   # the minor val and test vertices

        # tiny units: blocks without a training label are merged with a sibling, so no unit with test nodes is
        # left untrained; and the reported accuracy follows minor_score
        res = {}
        run_phase3b(spark, spark.sparkContext, ["synth"], ["louvain"], use_global_mapping=True, dataset_cfg=cfg,
                    gcn_cfg={"hidden_dim": 16, "num_epochs": 5, "lr": 0.01, "dropout": 0.3,
                             "weight_decay": 5e-4, "seed": 0, "probe_head": False},
                    get_paths_fn=E.paths, timing={}, results=res, models=["sage"],
                    task_type="node_classification", min_size=10, force_rerun=True, local_data_dir=E.ROOT,
                    context="macro", tiny_comm_handling="drop", max_nodes_per_community=2, minor_score="unit")
        df = res[("synth", "louvain", "sage")]
        units = df[(df["community_id"] >= 0) & (df["n_test"] > 0)]
        assert len(units) > E.N_COMM and (units["n_train"] > 0).all(), units[["community_id", "n_train", "n_test"]]
        assert df.attrs["weighted_comm_acc"] == df.attrs["acc_minor_unit"], df.attrs
        print(f"ok: minor nodes scored by units too ({a['acc_minor_unit']:.3f} vs global {a['acc_minor_global']:.3f}); "
              f"{len(units)} tiny units, all with training labels")
    finally:
        spark.stop()
        shutil.rmtree(E.ROOT, ignore_errors=True)


def test_score_minor():
    import pandas as pd
    from phases.phase3b_caan import _score_minor
    # node 7 only the global model holds; 8 and 9 go to the unit with the most edges (unit 1, then unit 2)
    comm_pd = pd.DataFrame({"community_id": [-1, 1, 2], "n_test": [3, 10, 10], "comm_test_acc": [1 / 3, .5, .5],
                            "n_val": [1, 4, 4], "comm_val_acc": [0., .5, .5],
                            "minor_scores": ["7,2,1,-1;8,2,0,-1;9,2,0,-1;6,1,0,-1", "8,2,1,3;9,2,0,1;6,1,1,1",
                                             "9,2,1,2"]})
    r = _score_minor(comm_pd)
    assert abs(r["acc_minor_global"] - 11 / 23) < 1e-12 and abs(r["acc_minor_unit"] - 13 / 23) < 1e-12, r
    assert abs(r["val_acc_minor_global"] - 4 / 9) < 1e-12 and abs(r["val_acc_minor_unit"] - 5 / 9) < 1e-12, r
    assert abs(r["minor_test_reached"] - 2 / 3) < 1e-12, r
    print("ok: _score_minor picks the unit with most edges and falls back to the global model")


if __name__ == "__main__":
    test_score_minor()
    main()

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
    finally:
        spark.stop()
        shutil.rmtree(E.ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()

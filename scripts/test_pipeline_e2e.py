#!/usr/bin/env python3
"""
Local end-to-end Spark run: Phase 1 -> Phase 2 -> Phase 3 on a synthetic graph.

Exercises the parts of the pipeline the Spark-free harness cannot reach — the
Delta reads/writes, the is_member column threading from Phase 2 into the Phase 3
UDF, the collect_list bundling, and the groupBy/applyInPandas execution — so that
edits to those paths are known to run before a cluster job depends on them.

    JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
    PYSPARK_PYTHON=$PWD/.venv-spark/bin/python \
    PYSPARK_DRIVER_PYTHON=$PWD/.venv-spark/bin/python \
    .venv-spark/bin/python scripts/test_pipeline_e2e.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

from phases.phase2_subgraph import run_phase2
from phases.phase3_training import run_phase3

ROOT = tempfile.mkdtemp(prefix="lakegrl_e2e_")
N_COMM, PER_COMM, FEAT_DIM, N_CLASSES = 4, 60, 16, 3


def paths(dataset, alg=None):
    base = f"file://{ROOT}/{dataset}"
    p = {"nodes": f"{base}/nodes/", "edges": f"{base}/edges/", "masks": f"{base}/masks/"}
    if alg:
        p.update({
            "communities": f"{base}/communities/{alg}/",
            "p2_nodes": f"{base}/p2_nodes/{alg}/",
            "p2_edges": f"{base}/p2_edges/{alg}/",
            "tag": f"e2e_{dataset}_{alg}",
        })
    return p


def build_graph(spark):
    """Planted-partition graph: dense within a community, sparse across."""
    rng = np.random.default_rng(7)
    n = N_COMM * PER_COMM
    comm = np.repeat(np.arange(N_COMM), PER_COMM)
    # Features carry the community signal so the task is learnable at all.
    x = rng.normal(comm[:, None] * 1.5, 1.0, size=(n, FEAT_DIM))
    y = comm % N_CLASSES

    edges = set()
    for c in range(N_COMM):
        members = np.where(comm == c)[0]
        for _ in range(PER_COMM * 6):
            u, v = rng.choice(members, 2, replace=False)
            edges.add((int(min(u, v)), int(max(u, v))))
    for _ in range(N_COMM * 8):                       # cross-community cut edges
        u, v = rng.choice(n, 2, replace=False)
        if comm[u] != comm[v]:
            edges.add((int(min(u, v)), int(max(u, v))))

    edges = sorted(edges)
    edges = edges + [(d, s) for s, d in edges]
    splits = ["train" if i % 5 < 3 else ("valid" if i % 5 == 3 else "test") for i in range(n)]

    p = paths("synth")
    spark.createDataFrame([(int(i), int(y[i]), [float(v) for v in x[i]]) for i in range(n)],
                          "id long, label long, features array<double>") \
         .write.format("delta").mode("overwrite").save(p["nodes"])
    spark.createDataFrame([(int(s), int(d)) for s, d in edges], "src long, dst long") \
         .write.format("delta").mode("overwrite").save(p["edges"])
    spark.createDataFrame([(int(i), splits[i]) for i in range(n)], "id long, split string") \
         .write.format("delta").mode("overwrite").save(p["masks"])
    spark.createDataFrame([(int(i), int(comm[i])) for i in range(n)], "id long, community_id long") \
         .write.format("delta").mode("overwrite").save(paths("synth", "louvain")["communities"])
    return n


def main():
    builder = (SparkSession.builder.appName("lakegrl-e2e").master("local[2]")
               .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
               .config("spark.sql.catalog.spark_catalog",
                       "org.apache.spark.sql.delta.catalog.DeltaCatalog")
               .config("spark.driver.memory", "3g")
               .config("spark.sql.shuffle.partitions", "4")
               .config("spark.sql.execution.arrow.pyspark.enabled", "true"))
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    n = build_graph(spark)
    print(f"\n  synthetic graph: {n} nodes, {N_COMM} planted communities, {N_CLASSES} classes")

    timing, p2_results, p3_results = {}, {}, {}

    run_phase2(spark, spark.sparkContext, ["synth"], ["louvain"], use_global_mapping=True,
               min_size=10, get_paths_fn=paths, timing=timing, results=p2_results,
               expand_boundary_nodes=True, tiny_comm_handling="drop", force_rerun=True)

    run_phase3(spark, spark.sparkContext, ["synth"], ["louvain"], use_global_mapping=True,
               dataset_cfg={"synth": {"in_feats": FEAT_DIM, "num_classes": N_CLASSES}},
               gcn_cfg={"hidden_dim": 32, "num_epochs": 5, "lr": 0.01, "dropout": 0.3},
               get_paths_fn=paths, timing=timing, results=p3_results,
               models=["sage"], task_type="both", force_rerun=True,
               max_nodes_per_community=int(os.environ.get("E2E_MAX_NODES", 10000)),
               max_edges_per_community=30000,
               edge_sample_modulus=1, mlp_epochs=3, local_data_dir=ROOT)

    print(f"\n{'-'*66}\n  END-TO-END CHECKS\n{'-'*66}")
    checks = []
    key = ("synth", "louvain", "sage")
    df = p3_results.get(key)
    # With halo expansion a community's frame exceeds PER_COMM, so the exact block
    # count depends on halo size. Assert the property that matters: a cap below the
    # community size must split it, and no cap must leave it intact.
    cap = int(os.environ.get("E2E_MAX_NODES", 10000))
    splitting = cap < PER_COMM
    checks.append(("Phase 3 produced a result frame", df is not None and len(df) > 0,
                   f"{0 if df is None else len(df)} row(s)"))
    if df is not None:
        a = df.attrs
        units = int(a.get("n_communities", 0))
        checks.append((("oversized communities split into blocks" if splitting
                        else "one unit per community when under the cap"),
                       (units > N_COMM) if splitting else (units == N_COMM),
                       f"{units} units from {N_COMM} communities (cap {cap})"))
        checks.append(("every vertex retained across units",
                       int(a.get("total_nodes", 0) or df['n_nodes'].iloc[0]) >= n,
                       f"{int(df['n_nodes'].iloc[0])} of {n} vertices"))
        checks.append(("weighted community accuracy is a real number",
                       0.0 < float(a.get("weighted_comm_acc", 0)) <= 1.0,
                       f"{a.get('weighted_comm_acc', 0):.4f}"))
        checks.append(("link AUC is not the 0.5 stub",
                       abs(float(a.get("weighted_comm_link_auc", 0.5)) - 0.5) > 1e-9,
                       f"{a.get('weighted_comm_link_auc', 0.5):.4f}"))

    p2 = spark.read.format("delta").load(paths("synth", "louvain")["p2_nodes"])
    cols = set(p2.columns)
    checks.append(("Phase 2 emits is_member", "is_member" in cols, ", ".join(sorted(cols))))
    if "is_member" in cols:
        halo = p2.filter(~p2.is_member).count()
        bnd = p2.filter(p2.is_boundary).count()
        checks.append(("halo vertices present and marked", halo > 0, f"{halo} halo rows"))
        checks.append(("boundary vertices detected under halo expansion", bnd > 0,
                       f"{bnd} boundary rows"))

    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<48s} {detail}")
        failed += (not ok)
    print(f"\n  {len(checks)-failed}/{len(checks)} end-to-end checks passed")

    spark.stop()
    shutil.rmtree(ROOT, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

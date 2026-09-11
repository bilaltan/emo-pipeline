#!/usr/bin/env python3
"""
Local Spark regression test for Phase 2's halo/member separation.

Builds a tiny two-community graph with a single cut edge, runs the real
run_phase2 with 1-hop boundary expansion on, and asserts that borrowed halo
vertices are marked is_member=False while a community's own vertices are True.

Without that distinction a boundary vertex is scored once per community that
borrows it, which inflates both node accuracy and link AUC.

    JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
    PYSPARK_PYTHON=$PWD/.venv-spark/bin/python \
    .venv-spark/bin/python scripts/test_phase2_halo.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

from phases.phase2_subgraph import run_phase2

ROOT = tempfile.mkdtemp(prefix="lakegrl_p2_test_")


def paths(dataset, alg=None):
    base = f"file://{ROOT}/{dataset}"
    p = {"nodes": f"{base}/nodes/", "edges": f"{base}/edges/", "masks": f"{base}/masks/"}
    if alg:
        p.update({
            "communities": f"{base}/communities/{alg}/",
            "p2_nodes": f"{base}/p2_nodes/{alg}/",
            "p2_edges": f"{base}/p2_edges/{alg}/",
            "tag": f"test_{dataset}_{alg}",
        })
    return p


def main():
    builder = (SparkSession.builder.appName("phase2-halo-test").master("local[2]")
               .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
               .config("spark.sql.catalog.spark_catalog",
                       "org.apache.spark.sql.delta.catalog.DeltaCatalog")
               .config("spark.driver.memory", "2g")
               .config("spark.sql.shuffle.partitions", "4"))
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Community 0 = {0,1,2}, community 1 = {3,4,5}, single cut edge 2--3.
    nodes = [(i, i % 2, [float(i), 1.0]) for i in range(6)]
    edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
    edges = edges + [(d, s) for s, d in edges]          # symmetrize
    comms = [(i, 0 if i < 3 else 1) for i in range(6)]

    p = paths("toy")
    spark.createDataFrame(nodes, "id long, label long, features array<double>") \
         .write.format("delta").mode("overwrite").save(p["nodes"])
    spark.createDataFrame(edges, "src long, dst long") \
         .write.format("delta").mode("overwrite").save(p["edges"])
    spark.createDataFrame([(i, "train" if i % 3 else "test") for i in range(6)],
                          "id long, split string") \
         .write.format("delta").mode("overwrite").save(p["masks"])
    spark.createDataFrame(comms, "id long, community_id long") \
         .write.format("delta").mode("overwrite").save(paths("toy", "louvain")["communities"])

    run_phase2(spark, spark.sparkContext, ["toy"], ["louvain"],
               use_global_mapping=True, min_size=1, get_paths_fn=paths,
               timing={}, results={}, expand_boundary_nodes=True,
               tiny_comm_handling="drop", force_rerun=True)

    out = spark.read.format("delta").load(paths("toy", "louvain")["p2_nodes"]) \
               .select("id", "community_id", "is_member", "is_boundary") \
               .orderBy("community_id", "id").collect()

    print("\n  community_id  id  is_member  is_boundary")
    for r in out:
        print(f"  {r.community_id:>12}{r.id:>4}{str(r.is_member):>11}{str(r.is_boundary):>13}")

    by_comm = {}
    for r in out:
        by_comm.setdefault(r.community_id, {})[r.id] = r.is_member

    failures = []
    # Vertex 3 belongs to community 1 but is pulled into community 0 as halo.
    if by_comm.get(0, {}).get(3) is not False:
        failures.append("vertex 3 should be halo (is_member=False) in community 0")
    if by_comm.get(1, {}).get(2) is not False:
        failures.append("vertex 2 should be halo (is_member=False) in community 1")
    for c, own in ((0, [0, 1, 2]), (1, [3, 4, 5])):
        for v in own:
            if by_comm.get(c, {}).get(v) is not True:
                failures.append(f"vertex {v} should be a member of community {c}")

    print()
    if failures:
        for f in failures:
            print(f"  [FAIL] {f}")
    else:
        print("  [PASS] halo vertices marked is_member=False, members marked True")

    spark.stop()
    shutil.rmtree(ROOT, ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

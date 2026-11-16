#!/usr/bin/env python3
"""
Regression test for the two partition-table forms and the global join.

On a tiny two-community graph with 1-hop halo expansion this checks:

  1. Phase 2's default still writes the materialized partition table (the
     LakeGRL default): p2_nodes carries label/features/split with the partition;
  2. partition_table_mode='normalized' writes only id, community_id,
     is_boundary, is_member;
  3. the global join (join_features / load_unit_nodes) of the normalized table
     returns exactly the rows of the materialized table;
  4. the normalized commit pins the raw-table versions it was built from, and a
     pinned join keeps returning them after the raw nodes table is rewritten;
  5. attach_units() sends every joined row to the (community, block) unit and
     bin that the direct loader's scan predicate would select;
  6. (if pyarrow is installed) the direct per-bin loader on the materialized
     table and the global-join shuffle (groupBy(bin).cogroup(manifest)) deliver
     the same rows to every bin.

    PYSPARK_PYTHON=$PWD/.venv-spark/bin/python .venv-spark/bin/python scripts/test_partition_join.py
Set LAKEGRL_SPARK_JARS to local Delta jars to skip the Maven download.
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from phases.phase2_subgraph import run_phase2
from utils.partition_join import (MATERIALIZED_COLUMNS, PARTITION_COLUMNS, attach_units,
                                  join_features, load_unit_nodes, pinned_versions,
                                  source_versions)

ROOT = tempfile.mkdtemp(prefix="lakegrl_pjoin_test_")


def paths(dataset, alg=None):
    base = f"file://{ROOT}/{dataset}"
    p = {"nodes": f"{base}/nodes/", "edges": f"{base}/edges/", "masks": f"{base}/masks/"}
    if alg:
        p.update({"communities": f"{base}/communities/{alg}/",
                  "p2_nodes": f"{base}/p2_nodes/{alg}/",
                  "p2_edges": f"{base}/p2_edges/{alg}/",
                  "tag": f"test_{dataset}_{alg}"})
    return p


def spark_session():
    # Spark 3.5 / Hadoop 3.3 need Java 17: newer JDKs fail at startup with
    # "getSubject is not supported". Same JDK as runners/run_local.py.
    if sys.platform == "darwin" and os.path.isdir("/opt/homebrew/opt/openjdk@17"):
        os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    b = (SparkSession.builder.appName("partition-join-test").master("local[2]")
         .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
         .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
         .config("spark.driver.host", "127.0.0.1")
         .config("spark.driver.memory", "1g")
         .config("spark.ui.enabled", "false")
         .config("spark.sql.shuffle.partitions", "4"))
    jars = os.environ.get("LAKEGRL_SPARK_JARS")
    if jars:
        spark = b.config("spark.jars", jars).getOrCreate()
    else:
        from delta import configure_spark_with_delta_pip
        spark = configure_spark_with_delta_pip(b).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _rows(df):
    return sorted((int(r.id), int(r.community_id), r.label, tuple(r.features), r.split,
                   bool(r.is_boundary), bool(r.is_member)) for r in df.collect())


def main():
    spark = spark_session()
    failures = []

    def check(ok, msg):
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        if not ok:
            failures.append(msg)

    # Community 0 = {0,1,2}, community 1 = {3,4,5}, single cut edge 2--3.
    nodes = [(i, i % 2, [float(i), 10.0 + i]) for i in range(6)]
    edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]
    edges = edges + [(d, s) for s, d in edges]
    p = paths("toy")
    mv, nm = paths("toy", "mv"), paths("toy", "norm")
    spark.createDataFrame(nodes, "id long, label int, features array<float>") \
         .write.format("delta").mode("overwrite").save(p["nodes"])
    spark.createDataFrame(edges, "src long, dst long").write.format("delta").mode("overwrite").save(p["edges"])
    spark.createDataFrame([(i, "train" if i % 3 else "test") for i in range(6)], "id long, split string") \
         .write.format("delta").mode("overwrite").save(p["masks"])
    for pa_ in (mv, nm):
        spark.createDataFrame([(i, 0 if i < 3 else 1) for i in range(6)], "id long, community_id long") \
             .write.format("delta").mode("overwrite").save(pa_["communities"])

    common = dict(use_global_mapping=True, min_size=1, get_paths_fn=paths, timing={},
                  expand_boundary_nodes=True, tiny_comm_handling="drop", force_rerun=True)
    run_phase2(spark, spark.sparkContext, ["toy"], ["mv"], results={}, **common)
    run_phase2(spark, spark.sparkContext, ["toy"], ["norm"], results={},
               partition_table_mode="normalized", **common)

    mv_tab = spark.read.format("delta").load(mv["p2_nodes"])
    nm_tab = spark.read.format("delta").load(nm["p2_nodes"])
    check(tuple(mv_tab.columns) == MATERIALIZED_COLUMNS,
          f"default Phase 2 writes the materialized table {MATERIALIZED_COLUMNS} (got {tuple(mv_tab.columns)})")
    check(tuple(nm_tab.columns) == PARTITION_COLUMNS,
          f"normalized Phase 2 writes only {PARTITION_COLUMNS} (got {tuple(nm_tab.columns)})")

    mv_rows = _rows(load_unit_nodes(spark, mv))
    nm_rows = _rows(load_unit_nodes(spark, nm))
    check(mv_rows == _rows(mv_tab), "load_unit_nodes reads a materialized table as is")
    check(nm_rows == mv_rows and len(nm_rows) == mv_tab.count(),
          f"global join of the normalized table returns the materialized rows ({len(nm_rows)} rows)")
    check({(r[0], r[1], r[6]) for r in nm_rows} >= {(3, 0, False), (2, 1, False)},
          "halo rows (3 in community 0, 2 in community 1) are joined with is_member=False")

    pinned = pinned_versions(spark, nm)
    check(pinned == source_versions(spark, p) and set(pinned) == {"nodes", "masks", "edges"},
          f"normalized partition commit pins the raw versions it was built from ({pinned})")

    # Units: community 0 split into 2 blocks (id & 1), community 1 whole; 3 bins.
    work = pd.DataFrame({"community_id": [0, 0, 1], "_block": [0, 1, 0],
                         "_n_blocks": [2, 2, 1], "bin_id": [0, 1, 2]})
    tagged = attach_units(spark, load_unit_nodes(spark, nm), work).collect()
    ok = len(tagged) == len(nm_rows) and all(
        r._block == (r.id & 1 if r.community_id == 0 else 0)
        and r.bin_id == {(0, 0): 0, (0, 1): 1, (1, 0): 2}[(r.community_id, r._block)]
        for r in tagged)
    check(ok, "attach_units sends each row to the block (id & (n-1)) and bin of its unit")

    try:
        import pyarrow  # noqa: F401
        have_pa = True
    except Exception:
        have_pa = False
    if have_pa:
        from phases.phase3b_caan import _load_communities_data_batch, _load_unit_edges
        mv_url, edges_url = mv["p2_nodes"], mv["p2_edges"]

        def direct(c, b, n):
            nodes_pdf, _ = _load_communities_data_batch(mv_url, edges_url, [c], block=b, n_blocks=n)
            # float(x): with numpy >= 2, repr(np.float32(1.0)) is "np.float32(1.0)", not "1.0"
            return sorted((int(r.id), bool(r.is_member), tuple(float(x) for x in r.features), r.split)
                          for r in nodes_pdf.itertuples())

        def per_bin(key, nodes_pdf, manifest_pdf):
            return pd.DataFrame({"bin_id": [int(key[0])],
                                 "rows": [repr(sorted((int(r.id), bool(r.is_member),
                                                       tuple(float(x) for x in r.features), r.split)
                                                      for r in nodes_pdf.itertuples()))]})
        manifest = spark.createDataFrame(work)
        shuffled = (attach_units(spark, load_unit_nodes(spark, nm), work)
                    .repartition(3, "bin_id").groupBy("bin_id")
                    .cogroup(manifest.repartition(3, "bin_id").groupBy("bin_id"))
                    .applyInPandas(per_bin, "bin_id long, rows string").collect())
        got = {r.bin_id: r.rows for r in shuffled}
        expect = {b: repr(direct(c, k, n)) for c, k, n, b in work.itertuples(index=False)}
        check(got == expect, "global-join shuffle delivers to every bin the rows the direct loader reads")
        if got != expect:
            for b in sorted(set(got) | set(expect)):
                if got.get(b) != expect.get(b):
                    print(f"    bin {b}: shuffle {got.get(b)}\n    bin {b}: direct  {expect.get(b)}")
        e_direct = _load_communities_data_batch(mv_url, edges_url, [0])[1]
        check(len(e_direct) > 0 and e_direct.equals(_load_unit_edges(edges_url, [0])),
              f"_load_unit_edges returns the direct loader's edges ({len(e_direct)})")
    else:
        print("  [SKIP] pyarrow not installed: direct loader and cogroup delivery not exercised")

    # Rewrite the raw features: a pinned join keeps the snapshot Phase 2 used.
    raw = {r.id: r for r in spark.read.format("delta").load(p["nodes"]).collect()}
    spark.createDataFrame([(i, i % 2, [-1.0, -1.0]) for i in range(6)], "id long, label int, features array<float>") \
         .write.format("delta").mode("overwrite").save(p["nodes"])
    pinned_rows = join_features(spark, nm).collect()
    latest_rows = join_features(spark, nm, pin=False).collect()
    check(all(list(r.features) == list(raw[r.id].features) for r in pinned_rows),
          "after the raw table is rewritten, the pinned join still returns the original features")
    check(all(list(r.features) == [-1.0, -1.0] for r in latest_rows),
          "pin=False returns the rewritten features")

    spark.stop()
    shutil.rmtree(ROOT, ignore_errors=True)
    print("\n  ALL PASSED" if not failures else f"\n  {len(failures)} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

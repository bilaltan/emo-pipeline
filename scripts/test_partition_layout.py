#!/usr/bin/env python3
"""
Regression test for the Phase 2 table layouts (utils/partition_layout.py).

Toy graph: two large communities and four tiny ones, 1-hop halo, every
community kept as its own unit (tiny_comm_handling='keep'). Checks:

  1. range has no partition columns; hive is partitioned by community_id with
     one directory per community; hive_merged is partitioned by storage_part,
     with large communities in c<id> and the small ones packed into m<k>;
  2. all three layouts hold exactly the same node and edge rows;
  3. the merge plan puts every community in exactly one storage partition and no
     shared partition exceeds merge_target_rows (unless one community does);
  4. (pyarrow) Phase 3b's loader with table_layout() returns, per community, the
     rows Spark returns, in every layout;
  5. (pyarrow) after the same path is overwritten with another layout, the loader
     still returns the current rows (snapshot file list), while a plain
     directory scan also picks up the replaced files.

    PYSPARK_PYTHON=$PWD/.venv-spark/bin/python .venv-spark/bin/python scripts/test_partition_layout.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from phases.phase2_subgraph import run_phase2
from utils.partition_layout import merge_plan, partition_columns, table_layout

ROOT = tempfile.mkdtemp(prefix="lakegrl_layout_test_")


def paths(dataset, alg=None):
    base = f"file://{ROOT}/{dataset}"
    p = {"nodes": f"{base}/nodes/", "edges": f"{base}/edges/", "masks": f"{base}/masks/"}
    if alg:
        p.update({"communities": f"{base}/communities/",
                  "p2_nodes": f"{base}/p2_nodes/{alg}/", "p2_edges": f"{base}/p2_edges/{alg}/",
                  "tag": f"test_{dataset}_{alg}"})
    return p


def spark_session():
    if sys.platform == "darwin" and os.path.isdir("/opt/homebrew/opt/openjdk@17"):
        os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    b = (SparkSession.builder.appName("partition-layout-test").master("local[2]")
         .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
         .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
         .config("spark.driver.host", "127.0.0.1").config("spark.driver.memory", "1g")
         .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "4"))
    jars = os.environ.get("LAKEGRL_SPARK_JARS")
    if jars:
        spark = b.config("spark.jars", jars).getOrCreate()
    else:
        from delta import configure_spark_with_delta_pip
        spark = configure_spark_with_delta_pip(b).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def rows_of(spark, path, cols):
    df = spark.read.format("delta").load(path)
    return sorted(tuple(r) for r in df.select(*cols).collect())


def local(path):
    return path.replace("file://", "")


def main():
    spark = spark_session()
    failures = []

    def check(ok, msg):
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        if not ok:
            failures.append(msg)

    # communities: 0 = 0..5, 1 = 6..11 (large); 2 = {12,13}, 3 = {14}, 4 = {15,16}, 5 = {17} (tiny)
    comm = {i: 0 for i in range(6)} | {i: 1 for i in range(6, 12)} | {12: 2, 13: 2, 14: 3, 15: 4, 16: 4, 17: 5}
    und = [(a, b) for a in range(6) for b in range(a + 1, 6)] + \
          [(a, b) for a in range(6, 12) for b in range(a + 1, 12)] + \
          [(5, 6), (12, 13), (12, 0), (14, 1), (15, 16), (16, 7), (17, 15)]
    edges = und + [(b, a) for a, b in und]
    n = 18
    p = paths("toy")
    spark.createDataFrame([(i, i % 3, [float(i), 1.0]) for i in range(n)],
                          "id long, label int, features array<float>").write.format("delta").save(p["nodes"])
    spark.createDataFrame(edges, "src long, dst long").write.format("delta").save(p["edges"])
    spark.createDataFrame([(i, "train" if i % 2 else "test") for i in range(n)],
                          "id long, split string").write.format("delta").save(p["masks"])
    spark.createDataFrame(sorted(comm.items()), "id long, community_id long") \
         .write.format("delta").save(paths("toy", "x")["communities"])

    common = dict(use_global_mapping=True, min_size=3, get_paths_fn=paths, timing={},
                  expand_boundary_nodes=True, tiny_comm_handling="keep", force_rerun=True,
                  merge_target_rows=6)
    for layout in ("range", "hive", "hive_merged"):
        run_phase2(spark, spark.sparkContext, ["toy"], [layout], results={}, p2_layout=layout, **common)

    # tiny_comm_handling='misc' must keep tiny-community nodes (in negative-id bins), not drop them
    run_phase2(spark, spark.sparkContext, ["toy"], ["misc"], results={}, p2_layout="range",
               **{**common, "tiny_comm_handling": "misc"})
    misc = spark.read.format("delta").load(paths("toy", "misc")["p2_nodes"])
    members = {r.id: r.community_id for r in misc.filter("is_member").collect()}
    check(set(members) == set(range(n)) and all(members[i] < 0 for i in range(12, 18))
          and all(members[i] in (0, 1) for i in range(12)),
          f"misc: tiny-community nodes 12-17 kept as members of negative-id bins "
          f"({sorted(set(v for k, v in members.items() if k >= 12))})")

    pr, ph, pm = paths("toy", "range"), paths("toy", "hive"), paths("toy", "hive_merged")
    check(partition_columns(spark, pr["p2_nodes"]) == [], "range: no partition columns")
    check(partition_columns(spark, ph["p2_nodes"]) == ["community_id"]
          and partition_columns(spark, ph["p2_edges"]) == ["community_id"],
          "hive: p2_nodes and p2_edges partitioned by community_id")
    dirs = sorted(d for d in os.listdir(local(ph["p2_nodes"])) if d.startswith("community_id="))
    check(len(dirs) == 6, f"hive: one directory per community ({len(dirs)} of 6)")
    check(partition_columns(spark, pm["p2_nodes"]) == ["storage_part"], "hive_merged: partitioned by storage_part")
    parts = sorted(d.split("=", 1)[1] for d in os.listdir(local(pm["p2_nodes"])) if d.startswith("storage_part="))
    check(parts[:2] == ["c0", "c1"] and all(x.startswith("m") for x in parts[2:]) and len(parts) - 2 == 2,
          f"hive_merged: large communities own c<id>, the 4 tiny ones share m<k> ({parts})")

    ncols, ecols = ["id", "community_id", "label", "split", "is_member", "is_boundary"], ["src", "dst", "community_id"]
    ref_n, ref_e = rows_of(spark, pr["p2_nodes"], ncols), rows_of(spark, pr["p2_edges"], ecols)
    same = all(rows_of(spark, q["p2_nodes"], ncols) == ref_n and rows_of(spark, q["p2_edges"], ecols) == ref_e
               for q in (ph, pm))
    check(same and len(ref_n) > n, f"identical node ({len(ref_n)}) and edge ({len(ref_e)}) rows in all layouts")

    stats = (spark.read.format("delta").load(pr["p2_nodes"]).groupBy("community_id")
             .agg(F.count(F.lit(1)).alias("rows"), F.sum(F.col("is_member").cast("int")).alias("members"))
             .toPandas())
    plan = merge_plan(stats, merge_minor_members=3, merge_target_rows=6)
    shared = plan[plan["small"]].merge(stats, on="community_id").groupby("storage_part")["rows"].agg(["sum", "max", "count"])
    check(plan["community_id"].is_unique and set(plan.loc[~plan["small"], "community_id"]) == {0, 1}
          and all((r["sum"] <= 6) or (r["count"] == 1) for _, r in shared.iterrows()),
          f"merge plan: one part per community, shared parts <= target rows ({shared['sum'].tolist()})")

    try:
        import pyarrow  # noqa: F401
        have_pa = True
    except Exception:
        have_pa = False
    if have_pa:
        from phases.phase3b_caan import _load_communities_data_batch
        ok = True
        for q in (pr, ph, pm):
            lay = table_layout(spark, q)
            for c in range(6):
                nodes_pdf, edges_pdf = _load_communities_data_batch(q["p2_nodes"], q["p2_edges"], [c], layout=lay)
                got = sorted((int(r.id), int(r.community_id)) for r in nodes_pdf.itertuples())
                exp = sorted((r[0], r[1]) for r in ref_n if r[1] == c)
                got_e = sorted((int(r.src), int(r.dst)) for r in edges_pdf.itertuples())
                exp_e = sorted((r[0], r[1]) for r in ref_e if r[2] == c)
                ok &= got == exp and got_e == exp_e and "storage_part" not in nodes_pdf.columns
        check(ok, "Phase 3b loader with table_layout() returns each community's rows in every layout")

        # overwrite the range table's path with the hive layout: replaced files stay on disk until VACUUM
        run_phase2(spark, spark.sparkContext, ["toy"], ["range"], results={}, p2_layout="hive", **common)
        lay = table_layout(spark, pr)
        nodes_pdf, _ = _load_communities_data_batch(pr["p2_nodes"], pr["p2_edges"], [0], layout=lay)
        exp = sorted((r[0], r[1]) for r in ref_n if r[1] == 0)
        check(sorted((int(r.id), int(r.community_id)) for r in nodes_pdf.itertuples()) == exp,
              "after overwriting with another layout, the snapshot file list returns the current rows only")
        import pyarrow.dataset as ds
        all_files = ds.dataset(local(pr["p2_nodes"]), format="parquet", ignore_prefixes=["_delta_log", "."]).files
        check(len(all_files) > len(lay["nodes"]["files"]),
              f"(a directory scan would see {len(all_files)} files, the snapshot has {len(lay['nodes']['files'])})")
    else:
        print("  [SKIP] pyarrow not installed: loader checks not run")

    spark.stop()
    shutil.rmtree(ROOT, ignore_errors=True)
    print("\n  ALL PASSED" if not failures else f"\n  {len(failures)} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

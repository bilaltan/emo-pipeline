#!/usr/bin/env python3
"""
Local Spark test for distributed LPA and minor-community absorption.

Runs the real run_phase1 LPA path on a planted-partition graph whose community
structure is known, then checks the three properties that decide whether LPA is
usable at Papers100M scale: it converges rather than burning a fixed budget, it
recovers roughly the planted structure instead of fragmenting, and the long tail
of small labels is folded into major communities rather than handed to Phase 2.

    JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
    PYSPARK_PYTHON=$PWD/.venv-spark/bin/python \
    PYSPARK_DRIVER_PYTHON=$PWD/.venv-spark/bin/python \
    .venv-spark/bin/python scripts/test_phase1_lpa.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

from phases.phase1_community import run_phase1

ROOT = tempfile.mkdtemp(prefix="lakegrl_lpa_")
N_COMM, PER_COMM = 6, 150
P_IN, P_OUT = 0.06, 0.0007          # dense within, sparse across


def paths(dataset, alg=None):
    base = f"file://{ROOT}/{dataset}"
    p = {"nodes": f"{base}/nodes/", "edges": f"{base}/edges/",
         "masks": f"{base}/masks/", "checkpoints": f"{ROOT}/ckpt/"}
    if alg:
        p["communities"] = f"{base}/communities/{alg}/"
        p["tag"] = f"lpa_{dataset}_{alg}"
    return p


def main():
    builder = (SparkSession.builder.appName("lpa-test").master("local[2]")
               .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
               .config("spark.sql.catalog.spark_catalog",
                       "org.apache.spark.sql.delta.catalog.DeltaCatalog")
               .config("spark.driver.memory", "3g")
               .config("spark.sql.shuffle.partitions", "8"))
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    rng = np.random.default_rng(11)
    n = N_COMM * PER_COMM
    comm = np.repeat(np.arange(N_COMM), PER_COMM)

    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            p_edge = P_IN if comm[u] == comm[v] else P_OUT
            if rng.random() < p_edge:
                edges.add((u, v))
    edges = sorted(edges)
    sym = edges + [(d, s) for s, d in edges]          # Phase 0 symmetrizes
    print(f"\n  planted graph: {n} nodes, {N_COMM} communities, {len(sym):,} directed edges")

    p = paths("planted")
    spark.createDataFrame([(int(i), int(comm[i])) for i in range(n)], "id long, label long") \
         .write.format("delta").mode("overwrite").save(p["nodes"])
    spark.createDataFrame([(int(s), int(d)) for s, d in sym], "src long, dst long") \
         .write.format("delta").mode("overwrite").save(p["edges"])

    timing, results = {}, {}
    run_phase1(spark, spark.sparkContext, ["planted"], ["lpa"],
               lpa_max_iter=int(os.environ.get('LPA_ITERS', 20)),
               resolution=1.0, random_seed=42, min_size=40,
               dataset_cfg={"planted": {"in_feats": 1, "num_classes": N_COMM}},
               get_paths_fn=paths, timing=timing, results=results,
               force_rerun=True, lpa_tol=0.001, merge_minor_communities=True)

    out = spark.read.format("delta").load(paths("planted", "lpa")["communities"])
    sizes = out.groupBy("community_id").count().orderBy("count", ascending=False).collect()
    found = len(sizes)
    covered = sum(r["count"] for r in sizes)
    minor = sum(1 for r in sizes if r["count"] < 40)

    # Agreement with the planted partition, as NMI.
    from sklearn.metrics import normalized_mutual_info_score
    pred = {r["id"]: r["community_id"] for r in out.collect()}
    nmi = normalized_mutual_info_score([comm[i] for i in sorted(pred)],
                                       [pred[i] for i in sorted(pred)])

    print(f"\n{'-'*66}\n  LPA CHECKS\n{'-'*66}")
    print(f"  communities found        {found} (planted {N_COMM})")
    print(f"  largest sizes            {[r['count'] for r in sizes[:8]]}")
    print(f"  NMI vs planted           {nmi:.4f}")

    checks = [
        ("converges rather than fragmenting", found <= N_COMM * 3, f"{found} communities"),
        ("recovers the planted structure", nmi >= 0.70, f"NMI {nmi:.4f}"),
        ("no minor-community tail left", minor == 0, f"{minor} below min_size"),
        ("every vertex assigned", covered == n, f"{covered}/{n}"),
    ]
    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<42s} {detail}")
        failed += (not ok)
    print(f"\n  {len(checks)-failed}/{len(checks)} checks passed")

    spark.stop()
    shutil.rmtree(ROOT, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

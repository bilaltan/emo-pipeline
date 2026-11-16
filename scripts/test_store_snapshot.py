#!/usr/bin/env python3
"""
utils.store_io.snapshot reads a Delta table's current files from its _delta_log without Spark (the export for the
baselines and DistDGL on EMR uses it). It must list exactly the files Spark's Delta reader reads: on every
checkpointed table of the local store (overwritten many times) and a few with many commits.

    bash scripts/wsl_run.sh python scripts/test_store_snapshot.py
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.store_io import snapshot  # noqa: E402


def main():
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession
    store = os.environ.get("LAKEGRL_STORE", os.path.expanduser("~/lakegrl/store"))
    logs = glob.glob(f"{store}/**/_delta_log", recursive=True)
    with_ck = [os.path.dirname(l) for l in logs if os.path.exists(f"{l}/_last_checkpoint")]
    many = [os.path.dirname(l) for _, l in sorted((len(glob.glob(f"{l}/*.json")), l) for l in logs)[-4:]]
    tables = sorted(set(with_ck + many))
    assert with_ck, "no checkpointed table in the store to test the checkpoint path on"
    spark = configure_spark_with_delta_pip(
        SparkSession.builder.master("local[2]").config("spark.driver.memory", "2g")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    bad = []
    for t in tables:
        _, mine, _ = snapshot(t)
        theirs = [f.replace("file://", "").replace("file:", "") for f in spark.read.format("delta").load(t).inputFiles()]
        if sorted(map(os.path.normpath, mine)) != sorted(map(os.path.normpath, theirs)):
            bad.append(t)
    assert not bad, f"snapshot differs from Spark on {bad}"
    print(f"ok: {len(tables)} tables ({len(with_ck)} checkpointed) list the same files as Spark")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Storage-layer benchmark: what do Delta OPTIMIZE and ZORDER do to the edge table?

Three conditions, not two. OPTIMIZE both compacts files and (with ZORDER)
clusters them, so a two-way baseline/z-order comparison cannot say which of the
two produced any observed change:

    A baseline   the table exactly as phase 0 wrote it
    B optimize   OPTIMIZE delta.`path`                       (compaction only)
    C zorder     OPTIMIZE delta.`path` ZORDER BY (src, dst)  (compaction + clustering)

B is the control for C.

Each condition gets its own physical copy of the baseline table, so the three
start from identical bytes and can be measured in any order.

Metrics per condition
---------------------
    files_active     parquet files the Delta log currently points at
    files_on_disk    parquet files actually present (OPTIMIZE does not delete
                     the originals; earlier work on this project found tables
                     that were >95% stale)
    bytes_active     sum of active file sizes, from the log
    bytes_on_disk    directory size on disk, before VACUUM
    scan_seconds     full-column scan, median of N repetitions
    scan_bytes_read  input bytes Spark actually read for that scan
    filter_seconds   selective read with a predicate on src
    filter_bytes_read  input bytes for the selective read

bytes_read is the metric that survives a warm page cache. Wall-clock timings on
a laptop are partly measuring RAM, but bytes_read is computed from what Spark
asked the filesystem for, so it shows data skipping directly whether or not the
file was already cached.

The pipeline code under test is imported from a worktree pinned at the
submission commit (--repo), so none of the post-submission changes participate.

Usage
-----
    python experiments/zorder_bench.py --dataset ogbn-arxiv \
        --repo /Users/bilaltan/Desktop/emo-submitted \
        --out results/zorder/ogbn-arxiv.json
"""
import argparse
import json
import os
import shutil
import statistics
import sys
import time
import urllib.request


def build_spark(repo, local_data_dir, master):
    """Same session configuration as the submitted runners/run_local.py."""
    if sys.platform == "darwin":
        os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

    extra_java_opts = (
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.util=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
        "--add-opens=java.base/java.security=ALL-UNNAMED "
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
        "--add-opens=java.base/java.math=ALL-UNNAMED"
    )
    os.environ["JAVA_TOOL_OPTIONS"] = extra_java_opts

    from pyspark.sql import SparkSession
    spark = (SparkSession.builder
             .appName("zorder-bench")
             .config("spark.master", master)
             .config("spark.driver.host", "127.0.0.1")
             .config("spark.driver.memory", "16g")
             .config("spark.driver.maxResultSize", "2g")
             .config("spark.sql.shuffle.partitions", "2")
             .config("spark.default.parallelism", "2")
             .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
             .config("spark.driver.extraJavaOptions", extra_java_opts)
             .config("spark.executor.extraJavaOptions", extra_java_opts)
             .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
             .config("spark.sql.catalog.spark_catalog",
                     "org.apache.spark.sql.delta.catalog.DeltaCatalog")
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .config("spark.databricks.delta.retentionDurationCheck.enabled", "false")
             .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.0")
             .config("spark.local.dir", os.path.join(local_data_dir, "spark_tmp"))
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    return spark


class StageMetrics:
    """Input bytes and task counts for the stages a single action ran.

    Read from the live Spark UI REST API, which is the only place PySpark
    exposes per-stage inputBytes without writing a Scala listener.
    """

    def __init__(self, spark):
        self.base = None
        url = spark.sparkContext.uiWebUrl
        if url:
            self.base = "%s/api/v1/applications/%s" % (
                url, spark.sparkContext.applicationId)

    def _stages(self):
        if not self.base:
            return []
        try:
            with urllib.request.urlopen(self.base + "/stages", timeout=30) as r:
                return json.load(r)
        except Exception:
            return []

    def snapshot(self):
        return {s["stageId"] for s in self._stages()}

    def since(self, before):
        """Sum inputBytes/numTasks over stages that did not exist in `before`."""
        total_bytes, total_tasks, n = 0, 0, 0
        for s in self._stages():
            if s["stageId"] in before or s.get("status") != "COMPLETE":
                continue
            total_bytes += s.get("inputBytes", 0) or 0
            total_tasks += s.get("numTasks", 0) or 0
            n += 1
        return {"bytes_read": total_bytes, "tasks": total_tasks, "stages": n}


def dir_stats(path):
    """Parquet files and total bytes actually present on disk."""
    files, size = 0, 0
    for root, _dirs, names in os.walk(path):
        if "_delta_log" in root:
            continue
        for n in names:
            if n.endswith(".parquet"):
                files += 1
                size += os.path.getsize(os.path.join(root, n))
    return files, size


def log_stats(spark, path):
    """Active file count and byte size, as the Delta log sees them."""
    row = spark.sql("DESCRIBE DETAIL delta.`%s`" % path).collect()[0]
    return int(row["numFiles"]), int(row["sizeInBytes"])


def timed(fn, reps):
    """Median of `reps` timings, first run discarded as JVM warm-up."""
    times = []
    for i in range(reps + 1):
        t = time.time()
        fn()
        dt = time.time() - t
        if i:
            times.append(dt)
    return round(statistics.median(times), 3), [round(t, 3) for t in times]


def measure(spark, path, reps, filter_lo, filter_hi, key, scan_exprs):
    from pyspark.sql import functions as F

    metrics = StageMetrics(spark)

    # Full scan. The aggregate has to touch every column that a real load would
    # read: parquet is columnar, so summing the key alone would prune the
    # feature column and report a load time no pipeline stage could achieve.
    # count() alone is worse still - Delta answers it from metadata.
    def full_scan():
        spark.read.format("delta").load(path).selectExpr(*scan_exprs).collect()

    # Selective read: the query shape Z-ordering exists to accelerate.
    def filtered():
        (spark.read.format("delta").load(path)
         .filter((F.col(key) >= filter_lo) & (F.col(key) < filter_hi))
         .selectExpr(*scan_exprs).collect())

    # Bytes and tasks are captured around ONE execution each. Measuring them
    # around the timing loop would report the sum over every repetition.
    before = metrics.snapshot()
    full_scan()
    scan_m = metrics.since(before)
    before = metrics.snapshot()
    filtered()
    filt_m = metrics.since(before)

    scan_s, scan_all = timed(full_scan, reps)
    filt_s, filt_all = timed(filtered, reps)

    files_active, bytes_active = log_stats(spark, path)
    files_disk, bytes_disk = dir_stats(path)

    return {
        "files_active": files_active,
        "files_on_disk": files_disk,
        "bytes_active": bytes_active,
        "bytes_on_disk": bytes_disk,
        "mb_active": round(bytes_active / 1e6, 2),
        "mb_on_disk": round(bytes_disk / 1e6, 2),
        "scan_seconds": scan_s,
        "scan_seconds_all": scan_all,
        "scan_bytes_read": scan_m["bytes_read"],
        "scan_tasks": scan_m["tasks"],
        "filter_seconds": filt_s,
        "filter_seconds_all": filt_all,
        "filter_bytes_read": filt_m["bytes_read"],
        "filter_tasks": filt_m["tasks"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ogbn-arxiv")
    ap.add_argument("--repo", default="/Users/bilaltan/Desktop/emo-submitted",
                    help="worktree holding the pipeline code under test")
    ap.add_argument("--local-data-dir", default=None,
                    help="where delta-data lives (default: <repo>/local_data)")
    ap.add_argument("--master", default="local[*]")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--table", default="edges",
                    choices=["edges", "original_edges", "nodes"])
    ap.add_argument("--filter-width", type=int, default=1000,
                    help="width of the src range used for the selective read")
    ap.add_argument("--max-file-mb", type=float, default=None,
                    help="cap OPTIMIZE's output file size, applied to B and C "
                         "alike. Delta skips data at FILE granularity, so a "
                         "table small enough to compact into one file cannot "
                         "show any clustering effect at all. Capping the size "
                         "keeps file count equal between B and C and isolates "
                         "what Z-ordering itself contributes.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-ingest", action="store_true")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo)
    local_data_dir = args.local_data_dir or os.path.join(repo, "local_data")
    os.makedirs(local_data_dir, exist_ok=True)
    sys.path.insert(0, repo)

    spark = build_spark(repo, local_data_dir, args.master)
    from pipeline.utils.paths import get_paths
    import experiment_config as config

    paths = get_paths(args.dataset, local_data_dir=local_data_dir)
    src_table = paths[args.table]

    if not args.skip_ingest and not os.path.exists(src_table):
        print("  ingesting %s with the submitted phase 0 ..." % args.dataset)
        from pipeline.phases import run_phase0
        t0 = time.time()
        run_phase0(
            spark, spark.sparkContext,
            datasets=[args.dataset],
            run_phase0_flag=True,
            use_ogb_splits=True,
            random_seed=config.RANDOM_SEED,
            dataset_cfg=config.DATASET_CFG,
            get_paths_fn=lambda d, a=None: get_paths(
                d, a, local_data_dir=local_data_dir),
            timing={},
            force_reingest=False,
        )
        print("  ingest took %.1fs" % (time.time() - t0))

    if not os.path.exists(src_table):
        print("table not found: %s" % src_table)
        return 1

    # One physical copy per condition, all from the same baseline bytes.
    bench_root = os.path.join(local_data_dir, "zorder_bench", args.dataset, args.table)
    conditions = {
        "A_baseline": None,
        "B_optimize": "OPTIMIZE delta.`%s`",
        "C_zorder": "OPTIMIZE delta.`%s` ZORDER BY (src, dst)",
    }
    if args.table == "nodes":
        conditions["C_zorder"] = "OPTIMIZE delta.`%s` ZORDER BY (id)"

    # Key column for the selective read, and an aggregate that forces every
    # column of a realistic load to be read off disk.
    if args.table == "nodes":
        key, scan_exprs = "id", ["sum(id)", "sum(label)", "sum(size(features))"]
    else:
        key, scan_exprs = "src", ["sum(src)", "sum(dst)"]

    results, optimize_time = {}, {}
    for name, sql in conditions.items():
        dst = os.path.join(bench_root, name)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(src_table, dst)
        print("\n=== %s ===" % name)
        if sql and args.max_file_mb:
            spark.conf.set("spark.databricks.delta.optimize.maxFileSize",
                           str(int(args.max_file_mb * 1024 * 1024)))
        if sql:
            t0 = time.time()
            spark.sql(sql % dst).collect()
            optimize_time[name] = round(time.time() - t0, 2)
            print("  %s took %.1fs" % (sql.split("delta")[0].strip(),
                                       optimize_time[name]))
        results[name] = measure(spark, dst, args.reps, 0, args.filter_width,
                                key, scan_exprs)
        results[name]["optimize_seconds"] = optimize_time.get(name, 0.0)
        for k in ("files_active", "files_on_disk", "mb_active", "mb_on_disk",
                  "scan_seconds", "scan_bytes_read",
                  "filter_seconds", "filter_bytes_read"):
            print("  %-18s %s" % (k, results[name][k]))

    edges = spark.read.format("delta").load(src_table).count()

    doc = {
        "dataset": args.dataset,
        "table": args.table,
        "rows": edges,
        "code_commit": os.popen("git -C %s rev-parse HEAD" % repo).read().strip(),
        "spark_version": spark.version,
        "delta_version": "3.2.0",
        "master": args.master,
        "reps": args.reps,
        "filter_predicate": "%s >= 0 AND %s < %d" % (key, key, args.filter_width),
        "max_file_mb": args.max_file_mb,
        "caveats": [
            "Timings are warm-cache: macOS page cache is not purged between "
            "runs, so scan_seconds partly measures RAM. bytes_read is the "
            "cache-independent metric and is what shows data skipping.",
            "Local NVMe has no network latency, so any I/O saving here is a "
            "lower bound on what the same change would give on S3.",
            "bytes_on_disk includes files the Delta log no longer references; "
            "OPTIMIZE does not delete them. bytes_active is the logical size.",
        ],
        "conditions": results,
    }
    print("\n" + json.dumps(doc, indent=2)[:400] + " ...")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(doc, f, indent=2)
        print("\nwrote %s" % args.out)
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

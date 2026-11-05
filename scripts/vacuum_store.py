#!/usr/bin/env python3
"""
Delete old Delta versions (Bilal 2026-11-05): VACUUM RETAIN 0 HOURS on every Delta table under the given roots, so
only each table's current version stays on disk. An overwrite keeps the old files until VACUUM, and every Table B
run's Phase 2 rewrites p2_nodes / p2_edges: on 11-05 ~/lakegrl held 566 GB of Delta files for 200 GB of current data.
Afterwards a read of an older version (versionAsOf) fails; the current version of every table is unchanged.

Per table: Delta's DRY RUN first; a table where it would delete anything but Spark data files (part-*) is skipped and
reported. VACUUM writes no commit (vacuum logging off), so the version Phase 2 stamped on the macro-graph tables
still matches. Afterwards the version and current files must be unchanged and on disk, and the hidden .crc
checksums of the deleted files (VACUUM leaves hidden files) are removed. Run it with nothing writing to the store.

    bash scripts/wsl_run.sh python scripts/vacuum_store.py ~/lakegrl --out results/vacuum/vacuum.jsonl
"""
import argparse
import json
import os
import re
import sys
import time
from urllib.parse import unquote, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delta import configure_spark_with_delta_pip  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402

from pipeline.utils.store_io import snapshot  # noqa: E402

PART = re.compile(r"^part-.*\.parquet$")


def tables(roots):
    for root in roots:
        for r, ds, _ in os.walk(root):
            if "_delta_log" in ds:
                ds[:] = []   # a table: no tables inside it
                yield r
            ds.sort()


def du(d):
    n = 0
    for r, _, fs in os.walk(d):
        for f in fs:
            try:
                n += os.lstat(os.path.join(r, f)).st_size
            except OSError:
                pass
    return n


def drop_orphan_crc(d):
    """Remove .NAME.crc where NAME is gone, outside _delta_log. Returns (files, bytes)."""
    n = b = 0
    for r, ds, fs in os.walk(d):
        if "_delta_log" in ds:
            ds.remove("_delta_log")
        have = set(fs)
        for f in fs:
            if f.startswith(".") and f.endswith(".crc") and f[1:-4] not in have:
                p = os.path.join(r, f)
                b += os.lstat(p).st_size
                os.remove(p)
                n += 1
    return n, b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    b = (SparkSession.builder.appName("vacuum-store").master("local[8]")
         .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
         .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
         .config("spark.databricks.delta.retentionDurationCheck.enabled", "false")
         .config("spark.databricks.delta.vacuum.logging.enabled", "false")
         .config("spark.databricks.delta.vacuum.parallelDelete.enabled", "true")
         # VACUUM lists files in this many tasks (default 10,000: 14 s per arxiv table, mostly task overhead)
         .config("spark.sql.sources.parallelPartitionDiscovery.parallelism", "32")
         .config("spark.driver.memory", "4g"))
    spark = configure_spark_with_delta_pip(b).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tot = {"before": 0, "after": 0, "tables": 0, "skipped": 0, "failed": 0}
    t_all = time.time()
    with open(args.out, "a") as fh:
        for t in tables([os.path.abspath(os.path.expanduser(r)) for r in args.roots]):
            t0 = time.time()
            rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "table": t, "dry_run": args.dry_run}
            try:
                _, live0, v0 = snapshot(t)
                rec.update(version=v0, before=du(t))
                if v0 < 0:   # _delta_log without a commit (a write that failed first): not a table, leave it
                    rec.update(skipped="no commit in _delta_log", after=rec["before"], ok=True)
                    raise StopIteration
                q = f"VACUUM delta.`file://{t}` RETAIN 0 HOURS"
                cand = [unquote(urlparse(r[0]).path) for r in spark.sql(q + " DRY RUN").collect()]
                assert all(c.startswith(t + "/") for c in cand), f"path outside the table: {cand[:1]}"
                assert not set(live0) & set(cand), "DRY RUN lists a current file"
                odd = [c for c in cand if not (PART.match(os.path.basename(c)) or os.path.isdir(c))]
                rec.update(delete=len(cand), odd=len(odd), odd_e_g=odd[:3])
                if odd:
                    rec["skipped"] = "would delete files that are not Spark data files"
                elif not args.dry_run:
                    if cand:
                        spark.sql(q)
                    rec["crc"], rec["crc_bytes"] = drop_orphan_crc(t)
                    _, live1, v1 = snapshot(t)
                    assert (v1, live1) == (v0, live0), f"current version changed: {v0} -> {v1}"
                    gone = [p for p in live1 if not os.path.exists(p)]
                    assert not gone, f"{len(gone)} current files missing, e.g. {gone[0]}"
                rec.update(after=du(t), ok=True)
            except StopIteration:
                pass
            except Exception as e:
                rec.update(ok=False, error=f"{type(e).__name__}: {e}"[:500])
            rec["s"] = round(time.time() - t0, 2)
            tot["tables"] += 1
            tot["skipped"] += "skipped" in rec
            tot["failed"] += not rec["ok"]
            tot["before"] += rec.get("before", 0)
            tot["after"] += rec.get("after", rec.get("before", 0))
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{'ok  ' if rec['ok'] else 'FAIL'} {t}  {rec.get('before', 0) / 1e9:.2f} -> "
                  f"{rec.get('after', 0) / 1e9:.2f} GB  delete {rec.get('delete')}  {rec.get('skipped', '')}"
                  f"{rec.get('error', '')}", flush=True)
    tot.update(freed_gb=round((tot["before"] - tot["after"]) / 1e9, 1), s=round(time.time() - t_all, 1),
               dry_run=args.dry_run)
    print(json.dumps(tot), flush=True)
    sys.exit(1 if tot["failed"] else 0)


if __name__ == "__main__":
    main()

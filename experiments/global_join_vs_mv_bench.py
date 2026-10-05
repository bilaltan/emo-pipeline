#!/usr/bin/env python3
"""
How node rows reach the training units: materialized view vs global join.

  mv_direct       p2_nodes is the materialized view (LakeGRL default). Each bin
                  reads its own units from p2_nodes/p2_edges inside the worker
                  with phase3b's pyarrow loader (community_id predicate, data
                  skipping). No data moves between workers.
  global_join     p2_nodes is normalized (ids + community only). One Spark join
                  of the whole partition table with the whole raw nodes/masks
                  tables (the partition table is broadcast), attach_units, shuffle
                  by bin, cogroup with the manifest. Edges are read as in mv_direct.
  global_join_smj Same, with broadcast joins disabled: what the join costs once
                  the partition table is too large to broadcast (sort-merge join,
                  so the raw nodes table is shuffled as well).

All modes run Phase 3b's work manifest (units, bounded blocks, bins) and stop
once every unit's nodes and edges are in its worker as numpy arrays, i.e. right
before training. Phase 2 is run in both partition_table_modes first, to measure
its write time and the size of each partition table.

Per run: wall time, in-worker time, bytes read from storage (Spark scan input
for the join, parquet column chunks read by pyarrow in the workers), shuffle
bytes (Spark REST API), and per-unit checksums that must agree across modes.

Run from the repository root, e.g.:

  PYSPARK_PYTHON=$PWD/.venv-spark/bin/python .venv-spark/bin/python \
    experiments/global_join_vs_mv_bench.py \
    --data-root local_data/joinbench/delta-data --dataset ogbn-arxiv --alg louvain \
    --bench-dir local_data/gjbench --out results/partition_join/global_join_vs_mv.jsonl
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# Python workers must import phases/utils (and 'pipeline', a symlink to ROOT).
os.environ["PYTHONPATH"] = ROOT + (os.pathsep + os.environ["PYTHONPATH"]
                                   if os.environ.get("PYTHONPATH") else "")

MODES = ("mv_direct", "global_join", "global_join_smj")


def total_ram_gb():
    import subprocess
    try:
        if sys.platform == "darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2 ** 30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except Exception:
        return None


# --------------------------------------------------------------------------- setup
def build_spark(args):
    # Spark 3.5 / Hadoop 3.3 need Java 17: newer JDKs fail at startup with
    # "getSubject is not supported". Same JDK as runners/run_local.py.
    if sys.platform == "darwin" and os.path.isdir("/opt/homebrew/opt/openjdk@17"):
        os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    from pyspark.sql import SparkSession
    b = (SparkSession.builder.appName("global-join-vs-mv-bench")
         .master(f"local[{args.cores}]")
         .config("spark.driver.host", "127.0.0.1")
         .config("spark.driver.memory", args.mem)
         .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
         .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
         # Same settings as scripts/run_local_pipeline.py.
         .config("spark.sql.shuffle.partitions", str(args.cores * 2))
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .config("spark.sql.parquet.columnarReaderBatchSize", "512")
         .config("spark.sql.execution.arrow.maxRecordsPerBatch", "500")
         .config("spark.ui.enabled", "true")          # stage metrics come from the REST API
         .config("spark.ui.showConsoleProgress", "false"))
    if args.spark_tmp:
        b = b.config("spark.local.dir", args.spark_tmp)
    jars = os.environ.get("LAKEGRL_SPARK_JARS")
    if jars:
        spark = b.config("spark.jars", jars).getOrCreate()
    else:
        from delta import configure_spark_with_delta_pip
        spark = configure_spark_with_delta_pip(b).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def make_paths(args):
    raw = os.path.abspath(os.path.join(args.data_root, args.dataset))
    out = os.path.abspath(os.path.join(args.bench_dir, args.dataset))

    def get_paths(dataset, alg=None):
        p = {"nodes": f"{raw}/nodes", "edges": f"{raw}/edges", "masks": f"{raw}/masks"}
        if alg:   # alg is the partition-table form ('materialized'/'normalized', + '_r<k>')
            p.update({"communities": f"{raw}/communities/{args.alg}",
                      "p2_nodes": f"{out}/{alg}/p2_nodes", "p2_edges": f"{out}/{alg}/p2_edges",
                      "tag": f"bench_{dataset}_{args.alg}_{alg}"})
        return p
    return get_paths


def table_size(spark, path):
    from delta.tables import DeltaTable
    d = DeltaTable.forPath(spark, path).detail().select("numFiles", "sizeInBytes").first()
    return {"files": int(d[0]), "bytes": int(d[1])}


def run_phase2_both(spark, args, get_paths, emit):
    """Phase 2 in both forms, alternating order. Every run writes a fresh
    directory: the direct loader lists parquet files in the table directory, so
    an overwritten table would also expose the previous run's files."""
    import shutil
    from phases.phase2_subgraph import run_phase2
    order = ["materialized", "normalized"]
    for r in range(args.phase2_repeats):
        for form in (order if r % 2 == 0 else order[::-1]):
            tag = form if r == 0 else f"{form}_r{r}"   # loading uses repeat 0's tables
            p = get_paths(args.dataset, tag)
            for k in ("p2_nodes", "p2_edges"):
                shutil.rmtree(p[k], ignore_errors=True)
            timing, res = {}, {}
            t0 = time.perf_counter()
            run_phase2(spark, spark.sparkContext, [args.dataset], [tag], use_global_mapping=True,
                       min_size=args.min_size, get_paths_fn=get_paths, timing=timing, results=res,
                       expand_boundary_nodes=True, tiny_comm_handling="drop", force_rerun=True,
                       partition_table_mode=form)
            wall = time.perf_counter() - t0
            rec = {"kind": "phase2", "dataset": args.dataset, "alg": args.alg, "form": form,
                   "repeat": r, "phase2_s": timing.get(("phase2", args.dataset, tag)), "wall_s": wall,
                   "p2_nodes": table_size(spark, p["p2_nodes"]),
                   "p2_edges": table_size(spark, p["p2_edges"])}
            emit(rec)
            print(f"  Phase 2 [{form:12s}] {rec['phase2_s']:.1f}s  p2_nodes "
                  f"{rec['p2_nodes']['bytes'] / 1e6:.1f} MB in {rec['p2_nodes']['files']} files")


def build_work(p2_table, cap, default_parallelism):
    """Phase 3b's manifest: units (communities, split into power-of-two blocks
    above ``cap`` rows) and their bins, exactly as run_phase3b builds it."""
    import numpy as np
    counts = (p2_table.select("community_id").groupBy("community_id").count().toPandas()
              .sort_values(by="count", ascending=False).reset_index(drop=True))

    def nblocks(cnt):
        need = int(np.ceil(float(cnt) / float(cap)))
        nb = 1
        while nb < need:
            nb *= 2
        return nb
    counts["_n_blocks"] = counts["count"].map(nblocks) if cap > 0 else 1
    work = counts.loc[counts.index.repeat(counts["_n_blocks"])].copy()
    work["_block"] = work.groupby("community_id").cumcount()
    work = work.reset_index(drop=True)
    n = len(work)
    num_bins = n if n <= 2000 else min(max(default_parallelism * 4, 1000), n)
    work["bin_id"] = [i % num_bins for i in range(n)]
    for c in ("community_id", "_n_blocks", "_block", "bin_id"):
        work[c] = work[c].astype("int64")
    return work, num_bins


# ------------------------------------------------------------------ worker side
RESULT_SCHEMA = ("bin_id long, community_id long, block long, n_nodes long, n_members long, "
                 "n_edges long, id_sum long, feat_sum double, label_sum long, n_train long, "
                 "bin_s double, load_s double")


def _unit_stats(pdf, epdf):
    """Turn a unit into the arrays training starts from; return checksums."""
    import numpy as np
    if pdf is None or len(pdf) == 0:
        return dict(n_nodes=0, n_members=0, n_edges=int(len(epdf)) if epdf is not None else 0,
                    id_sum=0, feat_sum=0.0, label_sum=0, n_train=0)
    ids = pdf["id"].to_numpy(np.int64)
    feats = np.stack(pdf["features"].to_numpy()).astype(np.float32)
    labels = pdf["label"].fillna(-1).to_numpy(np.int64)
    member = pdf["is_member"].astype(bool).to_numpy()
    train = ((pdf["split"] == "train").to_numpy() & member)
    n_edges = 0
    if epdf is not None and len(epdf) > 0:
        edge_index = np.stack([epdf["src"].to_numpy(np.int64), epdf["dst"].to_numpy(np.int64)])
        n_edges = int(edge_index.shape[1])
    return dict(n_nodes=int(len(ids)), n_members=int(member.sum()), n_edges=n_edges,
                id_sum=int(ids.sum()), feat_sum=float(feats.sum(dtype=np.float64)),
                label_sum=int(labels.sum()), n_train=int(train.sum()))


def _run_bin(bin_id, manifest_pdf, load_batch, load_item):
    """Phase 3b's per-bin loop (phase3b_caan._train_bin) minus the training."""
    import pandas as pd
    t0 = time.perf_counter()
    load_s = 0.0
    has_blocks = int(manifest_pdf["_n_blocks"].max()) > 1
    if not has_blocks:
        tl = time.perf_counter()
        nodes_all, edges_all = load_batch(manifest_pdf["community_id"].unique().tolist())
        load_s += time.perf_counter() - tl
    rows = []
    # (itertuples would rename the underscore columns)
    for c, b, n in manifest_pdf[["community_id", "_block", "_n_blocks"]].to_numpy().tolist():
        c, b, n = int(c), int(b), int(n)
        if has_blocks:
            tl = time.perf_counter()
            pdf, epdf = load_item(c, b, n)
            load_s += time.perf_counter() - tl
            if len(pdf) > 0 and int(pdf["is_member"].astype(bool).sum()) == 0:
                pdf = pdf.iloc[0:0]
        else:
            pdf = nodes_all[nodes_all["community_id"] == c] if len(nodes_all) > 0 else pd.DataFrame()
            epdf = edges_all[edges_all["community_id"] == c] if len(edges_all) > 0 else pd.DataFrame()
        rows.append(dict(bin_id=int(bin_id), community_id=c, block=b, **_unit_stats(pdf, epdf)))
    bin_s = time.perf_counter() - t0
    out = pd.DataFrame(rows)
    out["bin_s"], out["load_s"] = bin_s, load_s
    return out


def make_direct_udf(nodes_url, edges_url):
    from phases.phase3b_caan import _load_communities_data_batch

    def udf(key, manifest_pdf):
        return _run_bin(
            key[0], manifest_pdf,
            lambda ids: _load_communities_data_batch(nodes_url, edges_url, ids),
            lambda c, b, n: _load_communities_data_batch(nodes_url, edges_url, [c], block=b, n_blocks=n))
    return udf


def make_joined_udf(edges_url):
    from phases.phase3b_caan import _load_unit_edges

    def udf(key, nodes_pdf, manifest_pdf):
        import pandas as pd
        if len(nodes_pdf) > 0:
            nodes_pdf = nodes_pdf.drop_duplicates(subset=["id", "community_id"]).reset_index(drop=True)
        cols = [c for c in nodes_pdf.columns if c not in ("_block", "bin_id")]

        def batch(ids):
            return (nodes_pdf[cols] if len(nodes_pdf) > 0 else pd.DataFrame(),
                    _load_unit_edges(edges_url, ids))

        def item(c, b, n):
            if len(nodes_pdf) == 0:
                return pd.DataFrame(), _load_unit_edges(edges_url, [c], block=b, n_blocks=n)
            sel = nodes_pdf[(nodes_pdf["community_id"] == c) & (nodes_pdf["_block"] == b)]
            return (sel[cols].reset_index(drop=True) if len(sel) > 0 else pd.DataFrame(),
                    _load_unit_edges(edges_url, [c], block=b, n_blocks=n))
        return _run_bin(key[0], manifest_pdf, batch, item)
    return udf


# ------------------------------------------------------------------ measurement
def _get_json(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=30) as r:
        return json.loads(r.read().decode())


def stage_metrics(sc, group):
    """Sum task metrics over the executed stages of the jobs in ``group``."""
    base = f"{sc.uiWebUrl}/api/v1/applications/{sc.applicationId}"
    for _ in range(50):                       # wait for the listener to catch up
        jobs = [j for j in _get_json(f"{base}/jobs") if j.get("jobGroup") == group]
        if jobs and all(j["status"] in ("SUCCEEDED", "FAILED") for j in jobs):
            break
        time.sleep(0.2)
    tot = dict(input_bytes=0, input_records=0, shuffle_write_bytes=0, shuffle_read_bytes=0,
               executor_run_ms=0, stages=0, tasks=0)
    for sid in sorted({s for j in jobs for s in j["stageIds"]}):
        try:
            attempts = _get_json(f"{base}/stages/{sid}")
        except Exception:
            continue                            # never submitted (skipped)
        for att in attempts:
            if att.get("status") != "COMPLETE":
                continue                        # skipped stages did not run
            tot["input_bytes"] += att.get("inputBytes", 0)
            tot["input_records"] += att.get("inputRecords", 0)
            tot["shuffle_write_bytes"] += att.get("shuffleWriteBytes", 0)
            tot["shuffle_read_bytes"] += att.get("shuffleReadBytes", 0)
            tot["executor_run_ms"] += att.get("executorRunTime", 0)
            tot["stages"] += 1
            tot["tasks"] += att.get("numCompleteTasks", att.get("numTasks", 0))
    return tot


_FRAG_CACHE = {}
_BYTES_CACHE = {}


def pyarrow_read_bytes(url, comm_ids):
    """Parquet bytes the phase3b loader reads for a community_id predicate: all
    column chunks of every row group whose statistics may match (what pyarrow's
    scanner fetches). A block predicate (id & (n-1) == b) cannot be decided from
    min/max statistics, so a block reads exactly what its whole community reads."""
    import pyarrow.dataset as ds
    key = (url, tuple(sorted(int(c) for c in comm_ids)))
    if key in _BYTES_CACHE:
        return _BYTES_CACHE[key]
    if url not in _FRAG_CACHE:
        d = ds.dataset(url.replace("file://", ""), format="parquet", ignore_prefixes=["_delta_log", "."])
        frags = list(d.get_fragments())
        for f in frags:
            f.ensure_complete_metadata()
        _FRAG_CACHE[url] = frags
    filt = ds.field("community_id").isin(list(key[1]))
    total, files = 0, 0
    for f in _FRAG_CACHE[url]:
        md = f.metadata
        rgs = f.subset(filter=filt).row_groups
        if rgs:
            files += 1
        for rg in rgs:
            g = md.row_group(rg.id)
            total += sum(g.column(k).total_compressed_size for k in range(g.num_columns))
    _BYTES_CACHE[key] = (total, files)
    return total, files


def planned_pyarrow_io(work, nodes_url, edges_url):
    """Bytes/files the workers' pyarrow reads touch, summed over all bins."""
    has_blocks = int(work["_n_blocks"].max()) > 1
    out = dict(node_bytes=0, node_files=0, edge_bytes=0, edge_files=0)
    if has_blocks:   # one read per (community, block) work item
        items = [[int(c)] for c in work["community_id"].tolist()]
    else:            # one read per bin
        items = [g["community_id"].unique().tolist() for _, g in work.groupby("bin_id")]
    for ids in items:
        if nodes_url:
            nb, nf = pyarrow_read_bytes(nodes_url, ids)
            out["node_bytes"] += nb
            out["node_files"] += nf
        eb, ef = pyarrow_read_bytes(edges_url, ids)
        out["edge_bytes"] += eb
        out["edge_files"] += ef
    return out


def run_mode(spark, mode, work, num_bins, get_paths, args, tag):
    from pyspark.sql import functions as F  # noqa: F401
    from utils.partition_join import attach_units, load_unit_nodes
    sc = spark.sparkContext
    mv = get_paths(args.dataset, "materialized")
    nm = get_paths(args.dataset, "normalized")
    spark.catalog.clearCache()
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold",
                   "-1" if mode == "global_join_smj" else str(10 * 1024 * 1024))
    sc.setJobGroup(tag, tag)
    t0 = time.perf_counter()
    manifest = spark.createDataFrame(work[["community_id", "bin_id", "_block", "_n_blocks"]])
    if mode == "mv_direct":
        df = (manifest.repartition(num_bins, "bin_id").groupBy("bin_id")
              .applyInPandas(make_direct_udf(mv["p2_nodes"], mv["p2_edges"]), schema=RESULT_SCHEMA))
    else:
        units = attach_units(spark, load_unit_nodes(spark, nm), work)
        df = (units.repartition(num_bins, "bin_id").groupBy("bin_id")
              .cogroup(manifest.repartition(num_bins, "bin_id").groupBy("bin_id"))
              .applyInPandas(make_joined_udf(nm["p2_edges"]), schema=RESULT_SCHEMA))
    res = df.toPandas()
    wall = time.perf_counter() - t0
    sc.setJobGroup("", "")
    plan = df._jdf.queryExecution().executedPlan().toString()
    joins = sorted({j for j in ("BroadcastHashJoin", "SortMergeJoin", "ShuffledHashJoin") if j in plan})
    return res, wall, stage_metrics(sc, tag), joins


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="dir holding <dataset>/{nodes,edges,masks,communities}")
    ap.add_argument("--dataset", default="ogbn-arxiv")
    ap.add_argument("--alg", default="louvain")
    ap.add_argument("--bench-dir", required=True,
                    help="scratch dir for both Phase 2 outputs (its tables are deleted and rewritten)")
    ap.add_argument("--min-size", type=int, default=100)
    ap.add_argument("--caps", type=int, nargs="+", default=[10000, 0],
                    help="max_nodes_per_community values (0 = whole communities); 10000 is Phase 3b's default")
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1, help="unrecorded rounds before the repeats")
    ap.add_argument("--phase2-repeats", type=int, default=2)
    ap.add_argument("--skip-phase2", action="store_true", help="reuse the Phase 2 tables in --bench-dir")
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--mem", default="auto", help="driver memory; auto = 4g, 6g with >=16 GB RAM, 10g with >=32 GB")
    ap.add_argument("--spark-tmp", default=None)
    ap.add_argument("--out", default="results/partition_join/global_join_vs_mv.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    run_id = time.strftime("%Y%m%d-%H%M%S")

    def emit(rec):
        rec = {"run_id": run_id, "cores": args.cores, **rec}
        with open(args.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    ram = total_ram_gb()
    if args.mem == "auto":
        args.mem = "4g" if ram is None or ram < 15 else "6g" if ram < 30 else "10g"
    spark = build_spark(args)
    get_paths = make_paths(args)
    print(f"\n== {args.dataset}/{args.alg}  local[{args.cores}]  driver {args.mem}  "
          f"RAM {ram:.1f} GB  run {run_id}" if ram else f"\n== {args.dataset}/{args.alg}  run {run_id}")
    if not args.skip_phase2:
        run_phase2_both(spark, args, get_paths, emit)

    mv = get_paths(args.dataset, "materialized")
    nm = get_paths(args.dataset, "normalized")
    raw_nodes = table_size(spark, mv["nodes"])
    sizes = {"raw_nodes": raw_nodes, "p2_nodes_materialized": table_size(spark, mv["p2_nodes"]),
             "p2_nodes_normalized": table_size(spark, nm["p2_nodes"]),
             "p2_edges": table_size(spark, mv["p2_edges"])}
    emit({"kind": "sizes", "dataset": args.dataset, "alg": args.alg, **sizes})
    p2_mv = spark.read.format("delta").load(mv["p2_nodes"])

    for cap in args.caps:
        work, num_bins = build_work(p2_mv, cap, spark.sparkContext.defaultParallelism)
        n_split = int((work.groupby("community_id")["_n_blocks"].first() > 1).sum())
        print(f"\n-- cap={cap or 'none'}: {work['community_id'].nunique()} communities -> "
              f"{len(work)} units in {num_bins} bins ({n_split} communities split)")
        io_mv = planned_pyarrow_io(work, mv["p2_nodes"], mv["p2_edges"])
        io_gj = planned_pyarrow_io(work, None, nm["p2_edges"])
        print(f"   planned pyarrow reads: mv_direct {(io_mv['node_bytes'] + io_mv['edge_bytes']) / 1e9:.2f} GB, "
              f"global join (edges only) {io_gj['edge_bytes'] / 1e9:.2f} GB")
        reference = None
        for rep in range(-args.warmup, args.repeats):   # negative rep = warm-up, not recorded
            modes = args.modes if rep % 2 == 0 else args.modes[::-1]
            for mode in modes:
                tag = f"{mode}_cap{cap}_r{rep}"
                res, wall, sm, joins = run_mode(spark, mode, work, num_bins, get_paths, args, tag)
                key_cols = ["community_id", "block"]
                chk = (res.sort_values(key_cols)[key_cols + ["n_nodes", "n_members", "n_edges", "id_sum",
                                                          "label_sum", "n_train"]]
                       .reset_index(drop=True))
                fsum = round(float(res["feat_sum"].sum()), 2)
                if reference is None:
                    reference = (chk, fsum)
                same = chk.equals(reference[0]) and abs(fsum - reference[1]) <= 1e-6 * max(1.0, abs(fsum))
                per_bin = res.groupby("bin_id")[["bin_s", "load_s"]].first()
                io = io_mv if mode == "mv_direct" else io_gj
                rec = {"kind": "loading", "dataset": args.dataset, "alg": args.alg, "cap": cap,
                       "mode": mode, "repeat": rep, "units": len(work), "bins": num_bins,
                       "wall_s": wall, "sum_bin_s": float(per_bin["bin_s"].sum()),
                       "max_bin_s": float(per_bin["bin_s"].max()),
                       "sum_load_s": float(per_bin["load_s"].sum()),
                       "rows_delivered": int(res["n_nodes"].sum()), "edges_delivered": int(res["n_edges"].sum()),
                       "spark": sm, "pyarrow": io, "joins": joins, "checksums_match": bool(same)}
                if rep >= 0:
                    emit(rec)
                print(f"  {'warm-up' if rep < 0 else f'rep {rep}':7s} {mode:16s} wall {wall:6.2f}s  "
                      f"scan {sm['input_bytes'] / 1e6:7.1f} MB  pyarrow {(io['node_bytes'] + io['edge_bytes']) / 1e6:7.1f} MB  "
                      f"shuffle w {sm['shuffle_write_bytes'] / 1e6:7.1f} MB  joins {','.join(joins) or '-'}  "
                      f"rows {rec['rows_delivered']:,}  match={same}")

    spark.stop()
    summarize(args.out, run_id)


def summarize(path, run_id):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    recs = [r for r in recs if r.get("run_id") == run_id]
    print("\n== Summary (median over repeats)")
    for r in recs:
        if r["kind"] == "sizes":
            print(f"  raw nodes {r['raw_nodes']['bytes'] / 1e6:.1f} MB | p2_nodes materialized "
                  f"{r['p2_nodes_materialized']['bytes'] / 1e6:.1f} MB | normalized "
                  f"{r['p2_nodes_normalized']['bytes'] / 1e6:.2f} MB | p2_edges {r['p2_edges']['bytes'] / 1e6:.1f} MB")
    for form in ("materialized", "normalized"):
        ts = [r["phase2_s"] for r in recs if r["kind"] == "phase2" and r["form"] == form]
        if ts:
            print(f"  Phase 2 {form:12s}: {statistics.median(ts):.1f}s  (runs: {', '.join(f'{t:.1f}' for t in ts)})")
    loads = [r for r in recs if r["kind"] == "loading"]
    for cap in sorted({r["cap"] for r in loads}, reverse=True):
        print(f"\n  cap={cap or 'none'}  (units={next(r['units'] for r in loads if r['cap'] == cap)})")
        print(f"  {'mode':16s} {'wall s':>7s} {'worker s':>9s} {'scan MB':>8s} {'pyarrow MB':>11s} "
              f"{'shuffle MB':>11s} {'match':>6s}")
        for mode in MODES:
            rs = [r for r in loads if r["cap"] == cap and r["mode"] == mode]
            if not rs:
                continue
            med = lambda f: statistics.median(f(r) for r in rs)  # noqa: E731
            print(f"  {mode:16s} {med(lambda r: r['wall_s']):7.2f} {med(lambda r: r['sum_bin_s']):9.2f} "
                  f"{med(lambda r: r['spark']['input_bytes']) / 1e6:8.1f} "
                  f"{med(lambda r: r['pyarrow']['node_bytes'] + r['pyarrow']['edge_bytes']) / 1e6:11.1f} "
                  f"{med(lambda r: r['spark']['shuffle_write_bytes']) / 1e6:11.1f} "
                  f"{str(all(r['checksums_match'] for r in rs)):>6s}")


if __name__ == "__main__":
    main()

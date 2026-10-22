"""Physical layout of the Phase 2 tables (p2_nodes, p2_edges).

  range        (default) one Delta table without partition columns, written with
               repartitionByRange(community_id) and sorted by community_id, so
               each community sits in one or two files and Delta / parquet
               min-max statistics skip the rest.
  hive         .partitionBy("community_id"): one directory per community
               (community_id=<c>/), like a Hive table. Readers prune whole
               directories; the partition value lives in the path and in the
               Delta log, not in the parquet files.
  hive_merged  .partitionBy("storage_part"): every community large enough keeps
               its own directory (storage_part=c<id>); minor / small communities
               are packed into shared directories (storage_part=m<k>) of up to
               merge_target_rows rows, in community_id order, so a shared
               directory holds a contiguous run of small communities, still
               sorted by community_id inside its files. community_id stays a
               data column, so a training unit is still exactly one community.

A community is "small" if it has fewer than merge_minor_members member rows
(is_member, the paper's minor-community threshold K) or fewer than
merge_min_rows rows in total (members + 1-hop halo). Edges follow the
community's storage partition.

Readers: Spark / Delta handle all three transparently (partition pruning from
the log). The pyarrow per-unit loader in Phase 3b needs the partitioning and,
for hive_merged, the community -> storage_part map; table_layout() returns both,
plus the snapshot's file list so stale files left by an overwrite (until VACUUM)
are never read.
"""
LAYOUTS = ("range", "hive", "hive_merged")
STORAGE_PART = "storage_part"


def merge_plan(stats_pdf, merge_minor_members=100, merge_min_rows=0, merge_target_rows=250_000):
    """community_id -> storage_part.

    stats_pdf: pandas DataFrame with community_id, rows, members.
    Returns a DataFrame (community_id, storage_part, small) with one row per community.
    """
    import pandas as pd
    s = stats_pdf.sort_values("community_id").reset_index(drop=True)
    small = (s["members"] < int(merge_minor_members)) | (s["rows"] < int(merge_min_rows))
    parts, k, filled = [], 0, 0
    for cid, rows, is_small in zip(s["community_id"], s["rows"], small):
        if not is_small:
            parts.append(f"c{int(cid)}")
            continue
        if filled > 0 and filled + int(rows) > int(merge_target_rows):
            k, filled = k + 1, 0
        parts.append(f"m{k}")
        filled += int(rows)
    return pd.DataFrame({"community_id": s["community_id"].astype("int64"),
                         "storage_part": parts, "small": small.to_numpy()})


def community_stats(nodes_df):
    """Per community: rows (members + halo) and members, as pandas."""
    from pyspark.sql import functions as F
    member = F.col("is_member") if "is_member" in nodes_df.columns else F.lit(True)
    return (nodes_df.groupBy("community_id")
            .agg(F.count(F.lit(1)).alias("rows"),
                 F.sum(F.when(member, 1).otherwise(0)).alias("members"))
            .toPandas())


def with_storage_part(spark, df, plan_pdf):
    from pyspark.sql import functions as F
    m = spark.createDataFrame(plan_pdf[["community_id", "storage_part"]])
    return df.join(F.broadcast(m), on="community_id", how="inner")


def write_layout(df, path, layout, n_bins, user_metadata=None, max_records_per_file=0):
    """Write a Phase 2 table in the given layout (overwrite, schema and partitioning replaced)."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    key = STORAGE_PART if layout == "hive_merged" else "community_id"
    if layout == "hive_merged" and STORAGE_PART not in df.columns:
        raise ValueError("hive_merged needs a storage_part column (see with_storage_part)")
    order = [key, "community_id"] if key != "community_id" else ["community_id"]
    # _blk (add_blk) groups a community's rows by Phase 3b block, so a block's read skips the rest of the community
    sort = order + (["_blk"] if "_blk" in df.columns else [])
    out = df.repartitionByRange(int(n_bins), *order).sortWithinPartitions(*sort)
    w = (out.write.format("delta").mode("overwrite").option("overwriteSchema", "true"))
    if max_records_per_file:
        w = w.option("maxRecordsPerFile", int(max_records_per_file))
    if user_metadata:
        w = w.option("userMetadata", user_metadata)
    if layout != "range":
        w = w.partitionBy(key)
    w.save(path)


BLOCK_CAP_PROP = "lakegrl.block_cap"


def bit_reverse(x, bits):
    return int(format(int(x), f"0{bits}b")[::-1], 2) if bits else 0


def add_blk(spark, nodes_df, edges_df, cap):
    """Phase 2's tables with _blk: a row's Phase 3b block (id for nodes, src for edges, & (n - 1), n from
    utils.common.unit_blocks on the community's rows and `cap`), its k = log2 n bits reversed. A block of any split
    n_r <= n is then one contiguous _blk range (blk_range). Track B P4b (docs/trackB_storage_layout.md); a bit-reversed
    key is enough here because Phase 3b filters edges on src only. Returns (nodes, edges, units at this cap)."""
    from pyspark.sql import functions as F
    from pipeline.utils.common import unit_blocks
    cnt = nodes_df.groupBy("community_id").count().toPandas()
    cnt["_k"] = [unit_blocks(int(c), cap).bit_length() - 1 for c in cnt["count"]]
    cnt["_k"] = cnt["_k"].astype("int32")
    kmax = int(cnt["_k"].max()) if len(cnt) else 0
    kb = F.broadcast(spark.createDataFrame(cnt[["community_id", "_k"]]))
    rev = lambda key: F.expr(" + ".join(f"CASE WHEN _k > {d} THEN shiftleft(shiftright({key}, {d}) & 1, _k - {d + 1}) "
                                        f"ELSE 0 END" for d in range(kmax)) or "0").cast("int")
    nodes = nodes_df.join(kb, "community_id").withColumn("_blk", rev("id")).drop("_k")
    edges = edges_df.join(kb, "community_id").withColumn("_blk", rev("src")).drop("_k")
    return nodes, edges, int((2 ** cnt["_k"].astype("int64")).sum())


def blk_range(b, n_r, n_w):
    """Inclusive _blk range holding read block b of n_r on a table written with n_w blocks (add_blk). Finer reads
    (n_r > n_w) get their written block; the id / src filter keeps the rest."""
    k, m = n_w.bit_length() - 1, min(n_r, n_w).bit_length() - 1
    lo = bit_reverse(b & ((1 << m) - 1), m) << (k - m)
    return lo, lo + (1 << (k - m)) - 1


def sqrt_rg_mb(table_bytes, units, beta_mb=0.6):
    """Row-group size (MB, a power of two >= 1/4) from the √ rule s* = √(β T / (2 U)) (Track B P2: regret <= 1.46 on
    8 stores at every emulated latency). β = per-row-group overhead in bytes, about 0.6 MB locally."""
    import math
    s = math.sqrt(beta_mb * (table_bytes / (1 << 20)) / (2 * max(1, units)))
    return max(0.25, 2.0 ** round(math.log2(s))) if s > 0 else 0.25


def block_cap(spark, path):
    """The cap a Phase 2 table's _blk was written for (0: none)."""
    from delta.tables import DeltaTable
    df = spark.read.format("delta").load(path)
    if "_blk" not in df.columns:
        return 0
    props = DeltaTable.forPath(spark, path).detail().select("properties").first()[0] or {}
    return int(props.get(BLOCK_CAP_PROP, 0))


def current_files(spark, path):
    """Data files of the table's current snapshot.

    Reads at an explicit version: a plain load(path) is replaced by any cached
    DataFrame over the same path (e.g. Phase 3's persisted nodes), and
    inputFiles() of a cached relation is empty.
    """
    from delta.tables import DeltaTable
    v = int(DeltaTable.forPath(spark, path).history(1).select("version").first()[0])
    return sorted(spark.read.format("delta").option("versionAsOf", v).load(path).inputFiles())


def partition_columns(spark, path):
    from delta.tables import DeltaTable
    try:
        return list(DeltaTable.forPath(spark, path).detail().select("partitionColumns").first()[0])
    except Exception:
        return []


def layout_of(spark, path):
    cols = partition_columns(spark, path)
    if not cols:
        return "range"
    return "hive_merged" if cols == [STORAGE_PART] else "hive" if cols == ["community_id"] else "other"


def table_layout(spark, p_alg):
    """Everything a pyarrow worker needs to read the Phase 2 tables of one run.

    {"nodes": {"url", "files", "part_col", "part_type"}, "edges": {...},
     "layout": str, "comm_to_part": {community_id: storage_part} or None}
    """
    out = {}
    for name, key in (("nodes", "p2_nodes"), ("edges", "p2_edges")):
        url = p_alg[key]
        df = spark.read.format("delta").load(url)
        cols = partition_columns(spark, url)
        part_col = cols[0] if cols else None
        part_type = None
        if part_col:
            part_type = "string" if dict(df.dtypes)[part_col] == "string" else "int64"
        out[name] = {"url": url, "files": current_files(spark, url), "part_col": part_col,
                     "part_type": part_type}
    out["layout"] = {None: "range", "community_id": "hive", STORAGE_PART: "hive_merged"}.get(
        out["nodes"]["part_col"], "other")
    out["comm_to_part"] = None
    if out["layout"] == "hive_merged":
        rows = (spark.read.format("delta").load(p_alg["p2_nodes"])
                .select("community_id", STORAGE_PART).distinct().collect())
        out["comm_to_part"] = {int(r[0]): r[1] for r in rows}
    return out


def _local(uri):
    from urllib.parse import urlparse, unquote
    if uri.startswith("file:"):
        return unquote(urlparse(uri).path)
    return uri


def open_dataset(spec):
    """pyarrow dataset over one Phase 2 table described by table_layout()[name]."""
    import pyarrow as pa
    import pyarrow.dataset as ds
    files, url = spec.get("files"), spec["url"]
    part = None
    if spec.get("part_col"):
        typ = pa.string() if spec.get("part_type") == "string" else pa.int64()
        part = ds.partitioning(pa.schema([(spec["part_col"], typ)]), flavor="hive")
    if url.startswith(("s3://", "s3a://")):
        import pyarrow.fs as fs
        s3 = fs.S3FileSystem(region="us-east-1")
        strip = lambda u: u.split("://", 1)[1].rstrip("/")  # noqa: E731
        if files:
            return ds.dataset([strip(f) for f in files], filesystem=s3, format="parquet",
                              partitioning=part, partition_base_dir=strip(url) if part else None)
        return ds.dataset(strip(url), filesystem=s3, format="parquet", partitioning=part,
                          ignore_prefixes=["_delta_log", "."])
    base = _local(url).rstrip("/")
    if files:
        return ds.dataset([_local(f) for f in files], format="parquet", partitioning=part,
                          partition_base_dir=base if part else None)
    return ds.dataset(base, format="parquet", partitioning=part, ignore_prefixes=["_delta_log", "."])


def unit_filter(spec, comm_ids, comm_to_part=None):
    """Scan predicate for the given communities, with directory pruning when partitioned."""
    import pyarrow.dataset as ds
    f = ds.field("community_id").isin([int(c) for c in comm_ids])
    if spec.get("part_col") == STORAGE_PART and comm_to_part:
        parts = sorted({comm_to_part[int(c)] for c in comm_ids if int(c) in comm_to_part})
        f = ds.field(STORAGE_PART).isin(parts) & f
    return f


_RANGES = {}


def _hits(want, lo, hi):
    """Does the sorted list want hold a value in [lo, hi]?"""
    import bisect
    i = bisect.bisect_left(want, lo)
    return i < len(want) and want[i] <= hi


def unit_dataset(dataset, comm_ids, predicate=None):
    """The dataset restricted to the files and row groups whose community_id statistics can hold one of comm_ids.

    pyarrow's isin scan does not skip files or row groups by their statistics (Track B, arxiv/LPA: the units read
    87 GB of a 0.24 GB range table, 52 GB with hive_merged), so the reader skips them itself. predicate (the scan
    filter) first prunes partition directories; without it a partitioned dataset is returned as it is, since every
    footer would have to be read. Each file's footer is read once per process and cached; row groups without
    community_id statistics (hive files, where it is the directory) are kept. Rows read are unchanged.
    """
    import pyarrow.dataset as ds
    want = sorted({int(c) for c in comm_ids})
    frags = dataset.get_fragments() if predicate is None else dataset.get_fragments(filter=predicate)
    first = next(frags, None)
    if not want or (predicate is None and first is not None and str(first.partition_expression) != "true"):
        return dataset
    keep = []
    for frag in ([first] if first is not None else []) + list(frags):
        r = _RANGES.get(frag.path)
        if r is None:
            frag.ensure_complete_metadata()
            rgs = []
            for rg in frag.row_groups:
                st = (rg.statistics or {}).get("community_id")
                rgs.append((st["min"], st["max"]) if st and st.get("min") is not None else None)
            known = [mm for mm in rgs if mm is not None]
            whole = (min(m[0] for m in known), max(m[1] for m in known)) if known and len(known) == len(rgs) else None
            r = _RANGES[frag.path] = (whole, rgs)
        whole, rgs = r
        if whole is not None and not _hits(want, *whole):
            continue
        ids = [i for i, mm in enumerate(rgs) if mm is None or _hits(want, *mm)]
        if len(ids) == len(rgs):
            keep.append(frag)
        elif ids:
            keep.append(frag.subset(row_group_ids=ids))
    return ds.FileSystemDataset(keep, dataset.schema, dataset.format, dataset.filesystem)

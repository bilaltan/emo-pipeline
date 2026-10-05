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
    out = df.repartitionByRange(int(n_bins), *order).sortWithinPartitions(*order)
    w = (out.write.format("delta").mode("overwrite").option("overwriteSchema", "true"))
    if max_records_per_file:
        w = w.option("maxRecordsPerFile", int(max_records_per_file))
    if user_metadata:
        w = w.option("userMetadata", user_metadata)
    if layout != "range":
        w = w.partitionBy(key)
    w.save(path)


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

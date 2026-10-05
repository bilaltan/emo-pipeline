"""The two forms of the Phase 2 partition table, and how units get their features.

``materialized`` (LakeGRL default): ``p2_nodes`` is a materialized view whose
rows carry ``MATERIALIZED_COLUMNS``: the partitioning result plus a copy of
each node's label, features and split. A worker reads its own unit straight
from the table, with data skipping on ``community_id`` and no exchange between
workers. The price is that features are stored again (once per unit that
holds the node, halo included).

``normalized``: ``p2_nodes`` stores exactly ``PARTITION_COLUMNS``. Features and
labels stay in the raw ``nodes`` table and splits in ``masks`` (single source of
truth), and consumers get them back with one global Spark join
(``join_features``), after which the rows are shuffled to the workers by unit
(``attach_units``). Phase 2 records, in the partition table's commit metadata,
the Delta versions of ``nodes``/``masks``/``edges`` it was computed from, and the
join reads those pinned versions (time travel), so a partition and its
features always come from the same snapshot.

``load_unit_nodes`` returns unit rows with features for either form.
"""
import json

PARTITION_COLUMNS = ("id", "community_id", "is_boundary", "is_member")
MATERIALIZED_COLUMNS = ("id", "label", "features", "split",
                        "community_id", "is_boundary", "is_member")
FEATURE_COLUMNS = ("label", "features")
SOURCE_TABLES = ("nodes", "masks", "edges")


def delta_version(spark, path):
    """Latest committed version of the Delta table at ``path``."""
    from delta.tables import DeltaTable
    return int(DeltaTable.forPath(spark, path).history(1).select("version").first()[0])


def source_versions(spark, p):
    """Current versions of the raw tables a partition table is derived from."""
    out = {}
    for name in SOURCE_TABLES:
        try:
            out[name] = delta_version(spark, p[name])
        except Exception:
            pass
    return out


def partition_write_metadata(spark, p, algorithm=None):
    """JSON string for ``.option('userMetadata', ...)`` on the partition-table write."""
    return json.dumps({"lakegrl_partition_table": 1,
                       "algorithm": algorithm,
                       "source_versions": source_versions(spark, p)})


def pinned_versions(spark, p_alg):
    """Source versions recorded by Phase 2, or {} for tables written before this change.

    Takes the most recent commit that carries the metadata, so a later OPTIMIZE
    or other maintenance commit on the partition table does not drop the pin.
    """
    from delta.tables import DeltaTable
    from pyspark.sql import functions as F
    try:
        row = (DeltaTable.forPath(spark, p_alg["p2_nodes"]).history()
               .filter(F.col("userMetadata").contains("lakegrl_partition_table"))
               .orderBy(F.desc("version")).select("userMetadata").first())
        if row is None:
            return {}
        return {k: int(v) for k, v in json.loads(row[0]).get("source_versions", {}).items()}
    except Exception:
        return {}


def read_delta(spark, path, version=None):
    """Read a Delta table, at ``version`` when given (falls back to latest with a warning)."""
    if version is None:
        return spark.read.format("delta").load(path)
    try:
        df = spark.read.format("delta").option("versionAsOf", int(version)).load(path)
        df.schema  # resolve the snapshot now so a missing version fails here
        return df
    except Exception as e:
        print(f"  ⚠ Could not read {path} at pinned version {version} ({e}); using latest.")
        return spark.read.format("delta").load(path)


def is_materialized(df_or_columns):
    """True when a partition table (DataFrame or column list) carries features."""
    cols = getattr(df_or_columns, "columns", df_or_columns)
    return "features" in cols


def read_partition(spark, p_alg):
    """The partition table, reduced to its partitioning columns (either form)."""
    from pyspark.sql import functions as F
    df = spark.read.format("delta").load(p_alg["p2_nodes"])
    if "is_member" not in df.columns:
        df = df.withColumn("is_member", F.lit(True))
    if "is_boundary" not in df.columns:
        df = df.withColumn("is_boundary", F.lit(False))
    return df.select(*PARTITION_COLUMNS)


def join_features(spark, p_alg, partition_df=None, columns=FEATURE_COLUMNS,
                  with_split=True, pin=True):
    """Partition rows joined with their features (raw ``nodes``) and split (``masks``).

    Returns the columns the pre-change ``p2_nodes`` table carried:
    id, label, features, split, community_id, is_boundary, is_member.
    """
    part = read_partition(spark, p_alg) if partition_df is None else partition_df
    versions = pinned_versions(spark, p_alg) if pin else {}
    nodes = read_delta(spark, p_alg["nodes"], versions.get("nodes")).select("id", *columns)
    out = part.join(nodes, on="id", how="inner")
    if with_split:
        masks = read_delta(spark, p_alg["masks"], versions.get("masks")).select("id", "split")
        out = out.join(masks, on="id", how="left")
    ordered = ["id", *columns] + (["split"] if with_split else []) + \
              [c for c in PARTITION_COLUMNS if c != "id"]
    return out.select(*ordered)


def load_unit_nodes(spark, p_alg, p2_table=None):
    """Unit rows with features: id, label, features, split, community_id, is_boundary, is_member.

    Materialized partition table: the table itself, read as before (no join).
    Normalized partition table: the global join of the partition table with the
    raw nodes and masks tables at the versions Phase 2 recorded.
    """
    from pyspark.sql import functions as F
    df = spark.read.format("delta").load(p_alg["p2_nodes"]) if p2_table is None else p2_table
    if is_materialized(df):
        if "storage_part" in df.columns:          # hive_merged layout column, not data
            df = df.drop("storage_part")
        if "is_member" not in df.columns:
            df = df.withColumn("is_member", F.lit(True))
        return df
    return join_features(spark, p_alg)


def attach_units(spark, unit_nodes_df, work_pd):
    """Tag each unit row with its training unit (``_block``) and Spark bin (``bin_id``).

    ``work_pd`` is Phase 3b's work manifest (pandas) with community_id, _block,
    _n_blocks and bin_id. A community split into n blocks (n a power of two)
    sends node ``id`` to block ``id & (n - 1)``, the same rule the direct loader
    pushes into its parquet scan. Rows of communities not in the manifest are
    dropped. Returns the input columns plus ``_block`` and ``bin_id``.
    """
    from pyspark.sql import functions as F
    blocks = spark.createDataFrame(
        work_pd[["community_id", "_n_blocks"]].drop_duplicates().astype("int64"))
    units = spark.createDataFrame(
        work_pd[["community_id", "_block", "bin_id"]].astype("int64"))
    cols = list(unit_nodes_df.columns)
    return (unit_nodes_df
            .join(F.broadcast(blocks), on="community_id", how="inner")
            .withColumn("_block", F.col("id").bitwiseAND(F.col("_n_blocks") - F.lit(1)))
            .join(F.broadcast(units), on=["community_id", "_block"], how="inner")
            .select(*cols, "_block", "bin_id"))

"""CaaN's macro-graph (Eq. 1), stored by Phase 2 next to p2_nodes / p2_edges.

The macro-graph is part of the Phase 2 materialized view: one super-node per major community (>= min_size members;
features = the mean of its members' features) and, with tiny 'drop', every node of a minor community as itself;
edges = the raw edges with each end in a major community replaced by its super-node (-1000 - community_id), self
loops removed, distinct. Phase 3b built exactly this at the start of every run (phase3b_caan._build_macro), scanning
the whole raw edge table; with these tables it reads the macro-graph instead.

  p2_macro_nodes  macro_id (super-node -1000 - c, minor node its id), community_id, features, label, split
  p2_macro_edges  src, dst (macro ids)
  p2_unit_plan    community_id, count (p2_nodes rows: members + halo): Stage 3b plans its units from it instead of
                  reading p2_nodes' community_id column (one GET per row group)

All carry MACRO_PROP = the p2_nodes version Phase 2 wrote with them; Phase 3b uses them only when it matches.
"""
MACRO_PROP = "lakegrl.p2_nodes_version"
TABLES = ("p2_macro_nodes", "p2_macro_edges", "p2_unit_plan")


def mean_features(pdf):
    """One community's super-node features: the mean of its members' feature vectors (applyInPandas)."""
    import numpy as np
    import pandas as pd
    cid = pdf['community_id'].iloc[0]
    # rows in id order: Spark's group order varies, and the float32 sum differs with it (Reddit: up to 7.9e-6)
    feats = np.stack(pdf.sort_values('id', kind='mergesort')['features'].values)
    mean_feat = feats.mean(axis=0).tolist()
    return pd.DataFrame([{'community_id': cid, 'mean_features': mean_feat}])


def macro_tables(comms_df, major_df, nodes_df, masks_df, edges_df, with_minor):
    """(nodes, edges) Spark DataFrames of the macro-graph. comms_df: Phase 1's (id, community_id); major_df: the
    major communities' community_id; with_minor: tiny 'drop' (minor-community nodes are the global model's)."""
    from pyspark.sql import functions as F
    c = (comms_df.select('id', 'community_id')
         .join(F.broadcast(major_df.select('community_id').withColumn('_maj', F.lit(True))), 'community_id', 'left'))
    sup = (nodes_df.select('id', 'features').join(F.broadcast(c.filter(F.col('_maj')).select('id', 'community_id')), 'id')
           .groupBy('community_id')
           .applyInPandas(mean_features, schema='community_id long, mean_features array<float>')
           .select((F.lit(-1000) - F.col('community_id')).alias('macro_id'), 'community_id',
                   F.col('mean_features').alias('features'),
                   F.lit(None).cast('long').alias('label'), F.lit(None).cast('string').alias('split')))
    nodes = sup
    if with_minor:
        minor = (nodes_df.select('id', 'label', 'features')
                 .join(F.broadcast(c.filter(F.col('_maj').isNull()).select('id', 'community_id')), 'id')
                 .join(masks_df.select('id', 'split'), 'id', 'left')
                 .select(F.col('id').alias('macro_id'), 'community_id', 'features', 'label', 'split'))
        nodes = sup.unionByName(minor)
    m = c.select('id', F.when(F.col('_maj'), F.lit(-1000) - F.col('community_id')).otherwise(F.col('id')).alias('m'))
    edges = (edges_df.select('src', 'dst')
             .join(F.broadcast(m.toDF('src', 'u')), 'src', 'left')
             .join(F.broadcast(m.toDF('dst', 'v')), 'dst', 'left')
             .select(F.coalesce('u', 'src').alias('src'), F.coalesce('v', 'dst').alias('dst'))
             .filter(F.col('src') != F.col('dst'))
             .distinct())
    return nodes, edges


def write_macro(spark, p_alg, nodes, edges, stats):
    """One file per table (super-nodes first), stamped with the p2_nodes version they belong to. stats: Phase 2's
    community_stats of the p2_nodes rows (community_id, rows)."""
    import os
    from pyspark.sql import functions as F
    from pipeline.utils.partition_join import delta_version
    if os.environ.get('LAKEGRL_MACRO_F16'):   # trial: features as float16 bytes (Spark has no float16 type)
        import numpy as np
        nodes = nodes.withColumn('features', F.udf(lambda f: np.asarray(f, np.float16).tobytes(), 'binary')('features'))
    v = delta_version(spark, p_alg['p2_nodes'])
    plan = spark.createDataFrame(stats[['community_id', 'rows']].rename(columns={'rows': 'count'}).astype('int64'))
    for df, k, order in ((edges, 'p2_macro_edges', ['src', 'dst']),
                         (plan, 'p2_unit_plan', ['community_id']),
                         (nodes, 'p2_macro_nodes', [F.col('macro_id') >= 0, 'community_id', 'macro_id'])):
        (df.repartition(1).sortWithinPartitions(*order)
         .write.format('delta').mode('overwrite').option('overwriteSchema', 'true').save(p_alg[k]))
        spark.sql(f"ALTER TABLE delta.`{p_alg[k]}` SET TBLPROPERTIES ('{MACRO_PROP}' = '{v}')")


def stored(spark, p_alg):
    """True if Phase 2 stored the macro-graph with the current p2_nodes."""
    from pipeline.utils.store_io import properties, snapshot
    if not all(k in p_alg for k in TABLES):
        return False
    try:   # read from the Delta logs: DeltaTable.detail() took ~3 s of each run's macro-graph load
        v = str(snapshot(p_alg['p2_nodes'])[2])
        return all(properties(p_alg[k]).get(MACRO_PROP) == v for k in TABLES)
    except Exception:   # no such table: Phase 2 ran without macro_graph
        return False


def read_small(uri):
    """A small Delta table as pandas, read with pyarrow from its current files (store_io.snapshot), no Spark job:
    a Spark read + toPandas cost 0.5-1.3 s per table of each products run's macro-graph load."""
    import pyarrow.parquet as pq
    from pipeline.utils.store_io import snapshot
    fs, files, _ = snapshot(uri)
    return pq.read_table(files, filesystem=fs).to_pandas()


def load_macro(spark, p_alg):
    """(super_nodes_dict, minor_nodes_dict, caan_edges) in the form phase3b_caan._build_macro returns them."""
    import numpy as np
    import pandas as pd
    mn = read_small(p_alg['p2_macro_nodes'])
    if len(mn) and isinstance(mn['features'].iloc[0], (bytes, bytearray)):   # stored as float16 (LAKEGRL_MACRO_F16)
        mn['features'] = [np.frombuffer(f, np.float16).astype(np.float32) for f in mn['features']]
    sup, minor = mn[mn['macro_id'] < 0], mn[mn['macro_id'] >= 0]
    # features as float32 arrays, not lists of floats (0.7 s on products, and 0.17 s per unit to turn back)
    super_nodes_dict = {int(c): np.asarray(f, np.float32) for c, f in zip(sup['community_id'], sup['features'])}
    minor_nodes_dict = {int(i): {'features': np.asarray(f, np.float32), 'label': int(l) if not pd.isna(l) else -1,
                                 'split': str(s)}
                        for i, f, l, s in zip(minor['macro_id'], minor['features'], minor['label'], minor['split'])}
    me = read_small(p_alg['p2_macro_edges'])
    return super_nodes_dict, minor_nodes_dict, list(zip(me['src'].tolist(), me['dst'].tolist()))


def load_plan(spark, p_alg):
    """p2_nodes rows per community (community_id, count), as p2_nodes.groupBy('community_id').count()."""
    return read_small(p_alg['p2_unit_plan'])

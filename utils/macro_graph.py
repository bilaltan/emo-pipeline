"""CaaN's macro-graph (Eq. 1), stored by Phase 2 next to p2_nodes / p2_edges (advisor 2026-11-04).

The macro-graph is part of the Phase 2 materialized view: one super-node per major community (>= min_size members;
features = the mean of its members' features) and, with tiny 'drop', every node of a minor community as itself;
edges = the raw edges with each end in a major community replaced by its super-node (-1000 - community_id), self
loops removed, distinct. Phase 3b built exactly this at the start of every run (phase3b_caan._build_macro), scanning
the whole raw edge table; with these tables it reads the macro-graph instead.

  p2_macro_nodes  macro_id (super-node -1000 - c, minor node its id), community_id, features, label, split
  p2_macro_edges  src, dst (macro ids)

Both carry MACRO_PROP = the p2_nodes version Phase 2 wrote with them; Phase 3b uses them only when it matches.
"""
MACRO_PROP = "lakegrl.p2_nodes_version"
TABLES = ("p2_macro_nodes", "p2_macro_edges")


def mean_features(pdf):
    """One community's super-node features: the mean of its members' feature vectors (applyInPandas)."""
    import numpy as np
    import pandas as pd
    cid = pdf['community_id'].iloc[0]
    feats = np.stack(pdf['features'].values)
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


def write_macro(spark, p_alg, nodes, edges):
    """One file per table (super-nodes first), stamped with the p2_nodes version they belong to."""
    from pyspark.sql import functions as F
    from pipeline.utils.partition_join import delta_version
    v = delta_version(spark, p_alg['p2_nodes'])
    for df, k, order in ((edges, 'p2_macro_edges', ['src', 'dst']),
                         (nodes, 'p2_macro_nodes', [F.col('macro_id') >= 0, 'community_id', 'macro_id'])):
        (df.repartition(1).sortWithinPartitions(*order)
         .write.format('delta').mode('overwrite').option('overwriteSchema', 'true').save(p_alg[k]))
        spark.sql(f"ALTER TABLE delta.`{p_alg[k]}` SET TBLPROPERTIES ('{MACRO_PROP}' = '{v}')")


def stored(spark, p_alg):
    """True if Phase 2 stored the macro-graph with the current p2_nodes."""
    from delta.tables import DeltaTable
    from pipeline.utils.partition_join import delta_version
    if not all(k in p_alg for k in TABLES):
        return False
    try:
        v = str(delta_version(spark, p_alg['p2_nodes']))
        return all((DeltaTable.forPath(spark, p_alg[k]).detail().first()['properties'] or {}).get(MACRO_PROP) == v
                   for k in TABLES)
    except Exception:   # no such table: Phase 2 ran without macro_graph
        return False


def load_macro(spark, p_alg):
    """(super_nodes_dict, minor_nodes_dict, caan_edges) in the form phase3b_caan._build_macro returns them."""
    import pandas as pd
    mn = spark.read.format('delta').load(p_alg['p2_macro_nodes']).toPandas()
    sup, minor = mn[mn['macro_id'] < 0], mn[mn['macro_id'] >= 0]
    super_nodes_dict = {int(c): [float(x) for x in f] for c, f in zip(sup['community_id'], sup['features'])}
    minor_nodes_dict = {int(i): {'features': [float(x) for x in f], 'label': int(l) if not pd.isna(l) else -1,
                                 'split': str(s)}
                        for i, f, l, s in zip(minor['macro_id'], minor['features'], minor['label'], minor['split'])}
    me = spark.read.format('delta').load(p_alg['p2_macro_edges']).toPandas()
    return super_nodes_dict, minor_nodes_dict, list(zip(me['src'].tolist(), me['dst'].tolist()))

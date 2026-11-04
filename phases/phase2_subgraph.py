import os
import sys
import time

def run_phase2(spark, sc, datasets, algorithms, use_global_mapping, min_size,
               get_paths_fn, timing, results, **kwargs):
    """
    Build induced subgraphs per community for each (dataset, algorithm).
    Adds is_boundary flag to nodes (loses ≥1 edge due to inter-community cut).
    ISOLATION: writes to phase2_nodes/{tag}/ and phase2_edges/{tag}/.

    partition_table_mode (kwarg):
      'materialized' (default) -- phase2_nodes is a materialized view: each
          partition row also carries the node's label, features and split, so a
          worker reads its unit directly with data skipping on community_id.
      'normalized' -- phase2_nodes stores only the partitioning result
          (id, community_id, is_boundary, is_member). Features stay in the raw
          nodes table and splits in masks; readers join them back
          (utils/partition_join.py). The commit records the raw-table versions
          the partition was computed from, and readers join against those.
    """
    tiny_comm_handling = kwargs.get('tiny_comm_handling', 'misc')
    expand_boundary_nodes = kwargs.get('expand_boundary_nodes', False)
    partition_table_mode = kwargs.get('partition_table_mode', 'materialized')
    # Physical layout of p2_nodes / p2_edges (utils/partition_layout.py):
    # 'range' (default), 'hive' (partitionBy community_id) or 'hive_merged'
    # (partitionBy storage_part, small communities packed into shared parts).
    p2_layout = kwargs.get('p2_layout', 'range')
    merge_minor_members = int(kwargs.get('merge_minor_members', min_size))
    merge_min_rows = int(kwargs.get('merge_min_rows', 0))
    merge_target_rows = int(kwargs.get('merge_target_rows', 250_000))
    from pipeline.utils.partition_layout import LAYOUTS as _P2_LAYOUTS
    if p2_layout not in _P2_LAYOUTS:
        raise ValueError(f"p2_layout must be one of {_P2_LAYOUTS}, got {p2_layout!r}")
    if partition_table_mode not in ('materialized', 'normalized'):
        raise ValueError(f"partition_table_mode must be 'materialized' or 'normalized', "
                         f"got {partition_table_mode!r}")
    normalized = partition_table_mode == 'normalized'
    
    from pyspark.sql import functions as F

    for dataset in datasets:
        p = get_paths_fn(dataset)
        print(f"\n{'='*60}\n  PHASE 2 — Partitioning: {dataset}\n{'='*60}")

        nodes_df  = spark.read.format('delta').load(p['nodes'])
        edges_df  = spark.read.format('delta').load(p['edges'])
        masks_df  = spark.read.format('delta').load(p['masks'])
        n_edges   = edges_df.count()

        for alg in algorithms:
            key   = (dataset, alg)
            p_alg = get_paths_fn(dataset, alg)
            
            # Check if checkpoint exists
            force_rerun = kwargs.get('force_rerun', False)
            if not force_rerun:
                try:
                    # Try loading target paths
                    nodes_final = spark.read.format('delta').load(p_alg['p2_nodes'])
                    edges_part = spark.read.format('delta').load(p_alg['p2_edges'])
                    # A checkpoint written in the other partition_table_mode is
                    # not reusable; fall through and rebuild it.
                    if ('features' in nodes_final.columns) == normalized:
                        raise ValueError('phase2_nodes checkpoint has the other partition_table_mode')
                    from pipeline.utils.partition_layout import layout_of
                    if layout_of(spark, p_alg['p2_nodes']) != p2_layout:
                        raise ValueError('phase2_nodes checkpoint has another p2_layout')
                    
                    comms_df = spark.read.format('delta').load(p_alg['communities'])
                    n_raw = comms_df.select('community_id').distinct().count()
                    
                    # Compute stats
                    n_kept = nodes_final.count()
                    n_boundary = nodes_final.filter(F.col('is_boundary')).count()
                    n_intra = edges_part.count()
                    pct_kept = 100 * n_intra / n_edges if n_edges > 0 else 0.0
                    
                    # Calculate valid comms (excluding tiny comm group community_id = -1)
                    n_valid = nodes_final.filter(F.col('community_id') >= 0).select('community_id').distinct().count()
                    
                    print(f"\n  ── [{alg.upper()}] (Loaded Subgraphs from S3 Checkpoint) ──")
                    print(f"    ✓ Loaded {n_kept:,} nodes, {n_intra:,} intra-edges. Skipping subgraph extraction.")
                    
                    timing[('phase2', dataset, alg)] = 0.0
                    results[key] = {
                        'n_comms_raw':   n_raw,  'n_valid_comms': n_valid,
                        'n_nodes_kept':  n_kept, 'n_boundary':    n_boundary,
                        'n_internal':    n_kept - n_boundary,
                        'n_intra_edges': n_intra,'pct_edges_kept': pct_kept,
                        'shuffle_s':     0.0,
                    }
                    continue
                except Exception:
                    pass

            t0    = time.time()
            print(f"\n  ── [{alg.upper()}]  tag={p_alg['tag']} ──")

            # advisor 2026-11-04: no diagnostic counts in Phase 2's time; only the counts the layout needs run
            comms_df = spark.read.format('delta').load(p_alg['communities']).cache()

            # Filter by minimum size
            comm_sizes   = comms_df.groupBy('community_id')\
                                   .agg(F.count('*').alias('comm_size'))
            valid_comms  = comm_sizes.filter(F.col('comm_size') >= min_size).cache()
            n_valid      = valid_comms.count()

            if tiny_comm_handling in ('misc', 'micro_chunk'):
                invalid_comms = comm_sizes.filter(F.col('comm_size') < min_size)
                # Micro-chunk tiny communities into ~1,000 balanced bins of ~1,500 nodes each
                # Hashing community_id keeps adjacent nodes together, avoiding halo neighbor explosion
                tiny_nodes = comms_df.join(invalid_comms.select('community_id'), on='community_id', how='inner')
                tiny_nodes_binned = (tiny_nodes
                                     .withColumn('new_community_id', -1 - (F.abs(F.hash('community_id')) % 1000).cast('long'))
                                     .select('id', F.col('new_community_id').alias('community_id')))
                
                valid_nodes = comms_df.join(valid_comms.select('community_id'), on='community_id', how='inner')
                # unionByName: the join puts community_id first, so a positional
                # union swapped id and community_id in every tiny-node row, and
                # those rows then matched no node (tiny communities were dropped).
                comms_filt = valid_nodes.unionByName(tiny_nodes_binned)
            elif tiny_comm_handling == 'keep':
                # Every community stays its own unit, however small.
                comms_filt = comms_df
            else:
                comms_filt = comms_df.join(valid_comms.select('community_id'), on='community_id', how='inner')

            comms_filt = comms_filt.select('id', 'community_id')

            print(f"  Valid(≥{min_size})={n_valid:,}")

            # Build partitioned nodes. The materialized view copies each node's
            # label, features and split into its partition rows; the normalized
            # table keeps ids only, so this phase never reads the feature column.
            node_comm = comms_filt.select('id', 'community_id').cache()
            if normalized:
                nodes_w_split = nodes_df.select('id')
                node_cols = ['id']
            else:
                nodes_w_split = nodes_df.join(masks_df.select('id', 'split'),
                                              on='id', how='left')
                node_cols = ['id', 'label', 'features', 'split']
            nodes_part = (nodes_w_split
                          .join(node_comm, on='id', how='inner')
                          .select(*node_cols, 'community_id'))

            # Intra-community edges
            e_src = (edges_df.join(node_comm.withColumnRenamed('id','src').withColumnRenamed('community_id','src_comm'), on='src', how='inner'))
            # 'drop': minor-community vertices leave the units but stay as major units' 1-hop halo (their edges kept),
            # as in CaaN Eq. 2, where a community's local graph includes the minor nodes; Phase 3b's global model
            # scores them. Looking dst up in the kept units only would cut those edges.
            dst_map = comms_df.select('id', 'community_id') if tiny_comm_handling == 'drop' else node_comm
            e_full = (e_src.join(dst_map.withColumnRenamed('id','dst').withColumnRenamed('community_id','dst_comm'), on='dst', how='inner'))
            
            if expand_boundary_nodes:
                edges_part = e_full.select('src', 'dst', F.col('src_comm').alias('community_id'))
                halo_nodes = e_full.filter(F.col('src_comm') != F.col('dst_comm')).select(F.col('dst').alias('id'), F.col('src_comm').alias('community_id')).distinct()
                halo_cap = int(kwargs.get('halo_cap') or 0)
                if halo_cap:
                    # The 1:1 capped halo (advisor, 2026-10-30): a unit keeps at most min(members, C_max - members)
                    # halo vertices, those with the most edges into it (ties: lower id; the halo sub-table's capped
                    # rule, R = 1), and loses its edges to the others. Phase 1 keeps members <= C_max / 2, so a unit
                    # stores <= C_max rows.
                    from pyspark.sql import Window
                    over = (node_comm.groupBy('community_id').agg(F.count('*').alias('_m'))
                            .join(halo_nodes.groupBy('community_id').agg(F.count('*').alias('_h')), 'community_id')
                            .withColumn('_m', F.least('_m', F.lit(halo_cap) - F.col('_m')))   # the halo budget
                            .filter(F.col('_h') > F.col('_m')).select('community_id', '_m').toPandas())
                    if len(over):
                        ob = F.broadcast(spark.createDataFrame(over).withColumnRenamed('community_id', 'src_comm'))
                        drop = (e_full.filter(F.col('src_comm') != F.col('dst_comm')).join(ob, 'src_comm')
                                .groupBy('src_comm', 'dst', '_m').count()
                                .withColumn('_r', F.row_number().over(
                                    Window.partitionBy('src_comm').orderBy(F.desc('count'), 'dst')))
                                .filter(F.col('_r') > F.col('_m'))
                                .select(F.col('dst').alias('id'), F.col('src_comm').alias('community_id')).cache())
                        halo_nodes = halo_nodes.join(F.broadcast(drop), ['id', 'community_id'], 'left_anti')
                        edges_part = edges_part.join(F.broadcast(drop.withColumnRenamed('id', 'dst')),
                                                     ['dst', 'community_id'], 'left_anti')
                        print(f"  Halo capped at min(members, {halo_cap:,} - members): {len(over):,} units above it")
                # is_member separates a community's own vertices from the 1-hop halo
                # pulled in from neighbours. Without it a halo vertex is
                # indistinguishable from a member downstream, so it gets scored as a
                # link-prediction target and counted in this community's accuracy —
                # once per community that borrows it.
                expanded_node_comm = (node_comm.withColumn('is_member', F.lit(True))
                                      .union(halo_nodes.withColumn('is_member', F.lit(False)))
                                      .groupBy('id', 'community_id')
                                      .agg(F.max('is_member').alias('is_member')))
                nodes_part = nodes_w_split.join(expanded_node_comm, on='id', how='inner').select(*node_cols, 'community_id', 'is_member')
            else:
                edges_part = e_full.filter(F.col('src_comm') == F.col('dst_comm')).select('src', 'dst', F.col('src_comm').alias('community_id'))
                nodes_part = nodes_w_split.join(node_comm, on='id', how='inner').select(*node_cols, 'community_id').withColumn('is_member', F.lit(True))
            
            n_intra    = edges_part.count()
            pct_kept   = 100 * n_intra / n_edges if n_edges > 0 else 0.0

            # Boundary flag: Compute degrees directly from edges and join directly to nodes_part (zero edge-level joins against node_comm)
            orig_deg = (edges_df
                        .groupBy('src').count()
                        .withColumnRenamed('src', 'id')
                        .withColumnRenamed('count', 'orig_deg'))
            # Boundary is a property of the partition, not of which edges were kept:
            # b(v)=1 iff v has a neighbour in another community (Eq. 1). Deriving it
            # from edges_part makes it identically False under 1-hop expansion, since
            # that variant retains the cut edges it is meant to detect.
            intra_deg = (e_full
                         .filter(F.col('src_comm') == F.col('dst_comm'))
                         .groupBy('src').count()
                         .withColumnRenamed('src', 'id')
                         .withColumnRenamed('count', 'intra_deg'))
            
            nodes_final = (nodes_part
                           .join(orig_deg, on='id', how='left')
                           .join(intra_deg, on='id', how='left')
                           .withColumn('orig_deg', F.coalesce('orig_deg', F.lit(0)))
                           .withColumn('intra_deg', F.coalesce('intra_deg', F.lit(0)))
                           .withColumn('is_boundary', (F.col('orig_deg') > F.col('intra_deg')).cast('boolean'))
                           .withColumn('is_boundary', F.coalesce('is_boundary', F.lit(False)))
                           .select(*node_cols, 'community_id', 'is_boundary', 'is_member'))

            p2_times = {'build_s': time.time() - t0}   # advisor 11-03: Phase 2 runtime breakdown
            print(f"  Edges kept: {n_intra:,} ({pct_kept:.1f}%)")

            # Write — isolated per tag
            # Range, not hash. Phase 3b reads these tables with a predicate on
            # community_id, once per (community, block) unit
            # (phase3b_caan.py:1627,1636), and parquet prunes on each file's
            # [min,max]. A hash shuffle scatters neighbouring ids across every
            # file, so those ranges span almost the whole id space and nothing
            # can be skipped: measured 109 MB read per unit from a 375 MB table
            # on ogbn-mag, 3.48 GB across 34 units. Range partitioning keeps a
            # community's rows together and cuts that to 10.2 MB per unit,
            # 97.3% skipped (results/zorder/LAYOUT.md).
            # Phase 3 itself does not use this path: its copy of the pyarrow
            # loader (phase3_training.py:66) is never called, and units are fed
            # by groupBy().applyInPandas() instead.
            p2_bins = min(200, n_valid) if n_valid > 0 else 200
            # write_layout overwrites with overwriteSchema: the path may hold a
            # table written in another partition_table_mode or p2_layout.
            from pipeline.utils.partition_layout import (write_layout, community_stats,
                                                         merge_plan, with_storage_part,
                                                         add_blk, sqrt_rg_mb, BLOCK_CAP_PROP)
            meta = None
            if normalized:
                from pipeline.utils.partition_join import partition_write_metadata
                meta = partition_write_metadata(spark, p, alg)
            nodes_out, edges_out = nodes_final, edges_part
            # Track B storage layout (docs/trackB_storage_layout.md): physical only, every unit reads the same rows.
            # block_cap: rows grouped by Phase 3b block (_blk) for that cap, so a block read skips the rest of its
            # community. rg_mb: row-group size; 'auto' = the √ rule from the table size and the unit count.
            blk_cap = int(kwargs.get('block_cap') or 0)
            n_units = n_valid
            if blk_cap:
                nodes_out, edges_out, n_units = add_blk(spark, nodes_out, edges_out, blk_cap)
            rg_mb = kwargs.get('rg_mb')
            # rows per community (members + halo): the merge plan, and their sum is the √ rule's table size
            stats = community_stats(nodes_final) if (p2_layout == 'hive_merged' or rg_mb == 'auto') else None
            n_total_nodes = int(stats['rows'].sum()) if stats is not None else None
            if rg_mb == 'auto':
                from delta.tables import DeltaTable
                size = lambda u: int(DeltaTable.forPath(spark, u).detail().first()['sizeInBytes'])
                t_est = (size(p['nodes']) * n_total_nodes / max(1, nodes_df.count())
                         + size(p['edges']) * n_intra / max(1, n_edges))
                rg_mb = sqrt_rg_mb(t_est, n_units, float(kwargs.get('rg_beta_mb', 0.6)),
                                   float(kwargs.get('rg_floor_mb') or 0.0))
                print(f"  Row group (√ rule): {rg_mb:g} MB from ~{t_est / 2**20:,.0f} MB over {n_units:,} units")
            hconf = sc._jsc.hadoopConfiguration()
            rg_prev = hconf.get('parquet.block.size')
            if rg_mb:
                hconf.set('parquet.block.size', str(int(float(rg_mb) * (1 << 20))))
            if p2_layout == 'hive_merged':
                plan = merge_plan(stats, merge_minor_members, merge_min_rows, merge_target_rows)
                n_shared = int(plan.loc[plan['small'], 'storage_part'].nunique())
                print(f"  p2_layout=hive_merged: {int(plan['small'].sum()):,} small communities -> "
                      f"{n_shared} shared partitions; {int((~plan['small']).sum()):,} own partitions")
                nodes_out = with_storage_part(spark, nodes_out, plan)
                edges_out = with_storage_part(spark, edges_out, plan)
                p2_bins = min(200, max(1, plan['storage_part'].nunique()))
            p2_times['plan_s'] = time.time() - t0 - sum(p2_times.values())
            if kwargs.get('time_split'):
                # Table B breakdown run only: the rows computed and cached first, so the write below is the
                # Parquet write alone (as Table A's write column); a normal run computes them inside the write
                from pyspark import StorageLevel
                nodes_out, edges_out = (d.persist(StorageLevel.MEMORY_AND_DISK) for d in (nodes_out, edges_out))
                nodes_out.count(), edges_out.count()
                p2_times['materialize_s'] = time.time() - t0 - sum(p2_times.values())
            # one_file_per_dir: one file per directory, as Table A rows 6-8 (row 8b = this + rg floor 2 MB)
            one_file = bool(kwargs.get('one_file_per_dir', False))
            write_layout(nodes_out, p_alg['p2_nodes'], p2_layout, p2_bins, user_metadata=meta,
                         one_file_per_dir=one_file)
            write_layout(edges_out, p_alg['p2_edges'], p2_layout, p2_bins, one_file_per_dir=one_file)
            if rg_mb:   # the row group applies to Phase 2's tables only
                hconf.set('parquet.block.size', rg_prev) if rg_prev is not None else hconf.unset('parquet.block.size')
            if blk_cap:
                for k in ('p2_nodes', 'p2_edges'):
                    spark.sql(f"ALTER TABLE delta.`{p_alg[k]}` SET TBLPROPERTIES ('{BLOCK_CAP_PROP}' = '{blk_cap}')")

            p2_times['write_s'] = time.time() - t0 - sum(p2_times.values())
            if kwargs.get('time_split'):
                nodes_out.unpersist(), edges_out.unpersist()
            if kwargs.get('macro_graph'):
                # CaaN's macro-graph, stored with the view (advisor 2026-11-04); Phase 3b reads it (utils.macro_graph)
                from pipeline.utils.macro_graph import macro_tables, write_macro
                write_macro(spark, p_alg, *macro_tables(comms_df, valid_comms, nodes_df, masks_df, edges_df,
                                                        with_minor=tiny_comm_handling == 'drop'),
                            stats if stats is not None else community_stats(nodes_final))
                p2_times['macro_s'] = time.time() - t0 - sum(p2_times.values())
            elapsed = time.time() - t0
            timing[('phase2', dataset, alg)] = elapsed
            results[key] = {
                'n_comms_raw':   None,  'n_valid_comms': n_valid,   # the None fields: diagnostic counts, not run
                'n_nodes_kept':  None, 'n_boundary':    None,
                'n_internal':    None,
                'n_intra_edges': n_intra,'pct_edges_kept': pct_kept,
                'shuffle_s':    None,
                'partition_table_mode': partition_table_mode,
                'p2_layout': p2_layout,
                'block_cap': blk_cap, 'rg_mb': rg_mb, 'n_units': n_units, 'one_file_per_dir': one_file,
                'times': p2_times,
            }
            print(f"  ✓ [{alg}] Phase 2 done in {elapsed:.1f}s ({partition_table_mode} partition table, {p2_layout} layout)")
            print("    steps: " + ", ".join(f"{k} {v:.1f}" for k, v in p2_times.items()))
            
            node_comm.unpersist()
            valid_comms.unpersist()
            comms_df.unpersist()
            spark._jvm.System.gc()

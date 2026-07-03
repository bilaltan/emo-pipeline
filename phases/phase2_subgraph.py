import os
import sys
import time

def run_phase2(spark, sc, datasets, algorithms, use_global_mapping, min_size,
               get_paths_fn, timing, results, **kwargs):
    """
    Build induced subgraphs per community for each (dataset, algorithm).
    Adds is_boundary flag to nodes (loses ≥1 edge due to inter-community cut).
    ISOLATION: writes to phase2_nodes/{tag}/ and phase2_edges/{tag}/.
    """
    tiny_comm_handling = kwargs.get('tiny_comm_handling', 'misc')
    expand_boundary_nodes = kwargs.get('expand_boundary_nodes', False)
    
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

            comms_df = spark.read.format('delta').load(p_alg['communities']).cache()
            n_raw    = comms_df.select('community_id').distinct().count()

            # Filter by minimum size
            comm_sizes   = comms_df.groupBy('community_id')\
                                   .agg(F.count('*').alias('comm_size'))
            valid_comms  = comm_sizes.filter(F.col('comm_size') >= min_size).cache()
            n_valid      = valid_comms.count()
            
            if n_valid > 0:
                size_stats   = valid_comms.agg(
                    F.min('comm_size').alias('mn'), F.max('comm_size').alias('mx'),
                    F.avg('comm_size').alias('avg')).collect()[0]
                mn, mx, avg = size_stats['mn'], size_stats['mx'], size_stats['avg']
            else:
                mn, mx, avg = 0, 0, 0.0

            comms_filt = comms_df.join(valid_comms.select('community_id'), on='community_id', how='inner')
            if tiny_comm_handling == 'misc':
                invalid_comms = comm_sizes.filter(F.col('comm_size') < min_size)
                # Assign community_id = -1 to tiny communities
                comms_w_misc = comms_df.join(invalid_comms.select('community_id').withColumn('is_tiny', F.lit(True)), on='community_id', how='left')
                comms_w_misc = comms_w_misc.withColumn('community_id', F.when(F.col('is_tiny'), F.lit(-1)).otherwise(F.col('community_id'))).drop('is_tiny')
                comms_filt = comms_w_misc

            print(f"  Raw={n_raw:,}  Valid(≥{min_size})={n_valid:,}  "
                  f"min={mn:,}  max={mx:,}  "
                  f"avg={avg:.1f}")

            # Build partitioned nodes
            node_comm = comms_filt.select('id', 'community_id').cache()
            nodes_w_split = nodes_df.join(masks_df.select('id', 'split'),
                                          on='id', how='left')
            nodes_part = (nodes_w_split
                          .join(node_comm, on='id', how='inner')
                          .select('id', 'label', 'features', 'split', 'community_id'))
            n_kept = nodes_part.count()

            # Intra-community edges
            e_src = (edges_df.join(node_comm.withColumnRenamed('id','src').withColumnRenamed('community_id','src_comm'), on='src', how='inner'))
            e_full = (e_src.join(node_comm.withColumnRenamed('id','dst').withColumnRenamed('community_id','dst_comm'), on='dst', how='inner'))
            
            if expand_boundary_nodes:
                edges_part = e_full.select('src', 'dst', F.col('src_comm').alias('community_id'))
                halo_nodes = e_full.filter(F.col('src_comm') != F.col('dst_comm')).select(F.col('dst').alias('id'), F.col('src_comm').alias('community_id')).distinct()
                expanded_node_comm = node_comm.union(halo_nodes).distinct()
                nodes_part = nodes_w_split.join(expanded_node_comm, on='id', how='inner').select('id', 'label', 'features', 'split', 'community_id')
            else:
                edges_part = e_full.filter(F.col('src_comm') == F.col('dst_comm')).select('src', 'dst', F.col('src_comm').alias('community_id'))
                nodes_part = nodes_w_split.join(node_comm, on='id', how='inner').select('id', 'label', 'features', 'split', 'community_id')
            
            n_intra    = edges_part.count()
            pct_kept   = 100 * n_intra / n_edges if n_edges > 0 else 0.0

            # Boundary flag
            orig_deg  = (edges_df
                         .join(node_comm.withColumnRenamed('id', 'src'), on='src', how='inner')
                         .groupBy('src').count()
                         .withColumnRenamed('src', 'id')
                         .withColumnRenamed('count', 'orig_deg'))
            intra_deg = (edges_part
                         .groupBy('src').count()
                         .withColumnRenamed('src', 'id')
                         .withColumnRenamed('count', 'intra_deg'))
            deg_df    = (orig_deg.join(intra_deg, on='id', how='left')
                         .withColumn('intra_deg', F.coalesce('intra_deg', F.lit(0)))
                         .withColumn('is_boundary',
                                     (F.col('orig_deg') > F.col('intra_deg'))
                                     .cast('boolean'))
                         .select('id', 'is_boundary'))
            nodes_final = (nodes_part.join(deg_df, on='id', how='left')
                           .withColumn('is_boundary',
                                       F.coalesce('is_boundary', F.lit(False))))

            n_boundary = nodes_final.filter(F.col('is_boundary')).count()
            print(f"  Edges kept: {n_intra:,} ({pct_kept:.1f}%)  "
                  f"Boundary nodes: {n_boundary:,}  Internal: {n_kept-n_boundary:,}")

            # Shuffle overhead metric
            sc.setJobDescription(f'shuffle_p2_{dataset}_{alg}')
            t_sh = time.time()
            sh_df = nodes_final.groupBy('community_id').count().cache()
            sh_df.count()
            shuffle_s = time.time() - t_sh
            sh_df.unpersist()
            sc.setJobDescription('')
            print(f"  Shuffle overhead (groupBy community_id): {shuffle_s:.1f}s")

            # Write — isolated per tag
            nodes_final.write.format('delta').mode('overwrite')\
                       .save(p_alg['p2_nodes'])
            edges_part.write.format('delta').mode('overwrite')\
                       .save(p_alg['p2_edges'])

            elapsed = time.time() - t0
            timing[('phase2', dataset, alg)] = elapsed
            results[key] = {
                'n_comms_raw':   n_raw,  'n_valid_comms': n_valid,
                'n_nodes_kept':  n_kept, 'n_boundary':    n_boundary,
                'n_internal':    n_kept - n_boundary,
                'n_intra_edges': n_intra,'pct_edges_kept': pct_kept,
                'shuffle_s':    shuffle_s,
            }
            print(f"  ✓ [{alg}] Phase 2 done in {elapsed:.1f}s")
            
            node_comm.unpersist()
            valid_comms.unpersist()
            comms_df.unpersist()
            spark._jvm.System.gc()

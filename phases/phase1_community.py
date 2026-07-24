import os
import sys
import time
from pipeline.utils.common import silence_all

def _dgl_metis_worker(src, dst, num_nodes, k, conn):
    try:
        import dgl
        import torch
        g_dgl = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=num_nodes)
        partition_ids = dgl.metis_partition_assignment(g_dgl, k).numpy()
        conn.send(partition_ids)
    except Exception as e:
        conn.send(e)

def run_phase1(spark, sc, datasets, algorithms, lpa_max_iter, resolution,
               random_seed, min_size, dataset_cfg, get_paths_fn, timing, results, **kwargs):
    """
    Run community detection for each (dataset, algorithm) independently.
    Each algorithm writes to its own Delta path: communities/{alg}/
    Results stored in results[(dataset, alg)] — isolated, never mixed.
    """
    from pyspark.sql import functions as F
    from sklearn.metrics import normalized_mutual_info_score

    for dataset in datasets:
        p    = get_paths_fn(dataset)
        print(f"\n{'='*60}\n  PHASE 1 — Community Detection: {dataset}\n{'='*60}")

        t_load    = time.time()
        nodes_df  = spark.read.format('delta').load(p['nodes']).select('id', 'label').cache()
        edges_df  = spark.read.format('delta').load(p['edges']).cache()
        n_nodes   = nodes_df.count()
        n_edges   = edges_df.count()
        print(f"  Graph loaded: {n_nodes:,} nodes | {n_edges:,} edges  [{time.time()-t_load:.1f}s]")

        for alg in algorithms:
            key   = (dataset, alg)
            p_alg = get_paths_fn(dataset, alg)
            
            # Check if checkpoint exists
            force_rerun = kwargs.get('force_rerun', False)
            if not force_rerun:
                try:
                    communities_df = spark.read.format('delta').load(p_alg['communities'])
                    n_comms = communities_df.select('community_id').distinct().count()
                    print(f"\n  ── [{alg.upper()}] (Loaded from S3 Checkpoint) ──")
                    print(f"    ✓ Loaded {n_comms:,} communities, skipping detection.")
                    results[key] = {'n_comms': n_comms, 'runtime_s': 0.0, 'nmi': 0.0}
                    timing[('phase1', dataset, alg)] = 0.0
                    continue
                except Exception:
                    pass

            print(f"\n  ── [{alg.upper()}] ──────────────────────────────")
            t_alg = time.time()

            # ── LPA ─────────────────────────────────────────────────────────
            if alg == 'lpa':
                sc.setCheckpointDir(p['checkpoints'] + 'lpa/')
                num_parts = 336 if n_nodes > 10_000_000 else 112
                if n_nodes > 10_000_000:
                    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
                else:
                    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", str(256 * 1024 * 1024))
                
                curr = (nodes_df.select('id')
                        .withColumn('community_id', F.col('id'))
                        .repartition(num_parts).cache())
                curr.count()
                for i in range(lpa_max_iter):
                    t_i = time.time()
                    curr_rename = curr.withColumnRenamed('id', 'src').withColumnRenamed('community_id', 'src_comm')
                    
                    if n_nodes > 10_000_000:
                        proposed = (edges_df
                                    .join(curr_rename, on='src', how='inner')
                                    .select(F.col('dst').alias('id'),
                                            F.col('src_comm').alias('proposed_comm')))
                    else:
                        proposed = (edges_df
                                    .join(F.broadcast(curr_rename), on='src', how='inner')
                                    .select(F.col('dst').alias('id'),
                                            F.col('src_comm').alias('proposed_comm')))
                                            
                    freq = proposed.groupBy('id', 'proposed_comm').count()
                    from pyspark.sql.window import Window
                    from pyspark.sql.functions import row_number
                    w = Window.partitionBy('id').orderBy(F.desc('count'), F.desc('proposed_comm'))
                    best = freq.withColumn('rn', row_number().over(w))\
                               .filter(F.col('rn') == 1)\
                               .select('id', F.col('proposed_comm').alias('new_comm'))
                    
                    old_curr = curr
                    curr = (curr.join(best, on='id', how='left')
                                .withColumn('community_id',
                                            F.coalesce(F.col('new_comm'),
                                                       F.col('community_id')))
                                .select('id', 'community_id')
                                .repartition(num_parts))
                    
                    if i % 2 == 1:
                        curr = curr.localCheckpoint()
                    else:
                        curr = curr.cache()
                        
                    nc = curr.select('community_id').distinct().count()
                    print(f"    Iter {i+1}/{lpa_max_iter}: {nc:,} comms  [{time.time()-t_i:.1f}s]")
                    try:
                        old_curr.unpersist()
                    except Exception:
                        pass
                communities_df = curr.cache()

            # ── Louvain / Leiden / igraph_lpa ─────────────────────────────────────────────
            elif alg in ('louvain', 'leiden', 'igraph_lpa'):
                import igraph as ig, numpy as np
                from pyspark.sql.types import StructType, StructField, LongType as _LT
                print(f"  Collecting edges to driver ...")
                edges_pd = (edges_df
                            .filter(F.col('src') < F.col('dst'))
                            .select(F.col('src').alias('u'), F.col('dst').alias('v'))
                            .toPandas())
                all_nids = np.union1d(edges_pd['u'].values, edges_pd['v'].values)
                print(f"  Mapping node IDs using vectorized binary search ...")
                u_mapped = np.searchsorted(all_nids, edges_pd['u'].values)
                v_mapped = np.searchsorted(all_nids, edges_pd['v'].values)
                el = zip(u_mapped, v_mapped)
                print(f"  Building igraph ({len(all_nids):,} nodes) ...")
                G = ig.Graph(n=len(all_nids), edges=el, directed=False)
                print(f"  Running {alg} (resolution={resolution}) ...")
                if alg == 'louvain':
                    partition = G.community_multilevel(resolution=resolution)
                elif alg == 'leiden':
                    import leidenalg
                    partition = leidenalg.find_partition(
                        G, leidenalg.RBConfigurationVertexPartition,
                        resolution_parameter=resolution, seed=random_seed)
                elif alg == 'igraph_lpa':
                    partition = G.community_label_propagation()
                id_to_comm = {}
                for ci, members in enumerate(partition):
                    for li in members:
                        id_to_comm[int(all_nids[li])] = ci
                comm_rows = list(id_to_comm.items())
                cschema   = StructType([StructField('id', _LT(), False),
                                        StructField('community_id', _LT(), False)])
                communities_df = None
                for s in range(0, len(comm_rows), 500_000):
                    chunk = spark.createDataFrame(comm_rows[s:s+500_000], schema=cschema)
                    communities_df = (chunk if communities_df is None
                                      else communities_df.union(chunk))
                communities_df = communities_df.cache()

            # ── METIS Baseline ─────────────────────────────────────────────
            elif alg == 'metis':
                import numpy as np
                from pyspark.sql.types import StructType, StructField, LongType as _LT
                k = kwargs.get('metis_k', 100)
                print(f"  Partitioning with METIS (k={k}) ...")
                
                print(f"  Collecting edges to driver ...")
                edges_pd = (edges_df
                            .filter(F.col('src') < F.col('dst'))
                            .select(F.col('src').alias('u'), F.col('dst').alias('v'))
                            .toPandas())
                all_nids = np.union1d(edges_pd['u'].values, edges_pd['v'].values)
                nmap     = {int(n): i for i, n in enumerate(all_nids)}
                
                partition_ids = None
                try:
                    import subprocess
                    import tempfile
                    import torch
                    import dgl
                    
                    print(f"  Mapping node IDs using vectorized binary search ...")
                    u_mapped = np.searchsorted(all_nids, edges_pd['u'].values)
                    v_mapped = np.searchsorted(all_nids, edges_pd['v'].values)
                    src_t = torch.tensor(u_mapped, dtype=torch.int64)
                    dst_t = torch.tensor(v_mapped, dtype=torch.int64)
                    
                    cat_src = torch.cat([src_t, dst_t]).numpy()
                    cat_dst = torch.cat([dst_t, src_t]).numpy()
                    num_nodes_val = len(all_nids)
                    
                    temp_dir = tempfile.gettempdir()
                    edges_npz = os.path.join(temp_dir, f"metis_edges_{os.getpid()}.npz")
                    partition_npy = os.path.join(temp_dir, f"metis_parts_{os.getpid()}.npy")
                    
                    np.savez(edges_npz, src=cat_src, dst=cat_dst, num_nodes=num_nodes_val)
                    
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    runner_path = os.path.join(script_dir, "metis_runner.py")
                    
                    res = subprocess.run(
                        [sys.executable, runner_path, edges_npz, str(k), partition_npy],
                        capture_output=True,
                        text=True
                    )
                    
                    if os.path.exists(edges_npz):
                        os.remove(edges_npz)
                        
                    if res.returncode != 0:
                        err_msg = res.stderr or res.stdout
                        raise RuntimeError(
                            f"METIS partition subprocess exited with code {res.returncode}. Error: {err_msg}"
                        )
                        
                    partition_ids = np.load(partition_npy)
                    if os.path.exists(partition_npy):
                        os.remove(partition_npy)
                        
                    print("    Successfully partitioned using DGL METIS.")
                except Exception as e:
                    print(f"    DGL METIS failed or not available ({e}). Trying pymetis...")
                    try:
                        import pymetis
                        adjacency_list = [[] for _ in range(len(all_nids))]
                        for u, v in zip(edges_pd['u'], edges_pd['v']):
                            idx_u, idx_v = nmap[int(u)], nmap[int(v)]
                            adjacency_list[idx_u].append(idx_v)
                            adjacency_list[idx_v].append(idx_u)
                        n_cuts, membership = pymetis.partition(k, adjacency_list)
                        partition_ids = np.array(membership)
                        print("    Successfully partitioned using pymetis.")
                    except Exception as e2:
                        print(f"    pymetis failed or not available ({e2}). Falling back to igraph multilevel...")
                        import igraph as ig
                        u_mapped = np.searchsorted(all_nids, edges_pd['u'].values)
                        v_mapped = np.searchsorted(all_nids, edges_pd['v'].values)
                        el = zip(u_mapped, v_mapped)
                        G = ig.Graph(n=len(all_nids), edges=el, directed=False)
                        partition = G.community_multilevel()
                        partition_ids = np.zeros(len(all_nids), dtype=np.int32)
                        for ci, members in enumerate(partition):
                            for li in members:
                                partition_ids[li] = ci % k
                        print("    Successfully partitioned using igraph fallback.")
                
                id_to_comm = {int(all_nids[i]): int(partition_ids[i]) for i in range(len(all_nids))}
                comm_rows = list(id_to_comm.items())
                cschema   = StructType([StructField('id', _LT(), False),
                                        StructField('community_id', _LT(), False)])
                communities_df = None
                for s in range(0, len(comm_rows), 500_000):
                    chunk = spark.createDataFrame(comm_rows[s:s+500_000], schema=cschema)
                    communities_df = (chunk if communities_df is None
                                      else communities_df.union(chunk))
                communities_df = communities_df.cache()
            else:
                raise ValueError(f"Unknown algorithm '{alg}'. "
                                 "Use 'lpa', 'louvain', 'leiden', 'igraph_lpa', or 'metis'.")

            communities_df.write.format('delta').mode('overwrite')\
                          .save(p_alg['communities'])

            runtime  = time.time() - t_alg
            n_comms  = communities_df.select('community_id').distinct().count()

            joined   = communities_df.join(nodes_df.select('id', 'label'),
                                           on='id', how='inner').toPandas()
            nmi      = normalized_mutual_info_score(
                           joined['label'].values, joined['community_id'].values)

            timing[('phase1', dataset, alg)] = runtime
            results[key] = {'n_comms': n_comms, 'runtime_s': runtime, 'nmi': nmi}
            print(f"  ✓ [{alg}] {n_comms:,} communities | NMI={nmi:.4f} | {runtime:.1f}s")
            
            communities_df.unpersist()
            spark._jvm.System.gc()

        nodes_df.unpersist()
        edges_df.unpersist()
        spark._jvm.System.gc()

def print_phase1_stats(spark, datasets, algorithms, min_size, get_paths_fn, results):
    """Print per-algorithm community detection summary table with detailed vertex assignment stats."""
    from pyspark.sql import functions as F
    print("\n" + "="*70)
    print("  PHASE 1 SUMMARY")
    print("="*70)
    for dataset in datasets:
        p = get_paths_fn(dataset)
        print(f"\n  Dataset: {dataset}")
        print(f"  {'Algorithm':<12} {'#Comms':>10} {'NMI':>8} {'Runtime':>10}")
        print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*10}")
        for alg in algorithms:
            key = (dataset, alg)
            r   = results.get(key)
            if r is None:
                print(f"  {alg:<12}  (not run)")
                continue
            print(f"  {alg:<12} {r['n_comms']:>10,} {r['nmi']:>8.4f} {r['runtime_s']:>9.1f}s")

        try:
            total_graph_nodes = spark.read.format('delta').load(p['nodes']).count()
        except Exception:
            total_graph_nodes = None

        for alg in algorithms:
            p_alg = get_paths_fn(dataset, alg)
            print(f"\n  [{alg}] Community assignment details:")
            try:
                cd    = spark.read.format('delta').load(p_alg['communities'])
                sizes = cd.groupBy('community_id').count()
                st    = sizes.agg(F.min('count').alias('mn'),
                                  F.max('count').alias('mx'),
                                  F.avg('count').alias('avg'),
                                  F.percentile_approx('count', 0.5).alias('med')
                                  ).collect()[0]
                nt    = sizes.count()
                nv    = sizes.filter(F.col('count') >= min_size).count()

                total_assigned = cd.count()
                unique_assigned = cd.select('id').distinct().count()
                if total_graph_nodes:
                    unassigned = total_graph_nodes - unique_assigned
                    coverage_pct = 100.0 * unique_assigned / total_graph_nodes
                    print(f"    Vertices assigned: {unique_assigned:,} / {total_graph_nodes:,} "
                          f"({coverage_pct:.1f}% coverage)")
                    if unassigned > 0:
                        print(f"    Unassigned vertices: {unassigned:,}")
                else:
                    print(f"    Vertices assigned: {unique_assigned:,}")

                print(f"    Total communities: {nt:,} | Valid (≥{min_size}): {nv:,}")
                print(f"    Size stats: min={st['mn']:,}  max={st['mx']:,}  "
                      f"avg={st['avg']:.1f}  median={st['med']:,}")

                buckets = [
                    ('1-9',     1,     9),
                    ('10-99',   10,    99),
                    ('100-999', 100,   999),
                    ('1K-9999', 1000,  9999),
                    ('10K+',    10000, None),
                ]
                bucket_strs = []
                for label, lo, hi in buckets:
                    if hi is not None:
                        cnt = sizes.filter((F.col('count') >= lo) & (F.col('count') <= hi)).count()
                    else:
                        cnt = sizes.filter(F.col('count') >= lo).count()
                    if cnt > 0:
                        bucket_strs.append(f"{label}:{cnt:,}")
                print(f"    Size buckets: {' | '.join(bucket_strs)}")

                top10 = sizes.orderBy(F.desc('count')).limit(10).collect()
                top10_str = ', '.join([f"c{r['community_id']}={r['count']:,}" for r in top10])
                print(f"    Top-10 largest: {top10_str}")

            except Exception as e:
                print(f"    Could not load: {e}")

import os
import sys
import time
from pipeline.utils.common import _patch_torch_load, _delta_exists, silence_all

def run_phase0(spark, sc, datasets, run_phase0_flag, use_ogb_splits,
               random_seed, dataset_cfg, get_paths_fn, timing, force_reingest=False):
    """Ingest each dataset from OGB or DGL into Delta Lake (nodes, edges, masks).
    
    Also saves original (directed, as-is) graph as separate Delta tables:
      - original_nodes: [id, label, features]  (raw node data before any processing)
      - original_edges: [src, dst]             (directed edges exactly as from OGB/DGL)
    
    If processed Delta tables already exist for a dataset, the download is skipped.
    """
    _patch_torch_load()
    if not run_phase0_flag:
        print("[Phase 0 SKIPPED]  RUN_PHASE0 = False")
        print("  → Set RUN_PHASE0 = True in experiment_config.py when using a new dataset.")
        return

    import builtins
    import numpy as np
    from unittest.mock import patch
    from pyspark.sql.types import (StructType, StructField,
                                   LongType, IntegerType, ArrayType,
                                   FloatType, StringType)

    for dataset in datasets:
        p       = get_paths_fn(dataset)
        t_total = time.time()
        print(f"\n{'='*60}\n  PHASE 0 — Ingesting: {dataset}\n{'='*60}")

        # Skip if Delta tables already exist (unless force_reingest is enabled)
        if not force_reingest and (_delta_exists(spark, p['nodes']) and _delta_exists(spark, p['edges']) and
            _delta_exists(spark, p['original_nodes']) and _delta_exists(spark, p['original_edges']) and
            _delta_exists(spark, p['masks'])):
            print(f"  ✓ Delta tables already exist for '{dataset}' — skipping download.")
            print(f"    nodes: {p['nodes']}")
            print(f"    edges: {p['edges']}")
            print(f"    original_nodes: {p['original_nodes']}")
            print(f"    original_edges: {p['original_edges']}")
            elapsed = time.time() - t_total
            timing[('phase0', dataset)] = elapsed
            continue

        if dataset in ('reddit', 'flickr', 'wikics', 'coauthor-cs', 'coauthor-physics', 'deezereurope'):
            if dataset == 'reddit':
                from dgl.data import RedditDataset
                ds = RedditDataset(self_loop=False)
                g = ds[0]
                n_nodes = g.num_nodes()
                node_feat = np.asarray(g.ndata['feat'].numpy(), dtype=np.float32)
                src_r, dst_r = g.edges()
                src_r = np.asarray(src_r.numpy(), dtype=np.int64)
                dst_r = np.asarray(dst_r.numpy(), dtype=np.int64)
                lbl = np.asarray(g.ndata['label'].numpy(), dtype=np.int32).flatten()
                train_idx = np.where(g.ndata['train_mask'].numpy())[0]
                valid_idx = np.where(g.ndata['val_mask'].numpy())[0]
                test_idx = np.where(g.ndata['test_mask'].numpy())[0]
            elif dataset == 'flickr':
                from dgl.data import FlickrDataset
                ds = FlickrDataset()
                g = ds[0]
                n_nodes = g.num_nodes()
                feat = g.ndata['feat']
                if hasattr(feat, 'toarray'):
                    feat = feat.toarray()
                elif hasattr(feat, 'numpy'):
                    feat = feat.numpy()
                node_feat = np.asarray(feat, dtype=np.float32)
                src_r, dst_r = g.edges()
                src_r = np.asarray(src_r.numpy(), dtype=np.int64)
                dst_r = np.asarray(dst_r.numpy(), dtype=np.int64)
                lbl = np.asarray(g.ndata['label'].numpy(), dtype=np.int32).flatten()
                train_idx = np.where(g.ndata['train_mask'].numpy())[0]
                valid_idx = np.where(g.ndata['val_mask'].numpy())[0]
                test_idx = np.where(g.ndata['test_mask'].numpy())[0]
            elif dataset == 'wikics':
                from dgl.data import WikiCSDataset
                ds = WikiCSDataset()
                g = ds[0]
                n_nodes = g.num_nodes()
                node_feat = np.asarray(g.ndata['feat'].numpy(), dtype=np.float32)
                src_r, dst_r = g.edges()
                src_r = np.asarray(src_r.numpy(), dtype=np.int64)
                dst_r = np.asarray(dst_r.numpy(), dtype=np.int64)
                lbl = np.asarray(g.ndata['label'].numpy(), dtype=np.int32).flatten()
                train_idx = np.where(g.ndata['train_mask'][:, 0].numpy())[0]
                valid_idx = np.where(g.ndata['val_mask'][:, 0].numpy())[0]
                test_idx = np.where(g.ndata['test_mask'].numpy())[0]
            elif dataset == 'coauthor-cs':
                from dgl.data import CoauthorCSDataset
                ds = CoauthorCSDataset()
                g = ds[0]
                n_nodes = g.num_nodes()
                node_feat = np.asarray(g.ndata['feat'].numpy(), dtype=np.float32)
                src_r, dst_r = g.edges()
                src_r = np.asarray(src_r.numpy(), dtype=np.int64)
                dst_r = np.asarray(dst_r.numpy(), dtype=np.int64)
                lbl = np.asarray(g.ndata['label'].numpy(), dtype=np.int32).flatten()
                rng = np.random.default_rng(random_seed)
                perm = rng.permutation(n_nodes)
                n_tr, n_va = int(.6 * n_nodes), int(.2 * n_nodes)
                train_idx = perm[:n_tr]
                valid_idx = perm[n_tr:n_tr+n_va]
                test_idx = perm[n_tr+n_va:]
            elif dataset == 'coauthor-physics':
                from dgl.data import CoauthorPhysicsDataset
                ds = CoauthorPhysicsDataset()
                g = ds[0]
                n_nodes = g.num_nodes()
                node_feat = np.asarray(g.ndata['feat'].numpy(), dtype=np.float32)
                src_r, dst_r = g.edges()
                src_r = np.asarray(src_r.numpy(), dtype=np.int64)
                dst_r = np.asarray(dst_r.numpy(), dtype=np.int64)
                lbl = np.asarray(g.ndata['label'].numpy(), dtype=np.int32).flatten()
                rng = np.random.default_rng(random_seed)
                perm = rng.permutation(n_nodes)
                n_tr, n_va = int(.6 * n_nodes), int(.2 * n_nodes)
                train_idx = perm[:n_tr]
                valid_idx = perm[n_tr:n_tr+n_va]
                test_idx = perm[n_tr+n_va:]
            elif dataset == 'deezereurope':
                import zipfile, json, collections, ssl, urllib.request
                import pandas as pd
                zip_path = 'deezer_europe.zip'
                if not os.path.exists(zip_path):
                    ssl._create_default_https_context = ssl._create_unverified_context
                    print("  Downloading DeezerEurope from SNAP...")
                    urllib.request.urlretrieve('https://snap.stanford.edu/data/deezer_europe.zip', zip_path)
                extract_dir = 'deezer_europe_extracted'
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(extract_dir)
                edges_df = pd.read_csv(os.path.join(extract_dir, 'deezer_europe', 'deezer_europe_edges.csv'))
                src_r = edges_df['node_1'].values.astype(np.int64)
                dst_r = edges_df['node_2'].values.astype(np.int64)
                target_df = pd.read_csv(os.path.join(extract_dir, 'deezer_europe', 'deezer_europe_target.csv'))
                lbl = np.zeros(len(target_df), dtype=np.int32)
                lbl[target_df['id'].values.astype(np.int64)] = target_df['target'].values.astype(np.int32)
                with open(os.path.join(extract_dir, 'deezer_europe', 'deezer_europe_features.json'), 'r') as f:
                    features_dict = json.load(f)
                n_nodes = len(features_dict)
                all_artists = []
                for v in features_dict.values():
                    all_artists.extend(v)
                counter = collections.Counter(all_artists)
                top_128 = [artist for artist, count in counter.most_common(128)]
                node_feat = np.zeros((n_nodes, 128), dtype=np.float32)
                for node_id_str, liked_artists in features_dict.items():
                    node_id = int(node_id_str)
                    node_id_liked = set(liked_artists)
                    node_feat[node_id] = [1.0 if artist in node_id_liked else 0.0 for artist in top_128]
                rng = np.random.default_rng(random_seed)
                perm = rng.permutation(n_nodes)
                n_tr, n_va = int(.6 * n_nodes), int(.2 * n_nodes)
                train_idx = perm[:n_tr]
                valid_idx = perm[n_tr:n_tr+n_va]
                test_idx = perm[n_tr+n_va:]
        else:
            # Download OGB
            from ogb.nodeproppred import NodePropPredDataset
            with patch.object(builtins, 'input', lambda _: 'y'):
                with silence_all():
                    ogb_ds = NodePropPredDataset(name=dataset, root=os.path.join(os.environ.get('TMPDIR', '/tmp'), 'ogb_data'))
            graph, labels = ogb_ds[0]

            if dataset == 'ogbn-mag':
                n_nodes     = graph['num_nodes_dict']['paper']
                node_feat   = np.asarray(graph['node_feat_dict']['paper'], dtype=np.float32)
                ei          = graph['edge_index_dict'][('paper', 'cites', 'paper')]
                src_r, dst_r = np.asarray(ei[0], np.int64), np.asarray(ei[1], np.int64)
                lbl         = np.asarray(labels['paper'], np.int32).flatten()
            else:
                n_nodes     = graph['num_nodes']
                if graph.get('node_feat') is not None:
                    node_feat   = np.asarray(graph['node_feat'], dtype=np.float32)
                else:
                    print(f"  [Warning] No node_feat found for {dataset}, generating dummy features.")
                    node_feat   = np.ones((n_nodes, dataset_cfg[dataset]['in_feats']), dtype=np.float32)
                src_r = np.asarray(graph['edge_index'][0], np.int64)
                dst_r = np.asarray(graph['edge_index'][1], np.int64)
                lbl   = np.asarray(labels, np.int32).flatten()

        # Save ORIGINAL (raw) nodes and edges as Delta tables
        import pandas as pd
        ns = StructType([StructField('id', LongType(), False),
                         StructField('label', IntegerType(), True),
                         StructField('features', ArrayType(FloatType()), True)])
        es_orig = StructType([StructField('src', LongType(), False),
                              StructField('dst', LongType(), False)])

        print(f"  Saving original (directed) graph: {n_nodes:,} nodes | {len(src_r):,} edges ...")

        # Write original nodes in chunks to prevent JVM OOM for high-dimensional datasets
        feat_dim = node_feat.shape[1] if len(node_feat.shape) > 1 else 1
        CHUNK = max(100, min(50000, 1000000 // feat_dim))
        print(f"  Feature dimension: {feat_dim} | Ingesting in chunks of {CHUNK:,} nodes...")
        for s in range(0, n_nodes, CHUNK):
            e    = min(s + CHUNK, n_nodes)
            pdf_nodes = pd.DataFrame({
                'id':       np.arange(s, e, dtype=np.int64),
                'label':    lbl[s:e].astype(np.int32),
                'features': [x.tolist() for x in node_feat[s:e]]
            })
            df   = spark.createDataFrame(pdf_nodes, schema=ns)
            df.coalesce(1).write.format('delta').mode('overwrite' if s == 0 else 'append').save(p['original_nodes'])
        print(f"  Original nodes written → {p['original_nodes']}")

        # Write original (directed) edges
        ECHUNK = 5_000_000
        for s in range(0, len(src_r), ECHUNK):
            e    = min(s + ECHUNK, len(src_r))
            pdf_edges = pd.DataFrame({
                'src': src_r[s:e].astype(np.int64),
                'dst': dst_r[s:e].astype(np.int64)
            })
            df   = spark.createDataFrame(pdf_edges, schema=es_orig)
            df.coalesce(1).write.format('delta').mode('overwrite' if s == 0 else 'append').save(p['original_edges'])
        print(f"  Original edges written → {p['original_edges']}")

        # Undirected deduped edges
        all_s = np.concatenate([src_r, dst_r])
        all_d = np.concatenate([dst_r, src_r])
        m = all_s != all_d
        all_s, all_d = all_s[m], all_d[m]
        lo, hi = np.minimum(all_s, all_d), np.maximum(all_s, all_d)
        canon  = np.unique(np.stack([lo, hi], axis=1), axis=0)
        src_f  = np.concatenate([canon[:, 0], canon[:, 1]])
        dst_f  = np.concatenate([canon[:, 1], canon[:, 0]])
        print(f"  Nodes: {n_nodes:,} | Edges (both dirs): {len(src_f):,}")

        from pipeline.utils.graph_validator import validate_graph_properties
        validate_graph_properties(n_nodes, src_f, dst_f)

        # Write processed nodes
        es = StructType([StructField('src', LongType(), False),
                         StructField('dst', LongType(), False)])
        for s in range(0, n_nodes, CHUNK):
            e    = min(s + CHUNK, n_nodes)
            pdf_nodes = pd.DataFrame({
                'id':       np.arange(s, e, dtype=np.int64),
                'label':    lbl[s:e].astype(np.int32),
                'features': [x.tolist() for x in node_feat[s:e]]
            })
            df   = spark.createDataFrame(pdf_nodes, schema=ns)
            df.coalesce(1).write.format('delta').mode('overwrite' if s == 0 else 'append').save(p['nodes'])
        print(f"  Nodes written.")

        # Write processed edges
        for s in range(0, len(src_f), ECHUNK):
            e    = min(s + ECHUNK, len(src_f))
            pdf_edges = pd.DataFrame({
                'src': src_f[s:e].astype(np.int64),
                'dst': dst_f[s:e].astype(np.int64)
            })
            df   = spark.createDataFrame(pdf_edges, schema=es)
            df.coalesce(1).write.format('delta').mode('overwrite' if s == 0 else 'append').save(p['edges'])
        print(f"  Edges written.")

        # Write masks
        ms = StructType([StructField('id', LongType(), False),
                         StructField('split', StringType(), True)])
        if use_ogb_splits:
            if dataset in ('reddit', 'flickr', 'wikics', 'coauthor-cs', 'coauthor-physics', 'deezereurope'):
                ids = np.concatenate([train_idx.flatten(), valid_idx.flatten(), test_idx.flatten()])
                splits = (['train'] * len(train_idx.flatten()) +
                          ['valid'] * len(valid_idx.flatten()) +
                          ['test']  * len(test_idx.flatten()))
                pdf_masks = pd.DataFrame({'id': ids.astype(np.int64), 'split': splits})
            else:
                idx   = ogb_ds.get_idx_split()
                if dataset == 'ogbn-mag':
                    train_idx = idx['train']['paper']
                    valid_idx = idx['valid']['paper']
                    test_idx  = idx['test']['paper']
                else:
                    train_idx = idx['train']
                    valid_idx = idx['valid']
                    test_idx  = idx['test']
                
                ids = np.concatenate([train_idx.flatten(), valid_idx.flatten(), test_idx.flatten()])
                splits = (['train'] * len(train_idx.flatten()) +
                          ['valid'] * len(valid_idx.flatten()) +
                          ['test']  * len(test_idx.flatten()))
                pdf_masks = pd.DataFrame({'id': ids.astype(np.int64), 'split': splits})
        else:
            rng   = np.random.default_rng(random_seed)
            perm  = rng.permutation(n_nodes)
            n_tr, n_va = int(.6 * n_nodes), int(.2 * n_nodes)
            
            ids = perm
            splits = (['train'] * n_tr +
                      ['valid'] * n_va +
                      ['test']  * (n_nodes - n_tr - n_va))
            pdf_masks = pd.DataFrame({'id': ids.astype(np.int64), 'split': splits})
            
        spark.createDataFrame(pdf_masks, schema=ms)\
             .coalesce(1).write.format('delta').mode('overwrite').save(p['masks'])
        print(f"  Masks written.")

        # Compact the Delta tables
        print(f"  Compacting Delta tables (running OPTIMIZE & VACUUM) for {dataset}...")
        spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
        for table_key in ['original_nodes', 'original_edges', 'nodes', 'edges', 'masks']:
            table_path = p[table_key]
            try:
                t_opt = time.time()
                spark.sql(f"OPTIMIZE '{table_path}'")
                spark.sql(f"VACUUM '{table_path}' RETAIN 0 HOURS")
                print(f"    ✓ Compacted & Cleaned: {table_path} ({time.time() - t_opt:.1f}s)")
            except Exception as e:
                print(f"    ⚠ Warning: Could not optimize/vacuum {table_path}: {e}")

        # Clean up local raw downloaded files
        import shutil
        if dataset in ('reddit', 'flickr', 'wikics', 'coauthor-cs', 'coauthor-physics', 'deezereurope'):
            dgl_dir = os.environ.get('DGL_DOWNLOAD_DIR', '/tmp/.dgl')
            ds_dir = os.path.join(dgl_dir, dataset)
            if os.path.exists(ds_dir):
                print(f"  Cleaning up local DGL cache for {dataset} at {ds_dir}...")
                shutil.rmtree(ds_dir, ignore_errors=True)
            if dataset == 'deezereurope' and os.path.exists('deezer_europe_extracted'):
                print("  Cleaning up extracted DeezerEurope folder...")
                shutil.rmtree('deezer_europe_extracted', ignore_errors=True)
        else:
            normalized = dataset.replace('-', '_')
            if dataset.startswith('ogbn-'):
                normalized = dataset[5:].replace('-', '_')
            
            ogb_dir = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'ogb_data')
            for folder in [normalized, dataset]:
                ds_dir = os.path.join(ogb_dir, folder)
                if os.path.exists(ds_dir):
                    print(f"  Cleaning up local OGB cache for {dataset} at {ds_dir}...")
                    shutil.rmtree(ds_dir, ignore_errors=True)
            
            if os.path.exists('/tmp/ogb_data'):
                for folder in [normalized, dataset]:
                    ds_dir = os.path.join('/tmp/ogb_data', folder)
                    if os.path.exists(ds_dir):
                        print(f"  Cleaning up local fallback OGB cache for {dataset} at {ds_dir}...")
                        shutil.rmtree(ds_dir, ignore_errors=True)

        elapsed = time.time() - t_total
        timing[('phase0', dataset)] = elapsed
        print(f"  ✓ {dataset} ingested in {elapsed:.1f}s")

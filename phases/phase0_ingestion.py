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

        if dataset == 'ogbn-papers100M':
            import zipfile, glob, urllib.request, shutil
            import pandas as pd
            import builtins
            from unittest.mock import patch
            from pyspark.sql.types import (StructType, StructField, LongType, IntegerType, ArrayType, FloatType, StringType)
            
            print(f"\n{'='*60}\n  PHASE 0 — Zero-RAM Direct Ingestion: {dataset}\n{'='*60}")
            
            # Discover all candidate writable mounts and search paths
            candidate_bases = [
                '/mnt/tmp', '/mnt/spark', '/mnt/var/tmp',
                '/tmp', '/var/tmp',
                '/mnt1/tmp', '/mnt1/spark', '/mnt1/var/tmp',
                '/mnt2/tmp', '/mnt2/spark', '/mnt2/var/tmp',
                os.environ.get('TMPDIR', '/tmp'),
                os.environ.get('TEMP', '/tmp'),
                os.environ.get('TMP', '/tmp'),
                os.path.expanduser('~')
            ]
            candidate_bases = list(dict.fromkeys([b for b in candidate_bases if b and os.path.exists(b)]))
            
            search_paths = []
            for b in candidate_bases:
                search_paths.extend([
                    b,
                    os.path.join(b, 'ogb_data'),
                    os.path.join(b, 'ogbn_papers100M'),
                    os.path.join(b, 'papers100M'),
                    os.path.join(b, 'papers100M-bin'),
                    os.path.join(b, '.dgl')
                ])
            search_paths = list(dict.fromkeys(search_paths))
            print(f"  ► Scanning candidate storage volumes across {len(candidate_bases)} mounts for existing raw dataset files...")

            def locate_raw_files(paths):
                f_file, l_file, r_dir = None, None, None
                for search_base in paths:
                    if not os.path.exists(search_base):
                        continue
                    for root_path, dirs, files in os.walk(search_base):
                        for f in files:
                            if f in ['data.npz', 'node-feat.npy', 'node_feat.npy', 'node-feat.bin', 'node_feat.bin']:
                                f_file = os.path.join(root_path, f)
                                r_dir = root_path
                            elif f in ['node-label.npz', 'node_label.npz', 'node-label.npy', 'node_label.npy', 'node-label.bin', 'node_label.bin']:
                                l_file = os.path.join(root_path, f)
                        if f_file is not None:
                            break
                    if f_file is not None:
                        break
                return f_file, l_file, r_dir

            def locate_zip_archive(paths):
                for search_base in paths:
                    if not os.path.exists(search_base):
                        continue
                    for root_path, dirs, files in os.walk(search_base):
                        for f in files:
                            if f in ['papers100M-bin.zip', 'ogbn_papers100M.zip', 'papers100M.zip']:
                                return os.path.join(root_path, f)
                return None

            def purge_zip_archives(bases):
                freed_b = 0
                zip_names = ['papers100M-bin.zip', 'ogbn_papers100M.zip', 'papers100M.zip', 'raw.zip', 'deezer_europe.zip', 'reddit.zip']
                for b in bases:
                    if not os.path.exists(b):
                        continue
                    for root_path, dirs, files in os.walk(b):
                        for f in files:
                            if f in zip_names or (f.endswith('.zip') and ('papers' in f.lower() or 'ogb' in f.lower())):
                                zf = os.path.join(root_path, f)
                                try:
                                    sz = os.path.getsize(zf)
                                    os.remove(zf)
                                    freed_b += sz
                                    print(f"  ✓ Purged redundant zip archive: {zf} (freed {sz / (1024**3):.2f} GB)")
                                except Exception:
                                    pass
                return freed_b

            feat_file, label_file, raw_dir = locate_raw_files(search_paths)

            if feat_file is None:
                zip_path = locate_zip_archive(search_paths)
                
                # Pick volume with maximum free disk space
                best_base = candidate_bases[0]
                best_free = 0
                for b in candidate_bases:
                    try:
                        free_b = shutil.disk_usage(b).free
                        if free_b > best_free:
                            best_free = free_b
                            best_base = b
                    except Exception:
                        pass
                
                ogb_root = os.path.join(best_base, 'ogb_data')
                os.makedirs(ogb_root, exist_ok=True)

                if zip_path is None or not os.path.exists(zip_path):
                    url = "http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip"
                    zip_path = os.path.join(ogb_root, "papers100M-bin.zip")
                    print(f"  ► Downloading ogbn-papers100M from {url} to {zip_path} ({best_free / (1024**3):.2f} GB free)...")
                    from ogb.utils.url import download_url
                    download_url(url, ogb_root)

                print(f"  ► Extracting raw binary archive {zip_path} into {ogb_root} ...")
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(ogb_root)
                
                # Immediately purge compressed zip to free ~56.2 GB disk space
                purge_zip_archives(candidate_bases)

                feat_file, label_file, raw_dir = locate_raw_files(search_paths)

            if feat_file is None:
                raise FileNotFoundError(
                    f"Could not find data.npz or node-feat.npy in any scanned candidate directories: {candidate_bases}. "
                    f"Please verify that 'ogbn-papers100M' raw files are extracted into one of these candidate volumes."
                )

            print(f"  ✓ Located node features file at: {feat_file}")
            
            # Clean up extracted zip archives immediately to ensure maximum free disk space
            purge_zip_archives(candidate_bases)
            
            # Memory-mapped data loading (0 MB driver RAM)
            edge_index_arr = None
            if feat_file.endswith('.npz'):
                print(f"  ► Loading memory-mapped numpy data archive: {feat_file}")
                data_archive = np.load(feat_file, mmap_mode='r')
                if 'node_feat' in data_archive:
                    node_feat = data_archive['node_feat']
                elif 'node-feat' in data_archive:
                    node_feat = data_archive['node-feat']
                else:
                    node_feat = data_archive[data_archive.files[0]]

                if 'edge_index' in data_archive:
                    edge_index_arr = data_archive['edge_index']
                elif 'edge-index' in data_archive:
                    edge_index_arr = data_archive['edge-index']
            else:
                node_feat = np.load(feat_file, mmap_mode='r')

            if label_file and os.path.exists(label_file):
                print(f"  ► Loading memory-mapped labels from: {label_file}")
                if label_file.endswith('.npz'):
                    lbl_archive = np.load(label_file, mmap_mode='r')
                    if 'node_label' in lbl_archive:
                        lbl = lbl_archive['node_label']
                    elif 'node-label' in lbl_archive:
                        lbl = lbl_archive['node-label']
                    else:
                        lbl = lbl_archive[lbl_archive.files[0]]
                else:
                    lbl = np.load(label_file, mmap_mode='r')
                lbl = np.asarray(lbl).flatten()
            else:
                lbl = np.zeros(node_feat.shape[0], dtype=np.int32)

            n_nodes = node_feat.shape[0]
            feat_dim = node_feat.shape[1] if len(node_feat.shape) > 1 else 1
            print(f"  Nodes: {n_nodes:,} | Feature dimension: {feat_dim}")

            
            # 1. Stream Nodes to Delta Lake
            ns = StructType([StructField('id', LongType(), False),
                             StructField('label', IntegerType(), True),
                             StructField('features', ArrayType(FloatType()), True)])
            CHUNK = 200000
            print(f"  Streaming nodes to S3 Delta Lake in chunks of {CHUNK:,} nodes...")
            for s in range(0, n_nodes, CHUNK):
                e = min(s + CHUNK, n_nodes)
                lbl_chunk = np.nan_to_num(lbl[s:e], nan=-1).astype(np.int32)
                pdf_nodes = pd.DataFrame({
                    'id': np.arange(s, e, dtype=np.int64),
                    'label': lbl_chunk,
                    'features': node_feat[s:e].tolist()
                })
                df_node = spark.createDataFrame(pdf_nodes, schema=ns)
                df_node.write.format('delta').mode('overwrite' if s == 0 else 'append').save(p['original_nodes'])
            print(f"  ✓ Original nodes written to Delta Lake.")

            print(f"  Writing processed nodes Delta table via Spark distributed copy...")
            spark.read.format('delta').load(p['original_nodes']).write.format('delta').mode('overwrite').save(p['nodes'])
            print(f"  ✓ Processed nodes written to Delta Lake.")
            
            # 2. Stream Original Edges
            es_orig = StructType([StructField('src', LongType(), False),
                                  StructField('dst', LongType(), False)])
            
            # Resilient edge file search
            edge_csv_gz = os.path.join(raw_dir, 'edge.csv.gz')
            edge_npy = os.path.join(raw_dir, 'edge_index.npy')
            if edge_index_arr is None and not os.path.exists(edge_csv_gz) and not os.path.exists(edge_npy):
                for r_path, _, files in os.walk(raw_dir):
                    for f in files:
                        if f in ['edge_index.npy', 'edge_idx.npy', 'edge-index.npy', 'edge-idx.npy']:
                            edge_npy = os.path.join(r_path, f)
                            break
                        elif f in ['edge.csv.gz', 'edges.csv.gz']:
                            edge_csv_gz = os.path.join(r_path, f)
                            break
                    if os.path.exists(edge_npy) or os.path.exists(edge_csv_gz):
                        break
            
            print(f"  Streaming original edges to S3 Delta Lake...")
            if edge_index_arr is not None:
                src_r, dst_r = edge_index_arr[0], edge_index_arr[1]
                ECHUNK = 2_500_000
                for s in range(0, len(src_r), ECHUNK):
                    e = min(s + ECHUNK, len(src_r))
                    pdf_e = pd.DataFrame({'src': src_r[s:e].astype(np.int64), 'dst': dst_r[s:e].astype(np.int64)})
                    df_e = spark.createDataFrame(pdf_e, schema=es_orig)
                    df_e.write.format('delta').mode('overwrite' if s == 0 else 'append').save(p['original_edges'])
            elif os.path.exists(edge_npy):
                ei = np.load(edge_npy, mmap_mode='r')
                src_r, dst_r = ei[0], ei[1]
                ECHUNK = 2_500_000
                for s in range(0, len(src_r), ECHUNK):
                    e = min(s + ECHUNK, len(src_r))
                    pdf_e = pd.DataFrame({'src': src_r[s:e].astype(np.int64), 'dst': dst_r[s:e].astype(np.int64)})
                    df_e = spark.createDataFrame(pdf_e, schema=es_orig)
                    df_e.write.format('delta').mode('overwrite' if s == 0 else 'append').save(p['original_edges'])
            elif os.path.exists(edge_csv_gz):
                for i, chunk_df in enumerate(pd.read_csv(edge_csv_gz, compression='gzip', header=None, names=['src', 'dst'], chunksize=2_500_000)):
                    df_e = spark.createDataFrame(chunk_df.astype(np.int64), schema=es_orig)
                    df_e.write.format('delta').mode('overwrite' if i == 0 else 'append').save(p['original_edges'])
            else:
                raise FileNotFoundError(f"Could not find original edges file (edge_index array in data.npz, edge.csv.gz, or edge_index.npy) under {raw_dir}")
            print(f"  ✓ Original edges written to Delta Lake.")

            # 3. Distributed Deduplicating & Symmetrizing Edges in PySpark (Zero Driver RAM)
            print("  Symmetrizing and deduplicating 1.6B edges in PySpark across YARN workers...")
            orig_df = spark.read.format('delta').load(p['original_edges'])
            rev_df = orig_df.selectExpr("dst as src", "src as dst")
            sym_df = orig_df.union(rev_df).filter("src != dst").dropDuplicates(["src", "dst"])
            sym_df.write.format('delta').mode('overwrite').save(p['edges'])
            print(f"  ✓ Undirected edges written to Delta Lake.")

            # 4. Stream Masks
            ms = StructType([StructField('id', LongType(), False),
                             StructField('split', StringType(), True)])
            
            # Resilient split discovery
            train_path = os.path.join(raw_dir, 'split', 'paper-split-structure', 'time', 'train.csv.gz')
            valid_path = os.path.join(raw_dir, 'split', 'paper-split-structure', 'time', 'valid.csv.gz')
            test_path  = os.path.join(raw_dir, 'split', 'paper-split-structure', 'time', 'test.csv.gz')
            
            if not os.path.exists(train_path):
                for r_path, _, files in os.walk(raw_dir):
                    if 'train.csv.gz' in files:
                        train_path = os.path.join(r_path, 'train.csv.gz')
                        valid_path = os.path.join(r_path, 'valid.csv.gz')
                        test_path  = os.path.join(r_path, 'test.csv.gz')
                        break

            if use_ogb_splits and os.path.exists(train_path):
                train_df = pd.read_csv(train_path, header=None, names=['id'])
                valid_df = pd.read_csv(valid_path, header=None, names=['id'])
                test_df  = pd.read_csv(test_path, header=None, names=['id'])
                train_df['split'] = 'train'
                valid_df['split'] = 'valid'
                test_df['split']  = 'test'
                mask_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
            else:
                rng = np.random.default_rng(random_seed)
                perm = rng.permutation(n_nodes)
                n_tr, n_va = int(.6 * n_nodes), int(.2 * n_nodes)
                mask_df = pd.DataFrame({
                    'id': perm.astype(np.int64),
                    'split': (['train'] * n_tr + ['valid'] * n_va + ['test'] * (n_nodes - n_tr - n_va))
                })
            spark.createDataFrame(mask_df, schema=ms).coalesce(1).write.format('delta').mode('overwrite').save(p['masks'])
            print(f"  ✓ Masks written to Delta Lake.")

            # Compact & Vacuum
            print(f"  Compacting Delta tables for {dataset}...")
            spark.conf.set("spark.databricks.delta.retentionDurationCheck.enabled", "false")
            for table_key in ['original_nodes', 'original_edges', 'nodes', 'edges', 'masks']:
                try:
                    spark.sql(f"OPTIMIZE '{p[table_key]}'")
                    spark.sql(f"VACUUM '{p[table_key]}' RETAIN 0 HOURS")
                except Exception as ex:
                    print(f"    ⚠ Warning vacuuming {p[table_key]}: {ex}")
            
            elapsed = time.time() - t_total
            timing[('phase0', dataset)] = elapsed
            print(f"  ✓ {dataset} ingested successfully in {elapsed:.1f}s!")
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
        else:
            # Download OGB
            from ogb.nodeproppred import NodePropPredDataset
            print(f"  ► Downloading & loading OGB dataset '{dataset}' (progress shown below)...")
            ogb_root = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'ogb_data')
            with patch.object(builtins, 'input', lambda _: 'y'):
                ogb_ds = NodePropPredDataset(name=dataset, root=ogb_root)


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

        # Clean up local raw downloaded files across ALL candidate storage volumes
        import shutil
        print(f"  Sweeping and purging raw dataset caches for '{dataset}' across all candidate volumes...")
        clean_candidates = [
            '/mnt/tmp', '/mnt/spark', '/mnt/var/tmp',
            '/tmp', '/var/tmp',
            '/mnt1/tmp', '/mnt1/spark', '/mnt1/var/tmp',
            '/mnt2/tmp', '/mnt2/spark', '/mnt2/var/tmp',
            os.environ.get('TMPDIR', '/tmp'),
            os.environ.get('DGL_DOWNLOAD_DIR', '/tmp/.dgl'),
            os.path.expanduser('~')
        ]
        clean_candidates = list(dict.fromkeys([c for c in clean_candidates if c and os.path.exists(c)]))
        
        normalized = dataset.replace('-', '_')
        folder_names = [dataset, normalized]
        if dataset.startswith('ogbn-'):
            folder_names.append(dataset[5:].replace('-', '_'))

        for base_c in clean_candidates:
            # Purge DGL caches
            dgl_ds = os.path.join(base_c, '.dgl', dataset)
            if os.path.exists(dgl_ds):
                shutil.rmtree(dgl_ds, ignore_errors=True)
                print(f"    ✓ Purged DGL cache at: {dgl_ds}")
            
            # Purge OGB caches & raw subfolders
            for folder in folder_names:
                for sub in ['', 'ogb_data', 'raw']:
                    target_dir = os.path.join(base_c, sub, folder) if sub else os.path.join(base_c, folder)
                    if os.path.exists(target_dir):
                        shutil.rmtree(target_dir, ignore_errors=True)
                        print(f"    ✓ Purged raw cache at: {target_dir}")

            # Purge extracted Deezer Europe directory if present
            de_dir = os.path.join(base_c, 'deezer_europe_extracted')
            if os.path.exists(de_dir):
                shutil.rmtree(de_dir, ignore_errors=True)
                print(f"    ✓ Purged Deezer Europe cache at: {de_dir}")

        # Purge remaining dataset zip files in candidate volumes
        zip_names = ['papers100M-bin.zip', 'ogbn_papers100M.zip', 'papers100M.zip', 'raw.zip', 'deezer_europe.zip', 'reddit.zip']
        for base_c in clean_candidates:
            for root_path, dirs, files in os.walk(base_c):
                for f in files:
                    if f in zip_names or (f.endswith('.zip') and ('papers' in f.lower() or 'ogb' in f.lower())):
                        zf = os.path.join(root_path, f)
                        try:
                            os.remove(zf)
                            print(f"    ✓ Purged leftover zip archive: {zf}")
                        except Exception:
                            pass

        elapsed = time.time() - t_total
        timing[('phase0', dataset)] = elapsed
        print(f"  ✓ {dataset} ingested in {elapsed:.1f}s")

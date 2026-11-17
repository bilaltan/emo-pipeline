import copy
import time
import resource
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

from pipeline.utils.common import _patch_torch_load

def run_phase4(spark, sc, datasets, dataset_cfg, baseline_cfg, get_paths_fn,
               timing, results, **kwargs):
    """
    Train one GraphSAGE on the full graph per dataset (not per algorithm).
    Uses the SAME masks as Phase 3 — ensuring fair comparison.
    """
    _patch_torch_load()
    from torch_geometric.nn import SAGEConv
    from torch_geometric.utils import coalesce, add_remaining_self_loops, negative_sampling
    import torch_geometric.transforms as T
    from torch_geometric.data import Data
    try:
        import dgl
        import dgl.nn as dglnn
        from dgl.dataloading import NeighborSampler, DataLoader as DGLDataLoader
    except Exception:
        # No DGL wheel for this platform. The link baseline is pure PyG and still
        # runs; the node baseline needs DGL's neighbour sampler, so it is skipped
        # with a clear message rather than failing the whole phase.
        dgl = dglnn = NeighborSampler = DGLDataLoader = None

    for dataset in datasets:
        p   = get_paths_fn(dataset)
        cfg = dataset_cfg[dataset]
        IN_FEATS    = cfg['in_feats']
        NUM_CLASSES = cfg['num_classes']
        EPOCHS      = baseline_cfg['epochs']

        print(f"\n{'='*60}\n  PHASE 4 — GraphSAGE Baseline: {dataset}\n{'='*60}")
        print(f"  epochs={EPOCHS}")

        t0       = time.time()
        nodes_df = spark.read.format('delta').load(p['nodes'])
        edges_df = spark.read.format('delta').load(p['edges'])
        masks_df = spark.read.format('delta').load(p['masks'])

        nodes_pd = nodes_df.orderBy('id').toPandas()
        edges_pd = edges_df.toPandas()
        masks_pd = masks_df.toPandas()
        print(f"  Loaded in {time.time()-t0:.1f}s")

        feats_np  = np.stack(nodes_pd['features'].values).astype(np.float32)
        feats_norms = np.linalg.norm(feats_np, axis=1, keepdims=True)
        feats_np = feats_np / np.where(feats_norms > 0, feats_norms, 1.0)
        labels_np = nodes_pd['label'].values.astype(np.int64)
        src_np    = edges_pd['src'].values.astype(np.int64)
        dst_np    = edges_pd['dst'].values.astype(np.int64)
        n_nodes   = len(nodes_pd)
        IN_FEATS  = feats_np.shape[1]

        id2split    = dict(zip(masks_pd['id'].astype(int), masks_pd['split']))
        train_mask  = np.array([id2split.get(i,'') == 'train' for i in range(n_nodes)])
        val_mask    = np.array([id2split.get(i,'') == 'valid' for i in range(n_nodes)])
        test_mask   = np.array([id2split.get(i,'') == 'test'  for i in range(n_nodes)])

        # Substrate parity with Phase 3.
        #
        # Phase 3 trains under per-community caps (max nodes, max edges, hash
        # sampling); an unconstrained full-graph baseline sees everything. Comparing
        # them directly conflates two losses: information lost by partitioning (the
        # method's cost) and information lost by the sampling budget (an implementation
        # limit). Setting edge_budget/node_budget to the totals Phase 3 actually
        # consumed — reported as n_nodes/n_edges on its summary row — gives a third
        # condition that separates the two.
        edge_budget = baseline_cfg.get('edge_budget') or 0
        node_budget = baseline_cfg.get('node_budget') or 0
        budget_rng = np.random.default_rng(42)
        remap = None          # set below when a node budget renumbers ids

        if node_budget and node_budget < n_nodes:
            keep_nodes = np.sort(budget_rng.choice(n_nodes, int(node_budget), replace=False))
            remap = np.full(n_nodes, -1, dtype=np.int64)
            remap[keep_nodes] = np.arange(len(keep_nodes))
            in_sub = (remap[src_np] >= 0) & (remap[dst_np] >= 0)
            src_np, dst_np = remap[src_np[in_sub]], remap[dst_np[in_sub]]
            feats_np = feats_np[keep_nodes]
            labels_np = labels_np[keep_nodes]
            train_mask = train_mask[keep_nodes]
            val_mask = val_mask[keep_nodes]
            test_mask = test_mask[keep_nodes]
            n_nodes = len(keep_nodes)
            print(f"  [budget] node budget: {n_nodes:,} nodes, {len(src_np):,} edges remain")

        if edge_budget and edge_budget < len(src_np):
            keep_e = budget_rng.choice(len(src_np), int(edge_budget), replace=False)
            src_np, dst_np = src_np[keep_e], dst_np[keep_e]
            print(f"  [budget] edge budget: {len(src_np):,} edges retained")

        if not (edge_budget or node_budget):
            print(f"  [budget] unconstrained full graph: {n_nodes:,} nodes, {len(src_np):,} edges")

        # Partition map for the like-for-like link metric below. Phase 4 exists to be
        # a fair yardstick for Phase 3, and Phase 3 draws negatives from inside one
        # community -- so a purely global baseline answers a harder question and
        # retention against it reads above 100%. Load the same partitioning here and
        # report both numbers rather than choosing between them.
        _algs = kwargs.get('algorithms') or []
        comm_dense = comm_order = comm_starts = comm_counts = None
        if _algs:
            try:
                _cp = get_paths_fn(dataset, _algs[0])['communities']
                _cpd = spark.read.format('delta').load(_cp).select('id', 'community_id').toPandas()
                _ids = _cpd['id'].values.astype(np.int64)
                _cid = _cpd['community_id'].values.astype(np.int64)
                if remap is not None:
                    # A node budget renumbered the graph into a dense reduced range,
                    # but the community table still speaks original ids. Translate
                    # first; nodes the budget dropped map to -1 and fall away.
                    _in = (_ids >= 0) & (_ids < len(remap))
                    _ids, _cid = remap[_ids[_in]], _cid[_in]
                    _live = _ids >= 0
                    _ids, _cid = _ids[_live], _cid[_live]
                _raw = np.full(n_nodes, -1, dtype=np.int64)
                _ok = (_ids >= 0) & (_ids < n_nodes)
                _raw[_ids[_ok]] = _cid[_ok]
                _uniq, comm_dense = np.unique(_raw, return_inverse=True)
                comm_order = np.argsort(comm_dense, kind='stable')
                comm_counts = np.bincount(comm_dense, minlength=len(_uniq))
                comm_starts = np.concatenate([[0], np.cumsum(comm_counts)[:-1]])
                print(f"  [partition metric] loaded {len(_uniq):,} communities from {_algs[0]}")
            except Exception as _e:
                print(f"  [partition metric] unavailable ({type(_e).__name__}); "
                      f"reporting global link AUC only")
                comm_dense = None

        full_src = torch.tensor(src_np, dtype=torch.long)
        full_dst = torch.tensor(dst_np, dtype=torch.long)

        # Build DGL Graph once outside the runs loop to optimize memory. Only the
        # node baseline needs it; the link baseline is pure PyG.
        if dgl is not None:
            g = dgl.graph((full_src, full_dst), num_nodes=n_nodes)
            g = dgl.add_self_loop(g)
            g.ndata['feat']  = torch.tensor(feats_np,  dtype=torch.float32)
            g.ndata['label'] = torch.tensor(labels_np, dtype=torch.int64)
        else:
            g = None

        # Reclaim massive Pandas/Numpy memory allocations immediately
        del nodes_pd, edges_pd, masks_pd, src_np, dst_np
        import gc; gc.collect()

        HIDDEN = baseline_cfg.get('hidden_dim', 256)
        dropout_val = baseline_cfg.get('dropout', 0.5)

        class GraphSAGE(nn.Module):
            def __init__(self, input_dim, hidden_channels):
                super().__init__()
                self.conv1 = SAGEConv(input_dim, hidden_channels)
                self.conv2 = SAGEConv(hidden_channels, hidden_channels)
                self.dropout = nn.Dropout(dropout_val)
            def encode(self, x, edge_index):
                x = self.conv1(x, edge_index).relu()
                x = self.dropout(x)
                x = self.conv2(x, edge_index)
                return x

        class LinkPredictor(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.fc1 = nn.Linear(h, h)
                self.fc2 = nn.Linear(h, 1)
            def forward(self, h_src, h_dst):
                x = h_src * h_dst
                x = torch.relu(self.fc1(x))
                return self.fc2(x).squeeze(-1)

        class GraphSAGENodeClassifier(nn.Module):
            def __init__(self, in_f, h, nc):
                super().__init__()
                self.conv1 = dglnn.SAGEConv(in_f, h, 'mean')
                self.conv2 = dglnn.SAGEConv(h,    h, 'mean')
                self.fc = nn.Linear(h, nc)
                self.dropout = nn.Dropout(dropout_val)
            def forward(self, blocks, x):
                h = x
                h = self.conv1(blocks[0], h).relu()
                h = self.dropout(h)
                h = self.conv2(blocks[1], h)
                return self.fc(h)

        task_type = kwargs.get('task_type', 'node_classification')
        run_node = (task_type in ('node_classification', 'both'))
        run_link = (task_type in ('link_prediction', 'both'))

        N_RUNS = kwargs.get('n_baseline_runs', 3)
        all_accs = []
        all_aucs = []
        all_node_times = []
        all_link_times = []
        all_aucs_part = []

        for run_idx in range(N_RUNS):
            print(f"\n  ── GraphSAGE Run {run_idx+1}/{N_RUNS} ──")
            acc = 0.0
            correct = 0
            total_t = 0
            node_train_time = 0.0
            baseline_link_auc = 0.5
            baseline_link_auc_partition = None
            link_train_time = 0.0

            # ── Node Classification Baseline ──────────────────────────────────────
            if run_node and dgl is None:
                print("    [skip] node baseline needs DGL's neighbour sampler; "
                      "unavailable on this platform. Link baseline still runs.")
            if run_node and dgl is not None and train_mask.sum() > 0:
                t_node_start = time.time()

                model = GraphSAGENodeClassifier(IN_FEATS, HIDDEN, NUM_CLASSES)
                opt = torch.optim.Adam(model.parameters(), lr=baseline_cfg.get('lr', 1e-3), weight_decay=5e-4)
                crit = nn.CrossEntropyLoss()

                train_nids = torch.where(torch.tensor(train_mask))[0]
                sampler = NeighborSampler(baseline_cfg.get('fanout', [15, 10]))
                train_dl = DGLDataLoader(g, train_nids, sampler,
                                         batch_size=baseline_cfg.get('batch', 1024),
                                         shuffle=True, drop_last=False)

                # Validation-tracked training: the baseline defines the upper bound
                # every retention figure is measured against, so it runs to
                # convergence and keeps the best-validation weights rather than
                # whatever a fixed epoch budget happened to land on.
                node_epochs = int(baseline_cfg.get('node_epochs', EPOCHS))
                node_patience = int(baseline_cfg.get('node_patience', 10))
                val_nids = torch.where(torch.tensor(val_mask))[0]
                val_dl = (DGLDataLoader(g, val_nids, NeighborSampler(baseline_cfg.get('fanout', [15, 10])),
                                        batch_size=256, shuffle=False, drop_last=False)
                          if len(val_nids) > 0 else None)
                best_val_acc, best_node_state, node_waited = -1.0, None, 0

                for epoch in range(node_epochs):
                    model.train()
                    total_loss = 0.0
                    nb = 0
                    for input_nodes, output_nodes, blocks in train_dl:
                        x = blocks[0].srcdata['feat']
                        labels = blocks[-1].dstdata['label']
                        logits = model(blocks, x)
                        loss = crit(logits, labels)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        total_loss += loss.item()
                        nb += 1
                    if (epoch + 1) % 5 == 0 or epoch == node_epochs - 1:
                        print(f"    SAGE-BL Epoch {epoch+1:2d}/{node_epochs} loss={total_loss/max(nb,1):.4f}")

                    if val_dl is not None:
                        model.eval()
                        v_ok = v_tot = 0
                        with torch.no_grad():
                            for _, _, vb in val_dl:
                                vx = vb[0].srcdata['feat']
                                vy = vb[-1].dstdata['label']
                                v_ok += (model(vb, vx).argmax(dim=1) == vy).sum().item()
                                v_tot += len(vy)
                        v_acc = v_ok / v_tot if v_tot else 0.0
                        if v_acc > best_val_acc:
                            best_val_acc, node_waited = v_acc, 0
                            best_node_state = copy.deepcopy(model.state_dict())
                        else:
                            node_waited += 1
                            if node_waited >= node_patience:
                                print(f"    Node baseline converged at epoch {epoch+1} "
                                      f"(best val acc {best_val_acc:.4f})")
                                break

                if best_node_state is not None:
                    model.load_state_dict(best_node_state)
                node_train_time = time.time() - t_node_start

                # Clean up training loader memory
                del train_dl, sampler, train_nids
                import gc; gc.collect()

                # Evaluate using test DGLDataLoader with a safe batch size (256) to prevent OOM
                test_nids = torch.where(torch.tensor(test_mask))[0]
                test_dl = DGLDataLoader(g, test_nids, NeighborSampler(baseline_cfg.get('fanout', [15, 10])),
                                        batch_size=256, shuffle=False, drop_last=False)
                model.eval()
                correct = 0
                total_t = 0
                with torch.no_grad():
                    for input_nodes, output_nodes, blocks in test_dl:
                        x = blocks[0].srcdata['feat']
                        labels = blocks[-1].dstdata['label']
                        preds = model(blocks, x).argmax(dim=1)
                        correct += (preds == labels).sum().item()
                        total_t += len(labels)
                acc = correct / total_t if total_t > 0 else 0.0

            # ── Link Prediction Baseline ──────────────────────────────────────────
            if run_link and len(full_src) >= 5:
                import gc; gc.collect()
                
                t_link_start = time.time()
                torch.manual_seed(42 + run_idx)

                # Downsample edges to prevent CPU RandomLinkSplit OOM on massive edge lists (e.g. products)
                if len(full_src) > 5000000:
                    print(f"    [Warning] Edge count is {len(full_src)}. Downsampling to 3M edges for memory-safe single-node Link Prediction split...")
                    perm = torch.randperm(len(full_src))[:3000000]
                    edge_index = torch.stack([full_src[perm], full_dst[perm]], dim=0)
                else:
                    edge_index = torch.stack([full_src, full_dst], dim=0)

                feat_t = torch.tensor(feats_np, dtype=torch.float32)
                graph_pyg = Data(x=feat_t, edge_index=edge_index)

                # Ratios must match Phase 3's per-community 80/10/10 split, or the
                # baseline measures a different task from the system under test.
                split = T.RandomLinkSplit(
                    num_val=float(baseline_cfg.get('link_val_frac', 0.10)),
                    num_test=float(baseline_cfg.get('link_test_frac', 0.10)),
                    is_undirected=True,
                    add_negative_train_samples=False,
                    neg_sampling_ratio=1.0,
                )

                try:
                    train_data, val_data, test_data = split(graph_pyg)
                except ValueError:
                    train_data, val_data, test_data = None, None, None

                if train_data is not None:
                    encoder = GraphSAGE(IN_FEATS, HIDDEN)
                    predictor = LinkPredictor(HIDDEN)
                    optimizer = torch.optim.Adam(
                        list(encoder.parameters()) + list(predictor.parameters()),
                        lr=baseline_cfg.get('lr', 1e-3), weight_decay=5e-4
                    )
                    criterion = torch.nn.BCEWithLogitsLoss()
                    from sklearn.metrics import roc_auc_score

                    # An upper-bound baseline has to be trained to convergence to
                    # mean anything. One full-batch step per configured epoch left
                    # this at ~10 updates and produced below-chance AUC, which made
                    # every "retention vs full graph" comparison meaningless.
                    link_epochs = int(baseline_cfg.get('link_epochs', 300))
                    patience = int(baseline_cfg.get('link_patience', 30))
                    eval_every = 5
                    best_val, best_state, waited = -1.0, None, 0

                    def _val_auc():
                        encoder.eval(); predictor.eval()
                        with torch.no_grad():
                            zz = encoder.encode(train_data.x, train_data.edge_index)
                            s = predictor(zz[val_data.edge_label_index[0]],
                                          zz[val_data.edge_label_index[1]]).view(-1)
                            yt = val_data.edge_label.cpu().numpy()
                            if len(np.unique(yt)) < 2:
                                return float('nan')
                            return float(roc_auc_score(yt, s.cpu().numpy()))

                    for epoch in range(1, link_epochs + 1):
                        encoder.train()
                        predictor.train()
                        optimizer.zero_grad()

                        z = encoder.encode(train_data.x, train_data.edge_index)

                        neg_edge_index = negative_sampling(
                            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
                            num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

                        pos_src = train_data.edge_label_index[0]
                        pos_dst = train_data.edge_label_index[1]
                        neg_src = neg_edge_index[0]
                        neg_dst = neg_edge_index[1]

                        pos_scores = predictor(z[pos_src], z[pos_dst])
                        neg_scores = predictor(z[neg_src], z[neg_dst])

                        scores = torch.cat([pos_scores, neg_scores])
                        labels = torch.cat([
                            torch.ones_like(pos_scores),
                            torch.zeros_like(neg_scores)
                        ])

                        loss = criterion(scores, labels)
                        loss.backward()
                        optimizer.step()

                        if epoch % eval_every == 0 or epoch == link_epochs:
                            v = _val_auc()
                            if v == v and v > best_val:
                                best_val, waited = v, 0
                                best_state = (copy.deepcopy(encoder.state_dict()),
                                              copy.deepcopy(predictor.state_dict()))
                            else:
                                waited += eval_every
                                if waited >= patience:
                                    print(f"    Link baseline converged at epoch {epoch} "
                                          f"(best val AUC {best_val:.4f})")
                                    break

                    if best_state is not None:
                        encoder.load_state_dict(best_state[0])
                        predictor.load_state_dict(best_state[1])

                    with torch.no_grad():
                        encoder.eval()
                        predictor.eval()
                        z = encoder.encode(feat_t, train_data.edge_index)
                        
                        test_src = test_data.edge_label_index[0]
                        test_dst = test_data.edge_label_index[1]
                        pos_scores = predictor(z[test_src], z[test_dst]).view(-1)
                        
                        y_true = test_data.edge_label.cpu().numpy()
                        y_scores = pos_scores.cpu().numpy()

                        from sklearn.metrics import roc_auc_score
                        try:
                            baseline_link_auc = float(roc_auc_score(y_true, y_scores))
                        except ValueError:
                            baseline_link_auc = 0.5

                        # Same positives, negatives restricted to the source vertex's
                        # own community: the exact question Phase 3 answers. Sampling is
                        # vectorised through a CSR-style index over the partition, with
                        # a few rejection rounds against real edges and self-pairs.
                        if comm_dense is not None:
                            _pm = (test_data.edge_label == 1)
                            _ps = test_data.edge_label_index[0][_pm].cpu().numpy().astype(np.int64)
                            _pd = test_data.edge_label_index[1][_pm].cpu().numpy().astype(np.int64)
                            if len(_ps) > 0:
                                _rng = np.random.default_rng(42)
                                _ei = train_data.edge_index.cpu().numpy().astype(np.int64)
                                _ek = np.unique(_ei[0] * np.int64(n_nodes) + _ei[1])
                                _c = comm_dense[_ps]
                                _w = np.full(len(_ps), -1, dtype=np.int64)
                                _todo = comm_counts[_c] > 1
                                for _ in range(8):
                                    if not _todo.any():
                                        break
                                    _idx = np.where(_todo)[0]
                                    _r = (_rng.random(len(_idx)) * comm_counts[_c[_idx]]).astype(np.int64)
                                    _cand = comm_order[comm_starts[_c[_idx]] + _r]
                                    _key = _ps[_idx] * np.int64(n_nodes) + _cand
                                    _good = (_cand != _ps[_idx]) & ~np.isin(_key, _ek)
                                    _w[_idx[_good]] = _cand[_good]
                                    _todo[_idx[_good]] = False
                                _keep = _w >= 0
                                if _keep.sum() > 0:
                                    with torch.no_grad():
                                        _npos = predictor(z[torch.as_tensor(_ps[_keep])],
                                                          z[torch.as_tensor(_pd[_keep])]).view(-1)
                                        _nneg = predictor(z[torch.as_tensor(_ps[_keep])],
                                                          z[torch.as_tensor(_w[_keep])]).view(-1)
                                    _yt = np.concatenate([np.ones(int(_keep.sum())),
                                                          np.zeros(int(_keep.sum()))])
                                    _ys = torch.cat([_npos, _nneg]).cpu().numpy()
                                    try:
                                        baseline_link_auc_partition = float(roc_auc_score(_yt, _ys))
                                    except ValueError:
                                        baseline_link_auc_partition = None
                else:
                    baseline_link_auc = 0.5
                link_train_time = time.time() - t_link_start

            all_accs.append(acc)
            all_aucs.append(baseline_link_auc)
            if baseline_link_auc_partition is not None:
                all_aucs_part.append(baseline_link_auc_partition)
            all_node_times.append(node_train_time)
            all_link_times.append(link_train_time)
            print(f"    Run {run_idx+1} — acc={acc:.4f}  auc={baseline_link_auc:.4f}")

        # A skipped node baseline (no DGL) records None, not an accuracy of 0.0.
        node_ran = run_node and dgl is not None and train_mask.sum() > 0
        mean_acc = float(np.mean(all_accs)) if node_ran else None
        std_acc = float(np.std(all_accs)) if node_ran else None
        mean_auc = np.mean(all_aucs)
        std_auc = np.std(all_aucs)
        mean_auc_part = float(np.mean(all_aucs_part)) if all_aucs_part else None
        mean_node_time = np.mean(all_node_times)
        mean_link_time = np.mean(all_link_times)

        peak_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024.0

        timing[('phase4_node', dataset)] = mean_node_time
        timing[('phase4_link', dataset)] = mean_link_time
        timing[('phase4', dataset)]      = mean_node_time + mean_link_time

        results[dataset] = {
            'test_acc':          mean_acc,
            'test_acc_std':      std_acc,
            'link_auc':          mean_auc,
            'link_auc_std':      std_auc,
            # Global = true single-machine upper bound, comparable to published work.
            # Partition = same task as Phase 3, so retention is a fair comparison.
            'link_auc_global':   mean_auc,
            'link_auc_partition': mean_auc_part,
            'node_train_time_s': mean_node_time,
            'link_train_time_s': mean_link_time,
            'train_time_s':      mean_node_time + mean_link_time,
            'peak_mem_gb':       peak_mem / 1e9,
            'n_test':            total_t,
            'n_correct':         correct,
            'all_accs':          all_accs,
            'all_aucs':          all_aucs,
        }
        acc_txt = f"{mean_acc:.4f}±{std_acc:.4f}" if node_ran else "not run"
        print(f"  ✓ [{dataset}] GraphSAGE Baseline  acc={acc_txt}  "
              f"auc={mean_auc:.4f}±{std_auc:.4f}  "
              f"time={mean_node_time + mean_link_time:.1f}s")
              
        # Cleanup graph structure at the end of run_phase4
        if 'g' in locals():
            del g
        import gc; gc.collect()

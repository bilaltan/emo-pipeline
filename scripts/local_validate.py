#!/usr/bin/env python3
"""
Spark-free local validation of the LakeGRL scientific path.

Drives the *real* Phase 3 / Phase 3b community UDFs on a locally available
dataset so the partitioning, training, fusion and evaluation logic can be
exercised without a cluster. Every claim the paper makes about Stage 3a and
Stage 3b passes through the same functions this harness calls.

The point is the invariants at the bottom: a run that silently degenerates
(one community, a stubbed 0.5 AUC, a leaking test mask) fails here loudly
instead of producing a plausible-looking number.

    python3 scripts/local_validate.py --dataset deezer
    python3 scripts/local_validate.py --dataset deezer --no-halo
    python3 scripts/local_validate.py --dataset reddit --max-communities 8
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phases.phase3_training import _train_gnn_community_single


# ────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ────────────────────────────────────────────────────────────────────────────

def load_deezer(root="data/deezer_europe_extracted/deezer_europe", feat_dim=128):
    """DeezerEurope: 28K nodes, binary target, sparse categorical features."""
    edges = pd.read_csv(os.path.join(root, "deezer_europe_edges.csv"))
    target = pd.read_csv(os.path.join(root, "deezer_europe_target.csv"))
    with open(os.path.join(root, "deezer_europe_features.json")) as fh:
        raw_feats = json.load(fh)

    n_nodes = int(target["id"].max()) + 1
    y = np.zeros(n_nodes, dtype=np.int64)
    y[target["id"].values.astype(int)] = target["target"].values.astype(int)

    # Hash the sparse feature ids into a fixed-width multi-hot block. Uses the
    # real feature lists rather than the random matrix the older scripts built.
    x = np.zeros((n_nodes, feat_dim), dtype=np.float32)
    for node_str, feat_ids in raw_feats.items():
        node = int(node_str)
        if node >= n_nodes:
            continue
        for f in feat_ids:
            x[node, int(f) % feat_dim] += 1.0
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.where(norms > 0, norms, 1.0)

    src = edges["node_1"].values.astype(np.int64)
    dst = edges["node_2"].values.astype(np.int64)
    return x, y, src, dst, n_nodes, 2


def load_reddit(root="data/reddit"):
    from torch_geometric.datasets import Reddit
    data = Reddit(root=root)[0]
    x = data.x.numpy().astype(np.float32)
    y = data.y.numpy().astype(np.int64)
    ei = data.edge_index.numpy()
    keep = ei[0] < ei[1]                       # undirected: keep one direction
    return x, y, ei[0][keep], ei[1][keep], x.shape[0], int(y.max()) + 1


# ────────────────────────────────────────────────────────────────────────────
# Partitioning and abstraction (mirrors Phase 1 / Phase 2 semantics)
# ────────────────────────────────────────────────────────────────────────────

def louvain_partition(src, dst, n_nodes, seed=42, resolution=1.0,
                      cache_dir=None, dataset=""):
    """Louvain modularity partitioning.

    Prefers igraph's C implementation — the same backend Phase 1 uses, and the
    only one that handles a 100M-edge graph in reasonable memory. Falls back to
    networkx for small graphs when igraph is not installed.

    Mirrors the cluster's Phase 1 contract: the partition is a reusable artifact
    keyed by (dataset, algorithm, resolution, graph shape). On EMR that artifact
    is the Delta table under communities/{alg}/; here it is a local .npy. Either
    way the expensive step runs once per graph, not once per experiment.
    """
    cache_path = None
    if cache_dir:
        key = f"{dataset}_louvain_r{resolution}_s{seed}_n{n_nodes}_e{len(src)}"
        cache_path = os.path.join(cache_dir, f"{key}.npy")
        if os.path.exists(cache_path):
            print(f"  partition    reusing cached Louvain artifact ({key})")
            return np.load(cache_path)

    assign = np.full(n_nodes, -1, dtype=np.int64)

    def _persist(a):
        if cache_path:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_path, a)
        return a
    try:
        import random
        import igraph as ig
        # community_multilevel is randomized; without a fixed generator the same
        # graph yields a different number of communities on every run.
        rng = random.Random(seed)
        ig.set_random_number_generator(rng)
        # Pass the edge array straight through; materializing a Python list of
        # tuples costs several GB on a 50M-edge graph.
        g = ig.Graph(n=int(n_nodes),
                     edges=np.stack([src, dst], axis=1).astype(np.int64),
                     directed=False)
        partition = g.community_multilevel(resolution=resolution)
        for cid, members in enumerate(partition):
            assign[np.asarray(members, dtype=np.int64)] = cid
        return _persist(assign)
    except ImportError:
        import networkx as nx
        g = nx.Graph()
        g.add_nodes_from(range(n_nodes))
        g.add_edges_from(zip(src.tolist(), dst.tolist()))
        for cid, members in enumerate(
                nx.community.louvain_communities(g, seed=seed, resolution=resolution)):
            for v in members:
                assign[v] = cid
        return _persist(assign)


def apply_size_policy(assign, min_size, handling="drop"):
    """Phase 2's tiny-community rule. Returns (assignment, kept_major_ids)."""
    ids, counts = np.unique(assign, return_counts=True)
    major = ids[counts >= min_size]
    if handling == "drop":
        out = np.where(np.isin(assign, major), assign, -1)
    else:                                       # 'misc': fold tiny into one bucket
        out = np.where(np.isin(assign, major), assign, -1)
    return out, major


def boundary_flags(assign, src, dst, n_nodes):
    """b(v) = 1 iff v has a neighbour in a different community (Eq. 1)."""
    cut = assign[src] != assign[dst]
    bnd = np.zeros(n_nodes, dtype=bool)
    bnd[src[cut]] = True
    bnd[dst[cut]] = True
    return bnd


def make_splits(y, n_nodes, seed=42, ratios=(0.6, 0.2, 0.2)):
    rng = np.random.default_rng(seed)
    split = np.array(["none"] * n_nodes, dtype=object)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_tr = int(ratios[0] * len(idx))
        n_va = int(ratios[1] * len(idx))
        split[idx[:n_tr]] = "train"
        split[idx[n_tr:n_tr + n_va]] = "valid"
        split[idx[n_tr + n_va:]] = "test"
    return split


# ────────────────────────────────────────────────────────────────────────────
# Driving the real UDF
# ────────────────────────────────────────────────────────────────────────────

HYPER = {
    "_num_classes": None, "_hidden": 256, "_epochs": 10, "_lr": 0.001,
    "_dropout": 0.5, "_task_type": "both", "_model_type": "sage",
    "_max_nodes": 10000, "_max_edges": 30000, "_mlp_epochs": 10,
    "_phase3_diagnostics": False,
}


def run_community(cid, members, x, y, split, bnd, src, dst, csrc, cdst,
                  num_classes, expand_boundary):
    """Build the pdf the Phase 3 UDF expects and call it for one community."""
    # csrc/cdst are the community ids of each edge's endpoints, precomputed once
    # for the whole graph; np.isin over every edge per community does not scale.
    s_in = csrc == cid
    d_in = cdst == cid
    in_c = s_in & d_in
    e_src, e_dst = src[in_c], dst[in_c]

    if expand_boundary:
        # 1-hop halo: retain cut edges incident to this community's boundary so
        # the local model still sees the severed connection.
        halo = s_in ^ d_in
        h_src, h_dst = src[halo], dst[halo]
        ext = np.unique(np.where(csrc[halo] == cid, h_dst, h_src))
        ext = ext[~np.isin(ext, members)]
        # Members first, halo after: the UDF treats the first _n_local rows as the
        # community proper and everything after as context.
        node_ids = np.concatenate([members, ext])
        e_src = np.concatenate([e_src, h_src])
        e_dst = np.concatenate([e_dst, h_dst])
    else:
        node_ids = members

    # Halo nodes are context only: they carry no label and no split, so they can
    # never contribute to this community's reported accuracy.
    is_member = np.arange(len(node_ids)) < len(members)
    node_df = pd.DataFrame({
        "id": node_ids,
        "label": np.where(is_member, y[node_ids], -1),
        "features": list(x[node_ids]),
        "split": [split[n] if m else "none" for n, m in zip(node_ids, is_member)],
        "is_boundary": bnd[node_ids],
        "community_id": cid,
    })
    # Same column Phase 2 now emits, so the harness exercises the production path.
    node_df["is_member"] = is_member
    for k, v in HYPER.items():
        node_df[k] = num_classes if k == "_num_classes" else v

    edge_df = pd.DataFrame({"src": e_src, "dst": e_dst})
    out = _train_gnn_community_single(node_df, comm_edges_pdf=edge_df)
    row = out.iloc[0].to_dict()
    row["n_members"] = len(members)
    row["n_halo"] = len(node_ids) - len(members)
    return row


# ────────────────────────────────────────────────────────────────────────────
# Stage 3b: global topological abstraction + late fusion
# ────────────────────────────────────────────────────────────────────────────

class _BC:
    """Stands in for a Spark broadcast so the real Phase 3b UDF runs unchanged."""
    def __init__(self, value):
        self.value = value


def build_caan_components(assign, major, x, y, split, src, dst, min_size):
    """Phase 2 of Algorithm 1: super-nodes, minor nodes, and the macro adjacency.

    Major communities collapse to a centroid super-node (Eq. 2); nodes in minor
    communities stay explicit; severed edges become meta-edges between them.
    """
    major_set = set(int(c) for c in major)

    super_nodes_dict = {
        int(cid): x[np.where(assign == cid)[0]].mean(axis=0).astype(np.float32)
        for cid in major
    }

    minor_mask = ~np.isin(assign, list(major_set))
    minor_nodes_dict = {
        int(n): {"features": x[n].tolist(), "label": int(y[n]), "split": str(split[n])}
        for n in np.where(minor_mask)[0]
    }

    node_to_comm = {int(n): int(c) for n, c in enumerate(assign)}

    # Map each endpoint to its super-node when it sits in a major community.
    def _map(nodes):
        comm = assign[nodes]
        is_major = np.isin(comm, list(major_set))
        return np.where(is_major, -1000 - comm, nodes)

    u_m, v_m = _map(src), _map(dst)
    keep = u_m != v_m                      # drop super-node self-loops
    u_k = np.concatenate([u_m[keep], v_m[keep]])
    v_k = np.concatenate([v_m[keep], u_m[keep]])

    # Deduplicate on a flat int64 key rather than np.unique(axis=0). A structured
    # sort of ~115M rows costs minutes and gigabytes; almost every edge collapses
    # onto the same handful of super-node pairs, so the unique set is tiny.
    shift = np.int64(-min(int(u_k.min()), int(v_k.min()), 0))
    span = np.int64(max(int(u_k.max()), int(v_k.max())) + shift + 1)
    _, first = np.unique(((u_k + shift) * span + (v_k + shift)), return_index=True)

    caan_adj = {}
    for u, v in zip(u_k[first].tolist(), v_k[first].tolist()):
        caan_adj.setdefault(int(u), []).append(int(v))

    return {
        "super_nodes_dict": super_nodes_dict,
        "minor_nodes_dict": minor_nodes_dict,
        "node_to_comm": node_to_comm,
        "caan_adj": caan_adj,
        "major_comms": major_set,
    }


def run_community_caan(cid, members, x, y, split, bnd, src, dst, csrc, cdst,
                       num_classes, caan_udf):
    """Call the real Phase 3b UDF for one community."""
    in_c = csrc == cid
    in_c_rev = cdst == cid
    # Orient every incident edge so src is the local node; cut edges are kept so
    # the UDF can resolve external endpoints to super-nodes.
    e_src = np.concatenate([src[in_c], dst[in_c_rev]])
    e_dst = np.concatenate([dst[in_c], src[in_c_rev]])

    node_df = pd.DataFrame({
        "id": members,
        "label": y[members],
        "features": list(x[members]),
        "split": [split[n] for n in members],
        "is_boundary": bnd[members],
        "community_id": cid,
    })
    for k, v in HYPER.items():
        node_df[k] = num_classes if k == "_num_classes" else v

    out = caan_udf(node_df, pd.DataFrame({"src": e_src, "dst": e_dst}))
    row = out.iloc[0].to_dict()
    row["n_members"] = len(members)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="deezer", choices=["deezer", "reddit"])
    ap.add_argument("--min-community-size", type=int, default=100)
    ap.add_argument("--max-communities", type=int, default=0,
                    help="0 = all; otherwise cap for a fast smoke run")
    ap.add_argument("--no-halo", action="store_true",
                    help="disable 1-hop boundary expansion (EXPAND_BOUNDARY_NODES=False)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json-out", default="",
                    help="append the run's metrics as a JSON line for aggregation")
    ap.add_argument("--cache-dir", default="data/.partition_cache",
                    help="where to persist the Louvain artifact; '' disables reuse")
    ap.add_argument("--max-nodes", type=int, default=10000,
                    help="per-community node cap (PHASE3_MAX_NODES_PER_COMMUNITY)")
    ap.add_argument("--max-edges", type=int, default=30000,
                    help="per-community edge cap (PHASE3_MAX_EDGES_PER_COMMUNITY)")
    ap.add_argument("--skip-3b", action="store_true",
                    help="run only Stage 3a (local training), skip CAAN fusion")
    ap.add_argument("--subsample", type=int, default=0,
                    help="induce a subgraph on N randomly chosen nodes "
                         "(keeps large graphs tractable for a local Louvain)")
    args = ap.parse_args()

    expand = not args.no_halo
    HYPER["_max_nodes"] = args.max_nodes
    HYPER["_max_edges"] = args.max_edges
    t0 = time.time()

    print(f"\n{'='*74}\n  LakeGRL local validation — {args.dataset}"
          f"  (halo={'on' if expand else 'off'})\n{'='*74}")

    loader = load_deezer if args.dataset == "deezer" else load_reddit
    x, y, src, dst, n_nodes, num_classes = loader()

    if args.subsample and args.subsample < n_nodes:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(n_nodes, args.subsample, replace=False))
        remap = np.full(n_nodes, -1, dtype=np.int64)
        remap[keep] = np.arange(len(keep))
        in_sub = np.isin(src, keep) & np.isin(dst, keep)
        src, dst = remap[src[in_sub]], remap[dst[in_sub]]
        x, y = x[keep], y[keep]
        n_nodes = len(keep)
        # Relabel classes so the label space stays contiguous after subsampling.
        classes, y = np.unique(y, return_inverse=True)
        num_classes = len(classes)
        print(f"  subsample    induced subgraph on {n_nodes:,} nodes "
              f"({num_classes} classes survive)")

    print(f"  graph        {n_nodes:,} nodes  {len(src):,} undirected edges  "
          f"{num_classes} classes  {x.shape[1]}-dim features")

    split = make_splits(y, n_nodes, seed=args.seed)
    t_part = time.time()
    assign = louvain_partition(src, dst, n_nodes, seed=args.seed,
                               cache_dir=args.cache_dir,
                               dataset=f"{args.dataset}{'_sub'+str(args.subsample) if args.subsample else ''}")
    assign, major = apply_size_policy(assign, args.min_community_size)
    print(f"  partition    {len(major)} communities >= {args.min_community_size} nodes "
          f"in {time.time()-t_part:.1f}s  "
          f"(dropped {int((assign == -1).sum()):,} nodes in tiny communities)")

    csrc, cdst = assign[src], assign[dst]
    bnd = boundary_flags(assign, src, dst, n_nodes)
    cut = int((assign[src] != assign[dst]).sum())
    print(f"  abstraction  {int(bnd.sum()):,} boundary nodes  {cut:,} cut edges "
          f"({100*cut/max(len(src),1):.1f}% of edges severed)")

    targets = major if args.max_communities == 0 else major[:args.max_communities]
    rows = []
    print(f"\n  training {len(targets)} communities through the real Phase 3 UDF...")
    for i, cid in enumerate(targets, 1):
        members = np.where(assign == cid)[0]
        row = run_community(cid, members, x, y, split, bnd, src, dst, csrc, cdst,
                            num_classes, expand)
        rows.append(row)
        print(f"    [{i:3d}/{len(targets)}] comm {int(cid):<5d} "
              f"n={row['n_members']:>6,} halo={row['n_halo']:>6,} "
              f"edges={row['n_edges']:>8,} "
              f"acc={row['comm_test_acc']:.4f} auc={row['comm_link_auc']:.4f}")

    def summarize(frame):
        w_node = frame["n_nodes"].clip(lower=1)
        w_edge = frame["n_edges"].clip(lower=1)
        return {
            "acc": float((frame["comm_test_acc"] * w_node).sum() / w_node.sum()),
            "auc": float((frame["comm_link_auc"] * w_edge).sum() / w_edge.sum()),
            "bnd": float((frame["boundary_acc"] * w_node).sum() / w_node.sum()),
            "int": float((frame["internal_acc"] * w_node).sum() / w_node.sum()),
        }

    df = pd.DataFrame(rows)
    s3a = summarize(df)
    acc, auc = s3a["acc"], s3a["auc"]
    coverage = df["n_members"].sum() / n_nodes

    print(f"\n{'-'*74}\n  STAGE 3a RESULT ({time.time()-t0:.1f}s total)\n{'-'*74}")
    print(f"  communities trained      {len(df)}")
    print(f"  node coverage            {coverage*100:.1f}%  "
          f"({df['n_members'].sum():,} / {n_nodes:,})")
    print(f"  weighted node accuracy   {acc:.4f}")
    print(f"  weighted link ROC-AUC    {auc:.4f}")
    print(f"  boundary / internal acc  {s3a['bnd']:.4f} / {s3a['int']:.4f}")

    # ── Stage 3b: macro abstraction + late fusion ──────────────────────────
    s3b = None
    if not args.skip_3b:
        from phases.phase3b_caan import make_caan_udf
        t3b = time.time()
        comp = build_caan_components(assign, major, x, y, split, src, dst,
                                     args.min_community_size)
        print(f"\n  macro-graph  {len(comp['super_nodes_dict'])} super-nodes  "
              f"{len(comp['minor_nodes_dict']):,} minor nodes  "
              f"{sum(len(v) for v in comp['caan_adj'].values()):,} meta-edge endpoints")

        minor_ids = list(comp["minor_nodes_dict"].keys())
        caan_udf = make_caan_udf(
            super_nodes_dict_bc=_BC(comp["super_nodes_dict"]),
            minor_node_to_idx_bc=_BC({int(n): i for i, n in enumerate(minor_ids)}),
            minor_feats_arr_bc=_BC([comp["minor_nodes_dict"][n]["features"] for n in minor_ids]),
            minor_labels_arr_bc=_BC([comp["minor_nodes_dict"][n]["label"] for n in minor_ids]),
            minor_splits_arr_bc=_BC([comp["minor_nodes_dict"][n]["split"] for n in minor_ids]),
            minor_ids_arr_bc=_BC([int(n) for n in minor_ids]),
            caan_adj_bc=_BC(comp["caan_adj"]),
            node_to_comm_bc=_BC(comp["node_to_comm"]),
            major_comms_bc=_BC(comp["major_comms"]),
        ).community_fn

        print(f"  training {len(targets)} communities through the real Phase 3b UDF...")
        rows_b = []
        for i, cid in enumerate(targets, 1):
            members = np.where(assign == cid)[0]
            rb = run_community_caan(cid, members, x, y, split, bnd, src, dst,
                                    csrc, cdst, num_classes, caan_udf)
            rows_b.append(rb)
            print(f"    [{i:3d}/{len(targets)}] comm {int(cid):<5d} "
                  f"n={rb['n_members']:>6,} edges={rb['n_edges']:>8,} "
                  f"acc={rb['comm_test_acc']:.4f} auc={rb['comm_link_auc']:.4f}")

        df_b = pd.DataFrame(rows_b)
        s3b = summarize(df_b)
        print(f"\n{'-'*74}\n  STAGE 3b RESULT — CAAN fusion ({time.time()-t3b:.1f}s)\n{'-'*74}")
        print(f"  weighted node accuracy   {s3b['acc']:.4f}   "
              f"({s3b['acc']-acc:+.4f} vs Stage 3a)")
        print(f"  weighted link ROC-AUC    {s3b['auc']:.4f}   "
              f"({s3b['auc']-auc:+.4f} vs Stage 3a)")
        print(f"  boundary / internal acc  {s3b['bnd']:.4f} / {s3b['int']:.4f}")

    # ── invariants ─────────────────────────────────────────────────────────
    print(f"\n{'-'*74}\n  INVARIANTS\n{'-'*74}")
    checks = []

    checks.append(("R2  more than one community trained", len(df) > 1,
                   f"{len(df)} trained"))
    checks.append(("R2  no single -1 misc bucket monopoly",
                   not (len(df) == 1 and int(df['community_id'].iloc[0]) == -1),
                   f"ids={sorted(int(c) for c in df['community_id'])[:6]}"))
    checks.append(("R1  link AUC is a measurement, not the 0.5 stub",
                   not np.allclose(df["comm_link_auc"].values, 0.5),
                   f"{df['comm_link_auc'].min():.4f} – {df['comm_link_auc'].max():.4f}"))
    checks.append(("     link AUC beats chance overall", auc > 0.55, f"{auc:.4f}"))
    checks.append(("     node accuracy beats majority class",
                   acc > float(np.bincount(y).max() / len(y)),
                   f"{acc:.4f} vs {float(np.bincount(y).max()/len(y)):.4f} majority"))
    checks.append(("     node coverage above 90%", coverage > 0.90,
                   f"{coverage*100:.1f}%"))
    checks.append(("     no community reports perfect accuracy (leakage probe)",
                   float(df["comm_test_acc"].max()) < 0.999,
                   f"max {df['comm_test_acc'].max():.4f}"))
    checks.append(("     every community has test nodes",
                   int(df["n_test"].min()) > 0, f"min {int(df['n_test'].min())}"))

    if s3b is not None:
        checks.append(("3b  CAAN link AUC is a measurement, not the 0.5 stub",
                       not np.allclose(df_b["comm_link_auc"].values, 0.5),
                       f"{df_b['comm_link_auc'].min():.4f} – {df_b['comm_link_auc'].max():.4f}"))
        checks.append(("3b  fusion does not degrade node accuracy",
                       s3b["acc"] >= acc - 0.01, f"{s3b['acc']:.4f} vs {acc:.4f}"))
        checks.append(("3b  no community reports perfect accuracy (leakage probe)",
                       float(df_b["comm_test_acc"].max()) < 0.999,
                       f"max {df_b['comm_test_acc'].max():.4f}"))

    failed = 0
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52s} {detail}")
        failed += (not ok)

    print(f"\n  {len(checks)-failed}/{len(checks)} invariants passed")

    if args.json_out:
        import subprocess
        try:
            sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
        except Exception:
            sha = "unknown"
        record = {
            "dataset": args.dataset, "subsample": args.subsample, "seed": args.seed,
            "halo": expand, "min_community_size": args.min_community_size,
            "max_nodes": args.max_nodes, "max_edges": args.max_edges,
            "n_communities": len(df), "coverage": coverage,
            "n_nodes": int(n_nodes), "n_edges": int(len(src)),
            "stage3a_node_acc": s3a["acc"], "stage3a_link_auc": s3a["auc"],
            "stage3a_boundary_acc": s3a["bnd"], "stage3a_internal_acc": s3a["int"],
            "stage3b_node_acc": s3b["acc"] if s3b else None,
            "stage3b_link_auc": s3b["auc"] if s3b else None,
            "stage3b_boundary_acc": s3b["bnd"] if s3b else None,
            "stage3b_internal_acc": s3b["int"] if s3b else None,
            "invariants_passed": len(checks) - failed, "invariants_total": len(checks),
            # Provenance: a result that cannot name the code that produced it is
            # not reusable evidence.
            "git_sha": sha, "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                           time.gmtime()),
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"  appended result to {args.json_out}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

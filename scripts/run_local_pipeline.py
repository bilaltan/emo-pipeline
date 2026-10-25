#!/usr/bin/env python3
"""
Full LakeGRL pipeline on a single machine, against real Spark and Delta.

Runs Phase 0 (ingest) -> 1 (community detection) -> 2 (partition + abstraction)
-> 3b (unit training with CaaN macro context; --stage3a adds the no-context run) on a
small dataset, so the whole chain can be validated before it costs cluster time.
Results are written as one JSON record per run.

    JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
    PYSPARK_PYTHON=$PWD/.venv-spark/bin/python \
    PYSPARK_DRIVER_PYTHON=$PWD/.venv-spark/bin/python \
    .venv-spark/bin/python scripts/run_local_pipeline.py --alg lpa
"""
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

from phases.phase1_community import run_phase1
from phases.phase2_subgraph import run_phase2
from phases.phase3b_caan import run_phase3b
from phases.phase4_baselines import run_phase4
from experiment_config import TRAIN, PHASE3_MAX_NODES_PER_COMMUNITY
from pipeline.utils.store_io import parse_emu
from pipeline.utils.common import spark_resources

# LAKEGRL_STORE puts the Delta tables on a fast local file system (scripts/wsl_run.sh sets it).
STORE = os.path.abspath(os.environ.get("LAKEGRL_STORE", "data/.local_pipeline"))
# Raw datasets shared with the other projects (C:\data on Windows, /mnt/c/data in WSL).
DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt/c/data")
# Set by --link-split: edges held out before Phase 0 writes the training graph.
LINK_HOLDOUT = {"n_val": 0, "n_test": 0, "seed": 0}


def paths(dataset, alg=None):
    base = f"file://{STORE}/{dataset}"
    p = {"nodes": f"{base}/nodes/", "edges": f"{base}/edges/", "masks": f"{base}/masks/",
         "link_split": f"{base}/link_split/",
         "checkpoints": f"{STORE}/ckpt/{dataset}/"}
    if alg:
        p.update({"communities": f"{base}/communities/{alg}/",
                  "p2_nodes": f"{base}/p2_nodes/{alg}/",
                  "p2_edges": f"{base}/p2_edges/{alg}/",
                  "tag": f"local_{dataset}_{alg}"})
    return p


def _hold_out_links(src, dst, n_val, n_test, seed):
    """Sample val/test positives from the undirected edges (src < dst) and remove them.

    Every method in Track A (docs/trackA_link_prediction.md) is scored on these pairs, and
    Phases 1-4 only ever see the remaining training graph.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(src))
    held = perm[:n_val + n_test]
    keep = np.ones(len(src), dtype=bool)
    keep[held] = False
    split = np.array(["val"] * n_val + ["test"] * n_test)
    held_df = pd.DataFrame({"src": src[held].astype(np.int64), "dst": dst[held].astype(np.int64), "split": split})
    return src[keep], dst[keep], held_df


def _write_delta(spark, name, x, y, src, dst, seed=42, split=None, link_split=None):
    """Phase 0 equivalent: arrays -> transactional Delta tables.

    `split` is a per-node array of train/valid/test (the dataset's official split); without it a
    stratified 60/20/20 split is drawn. `link_split`, if given, is written as the link_split table.
    """
    n = x.shape[0]
    if link_split is None and LINK_HOLDOUT["n_val"] + LINK_HOLDOUT["n_test"] > 0:
        src, dst, link_split = _hold_out_links(src, dst, LINK_HOLDOUT["n_val"], LINK_HOLDOUT["n_test"],
                                               LINK_HOLDOUT["seed"])
        print(f"  Link split: held out {len(link_split):,} undirected edges "
              f"({LINK_HOLDOUT['n_val']:,} val / {LINK_HOLDOUT['n_test']:,} test, seed {LINK_HOLDOUT['seed']})")
    sym = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
    if split is None:
        rng = np.random.default_rng(seed)
        split = np.array(["none"] * n, dtype=object)
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            rng.shuffle(idx)
            a, b = int(0.6 * len(idx)), int(0.8 * len(idx))
            split[idx[:a]] = "train"; split[idx[a:b]] = "valid"; split[idx[b:]] = "test"

    # Go through Arrow/Parquet rather than a Python list of tuples. Coauthor-Physics
    # is 8,415-dim, so the tuple form materialises ~290M Python floats before Spark
    # sees a row, which stalls ingestion for hours on a graph of 34k vertices.
    import pyarrow as pa
    import pyarrow.parquet as pq
    import tempfile

    p = paths(name)
    staging = tempfile.mkdtemp(prefix=f"ingest_{name}_")

    # float32 matches Phase 0's FloatType schema and what the UDF casts to on
    # arrival. Writing float64 doubled every feature payload through Delta,
    # Arrow and the training frame — on 8,415-dim Coauthor-Physics that is
    # 67KB per vertex instead of 34KB. Built from one flat buffer: a Python list of
    # 2.4M row arrays (ogbn-products) costs several GB before Arrow sees it.
    x32 = np.ascontiguousarray(x, dtype=np.float32)
    feats = pa.FixedSizeListArray.from_arrays(pa.array(x32.reshape(-1)), x32.shape[1]).cast(pa.list_(pa.float32()))
    pq.write_table(pa.table({
        "id": pa.array(np.arange(n, dtype=np.int64)),
        "label": pa.array(y.astype(np.int64)),
        "features": feats,
    }), os.path.join(staging, "nodes.parquet"), row_group_size=50_000)
    del x32, feats
    pq.write_table(pa.table({
        "src": pa.array(sym[0].astype(np.int64)),
        "dst": pa.array(sym[1].astype(np.int64)),
    }), os.path.join(staging, "edges.parquet"))
    pq.write_table(pa.table({
        "id": pa.array(np.arange(n, dtype=np.int64)),
        "split": pa.array([str(v) for v in split]),
    }), os.path.join(staging, "masks.parquet"))

    targets = [("nodes.parquet", "nodes"), ("edges.parquet", "edges"), ("masks.parquet", "masks")]
    if link_split is not None:
        pq.write_table(pa.Table.from_pandas(link_split, preserve_index=False),
                       os.path.join(staging, "link_split.parquet"))
        targets.append(("link_split.parquet", "link_split"))
    for fname, target in targets:
        # overwriteSchema: existing tables were written as array<double>; a plain
        # overwrite keeps the old schema and fails the merge.
        (spark.read.parquet(f"file://{os.path.join(staging, fname)}")
              .write.format("delta").mode("overwrite")
              .option("overwriteSchema", "true").save(p[target]))

    shutil.rmtree(staging, ignore_errors=True)
    return n, sym.shape[1], int(y.max()) + 1, x.shape[1]


def ingest_ogb(spark, name, store=None):
    """ogbn-arxiv via OGB, using the repo's torch.load patch for cached artifacts."""
    from utils.common import _patch_torch_load
    _patch_torch_load()
    from ogb.nodeproppred import PygNodePropPredDataset
    d = PygNodePropPredDataset(name=name, root="data/ogb")[0]
    x = d.x.numpy().astype(np.float32)
    y = d.y.numpy().reshape(-1).astype(np.int64)
    ei = d.edge_index.numpy()
    keep = ei[0] < ei[1]
    return _write_delta(spark, store or name, x, y, ei[0][keep], ei[1][keep])


def ingest_reddit(spark, name="reddit"):
    """Reddit (Hamilton et al.) from the DGL-format npz files in DATA_ROOT/raw/reddit.

    The graph file already holds both directions (114.6M entries); only src < dst is kept and
    _write_delta symmetrizes again. node_types 1/2/3 are the official train/val/test split.
    """
    root = os.path.join(DATA_ROOT, "raw", "reddit")
    d = np.load(os.path.join(root, "reddit_data.npz"))
    g = np.load(os.path.join(root, "reddit_graph.npz"))
    x = d["feature"].astype(np.float32)
    y = d["label"].astype(np.int64)
    split = np.array(["none", "train", "valid", "test"], dtype=object)[d["node_types"]]
    row, col = g["row"].astype(np.int64), g["col"].astype(np.int64)
    keep = row < col
    src, dst = row[keep], col[keep]
    del row, col, g
    return _write_delta(spark, name, x, y, src, dst, split=split)


def ingest_ogb_raw(spark, name, ogb_name):
    """OGB node datasets from DATA_ROOT/raw/ogb with the library-agnostic loader and official split.

    ogbn-products and ogbn-arxiv are stored in OGB's generic format (processed/data_processed), so no PyG or DGL
    conversion is needed. Edges become unique undirected pairs (arxiv's citations are directed).
    """
    from utils.common import _patch_torch_load
    _patch_torch_load()
    from ogb.nodeproppred import NodePropPredDataset
    ds = NodePropPredDataset(name=ogb_name, root=os.path.join(DATA_ROOT, "raw", "ogb"))
    graph, label = ds[0]
    n = graph["num_nodes"]
    split = np.array(["none"] * n, dtype=object)
    for k, v in ds.get_idx_split().items():
        split[v] = {"train": "train", "valid": "valid", "test": "test"}[k]
    ei = graph["edge_index"]
    lo, hi = np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1])
    und = np.unique(np.stack([lo[lo != hi], hi[lo != hi]]), axis=1)
    return _write_delta(spark, name, graph["node_feat"], label.reshape(-1).astype(np.int64),
                        und[0].astype(np.int64), und[1].astype(np.int64), split=split)


def ingest_pyg(spark, name, store=None):
    """WikiCS / Coauthor-CS / Coauthor-Physics via PyTorch Geometric."""
    if name == "WikiCS":
        from torch_geometric.datasets import WikiCS
        d = WikiCS(root="data/wikics")[0]
    elif name == "Coauthor-CS":
        from torch_geometric.datasets import Coauthor
        d = Coauthor(root="data/coauthor_cs", name="CS")[0]
    elif name == "Coauthor-Physics":
        from torch_geometric.datasets import Coauthor
        d = Coauthor(root="data/coauthor_physics", name="Physics")[0]
    else:
        raise ValueError(name)
    x = d.x.numpy().astype(np.float32)
    y = d.y.numpy().astype(np.int64)
    ei = d.edge_index.numpy()
    keep = ei[0] < ei[1]
    return _write_delta(spark, store or name, x, y, ei[0][keep], ei[1][keep])


def ingest_deezer(spark, root="data/deezer_europe_extracted/deezer_europe", feat_dim=128, store=None):
    """Phase 0 equivalent: raw files -> transactional Delta tables."""
    edges = pd.read_csv(os.path.join(root, "deezer_europe_edges.csv"))
    target = pd.read_csv(os.path.join(root, "deezer_europe_target.csv"))
    with open(os.path.join(root, "deezer_europe_features.json")) as fh:
        raw = json.load(fh)

    n = int(target["id"].max()) + 1
    y = np.zeros(n, dtype=np.int64)
    y[target["id"].values.astype(int)] = target["target"].values.astype(int)

    x = np.zeros((n, feat_dim), dtype=np.float32)
    for node, feats in raw.items():
        i = int(node)
        if i < n:
            for f in feats:
                x[i, int(f) % feat_dim] += 1.0
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.where(norms > 0, norms, 1.0)

    return _write_delta(spark, store or "DeezerEurope", x, y,
                        edges["node_1"].values.astype(np.int64),
                        edges["node_2"].values.astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="DeezerEurope",
                    choices=["DeezerEurope", "WikiCS", "Coauthor-CS",
                             "Coauthor-Physics", "ogbn-arxiv", "reddit", "ogbn-products"])
    # Track A: hold out N val and N test undirected edges before Phase 0; tables are stored
    # as <dataset>-lp so they never mix with the full graph's tables.
    ap.add_argument("--link-split", type=int, default=0, metavar="N")
    ap.add_argument("--link-seed", type=int, default=0)
    # Phase 4's link baseline uses its own split, so it is opt-in; baselines have their own scripts.
    ap.add_argument("--stop-after", default="3b", choices=["2", "3", "3b", "4"],
                    help="last phase to run (Track A runs Phases 0-2 once, then its own harness)")
    # drop (CaaN, default): communities below --min-size leave the units and Stage 3b scores their nodes with
    # CaaN's global model (super-nodes + minor nodes); misc packs them into shared bins; keep makes each a unit.
    ap.add_argument("--tiny-comm", default="drop", choices=["drop", "misc", "keep"])
    ap.add_argument("--minor-score", default="global", choices=["global", "unit"],
                    help="who scores minor-community nodes in stage3b_node_acc: CaaN's global model or the unit "
                         "they have most edges into (both are recorded)")
    ap.add_argument("--merge-minor", action="store_true",
                    help="Phase 1 moves small-community vertices into a major neighbour community (not in CaaN)")
    ap.add_argument("--reuse-phase1", action="store_true", help="keep an existing communities table")
    ap.add_argument("--alg", default="louvain", choices=["lpa", "louvain"])
    ap.add_argument("--seed", type=int, default=0, help="training seed for Phases 3, 3b and 4")
    # Per-unit link AUC is split inside each unit and is not comparable with the baseline; links are
    # scored on the held-out split by experiments/trackA/harness.py.
    ap.add_argument("--task", default="node_classification", choices=["node_classification", "both"])
    # Stage 3b (CaaN macro context) is the method; Stage 3a (no context) runs only when asked, as an ablation.
    ap.add_argument("--stage3a", action="store_true", help="also run Stage 3a (units without context)")
    ap.add_argument("--stage3a-halo", action="store_true", help="run Stage 3a with messages from the 1-hop halo")
    ap.add_argument("--model", default="sage",
                    choices=["sage", "gatv2", "gat", "transformer", "arma", "asap"])
    ap.add_argument("--min-size", type=int, default=100)
    ap.add_argument("--max-nodes", type=int, default=PHASE3_MAX_NODES_PER_COMMUNITY)
    # Small datasets exist to be run whole. The production per-unit edge cap
    # (30k) silently subsamples dense communities, so raise it above anything
    # these graphs contain and assert afterwards that nothing was capped.
    ap.add_argument("--max-edges", type=int, default=5_000_000)
    # Training settings come from experiment_config.TRAIN; these flags exist only for explicit sweeps.
    ap.add_argument("--epochs", type=int, default=TRAIN["epochs"])
    ap.add_argument("--lr", type=float, default=TRAIN["lr"])
    ap.add_argument("--node-patience", type=int, default=None, help="default: no early stopping")
    ap.add_argument("--probe-head", type=int, choices=[0, 1], default=None, help="override TRAIN['probe_head']")
    # Local mode has no executors: local[N] is N worker threads in the driver
    # JVM. Units train one-per-core, so N sets how many run concurrently.
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--driver-mem", default="8g")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--force-ingest", action="store_true",
                    help="rebuild Delta tables even if they already exist")
    ap.add_argument("--json-out", default="results/local_pipeline_runs.jsonl")
    # materialized: p2_nodes copies features (LakeGRL default, a materialized view).
    # normalized: p2_nodes holds ids + community only; features are joined back.
    ap.add_argument("--partition-table", default="materialized",
                    choices=["materialized", "normalized"])
    # Phase 3b: direct per-bin reads of a materialized p2_nodes, or a global Spark
    # join + shuffle (required for a normalized p2_nodes). auto picks by table form.
    ap.add_argument("--node-loading", default="auto",
                    choices=["auto", "direct", "global_join"])
    # Physical layout of the Phase 2 tables: range (default), hive (partitionBy
    # community_id) or hive_merged (small communities packed into shared partitions).
    ap.add_argument("--p2-layout", default="range", choices=["range", "hive", "hive_merged"])
    ap.add_argument("--merge-target-rows", type=int, default=250_000)
    # Track B storage layout of the Phase 2 tables (physical only: every unit reads the same rows).
    # Row group in MB, or auto = the sqrt rule from table size and unit count (0: Spark's 128 MB default).
    ap.add_argument("--p2-rg-mb", default="auto")
    ap.add_argument("--p2-plain", action="store_true",
                    help="Phase 2's order without _blk (block-ordered for --max-nodes by default)")
    ap.add_argument("--s3-emu", default=None, metavar="LAT_MS,MBPS|count",
                    help="Stage 3b reads the Phase 2 tables through an emulated S3 (e.g. 20,80: 20 ms + 80 MB/s "
                         "per GET), or 'count' to count GETs and bytes only (utils/store_io.py)")
    args = ap.parse_args()

    builder = (SparkSession.builder.appName("lakegrl-local-pipeline").master(f"local[{args.cores}]")
               .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
               .config("spark.sql.catalog.spark_catalog",
                       "org.apache.spark.sql.delta.catalog.DeltaCatalog")
               .config("spark.driver.memory", args.driver_mem)
               .config("spark.sql.shuffle.partitions", str(args.cores * 2))
               .config("spark.sql.execution.arrow.pyspark.enabled", "true")
               # Wide feature vectors (Coauthor-CS is 6,805-dim) overflow the Parquet
               # vectorized reader, which reserves a contiguous batch of 4,096 rows by
               # default: 4096 x 6805 x 4B is ~111MB per column batch and OOMs the heap.
               .config("spark.sql.parquet.columnarReaderBatchSize", "512")
               .config("spark.sql.execution.arrow.maxRecordsPerBatch", "500"))
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    sc = spark.sparkContext

    # Read at the start: a commit made while the run is going must not relabel it.
    import subprocess
    _git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True,
                                     cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).stdout.strip()
    # core.autocrlf=true as on the Windows side that owns the checkout; WSL's git would call every CRLF file modified
    git_hash = _git("rev-parse", "--short", "HEAD") + (
        "-dirty" if _git("-c", "core.autocrlf=true", "status", "--porcelain", "--untracked-files=no") else "")
    t_all = time.time()
    DS = args.dataset
    if args.link_split > 0:
        LINK_HOLDOUT.update(n_val=args.link_split, n_test=args.link_split, seed=args.link_seed)
        DS = f"{args.dataset}-lp"
    print(f"\n{'='*70}\n  LakeGRL full pipeline — {DS} / {args.alg} / {args.model}\n{'='*70}")

    t0 = time.time()
    # Phase 0 output is a function of the dataset alone, so re-deriving it for each
    # partitioner is pure repetition. Reuse the tables when they already hold the
    # right graph; the pipeline's own phases already checkpoint this way.
    def _existing_tables():
        try:
            p = paths(DS)
            n = spark.read.format("delta").load(p["nodes"])
            e = spark.read.format("delta").load(p["edges"])
            spark.read.format("delta").load(p["masks"]).limit(1).count()
            nn = n.count()
            feat_dim = len(n.select("features").first()[0])
            n_classes = int(n.agg({"label": "max"}).first()[0]) + 1
            return nn, e.count(), n_classes, feat_dim
        except Exception:
            return None

    cached = None if args.force_ingest else _existing_tables()
    if cached:
        n_nodes, n_edges, n_classes, feat_dim = cached
        print(f"  Phase 0: reusing existing Delta tables ({n_nodes:,} nodes, "
              f"{n_edges:,} edges)  [0.0s]")
    elif args.dataset == "DeezerEurope":
        n_nodes, n_edges, n_classes, feat_dim = ingest_deezer(spark, store=DS)
    elif args.dataset == "reddit":
        n_nodes, n_edges, n_classes, feat_dim = ingest_reddit(spark, DS)
    elif args.dataset in ("ogbn-products", "ogbn-arxiv"):
        n_nodes, n_edges, n_classes, feat_dim = ingest_ogb_raw(spark, DS, args.dataset)
    elif args.dataset.startswith("ogbn-"):
        n_nodes, n_edges, n_classes, feat_dim = ingest_ogb(spark, args.dataset, store=DS)
    else:
        n_nodes, n_edges, n_classes, feat_dim = ingest_pyg(spark, args.dataset, store=DS)
    t_p0 = time.time() - t0
    if not cached:
        print(f"  Phase 0: {n_nodes:,} nodes, {n_edges:,} directed edges -> Delta  [{t_p0:.1f}s]")

    ds_cfg = {DS: {"in_feats": feat_dim, "num_classes": n_classes}}
    args.node_patience = args.node_patience or args.epochs
    train = dict(TRAIN, epochs=args.epochs, lr=args.lr, seed=args.seed, node_patience=args.node_patience)
    if args.probe_head is not None:
        train["probe_head"] = bool(args.probe_head)
    gcn_cfg = {"hidden_dim": train["hidden_dim"], "num_epochs": train["epochs"], "lr": train["lr"],
               "dropout": train["dropout"], "weight_decay": train["weight_decay"], "seed": train["seed"],
               "probe_head": train["probe_head"]}
    timing, r1, r2, r3, r3b, r4 = {}, {}, {}, {}, {}, {}

    run_phase1(spark, sc, [DS], [args.alg], lpa_max_iter=20, resolution=1.0,
               random_seed=42, min_size=args.min_size, dataset_cfg=ds_cfg,
               get_paths_fn=paths, timing=timing, results=r1, force_rerun=not args.reuse_phase1,
               lpa_tol=0.001, merge_minor_communities=args.merge_minor)

    run_phase2(spark, sc, [DS], [args.alg], use_global_mapping=True,
               min_size=args.min_size, get_paths_fn=paths, timing=timing, results=r2,
               expand_boundary_nodes=True, tiny_comm_handling=args.tiny_comm, force_rerun=True,
               partition_table_mode=args.partition_table, p2_layout=args.p2_layout,
               merge_target_rows=args.merge_target_rows,
               rg_mb=args.p2_rg_mb if args.p2_rg_mb == "auto" else float(args.p2_rg_mb) or None,
               block_cap=None if args.p2_plain else args.max_nodes)

    if args.stop_after != "4":
        args.skip_baseline = True

    # Both stages run the same unit code (Phase 3b's) on the same units and differ only in context: Stage 3a
    # sees member-member edges (or the halo), Stage 3b the macro-graph. No try/except: a failed stage fails the run.
    stage_kw = dict(use_global_mapping=True, dataset_cfg=ds_cfg, gcn_cfg=gcn_cfg, get_paths_fn=paths,
                    models=[args.model], task_type=args.task, min_size=args.min_size, force_rerun=True,
                    local_data_dir=STORE, node_loading=args.node_loading, max_nodes_per_community=args.max_nodes,
                    node_patience=args.node_patience, tiny_comm_handling=args.tiny_comm,
                    s3_emu=parse_emu(args.s3_emu), minor_score=args.minor_score)
    if args.stop_after != "2" and (args.stage3a or args.stage3a_halo):
        t3 = {}
        run_phase3b(spark, sc, [DS], [args.alg], timing=t3, results=r3,
                    context="halo" if args.stage3a_halo else "none", **stage_kw)
        timing.update({("phase3",) + kk[1:]: vv for kk, vv in t3.items()})

    if args.stop_after in ("3b", "4"):
        run_phase3b(spark, sc, [DS], [args.alg], timing=timing, results=r3b, context="macro", **stage_kw)

    # The full-graph baseline is independent of Phase 1's algorithm — running it once
    # per (dataset, partitioner) computes the identical result twice.
    base_cache = os.path.join(STORE, "baseline_cache.json")
    base_key = f"{DS}|seed={args.seed}"
    cached_base = {}
    if os.path.exists(base_cache):
        try:
            cached_base = json.load(open(base_cache))
        except Exception:
            cached_base = {}

    if not args.skip_baseline and base_key in cached_base:
        r4[DS] = cached_base[base_key]
        print(f"  Phase 4: reusing cached full-graph baseline for {base_key}")
    elif not args.skip_baseline:
        run_phase4(spark, sc, [DS], ds_cfg,
                   {"epochs": 100, "batch": 1024, "fanout": [15, 10], "lr": 0.001,
                    "hidden_dim": 256, "dropout": 0.5, "link_epochs": 300,
                    "link_patience": 30, "link_val_frac": 0.10, "link_test_frac": 0.10,
                    "node_epochs": 100, "node_patience": 10},
                   paths, timing, r4, task_type=args.task, n_runs=1)
        if r4.get(DS):
            cached_base[base_key] = {k: v for k, v in r4[DS].items()
                                     if isinstance(v, (int, float, str, type(None)))}
            os.makedirs(os.path.dirname(base_cache), exist_ok=True)
            json.dump(cached_base, open(base_cache, "w"), indent=1)

    def attrs(store, key):
        d = store.get(key)
        return dict(d.attrs) if d is not None and hasattr(d, "attrs") else {}

    k = (DS, args.alg, args.model)
    a3, a3b = attrs(r3, k), attrs(r3b, k)
    base = r4.get(DS, {}) or {}

    # Coverage: test nodes scored / test nodes in the masks. Below 1.0 the accuracy is over part of the set.
    n_test_all = spark.read.format("delta").load(paths(DS)["masks"]).filter("split = 'test'").count()
    def scored(store):
        d = store.get(k)
        return int(d["n_test"].sum()) if d is not None and "n_test" in getattr(d, "columns", []) else None

    record = {
        "dataset": DS, "algorithm": args.alg, "model": args.model, "epochs": args.epochs,
        "seed": args.seed, "train": train, "task": args.task, "stage3a_halo": args.stage3a_halo, "tiny_comm": args.tiny_comm, "merge_minor": args.merge_minor,
        "min_size": args.min_size, "max_nodes": args.max_nodes, "git": git_hash,
        "n_test_all": n_test_all, "stage3a_n_test_scored": scored(r3), "stage3b_n_test_scored": scored(r3b),
        "partition_table": args.partition_table, "node_loading": args.node_loading,
        "p2_layout": args.p2_layout, "p2_rg_mb": r2.get((DS, args.alg), {}).get("rg_mb"),
        "p2_block_cap": r2.get((DS, args.alg), {}).get("block_cap"),
        "n_nodes": n_nodes, "n_edges": n_edges, "n_classes": n_classes,
        "n_communities_phase1": r1.get((DS, args.alg), {}).get("n_comms"),
        "n_units_phase3": len(r3[k]) if k in r3 else None,
        "n_units_stage3b": len(r3b[k]) if k in r3b else None,
        "stage3a_node_acc": a3.get("weighted_comm_acc"),
        "stage3a_link_auc": a3.get("weighted_comm_link_auc"),
        "stage3b_node_acc": a3b.get("weighted_comm_acc"),
        "stage3b_node_val_acc": a3b.get("weighted_comm_val_acc"), "minor_score": a3b.get("minor_score"),
        **{f"stage3b_{kk}": a3b.get(kk) for kk in ("acc_minor_global", "acc_minor_unit", "val_acc_minor_global",
                                                    "val_acc_minor_unit", "minor_test_reached")},
        "stage3b_link_auc": a3b.get("weighted_comm_link_auc"),
        "cores": args.cores, "s3_emu": args.s3_emu, "stage3b_read_s": a3b.get("read_s"), "stage3b_read_gets": a3b.get("read_gets"),
        "stage3b_read_bytes": a3b.get("read_bytes"),
        "baseline_node_acc": base.get("test_acc"),
        "baseline_link_auc": base.get("link_auc"),
        "phase_seconds": {str(kk): round(vv, 2) for kk, vv in timing.items()},
        "total_seconds": round(time.time() - t_all, 1),
        **spark_resources(sc),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    print(f"\n{'-'*70}\n  PIPELINE RESULT\n{'-'*70}")
    for label, key in [("Phase 1 communities", "n_communities_phase1"),
                       ("Phase 3 units", "n_units_phase3"),
                       ("Stage 3a node acc", "stage3a_node_acc"),
                       ("Stage 3a link AUC", "stage3a_link_auc"),
                       ("Stage 3b node acc", "stage3b_node_acc"),
                       ("Stage 3b link AUC", "stage3b_link_auc"),
                       ("Baseline node acc", "baseline_node_acc"),
                       ("Baseline link AUC", "baseline_link_auc")]:
        v = record[key]
        shown = f"{v:.4f}" if isinstance(v, float) else ("—" if v is None else v)
        print(f"  {label:<24} {shown}")
    print(f"  {'total wall clock':<24} {record['total_seconds']}s")
    for stage in ("stage3a", "stage3b"):
        n = record[f"{stage}_n_test_scored"]
        if n is not None and n != n_test_all:
            print(f"  WARNING {stage} scored {n:,} of {n_test_all:,} test nodes")

    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(f"\n  appended to {args.json_out}")

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

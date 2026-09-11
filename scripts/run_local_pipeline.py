#!/usr/bin/env python3
"""
Full LakeGRL pipeline on a single machine, against real Spark and Delta.

Runs Phase 0 (ingest) -> 1 (community detection) -> 2 (partition + abstraction)
-> 3 (decoupled training) -> 3b (CAAN fusion) -> 4 (full-graph baseline) on a
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
from phases.phase3_training import run_phase3
from phases.phase3b_caan import run_phase3b
from phases.phase4_baselines import run_phase4

STORE = os.path.abspath("data/.local_pipeline")


def paths(dataset, alg=None):
    base = f"file://{STORE}/{dataset}"
    p = {"nodes": f"{base}/nodes/", "edges": f"{base}/edges/", "masks": f"{base}/masks/",
         "checkpoints": f"{STORE}/ckpt/{dataset}/"}
    if alg:
        p.update({"communities": f"{base}/communities/{alg}/",
                  "p2_nodes": f"{base}/p2_nodes/{alg}/",
                  "p2_edges": f"{base}/p2_edges/{alg}/",
                  "tag": f"local_{dataset}_{alg}"})
    return p


def _write_delta(spark, name, x, y, src, dst, seed=42):
    """Phase 0 equivalent: arrays -> transactional Delta tables with a stratified split."""
    n = x.shape[0]
    sym = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
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

    pq.write_table(pa.table({
        "id": pa.array(np.arange(n, dtype=np.int64)),
        "label": pa.array(y.astype(np.int64)),
        # float32 matches Phase 0's FloatType schema and what the UDF casts to on
        # arrival. Writing float64 doubled every feature payload through Delta,
        # Arrow and the training frame — on 8,415-dim Coauthor-Physics that is
        # 67KB per vertex instead of 34KB.
        "features": pa.array(list(x.astype(np.float32)),
                             type=pa.list_(pa.float32())),
    }), os.path.join(staging, "nodes.parquet"))
    pq.write_table(pa.table({
        "src": pa.array(sym[0].astype(np.int64)),
        "dst": pa.array(sym[1].astype(np.int64)),
    }), os.path.join(staging, "edges.parquet"))
    pq.write_table(pa.table({
        "id": pa.array(np.arange(n, dtype=np.int64)),
        "split": pa.array([str(v) for v in split]),
    }), os.path.join(staging, "masks.parquet"))

    for fname, target in (("nodes.parquet", "nodes"), ("edges.parquet", "edges"),
                          ("masks.parquet", "masks")):
        # overwriteSchema: existing tables were written as array<double>; a plain
        # overwrite keeps the old schema and fails the merge.
        (spark.read.parquet(f"file://{os.path.join(staging, fname)}")
              .write.format("delta").mode("overwrite")
              .option("overwriteSchema", "true").save(p[target]))

    shutil.rmtree(staging, ignore_errors=True)
    return n, sym.shape[1], int(y.max()) + 1, x.shape[1]


def ingest_ogb(spark, name):
    """ogbn-arxiv via OGB, using the repo's torch.load patch for cached artifacts."""
    from utils.common import _patch_torch_load
    _patch_torch_load()
    from ogb.nodeproppred import PygNodePropPredDataset
    d = PygNodePropPredDataset(name=name, root="data/ogb")[0]
    x = d.x.numpy().astype(np.float32)
    y = d.y.numpy().reshape(-1).astype(np.int64)
    ei = d.edge_index.numpy()
    keep = ei[0] < ei[1]
    return _write_delta(spark, name, x, y, ei[0][keep], ei[1][keep])


def ingest_pyg(spark, name):
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
    return _write_delta(spark, name, x, y, ei[0][keep], ei[1][keep])


def ingest_deezer(spark, root="data/deezer_europe_extracted/deezer_europe", feat_dim=128):
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

    return _write_delta(spark, "DeezerEurope", x, y,
                        edges["node_1"].values.astype(np.int64),
                        edges["node_2"].values.astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="DeezerEurope",
                    choices=["DeezerEurope", "WikiCS", "Coauthor-CS",
                             "Coauthor-Physics", "ogbn-arxiv"])
    ap.add_argument("--alg", default="lpa", choices=["lpa", "louvain"])
    ap.add_argument("--model", default="sage",
                    choices=["sage", "gatv2", "gat", "transformer", "arma", "asap"])
    ap.add_argument("--min-size", type=int, default=100)
    ap.add_argument("--max-nodes", type=int, default=10000)
    # Small datasets exist to be run whole. The production per-unit edge cap
    # (30k) silently subsamples dense communities, so raise it above anything
    # these graphs contain and assert afterwards that nothing was capped.
    ap.add_argument("--max-edges", type=int, default=5_000_000)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--node-patience", type=int, default=20)
    # Local mode has no executors: local[N] is N worker threads in the driver
    # JVM. Units train one-per-core, so N sets how many run concurrently.
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--driver-mem", default="8g")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--force-ingest", action="store_true",
                    help="rebuild Delta tables even if they already exist")
    ap.add_argument("--json-out", default="results/local_pipeline_runs.jsonl")
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

    t_all = time.time()
    DS = args.dataset
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
    elif DS == "DeezerEurope":
        n_nodes, n_edges, n_classes, feat_dim = ingest_deezer(spark)
    elif DS.startswith("ogbn-"):
        n_nodes, n_edges, n_classes, feat_dim = ingest_ogb(spark, DS)
    else:
        n_nodes, n_edges, n_classes, feat_dim = ingest_pyg(spark, DS)
    t_p0 = time.time() - t0
    if not cached:
        print(f"  Phase 0: {n_nodes:,} nodes, {n_edges:,} directed edges -> Delta  [{t_p0:.1f}s]")

    ds_cfg = {DS: {"in_feats": feat_dim, "num_classes": n_classes}}
    gcn_cfg = {"hidden_dim": 256, "num_epochs": args.epochs, "lr": 0.001, "dropout": 0.5}
    timing, r1, r2, r3, r3b, r4 = {}, {}, {}, {}, {}, {}

    run_phase1(spark, sc, [DS], [args.alg], lpa_max_iter=20, resolution=1.0,
               random_seed=42, min_size=args.min_size, dataset_cfg=ds_cfg,
               get_paths_fn=paths, timing=timing, results=r1, force_rerun=True,
               lpa_tol=0.001, merge_minor_communities=True)

    run_phase2(spark, sc, [DS], [args.alg], use_global_mapping=True,
               min_size=args.min_size, get_paths_fn=paths, timing=timing, results=r2,
               expand_boundary_nodes=True, tiny_comm_handling="drop", force_rerun=True)

    run_phase3(spark, sc, [DS], [args.alg], use_global_mapping=True,
               dataset_cfg=ds_cfg, gcn_cfg=gcn_cfg, get_paths_fn=paths,
               timing=timing, results=r3, models=[args.model], task_type="both",
               force_rerun=True, max_nodes_per_community=args.max_nodes,
               max_edges_per_community=args.max_edges, edge_sample_modulus=1, mlp_epochs=150, mlp_patience=15,
               local_data_dir=STORE, block_oversized=True,
               node_patience=args.node_patience)

    try:
        run_phase3b(spark, sc, [DS], [args.alg], use_global_mapping=True,
                    dataset_cfg=ds_cfg, gcn_cfg=gcn_cfg, get_paths_fn=paths,
                    timing=timing, results=r3b, models=[args.model], task_type="both",
                    min_size=args.min_size, force_rerun=True, local_data_dir=STORE)
    except Exception as e:
        print(f"  [Phase 3b] failed: {type(e).__name__}: {str(e)[:200]}")

    # The full-graph baseline is independent of Phase 1's algorithm — running it once
    # per (dataset, partitioner) computes the identical result twice.
    base_cache = os.path.join(STORE, "baseline_cache.json")
    cached_base = {}
    if os.path.exists(base_cache):
        try:
            cached_base = json.load(open(base_cache))
        except Exception:
            cached_base = {}

    if not args.skip_baseline and DS in cached_base:
        r4[DS] = cached_base[DS]
        print(f"  Phase 4: reusing cached full-graph baseline for {DS}")
    elif not args.skip_baseline:
        try:
            run_phase4(spark, sc, [DS], ds_cfg,
                       {"epochs": 100, "batch": 1024, "fanout": [15, 10], "lr": 0.001,
                        "hidden_dim": 256, "dropout": 0.5, "link_epochs": 300,
                        "link_patience": 30, "link_val_frac": 0.10, "link_test_frac": 0.10,
                        "node_epochs": 100, "node_patience": 10},
                       paths, timing, r4, task_type="both", n_runs=1)
        except Exception as e:
            print(f"  [Phase 4] failed: {type(e).__name__}: {str(e)[:200]}")
        if r4.get(DS):
            cached_base[DS] = {k: v for k, v in r4[DS].items()
                               if isinstance(v, (int, float, str, type(None)))}
            os.makedirs(os.path.dirname(base_cache), exist_ok=True)
            json.dump(cached_base, open(base_cache, "w"), indent=1)

    def attrs(store, key):
        d = store.get(key)
        return dict(d.attrs) if d is not None and hasattr(d, "attrs") else {}

    k = (DS, args.alg, args.model)
    a3, a3b = attrs(r3, k), attrs(r3b, k)
    base = r4.get(DS, {}) or {}

    record = {
        "dataset": DS, "algorithm": args.alg, "model": args.model, "epochs": args.epochs,
        "n_nodes": n_nodes, "n_edges": n_edges, "n_classes": n_classes,
        "n_communities_phase1": r1.get((DS, args.alg), {}).get("n_comms"),
        "n_units_phase3": a3.get("n_communities"),
        "max_edges_cap": args.max_edges,
        "edges_retained_phase3": a3.get("total_edges"),
        "stage3a_node_acc": a3.get("weighted_comm_acc"),
        "stage3a_acc_gnn_head": a3.get("acc_gnn_head"),
        "stage3a_acc_mlp_head": a3.get("acc_mlp_head"),
        "stage3a_head_used": a3.get("head_used"),
        "stage3a_n_head_gnn": a3.get("n_head_gnn"),
        "stage3a_n_head_mlp": a3.get("n_head_mlp"),
        "stage3a_link_auc": a3.get("weighted_comm_link_auc"),
        "stage3b_node_acc": a3b.get("weighted_comm_acc"),
        "stage3b_link_auc": a3b.get("weighted_comm_link_auc"),
        "baseline_node_acc": base.get("test_acc"),
        "baseline_link_auc": base.get("link_auc"),
        "phase_seconds": {str(kk): round(vv, 2) for kk, vv in timing.items()},
        "total_seconds": round(time.time() - t_all, 1),
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

    os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
    with open(args.json_out, "a") as fh:
        fh.write(json.dumps(record) + "\n")
    print(f"\n  appended to {args.json_out}")

    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

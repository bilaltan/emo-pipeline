#!/usr/bin/env python3
"""
Track A: export one dataset's Delta tables (Phases 0-2 output) to a numpy cache.

Every Track A method then trains and scores from the same arrays, so differences between methods
come from the method, not from data loading. Reads the current Delta snapshot's Parquet files
directly with pyarrow (Spark only lists the files), which avoids a toPandas() of the feature column.

    bash scripts/wsl_run.sh python experiments/trackA/export_cache.py --dataset reddit-lp --alg louvain

Writes $LAKEGRL_STORE/<dataset>/trackA_<alg>/:
  x.npy        [N, F] float32 node features
  y.npy        [N] int64 labels
  edges.npy    [2, E] int64 undirected training edges, src < dst (held-out links already removed)
  comm.npy     [N] int64 community (Phase 2 unit) of each member vertex, or INT64_MIN if none
  halo.npy     [2, H] int64 (vertex, community) rows of Phase 2's 1-hop halo (is_member = False)
  split.npy    [N] int8 node split from the masks table: 0 train, 1 valid, 2 test, -1 none
  links.npy    [3, L] int64 (src, dst, split) held-out positives, split 0 = val, 1 = test
  meta.json    sizes, source table versions and file counts
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

NO_COMM = np.iinfo(np.int64).min


def snapshot_files(spark, url):
    """Current Delta snapshot: (local file paths, table version)."""
    from delta.tables import DeltaTable
    df = spark.read.format("delta").load(url)
    files = [f.replace("file://", "").replace("file:", "") for f in df.inputFiles()]
    version = DeltaTable.forPath(spark, url).history(1).select("version").first()[0]
    return files, int(version)


def read_columns(files, columns):
    return pa.concat_tables([pq.read_table(f, columns=columns) for f in files]).combine_chunks()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="store name, e.g. reddit-lp")
    ap.add_argument("--alg", default="louvain")
    args = ap.parse_args()

    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession
    from scripts.run_local_pipeline import STORE, paths
    from pipeline.utils.common import spark_resources

    spark = configure_spark_with_delta_pip(
        SparkSession.builder.appName("trackA-export").master("local[2]")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.driver.memory", "4g")).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    p = paths(args.dataset, args.alg)
    out = os.path.join(STORE, args.dataset, f"trackA_{args.alg}")
    os.makedirs(out, exist_ok=True)
    meta = {"dataset": args.dataset, "alg": args.alg, "tables": {}}
    t0 = time.time()

    # nodes: id is 0..N-1 (written that way by run_local_pipeline), sort to be safe
    files, ver = snapshot_files(spark, p["nodes"])
    meta["tables"]["nodes"] = {"version": ver, "files": len(files)}
    t = read_columns(files, ["id", "label", "features"])
    order = np.argsort(t.column("id").to_numpy())
    n = len(order)
    assert np.array_equal(t.column("id").to_numpy()[order], np.arange(n)), "node ids are not 0..N-1"
    feats = t.column("features")
    dim = len(feats[0].as_py())
    flat = pa.concat_arrays([c.flatten() for c in feats.chunks]) if hasattr(feats, "chunks") else feats.flatten()
    x = flat.to_numpy(zero_copy_only=False).astype(np.float32).reshape(n, dim)[order]
    np.save(os.path.join(out, "x.npy"), x)
    np.save(os.path.join(out, "y.npy"), t.column("label").to_numpy()[order].astype(np.int64))
    del t, feats, flat, x

    # edges: both directions are stored; keep src < dst
    files, ver = snapshot_files(spark, p["edges"])
    meta["tables"]["edges"] = {"version": ver, "files": len(files)}
    t = read_columns(files, ["src", "dst"])
    s, d = t.column("src").to_numpy(), t.column("dst").to_numpy()
    keep = s < d
    edges = np.unique(np.stack([s[keep], d[keep]]).astype(np.int64), axis=1)
    np.save(os.path.join(out, "edges.npy"), edges)
    del t, s, d

    # Phase 2 partition table: members give the community, non-members are the halo
    files, ver = snapshot_files(spark, p["p2_nodes"])
    meta["tables"]["p2_nodes"] = {"version": ver, "files": len(files)}
    t = read_columns(files, ["id", "community_id", "is_member"])
    ids = t.column("id").to_numpy().astype(np.int64)
    cid = t.column("community_id").to_numpy().astype(np.int64)
    mem = t.column("is_member").to_numpy(zero_copy_only=False).astype(bool)
    comm = np.full(n, NO_COMM, dtype=np.int64)
    comm[ids[mem]] = cid[mem]
    assert len(np.unique(ids[mem])) == mem.sum(), "a vertex is a member of more than one unit"
    np.save(os.path.join(out, "comm.npy"), comm)
    np.save(os.path.join(out, "halo.npy"), np.stack([ids[~mem], cid[~mem]]))

    # node split, the same masks table Phase 3b reads (node baselines train and score on it)
    files, ver = snapshot_files(spark, p["masks"])
    meta["tables"]["masks"] = {"version": ver, "files": len(files)}
    t = read_columns(files, ["id", "split"])
    code = {"train": 0, "valid": 1, "test": 2}
    split = np.full(n, -1, dtype=np.int8)
    split[t.column("id").to_numpy()] = [code.get(v, -1) for v in t.column("split").to_pylist()]
    np.save(os.path.join(out, "split.npy"), split)

    files, ver = snapshot_files(spark, p["link_split"])
    meta["tables"]["link_split"] = {"version": ver, "files": len(files)}
    t = read_columns(files, ["src", "dst", "split"])
    sp = np.array([0 if v == "val" else 1 for v in t.column("split").to_pylist()], dtype=np.int64)
    np.save(os.path.join(out, "links.npy"),
            np.stack([t.column("src").to_numpy(), t.column("dst").to_numpy(), sp]).astype(np.int64))

    meta.update(n_nodes=int(n), feat_dim=int(dim), n_edges_undirected=int(edges.shape[1]),
                n_members=int(mem.sum()), n_halo_rows=int((~mem).sum()),
                n_units=int(len(np.unique(cid[mem]))), n_links=int(len(sp)),
                export_seconds=round(time.time() - t0, 1), spark=spark_resources(spark.sparkContext))
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))
    spark.stop()


if __name__ == "__main__":
    main()

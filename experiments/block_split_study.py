#!/usr/bin/env python3
"""
Graph-aware splitting of oversized training units (Phase 3b's bounded blocks).

Phase 3b bounds every training unit at max_nodes_per_community rows (default
10,000). A larger community is cut into n = 2^k blocks by `id & (n-1)`. That
cut ignores the graph: an edge survives only if both endpoints land in the same
block, and every block re-reads its whole community, because the predicate
cannot be answered from parquet min/max statistics.

Strategies (--strategies), all under the same cap on rows per unit:

  id_pow2   Phase 3b today: id & (n-1), n = smallest power of two >= rows/cap
  id_equal  control: id % m, m = number of units the first graph strategy made,
            so it differs from that strategy only in how rows are grouped
  louvain   community detection on the unit's own subgraph (ML-GRL-style
  leiden    multi-level partitioning); clusters still above the cap are split
  lpa       again; clusters are packed first-fit-decreasing into units <= cap
  whole     reference only (--whole-max-rows): the community as one unit

Steps (--steps):
  quality   units, rows per unit, intra-unit edges kept
  accuracy  one 2-layer GraphSAGE per unit (identical trainer for every
            strategy), for each --seeds; accuracy on the members' test nodes
            of the split communities (unsplit communities are identical under
            every strategy, so they are left out)
  io        (cap in --io-caps) rewrites the materialized table ordered by
            (community_id, block_id) with small row groups and a unit_id
            column, and times loading every unit with pyarrow worker processes
            against today's table and `id & (n-1)` predicate (needs Spark)

Every (cap, strategy) result is also appended to <out-dir>/summary.jsonl.
quality and accuracy need only pyarrow, python-igraph, torch and PyG, so they
also run on a GPU server (--device auto picks CUDA when available).

  .venv-spark/bin/python experiments/block_split_study.py --dataset ogbn-arxiv
"""
import argparse
import json
import os
import random
import shutil
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("OMP_NUM_THREADS", "1")

GRAPH_METHODS = ("louvain", "leiden", "lpa")
STRATEGY_CHOICES = ("id_pow2", "id_equal") + GRAPH_METHODS


# ------------------------------------------------------------------ data access
_DS = {}


def dataset(path):
    import pyarrow.dataset as ds
    if path not in _DS:
        _DS[path] = ds.dataset(path, format="parquet", ignore_prefixes=["_delta_log", "."])
    return _DS[path]


def col_np(table, name):
    return table.column(name).to_pandas().to_numpy()


def read_unit(p2_nodes, p2_edges, c, with_features=False):
    """Rows (sorted by id) and edges (as local row indices) of community c."""
    import pyarrow.dataset as ds
    cols = ["id", "is_member"] + (["label", "split", "features"] if with_features else [])
    t = dataset(p2_nodes).to_table(columns=cols, filter=ds.field("community_id") == int(c))
    ids = col_np(t, "id").astype(np.int64)
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    u = {"ids": ids, "member": col_np(t, "is_member").astype(bool)[order]}
    if with_features:
        f = t.column("features").combine_chunks()
        dim = len(f[0]) if len(f) else 0
        x = f.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(f), dim)
        u["x"] = x[order]
        u["y"] = np.nan_to_num(col_np(t, "label").astype(float), nan=-1).astype(np.int64)[order]
        u["split"] = col_np(t, "split").astype(object)[order]
    e = dataset(p2_edges).to_table(columns=["src", "dst"], filter=ds.field("community_id") == int(c))
    src, dst = col_np(e, "src").astype(np.int64), col_np(e, "dst").astype(np.int64)
    si, di = np.searchsorted(ids, src), np.searchsorted(ids, dst)
    si, di = np.minimum(si, len(ids) - 1), np.minimum(di, len(ids) - 1)
    ok = (ids[si] == src) & (ids[di] == dst)
    u["s"], u["d"] = si[ok], di[ok]
    return u


def n_pow2(rows, cap):
    need = int(np.ceil(float(rows) / cap))
    nb = 1
    while nb < need:
        nb *= 2
    return nb


# ------------------------------------------------------------------ partitioning
def _cluster(g, method):
    if method == "louvain":
        return g.community_multilevel().membership
    if method == "leiden":
        return g.community_leiden(objective_function="modularity").membership
    if method == "lpa":
        return g.community_label_propagation().membership
    raise ValueError(method)


def graph_split(n, s, d, cap, seed, method="louvain"):
    """Local row index -> unit id. Recursive community detection on the unit
    graph, then first-fit-decreasing packing of the clusters into units <= cap."""
    import igraph as ig
    ig.set_random_number_generator(random.Random(seed))
    a, b = np.minimum(s, d), np.maximum(s, d)
    keep = a != b
    key = np.unique(a[keep] * np.int64(n) + b[keep])
    a, b = key // n, key % n
    pieces, stack = [], [np.arange(n, dtype=np.int64)]
    local = np.full(n, -1, dtype=np.int64)
    while stack:
        nodes = stack.pop()
        if len(nodes) <= cap:
            pieces.append(nodes)
            continue
        local[nodes] = np.arange(len(nodes))
        m = (local[a] >= 0) & (local[b] >= 0)
        la, lb = local[a[m]], local[b[m]]
        local[nodes] = -1
        g = ig.Graph(n=len(nodes), edges=list(zip(la.tolist(), lb.tolist())))
        memb = np.asarray(_cluster(g, method), dtype=np.int64)
        k = int(memb.max()) + 1 if len(memb) else 1
        if k <= 1:  # no community structure left: plain chunks
            pieces.extend(nodes[i:i + cap] for i in range(0, len(nodes), cap))
            continue
        order = np.argsort(memb, kind="stable")
        bounds = np.cumsum(np.bincount(memb, minlength=k))[:-1]
        for grp in np.split(nodes[order], bounds):
            (stack if len(grp) > cap else pieces).append(grp)
    pieces.sort(key=len, reverse=True)
    block = np.empty(n, dtype=np.int64)
    room = []
    for p in pieces:
        for i, r in enumerate(room):
            if r >= len(p):
                room[i] -= len(p)
                block[p] = i
                break
        else:
            room.append(cap - len(p))
            block[p] = len(room) - 1
    return block


def block_stats(blocks, member, s, d):
    sizes = np.bincount(blocks)
    mem = np.bincount(blocks, weights=member.astype(float), minlength=len(sizes))
    used = sizes > 0
    return {"blocks": int(used.sum()), "max_rows": int(sizes.max()),
            "mean_rows": float(sizes[used].mean()),
            "memberless_blocks": int((used & (mem == 0)).sum()),
            "edges_kept": int((blocks[s] == blocks[d]).sum())}


def partition_task(args):
    p2_nodes, p2_edges, c, cap, seed, strategies = args
    u = read_unit(p2_nodes, p2_edges, c)
    n = len(u["ids"])
    blocks, split_s = {}, {}
    graph = [m for m in strategies if m in GRAPH_METHODS]
    for m in graph:
        t0 = time.perf_counter()
        blocks[m] = graph_split(n, u["s"], u["d"], cap, seed, m)
        split_s[m] = round(time.perf_counter() - t0, 2)
    if "id_pow2" in strategies:
        blocks["id_pow2"] = u["ids"] & (n_pow2(n, cap) - 1)
    if "id_equal" in strategies:
        ref = blocks[graph[0]] if graph else u["ids"] & (n_pow2(n, cap) - 1)
        blocks["id_equal"] = u["ids"] % (int(ref.max()) + 1)
    out = {"community_id": int(c), "rows": n, "members": int(u["member"].sum()),
           "edge_rows": int(len(u["s"])), "split_s": split_s}
    for k in strategies:
        out[k] = block_stats(blocks[k], u["member"], u["s"], u["d"])
    return out, u["ids"], blocks


# ------------------------------------------------------------------ training
def train_task(args):
    (key, x, y, train_m, test_m, ei, fallback, n_cls, epochs, hidden, lr, seed, device) = args
    n_test = int(test_m.sum())
    if n_test == 0:
        return key, 0, 0, 1          # nothing to score; not a fallback
    if train_m.sum() == 0:
        return key, int((y[test_m] == fallback).sum()), n_test, 0
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    dev = torch.device(device)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = SAGEConv(x.shape[1], hidden)
            self.c2 = SAGEConv(hidden, n_cls)

        def forward(self, X, E):
            h = F.dropout(F.relu(self.c1(X, E)), p=0.5, training=self.training)
            return self.c2(h, E)

    X, Y = torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)
    E = torch.from_numpy(ei).to(dev)
    tr, te = torch.from_numpy(train_m).to(dev), torch.from_numpy(test_m).to(dev)
    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        F.cross_entropy(net(X, E)[tr], Y[tr]).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(X, E).argmax(1)
    return key, int((pred[te] == Y[te]).sum()), n_test, 1


def block_tasks(u, blocks, strategy, c, fallback, n_cls, args, seeds, device):
    """One training task per (unit, seed)."""
    train_all = u["member"] & (u["split"] == "train")
    test_all = u["member"] & (u["split"] == "test")
    a, b = np.minimum(u["s"], u["d"]), np.maximum(u["s"], u["d"])
    keep = a != b
    n = len(u["ids"])
    key = np.unique(a[keep] * np.int64(n) + b[keep])
    ua, ub = key // n, key % n
    same = blocks[ua] == blocks[ub]
    ua, ub = ua[same], ub[same]
    eb = blocks[ua]
    order_e = np.argsort(eb, kind="stable")
    ua, ub, eb = ua[order_e], ub[order_e], eb[order_e]
    tasks = []
    for bid in np.unique(blocks):
        idx = np.nonzero(blocks == bid)[0]
        if not u["member"][idx].any():
            continue
        loc = np.full(n, -1, dtype=np.int64)
        loc[idx] = np.arange(len(idx))
        lo, hi = np.searchsorted(eb, bid), np.searchsorted(eb, bid, side="right")
        e = np.stack([loc[ua[lo:hi]], loc[ub[lo:hi]]])
        ei = np.concatenate([e, e[::-1]], axis=1).astype(np.int64)
        xb, yb, trb, teb = u["x"][idx], u["y"][idx], train_all[idx], test_all[idx]
        for sd in seeds:
            tasks.append(((strategy, int(sd), int(c), int(bid)), xb, yb, trb, teb, ei, fallback, n_cls,
                          args.epochs, args.hidden, args.lr, int(sd), device))
    return tasks


# ------------------------------------------------------------------ io
def load_task(args):
    """Load one unit the way a Phase 3b worker does."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    key, nodes_path, edges_path, c, b, n, mode, unit = args
    t0 = time.perf_counter()
    base = ds.field("community_id") == int(c)
    if mode == "unit":
        fn = fe = ds.field("unit_id") == int(unit)
    elif mode == "block":
        fn = fe = base & (ds.field("block_id") == int(b))
    elif n > 1:
        fn = base & (pc.bit_wise_and(ds.field("id"), n - 1) == int(b))
        fe = (base & (pc.bit_wise_and(ds.field("src"), n - 1) == int(b))
              & (pc.bit_wise_and(ds.field("dst"), n - 1) == int(b)))
    else:
        fn = fe = base
    t = dataset(nodes_path).to_table(columns=["id", "label", "features", "split", "is_member"], filter=fn)
    f = t.column("features").combine_chunks()
    if len(f):
        f.flatten().to_numpy(zero_copy_only=False).reshape(len(f), -1)
    e = dataset(edges_path).to_table(columns=["src", "dst"], filter=fe)
    ids = col_np(t, "id")
    return key, len(ids), e.num_rows, int(ids.sum()), time.perf_counter() - t0


_FRAGS = {}


def planned_bytes(path, filt):
    """Parquet bytes a pyarrow scan with this filter fetches: all column chunks of
    every row group whose min/max statistics may satisfy it."""
    if path not in _FRAGS:
        frs = list(dataset(path).get_fragments())
        for f in frs:
            f.ensure_complete_metadata()
        _FRAGS[path] = frs
    total = 0
    for f in _FRAGS[path]:
        md = f.metadata
        for rg in f.subset(filter=filt).row_groups:
            g = md.row_group(rg.id)
            total += sum(g.column(k).total_compressed_size for k in range(g.num_columns))
    return total


def build_spark(args):
    if sys.platform == "darwin" and os.path.isdir("/opt/homebrew/opt/openjdk@17"):
        os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    from pyspark.sql import SparkSession
    b = (SparkSession.builder.appName("block-split-study").master(f"local[{min(args.cores, 8)}]")
         .config("spark.driver.host", "127.0.0.1")
         .config("spark.driver.memory", args.mem)
         .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
         .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
         .config("spark.sql.shuffle.partitions", "16")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .config("spark.sql.parquet.columnarReaderBatchSize", "512")
         # small row groups, so a unit predicate can skip inside a file
         .config("spark.hadoop.parquet.block.size", str(int(args.row_group_mb * 1024 * 1024)))
         .config("spark.ui.showConsoleProgress", "false"))
    jars = os.environ.get("LAKEGRL_SPARK_JARS")
    if jars:
        spark = b.config("spark.jars", jars).getOrCreate()
    else:
        from delta import configure_spark_with_delta_pip
        spark = configure_spark_with_delta_pip(b).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def write_blocked(spark, p2_nodes, p2_edges, assign_pdf, out_dir, n_files):
    """Materialized table + block_id + unit_id, ordered by unit; edges carry the
    unit only when both endpoints are in it (else -1)."""
    from pyspark.sql import functions as F
    shutil.rmtree(out_dir, ignore_errors=True)
    asg = spark.createDataFrame(assign_pdf)
    nodes = spark.read.format("delta").load(p2_nodes).join(asg, ["id", "community_id"], "inner")
    (nodes.repartitionByRange(n_files, "unit_id")
          .sortWithinPartitions("unit_id", "id")
          .write.format("delta").mode("overwrite").save(f"{out_dir}/p2_nodes"))
    sa = asg.select(F.col("id").alias("src"), "community_id",
                    F.col("block_id").alias("_sb"), F.col("unit_id").alias("_su"))
    da = asg.select(F.col("id").alias("dst"), "community_id", F.col("block_id").alias("_db"))
    same = F.col("_sb") == F.col("_db")
    edges = (spark.read.format("delta").load(p2_edges)
             .join(sa, ["src", "community_id"], "inner").join(da, ["dst", "community_id"], "inner")
             .withColumn("block_id", F.when(same, F.col("_sb")).otherwise(F.lit(-1)))
             .withColumn("unit_id", F.when(same, F.col("_su")).otherwise(F.lit(-1)))
             .select("src", "dst", "community_id", "block_id", "unit_id"))
    (edges.repartitionByRange(n_files, "unit_id")
          .sortWithinPartitions("unit_id")
          .write.format("delta").mode("overwrite").save(f"{out_dir}/p2_edges"))
    return f"{out_dir}/p2_nodes", f"{out_dir}/p2_edges"


# ------------------------------------------------------------------ steps
def run_accuracy(args, ctx, p2_nodes, p2_edges, split_c, assign, strategies, cap, tag, device):
    rng = random.Random(args.partition_seed)
    train_c = split_c if not args.train_communities else sorted(
        rng.sample(split_c, min(args.train_communities, len(split_c))))
    n_cls = int(np.nanmax(dataset(p2_nodes).to_table(columns=["label"]).column("label")
                          .to_pandas().to_numpy(dtype=float))) + 1
    acc = {}
    t0 = time.time()
    print(f"\n-- accuracy (cap {cap:,}): {len(train_c)} split communities, seeds {args.seeds}, "
          f"{args.epochs} epochs, GraphSAGE on {device}", flush=True)
    with ctx.Pool(args.cores) as pool:
        for i, c in enumerate(train_c):
            u = read_unit(p2_nodes, p2_edges, c, with_features=True)
            ids, blocks = assign[c]
            assert np.array_equal(ids, u["ids"])
            tr = u["member"] & (u["split"] == "train")
            fallback = int(np.bincount(u["y"][tr]).argmax()) if tr.any() else 0
            tasks = []
            for k in strategies:
                tasks += block_tasks(u, blocks[k], k, c, fallback, n_cls, args, args.seeds, device)
            if args.whole_max_rows and len(u["ids"]) <= args.whole_max_rows:
                tasks += block_tasks(u, np.zeros(len(u["ids"]), np.int64), "whole", c,
                                     fallback, n_cls, args, args.seeds, device)
            for (k, sd, cc, _), corr, n_test, trained in pool.imap_unordered(train_task, tasks):
                a = acc.setdefault(k, {}).setdefault(sd, {}).setdefault(cc, [0, 0, 0, 0])
                a[0] += corr
                a[1] += n_test
                a[2] += 1
                a[3] += 1 - trained
            print(f"  [{i + 1}/{len(train_c)}] community {c}: {len(u['ids']):,} rows, "
                  f"{len(tasks)} trainings  [{time.time() - t0:.0f}s]", flush=True)
    whole_c = sorted({c for per in acc.get("whole", {}).values() for c in per})
    out = {}
    print(f"\n  {'strategy':9s} {'test acc (mean ± std over seeds)':>34s} {'units':>6s} "
          f"{'fallback':>9s}" + (f"   on the {len(whole_c)} communities trainable whole" if whole_c else ""))
    for k in list(strategies) + (["whole"] if whole_c else []):
        per_seed, per_seed_w = [], []
        units = fb = n_test_tot = 0
        for sd, per in sorted(acc.get(k, {}).items()):
            corr, n_test = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
            per_seed.append(corr / max(n_test, 1))
            wc = sum(per[c][0] for c in whole_c if c in per)
            wt = sum(per[c][1] for c in whole_c if c in per)
            per_seed_w.append(wc / max(wt, 1))
            units = sum(v[2] for v in per.values())
            fb = sum(v[3] for v in per.values())
            n_test_tot = n_test
        out[k] = {"acc_mean": float(np.mean(per_seed)), "acc_std": float(np.std(per_seed)),
                  "acc_per_seed": [round(v, 4) for v in per_seed], "units": units,
                  "fallback_units": fb, "test_nodes": n_test_tot,
                  "acc_whole_set_mean": float(np.mean(per_seed_w)) if whole_c else None}
        o = out[k]
        print(f"  {k:9s} {100 * o['acc_mean']:24.2f} ± {100 * o['acc_std']:5.2f} {units:6d} {fb:9d}"
              + (f"   {100 * o['acc_whole_set_mean']:.2f}%" if whole_c else ""))
    json.dump({"communities": train_c, "whole_communities": whole_c, "seeds": args.seeds,
               "epochs": args.epochs, "hidden": args.hidden, "summary": out,
               "per_community": {k: {str(sd): {str(c): v for c, v in per.items()} for sd, per in d.items()}
                                 for k, d in acc.items()}},
              open(f"{args.out_dir}/{tag}_accuracy.json", "w"), indent=1)
    return out, len(train_c)


def run_io(args, ctx, p2_nodes, p2_edges, allc, comm, cids, sizes, assign, strategies, cap, tag):
    import pandas as pd
    import pyarrow.dataset as ds
    graph = [m for m in strategies if m in GRAPH_METHODS]
    if "id_pow2" not in strategies or not graph:
        print("  io needs id_pow2 and one graph strategy; skipped")
        return None
    gm = graph[0]
    work = os.path.abspath(args.work_dir or f"local_data/joinbench/blocksplit/{args.dataset}")
    ids_all = col_np(allc, "id").astype(np.int64)
    base = pd.DataFrame({"id": ids_all, "community_id": comm, "block_id": np.zeros(len(comm), np.int64)})
    base = base.sort_values(["community_id", "id"]).reset_index(drop=True)
    spark = build_spark(args)
    n_files = len(dataset(p2_nodes).files)
    units = {}
    tables = {}
    t_write = {}
    for k in ("id_pow2", gm):
        pdf = base.copy()
        for c, (ids, blocks) in assign.items():
            sel = (pdf["community_id"] == c).to_numpy()
            assert np.array_equal(pdf.loc[sel, "id"].to_numpy(), ids)
            pdf.loc[sel, "block_id"] = blocks[k]
        pdf["unit_id"] = pdf.groupby(["community_id", "block_id"], sort=True).ngroup().astype(np.int64)
        units[k] = pdf.drop_duplicates(["community_id", "block_id"])[["community_id", "block_id", "unit_id"]]
        t0 = time.time()
        tables[k] = write_blocked(spark, p2_nodes, p2_edges, pdf, f"{work}/{k}", n_files)
        t_write[k] = round(time.time() - t0, 1)
    spark.stop()

    def items(k, mode, npath, epath):
        out = []
        for c, b, u in units[k].itertuples(index=False):
            c, b, u = int(c), int(b), int(u)
            n = n_pow2(int(sizes[list(cids).index(c)]), cap) if (mode == "bits" and c in assign) else 1
            out.append(((c, b), npath, epath, c, b, n, mode, u))
        return out

    layouts = {
        "today: id & (n-1) predicate": ("id_pow2", "bits", p2_nodes, p2_edges),
        "sorted, (community, block) predicate, id_pow2 units": ("id_pow2", "block") + tables["id_pow2"],
        "sorted, unit_id predicate, id_pow2 units": ("id_pow2", "unit") + tables["id_pow2"],
        f"sorted, (community, block) predicate, {gm} units": (gm, "block") + tables[gm],
        f"sorted, unit_id predicate, {gm} units": (gm, "unit") + tables[gm],
    }
    names = list(layouts)
    io = {}
    print(f"\n-- io (cap {cap:,}): loading every unit with {args.cores} pyarrow worker processes, "
          f"{args.row_group_mb} MB row groups in the rewritten tables", flush=True)
    with ctx.Pool(args.cores) as pool:
        for rep in range(-1, args.io_repeats):
            for name in (names if rep % 2 == 0 else names[::-1]):
                k, mode, npath, epath = layouts[name]
                its = items(k, mode, npath, epath)
                t0 = time.perf_counter()
                out = pool.map(load_task, its, chunksize=1)
                wall = time.perf_counter() - t0
                if rep < 0:
                    continue
                r = io.setdefault(name, {"wall_s": [], "units": len(its)})
                r["wall_s"].append(round(wall, 2))
                r["rows"] = sum(o[1] for o in out)
                r["edges"] = sum(o[2] for o in out)
                r["checksum"] = {f"{o[0][0]}:{o[0][1]}": [o[1], o[2], o[3]] for o in out}
    for name in names:
        k, mode, npath, epath = layouts[name]
        total = 0
        for it in items(k, mode, npath, epath):
            _, _, _, c, b, n, m, u = it
            if m == "unit":
                f = ds.field("unit_id") == u
            elif m == "block":
                f = (ds.field("community_id") == c) & (ds.field("block_id") == b)
            else:
                f = ds.field("community_id") == c
            total += planned_bytes(npath, f) + planned_bytes(epath, f)
        io[name]["read_bytes"] = total
    checks = {"id_pow2: today == sorted (community, block)": io[names[0]]["checksum"] == io[names[1]]["checksum"],
              "id_pow2: today == sorted unit_id": io[names[0]]["checksum"] == io[names[2]]["checksum"],
              f"{gm}: (community, block) == unit_id": io[names[3]]["checksum"] == io[names[4]]["checksum"]}
    table_bytes = {}
    for k in tables:
        table_bytes[k] = sum(os.path.getsize(f) for p in tables[k] for f in dataset(p).files)
    print(f"  {'layout':52s} {'units':>6s} {'load':>8s} {'read':>9s} {'edges':>12s}")
    for name in names:
        r = io[name]
        print(f"  {name:52s} {r['units']:6d} {np.median(r['wall_s']):7.2f}s "
              f"{r['read_bytes'] / 1e9:7.2f}GB {r['edges']:12,d}")
    print(f"  identical data per unit: {checks}")
    print(f"  rewritten table sizes (nodes+edges): "
          f"{ {k: round(v / 1e9, 2) for k, v in table_bytes.items()} } GB; write time {t_write} s")
    for r in io.values():
        r.pop("checksum")
    res = {"layouts": io, "checks": checks, "write_s": t_write, "row_group_mb": args.row_group_mb,
           "rewritten_table_bytes": table_bytes}
    json.dump(res, open(f"{args.out_dir}/{tag}_io.json", "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ogbn-arxiv")
    ap.add_argument("--p2-root", default=None,
                    help="dir with the materialized p2_nodes/p2_edges "
                         "(default local_data/joinbench/gjbench/<dataset>/materialized)")
    ap.add_argument("--steps", nargs="+", default=["quality", "accuracy", "io"],
                    choices=["quality", "accuracy", "io"])
    ap.add_argument("--caps", type=int, nargs="+", default=[10000],
                    help="Phase 3b max_nodes_per_community values")
    ap.add_argument("--strategies", nargs="+", default=["id_pow2", "id_equal", "louvain"],
                    choices=STRATEGY_CHOICES)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="training seeds")
    ap.add_argument("--partition-seed", type=int, default=42)
    ap.add_argument("--cores", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or auto (cuda when available)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--train-communities", type=int, default=0,
                    help="train on this many split communities (seeded sample); 0 = all")
    ap.add_argument("--whole-max-rows", type=int, default=60000,
                    help="also train split communities up to this size as one unit (0 = off)")
    ap.add_argument("--io-caps", type=int, nargs="*", default=[10000])
    ap.add_argument("--row-group-mb", type=float, default=4.0)
    ap.add_argument("--io-repeats", type=int, default=2)
    ap.add_argument("--mem", default="8g")
    ap.add_argument("--work-dir", default=None,
                    help="scratch for the rewritten tables (default local_data/joinbench/blocksplit/<dataset>)")
    ap.add_argument("--out-dir", default="results/block_split")
    ap.add_argument("--tag", default="", help="suffix for output files and summary rows")
    args = ap.parse_args()

    import multiprocessing as mp
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    p2_root = args.p2_root or f"local_data/joinbench/gjbench/{args.dataset}/materialized"
    p2_nodes, p2_edges = os.path.abspath(f"{p2_root}/p2_nodes"), os.path.abspath(f"{p2_root}/p2_edges")
    os.makedirs(args.out_dir, exist_ok=True)
    ctx = mp.get_context("spawn")
    strategies = list(dict.fromkeys(args.strategies))
    run_id = time.strftime("%Y%m%d-%H%M%S")

    allc = dataset(p2_nodes).to_table(columns=["id", "community_id"])
    comm = col_np(allc, "community_id").astype(np.int64)
    counts = np.bincount(comm - comm.min())
    cids = np.nonzero(counts)[0] + comm.min()
    sizes = counts[cids - comm.min()]
    print(f"== {args.dataset}: {len(cids)} communities, {int(sizes.sum()):,} unit rows; "
          f"{args.cores} processes, device {device}, strategies {strategies}", flush=True)

    for cap in args.caps:
        tag = f"{args.dataset}_cap{cap}" + (f"_{args.tag}" if args.tag else "")
        split_c = [int(c) for c, s in zip(cids, sizes) if s > cap]
        t0 = time.time()
        with ctx.Pool(args.cores) as pool:
            res = pool.map(partition_task,
                           [(p2_nodes, p2_edges, c, cap, args.partition_seed, strategies) for c in split_c],
                           chunksize=1)
        quality = [r[0] for r in res]
        assign = {r[0]["community_id"]: (r[1], r[2]) for r in res}
        tot_e = sum(q["edge_rows"] for q in quality)
        print(f"\n-- quality (cap {cap:,}): {len(split_c)} of {len(cids)} communities split, "
              f"{sum(q['rows'] for q in quality):,} rows; partitioning {time.time() - t0:.1f}s wall", flush=True)
        print(f"  {'strategy':9s} {'units':>7s} {'max rows':>9s} {'mean rows':>10s} {'edges kept':>11s} "
              f"{'split s (cpu)':>14s}")
        qsum = {}
        for k in strategies:
            units = sum(q[k]["blocks"] for q in quality)
            kept = sum(q[k]["edges_kept"] for q in quality)
            qsum[k] = {"units": units, "max_rows": max(q[k]["max_rows"] for q in quality),
                       "mean_rows": round(sum(q["rows"] for q in quality) / max(units, 1), 1),
                       "edges_kept_frac": round(kept / max(tot_e, 1), 4),
                       "memberless_units": sum(q[k]["memberless_blocks"] for q in quality),
                       "split_cpu_s": round(sum(q["split_s"].get(k, 0) for q in quality), 1)}
            s = qsum[k]
            print(f"  {k:9s} {units:7d} {s['max_rows']:9,d} {s['mean_rows']:10,.0f} "
                  f"{100 * s['edges_kept_frac']:10.1f}% {s['split_cpu_s']:14.1f}")
        json.dump({"cap": cap, "split_communities": len(split_c), "summary": qsum, "communities": quality},
                  open(f"{args.out_dir}/{tag}_quality.json", "w"), indent=1)

        acc, n_train_c = ({}, 0)
        if "accuracy" in args.steps and split_c:
            acc, n_train_c = run_accuracy(args, ctx, p2_nodes, p2_edges, split_c, assign,
                                          strategies, cap, tag, device)
        io = None
        if "io" in args.steps and cap in args.io_caps and split_c:
            io = run_io(args, ctx, p2_nodes, p2_edges, allc, comm, cids, sizes, assign, strategies, cap, tag)

        with open(f"{args.out_dir}/summary.jsonl", "a") as fh:
            for k in list(strategies) + (["whole"] if "whole" in acc else []):
                row = {"run_id": run_id, "tag": args.tag, "dataset": args.dataset, "cap": cap,
                       "strategy": k, "split_communities": len(split_c),
                       "trained_communities": n_train_c, "epochs": args.epochs, "seeds": args.seeds,
                       "device": device, **qsum.get(k, {}), **{f"acc_{kk}" if not kk.startswith("acc") else kk: vv
                                                               for kk, vv in acc.get(k, {}).items()}}
                fh.write(json.dumps(row) + "\n")
        print(f"\n  results -> {args.out_dir}/{tag}_*.json, {args.out_dir}/summary.jsonl", flush=True)


if __name__ == "__main__":
    main()

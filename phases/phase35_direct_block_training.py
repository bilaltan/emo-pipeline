"""Phase 3.5: bounded direct-Delta GNN validation on complete seed units.

This phase is deliberately separate from the legacy community trainer.  Spark
uses the Phase 2.7 manifest to select complete source-seed units, joins only
their source/halo node records from Phase 2.6 Delta tables, and sends one
bounded unit at a time to ``applyInPandas``.  It never builds an unbounded
``collect_list`` adjacency payload.

The validation mode trains an independent model per selected unit.  It proves
direct lossless block loading and bounded GNN execution; it is not a
synchronized full-graph model.
"""

import time


def _result_schema():
    from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType
    return StructType([
        StructField("community_id", LongType()),
        StructField("n_nodes", LongType()),
        StructField("n_edges", LongType()),
        StructField("n_train", LongType()),
        StructField("n_val", LongType()),
        StructField("n_test", LongType()),
        StructField("n_boundary", LongType()),
        StructField("n_internal", LongType()),
        StructField("comm_test_acc", DoubleType()),
        StructField("boundary_acc", DoubleType()),
        StructField("internal_acc", DoubleType()),
        StructField("comm_link_auc", DoubleType()),
        StructField("size_bucket", StringType()),
        StructField("load_time_s", DoubleType()),
        StructField("node_train_time_s", DoubleType()),
        StructField("link_train_time_s", DoubleType()),
        StructField("peak_mem_mb", DoubleType()),
    ])


def _train_complete_seed_unit(pdf):
    """Train a one-layer GraphSAGE classifier for one bounded source seed unit."""
    import os
    import resource
    import time
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    import dgl
    import dgl.nn as dglnn

    started = time.time()
    node_rows = pdf[pdf["record_type"] == "node"].copy()
    edge_rows = pdf[pdf["record_type"] == "edge"]
    shard = int(pdf["src_shard"].iloc[0])
    seed_block = int(pdf["seed_block"].iloc[0])
    unit_id = shard * 1000 + seed_block

    if node_rows.empty:
        raise RuntimeError(f"Phase 3.5 unit {shard}/{seed_block} has no node rows")

    node_rows = node_rows.drop_duplicates("id", keep="first").reset_index(drop=True)
    node_ids = node_rows["id"].to_numpy(dtype=np.int64)
    sorted_order = np.argsort(node_ids)
    sorted_ids = node_ids[sorted_order]

    src_ids = edge_rows["src"].dropna().to_numpy(dtype=np.int64)
    dst_ids = edge_rows["dst"].dropna().to_numpy(dtype=np.int64)
    src_pos = np.searchsorted(sorted_ids, src_ids)
    dst_pos = np.searchsorted(sorted_ids, dst_ids)
    valid = (
        (src_pos < len(sorted_ids)) & (dst_pos < len(sorted_ids))
        & (sorted_ids[np.minimum(src_pos, len(sorted_ids) - 1)] == src_ids)
        & (sorted_ids[np.minimum(dst_pos, len(sorted_ids) - 1)] == dst_ids)
    )
    src_local = sorted_order[src_pos[valid]]
    dst_local = sorted_order[dst_pos[valid]]

    features = np.stack(node_rows["features"].map(lambda x: np.asarray(x, dtype=np.float32)).to_numpy())
    labels = node_rows["label"].fillna(-1).to_numpy(dtype=np.int64)
    seed_mask = node_rows["is_seed"].fillna(False).to_numpy(dtype=bool)
    splits = node_rows["split"].fillna("").to_numpy(dtype=str)
    train_mask = seed_mask & (splits == "train") & (labels >= 0)
    test_mask = seed_mask & (splits == "test") & (labels >= 0)

    hidden_dim = int(node_rows["_hidden"].iloc[0])
    epochs = int(node_rows["_epochs"].iloc[0])
    lr = float(node_rows["_lr"].iloc[0])
    num_classes = int(node_rows["_num_classes"].iloc[0])
    torch.manual_seed(42 + unit_id)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))

    # Blocks are source-owned, while DGL aggregates incoming messages. Reverse
    # the stored (source, destination) relation so a source seed receives its
    # complete destination-halo neighborhood for this one-hop validation.
    graph = dgl.graph((torch.as_tensor(dst_local), torch.as_tensor(src_local)), num_nodes=len(node_rows))
    graph = dgl.add_self_loop(graph)
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels)
    train_idx = torch.from_numpy(np.flatnonzero(train_mask))
    test_idx = torch.from_numpy(np.flatnonzero(test_mask))

    class SeedSAGE(torch.nn.Module):
        def __init__(self, in_dim, hidden, classes):
            super().__init__()
            self.conv1 = dglnn.SAGEConv(in_dim, hidden, "mean")
            self.classifier = torch.nn.Linear(hidden, classes)

        def forward(self, graph_, x_):
            return self.classifier(F.relu(self.conv1(graph_, x_)))

    model = SeedSAGE(x.shape[1], hidden_dim, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    train_started = time.time()
    if len(train_idx):
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = model(graph, x)
            loss = F.cross_entropy(logits[train_idx], y[train_idx])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(graph, x).argmax(dim=1)
        correct = int((predictions[test_idx] == y[test_idx]).sum().item()) if len(test_idx) else 0
    n_test = int(len(test_idx))
    peak_mem_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    n_seed = int(seed_mask.sum())

    return pd.DataFrame([{
        "community_id": unit_id,
        "n_nodes": int(len(node_rows)),
        "n_edges": int(len(src_local)),
        "n_train": int(train_mask.sum()),
        "n_val": int((seed_mask & (splits == "valid") & (labels >= 0)).sum()),
        "n_test": n_test,
        "n_boundary": 0,
        "n_internal": n_seed,
        "comm_test_acc": float(correct / n_test) if n_test else 0.0,
        "boundary_acc": 0.0,
        "internal_acc": float(correct / n_test) if n_test else 0.0,
        "comm_link_auc": 0.5,
        "size_bucket": "direct_seed_unit",
        "load_time_s": float(train_started - started),
        "node_train_time_s": float(time.time() - train_started),
        "link_train_time_s": 0.0,
        "peak_mem_mb": peak_mem_mb,
    }])


def run_phase35(spark, datasets, algorithms, get_paths_fn, timing, results, **kwargs):
    """Run direct-Delta bounded GNN validation for a deterministic manifest subset."""
    from pyspark.sql import functions as F

    dataset_cfg = kwargs["dataset_cfg"]
    gcn_cfg = kwargs["gcn_cfg"]
    max_units = int(kwargs.get("max_units", 8))
    if max_units < 1:
        raise ValueError("PHASE35_MAX_UNITS must be at least one")

    for dataset in datasets:
        for alg in algorithms:
            key = (dataset, alg, "sage")
            paths = get_paths_fn(dataset, alg)
            started = time.time()
            print(f"\n{'=' * 60}\n  PHASE 3.5 — Direct bounded GraphSAGE validation: {dataset} / {alg}\n{'=' * 60}")
            print(f"  Scheduling {max_units} complete source-seed units from the Phase 2.7 manifest; "
                  "no collect_list aggregation.")

            audit = spark.read.format("delta").load(paths["p27_manifest"])
            selected = (audit.orderBy(F.desc("n_edges"), "src_shard", "seed_block")
                .limit(max_units)
                .select("src_shard", "seed_block", "n_seed_nodes", "n_edges"))
            expected = selected.agg(
                F.count("*").alias("n_units"),
                F.sum("n_seed_nodes").alias("n_seed_nodes"),
                F.sum("n_edges").alias("n_edges"),
            ).first()

            edges = (spark.read.format("delta").load(paths["p26_edges"])
                .join(F.broadcast(selected.select("src_shard", "seed_block")), ["src_shard", "seed_block"], "inner")
                .select("src_shard", "seed_block", "src", "dst"))
            seeds = (spark.read.format("delta").load(paths["p26_nodes"])
                .join(F.broadcast(selected.select(F.col("src_shard").alias("node_shard"), "seed_block")),
                      ["node_shard", "seed_block"], "inner")
                .select("node_shard", "seed_block", "id", "label", "split", "features")
                .withColumnRenamed("node_shard", "src_shard")
                .withColumn("is_seed", F.lit(True)))

            # Fetch each selected unit's exact destination halo in Spark. The
            # final join is relational and bounded by selected units; no node or
            # edge arrays are materialized on the driver.
            halo_keys = edges.select("src_shard", "seed_block", F.col("dst").alias("id")).distinct()
            halos = (halo_keys.join(
                spark.read.format("delta").load(paths["p26_nodes"]).select("id", "label", "split", "features"),
                "id", "inner",
            ).withColumn("is_seed", F.lit(False)))
            unit_nodes = (seeds.unionByName(halos)
                .groupBy("src_shard", "seed_block", "id")
                .agg(
                    F.first("label", ignorenulls=True).alias("label"),
                    F.first("split", ignorenulls=True).alias("split"),
                    F.first("features", ignorenulls=True).alias("features"),
                    F.max(F.col("is_seed").cast("int")).cast("boolean").alias("is_seed"),
                ))

            node_records = (unit_nodes
                .withColumn("record_type", F.lit("node"))
                .withColumn("src", F.lit(None).cast("long"))
                .withColumn("dst", F.lit(None).cast("long")))
            edge_records = (edges
                .withColumn("record_type", F.lit("edge"))
                .withColumn("id", F.lit(None).cast("long"))
                .withColumn("label", F.lit(None).cast("int"))
                .withColumn("split", F.lit(None).cast("string"))
                .withColumn("features", F.lit(None).cast("array<float>"))
                .withColumn("is_seed", F.lit(False)))
            records = (node_records.unionByName(edge_records, allowMissingColumns=False)
                .withColumn("_hidden", F.lit(int(gcn_cfg["hidden_dim"])))
                .withColumn("_epochs", F.lit(int(gcn_cfg["num_epochs"])))
                .withColumn("_lr", F.lit(float(gcn_cfg["lr"])))
                .withColumn("_num_classes", F.lit(int(dataset_cfg[dataset]["num_classes"]))))

            output = records.groupBy("src_shard", "seed_block").applyInPandas(_train_complete_seed_unit, _result_schema())
            rows = output.toPandas()
            if len(rows) != int(expected["n_units"]):
                raise RuntimeError(f"Phase 3.5 result coverage failed: {len(rows)}/{expected['n_units']} units returned")
            if int(rows["n_edges"].sum()) != int(expected["n_edges"]):
                raise RuntimeError(f"Phase 3.5 edge coverage failed: {int(rows['n_edges'].sum())}/{expected['n_edges']}")
            if int(rows["n_internal"].sum()) != int(expected["n_seed_nodes"]):
                raise RuntimeError(
                    f"Phase 3.5 seed coverage failed: {int(rows['n_internal'].sum())}/{expected['n_seed_nodes']}"
                )

            total_test = int(rows["n_test"].sum())
            weighted_acc = float((rows["comm_test_acc"] * rows["n_test"]).sum() / total_test) if total_test else 0.0
            rows.attrs["weighted_comm_acc"] = weighted_acc
            results[key] = rows
            timing[("phase35", dataset, alg, "sage")] = time.time() - started
            print(f"  ✓ Direct unit coverage verified: {len(rows):,}/{expected['n_units']:,} units and "
                  f"{int(rows['n_edges'].sum()):,}/{int(expected['n_edges']):,} edges.")
            print(f"  ✓ Phase 3.5 bounded local-model validation accuracy: {weighted_acc:.4f}; "
                  f"elapsed {timing[('phase35', dataset, alg, 'sage')]:.1f}s.")

"""Phase 3.6: synchronized direct-block GraphSAGE validation.

A bounded validation subset of complete Phase 2.7 seed units trains one shared
model with synchronous, training-node-weighted FedAvg. Spark reads Phase 2.6
Delta blocks directly; no community adjacency is assembled with collect_list.

This is a shared-model proof of the distributed execution design. It is not yet
the final all-8,192-unit trainer: each round deliberately returns a small model
vector to the driver, which is appropriate only for the configured validation
subset.
"""

import time


def _delta_exists(spark, path):
    try:
        spark.read.format("delta").load(path).limit(1).count()
        return True
    except Exception:
        return False


def _round_schema(include_weights=True):
    from pyspark.sql.types import ArrayType, FloatType, LongType, StructField, StructType
    fields = [
        StructField("src_shard", LongType()),
        StructField("seed_block", LongType()),
        StructField("n_train", LongType()),
        StructField("n_test", LongType()),
        StructField("n_correct", LongType()),
    ]
    if include_weights:
        fields.append(StructField("weights", ArrayType(FloatType(), containsNull=False)))
    return StructType(fields)


def _model_class(torch, dglnn, in_dim, hidden_dim, num_classes):
    import torch.nn.functional as F

    class SharedSeedSAGE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = dglnn.SAGEConv(in_dim, hidden_dim, "mean")
            self.classifier = torch.nn.Linear(hidden_dim, num_classes)

        def forward(self, graph, features):
            return self.classifier(F.relu(self.conv1(graph, features)))

    return SharedSeedSAGE


def _load_weights(model, flat_weights):
    import numpy as np
    import torch

    flat = np.asarray(flat_weights, dtype=np.float32)
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            count = parameter.numel()
            if offset + count > len(flat):
                raise RuntimeError("Phase 3.6 received a truncated shared model vector")
            parameter.copy_(torch.from_numpy(flat[offset:offset + count]).view_as(parameter))
            offset += count
    if offset != len(flat):
        raise RuntimeError("Phase 3.6 received an oversized shared model vector")


def _flatten_weights(model):
    import numpy as np
    return np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in model.parameters()]).astype(np.float32)


def _aggregate_weight_partition(rows):
    """Return one weighted model sum per Spark output partition.

    Importing NumPy inside this function keeps driver-specific NumPy objects out
    of Spark closure serialization. The driver receives partition aggregates,
    not one complete model vector for every source-seed unit.
    """
    import numpy as np

    weighted_sum = None
    total_train = 0
    total_test = 0
    total_correct = 0
    n_units = 0
    for row in rows:
        n_train = int(row["n_train"])
        vector = np.asarray(row["weights"], dtype=np.float64)
        if weighted_sum is None:
            weighted_sum = np.zeros_like(vector)
        if n_train:
            weighted_sum += vector * n_train
            total_train += n_train
        total_test += int(row["n_test"])
        total_correct += int(row["n_correct"])
        n_units += 1
    if weighted_sum is not None:
        yield (n_units, total_train, total_test, total_correct, weighted_sum.astype(np.float32).tolist())


def _unit_arrays(pdf):
    """Build the bounded graph arrays and seed-only masks for one Spark group."""
    import numpy as np

    nodes = pdf[pdf["record_type"] == "node"].copy()
    edges = pdf[pdf["record_type"] == "edge"]
    if nodes.empty:
        raise RuntimeError("Phase 3.6 received an empty source-seed unit")
    nodes = nodes.drop_duplicates("id", keep="first").reset_index(drop=True)
    node_ids = nodes["id"].to_numpy(dtype=np.int64)
    sorted_order = np.argsort(node_ids)
    sorted_ids = node_ids[sorted_order]
    src_ids = edges["src"].dropna().to_numpy(dtype=np.int64)
    dst_ids = edges["dst"].dropna().to_numpy(dtype=np.int64)
    src_pos = np.searchsorted(sorted_ids, src_ids)
    dst_pos = np.searchsorted(sorted_ids, dst_ids)
    valid = (
        (src_pos < len(sorted_ids)) & (dst_pos < len(sorted_ids))
        & (sorted_ids[np.minimum(src_pos, len(sorted_ids) - 1)] == src_ids)
        & (sorted_ids[np.minimum(dst_pos, len(sorted_ids) - 1)] == dst_ids)
    )
    # Stored blocks are source-owned. Reverse them for DGL incoming-neighbor
    # aggregation so the source seed receives messages from its destination halo.
    graph_src = sorted_order[dst_pos[valid]]
    graph_dst = sorted_order[src_pos[valid]]
    features = np.stack(nodes["features"].map(lambda x: np.asarray(x, dtype=np.float32)).to_numpy())
    labels = nodes["label"].fillna(-1).to_numpy(dtype=np.int64)
    is_seed = nodes["is_seed"].fillna(False).to_numpy(dtype=bool)
    splits = nodes["split"].fillna("").to_numpy(dtype=str)
    train_mask = is_seed & (splits == "train") & (labels >= 0)
    test_mask = is_seed & (splits == "test") & (labels >= 0)
    return nodes, graph_src, graph_dst, features, labels, train_mask, test_mask


def _make_train_round(shared_weights_bc, local_epochs):
    """Create a Pandas-group UDF closure for one FedAvg local epoch."""
    def train_round(pdf):
        import os
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn.functional as F
        import dgl
        import dgl.nn as dglnn

        nodes, graph_src, graph_dst, features, labels, train_mask, test_mask = _unit_arrays(pdf)
        shard = int(pdf["src_shard"].iloc[0])
        seed_block = int(pdf["seed_block"].iloc[0])
        hidden = int(nodes["_hidden"].iloc[0])
        classes = int(nodes["_num_classes"].iloc[0])
        lr = float(nodes["_lr"].iloc[0])
        torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
        torch.manual_seed(42 + shard * 1000 + seed_block)
        graph = dgl.add_self_loop(dgl.graph((torch.as_tensor(graph_src), torch.as_tensor(graph_dst)), num_nodes=len(nodes)))
        model_type = _model_class(torch, dglnn, features.shape[1], hidden, classes)
        model = model_type()
        # The broadcast deliberately contains a plain Python list. Capturing a
        # driver NumPy array in this closure makes cloudpickle import the
        # driver's NumPy-internal module path on executors, which is unsafe
        # when the driver and executor NumPy minor versions differ.
        _load_weights(model, shared_weights_bc.value)
        train_idx = torch.from_numpy(np.flatnonzero(train_mask))
        test_idx = torch.from_numpy(np.flatnonzero(test_mask))
        x = torch.from_numpy(features)
        y = torch.from_numpy(labels)
        if len(train_idx):
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            for _ in range(local_epochs):
                optimizer.zero_grad()
                logits = model(graph, x)
                F.cross_entropy(logits[train_idx], y[train_idx]).backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions = model(graph, x).argmax(dim=1)
            correct = int((predictions[test_idx] == y[test_idx]).sum().item()) if len(test_idx) else 0
        return pd.DataFrame([{
            "src_shard": shard,
            "seed_block": seed_block,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_correct": correct,
            "weights": _flatten_weights(model).tolist(),
        }])
    return train_round


def _make_evaluate(shared_weights_bc):
    """Create a Pandas-group UDF closure that evaluates the final shared model."""
    def evaluate(pdf):
        import os
        import numpy as np
        import pandas as pd
        import torch
        import dgl
        import dgl.nn as dglnn

        nodes, graph_src, graph_dst, features, labels, train_mask, test_mask = _unit_arrays(pdf)
        shard = int(pdf["src_shard"].iloc[0])
        seed_block = int(pdf["seed_block"].iloc[0])
        hidden = int(nodes["_hidden"].iloc[0])
        classes = int(nodes["_num_classes"].iloc[0])
        torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
        graph = dgl.add_self_loop(dgl.graph((torch.as_tensor(graph_src), torch.as_tensor(graph_dst)), num_nodes=len(nodes)))
        model_type = _model_class(torch, dglnn, features.shape[1], hidden, classes)
        model = model_type()
        _load_weights(model, shared_weights_bc.value)
        test_idx = torch.from_numpy(np.flatnonzero(test_mask))
        x = torch.from_numpy(features)
        y = torch.from_numpy(labels)
        model.eval()
        with torch.no_grad():
            predictions = model(graph, x).argmax(dim=1)
            correct = int((predictions[test_idx] == y[test_idx]).sum().item()) if len(test_idx) else 0
        return pd.DataFrame([{
            "src_shard": shard,
            "seed_block": seed_block,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "n_correct": correct,
        }])
    return evaluate


def _build_records(spark, paths, selected, gcn_cfg, dataset_cfg):
    """Build mixed node/edge records with Spark joins, without list aggregation."""
    from pyspark.sql import functions as F

    selected_keys = selected.select("src_shard", "seed_block")
    edges = (spark.read.format("delta").load(paths["p26_edges"])
        .join(F.broadcast(selected_keys), ["src_shard", "seed_block"], "inner")
        .select("src_shard", "seed_block", "src", "dst"))
    seeds = (spark.read.format("delta").load(paths["p26_nodes"])
        .join(F.broadcast(selected_keys.select(F.col("src_shard").alias("node_shard"), "seed_block")),
              ["node_shard", "seed_block"], "inner")
        .select("node_shard", "seed_block", "id", "label", "split", "features")
        .withColumnRenamed("node_shard", "src_shard")
        .withColumn("is_seed", F.lit(True)))
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
    nodes = (unit_nodes.withColumn("record_type", F.lit("node"))
        .withColumn("src", F.lit(None).cast("long"))
        .withColumn("dst", F.lit(None).cast("long")))
    edge_records = (edges.withColumn("record_type", F.lit("edge"))
        .withColumn("id", F.lit(None).cast("long"))
        .withColumn("label", F.lit(None).cast("int"))
        .withColumn("split", F.lit(None).cast("string"))
        .withColumn("features", F.lit(None).cast("array<float>"))
        .withColumn("is_seed", F.lit(False)))
    return (nodes.unionByName(edge_records)
        .withColumn("_hidden", F.lit(int(gcn_cfg["hidden_dim"])))
        .withColumn("_lr", F.lit(float(gcn_cfg["lr"])))
        .withColumn("_num_classes", F.lit(int(dataset_cfg["num_classes"]))))


def _initial_weights(in_dim, hidden_dim, classes):
    """Deterministic Xavier-like initialization matching the shared model layout."""
    import numpy as np
    rng = np.random.default_rng(42)
    parts = [
        rng.normal(0.0, (2.0 / (in_dim + hidden_dim)) ** 0.5, hidden_dim * in_dim),
        rng.normal(0.0, (2.0 / (in_dim + hidden_dim)) ** 0.5, hidden_dim * in_dim),
        np.zeros(hidden_dim),
        rng.normal(0.0, (2.0 / (hidden_dim + classes)) ** 0.5, classes * hidden_dim),
        np.zeros(classes),
    ]
    return np.concatenate(parts).astype(np.float32)


def run_phase36(spark, datasets, algorithms, get_paths_fn, timing, results, **kwargs):
    """Run a bounded synchronous shared-model validation on direct graph blocks."""
    from pyspark import StorageLevel
    from pyspark.sql import functions as F
    import numpy as np

    dataset_cfgs = kwargs["dataset_cfg"]
    gcn_cfg = kwargs["gcn_cfg"]
    train_units = int(kwargs.get("train_units", kwargs.get("max_units", 8)))
    holdout_units = int(kwargs.get("holdout_units", 64))
    rounds = int(kwargs.get("rounds", 5))
    local_epochs = int(kwargs.get("local_epochs", 1))
    aggregation_partitions = int(kwargs.get("aggregation_partitions", 64))
    server_optimizer = str(kwargs.get("server_optimizer", "fedavg")).lower()
    server_lr = float(kwargs.get("server_lr", 0.003))
    server_beta1 = float(kwargs.get("server_beta1", 0.9))
    server_beta2 = float(kwargs.get("server_beta2", 0.99))
    server_epsilon = float(kwargs.get("server_epsilon", 1e-8))
    use_workset_checkpoint = bool(kwargs.get("use_workset_checkpoint", True))
    repartition_by_unit = bool(kwargs.get("repartition_by_unit", True))
    if train_units < 1 or holdout_units < 1 or rounds < 1 or local_epochs < 1 or aggregation_partitions < 1:
        raise ValueError("Phase 3.6 requires positive training-unit, holdout-unit, round, and local-epoch counts")
    if server_optimizer not in {"fedavg", "fedadam"}:
        raise ValueError("PHASE36_SERVER_OPTIMIZER must be 'fedavg' or 'fedadam'")
    if server_lr <= 0 or not 0 <= server_beta1 < 1 or not 0 <= server_beta2 < 1 or server_epsilon <= 0:
        raise ValueError("Phase 3.6 FedAdam hyperparameters are invalid")

    for dataset in datasets:
        for alg in algorithms:
            key = (dataset, alg)
            paths = get_paths_fn(dataset, alg)
            started = time.time()
            print(f"\n{'=' * 60}\n  PHASE 3.6 — Synchronized direct-block GraphSAGE: {dataset} / {alg}\n{'=' * 60}")
            print(f"  Shared-model synchronized training: {train_units} training units + {holdout_units} fixed holdout units × "
                f"{rounds} rounds × {local_epochs} local epochs; {aggregation_partitions} partition aggregates/round; "
                  "no collect_list aggregation.")
            if server_optimizer == "fedadam":
                print(f"  Server optimizer: FedAdam (lr={server_lr}, beta1={server_beta1}, beta2={server_beta2}).")
            from pyspark.sql import Window
            ranked = (spark.read.format("delta").load(paths["p27_manifest"])
                # Stable hash rank is independent of unit size and creates a
                # disjoint, permanently fixed holdout before training rows.
                .select("src_shard", "seed_block", "n_seed_nodes", "n_edges")
                .withColumn("_rank", F.row_number().over(Window.orderBy(F.xxhash64("src_shard", "seed_block")))))
            holdout = ranked.filter(F.col("_rank") <= holdout_units).drop("_rank")
            selected = (ranked
                .filter((F.col("_rank") > holdout_units) & (F.col("_rank") <= holdout_units + train_units))
                .drop("_rank"))
            expected = selected.agg(F.count("*").alias("n_units"), F.sum("n_seed_nodes").alias("n_seeds"), F.sum("n_edges").alias("n_edges")).first()
            holdout_expected = holdout.agg(F.count("*").alias("n_units"), F.sum("n_seed_nodes").alias("n_seeds"), F.sum("n_edges").alias("n_edges")).first()
            if int(expected["n_units"]) != train_units or int(holdout_expected["n_units"]) != holdout_units:
                raise RuntimeError("Phase 3.6 could not construct the requested disjoint training/holdout unit sets")

            workset_started = time.time()

            workset_base = (
                f"{paths['checkpoints'].rstrip('/')}/phase36_worksets/{paths['tag']}/"
                f"opt_{server_optimizer}_tr{train_units}_ho{holdout_units}"
            )
            train_records_path = f"{workset_base}/train_records"
            holdout_records_path = f"{workset_base}/holdout_records"

            if use_workset_checkpoint and _delta_exists(spark, train_records_path) and _delta_exists(spark, holdout_records_path):
                print("  Reusing Phase 3.6 Delta workset checkpoints for train/holdout records.")
                records = spark.read.format("delta").load(train_records_path)
                holdout_records = spark.read.format("delta").load(holdout_records_path)
            else:
                print("  Building Phase 3.6 train/holdout worksets from Phase 2.6 blocks.")
                records = _build_records(spark, paths, selected, gcn_cfg, dataset_cfgs[dataset])
                holdout_records = _build_records(spark, paths, holdout, gcn_cfg, dataset_cfgs[dataset])
                if use_workset_checkpoint:
                    (records.write.format("delta").mode("overwrite")
                        .partitionBy("src_shard", "seed_block", "record_type")
                        .save(train_records_path))
                    (holdout_records.write.format("delta").mode("overwrite")
                        .partitionBy("src_shard", "seed_block", "record_type")
                        .save(holdout_records_path))
                    records = spark.read.format("delta").load(train_records_path)
                    holdout_records = spark.read.format("delta").load(holdout_records_path)

            if repartition_by_unit:
                records = records.repartition(max(train_units, aggregation_partitions), "src_shard", "seed_block")
                holdout_records = holdout_records.repartition(max(holdout_units, 16), "src_shard", "seed_block")

            records = records.persist(StorageLevel.MEMORY_AND_DISK)
            holdout_records = holdout_records.persist(StorageLevel.MEMORY_AND_DISK)
            records.count()
            holdout_records.count()
            workset_elapsed = time.time() - workset_started
            print(f"  Worksets ready in {workset_elapsed:.1f}s.")

            rounds_started = time.time()
            weights = _initial_weights(dataset_cfgs[dataset]["in_feats"], int(gcn_cfg["hidden_dim"]), dataset_cfgs[dataset]["num_classes"])
            server_m = np.zeros_like(weights)
            server_v = np.zeros_like(weights)

            for round_index in range(1, rounds + 1):
                broadcast = spark.sparkContext.broadcast(weights.tolist())
                updates = records.groupBy("src_shard", "seed_block").applyInPandas(
                    _make_train_round(broadcast, local_epochs), _round_schema(include_weights=True)
                )
                partition_updates = (updates.coalesce(aggregation_partitions).rdd
                    .mapPartitions(_aggregate_weight_partition).collect())
                broadcast.unpersist(blocking=False)
                returned_units = sum(item[0] for item in partition_updates)
                train_weight = sum(item[1] for item in partition_updates)
                if returned_units != int(expected["n_units"]):
                    raise RuntimeError(f"Phase 3.6 round {round_index}: returned {returned_units}/{expected['n_units']} units")
                if train_weight == 0:
                    raise RuntimeError("Phase 3.6 has no training seed nodes in its selected units")
                client_average = (np.sum(
                    [np.asarray(item[4], dtype=np.float64) for item in partition_updates], axis=0
                ) / train_weight).astype(np.float32)
                if server_optimizer == "fedavg":
                    weights = client_average
                else:
                    # FedAdam treats the weighted client-model displacement as
                    # a server gradient direction. Bias correction keeps the
                    # early rounds comparable to later optimization rounds.
                    delta = client_average - weights
                    server_m = server_beta1 * server_m + (1.0 - server_beta1) * delta
                    server_v = server_beta2 * server_v + (1.0 - server_beta2) * np.square(delta)
                    m_hat = server_m / (1.0 - server_beta1 ** round_index)
                    v_hat = server_v / (1.0 - server_beta2 ** round_index)
                    weights = weights + server_lr * m_hat / (np.sqrt(v_hat) + server_epsilon)
                round_test = sum(item[2] for item in partition_updates)
                round_correct = sum(item[3] for item in partition_updates)
                round_acc = float(round_correct / round_test) if round_test else 0.0
                print(f"    Round {round_index}/{rounds}: weighted local-post-update accuracy={round_acc:.4f}; "
                      f"training seeds={train_weight:,}; partition aggregates={len(partition_updates):,}.")
            rounds_elapsed = time.time() - rounds_started

            holdout_started = time.time()
            broadcast = spark.sparkContext.broadcast(weights.tolist())
            evaluation = holdout_records.groupBy("src_shard", "seed_block").applyInPandas(
                _make_evaluate(broadcast), _round_schema(include_weights=False)
            ).toPandas()
            broadcast.unpersist(blocking=False)
            records.unpersist(blocking=False)
            holdout_records.unpersist(blocking=False)
            if len(evaluation) != int(holdout_expected["n_units"]):
                raise RuntimeError(f"Phase 3.6 holdout evaluation returned {len(evaluation)}/{holdout_expected['n_units']} units")
            holdout_elapsed = time.time() - holdout_started
            total_test = int(evaluation["n_test"].sum())
            accuracy = float(evaluation["n_correct"].sum() / total_test) if total_test else 0.0
            elapsed = time.time() - started
            results[key] = {
                "mode": f"synchronous_{server_optimizer}_validation",
                "n_units": int(expected["n_units"]),
                "n_seed_nodes": int(expected["n_seeds"]),
                "n_edges": int(expected["n_edges"]),
                "n_train": int(evaluation["n_train"].sum()),
                "n_test": total_test,
                "holdout_test_acc": accuracy,
                "rounds": rounds,
                "local_epochs": local_epochs,
                "server_optimizer": server_optimizer,
                "server_lr": server_lr if server_optimizer == "fedadam" else None,
                "holdout_units": int(holdout_expected["n_units"]),
                "holdout_seed_nodes": int(holdout_expected["n_seeds"]),
                "holdout_edges": int(holdout_expected["n_edges"]),
                "workset_time_s": workset_elapsed,
                "rounds_time_s": rounds_elapsed,
                "holdout_time_s": holdout_elapsed,
            }
            timing[("phase36", dataset, alg)] = elapsed
            print(f"  ✓ Training coverage verified: {expected['n_units']:,}/{train_units:,} units, "
                  f"{int(expected['n_seeds']):,} seeds, {int(expected['n_edges']):,} edges.")
            print(f"  ✓ Fixed holdout coverage verified: {len(evaluation):,}/{holdout_expected['n_units']:,} units, "
                f"{int(holdout_expected['n_seeds']):,} seeds, {int(holdout_expected['n_edges']):,} edges.")
            print(f"  Timing breakdown: worksets={workset_elapsed:.1f}s, rounds={rounds_elapsed:.1f}s, holdout={holdout_elapsed:.1f}s.")
            print(f"  ✓ Phase 3.6 shared-model fixed-holdout accuracy: {accuracy:.4f} after {rounds} rounds; "
                  f"elapsed {elapsed:.1f}s.")

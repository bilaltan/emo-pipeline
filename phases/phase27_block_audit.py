"""Phase 2.7: audit complete source-seed units before direct GNN training.

Phase 2.6 writes each source seed unit into several destination-hash edge
partitions.  Those partitions are storage units, not independent GNN examples.
This phase combines their *statistics* (without collecting any adjacency list)
to measure the complete one-hop workload for every ``(src_shard, seed_block)``.
"""

import time


def _delta_exists(spark, path):
    try:
        spark.read.format("delta").load(path).limit(1).count()
        return True
    except Exception:
        return False


def run_phase27(spark, datasets, algorithms, get_paths_fn, timing, results, **kwargs):
    """Create a manifest for complete source-seed units.

    The audit is intentionally relational: Spark computes exact counts and
    distinct destination halos, while no edge list is sent to the driver or a
    Pandas UDF.  ``max_resident_nodes_upper_bound`` is conservative because a
    destination halo can overlap a seed node; it is suitable for memory safety
    planning before implementing cross-block message passing.
    """
    from pyspark.sql import functions as F

    feature_dim = int(kwargs.get("feature_dim", 128))
    dataset_cfg = kwargs.get("dataset_cfg", {})
    headroom_multiplier = float(kwargs.get("headroom_multiplier", 4.0))
    force_rerun = bool(kwargs.get("force_rerun", False))
    if feature_dim < 1 or headroom_multiplier < 1:
        raise ValueError("Phase 2.7 feature_dim and headroom_multiplier must be positive")

    for dataset in datasets:
        for alg in algorithms:
            current_feature_dim = int(dataset_cfg.get(dataset, {}).get("in_feats", feature_dim))
            key = (dataset, alg)
            paths = get_paths_fn(dataset, alg)
            audit_path = paths["p27_manifest"]

            if not force_rerun and _delta_exists(spark, audit_path):
                audit = spark.read.format("delta").load(audit_path)
                summary = audit.agg(
                    F.count("*").alias("n_seed_units"),
                    F.sum("n_seed_nodes").alias("n_nodes"),
                    F.sum("n_edges").alias("n_edges"),
                    F.max("n_edges").alias("max_edges"),
                    F.max("n_unique_dst").alias("max_halo_nodes"),
                    F.max("estimated_working_set_gib").alias("max_working_set_gib"),
                ).first()
                results[key] = {
                    "n_seed_units": int(summary["n_seed_units"] or 0),
                    "n_nodes": int(summary["n_nodes"] or 0),
                    "n_edges": int(summary["n_edges"] or 0),
                    "max_edges_per_seed_unit": int(summary["max_edges"] or 0),
                    "max_halo_nodes_per_seed_unit": int(summary["max_halo_nodes"] or 0),
                    "max_estimated_working_set_gib": float(summary["max_working_set_gib"] or 0.0),
                    "loaded_checkpoint": True,
                }
                timing[("phase27", dataset, alg)] = 0.0
                print(f"  [Phase 2.7] Loaded full-neighborhood audit for {dataset}/{alg}: "
                      f"{results[key]['n_seed_units']:,} seed units.")
                continue

            started = time.time()
            print(f"\n{'=' * 60}\n  PHASE 2.7 — Complete seed-unit safety audit: {dataset} / {alg}\n{'=' * 60}")
            print("  Combining all destination partitions logically per seed unit; "
                  "no adjacency arrays are collected.")

            nodes = spark.read.format("delta").load(paths["p26_nodes"])
            edges = spark.read.format("delta").load(paths["p26_edges"])

            # One row per seed unit, including units whose nodes have no outgoing
            # edge.  These counts establish exact one-time seed ownership.
            seed_stats = nodes.groupBy("node_shard", "seed_block").agg(
                F.count("*").alias("n_seed_nodes"),
                F.sum(F.when(F.col("split") == "train", 1).otherwise(0)).alias("n_train"),
                F.sum(F.when(F.col("split") == "valid", 1).otherwise(0)).alias("n_val"),
                F.sum(F.when(F.col("split") == "test", 1).otherwise(0)).alias("n_test"),
            ).alias("seeds")

            # Dropping neighbor_block logically reunites all four Phase 2.6
            # edge partitions for one source seed unit. countDistinct(dst) is
            # exact: it is not the sum of the four per-partition halo counts.
            edge_stats = edges.groupBy("src_shard", "seed_block").agg(
                F.count("*").alias("n_edges"),
                F.countDistinct("src").alias("n_unique_src"),
                F.countDistinct("dst").alias("n_unique_dst"),
            ).alias("edges")

            audit = (seed_stats.join(
                edge_stats,
                (F.col("seeds.node_shard") == F.col("edges.src_shard"))
                & (F.col("seeds.seed_block") == F.col("edges.seed_block")),
                "left",
            ).select(
                F.col("seeds.node_shard").cast("int").alias("src_shard"),
                F.col("seeds.seed_block").cast("int").alias("seed_block"),
                F.col("seeds.n_seed_nodes").cast("long").alias("n_seed_nodes"),
                F.coalesce(F.col("edges.n_edges"), F.lit(0)).cast("long").alias("n_edges"),
                F.coalesce(F.col("edges.n_unique_src"), F.lit(0)).cast("long").alias("n_unique_src"),
                F.coalesce(F.col("edges.n_unique_dst"), F.lit(0)).cast("long").alias("n_unique_dst"),
                F.col("seeds.n_train").cast("long").alias("n_train"),
                F.col("seeds.n_val").cast("long").alias("n_val"),
                F.col("seeds.n_test").cast("long").alias("n_test"),
            ).withColumn(
                "max_resident_nodes_upper_bound",
                (F.col("n_seed_nodes") + F.col("n_unique_dst")).cast("long"),
            ).withColumn(
                # Float32 seed/halo features plus one int64 source and
                # destination index per edge. This is a lower bound; the next
                # column adds configurable model/activation safety headroom.
                "estimated_min_payload_gib",
                ((F.col("max_resident_nodes_upper_bound") * F.lit(current_feature_dim * 4))
                 + (F.col("n_edges") * F.lit(16))) / F.lit(1024 ** 3),
            ).withColumn(
                "estimated_working_set_gib",
                F.col("estimated_min_payload_gib") * F.lit(headroom_multiplier),
            ))

            audit.write.format("delta").mode("overwrite").save(audit_path)

            source_nodes = nodes.count()
            source_edges = edges.count()
            summary = audit.agg(
                F.count("*").alias("n_seed_units"),
                F.sum("n_seed_nodes").alias("n_nodes"),
                F.sum("n_edges").alias("n_edges"),
                F.min("n_edges").alias("min_edges"),
                F.avg("n_edges").alias("avg_edges"),
                F.max("n_edges").alias("max_edges"),
                F.max("n_unique_dst").alias("max_halo_nodes"),
                F.max("estimated_working_set_gib").alias("max_working_set_gib"),
            ).first()

            if int(summary["n_nodes"] or 0) != source_nodes or int(summary["n_edges"] or 0) != source_edges:
                raise RuntimeError(
                    "Phase 2.7 coverage check failed: "
                    f"nodes {summary['n_nodes']}/{source_nodes}, edges {summary['n_edges']}/{source_edges}"
                )

            elapsed = time.time() - started
            results[key] = {
                "n_seed_units": int(summary["n_seed_units"]),
                "n_nodes": int(summary["n_nodes"]),
                "n_edges": int(summary["n_edges"]),
                "min_edges_per_seed_unit": int(summary["min_edges"] or 0),
                "avg_edges_per_seed_unit": float(summary["avg_edges"] or 0.0),
                "max_edges_per_seed_unit": int(summary["max_edges"] or 0),
                "max_halo_nodes_per_seed_unit": int(summary["max_halo_nodes"] or 0),
                "max_estimated_working_set_gib": float(summary["max_working_set_gib"] or 0.0),
                "loaded_checkpoint": False,
            }
            timing[("phase27", dataset, alg)] = elapsed
            print(f"  ✓ Exact coverage verified: {summary['n_nodes']:,}/{source_nodes:,} seeds and "
                  f"{summary['n_edges']:,}/{source_edges:,} edges.")
            print(f"    Complete seed units: {summary['n_seed_units']:,}; edges/unit min/avg/max = "
                  f"{summary['min_edges']:,}/{summary['avg_edges']:.1f}/{summary['max_edges']:,}.")
            print(f"    Max exact destination halo/unit: {summary['max_halo_nodes']:,}; "
                  f"conservative working-set estimate: {summary['max_working_set_gib']:.2f} GiB.")
            print(f"  ✓ Phase 2.7 done in {elapsed:.1f}s.")

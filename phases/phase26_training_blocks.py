"""Phase 2.6: materialize lossless bounded source-owned training blocks.

Every Phase 2.5 node remains a seed exactly once. Every Phase 2.5 edge remains
in exactly one source-owned edge block. The phase does not train a model; it
creates the directly readable layout required to remove Phase 3's unbounded
community ``collect_list`` aggregation.
"""

import time


def _delta_exists(spark, path):
    try:
        spark.read.format("delta").load(path).limit(1).count()
        return True
    except Exception:
        return False


def run_phase26(spark, datasets, algorithms, get_paths_fn, timing, results, **kwargs):
    """Build bounded source-seed and adjacency blocks from the Phase 2.5 store.

    ``seed_block`` subdivides each source shard into roughly equal seed sets.
    ``neighbor_block`` subdivides each seed set's outgoing edges by destination
    hash. Thus a directed edge is assigned to exactly one block:

    ``(src_shard, hash(src, seed-salt) % seed_blocks, hash(dst) % neighbor_blocks)``.

    This is a storage layout, not edge sampling. A later trainer can read one
    adjacency block at a time and fetch destination halo features on demand.
    """
    from pyspark.sql import functions as F

    seed_blocks = int(kwargs.get("seed_blocks", 16))
    neighbor_blocks = int(kwargs.get("neighbor_blocks", 4))
    force_rerun = bool(kwargs.get("force_rerun", False))
    if seed_blocks < 1 or neighbor_blocks < 1:
        raise ValueError("Phase 2.6 block counts must be positive")

    for dataset in datasets:
        for alg in algorithms:
            key = (dataset, alg)
            paths = get_paths_fn(dataset, alg)
            node_blocks_path = paths["p26_nodes"]
            edge_blocks_path = paths["p26_edges"]
            manifest_path = paths["p26_manifest"]

            if not force_rerun and all(
                _delta_exists(spark, path)
                for path in (node_blocks_path, edge_blocks_path, manifest_path)
            ):
                n_nodes = spark.read.format("delta").load(node_blocks_path).count()
                n_edges = spark.read.format("delta").load(edge_blocks_path).count()
                results[key] = {
                    "n_nodes": int(n_nodes),
                    "n_edges": int(n_edges),
                    "loaded_checkpoint": True,
                }
                timing[("phase26", dataset, alg)] = 0.0
                print(f"  [Phase 2.6] Loaded training-block checkpoint for {dataset}/{alg}: "
                      f"{results[key]['n_nodes']:,} seeds, {results[key]['n_edges']:,} edges.")
                continue

            started = time.time()
            print(f"\n{'=' * 60}\n  PHASE 2.6 — Lossless source-owned training blocks: {dataset} / {alg}\n{'=' * 60}")
            print(f"  Splitting each source shard into {seed_blocks} seed blocks and "
                  f"{neighbor_blocks} destination-neighbor blocks; no node or edge sampling.")

            graph_nodes = spark.read.format("delta").load(paths["p25_nodes"])
            graph_edges = spark.read.format("delta").load(paths["p25_edges"])

            # Do not derive seed_block from the same hash used by node_shard.
            # With 512 source shards and 16 seed blocks, hash(id) % 16 is
            # implied by hash(id) % 512 and produces no subdivision at all.
            # A stable second hash argument creates an independent layout.
            node_blocks = graph_nodes.withColumn(
                "seed_block",
                F.pmod(F.xxhash64("id", F.lit("phase26-seed-v1")), F.lit(seed_blocks)).cast("int"),
            )
            edge_blocks = (graph_edges
                .withColumn(
                    "seed_block",
                    F.pmod(F.xxhash64("src", F.lit("phase26-seed-v1")), F.lit(seed_blocks)).cast("int"),
                )
                .withColumn("neighbor_block", F.pmod(F.xxhash64("dst"), F.lit(neighbor_blocks)).cast("int")))

            # Keep node seeds separate from edge blocks. Node seeds are written
            # once; edges are written once. The future direct-Parquet trainer
            # joins/fetches only the halo features it needs for one edge block.
            (node_blocks.repartition("node_shard", "seed_block")
                .write.format("delta").mode("overwrite")
                .partitionBy("node_shard", "seed_block").save(node_blocks_path))
            (edge_blocks.repartition("src_shard", "seed_block", "neighbor_block")
                .write.format("delta").mode("overwrite")
                .partitionBy("src_shard", "seed_block", "neighbor_block").save(edge_blocks_path))

            seed_counts = node_blocks.groupBy("node_shard", "seed_block").agg(
                F.count("*").alias("n_seed_nodes"),
                F.sum(F.when(F.col("split") == "train", 1).otherwise(0)).alias("n_train"),
                F.sum(F.when(F.col("split") == "valid", 1).otherwise(0)).alias("n_val"),
                F.sum(F.when(F.col("split") == "test", 1).otherwise(0)).alias("n_test"),
            )
            edge_counts = edge_blocks.groupBy("src_shard", "seed_block", "neighbor_block").agg(
                F.count("*").alias("n_edges"),
                F.approx_count_distinct("src").alias("n_source_nodes"),
                F.approx_count_distinct("dst").alias("n_halo_nodes"),
            )
            edge_counts = edge_counts.alias("edges")
            seed_counts = seed_counts.alias("seeds")
            manifest = (edge_counts.join(
                seed_counts,
                (F.col("edges.src_shard") == F.col("seeds.node_shard"))
                & (F.col("edges.seed_block") == F.col("seeds.seed_block")),
                "left",
            ).select(
                F.col("edges.src_shard").cast("int").alias("src_shard"),
                F.col("edges.seed_block").cast("int").alias("seed_block"),
                F.col("edges.neighbor_block").cast("int").alias("neighbor_block"),
                F.coalesce(F.col("seeds.n_seed_nodes"), F.lit(0)).cast("long").alias("n_seed_nodes"),
                F.col("edges.n_source_nodes").cast("long").alias("n_source_nodes"),
                F.col("edges.n_halo_nodes").cast("long").alias("n_halo_nodes"),
                F.col("edges.n_edges").cast("long").alias("n_edges"),
                F.coalesce(F.col("seeds.n_train"), F.lit(0)).cast("long").alias("n_train"),
                F.coalesce(F.col("seeds.n_val"), F.lit(0)).cast("long").alias("n_val"),
                F.coalesce(F.col("seeds.n_test"), F.lit(0)).cast("long").alias("n_test"),
            ))
            manifest.write.format("delta").mode("overwrite").save(manifest_path)

            source_node_total = graph_nodes.count()
            source_edge_total = graph_edges.count()
            block_node_total = node_blocks.count()
            block_edge_total = edge_blocks.count()
            seed_layout = node_blocks.agg(
                F.countDistinct("node_shard").alias("n_source_shards"),
                F.countDistinct(F.struct("node_shard", "seed_block")).alias("n_seed_units"),
            ).first()
            summary = manifest.agg(
                F.count("*").alias("n_edge_blocks"),
                F.min("n_edges").alias("min_edges"),
                F.max("n_edges").alias("max_edges"),
                F.avg("n_edges").alias("avg_edges"),
                F.max("n_halo_nodes").alias("max_halo_nodes"),
            ).first()

            if source_node_total != block_node_total or source_edge_total != block_edge_total:
                raise RuntimeError(
                    "Phase 2.6 coverage check failed: "
                    f"nodes {block_node_total}/{source_node_total}, edges {block_edge_total}/{source_edge_total}"
                )
            if seed_blocks > 1 and seed_layout["n_seed_units"] <= seed_layout["n_source_shards"]:
                raise RuntimeError(
                    "Phase 2.6 seed blocks did not subdivide source shards. "
                    "Use an independent seed-block hash."
                )

            elapsed = time.time() - started
            results[key] = {
                "n_nodes": int(block_node_total),
                "n_edges": int(block_edge_total),
                "n_edge_blocks": int(summary["n_edge_blocks"]),
                "n_seed_units": int(seed_layout["n_seed_units"]),
                "min_edges_per_block": int(summary["min_edges"] or 0),
                "max_edges_per_block": int(summary["max_edges"] or 0),
                "avg_edges_per_block": float(summary["avg_edges"] or 0.0),
                "max_halo_nodes": int(summary["max_halo_nodes"] or 0),
                "loaded_checkpoint": False,
            }
            timing[("phase26", dataset, alg)] = elapsed
            print(f"  ✓ Coverage verified: {block_node_total:,}/{source_node_total:,} seed nodes and "
                  f"{block_edge_total:,}/{source_edge_total:,} edges retained exactly once.")
            print(f"    Seed layout: {seed_layout['n_seed_units']:,} source-seed units across "
                f"{seed_layout['n_source_shards']:,} source shards.")
            print(f"    Edge blocks: {results[key]['n_edge_blocks']:,}; edges/block min/avg/max = "
                  f"{results[key]['min_edges_per_block']:,}/"
                  f"{results[key]['avg_edges_per_block']:.1f}/"
                  f"{results[key]['max_edges_per_block']:,}; "
                  f"max halo nodes/block = {results[key]['max_halo_nodes']:,}.")
            print(f"  ✓ Phase 2.6 done in {elapsed:.1f}s.")

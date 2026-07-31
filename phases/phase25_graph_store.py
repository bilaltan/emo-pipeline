"""Phase 2.5: build a bounded, shard-addressable graph store for Phase 3.

The existing Phase 3 path groups every edge of a community into a Python list.
That does not scale to dense Papers100M communities. This phase preserves the
Phase 2 graph as sharded Delta/Parquet tables so future workers can read only
bounded node/adjacency shards directly from distributed storage.
"""

import time


def _table_exists(spark, path):
    try:
        spark.read.format("delta").load(path).limit(1).count()
        return True
    except Exception:
        return False


def run_phase25(spark, datasets, algorithms, get_paths_fn, timing, results, **kwargs):
    """Materialize Phase 2 nodes/edges as source-addressable graph-store shards.

    The output retains every Phase 2 row. It deliberately does not sample or
    aggregate adjacency lists. ``src_shard`` is a deterministic owner for every
    directed edge, allowing later bounded neighbor sampling without a global
    ``collect_list`` or driver collection.
    """
    from pyspark.sql import functions as F

    num_shards = int(kwargs.get("num_shards", 512))
    force_rerun = bool(kwargs.get("force_rerun", False))

    if num_shards < 1:
        raise ValueError("Phase 2.5 num_shards must be positive")

    for dataset in datasets:
        for alg in algorithms:
            key = (dataset, alg)
            paths = get_paths_fn(dataset, alg)
            nodes_path = paths["p25_nodes"]
            edges_path = paths["p25_edges"]
            manifest_path = paths["p25_manifest"]

            if not force_rerun and all(
                _table_exists(spark, path)
                for path in (nodes_path, edges_path, manifest_path)
            ):
                manifest = spark.read.format("delta").load(manifest_path)
                summary = manifest.agg(
                    F.sum("n_nodes").alias("n_nodes"),
                    F.sum("n_edges").alias("n_edges"),
                ).first()
                results[key] = {
                    "n_shards": num_shards,
                    "n_nodes": int(summary["n_nodes"] or 0),
                    "n_edges": int(summary["n_edges"] or 0),
                    "loaded_checkpoint": True,
                }
                timing[("phase25", dataset, alg)] = 0.0
                print(f"  [Phase 2.5] Loaded graph-store checkpoint for {dataset}/{alg}: "
                      f"{results[key]['n_nodes']:,} nodes, {results[key]['n_edges']:,} edges.")
                continue

            started = time.time()
            print(f"\n{'=' * 60}\n  PHASE 2.5 — Graph-store materialization: {dataset} / {alg}\n{'=' * 60}")
            print(f"  Writing {num_shards} deterministic node/source-adjacency shards; no edge sampling or aggregation.")

            source_nodes = spark.read.format("delta").load(paths["p2_nodes"])
            source_edges = spark.read.format("delta").load(paths["p2_edges"])

            # pmod produces [0, num_shards) even when xxhash64 returns negatives.
            nodes = source_nodes.withColumn(
                "node_shard", F.pmod(F.xxhash64("id"), F.lit(num_shards)).cast("int")
            )
            edges = (source_edges
                .withColumn("src_shard", F.pmod(F.xxhash64("src"), F.lit(num_shards)).cast("int"))
                .withColumn("dst_shard", F.pmod(F.xxhash64("dst"), F.lit(num_shards)).cast("int")))

            # Materialize independently. Every directed Phase 2 edge has exactly
            # one source shard, which is the foundation for lossless adjacency
            # sampling in the next Phase 3 implementation.
            (nodes.repartition(num_shards, "node_shard")
                .write.format("delta").mode("overwrite")
                .partitionBy("node_shard").save(nodes_path))
            (edges.repartition(num_shards, "src_shard")
                .write.format("delta").mode("overwrite")
                .partitionBy("src_shard").save(edges_path))

            node_counts = nodes.groupBy("node_shard").agg(
                F.count("*").alias("n_nodes"),
                F.sum(F.when(F.col("split") == "train", 1).otherwise(0)).alias("n_train"),
                F.sum(F.when(F.col("split") == "valid", 1).otherwise(0)).alias("n_val"),
                F.sum(F.when(F.col("split") == "test", 1).otherwise(0)).alias("n_test"),
            )
            edge_counts = edges.groupBy("src_shard").agg(F.count("*").alias("n_edges"))
            manifest = (node_counts.join(
                edge_counts, node_counts.node_shard == edge_counts.src_shard, "full_outer"
            ).select(
                F.coalesce(node_counts.node_shard, edge_counts.src_shard).cast("int").alias("shard_id"),
                F.coalesce("n_nodes", F.lit(0)).cast("long").alias("n_nodes"),
                F.coalesce("n_edges", F.lit(0)).cast("long").alias("n_edges"),
                F.coalesce("n_train", F.lit(0)).cast("long").alias("n_train"),
                F.coalesce("n_val", F.lit(0)).cast("long").alias("n_val"),
                F.coalesce("n_test", F.lit(0)).cast("long").alias("n_test"),
            ))
            manifest.write.format("delta").mode("overwrite").save(manifest_path)

            summary = manifest.agg(
                F.count("*").alias("n_shards"),
                F.sum("n_nodes").alias("n_nodes"),
                F.sum("n_edges").alias("n_edges"),
                F.min("n_nodes").alias("min_nodes"),
                F.max("n_nodes").alias("max_nodes"),
                F.max("n_edges").alias("max_edges"),
            ).first()
            elapsed = time.time() - started
            results[key] = {
                "n_shards": int(summary["n_shards"]),
                "n_nodes": int(summary["n_nodes"] or 0),
                "n_edges": int(summary["n_edges"] or 0),
                "min_nodes_per_shard": int(summary["min_nodes"] or 0),
                "max_nodes_per_shard": int(summary["max_nodes"] or 0),
                "max_edges_per_shard": int(summary["max_edges"] or 0),
                "loaded_checkpoint": False,
            }
            timing[("phase25", dataset, alg)] = elapsed
            print(f"  ✓ Phase 2.5 retained {results[key]['n_nodes']:,} nodes and {results[key]['n_edges']:,} edges "
                  f"across {results[key]['n_shards']:,} shards in {elapsed:.1f}s.")
            print(f"    Nodes/shard: {results[key]['min_nodes_per_shard']:,} to {results[key]['max_nodes_per_shard']:,}; "
                  f"largest source-adjacency shard: {results[key]['max_edges_per_shard']:,} edges.")

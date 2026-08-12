"""Phase 3.7: Spark + Delta cached feature propagation for scalable CPU training.

This is a SIGN-style preprocessing phase.  It performs a small, fixed number
of sparse neighborhood mean propagations, stores every hop in Delta, and avoids
repeating graph message passing during classifier epochs.  The aggregation uses
Spark ML's vector ``Summarizer.mean`` rather than ``collect_list`` or a Python
edge UDF.
"""

import time


def _delta_exists(spark, path):
    try:
        spark.read.format("delta").load(path).limit(1).count()
        return True
    except Exception:
        return False


def _load_graph(spark, paths, graph_source):
    """Load either the prototype Phase 2 graph or full Phase 0 graph tables."""
    from pyspark.sql import functions as F

    if graph_source == "phase2":
        nodes = spark.read.format("delta").load(paths["p2_nodes"]).select(
            "id", "label", "split", "features"
        )
        edges = spark.read.format("delta").load(paths["p2_edges"]).select("src", "dst")
    elif graph_source == "phase0":
        if not _delta_exists(spark, paths["original_nodes"]):
            raise RuntimeError(
                f"Delta table 'original_nodes' not found at '{paths['original_nodes']}'. "
                f"Ensure Phase 0 ingestion (RUN_PHASE0 / --run-phase0) is enabled so missing dataset tables are generated."
            )
        # Phase 0 nodes include all graph nodes; masks exist only for the
        # labelled OGB subset. Null split is intentional for unlabeled nodes.
        nodes = (spark.read.format("delta").load(paths["original_nodes"])
            .join(
                spark.read.format("delta").load(paths["masks"]).select("id", "split"),
                "id", "left",
            ).select("id", "label", "split", "features"))
        # ``edges`` is Phase 0's already symmetrized graph, which gives each
        # node both citation directions for propagation.
        edges = spark.read.format("delta").load(paths["edges"]).select("src", "dst")
    else:
        raise ValueError("PHASE37_GRAPH_SOURCE must be 'phase2' or 'phase0'")
    return nodes, edges


def run_phase37(spark, datasets, algorithms, get_paths_fn, timing, results, **kwargs):
    """Materialize $K$ cached mean-propagation feature hops in Delta.

    For hop $k$, each destination receives the mean previous-hop vector of its
    incoming neighbors. Isolated nodes retain their own previous-hop feature.
    Delta output makes later MLP/GAMLP experiments edge-free and resumable.
    """
    from pyspark.ml.functions import array_to_vector, vector_to_array
    from pyspark.ml.stat import Summarizer
    from pyspark.sql import functions as F

    graph_source = str(kwargs.get("graph_source", "phase2")).lower()
    num_hops = int(kwargs.get("num_hops", 2))
    num_partitions = int(kwargs.get("num_partitions", 512))
    force_rerun = bool(kwargs.get("force_rerun", False))
    if num_hops < 1 or num_partitions < 1:
        raise ValueError("Phase 3.7 requires positive hop and partition counts")

    for dataset in datasets:
        for alg in algorithms:
            key = (dataset, alg)
            paths = get_paths_fn(dataset, alg)
            base_path = f"{paths['p37_base'].rstrip('/')}/{graph_source}"
            final_path = f"{base_path}/features_k{num_hops}"
            started = time.time()

            print(f"\n{'=' * 60}\n  PHASE 3.7 — Delta cached feature propagation: {dataset} / {alg}\n{'=' * 60}")
            print(f"  Graph source={graph_source}; {num_hops} mean-propagation hops; "
                  f"{num_partitions} Spark partitions; no Python edge UDFs.")

            if not force_rerun and _delta_exists(spark, final_path):
                cached = spark.read.format("delta").load(final_path)
                summary = cached.agg(F.count("*").alias("n_nodes")).first()
                results[key] = {
                    "n_nodes": int(summary["n_nodes"]),
                    "num_hops": num_hops,
                    "graph_source": graph_source,
                    "loaded_checkpoint": True,
                }
                timing[("phase37", dataset, alg)] = 0.0
                print(f"  ✓ Reusing Delta propagated-feature checkpoint: {summary['n_nodes']:,} nodes.")
                continue

            nodes, edges = _load_graph(spark, paths, graph_source)
            # Stable materialization protects subsequent hops from recomputing
            # source graph joins; it also provides exact input size reporting.
            n_nodes = nodes.count()
            n_edges = edges.count()
            print(f"  Input graph: {n_nodes:,} nodes, {n_edges:,} directed propagation edges.")

            previous = nodes.select("id", F.col("features").alias("features"))
            hop_columns = [F.col("features").alias("x0")]
            for hop in range(1, num_hops + 1):
                hop_path = f"{base_path}/hop_{hop}"
                if not force_rerun and _delta_exists(spark, hop_path):
                    print(f"  Hop {hop}/{num_hops}: reusing Delta checkpoint.")
                    previous = spark.read.format("delta").load(hop_path).select("id", "features")
                else:
                    print(f"  Hop {hop}/{num_hops}: Spark vector mean aggregation and Delta checkpoint...")
                    # Summarizer.mean operates as a JVM-side vector aggregate.
                    # It avoids materializing neighbor feature lists in Python.
                    source_features = previous.select(
                        F.col("id").alias("src"),
                        array_to_vector(F.col("features")).alias("_src_vector"),
                    )
                    neighbor_means = (edges
                        .join(source_features, "src", "inner")
                        .repartition(num_partitions, "dst")
                        .groupBy("dst")
                        .agg(Summarizer.mean(F.col("_src_vector")).alias("_mean_vector"))
                        .select(
                            F.col("dst").alias("id"),
                            vector_to_array(F.col("_mean_vector")).alias("features"),
                        ))
                    # Preserve isolated nodes and all node identifiers.
                    next_hop = (previous.alias("old")
                        .join(neighbor_means.alias("mean"), "id", "left")
                        .select(
                            F.col("id"),
                            F.coalesce(F.col("mean.features"), F.col("old.features")).alias("features"),
                        ))
                    (next_hop.repartition(num_partitions, "id")
                        .write.format("delta").mode("overwrite").save(hop_path))
                    previous = spark.read.format("delta").load(hop_path).select("id", "features")
                hop_columns.append(F.col(f"h{hop}.features").alias(f"x{hop}"))

            # Build the compact, reusable classifier input once. Concatenation
            # is lossless over x0...xK; no graph edge access is needed later.
            features = nodes.alias("n")
            for hop in range(1, num_hops + 1):
                features = features.join(
                    spark.read.format("delta").load(f"{base_path}/hop_{hop}").alias(f"h{hop}"),
                    F.col("n.id") == F.col(f"h{hop}.id"),
                    "inner",
                )
            concat_terms = [F.col("n.features")] + [F.col(f"h{hop}.features") for hop in range(1, num_hops + 1)]
            final_features = features.select(
                F.col("n.id").alias("id"),
                F.col("n.label").alias("label"),
                F.col("n.split").alias("split"),
                F.concat(*concat_terms).alias("features"),
            )
            (final_features.repartition(num_partitions, "id")
                .write.format("delta").mode("overwrite").save(final_path))

            output_nodes = spark.read.format("delta").load(final_path).count()
            if output_nodes != n_nodes:
                raise RuntimeError(f"Phase 3.7 coverage failed: {output_nodes}/{n_nodes} nodes")
            elapsed = time.time() - started
            results[key] = {
                "n_nodes": int(output_nodes),
                "n_edges": int(n_edges),
                "num_hops": num_hops,
                "graph_source": graph_source,
                "loaded_checkpoint": False,
            }
            timing[("phase37", dataset, alg)] = elapsed
            print(f"  ✓ Propagation coverage verified: {output_nodes:,}/{n_nodes:,} nodes; "
                  f"{num_hops} Delta hops materialized in {elapsed:.1f}s.")
            print(f"    Classifier features are cached at {final_path}.")

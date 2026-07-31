"""Phase 3.8: edge-free global classifier over Phase 3.7 Delta features.

A multinomial logistic probe is intentionally the first classifier for the
cached SIGN-style features. Spark ML optimizes one global model using distributed
objective/gradient aggregation; this is fundamentally different from fitting
one disconnected model per Spark partition. No graph edges are read here.
"""

import time


def run_phase38(spark, datasets, algorithms, get_paths_fn, timing, results, **kwargs):
    """Fit and score a global Spark ML multinomial logistic classifier.

    Output metrics use the OGB train/valid/test split stored alongside Phase 3.7
    features. The phase is a scalable linear probe, not a claim of an MLP/GNN.
    """
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.functions import array_to_vector
    from pyspark.sql import functions as F
    from pyspark.storagelevel import StorageLevel

    graph_source = str(kwargs.get("graph_source", "phase2")).lower()
    num_hops = int(kwargs.get("num_hops", 2))
    max_iter = int(kwargs.get("max_iter", 30))
    reg_param = float(kwargs.get("reg_param", 1e-4))
    elastic_net_param = float(kwargs.get("elastic_net_param", 0.0))
    if max_iter < 1 or reg_param < 0 or not 0 <= elastic_net_param <= 1:
        raise ValueError("Invalid Phase 3.8 logistic-regression settings")

    def score(model, frame, split_name):
        predicted = model.transform(frame).select("label", "prediction")
        metrics = predicted.agg(
            F.count("*").alias("n"),
            F.sum(F.when(F.col("label") == F.col("prediction"), 1).otherwise(0)).alias("correct"),
        ).first()
        n = int(metrics["n"])
        if n == 0:
            raise RuntimeError(f"Phase 3.8 has no labelled {split_name} examples")
        return n, float(metrics["correct"]) / n

    for dataset in datasets:
        for alg in algorithms:
            key = (dataset, alg)
            paths = get_paths_fn(dataset, alg)
            feature_path = (
                f"{paths['p37_base'].rstrip('/')}/{graph_source}/features_k{num_hops}"
            )
            started = time.time()
            print(f"\n{'=' * 60}\n  PHASE 3.8 — Global edge-free multinomial classifier: {dataset} / {alg}\n{'=' * 60}")
            print("  Training one Spark ML global model on cached propagation features; no graph edges are read.")

            raw = spark.read.format("delta").load(feature_path)
            labelled = (raw
                .filter(F.col("label").isNotNull() & (F.col("label") >= 0) & F.col("split").isNotNull())
                .select(
                    F.col("label").cast("double").alias("label"),
                    F.col("split"),
                    array_to_vector(F.col("features")).alias("features"),
                )
                .persist(StorageLevel.MEMORY_AND_DISK))
            split_counts = {
                row["split"]: int(row["count"])
                for row in labelled.groupBy("split").count().collect()
            }
            required = ("train", "valid", "test")
            missing = [name for name in required if split_counts.get(name, 0) == 0]
            if missing:
                labelled.unpersist()
                raise RuntimeError(
                    "Phase 3.8 requires OGB train/valid/test labels in the Phase 3.7 graph; "
                    f"missing: {', '.join(missing)}"
                )
            print("  Labelled examples: " + ", ".join(
                f"{name}={split_counts[name]:,}" for name in required
            ))

            train = labelled.filter(F.col("split") == "train").select("label", "features")
            valid = labelled.filter(F.col("split") == "valid").select("label", "features")
            test = labelled.filter(F.col("split") == "test").select("label", "features")
            classifier = LogisticRegression(
                featuresCol="features",
                labelCol="label",
                predictionCol="prediction",
                family="multinomial",
                maxIter=max_iter,
                regParam=reg_param,
                elasticNetParam=elastic_net_param,
                standardization=True,
                aggregationDepth=4,
            )
            print(f"  Fitting distributed multinomial logistic model (maxIter={max_iter}, regParam={reg_param:g})...")
            model = classifier.fit(train)
            n_valid, valid_acc = score(model, valid, "valid")
            n_test, test_acc = score(model, test, "test")
            elapsed = time.time() - started
            labelled.unpersist()

            results[key] = {
                "model": "spark_multinomial_logistic",
                "graph_source": graph_source,
                "num_hops": num_hops,
                "n_train": split_counts["train"],
                "n_valid": n_valid,
                "n_test": n_test,
                "valid_acc": valid_acc,
                "test_acc": test_acc,
                "iterations": int(model.summary.totalIterations),
            }
            timing[("phase38", dataset, alg)] = elapsed
            print(f"  ✓ Global classifier complete in {elapsed:.1f}s; "
                  f"valid accuracy={valid_acc:.4f}, test accuracy={test_acc:.4f}; "
                  f"iterations={model.summary.totalIterations}.")

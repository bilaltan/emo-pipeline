#!/usr/bin/env python3
"""
GRL Experiment Runner - Local GNN Pipeline
Replaces pipeline_local.py and imports code directly from the modular `pipeline` package.
"""
import os
import sys
import time
import argparse

# Configure Java JVM arguments globally for Apache Arrow compatibility on Java 17/21
os.environ["JAVA_TOOL_OPTIONS"] = (
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
    "--add-opens=java.base/java.security=ALL-UNNAMED "
    "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
    "--add-opens=java.base/java.math=ALL-UNNAMED"
)

def load_baseline_checkpoint(dataset, suffix, results_dict, timing, s3_bucket, experiment_name, local_data_dir=None):
    """Loads baseline result dictionary and timings from S3/local checkpoint."""
    if local_data_dir:
        ckpt_path = os.path.join(local_data_dir, "gnn-bench-checkpoint", "phase4", experiment_name, f"{dataset}_{suffix}.json")
        if os.path.exists(ckpt_path):
            import json
            try:
                with open(ckpt_path, "r") as f:
                    data = json.load(f)
                results_dict[dataset] = data['results']
                for k, v in data['timing'].items():
                    timing[eval(k)] = v
                print(f"    ✓ Loaded baseline {suffix} from local checkpoint.")
                return True
            except Exception as e:
                print(f"    ⚠️ Warning loading local baseline checkpoint: {e}")
    else:
        import boto3
        import json
        import tempfile
        from botocore.exceptions import ClientError
        s3_client = boto3.client('s3')
        s3_key = f"gnn-bench-checkpoint/phase4/{experiment_name}/{dataset}_{suffix}.json"
        tmp_file = tempfile.mktemp(suffix=".json")
        try:
            s3_client.download_file(s3_bucket, s3_key, tmp_file)
            with open(tmp_file, "r") as f:
                data = json.load(f)
            results_dict[dataset] = data['results']
            for k, v in data['timing'].items():
                timing[eval(k)] = v
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
            print(f"    ✓ Loaded baseline {suffix} from S3 checkpoint.")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] != "404":
                print(f"    ⚠️ Warning loading S3 baseline checkpoint: {e}")
        except Exception as e:
            print(f"    ⚠️ Warning loading baseline checkpoint: {e}")
    return False

def save_baseline_checkpoint(dataset, suffix, results_dict, timing, s3_bucket, experiment_name, local_data_dir=None):
    """Saves baseline result dictionary and timings to S3/local checkpoint."""
    try:
        import json
        # Extract related timing keys
        timing_data = {}
        for k, v in timing.items():
            if isinstance(k, tuple) and len(k) >= 2 and k[0] in (f'phase{suffix}', f'phase{suffix}_node', f'phase{suffix}_link') and k[1] == dataset:
                timing_data[repr(k)] = v
        
        payload = {
            'results': results_dict.get(dataset),
            'timing': timing_data
        }
        
        if local_data_dir:
            ckpt_dir = os.path.join(local_data_dir, "gnn-bench-checkpoint", "phase4", experiment_name)
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"{dataset}_{suffix}.json")
            with open(ckpt_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"    ✓ Saved baseline {suffix} checkpoint locally.")
        else:
            import tempfile
            import boto3
            tmp_file = tempfile.mktemp(suffix=".json")
            with open(tmp_file, "w") as f:
                json.dump(payload, f, indent=2)
            
            s3_client = boto3.client('s3')
            s3_key = f"gnn-bench-checkpoint/phase4/{experiment_name}/{dataset}_{suffix}.json"
            s3_client.upload_file(tmp_file, s3_bucket, s3_key)
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
            print(f"    ✓ Saved baseline {suffix} checkpoint to S3.")
    except Exception as e:
        print(f"    ⚠️ Failed to save baseline checkpoint: {e}")

def main():
    t_pipeline_start = time.time()

    # 1. PARSE COMMAND LINE ARGUMENTS
    parser = argparse.ArgumentParser(
        description="GRL Experiment Runner - Local GNN Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--experiment-name", type=str, default="local_run3",
                        help="Unique label identifying local outputs")
    parser.add_argument("--datasets", type=str, default="ogbn-mag",
                        help="Comma-separated list of datasets to run")
    parser.add_argument("--algorithms", type=str, default="lpa, louvain, leiden, igraph_lpa, metis",
                        help="Algorithms to run (lpa, louvain, leiden, igraph_lpa, metis)")
    parser.add_argument("--task-type", type=str, default="both", choices=["both", "node_classification", "link_prediction"],
                        help="GNN evaluation task types to execute")
    
    parser.add_argument("--run-phase0", action="store_true", default=True,
                        help="Force Delta Lake ingestion phase")
    parser.add_argument("--no-phase0", action="store_false", dest="run_phase0",
                        help="Skip Delta Lake ingestion phase")
    parser.add_argument("--force-reingest", action="store_true", default=False,
                        help="Force Phase 0 to re-download and overwrite existing Delta tables")
    parser.add_argument("--use-ogb-splits", type=str, default="true", choices=["true", "false"],
                        help="Use OGB official splits (true) or stratified 60/20/20 random split (false)")
    parser.add_argument("--min-community-size", type=int, default=10,
                        help="Filter out communities smaller than this size threshold")
    parser.add_argument("--tiny-comm-handling", type=str, default="misc", choices=["misc", "drop"],
                        help="Drop small communities or group them into community_id = -1 ('misc')")
    parser.add_argument("--expand-boundary-nodes", type=str, default="true", choices=["true", "false"],
                        help="Enable 1-hop boundary node expansion for local subgraphs")
    parser.add_argument("--global-mapping", type=str, default="true", choices=["true", "false"],
                        help="Use global OGB masks for local GNN UDF training")
    parser.add_argument("--run-phase3", action="store_true", default=True,
                        help="Run Phase 3: Standard parallel UDF training")
    parser.add_argument("--no-phase3", action="store_false", dest="run_phase3",
                        help="Skip Phase 3: Standard parallel UDF training")
    parser.add_argument("--run-phase3b", action="store_true", default=True,
                        help="Run Phase 3b: GNN parallel training with CaaN Global Graph")
    parser.add_argument("--no-phase3b", action="store_false", dest="run_phase3b",
                        help="Skip Phase 3b: GNN parallel training with CaaN Global Graph")
    parser.add_argument("--run-phase4", action="store_true", default=True,
                        help="Run Phase 4 & 4b-4g: Full-graph baselines")
    parser.add_argument("--no-phase4", action="store_false", dest="run_phase4",
                        help="Skip Phase 4 & 4b-4g: Full-graph baselines")

    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden dimension for GCN model layers")
    parser.add_argument("--num-epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout probability")

    args = parser.parse_args()

    # 2. INITIALIZE SPARK SESSION FOR LOCAL RUNNING
    print("\n" + "="*80)
    print("  INITIALIZING SPARK SESSION (LOCAL SINGLE-NODE MODE)")
    print("="*80)
    
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    local_data_dir = os.path.join(script_dir, "local_data")
    os.makedirs(local_data_dir, exist_ok=True)
    
    if sys.platform == "darwin":
        os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"
    else:
        if os.path.exists("/usr/lib/jvm/java-17-openjdk-amd64"):
            os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
            
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

    extra_java_opts = (
        "--add-opens=java.base/java.nio=ALL-UNNAMED "
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
        "--add-opens=java.base/java.lang=ALL-UNNAMED "
        "--add-opens=java.base/java.util=ALL-UNNAMED "
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
        "--add-opens=java.base/java.security=ALL-UNNAMED "
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
        "--add-opens=java.base/java.math=ALL-UNNAMED"
    )
    
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName(f"GRL-Local-{args.experiment_name}") \
        .config("spark.master", "local[*]") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.memory", "24g") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.default.parallelism", "2") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.driver.extraJavaOptions", extra_java_opts) \
        .config("spark.executor.extraJavaOptions", extra_java_opts) \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.databricks.delta.retentionDurationCheck.enabled", "false") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.0,graphframes:graphframes:0.8.3-spark3.5-s_2.12") \
        .config("spark.local.dir", os.path.join(local_data_dir, "spark_tmp")) \
        .getOrCreate()

    sc = spark.sparkContext
    print("  ✓ SparkSession successfully configured and initialized locally.")

    # 3. IMPORT CORE CONFIG AND PIPELINE MODULES
    # Ensure current directory is on PYTHONPATH
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        
    import experiment_config as config
    from pipeline.phases import (
        run_phase0,
        run_phase1,
        print_phase1_stats,
        run_phase2,
        run_phase3,
        run_phase3b,
        run_phase4,
        run_phase4b,
        run_phase4c,
        run_phase4d,
        run_phase4e,
        run_phase4f,
        run_phase4g,
        run_phase4h,
        print_accuracy_table,
        print_timing_table,
        save_plots_and_xlsx,
        print_summary
    )
    from pipeline.utils.paths import get_paths

    # Set parameters
    EXPERIMENT_NAME = args.experiment_name
    DATASETS_TO_RUN = [d.strip() for d in args.datasets.split(",") if d.strip()]
    ALGORITHMS_TO_RUN = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    TASK_TYPE = args.task_type
    
    RUN_PHASE0 = args.run_phase0
    FORCE_REINGEST = args.force_reingest
    USE_OGB_SPLITS = (args.use_ogb_splits == "true")
    MIN_COMMUNITY_SIZE = args.min_community_size
    TINY_COMM_HANDLING = args.tiny_comm_handling
    EXPAND_BOUNDARY_NODES = (args.expand_boundary_nodes == "true")
    USE_GLOBAL_MAPPING = (args.global_mapping == "true")

    GCN_CFG = {
        'hidden_dim': args.hidden_dim,
        'num_epochs': args.num_epochs,
        'lr':         args.lr,
        'dropout':    args.dropout,
    }
    
    BASELINE_CFG = {
        'epochs':     args.num_epochs,
        'batch':      getattr(config, 'BASELINE_BATCH', 1024),
        'fanout':     [10, 10],
        'lr':         args.lr,
        'hidden_dim': args.hidden_dim,
        'dropout':    args.dropout,
    }

    timing = {}
    phase1_results = {}
    phase2_results = {}
    phase3_results = {}
    phase3b_results = {}
    phase4_results = {}
    phase4b_results = {}
    phase4c_results = {}
    phase4d_results = {}
    phase4e_results = {}
    phase4f_results = {}
    phase4g_results = {}
    phase4h_results = {}

    print(f"\n  Run Profile: {EXPERIMENT_NAME}")
    print(f"  └─ Datasets: {DATASETS_TO_RUN}")
    print(f"  └─ Algorithms: {ALGORITHMS_TO_RUN}")
    print(f"  └─ GNN Architecture: GCN (epochs={args.num_epochs}, lr={args.lr}, hidden={args.hidden_dim})")
    print(f"  └─ Global Mapping: {USE_GLOBAL_MAPPING} | Boundary Expansion: {EXPAND_BOUNDARY_NODES}")
    print(f"  └─ Ingestion Phase (Phase 0): {RUN_PHASE0}")

    # Build curried path function bound to local data directory
    get_paths_fn = lambda dataset, alg=None: get_paths(
        dataset, alg,
        experiment_name=EXPERIMENT_NAME,
        local_data_dir=local_data_dir
    )

    # Phase 0: Delta Ingestion
    run_phase0(
        spark, sc,
        datasets        = DATASETS_TO_RUN,
        run_phase0_flag = RUN_PHASE0,
        use_ogb_splits  = USE_OGB_SPLITS,
        random_seed     = config.RANDOM_SEED,
        dataset_cfg     = config.DATASET_CFG,
        get_paths_fn    = get_paths_fn,
        timing          = timing,
        force_reingest  = FORCE_REINGEST
    )

    # Resolve FORCE_RERUN config parameter
    FORCE_RERUN = getattr(config, 'FORCE_RERUN', False)

    # Phase 1: Community Detection
    run_phase1(
        spark, sc,
        datasets     = DATASETS_TO_RUN,
        algorithms   = ALGORITHMS_TO_RUN,
        lpa_max_iter = config.LPA_MAX_ITER,
        resolution   = getattr(config, 'RESOLUTION', 1.0),
        random_seed  = config.RANDOM_SEED,
        min_size     = MIN_COMMUNITY_SIZE,
        dataset_cfg  = config.DATASET_CFG,
        get_paths_fn = get_paths_fn,
        timing       = timing,
        results      = phase1_results,
        metis_k      = getattr(config, 'METIS_K', 100),
        force_rerun  = FORCE_RERUN
    )

    # Print Phase 1 stats
    print_phase1_stats(
        spark,
        datasets     = DATASETS_TO_RUN,
        algorithms   = ALGORITHMS_TO_RUN,
        min_size     = MIN_COMMUNITY_SIZE,
        get_paths_fn = get_paths_fn,
        results      = phase1_results
    )

    # Phase 2: Induced Subgraph Extraction
    run_phase2(
        spark, sc,
        datasets           = DATASETS_TO_RUN,
        algorithms         = ALGORITHMS_TO_RUN,
        use_global_mapping = USE_GLOBAL_MAPPING,
        min_size           = MIN_COMMUNITY_SIZE,
        get_paths_fn       = get_paths_fn,
        timing             = timing,
        results            = phase2_results,
        tiny_comm_handling = TINY_COMM_HANDLING,
        expand_boundary_nodes = EXPAND_BOUNDARY_NODES,
        force_rerun        = FORCE_RERUN
    )

    # Phase 3: Parallel GNN UDF Training
    if getattr(args, 'run_phase3', True):
        run_phase3(
            spark, sc,
            datasets           = DATASETS_TO_RUN,
            algorithms         = ALGORITHMS_TO_RUN,
            use_global_mapping = USE_GLOBAL_MAPPING,
            dataset_cfg        = config.DATASET_CFG,
            gcn_cfg            = GCN_CFG,
            get_paths_fn       = get_paths_fn,
            timing             = timing,
            results            = phase3_results,
            task_type          = TASK_TYPE,
            models             = config.GNN_MODELS,
            force_rerun        = FORCE_RERUN,
            local_data_dir     = local_data_dir,
            experiment_name    = EXPERIMENT_NAME
        )

    # Phase 3b: CaaN Global Graph GNN Training
    if args.run_phase3b:
        run_phase3b(
            spark, sc,
            datasets           = DATASETS_TO_RUN,
            algorithms         = ALGORITHMS_TO_RUN,
            use_global_mapping = USE_GLOBAL_MAPPING,
            dataset_cfg        = config.DATASET_CFG,
            gcn_cfg            = GCN_CFG,
            get_paths_fn       = get_paths_fn,
            timing             = timing,
            results            = phase3b_results,
            task_type          = TASK_TYPE,
            models             = config.GNN_MODELS,
            min_size           = MIN_COMMUNITY_SIZE,
            force_rerun        = FORCE_RERUN,
            local_data_dir     = local_data_dir,
            experiment_name    = EXPERIMENT_NAME
        )

    # Phase 4: Full-Graph Baselines
    if getattr(args, 'run_phase4', True):
        # 1. GraphSAGE Baseline
        if 'sage' in config.GNN_MODELS:
            datasets_for_4 = []
            for dataset in DATASETS_TO_RUN:
                if not FORCE_RERUN and load_baseline_checkpoint(dataset, '4', phase4_results, timing, None, EXPERIMENT_NAME, local_data_dir):
                    pass
                else:
                    datasets_for_4.append(dataset)
            if datasets_for_4:
                print("\n[PHASE 4] - GraphSAGE Full-Graph Global Baseline (PyG)")
                run_phase4(
                    spark, sc,
                    datasets     = datasets_for_4,
                    dataset_cfg  = config.DATASET_CFG,
                    baseline_cfg = BASELINE_CFG,
                    get_paths_fn = get_paths_fn,
                    timing       = timing,
                    results      = phase4_results,
                    task_type    = TASK_TYPE,
                    n_baseline_runs = getattr(config, 'N_BASELINE_RUNS', 3)
                )
                for dataset in datasets_for_4:
                    save_baseline_checkpoint(dataset, '4', phase4_results, timing, None, EXPERIMENT_NAME, local_data_dir)

        # 2. DistDGL Baseline Simulation
        datasets_for_4b = []
        for dataset in DATASETS_TO_RUN:
            if not FORCE_RERUN and load_baseline_checkpoint(dataset, '4b', phase4b_results, timing, None, EXPERIMENT_NAME, local_data_dir):
                pass
            else:
                datasets_for_4b.append(dataset)
        if datasets_for_4b:
            print("\n[PHASE 4b] - DistDGL Distributed Simulation Baseline")
            run_phase4b(
                spark, sc,
                datasets     = datasets_for_4b,
                dataset_cfg  = config.DATASET_CFG,
                baseline_cfg = BASELINE_CFG,
                get_paths_fn = get_paths_fn,
                timing       = timing,
                results      = phase4b_results,
                task_type    = TASK_TYPE
            )
            for dataset in datasets_for_4b:
                save_baseline_checkpoint(dataset, '4b', phase4b_results, timing, None, EXPERIMENT_NAME, local_data_dir)

        # 3. ARMA Baseline
        datasets_for_4c = []
        for dataset in DATASETS_TO_RUN:
            if not FORCE_RERUN and load_baseline_checkpoint(dataset, '4c', phase4c_results, timing, None, EXPERIMENT_NAME, local_data_dir):
                pass
            else:
                datasets_for_4c.append(dataset)
        if datasets_for_4c:
            print("\n[PHASE 4c] - ARMA Full-Graph Global Baseline (PyG)")
            run_phase4c(
                spark, sc,
                datasets     = datasets_for_4c,
                dataset_cfg  = config.DATASET_CFG,
                baseline_cfg = BASELINE_CFG,
                get_paths_fn = get_paths_fn,
                timing       = timing,
                results      = phase4c_results,
                task_type    = TASK_TYPE,
                n_baseline_runs = getattr(config, 'N_BASELINE_RUNS', 3)
            )
            for dataset in datasets_for_4c:
                save_baseline_checkpoint(dataset, '4c', phase4c_results, timing, None, EXPERIMENT_NAME, local_data_dir)

        # 4. ASAP Baseline
        datasets_for_4d = []
        for dataset in DATASETS_TO_RUN:
            if not FORCE_RERUN and load_baseline_checkpoint(dataset, '4d', phase4d_results, timing, None, EXPERIMENT_NAME, local_data_dir):
                pass
            else:
                datasets_for_4d.append(dataset)
        if datasets_for_4d:
            print("\n[PHASE 4d] - ASAP Full-Graph Global Baseline (PyG)")
            run_phase4d(
                spark, sc,
                datasets     = datasets_for_4d,
                dataset_cfg  = config.DATASET_CFG,
                baseline_cfg = BASELINE_CFG,
                get_paths_fn = get_paths_fn,
                timing       = timing,
                results      = phase4d_results,
                task_type    = TASK_TYPE,
                n_baseline_runs = getattr(config, 'N_BASELINE_RUNS', 3)
            )
            for dataset in datasets_for_4d:
                save_baseline_checkpoint(dataset, '4d', phase4d_results, timing, None, EXPERIMENT_NAME, local_data_dir)

        # 5. GAT Baseline
        if 'gat' in config.GNN_MODELS:
            datasets_for_4e = []
            for dataset in DATASETS_TO_RUN:
                if not FORCE_RERUN and load_baseline_checkpoint(dataset, '4e', phase4e_results, timing, None, EXPERIMENT_NAME, local_data_dir):
                    pass
                else:
                    datasets_for_4e.append(dataset)
            if datasets_for_4e:
                print("\n[PHASE 4e] - GAT Full-Graph Global Baseline (PyG)")
                run_phase4e(
                    spark, sc,
                    datasets     = datasets_for_4e,
                    dataset_cfg  = config.DATASET_CFG,
                    baseline_cfg = BASELINE_CFG,
                    get_paths_fn = get_paths_fn,
                    timing       = timing,
                    results      = phase4e_results,
                    task_type    = TASK_TYPE,
                    n_baseline_runs = getattr(config, 'N_BASELINE_RUNS', 3)
                )
                for dataset in datasets_for_4e:
                    save_baseline_checkpoint(dataset, '4e', phase4e_results, timing, None, EXPERIMENT_NAME, local_data_dir)

        # 6. Graph Transformer Baseline
        if 'transformer' in config.GNN_MODELS:
            datasets_for_4f = []
            for dataset in DATASETS_TO_RUN:
                if not FORCE_RERUN and load_baseline_checkpoint(dataset, '4f', phase4f_results, timing, None, EXPERIMENT_NAME, local_data_dir):
                    pass
                else:
                    datasets_for_4f.append(dataset)
            if datasets_for_4f:
                print("\n[PHASE 4f] - Graph Transformer Full-Graph Global Baseline (PyG)")
                run_phase4f(
                    spark, sc,
                    datasets     = datasets_for_4f,
                    dataset_cfg  = config.DATASET_CFG,
                    baseline_cfg = BASELINE_CFG,
                    get_paths_fn = get_paths_fn,
                    timing       = timing,
                    results      = phase4f_results,
                    task_type    = TASK_TYPE,
                    n_baseline_runs = getattr(config, 'N_BASELINE_RUNS', 3)
                )
                for dataset in datasets_for_4f:
                    save_baseline_checkpoint(dataset, '4f', phase4f_results, timing, None, EXPERIMENT_NAME, local_data_dir)

        # 7. ClusterSCL Baseline
        if 'clusterscl' in config.GNN_MODELS:
            datasets_for_4g = []
            for dataset in DATASETS_TO_RUN:
                if not FORCE_RERUN and load_baseline_checkpoint(dataset, '4g', phase4g_results, timing, None, EXPERIMENT_NAME, local_data_dir):
                    pass
                else:
                    datasets_for_4g.append(dataset)
            if datasets_for_4g:
                print("\n[PHASE 4g] - ClusterSCL Full-Graph Global Baseline (PyG)")
                run_phase4g(
                    spark, sc,
                    datasets     = datasets_for_4g,
                    dataset_cfg  = config.DATASET_CFG,
                    baseline_cfg = BASELINE_CFG,
                    get_paths_fn = get_paths_fn,
                    timing       = timing,
                    results      = phase4g_results,
                    task_type    = TASK_TYPE,
                    n_baseline_runs = getattr(config, 'N_BASELINE_RUNS', 3)
                )
                for dataset in datasets_for_4g:
                    save_baseline_checkpoint(dataset, '4g', phase4g_results, timing, None, EXPERIMENT_NAME, local_data_dir)

        # 8. GATv2 Baseline
        if 'gatv2' in config.GNN_MODELS:
            datasets_for_4h = []
            for dataset in DATASETS_TO_RUN:
                if not FORCE_RERUN and load_baseline_checkpoint(dataset, '4h', phase4h_results, timing, None, EXPERIMENT_NAME, local_data_dir):
                    pass
                else:
                    datasets_for_4h.append(dataset)
            if datasets_for_4h:
                print("\n[PHASE 4h] - GATv2 Full-Graph Global Baseline (PyG)")
                run_phase4h(
                    spark, sc,
                    datasets     = datasets_for_4h,
                    dataset_cfg  = config.DATASET_CFG,
                    baseline_cfg = BASELINE_CFG,
                    get_paths_fn = get_paths_fn,
                    timing       = timing,
                    results      = phase4h_results,
                    task_type    = TASK_TYPE,
                    n_baseline_runs = getattr(config, 'N_BASELINE_RUNS', 3)
                )
                for dataset in datasets_for_4h:
                    save_baseline_checkpoint(dataset, '4h', phase4h_results, timing, None, EXPERIMENT_NAME, local_data_dir)

    # Phase 5: Metrics Aggregation & Reporting
    print("\n[PHASE 5] - Metrics Analysis & Local Excel Export")
    print_accuracy_table(
        datasets        = DATASETS_TO_RUN,
        algorithms      = ALGORITHMS_TO_RUN,
        phase3_results  = phase3_results,
        phase4_results  = phase4_results,
        phase4b_results = phase4b_results,
        phase4c_results = phase4c_results,
        phase4d_results = phase4d_results,
        phase4e_results = phase4e_results,
        phase4f_results = phase4f_results,
        phase4g_results = phase4g_results,
        phase4h_results = phase4h_results,
        phase3b_results = phase3b_results,
        gnn_models      = config.GNN_MODELS
    )

    print_timing_table(
        datasets   = DATASETS_TO_RUN,
        algorithms = ALGORITHMS_TO_RUN,
        timing     = timing,
        gnn_models = config.GNN_MODELS
    )

    save_plots_and_xlsx(
        datasets        = DATASETS_TO_RUN,
        algorithms      = ALGORITHMS_TO_RUN,
        phase3_results  = phase3_results,
        phase4_results  = phase4_results,
        timing          = timing,
        experiment_name = EXPERIMENT_NAME,
        s3_bucket       = None,  # Bypassed
        phase4b_results = phase4b_results,
        phase4c_results = phase4c_results,
        phase4d_results = phase4d_results,
        phase4e_results = phase4e_results,
        phase4f_results = phase4f_results,
        phase4g_results = phase4g_results,
        phase4h_results = phase4h_results,
        phase3b_results = phase3b_results,
        local_data_dir  = local_data_dir,
        gnn_models      = config.GNN_MODELS
    )

    # Final Summary Report
    print("\n" + "="*80)
    print("  FINAL PIPELINE REPORT SUMMARY")
    print("="*80)
    print_summary(
        experiment_name    = EXPERIMENT_NAME,
        datasets           = DATASETS_TO_RUN,
        algorithms         = ALGORITHMS_TO_RUN,
        use_global_mapping = USE_GLOBAL_MAPPING,
        min_size           = MIN_COMMUNITY_SIZE,
        phase1_results     = phase1_results,
        phase2_results     = phase2_results,
        phase3_results     = phase3_results,
        phase4_results     = phase4_results,
        timing             = timing,
        phase4b_results    = phase4b_results,
        phase4c_results    = phase4c_results,
        phase4d_results    = phase4d_results,
        phase4e_results    = phase4e_results,
        phase4f_results    = phase4f_results,
        phase4g_results    = phase4g_results,
        phase4h_results    = phase4h_results,
        phase3b_results    = phase3b_results,
        gnn_models         = config.GNN_MODELS
    )

    t_elapsed = time.time() - t_pipeline_start
    print(f"\n[SUCCESS] Local pipeline execution completed in {t_elapsed:.1f} seconds.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

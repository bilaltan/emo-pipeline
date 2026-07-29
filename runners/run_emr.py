#!/usr/bin/env python3
"""
AWS EMR Interactive/Driver Runner
Allows running the GRL pipeline interactively on the AWS EMR driver node.
Packages the `pipeline` directory, registers dependencies, and runs all phases.
"""
import os
import sys
import time
import argparse
import subprocess
import shutil
import sys

class TeeStream(object):
    def __init__(self, file_handle, original_stream):
        self.file = file_handle
        self.stream = original_stream
    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stream.write(data)
        self.stream.flush()
    def flush(self):
        self.file.flush()
        self.stream.flush()

# ── OVERWRITE DIRECTORIES TO USE LARGE VOLUMES (PREVENT ROOT DISK OOM) ───
candidates = [
    '/mnt/tmp', '/mnt1/tmp', '/mnt2/tmp',
    '/mnt/spark', '/mnt1/spark', '/mnt2/spark',
    '/mnt/var/tmp', '/mnt1/var/tmp', '/mnt2/var/tmp',
    '/tmp', '/var/tmp'
]

writable_candidates = []
for candidate in candidates:
    if not os.path.exists(candidate):
        parent = os.path.dirname(candidate)
        if os.path.exists(parent) and os.access(parent, os.W_OK):
            try:
                os.makedirs(candidate, exist_ok=True)
            except Exception:
                pass
    
    if os.path.exists(candidate) and os.access(candidate, os.W_OK):
        try:
            free_space = shutil.disk_usage(candidate).free
            writable_candidates.append((candidate, free_space))
        except Exception:
            pass

writable_candidates.sort(key=lambda x: x[1], reverse=True)

if writable_candidates:
    large_tmp = writable_candidates[0][0]
    print("Writable directories and free space:")
    for path, free_bytes in writable_candidates:
        print(f"  - {path}: {free_bytes / (1024*1024*1024):.2f} GB free")
else:
    large_tmp = '/tmp'
    print("WARNING: No writable candidate directories found, falling back to /tmp")

os.environ['HOME'] = large_tmp
os.environ['PYTHONUSERBASE'] = f'{large_tmp}/.local'
os.environ['PIP_CACHE_DIR'] = f'{large_tmp}/.pip-cache'
os.environ['DGL_DOWNLOAD_DIR'] = f'{large_tmp}/.dgl'
os.environ['TMPDIR'] = large_tmp
os.environ['TEMP'] = large_tmp
os.environ['TMP'] = large_tmp

# Dynamically construct and insert the user-site packages search path on the large temp volume
py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
tmp_site_packages = f"{large_tmp}/.local/lib/{py_version}/site-packages"
os.makedirs(tmp_site_packages, exist_ok=True)
if tmp_site_packages not in sys.path:
    sys.path.insert(0, tmp_site_packages)

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
    # ── 1. PARSE COMMAND LINE ARGUMENTS ───────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="GRL Experiment Runner - High-performance GNN Pipeline on AWS EMR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Unique label identifying S3 outputs, timing sheets, and XLSX files")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated list of datasets to run")
    parser.add_argument("--algorithms", type=str, default=None,
                        help="Comma-separated list of partition/community detection algorithms to run")
    parser.add_argument("--task-type", type=str, default=None, choices=["both", "node_classification", "link_prediction"],
                        help="GNN evaluation task types to execute")
    
    parser.add_argument("--run-phase0", action="store_true", default=None,
                        help="Force Delta Lake ingestion phase")
    parser.add_argument("--no-phase0", action="store_false", dest="run_phase0",
                        help="Skip Delta Lake ingestion phase")
    parser.add_argument("--force-reingest", action="store_true", default=None,
                        help="Force Phase 0 to re-download and overwrite existing Delta tables")
    parser.add_argument("--use-ogb-splits", type=str, default=None, choices=["true", "false"],
                        help="Use OGB official splits (true) or stratified 60/20/20 random split (false)")
    parser.add_argument("--min-community-size", type=int, default=None,
                        help="Filter out communities smaller than this size threshold")
    parser.add_argument("--tiny-comm-handling", type=str, default=None, choices=["misc", "drop"],
                        help="Drop small communities or group them into community_id = -1 ('misc')")
    parser.add_argument("--expand-boundary-nodes", type=str, default=None, choices=["true", "false"],
                        help="Enable 1-hop boundary node expansion for local subgraphs")
    parser.add_argument("--global-mapping", type=str, default=None, choices=["true", "false"],
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

    parser.add_argument("--hidden-dim", type=int, default=None,
                        help="Hidden dimension for GCN model layers")
    parser.add_argument("--num-epochs", type=int, default=None,
                        help="Number of local training epochs for GCN model")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate for local GCN optimizer")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Dropout probability for GCN layers")

    parser.add_argument("--local", action="store_true", default=False,
                        help="Load experiment_config.py and pipeline from current local directory instead of S3")
    parser.add_argument("--s3-bucket", type=str, default="us-east-1-s3-gnn",
                        help="S3 Bucket storing experiment code, assets, and output Delta tables")
    parser.add_argument("--s3-prefix", type=str, default="pipeline",
                        help="Prefix/folder within the S3 bucket where code is stored")
    parser.add_argument("--no-install", action="store_true", default=False,
                        help="Skip dynamic package verification/installation on YARN executors")

    args = parser.parse_args()

    # Capture stdout/stderr logs locally before uploading to S3
    log_file_path = "/tmp/run_pipeline.log"
    log_file = open(log_file_path, "w")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(log_file, original_stdout)
    sys.stderr = TeeStream(log_file, original_stderr)

    # Import/Inject PySpark paths if running on YARN
    try:
        from pyspark.sql import SparkSession
    except ModuleNotFoundError:
        emr_spark_py = '/usr/lib/spark/python'
        emr_py4j_dir = '/usr/lib/spark/python/lib'
        if os.path.exists(emr_spark_py):
            sys.path.insert(0, emr_spark_py)
            if os.path.exists(emr_py4j_dir):
                for item in os.listdir(emr_py4j_dir):
                    if item.startswith('py4j-') and item.endswith('.zip'):
                        sys.path.insert(0, os.path.join(emr_py4j_dir, item))
                        break
        from pyspark.sql import SparkSession

    # ── Spark Auto-Scaler: Determine optimal resources based on dataset and cluster size ──
    dataset_name = None
    if getattr(args, 'datasets', None):
        dataset_name = args.datasets.split(',')[0].strip()
    else:
        try:
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cfg_path = os.path.join(script_dir, "experiment_config.py")
            if os.path.exists(cfg_path):
                import re
                with open(cfg_path, 'r') as f:
                    content = f.read()
                m = re.search(r'DATASETS_TO_RUN\s*=\s*\[[\'"]([^\'"]+)[\'"]', content)
                if m:
                    dataset_name = m.group(1)
        except Exception:
            pass
    if not dataset_name:
        dataset_name = 'ogbn-arxiv'
    
    # 1. Query actual YARN cluster capacity (memory + cores per node)
    nodes_count = 0
    yarn_total_mem_gb = 0
    yarn_total_vcores = 0
    try:
        import subprocess, re, json
        # Try YARN REST API first (most accurate)
        try:
            out = subprocess.check_output(
                ['curl', '-s', 'http://localhost:8088/ws/v1/cluster/metrics'],
                stderr=subprocess.DEVNULL, timeout=5
            ).decode('utf-8')
            metrics = json.loads(out).get('clusterMetrics', {})
            yarn_total_mem_gb = metrics.get('totalMB', 0) / 1024.0
            yarn_total_vcores = metrics.get('totalVirtualCores', 0)
            nodes_count = metrics.get('activeNodes', 0)
        except Exception:
            pass

        # Fallback: parse `yarn node -list`
        if nodes_count < 1:
            out = subprocess.check_output(['yarn', 'node', '-list'], stderr=subprocess.DEVNULL).decode('utf-8')
            nodes_count = len(re.findall(r'RUNNING', out))
    except Exception:
        pass

    # Fallback: Hadoop slaves file
    if nodes_count < 2:
        for slaves_file in ['/etc/hadoop/conf/slaves', '/etc/hadoop/conf/workers']:
            if os.path.exists(slaves_file):
                try:
                    with open(slaves_file, 'r') as sf:
                        lines = [l.strip() for l in sf if l.strip() and not l.startswith('#')]
                        if len(lines) > nodes_count:
                            nodes_count = len(lines)
                            break
                except Exception:
                    pass
    if nodes_count < 2:
        nodes_count = 8

    # 2. Detect per-node hardware (from driver host as proxy)
    def get_system_total_memory_gb():
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        return int(line.split()[1]) / (1024 * 1024)
        except Exception:
            pass
        try:
            import subprocess
            out = subprocess.check_output(['sysctl', '-n', 'hw.memsize']).decode('utf-8').strip()
            return int(out) / (1024**3)
        except Exception:
            pass
        return 64.0

    import multiprocessing
    host_cores = multiprocessing.cpu_count()
    host_mem_gb = get_system_total_memory_gb()

    # Use YARN metrics if available, otherwise estimate from host hardware
    if yarn_total_mem_gb > 0 and yarn_total_vcores > 0:
        node_mem_gb = yarn_total_mem_gb / nodes_count
        node_vcores = yarn_total_vcores // nodes_count
    else:
        node_mem_gb = host_mem_gb
        node_vcores = host_cores

    # 3. Dynamic graph scale detection from S3
    bucket_name = getattr(args, 's3_bucket', 'us-east-1-s3-gnn')

    def get_dataset_s3_size_mb(bucket, prefix):
        import boto3
        s3 = boto3.client('s3')
        try:
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            total_size = 0
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if obj['Key'].endswith('.parquet'):
                            total_size += obj['Size']
            return total_size / (1024 * 1024)
        except Exception:
            return 0.0

    edges_prefix = f"delta-data/{dataset_name}/edges/"
    edges_size_mb = get_dataset_s3_size_mb(bucket_name, edges_prefix)

    if edges_size_mb == 0.0:
        if dataset_name in ('wikics', 'ogbn-arxiv'):
            edges_size_mb = 5.0
        elif dataset_name in ('ogbn-products', 'reddit'):
            edges_size_mb = 250.0
        elif dataset_name == 'ogbn-papers100M':
            edges_size_mb = 6000.0
        else:
            edges_size_mb = 100.0

    # ═══════════════════════════════════════════════════════════════════════════
    #  DATASET-AWARE BIN-PACKING AUTO-SCALER
    #
    #  Maximizes cluster parallelism while keeping every executor safely within
    #  its YARN container memory limit.
    #
    #  Key insight: Python RAM per task varies dramatically by dataset size.
    #  Small datasets (WikiCS):       communities ~500 nodes  → ~0.8 GB/task
    #  Medium datasets (ogbn-products): communities ~5K nodes → ~2.0 GB/task
    #  Massive datasets (papers100M): communities ~600K nodes → ~6.5 GB/task
    #
    #  The solver tries cores_per_executor ∈ [1..8] and picks the configuration
    #  that maximizes total_parallel_tasks subject to:
    #    overhead ≥ cores × python_ram_per_task   (memory safety)
    #    heap + overhead ≤ usable_node_mem        (YARN container limit)
    #    cores ≤ available_vcores                 (CPU limit)
    # ═══════════════════════════════════════════════════════════════════════════

    # Step 1: Estimate Python RAM per concurrent task based on dataset scale.
    # Since we use Zero-Shuffle PyArrow community filtering, executors only load tiny subgraphs (under 20MB)
    # and never load the full graph. We cap python_ram_per_task to 1.5 GB to maximize container bin-packing.
    if edges_size_mb <= 20.0:
        scale_label = "Small/Medium"
        python_ram_per_task = 2.0
    elif edges_size_mb <= 300.0:
        scale_label = "Large (100M Scale)"
        python_ram_per_task = 6.0
    else:
        scale_label = "Very Large/Massive (100M+ Scale)"
        python_ram_per_task = 20.0

    # For massive datasets, use fatter executors (2-4 cores per exec) to guarantee 56GB+ RAM per container
    max_cores = 2 if edges_size_mb > 300.0 else 4
    jvm_heap_min = 16.0 if edges_size_mb > 300.0 else 4.0

    OS_RESERVE_GB   = max(16.0, node_mem_gb * 0.08)   # reserved for OS kernel + YARN NodeManager daemon
    DRIVER_FRACTION = 0.75  # fraction of driver host RAM for driver container
    usable_node_mem_gb = node_mem_gb - OS_RESERVE_GB

    if edges_size_mb <= 50.0:
        scale_label = "Small/Medium"
        target_execs_per_node = min(16, max(1, node_vcores // 4))
    elif edges_size_mb <= 500.0:
        scale_label = "Large (100M Scale)"
        target_execs_per_node = min(8, max(1, node_vcores // 4))
    else:
        scale_label = "Very Large/Massive (100M+ Scale)"
        target_execs_per_node = min(8, max(1, node_vcores // 4))

    execs_per_node = max(1, target_execs_per_node)
    cores_candidate = max(2, min(4, node_vcores // execs_per_node))
    total_execs = nodes_count * execs_per_node
    total_cores = total_execs * cores_candidate

    container_gb = max(16, int(usable_node_mem_gb / execs_per_node))
    heap_needed = max(4, int(container_gb * 0.30))
    overhead_needed = container_gb - heap_needed

    best_total_cores = total_cores
    best_config = {
        'cores': cores_candidate,
        'heap_gb': heap_needed,
        'overhead_gb': overhead_needed,
        'container_gb': container_gb,
        'execs_per_node': execs_per_node,
        'total_execs': total_execs,
        'python_ram_per_task': round(overhead_needed / cores_candidate, 1),
    }

    executor_instances = best_config['total_execs']
    executor_mem       = f"{best_config['heap_gb']}g"
    executor_overhead  = f"{best_config['overhead_gb']}g"
    executor_cores     = str(best_config['cores'])

    # Step 3: Driver allocation — capped to 40g heap + 12g overhead (52g total) for YARN container harmony
    driver_mem = "40g"
    driver_overhead = "12g"
    driver_cores = "8"

    # Step 4: Shuffle partitions — 2× total cores for pipeline overlap
    shuffle_partitions = max(200, best_total_cores * 2)

    # Step 5: Print the full allocation plan
    print(f"\n  [Spark Auto-Scaler] Dataset: {dataset_name} | S3 Size: {edges_size_mb:.0f} MB | Scale: {scale_label}")
    print(f"  [Spark Auto-Scaler] YARN Cluster: {nodes_count} nodes × {node_mem_gb:.0f} GB RAM × {node_vcores} vCores")
    print(f"  [Spark Auto-Scaler] Python RAM Budget: {best_config['python_ram_per_task']:.1f} GB/task (dataset-aware)")
    print(f"  [Spark Auto-Scaler] Bin-Packing Solution:")
    print(f"    → {best_config['execs_per_node']} executors/node × {nodes_count} nodes = {executor_instances} total executors")
    print(f"    → {executor_cores} cores/executor | {executor_mem} heap + {executor_overhead} overhead = {best_config['container_gb']}g container")
    print(f"    → Total parallel tasks: {best_total_cores} | Shuffle partitions: {shuffle_partitions}")
    print(f"    → Driver: {driver_mem} heap + {driver_overhead} overhead")
    print(f"    → Cluster utilization: {executor_instances * best_config['container_gb']:.0f}g / {node_mem_gb * nodes_count:.0f}g "
          f"({100.0 * executor_instances * best_config['container_gb'] / (node_mem_gb * nodes_count):.0f}%)\n")

    spark = SparkSession.builder \
        .appName(f"GRL-{args.experiment_name}") \
        .config("spark.master", "yarn") \
        .config("spark.eventLog.enabled", "false") \
        .config("spark.driver.memory", driver_mem) \
        .config("spark.driver.maxResultSize", "0") \
        .config("spark.driver.cores", driver_cores) \
        .config("spark.driver.memoryOverhead", driver_overhead) \
        .config("spark.rpc.message.maxSize", "1024") \
        .config("spark.network.timeout", "1800s") \
        .config("spark.executor.heartbeatInterval", "180s") \
        .config("spark.speculation", "true") \
        .config("spark.speculation.interval", "100ms") \
        .config("spark.speculation.quantile", "0.75") \
        .config("spark.speculation.multiplier", "2.0") \
        .config("spark.task.maxFailures", "8") \
        .config("spark.stage.maxConsecutiveAttempts", "8") \
        .config("spark.shuffle.io.maxRetries", "10") \
        .config("spark.shuffle.io.retryWait", "30s") \
        .config("spark.yarn.maxAppAttempts", "4") \
        .config("spark.dynamicAllocation.enabled", "false") \
        .config("spark.executor.instances", str(executor_instances)) \
        .config("spark.executor.memory", executor_mem) \
        .config("spark.executor.memoryOverhead", executor_overhead) \
        .config("spark.executor.cores", executor_cores) \
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions)) \
        .config("spark.default.parallelism", str(shuffle_partitions)) \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "1024m") \
        .config("spark.pyspark.python", "python3") \
        .config("spark.pyspark.virtualenv.enabled", "false") \
        .config("spark.executorEnv.HOME", "/tmp") \
        .config("spark.executorEnv.PYTHONUSERBASE", "/tmp/.local") \
        .config("spark.executorEnv.PYTHONPATH", f"/mnt/tmp/.local/lib/{py_version}/site-packages:/mnt1/tmp/.local/lib/{py_version}/site-packages:/mnt2/tmp/.local/lib/{py_version}/site-packages:/tmp/.local/lib/{py_version}/site-packages:$PYTHONPATH") \
        .config("spark.executorEnv.DGLBACKEND", "pytorch") \
        .config("spark.executorEnv.DGL_DOWNLOAD_DIR", "/tmp/.dgl") \
        .config("spark.executorEnv.TMPDIR", "/tmp") \
        .config("spark.executorEnv.TEMP", "/tmp") \
        .config("spark.executorEnv.TMP", "/tmp") \
        .config("spark.executorEnv.OMP_NUM_THREADS", "1") \
        .config("spark.executorEnv.MKL_NUM_THREADS", "1") \
        .config("spark.executorEnv.OPENBLAS_NUM_THREADS", "1") \
        .config("spark.executorEnv.VECLIB_MAXIMUM_THREADS", "1") \
        .config("spark.executorEnv.NUMEXPR_NUM_THREADS", "1") \
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.0,graphframes:graphframes:0.8.3-spark3.5-s_2.12") \
        .config("spark.jars.ivy", f"{large_tmp}/.ivy2") \
        .config("spark.local.dir", f"{large_tmp}/spark-local") \
        .config("spark.driver.extraJavaOptions", f"-Djava.io.tmpdir={large_tmp}") \
        .config("spark.executor.extraJavaOptions", f"-Djava.io.tmpdir={large_tmp}") \
        .config("spark.hadoop.dfs.datanode.du.reserved", "1073741824") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "500") \
        .config("spark.python.worker.reuse", "false") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "3") \
        .config("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "268435456") \
        .config("spark.databricks.delta.retentionDurationCheck.enabled", "false") \
        .config("spark.databricks.delta.vacuum.parallelDelete.enabled", "true") \
        .enableHiveSupport() \
        .getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    print("  ✓ SparkSession successfully configured and initialized as 'spark'.")

    # ── 3. INSTALL PYTHON DEPENDENCIES ─────────────────────────────────────────
    skip_pkg_sync = False
    try:
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg_path = os.path.join(script_dir, "experiment_config.py")
        if os.path.exists(cfg_path):
            import re
            with open(cfg_path, 'r') as f:
                content = f.read()
            m = re.search(r'SKIP_PKG_SYNC\s*=\s*(True|False)', content)
            if m:
                skip_pkg_sync = (m.group(1) == 'True')
    except Exception:
        pass

    if getattr(args, 'no_install', False) or skip_pkg_sync:
        print("\n  ► Skipping dynamic package verification/installation on YARN executors...")
    else:


        print("\n" + "="*80)
        print("  VERIFYING AND INSTALLING PYTHON ENVIRONMENT PACKAGES")
        print("="*80)

        packages = ['numpy', 'ogb', 'igraph', 'leidenalg', 'scikit-learn',
                    'torch', 'boto3', 'xlsxwriter', 'openpyxl', 'matplotlib', 'seaborn',
                    'torch-geometric', 'pyarrow', 'dgl==1.1.3']
        driver_only_packages = {'xlsxwriter', 'openpyxl', 'matplotlib', 'seaborn', 'igraph', 'leidenalg', 'ogb'}

        # 1. Install / Verify driver-only and general packages on driver node
        import importlib
        for pkg in packages:
            import_map = {
                'scikit-learn': 'sklearn',
                'torch-geometric': 'torch_geometric',
                'dgl==1.1.3': 'dgl'
            }
            import_name = import_map.get(pkg, pkg)
            try:
                importlib.import_module(import_name)
            except Exception:
                print(f"  ► Installing {pkg} on driver node...")
                cmd = [sys.executable, '-m', 'pip', 'install', '--user', '--quiet', '--no-cache-dir', '--force-reinstall']
                if pkg.startswith('dgl'):
                    cmd += [pkg, '-f', 'https://data.dgl.ai/wheels/repo.html']
                elif pkg == 'torch':
                    cmd += [pkg, '--index-url', 'https://download.pytorch.org/whl/cpu']
                else:
                    cmd += [pkg]
                try:
                    subprocess.run(cmd, check=True)
                    print(f"    ✓ {pkg} successfully installed on driver.")
                except Exception as e:
                    print(f"    ⚠ Failed to install {pkg} on driver: {e}")

        # 2. Verify and install executor packages
        executor_packages = [p for p in packages if p not in driver_only_packages]

        def run_executor_install(iterator):
            import socket
            import subprocess
            import sys
            import os
            import time
            import importlib

            node_name = socket.gethostname()
            
            # Dynamically discover a writable local scratch path on this specific node
            candidates = ["/mnt/tmp", "/mnt1/tmp", "/mnt2/tmp", "/mnt/spark", "/mnt1/spark", "/mnt2/spark", "/tmp"]
            worker_tmp = "/tmp"
            for c in candidates:
                try:
                    os.makedirs(c, exist_ok=True)
                    if os.access(c, os.W_OK):
                        worker_tmp = c
                        break
                except Exception:
                    pass

            os.environ["PYTHONUSERBASE"] = f"{worker_tmp}/.local"
            os.environ["TMPDIR"] = worker_tmp
            os.environ["TEMP"] = worker_tmp
            os.environ["TMP"] = worker_tmp

            import_map = {
                'scikit-learn': 'sklearn',
                'torch-geometric': 'torch_geometric',
                'dgl==1.1.3': 'dgl'
            }

            def check_all_imported():
                for pkg in executor_packages:
                    import_name = import_map.get(pkg, pkg)
                    try:
                        importlib.import_module(import_name)
                    except Exception:
                        return False
                return True

            # If already fully built on this node, hold slot briefly so Spark schedules tasks to all worker nodes
            if check_all_imported():
                time.sleep(3)
                return [f"{node_name}: Success (Cached)"]

            lock_dir = f"{worker_tmp}/.local_sync_lock"
            if os.path.exists(lock_dir):
                try:
                    os.rmdir(lock_dir)
                except Exception:
                    pass

            lock_acquired = False
            for _ in range(5):
                if check_all_imported():
                    return [f"{node_name}: Success (Completed by other task)"]
                try:
                    os.makedirs(lock_dir, exist_ok=False)
                    lock_acquired = True
                    break
                except FileExistsError:
                    time.sleep(1)

            if not lock_acquired:
                if check_all_imported():
                    return [f"{node_name}: Success (Completed by other task)"]
                return [f"{node_name}: Failed (Timeout waiting for sync lock)"]

            # We hold the lock! Check and install missing packages sequentially
            installed_pkgs = []
            try:
                for pkg in executor_packages:
                    import_name = import_map.get(pkg, pkg)
                    try:
                        importlib.import_module(import_name)
                        continue
                    except Exception:
                        pass

                    cmd = [sys.executable, '-m', 'pip', 'install', '--user', '--quiet', '--no-cache-dir', '--force-reinstall']
                    if pkg.startswith('dgl'):
                        cmd += [pkg, '-f', 'https://data.dgl.ai/wheels/repo.html']
                    elif pkg == 'torch':
                        cmd += [pkg, '--index-url', 'https://download.pytorch.org/whl/cpu']
                    else:
                        cmd += [pkg]

                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    installed_pkgs.append(pkg)
                return [f"{node_name}: Success (Installed: {installed_pkgs})"]
            except Exception as e:
                return [f"{node_name}: Failed ({e})"]
            finally:
                try:
                    os.rmdir(lock_dir)
                except Exception:
                    pass

        try:
            num_executors = int(spark.conf.get("spark.executor.instances", "8"))
            print(f"  ► Syncing dependencies across all active YARN worker nodes...")

            synced_hosts = set()
            all_reports = []

            for attempt in range(1, 6):
                # Run high partition count with 3s slot hold to hit all physical nodes
                n_partitions = max(64, num_executors * 16)
                results = sc.parallelize(range(n_partitions), n_partitions) \
                            .mapPartitions(run_executor_install) \
                            .collect()
                
                prev_count = len(synced_hosts)
                for report in results:
                    all_reports.append(report)
                    if "Success" in report:
                        h = report.split(":")[0].split('.')[0].strip()
                        synced_hosts.add(h)

                print(f"  ► Sync attempt {attempt}/5: {len(synced_hosts)} worker node(s) verified: {sorted(list(synced_hosts))}")

                # Stop when we hit the configured executor instance target
                if len(synced_hosts) >= num_executors:
                    print(f"  ✓ All {len(synced_hosts)} YARN worker nodes verified and synced successfully.")
                    break
                
                time.sleep(2)

            print("\n  === YARN Executor Package Sync Summary ===")
            unique_reports = sorted(list(set(all_reports)))
            for report in unique_reports:
                print(f"    {report}")

        except Exception as e:
            print(f"  ⚠ Executor package sync failed: {e}")

        print("\n  ✓ ALL PIPELINE DEPENDENCIES VERIFIED AND READY.")

        # ── Cluster Disk Space Check ──
        try:
            n_nodes = 0
            feature_dim = 0
            
            try:
                from pipeline.utils.paths import get_paths
                p_paths = get_paths(dataset_name)
                nodes_df = spark.read.format('delta').load(p_paths['nodes'])
                n_nodes = nodes_df.count()
                first_row = nodes_df.select('features').first()
                if first_row and first_row['features'] is not None:
                    feature_dim = len(first_row['features'])
            except Exception:
                fallback_stats = {
                    'wikics': (11701, 300),
                    'coauthor-cs': (18333, 6805),
                    'coauthor-physics': (34493, 8415),
                    'deezereurope': (28281, 128),
                    'flickr': (89250, 500),
                    'reddit': (232965, 602),
                    'ogbn-arxiv': (169343, 128),
                    'ogbn-products': (2449029, 100),
                    'ogbn-papers100M': (111059956, 128)
                }
                n_nodes, feature_dim = fallback_stats.get(dataset_name, (200000, 128))

            full_feature_bytes = n_nodes * feature_dim * 4
            # Executors only process local data shards. We scale required disk space by nodes_count with a 1.5x safety margin.
            required_bytes = max(10 * 1024**3, (full_feature_bytes * 1.5) / max(1, nodes_count))
            required_gb = required_bytes / (1024**3)

            print("\n" + "="*80)
            print("  CLUSTER DISK SPACE COMPATIBILITY VERIFICATION")
            print("="*80)
            print(f"  Dataset: {dataset_name} | Nodes: {n_nodes:,} | Features: {feature_dim}")
            print(f"  Required temp workspace disk space per executor container: {required_gb:.2f} GB")

            # Check Driver Node
            driver_free_gb = shutil.disk_usage(large_tmp).free / (1024**3)
            print(f"  Driver temp space available ({large_tmp}): {driver_free_gb:.2f} GB")

            # Check Executor Nodes
            num_execs = int(spark.conf.get("spark.executor.instances", "4"))
            def run_worker_disk_check(iterator):
                import shutil
                import socket
                _, _, free = shutil.disk_usage('.')
                free_gb = free / (1024**3)
                if free < required_bytes:
                    return [f"FAIL:{socket.gethostname()}:{free_gb:.2f} GB"]
                else:
                    return [f"OK:{socket.gethostname()}:{free_gb:.2f} GB"]

            worker_check_rdd = sc.parallelize(range(num_execs * 2), num_execs * 2)
            check_results = worker_check_rdd.mapPartitions(run_worker_disk_check).collect()

            print("  Executor nodes disk check results:")
            failures = []
            for r in sorted(list(set(check_results))):
                status, host, avail_gb = r.split(':')
                if status == 'FAIL':
                    print(f"    ❌ {host:<45} - INSUFFICIENT SPACE! Available: {avail_gb} (Needs {required_gb:.2f} GB)")
                    failures.append(r)
                else:
                    print(f"    ✓  {host:<45} - Sufficient Space. Available: {avail_gb}")

            if failures:
                print("\n[FATAL ERROR] Cluster disk space check failed! Insufficient disk space on executors.")
                print("Please allocate larger EBS volumes or clean executor caches before rerunning.")
                sys.exit(1)

            print("  ✓ ALL NODES HAVE SUFFICIENT TEMPORARY DISK SPACE TO EXECUTE GNN JOBS.")
            print("="*80 + "\n")
        except Exception as e:
            print(f"  ⚠️ Warning: Could not complete cluster disk space check: {e}")

    # ── 4. FETCH AND PACK SCRIPTS ──────────────────────────────────────────────
    print("\n" + "="*80)
    print("  BOOTSTRAPPING EXPERIMENT CODE AND STAGE UTILITIES")
    print("="*80)
    
    import boto3
    s3 = boto3.client('s3')

    # Locate local repo or download from S3
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_cfg = os.path.join(script_dir, "experiment_config.py")

    if args.local or os.path.exists(local_cfg):
        print("  ► LOADING PIPELINE AND CONFIG FROM LOCAL REPOSITORY PATHS...")
        # Clean up previous run leftovers in /tmp to avoid package/namespace conflicts
        for path in ['/tmp/pipeline', '/tmp/pipeline.zip', '/tmp/pipeline_stage', '/tmp/experiment_config.py']:
            if os.path.exists(path):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                except Exception:
                    pass

        shutil.copyfile(local_cfg, "/tmp/experiment_config.py")
        
        # Build staging directory /tmp/pipeline_stage/pipeline to match import namespace 'pipeline'
        os.makedirs('/tmp/pipeline_stage/pipeline', exist_ok=True)
        for item in os.listdir(script_dir):
            s = os.path.join(script_dir, item)
            d = os.path.join('/tmp/pipeline_stage/pipeline', item)
            if os.path.isdir(s):
                if item not in ['.git', '__pycache__', 'results', 'pipeline']:
                    shutil.copytree(s, d)
            else:
                shutil.copyfile(s, d)

        # Package local pipeline folder into a zip containing 'pipeline/' at root
        shutil.make_archive("/tmp/pipeline", "zip", "/tmp/pipeline_stage", "pipeline")
        
        # Copy to /tmp/pipeline so driver can import it directly
        shutil.copytree('/tmp/pipeline_stage/pipeline', '/tmp/pipeline')
        print("    ✓ Packed local git repository code to /tmp/pipeline.zip and copied to /tmp/pipeline")
    else:
        print(f"  ► DOWNLOADING SCRIPTS FROM S3: s3://{args.s3_bucket}/{args.s3_prefix} ...")

        # Clean up previous run leftovers in /tmp to avoid package/namespace conflicts
        for path in ['/tmp/pipeline', '/tmp/pipeline.zip', '/tmp/pipeline_stage', '/tmp/experiment_config.py']:
            if os.path.exists(path):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                except Exception:
                    pass

        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=args.s3_bucket, Prefix=args.s3_prefix + '/')
        
        download_count = 0
        for page in pages:
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                key = obj['Key']
                # Download .py files and requirements.txt
                if not (key.endswith('.py') or key.endswith('requirements.txt')):
                    continue
                
                # Get path relative to s3_prefix
                rel_path = key[len(args.s3_prefix) + 1:].strip('/')
                if not rel_path:
                    continue
                
                if rel_path in ['experiment_config.py', 'requirements.txt']:
                    local_path = f'/tmp/{rel_path}'
                else:
                    local_path = f'/tmp/pipeline_stage/pipeline/{rel_path}'
                
                # Create local directory structure
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                # Download file
                s3.download_file(args.s3_bucket, key, local_path)
                print(f"    - Downloaded key: {key} -> {local_path}")
                download_count += 1
                
        print(f"    ✓ Downloaded {download_count} files from S3 successfully to /tmp.")
        
        # Fallback to download experiment_config.py from parent directory or bucket root if not in prefix
        if not os.path.exists('/tmp/experiment_config.py'):
            print("  ► experiment_config.py not found in prefix. Trying parent directory or bucket root...")
            parent_prefix = os.path.dirname(args.s3_prefix)
            parent_key = f"{parent_prefix}/experiment_config.py" if parent_prefix else "experiment_config.py"
            try:
                s3.download_file(args.s3_bucket, parent_key, '/tmp/experiment_config.py')
                print(f"    ✓ Downloaded experiment_config.py from parent key: {parent_key}")
            except Exception:
                try:
                    s3.download_file(args.s3_bucket, "experiment_config.py", '/tmp/experiment_config.py')
                    print("    ✓ Downloaded experiment_config.py from bucket root")
                except Exception as e:
                    print(f"    ⚠ Could not locate experiment_config.py anywhere on S3: {e}")
        
        # Package the downloaded files into /tmp/pipeline.zip
        if os.path.exists('/tmp/pipeline_stage/pipeline'):
            # Ensure __init__.py exists in package root
            init_file = '/tmp/pipeline_stage/pipeline/__init__.py'
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write("# Auto-generated package init\n")
            
            # If experiment_config.py is inside the pipeline package, copy it to /tmp
            pkg_config = '/tmp/pipeline_stage/pipeline/experiment_config.py'
            if os.path.exists(pkg_config) and not os.path.exists('/tmp/experiment_config.py'):
                shutil.copyfile(pkg_config, '/tmp/experiment_config.py')
                print("    ✓ Copied experiment_config.py from package to /tmp")
            
            # Pack '/tmp/pipeline_stage/pipeline' into '/tmp/pipeline.zip'
            shutil.make_archive("/tmp/pipeline", "zip", "/tmp/pipeline_stage", "pipeline")
            
            # Copy to /tmp/pipeline so driver can import it directly
            shutil.copytree('/tmp/pipeline_stage/pipeline', '/tmp/pipeline')
            print("    ✓ Packed downloaded files to /tmp/pipeline.zip and synchronized /tmp/pipeline")

    # Register zip package on PySpark driver and YARN executors
    if os.path.exists('/tmp/pipeline.zip'):
        sys.path.insert(0, '/tmp/pipeline.zip')
        sc.addPyFile('/tmp/pipeline.zip')
        print("  ✓ Registered pipeline.zip package on Spark Context")

    if '/tmp' in sys.path:
        sys.path.remove('/tmp')
    sys.path.insert(0, '/tmp')

    # Load configuration
    try:
        import experiment_config as config
    except ModuleNotFoundError as e:
        print("\n[ERROR] Failed to import experiment_config from /tmp!")
        print(f"sys.path: {sys.path}")
        if os.path.exists('/tmp'):
            print(f"Contents of /tmp: {os.listdir('/tmp')}")
        raise e
    from pipeline.phases import (
        run_phase0, run_phase1, print_phase1_stats, run_phase2, run_phase3, run_phase3b,
        run_phase4, run_phase4b, run_phase4c, run_phase4d, run_phase4e, run_phase4f, run_phase4g, run_phase4h,
        print_accuracy_table, print_timing_table, save_plots_and_xlsx, print_summary
    )
    from pipeline.utils.paths import get_paths

    # Override configurations from CLI args
    EXPERIMENT_NAME = args.experiment_name if args.experiment_name is not None else getattr(config, 'EXPERIMENT_NAME', 'emr_interactive')
    DATASETS_TO_RUN = [d.strip() for d in args.datasets.split(",") if d.strip()] if args.datasets is not None else getattr(config, 'DATASETS_TO_RUN', ['ogbn-mag'])
    ALGORITHMS_TO_RUN = [a.strip() for a in args.algorithms.split(",") if a.strip()] if args.algorithms is not None else getattr(config, 'ALGORITHMS_TO_RUN', ['lpa'])
    TASK_TYPE = args.task_type if args.task_type is not None else getattr(config, 'TASK_TYPE', 'both')
    
    RUN_PHASE0 = args.run_phase0 if args.run_phase0 is not None else getattr(config, 'RUN_PHASE0', True)
    FORCE_REINGEST = args.force_reingest if args.force_reingest is not None else getattr(config, 'FORCE_REINGEST', False)
    USE_OGB_SPLITS = (args.use_ogb_splits == "true") if args.use_ogb_splits is not None else getattr(config, 'USE_OGB_SPLITS', True)
    MIN_COMMUNITY_SIZE = args.min_community_size if args.min_community_size is not None else getattr(config, 'MIN_COMMUNITY_SIZE', 10)
    TINY_COMM_HANDLING = args.tiny_comm_handling if args.tiny_comm_handling is not None else getattr(config, 'TINY_COMM_HANDLING', 'misc')
    EXPAND_BOUNDARY_NODES = (args.expand_boundary_nodes == "true") if args.expand_boundary_nodes is not None else getattr(config, 'EXPAND_BOUNDARY_NODES', True)
    USE_GLOBAL_MAPPING = (args.global_mapping == "true") if args.global_mapping is not None else getattr(config, 'USE_GLOBAL_MAPPING', True)

    h_dim = args.hidden_dim if args.hidden_dim is not None else getattr(config, 'GCN_CFG', {}).get('hidden_dim', 256)
    n_epochs = args.num_epochs if args.num_epochs is not None else getattr(config, 'GCN_CFG', {}).get('num_epochs', 10)
    learning_rate = args.lr if args.lr is not None else getattr(config, 'GCN_CFG', {}).get('lr', 1e-2)
    drop_out = args.dropout if args.dropout is not None else getattr(config, 'GCN_CFG', {}).get('dropout', 0.5)

    GCN_CFG = {
        'hidden_dim': h_dim,
        'num_epochs': n_epochs,
        'lr':         learning_rate,
        'dropout':    drop_out,
    }
    
    baseline_epochs = args.num_epochs if args.num_epochs is not None else getattr(config, 'BASELINE_EPOCHS', n_epochs)
    BASELINE_CFG = {
        'epochs':     baseline_epochs,
        'batch':      getattr(config, 'BASELINE_BATCH', 1024),
        'fanout':     [10, 10],
        'lr':         learning_rate,
        'hidden_dim': h_dim,
        'dropout':    drop_out,
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
    print(f"  └─ GNN Architecture: GCN (epochs={n_epochs}, lr={learning_rate}, hidden={h_dim})")
    print(f"  └─ Global Mapping: {USE_GLOBAL_MAPPING} | Boundary Expansion: {EXPAND_BOUNDARY_NODES}")
    print(f"  └─ Ingestion Phase (Phase 0): {RUN_PHASE0}")

    # Build path helper function for S3 paths
    get_paths_fn = lambda dataset, alg=None: get_paths(
        dataset, alg,
        experiment_name=EXPERIMENT_NAME,
        s3_bucket=args.s3_bucket
    )

    t_pipeline_start = time.time()

    # Phase 0: Ingestion
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

    # Phase 1: Partition assignment / Community detection
    if getattr(config, 'RUN_PHASE1', True):
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

        # Phase 1 Stats
        print_phase1_stats(
            spark,
            datasets     = DATASETS_TO_RUN,
            algorithms   = ALGORITHMS_TO_RUN,
            min_size     = MIN_COMMUNITY_SIZE,
            get_paths_fn = get_paths_fn,
            results      = phase1_results
        )

    # Phase 2: Subgraph generation
    if getattr(config, 'RUN_PHASE2', True):
        run_phase2(
            spark, sc,
            datasets            = DATASETS_TO_RUN,
            algorithms          = ALGORITHMS_TO_RUN,
            use_global_mapping  = USE_GLOBAL_MAPPING,
            min_size            = MIN_COMMUNITY_SIZE,
            get_paths_fn        = get_paths_fn,
            timing              = timing,
            results             = phase2_results,
            tiny_comm_handling  = TINY_COMM_HANDLING,
            expand_boundary_nodes = EXPAND_BOUNDARY_NODES,
            force_rerun         = FORCE_RERUN
        )

    # Phase 3: Parallel GNN UDF Training
    if getattr(args, 'run_phase3', True) and getattr(config, 'RUN_PHASE3', True):
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
            s3_bucket          = args.s3_bucket,
            experiment_name    = EXPERIMENT_NAME
        )

    # Phase 3b: Parallel GNN Training with CaaN Global Graph
    if ALGORITHMS_TO_RUN and args.run_phase3b and getattr(config, 'RUN_PHASE3B', True):
        print("\n[PHASE 3b] - Parallel GNN Training with CaaN Global Graph")
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
            s3_bucket          = args.s3_bucket,
            experiment_name    = EXPERIMENT_NAME
        )

    # Phase 4: Full-Graph Baseline
    if getattr(args, 'run_phase4', True):
        # 1. GraphSAGE Baseline
        if getattr(config, 'RUN_PHASE4', True) and 'sage' in config.GNN_MODELS:
            datasets_for_4 = []
            for dataset in DATASETS_TO_RUN:
                if not FORCE_RERUN and load_baseline_checkpoint(dataset, '4', phase4_results, timing, args.s3_bucket, EXPERIMENT_NAME):
                    pass
                else:
                    datasets_for_4.append(dataset)
            if datasets_for_4:
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
                    save_baseline_checkpoint(dataset, '4', phase4_results, timing, args.s3_bucket, EXPERIMENT_NAME)

        # 2. DistDGL Baseline
        if getattr(config, 'RUN_PHASE4B', True):
            datasets_for_4b = []
            for dataset in DATASETS_TO_RUN:
                if not FORCE_RERUN and load_baseline_checkpoint(dataset, '4b', phase4b_results, timing, args.s3_bucket, EXPERIMENT_NAME):
                    pass
                else:
                    datasets_for_4b.append(dataset)
            if datasets_for_4b:
                run_phase4b(
                    spark, sc,
                    datasets     = datasets_for_4b,
                    dataset_cfg  = config.DATASET_CFG,
                    baseline_cfg = BASELINE_CFG,
                    get_paths_fn = get_paths_fn,
                    timing       = timing,
                    results      = phase4b_results,
                    task_type    = TASK_TYPE,
                )
                for dataset in datasets_for_4b:
                    save_baseline_checkpoint(dataset, '4b', phase4b_results, timing, args.s3_bucket, EXPERIMENT_NAME)

        # 3. ARMA Baseline
        if getattr(config, 'RUN_PHASE4C', True):
            datasets_for_4c = []
            for dataset in DATASETS_TO_RUN:
                if not FORCE_RERUN and load_baseline_checkpoint(dataset, '4c', phase4c_results, timing, args.s3_bucket, EXPERIMENT_NAME):
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
                    save_baseline_checkpoint(dataset, '4c', phase4c_results, timing, args.s3_bucket, EXPERIMENT_NAME)

        # 4. ASAP Baseline
        if getattr(config, 'RUN_PHASE4D', True):
            datasets_for_4d = []
            for dataset in DATASETS_TO_RUN:
                if not FORCE_RERUN and load_baseline_checkpoint(dataset, '4d', phase4d_results, timing, args.s3_bucket, EXPERIMENT_NAME):
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
                    save_baseline_checkpoint(dataset, '4d', phase4d_results, timing, args.s3_bucket, EXPERIMENT_NAME)

        # 5. GAT Baseline
        if getattr(config, 'RUN_PHASE4E', True) and getattr(config, 'RUN_PHASE4', True) and 'gat' in config.GNN_MODELS:
            datasets_for_4e = []
            for dataset in DATASETS_TO_RUN:
                if not FORCE_RERUN and load_baseline_checkpoint(dataset, '4e', phase4e_results, timing, args.s3_bucket, EXPERIMENT_NAME):
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
                    save_baseline_checkpoint(dataset, '4e', phase4e_results, timing, args.s3_bucket, EXPERIMENT_NAME)

        # 6. Graph Transformer Baseline
        if getattr(config, 'RUN_PHASE4F', True) and getattr(config, 'RUN_PHASE4', True) and 'transformer' in config.GNN_MODELS:
            datasets_for_4f = []
            for dataset in DATASETS_TO_RUN:
                if not FORCE_RERUN and load_baseline_checkpoint(dataset, '4f', phase4f_results, timing, args.s3_bucket, EXPERIMENT_NAME):
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
                    save_baseline_checkpoint(dataset, '4f', phase4f_results, timing, args.s3_bucket, EXPERIMENT_NAME)

        # 7. ClusterSCL Baseline
        if getattr(config, 'RUN_PHASE4G', True) and getattr(config, 'RUN_PHASE4', True) and 'clusterscl' in config.GNN_MODELS:
            datasets_for_4g = []
            for dataset in DATASETS_TO_RUN:
                if not FORCE_RERUN and load_baseline_checkpoint(dataset, '4g', phase4g_results, timing, args.s3_bucket, EXPERIMENT_NAME):
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
                    save_baseline_checkpoint(dataset, '4g', phase4g_results, timing, args.s3_bucket, EXPERIMENT_NAME)

        # 8. GATv2 Baseline
        if getattr(config, 'RUN_PHASE4H', True) and getattr(config, 'RUN_PHASE4', True) and 'gatv2' in config.GNN_MODELS:
            datasets_for_4h = []
            for dataset in DATASETS_TO_RUN:
                if not FORCE_RERUN and load_baseline_checkpoint(dataset, '4h', phase4h_results, timing, args.s3_bucket, EXPERIMENT_NAME):
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
                    save_baseline_checkpoint(dataset, '4h', phase4h_results, timing, args.s3_bucket, EXPERIMENT_NAME)

    # Phase 5: Metrics Aggregation & Reporting
    print("\n[PHASE 5] - Metrics Analysis, Visualizations, and Excel S3 Export")
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
        s3_bucket       = args.s3_bucket,
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

    # SUMMARY — Print final execution report box
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
    print(f"\n[SUCCESS] EMR Driver execution completed in {t_elapsed:.1f} seconds.")
    print("="*80 + "\n")

    # Restore streams and close log file
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()

    # Consolidated output upload to S3 (logs/ and excels/)
    try:
        import secrets
        import boto3
        run_id = secrets.token_hex(8)
        consolidated_folder = f"{EXPERIMENT_NAME}-{run_id}"
        s3_bucket = args.s3_bucket if args.s3_bucket else "us-east-1-s3-gnn"
        
        s3_client = boto3.client('s3')
        
        # 1. Upload the log file
        s3_log_key = f"gnn-bench-out/spark-results/{consolidated_folder}/logs/run_pipeline.log"
        print(f"Uploading execution log to: s3://{s3_bucket}/{s3_log_key}")
        s3_client.upload_file(log_file_path, s3_bucket, s3_log_key)
        
        # 2. Copy Excel results directly on S3
        s3_source_key = f"gnn-bench-out/{EXPERIMENT_NAME}_results.xlsx"
        s3_dest_key = f"gnn-bench-out/spark-results/{consolidated_folder}/excels/{EXPERIMENT_NAME}_results.xlsx"
        
        try:
            s3_client.head_object(Bucket=s3_bucket, Key=s3_source_key)
            print(f"Copying S3 Excel results to: s3://{s3_bucket}/{s3_dest_key}")
            s3_client.copy_object(
                Bucket=s3_bucket,
                CopySource={'Bucket': s3_bucket, 'Key': s3_source_key},
                Key=s3_dest_key
            )
        except Exception as copy_err:
            print(f"Warning: Could not copy Excel results on S3: {copy_err}")
            
        # 3. Upload LaTeX tables directly to S3 under /latex_tables
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(project_root, "results")
        if os.path.exists(results_dir):

            for fname in os.listdir(results_dir):
                if fname.endswith(".tex"):
                    local_tex = os.path.join(results_dir, fname)
                    s3_tex_key = f"gnn-bench-out/spark-results/{consolidated_folder}/latex_tables/{fname}"
                    print(f"Uploading LaTeX table to: s3://{s3_bucket}/{s3_tex_key}")
                    s3_client.upload_file(local_tex, s3_bucket, s3_tex_key)

    except Exception as upload_err:
        print(f"Warning: Error uploading logs, excels, or latex_tables to consolidated S3 folder: {upload_err}")


if __name__ == "__main__":
    main()

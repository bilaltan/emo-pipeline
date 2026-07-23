#!/usr/bin/env python3
"""
AWS EMR Background Step Runner
Runs the entire GRL pipeline in the background as an EMR Step.
Loads code and config from S3, registers dependencies, and runs all phases.
"""
import os
import sys
import time
import subprocess
import shutil

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
py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
os.environ['TMP'] = large_tmp

# Initialize Spark Session (configured for AWS YARN)
from pyspark.sql import SparkSession

print("Starting Spark Application Step...")
spark = SparkSession.builder \
    .appName("GRL-Pipeline-Background-Step") \
    .config("spark.master", "yarn") \
    .config("spark.driver.memory", "30g") \
    .config("spark.driver.maxResultSize", "24g") \
    .config("spark.driver.cores", "4") \
    .config("spark.driver.memoryOverhead", "8g") \
    .config("spark.dynamicAllocation.enabled", "false") \
    .config("spark.executor.instances", "4") \
    .config("spark.executor.memory", "28g") \
    .config("spark.executor.memoryOverhead", "12g") \
    .config("spark.executor.cores", "2") \
    .config("spark.sql.shuffle.partitions", "336") \
    .config("spark.default.parallelism", "336") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryoserializer.buffer.max", "1024m") \
    .config("spark.pyspark.python", "python3") \
    .config("spark.pyspark.virtualenv.enabled", "false") \
    .config("spark.executorEnv.HOME", "/tmp") \
    .config("spark.executorEnv.PYTHONUSERBASE", "/tmp/.local") \
    .config("spark.executorEnv.PYTHONPATH", f"/tmp/.local/lib/{py_version}/site-packages:$PYTHONPATH") \
    .config("spark.executorEnv.DGLBACKEND", "pytorch") \
    .config("spark.executorEnv.DGL_DOWNLOAD_DIR", "/tmp/.dgl") \
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.0,graphframes:graphframes:0.8.3-spark3.5-s_2.12") \
    .config("spark.jars.ivy", f"{large_tmp}/.ivy2") \
    .config("spark.local.dir", f"{large_tmp}") \
    .config("spark.driver.extraJavaOptions", f"-Djava.io.tmpdir={large_tmp}") \
    .config("spark.executor.extraJavaOptions", f"-Djava.io.tmpdir={large_tmp}") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.databricks.delta.retentionDurationCheck.enabled", "false") \
    .config("spark.databricks.delta.vacuum.parallelDelete.enabled", "true") \
    .enableHiveSupport() \
    .getOrCreate()
sc = spark.sparkContext

# ── 1. CONFIGURATION ──────────────────────────────────────────────────────────
S3_BUCKET = 'us-east-1-s3-gnn'
S3_PREFIX = 'pipeline'

# ── 1.5. BOOTSTRAP BOTO3 ──────────────────────────────────────────────────────
try:
    import boto3
except ImportError:
    print("boto3 not found. Installing boto3 dynamically before download...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "boto3"])
    except Exception as e:
        print(f"Warning: Failed to install boto3: {e}")
    import boto3

# ── 2. DOWNLOAD PIPELINE ZIP AND CONFIG FROM S3 ───────────────────────────────────
print(f"Downloading latest scripts from s3://{S3_BUCKET}/{S3_PREFIX} ...")
s3 = boto3.client('s3')

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

# We will download experiment_config.py, requirements.txt, and reconstruct the pipeline module
paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX + '/')

download_count = 0
for page in pages:
    if 'Contents' not in page:
        continue
    for obj in page['Contents']:
        key = obj['Key']
        # Download files ending in .py or requirements.txt
        if not (key.endswith('.py') or key.endswith('requirements.txt')):
            continue
            
        # Get path relative to S3_PREFIX
        rel_path = key[len(S3_PREFIX) + 1:].strip('/')
        if not rel_path:
            continue
            
        if rel_path in ['experiment_config.py', 'requirements.txt']:
            local_path = f'/tmp/{rel_path}'
        else:
            local_path = f'/tmp/pipeline_stage/pipeline/{rel_path}'
            
        # Create local directory structure
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download file
        s3.download_file(S3_BUCKET, key, local_path)
        print(f"  - Downloaded key: {key} -> {local_path}")
        download_count += 1

print(f"  ✓ Downloaded {download_count} files from S3.")

# Fallback to download experiment_config.py from parent directory or bucket root if not in prefix
if not os.path.exists('/tmp/experiment_config.py'):
    print("  ► experiment_config.py not found in prefix. Trying parent directory or bucket root...")
    parent_prefix = os.path.dirname(S3_PREFIX)
    parent_key = f"{parent_prefix}/experiment_config.py" if parent_prefix else "experiment_config.py"
    try:
        s3.download_file(S3_BUCKET, parent_key, '/tmp/experiment_config.py')
        print(f"  ✓ Downloaded experiment_config.py from parent key: {parent_key}")
    except Exception:
        try:
            s3.download_file(S3_BUCKET, "experiment_config.py", '/tmp/experiment_config.py')
            print("  ✓ Downloaded experiment_config.py from bucket root")
        except Exception as e:
            print(f"  ⚠ Could not locate experiment_config.py: {e}")

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
        print("  ✓ Copied experiment_config.py from package to /tmp")
            
    # Pack '/tmp/pipeline_stage/pipeline' into '/tmp/pipeline.zip'
    shutil.make_archive("/tmp/pipeline", "zip", "/tmp/pipeline_stage", "pipeline")
    
    # Copy to /tmp/pipeline so driver can import it directly
    shutil.copytree('/tmp/pipeline_stage/pipeline', '/tmp/pipeline')
    print("  ✓ Zipped codebase package to /tmp/pipeline.zip and synchronized /tmp/pipeline")

# Register package zip on driver and YARN executors
if os.path.exists('/tmp/pipeline.zip'):
    sys.path.insert(0, '/tmp/pipeline.zip')
    sc.addPyFile('/tmp/pipeline.zip')
    print("  ✓ Registered pipeline.zip package on Spark Context")

# ── 2.5. INSTALL DEPENDENCIES DYNAMICALLY ─────────────────────────────────────
import contextlib
@contextlib.contextmanager
def silence_all():
    import os
    import sys
    null_fds = []
    try:
        null_file = open(os.devnull, 'w', encoding='utf-8')
        null_fd = null_file.fileno()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        null_fds.extend([saved_stdout_fd, saved_stderr_fd])
        sys.stdout = null_file
        sys.stderr = null_file
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        yield
    finally:
        if len(null_fds) >= 2:
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            for fd in null_fds:
                try:
                    os.close(fd)
                except Exception:
                    pass
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        try:
            null_file.close()
        except Exception:
            pass

print("\nVerifying and installing GRL Pipeline dependencies on driver and executors...")
def safe_install(pkg):
    import importlib
    import_map = {
        'scikit-learn': 'sklearn',
        'torch-geometric': 'torch_geometric',
        'dgl==1.1.3': 'dgl'
    }
    import_name = import_map.get(pkg, pkg)

    # Some packages are only imported on the driver node (plotting, excel report generation)
    # Installing them on executors is unnecessary and can cause lock conflicts or path pollution
    driver_only_packages = {'xlsxwriter', 'openpyxl', 'matplotlib', 'seaborn'}
    is_driver_only = pkg in driver_only_packages

    # Check driver
    driver_ok = False
    try:
        importlib.import_module(import_name)
        driver_ok = True
    except Exception:
        pass

    # Check executors
    executors_ok = False
    if is_driver_only:
        executors_ok = True
    else:
        try:
            num_executors = int(spark.conf.get("spark.executor.instances", "4"))
            def check_executor(iterator):
                import importlib
                try:
                    importlib.import_module(import_name)
                    return ["OK"]
                except Exception:
                    return ["Missing"]
            results = sc.parallelize(range(num_executors * 4), num_executors * 4) \
                        .mapPartitions(check_executor) \
                        .collect()
            if len(results) > 0 and all(r == "OK" for r in results):
                executors_ok = True
        except Exception:
            pass

    if driver_ok and executors_ok:
        print(f"  ✓ {pkg:<15} - Already installed on driver and executors (skipping)")
        return

    # 1. Install on Driver Node
    if not driver_ok:
        try:
            cmd = [sys.executable, '-m', 'pip', 'install', '--user', '--quiet', '--no-cache-dir']
            if pkg.startswith('dgl'):
                cmd += [pkg, '-f', 'https://data.dgl.ai/wheels/repo.html']
            else:
                cmd += [pkg]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"  ✓ {pkg:<15} - Package synced successfully on cluster driver node")
        except Exception as e:
            print(f"  ⚠ {pkg:<15} - Driver sync encountered a warning (may already exist): {e}")
    else:
        print(f"  ✓ {pkg:<15} - Already installed on cluster driver node")

    # 2. Parallel Install on YARN executors
    if not executors_ok:
        try:
            num_executors = int(spark.conf.get("spark.executor.instances", "4"))
            def run_executor_install(iterator):
                import subprocess
                import sys
                import os
                import time
                import importlib

                import_map = {
                    'scikit-learn': 'sklearn',
                    'torch-geometric': 'torch_geometric',
                    'dgl==1.1.3': 'dgl'
                }
                import_name = import_map.get(pkg, pkg)

                # File/Directory lock to serialize installs on the same node
                lock_dir = f"/tmp/.local_{pkg}_lock"
                for _ in range(300):
                    try:
                        os.makedirs(lock_dir, exist_ok=False)
                        # We acquired the lock!
                        break
                    except FileExistsError:
                        # Lock is held by another task, check if already installed
                        # Catch all exceptions: if concurrent pip write is half-done,
                        # it may raise AttributeError / KeyError. We sleep and retry.
                        try:
                            importlib.import_module(import_name)
                            return ["Success"]
                        except Exception:
                            time.sleep(1)
                else:
                    return [f"Failed: Timeout waiting for lock for {pkg}"]

                try:
                    # Verify if already installed before launching pip
                    try:
                        importlib.import_module(import_name)
                        return ["Success"]
                    except Exception:
                        pass

                    try:
                        # Try installing globally using sudo /usr/bin/pip3
                        cmd = ['sudo', '/usr/bin/pip3', 'install', '--quiet', '--no-cache-dir']
                        if pkg.startswith('dgl'):
                            cmd += [pkg, '-f', 'https://data.dgl.ai/wheels/repo.html']
                        else:
                            cmd += [pkg]
                        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                        return ["Success"]
                    except Exception as e:
                        # Fallback to --user
                        try:
                            cmd = [sys.executable, '-m', 'pip', 'install', '--user', '--quiet', '--no-cache-dir']
                            if pkg.startswith('dgl'):
                                cmd += [pkg, '-f', 'https://data.dgl.ai/wheels/repo.html']
                            else:
                                cmd += [pkg]
                            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                            return ["Success"]
                        except Exception as e2:
                            return [f"Failed: {e2}"]
                finally:
                    try:
                        os.rmdir(lock_dir)
                    except Exception:
                        pass
            # Run parallelized mapPartitions to trigger installation on all executors
            results = sc.parallelize(range(num_executors * 4), num_executors * 4) \
                        .mapPartitions(run_executor_install) \
                        .collect()
            failures = [r for r in results if r != "Success"]
            if failures:
                raise RuntimeError(f"Sync failed on some executors: {failures}")
            print(f"  ✓ {pkg:<15} - PyPI package successfully synced on YARN cluster executors ({len(results)} tasks)")
        except Exception as e:
            print(f"  ⚠ {pkg:<15} - YARN executor sync failed: {e}")
    else:
        print(f"  ✓ {pkg:<15} - Already installed on YARN cluster executors")

packages = ['numpy', 'ogb', 'igraph', 'leidenalg', 'scikit-learn',
            'torch', 'boto3', 'xlsxwriter', 'openpyxl', 'matplotlib', 'seaborn',
            'torch-geometric', 'pyarrow']

for p in packages:
    safe_install(p)

print("\n  ► Installing DGL wheels onto cluster driver python environment...")
dgl_res = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '--user', '--quiet', '--no-cache-dir', 'dgl==1.1.3', '-f', 'https://data.dgl.ai/wheels/repo.html'],
    capture_output=True, text=True
)
if dgl_res.returncode == 0:
    print("  ✓ dgl==1.1.3      - Driver successfully linked with DGL engine")
else:
    print(f"  ⚠ DGL install returned a warning status (may already exist): {dgl_res.stderr[:120]}")

print("✓ Package installation completed successfully.")

import glob
local_site_packages = f'{large_tmp}/.local/lib/python*/site-packages'
for path in glob.glob(local_site_packages):
    if path not in sys.path:
        sys.path.insert(0, path)
        print(f"Added {path} to python path.")

# Add /tmp to python path for experiment_config.py loading
if '/tmp' in sys.path:
    sys.path.remove('/tmp')
sys.path.insert(0, '/tmp')

# Load the experiment configuration
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

print("\nConfig and Modular Pipeline fully loaded into EMR Context.")

# Initialize dynamic registries
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

# Override settings from experiment_config
EXPERIMENT_NAME = getattr(config, 'EXPERIMENT_NAME', 'emr_step')
DATASETS_TO_RUN = getattr(config, 'DATASETS_TO_RUN', ['ogbn-mag'])
ALGORITHMS_TO_RUN = getattr(config, 'ALGORITHMS_TO_RUN', ['lpa'])
TASK_TYPE = getattr(config, 'TASK_TYPE', 'both')

RUN_PHASE0 = getattr(config, 'RUN_PHASE0', True)
FORCE_REINGEST = getattr(config, 'FORCE_REINGEST', False)
USE_OGB_SPLITS = getattr(config, 'USE_OGB_SPLITS', True)
MIN_COMMUNITY_SIZE = getattr(config, 'MIN_COMMUNITY_SIZE', 10)
TINY_COMM_HANDLING = getattr(config, 'TINY_COMM_HANDLING', 'misc')
EXPAND_BOUNDARY_NODES = getattr(config, 'EXPAND_BOUNDARY_NODES', True)
USE_GLOBAL_MAPPING = getattr(config, 'USE_GLOBAL_MAPPING', True)

GCN_CFG = getattr(config, 'GCN_CFG', {})
BASELINE_CFG = getattr(config, 'BASELINE_CFG', {})

print("\n" + "="*60 + "\n  STARTING PIPELINE EXECUTION IN BACKGROUND\n" + "="*60)
t_pipeline_start = time.time()

# Build path helper function for S3 paths
get_paths_fn = lambda dataset, alg=None: get_paths(
    dataset, alg,
    experiment_name=EXPERIMENT_NAME,
    s3_bucket=S3_BUCKET
)

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

# Phase 1: Partition assignment / Community detection
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
    metis_k      = getattr(config, 'METIS_K', 100)
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
    expand_boundary_nodes = EXPAND_BOUNDARY_NODES
)

# Phase 3: Parallel GNN UDF Training
if getattr(config, 'RUN_PHASE3', True):
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
        models             = config.GNN_MODELS
    )

# Phase 3b: Parallel GNN Training with CaaN Global Graph
if ALGORITHMS_TO_RUN and getattr(config, 'RUN_PHASE3B', True):
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
        min_size           = MIN_COMMUNITY_SIZE
    )

# Phase 4: Full-Graph Baseline
if getattr(config, 'RUN_PHASE4', True):
    run_phase4(
        spark, sc,
        datasets     = DATASETS_TO_RUN,
        dataset_cfg  = config.DATASET_CFG,
        baseline_cfg = BASELINE_CFG,
        get_paths_fn = get_paths_fn,
        timing       = timing,
        results      = phase4_results,
        task_type    = TASK_TYPE,
        n_baseline_runs = getattr(config, 'N_BASELINE_RUNS', 3)
    )

    # Phase 4b: DistDGL Baseline
    run_phase4b(
        spark, sc,
        datasets     = DATASETS_TO_RUN,
        dataset_cfg  = config.DATASET_CFG,
        baseline_cfg = BASELINE_CFG,
        get_paths_fn = get_paths_fn,
        timing       = timing,
        results      = phase4b_results,
        task_type    = TASK_TYPE
    )

    # Phase 4c: ARMA Baseline
    print("\n[PHASE 4c] - ARMA Full-Graph Global Baseline (PyG)")
    run_phase4c(
        spark, sc,
        datasets     = DATASETS_TO_RUN,
        dataset_cfg  = config.DATASET_CFG,
        baseline_cfg = BASELINE_CFG,
        get_paths_fn = get_paths_fn,
        timing       = timing,
        results      = phase4c_results,
        task_type    = TASK_TYPE,
        n_baseline_runs = getattr(config, 'N_BASELINE_RUNS', 3)
    )

    # Phase 4d: ASAP Baseline
    print("\n[PHASE 4d] - ASAP Full-Graph Global Baseline (PyG)")
    run_phase4d(
        spark, sc,
        datasets     = DATASETS_TO_RUN,
        dataset_cfg  = config.DATASET_CFG,
        baseline_cfg = BASELINE_CFG,
        get_paths_fn = get_paths_fn,
        timing       = timing,
        results      = phase4d_results,
        task_type    = TASK_TYPE,
        n_baseline_runs = getattr(config, 'N_BASELINE_RUNS', 3)
    )

    # Phase 4e: GAT Baseline
    print("\n[PHASE 4e] - GAT Full-Graph Global Baseline (PyG)")
    run_phase4e(
        spark, sc,
        datasets     = DATASETS_TO_RUN,
        dataset_cfg  = config.DATASET_CFG,
        baseline_cfg = BASELINE_CFG,
        get_paths_fn = get_paths_fn,
        timing       = timing,
        results      = phase4e_results,
        task_type    = TASK_TYPE,
        n_baseline_runs = getattr(config, 'N_BASELINE_RUNS', 3)
    )

    # Phase 4f: Graph Transformer Baseline
    print("\n[PHASE 4f] - Graph Transformer Full-Graph Global Baseline (PyG)")
    run_phase4f(
        spark, sc,
        datasets     = DATASETS_TO_RUN,
        dataset_cfg  = config.DATASET_CFG,
        baseline_cfg = BASELINE_CFG,
        get_paths_fn = get_paths_fn,
        timing       = timing,
        results      = phase4f_results,
        task_type    = TASK_TYPE,
        n_baseline_runs = getattr(config, 'N_BASELINE_RUNS', 3)
    )

    # Phase 4g: ClusterSCL Baseline
    print("\n[PHASE 4g] - ClusterSCL Full-Graph Global Baseline (PyG)")
    run_phase4g(
        spark, sc,
        datasets     = DATASETS_TO_RUN,
        dataset_cfg  = config.DATASET_CFG,
        baseline_cfg = BASELINE_CFG,
        get_paths_fn = get_paths_fn,
        timing       = timing,
        results      = phase4g_results,
        task_type    = TASK_TYPE,
        n_baseline_runs = getattr(config, 'N_BASELINE_RUNS', 3)
    )

    # Phase 4h: GATv2 Baseline
    print("\n[PHASE 4h] - GATv2 Full-Graph Global Baseline (PyG)")
    run_phase4h(
        spark, sc,
        datasets     = DATASETS_TO_RUN,
        dataset_cfg  = config.DATASET_CFG,
        baseline_cfg = BASELINE_CFG,
        get_paths_fn = get_paths_fn,
        timing       = timing,
        results      = phase4h_results,
        task_type    = TASK_TYPE,
        n_baseline_runs = getattr(config, 'N_BASELINE_RUNS', 3)
    )

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
    s3_bucket       = S3_BUCKET,
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
print(f"\n[SUCCESS] EMR BACKGROUND STEP COMPLETED SUCCESSFULLY in {t_elapsed:.1f} seconds.")
print("="*80 + "\n")

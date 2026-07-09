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

# ── OVERWRITE DIRECTORIES TO USE LARGE VOLUMES (PREVENT ROOT DISK OOM) ───
candidates = [
    '/mnt/tmp', '/mnt1/tmp', '/mnt2/tmp',
    '/mnt/spark', '/mnt1/spark', '/mnt2/spark',
    '/mnt/var/tmp', '/mnt1/var/tmp',
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

    # ── 2. INITIALIZE SPARK SESSION WITH PRE-CONFIGURED SETTINGS ───────────────
    print("\n" + "="*80)
    print("  INITIALIZING SPARK SESSION (AWS YARN CONFIGURATION)")
    print("="*80)
    
    spark = SparkSession.builder \
        .appName(f"GRL-{args.experiment_name}") \
        .config("spark.master", "yarn") \
        .config("spark.driver.memory", "45g") \
        .config("spark.driver.maxResultSize", "24g") \
        .config("spark.driver.cores", "8") \
        .config("spark.driver.memoryOverhead", "8g") \
        .config("spark.dynamicAllocation.enabled", "false") \
        .config("spark.executor.instances", "4") \
        .config("spark.executor.memory", "200g") \
        .config("spark.executor.memoryOverhead", "32g") \
        .config("spark.executor.cores", "28") \
        .config("spark.sql.shuffle.partitions", "336") \
        .config("spark.default.parallelism", "336") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "1024m") \
        .config("spark.pyspark.python", "python3") \
        .config("spark.pyspark.virtualenv.enabled", "false") \
        .config("spark.executorEnv.PYTHONUSERBASE", "/tmp/.local") \
        .config("spark.executorEnv.PYTHONPATH", f"/tmp/.local/lib/{py_version}/site-packages") \
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
    print("  ✓ SparkSession successfully configured and initialized as 'spark'.")

    # ── 3. INSTALL PYTHON DEPENDENCIES ─────────────────────────────────────────
    if getattr(args, 'no_install', False):
        print("\n  ► Skipping dynamic package verification/installation on YARN executors...")
    else:
        print("\n" + "="*80)
        print("  VERIFYING AND INSTALLING PYTHON ENVIRONMENT PACKAGES")
        print("="*80)
        
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

        def safe_install(pkg):
            import importlib
            import_map = {
                'scikit-learn': 'sklearn',
                'torch-geometric': 'torch_geometric',
                'dgl==1.1.3': 'dgl'
            }
            import_name = import_map.get(pkg, pkg)

            # Check driver
            driver_ok = False
            try:
                importlib.import_module(import_name)
                driver_ok = True
            except ImportError:
                pass

            # Check executors
            executors_ok = False
            try:
                num_executors = int(spark.conf.get("spark.executor.instances", "4"))
                def check_executor(iterator):
                    import importlib
                    try:
                        importlib.import_module(import_name)
                        return ["OK"]
                    except ImportError:
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
                                try:
                                    importlib.import_module(import_name)
                                    return ["Success"]
                                except ImportError:
                                    time.sleep(1)
                        else:
                            return [f"Failed: Timeout waiting for lock for {pkg}"]

                        try:
                            # Verify if already installed before launching pip
                            try:
                                importlib.import_module(import_name)
                                return ["Success"]
                            except ImportError:
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
                    'torch-geometric', 'pyarrow', 'dgl==1.1.3']
        
        for p in packages:
            safe_install(p)

        print("\n  ✓ ALL PIPELINE DEPENDENCIES VERIFIED AND READY.")

    # ── 4. FETCH AND PACK SCRIPTS ──────────────────────────────────────────────
    print("\n" + "="*80)
    print("  BOOTSTRAPPING EXPERIMENT CODE AND STAGE UTILITIES")
    print("="*80)
    
    import boto3
    s3 = boto3.client('s3')

    # Locate/Download the config
    if args.local:
        print("  ► LOADING PIPELINE AND CONFIG FROM LOCAL PATHS...")
        # Clean up previous run leftovers in /tmp to avoid package/namespace conflicts
        for path in ['/tmp/pipeline', '/tmp/pipeline.zip', '/tmp/experiment_config.py']:
            if os.path.exists(path):
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                except Exception:
                    pass

        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cfg_path = os.path.join(script_dir, "experiment_config.py")
        shutil.copyfile(cfg_path, "/tmp/experiment_config.py")
        
        # Package local pipeline folder into a zip
        parent_dir = os.path.dirname(script_dir)
        base_dir = os.path.basename(script_dir)
        shutil.make_archive("/tmp/pipeline", "zip", parent_dir, base_dir)
        
        # Copy to /tmp/pipeline so driver can import it directly
        shutil.copytree(script_dir, '/tmp/pipeline')
        print("    ✓ Packed local pipeline folder to /tmp/pipeline.zip and copied to /tmp/pipeline")
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
    
    BASELINE_CFG = {
        'epochs':     n_epochs,
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
            s3_bucket          = args.s3_bucket,
            experiment_name    = EXPERIMENT_NAME
        )

    # Phase 3b: Parallel GNN Training with CaaN Global Graph
    if ALGORITHMS_TO_RUN and args.run_phase3b:
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
        if 'sage' in config.GNN_MODELS:
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
        if 'gat' in config.GNN_MODELS:
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
        if 'transformer' in config.GNN_MODELS:
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
        if 'clusterscl' in config.GNN_MODELS:
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
        if 'gatv2' in config.GNN_MODELS:
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

if __name__ == "__main__":
    main()

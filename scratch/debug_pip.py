import sys
import os

# Set up Spark import paths on EMR
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

def main():
    spark = SparkSession.builder \
        .appName("DebugAllPipInstalls") \
        .config("spark.master", "yarn") \
        .getOrCreate()
        
    sc = spark.sparkContext
    num_executors = int(spark.conf.get("spark.executor.instances", "2"))
    
    # We want to debug installation of all executor dependencies
    def run_debug_pip_all(iterator):
        import subprocess
        import sys
        import os
        
        large_tmp = "/mnt/tmp"
        os.environ["PYTHONUSERBASE"] = f"{large_tmp}/.local"
        os.environ["TMPDIR"] = large_tmp
        
        # Test libraries
        libs = ['numpy', 'scikit-learn', 'torch', 'torch-geometric', 'pyarrow', 'dgl==1.1.3']
        results = []
        
        for lib in libs:
            # First check if import works
            import_name = 'sklearn' if lib == 'scikit-learn' else 'torch_geometric' if lib == 'torch-geometric' else lib.split('==')[0]
            try:
                import importlib
                importlib.import_module(import_name)
                results.append(f"{lib:<20} | IMPORT: OK")
                continue
            except Exception:
                pass
                
            # If not importable, attempt pip install
            cmd = [sys.executable, '-m', 'pip', 'install', '--user', '--no-cache-dir']
            if lib.startswith('dgl'):
                cmd += [lib, '-f', 'https://data.dgl.ai/wheels/repo.html']
            else:
                cmd += [lib]
                
            try:
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                results.append(f"{lib:<20} | PIP INSTALL: SUCCESS\nstdout: {res.stdout.strip()[:100]}...")
            except subprocess.CalledProcessError as e:
                results.append(f"{lib:<20} | PIP INSTALL: FAILED (code={e.returncode})\nstdout: {e.stdout.strip()}\nstderr: {e.stderr.strip()}")
                
        return ["\n".join(results)]
            
    results = sc.parallelize(range(num_executors), num_executors) \
                .mapPartitions(run_debug_pip_all) \
                .collect()
                
    print("=== PIP VERIFICATION RESULTS FOR ALL EXECUTORS ===")
    for idx, r in enumerate(results):
        print(f"\nNode / Executor Partition {idx}:\n{r}")
        
    spark.stop()

if __name__ == "__main__":
    main()

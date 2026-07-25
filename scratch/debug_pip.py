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
        .appName("DebugPipInstall") \
        .config("spark.master", "yarn") \
        .getOrCreate()
        
    sc = spark.sparkContext
    num_executors = int(spark.conf.get("spark.executor.instances", "2"))
    
    # We want to run this on the executors
    def run_debug_pip(iterator):
        import subprocess
        import sys
        import os
        
        large_tmp = "/mnt/tmp"
        os.environ["PYTHONUSERBASE"] = f"{large_tmp}/.local"
        os.environ["TMPDIR"] = large_tmp
        
        cmd = [sys.executable, '-m', 'pip', 'install', '--user', '--no-cache-dir', 'numpy']
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            return [f"SUCCESS: {res.stdout}"]
        except subprocess.CalledProcessError as e:
            return [f"FAILED: code={e.returncode}\nstdout={e.stdout}\nstderr={e.stderr}"]
            
    results = sc.parallelize(range(num_executors * 2), num_executors * 2) \
                .mapPartitions(run_debug_pip) \
                .collect()
                
    print("=== PIP EXECUTION RESULTS ===")
    for idx, r in enumerate(results):
        print(f"\nExecutor {idx}:\n{r}")
        
    spark.stop()

if __name__ == "__main__":
    main()

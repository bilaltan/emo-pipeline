import sys
import os
import socket
import subprocess
from pyspark.sql import SparkSession

def check_worker_pip(iterator):
    node_name = socket.gethostname()
    
    results = [
        f"\n=======================================================",
        f"  DIAGNOSTIC REPORT FOR NODE: {node_name}",
        f"======================================================="
    ]
    
    # Discover writeable path
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
    results.append(f"Resolved PYTHONUSERBASE: {os.environ['PYTHONUSERBASE']}")
    
    try:
        import torch
        results.append(f"✓ torch is already importable on this node: {torch.__file__}")
        return results
    except Exception as e:
        results.append(f"✗ torch import failed currently: {e}")

    # Run the CPU installation test
    cmd = [
        sys.executable, '-m', 'pip', 'install', 
        '--user', '--no-cache-dir', 'torch', 
        '--index-url', 'https://download.pytorch.org/whl/cpu'
    ]
    results.append(f"Executing test command: {' '.join(cmd)}")
    
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        results.append(f"Exit code: {proc.returncode}")
        results.append(f"--- STDOUT --- \n{proc.stdout}")
        results.append(f"--- STDERR --- \n{proc.stderr}")
    except Exception as e:
        results.append(f"Failed to execute command: {e}")
        
    return results

if __name__ == "__main__":
    spark = SparkSession.builder.appName("DiagnoseWorkers").getOrCreate()
    sc = spark.sparkContext
    num_executors = int(spark.conf.get("spark.executor.instances", "8"))
    
    print("Launching diagnostics Spark job across executors...")
    reports = sc.parallelize(range(num_executors * 4), num_executors * 4) \
                .mapPartitions(check_worker_pip) \
                .collect()
                
    # Filter to show unique reports per host
    seen_hosts = set()
    for line in reports:
        print(line)

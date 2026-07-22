#!/usr/bin/env python3
import os
import sys
import secrets
import boto3
from experiment_config import EXPERIMENT_NAME

S3_BUCKET = "us-east-1-s3-gnn"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def upload_latex_tables_and_results(run_id=None, s3_bucket=S3_BUCKET):
    """
    Uploads LaTeX tables, execution log, and Excel results to S3 in a unique consolidated run folder.
    S3 structure:
      s3://{s3_bucket}/gnn-bench-out/spark-results/{EXPERIMENT_NAME}-{run_id}/logs/run_pipeline.log
      s3://{s3_bucket}/gnn-bench-out/spark-results/{EXPERIMENT_NAME}-{run_id}/excels/{EXPERIMENT_NAME}_results.xlsx
      s3://{s3_bucket}/gnn-bench-out/spark-results/{EXPERIMENT_NAME}-{run_id}/latex_tables/*.tex
    """
    if run_id is None:
        run_id = secrets.token_hex(8)
    
    consolidated_folder = f"{EXPERIMENT_NAME}-{run_id}"
    s3_client = boto3.client('s3')
    print(f"=== Uploading Results to S3 Folder: s3://{s3_bucket}/gnn-bench-out/spark-results/{consolidated_folder}/ ===")

    # 1. Upload log file if present
    log_path = os.path.join(PROJECT_ROOT, "run_pipeline.log")
    if os.path.exists(log_path):
        s3_log_key = f"gnn-bench-out/spark-results/{consolidated_folder}/logs/run_pipeline.log"
        print(f"Uploading log: s3://{s3_bucket}/{s3_log_key}")
        s3_client.upload_file(log_path, s3_bucket, s3_log_key)
    
    # 2. Upload Excel file if present
    excel_path = os.path.join(PROJECT_ROOT, "results", f"{EXPERIMENT_NAME}_results.xlsx")
    if not os.path.exists(excel_path):
        excel_path = os.path.join(PROJECT_ROOT, f"{EXPERIMENT_NAME}_results.xlsx")
    
    if os.path.exists(excel_path):
        s3_excel_key = f"gnn-bench-out/spark-results/{consolidated_folder}/excels/{EXPERIMENT_NAME}_results.xlsx"
        print(f"Uploading excel: s3://{s3_bucket}/{s3_excel_key}")
        s3_client.upload_file(excel_path, s3_bucket, s3_excel_key)

    # 3. Upload LaTeX tables from results/*.tex to /latex_tables
    results_dir = os.path.join(PROJECT_ROOT, "results")
    if os.path.exists(results_dir):
        for fname in os.listdir(results_dir):
            if fname.endswith(".tex"):
                local_tex = os.path.join(results_dir, fname)
                s3_tex_key = f"gnn-bench-out/spark-results/{consolidated_folder}/latex_tables/{fname}"
                print(f"Uploading LaTeX table: s3://{s3_bucket}/{s3_tex_key}")
                s3_client.upload_file(local_tex, s3_bucket, s3_tex_key)

    print("=== Upload Complete ===")

def upload_code_to_s3(s3_bucket=S3_BUCKET):
    """
    Uploads all local code files (experiment_config.py, phases/*.py, utils/*.py, runners/*.py)
    to s3://{s3_bucket}/pipeline/ so EMR nodes execute the latest local code.
    """
    s3_client = boto3.client('s3')
    print(f"=== Syncing Local Code to S3: s3://{s3_bucket}/pipeline/ ===")
    
    root_files = ['experiment_config.py', '__init__.py']
    for rf in root_files:
        local_f = os.path.join(PROJECT_ROOT, rf)
        if os.path.exists(local_f):
            s3_key = f"pipeline/{rf}"
            s3_client.upload_file(local_f, s3_bucket, s3_key)
            print(f"  ✓ Uploaded {rf} -> s3://{s3_bucket}/{s3_key}")
            
    for sub in ['phases', 'utils', 'runners']:
        sub_dir = os.path.join(PROJECT_ROOT, sub)
        if os.path.exists(sub_dir):
            for fname in os.listdir(sub_dir):
                if fname.endswith('.py'):
                    local_f = os.path.join(sub_dir, fname)
                    s3_key = f"pipeline/{sub}/{fname}"
                    s3_client.upload_file(local_f, s3_bucket, s3_key)
                    print(f"  ✓ Uploaded {sub}/{fname} -> s3://{s3_bucket}/{s3_key}")
    print("=== Code Sync Complete ===")

if __name__ == "__main__":
    upload_code_to_s3()
    upload_latex_tables_and_results()


#!/usr/bin/env python3
import os
import sys
import secrets
import boto3
from experiment_config import EXPERIMENT_NAME

S3_BUCKET = "us-east-1-s3-gnn"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def upload_latex_tables_and_results(run_id=None, s3_bucket=S3_BUCKET, dest_subdir=None):
    """
    Uploads LaTeX tables, execution log, and Excel results to S3 in a unique consolidated run folder.
    S3 structure:
      s3://{s3_bucket}/gnn-bench-out/{dest_subdir}/{EXPERIMENT_NAME}-{run_id}/logs/run_pipeline.log
      s3://{s3_bucket}/gnn-bench-out/{dest_subdir}/{EXPERIMENT_NAME}-{run_id}/excels/{EXPERIMENT_NAME}_results.xlsx
      s3://{s3_bucket}/gnn-bench-out/{dest_subdir}/{EXPERIMENT_NAME}-{run_id}/latex_tables/*.tex
    """
    if run_id is None:
        run_id = secrets.token_hex(8)
    
    if dest_subdir is None:
        dest_subdir = os.environ.get("S3_DEST_SUBDIR", "spark-results")
    
    consolidated_folder = f"{EXPERIMENT_NAME}-{run_id}"
    s3_client = boto3.client('s3')
    print(f"=== Uploading Results to S3 Folder: s3://{s3_bucket}/gnn-bench-out/{dest_subdir}/{consolidated_folder}/ ===")

    # 1. Upload log file if present
    log_path = os.path.join(PROJECT_ROOT, "run_pipeline.log")
    if os.path.exists(log_path):
        s3_log_key = f"gnn-bench-out/{dest_subdir}/{consolidated_folder}/logs/run_pipeline.log"
        print(f"Uploading log: s3://{s3_bucket}/{s3_log_key}")
        s3_client.upload_file(log_path, s3_bucket, s3_log_key)
    
    # 2. Upload Excel file if present
    excel_path = os.path.join(PROJECT_ROOT, "results", f"{EXPERIMENT_NAME}_results.xlsx")
    if not os.path.exists(excel_path):
        excel_path = os.path.join(PROJECT_ROOT, f"{EXPERIMENT_NAME}_results.xlsx")
    
    if os.path.exists(excel_path):
        s3_excel_key = f"gnn-bench-out/{dest_subdir}/{consolidated_folder}/excels/{EXPERIMENT_NAME}_results.xlsx"
        print(f"Uploading excel: s3://{s3_bucket}/{s3_excel_key}")
        s3_client.upload_file(excel_path, s3_bucket, s3_excel_key)

    # 3. Upload LaTeX tables from results/*.tex to /latex_tables
    results_dir = os.path.join(PROJECT_ROOT, "results")
    if os.path.exists(results_dir):
        for fname in os.listdir(results_dir):
            if fname.endswith(".tex"):
                local_tex = os.path.join(results_dir, fname)
                s3_tex_key = f"gnn-bench-out/{dest_subdir}/{consolidated_folder}/latex_tables/{fname}"
                print(f"Uploading LaTeX table: s3://{s3_bucket}/{s3_tex_key}")
                s3_client.upload_file(local_tex, s3_bucket, s3_tex_key)

    # 4. If 2worker folder, also copy directly to s3_latest_results_2worker top-level
    if "2worker" in dest_subdir:
        if os.path.exists(excel_path):
            s3_top_key = f"gnn-bench-out/s3_latest_results_2worker/{os.path.basename(excel_path)}"
            s3_client.upload_file(excel_path, s3_bucket, s3_top_key)

    print("=== Upload Complete ===")

def upload_code_to_s3(s3_bucket=S3_BUCKET):
    """
    Uploads all local code files (experiment_config.py, phases/*.py, utils/*.py, runners/*.py)
    to s3://{s3_bucket}/pipeline/ so EMR nodes execute the latest local code.
    """
    s3_client = boto3.client('s3')
    print(f"=== Uploading Local Code Files to s3://{s3_bucket}/pipeline/ ===")

    include_dirs = ["phases", "utils", "runners", "models", "data", "scripts"]
    include_files = ["experiment_config.py", "upload_to_s3.py"]

    for root_file in include_files:
        local_path = os.path.join(PROJECT_ROOT, root_file)
        if os.path.exists(local_path):
            s3_key = f"pipeline/{root_file}"
            print(f"  Uploading {root_file} -> s3://{s3_bucket}/{s3_key}")
            s3_client.upload_file(local_path, s3_bucket, s3_key)

    for d in include_dirs:
        dir_path = os.path.join(PROJECT_ROOT, d)
        if os.path.exists(dir_path):
            for root, _, files in os.walk(dir_path):
                for f in files:
                    if f.endswith(".py") or f.endswith(".sh") or f.endswith(".json") or f.endswith(".sql"):
                        local_path = os.path.join(root, f)
                        rel_path = os.path.relpath(local_path, PROJECT_ROOT)
                        s3_key = f"pipeline/{rel_path}"
                        print(f"  Uploading {rel_path} -> s3://{s3_bucket}/{s3_key}")
                        s3_client.upload_file(local_path, s3_bucket, s3_key)

    print("=== Code Upload Complete ===")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--code-only":
        upload_code_to_s3()
    else:
        upload_latex_tables_and_results()

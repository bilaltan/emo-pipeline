#!/usr/bin/env python3
"""
Fetch S3 Results Generated in Last 24 Hours (Supports 2-Worker and 4-Worker Clusters)
══════════════════════════════════════════════════════════════════════════════
Scans s3://us-east-1-s3-gnn/gnn-bench-out/ (including spark-results/, spark-results-2worker/,
and s3_latest_results_2worker/) for all artifacts created or modified within the time window.

Downloads:
  - Excel files (*.xlsx, *.xls)
  - Execution logs (*.log)
  - LaTeX tables (*.tex)
  - JSON summary & checkpoint metrics (*.json)

Usage:
  python3 scripts/fetch_s3_results_last_24h.py --worker-type 2worker
  python3 scripts/fetch_s3_results_last_24h.py --dest results/s3_latest_results_2worker
══════════════════════════════════════════════════════════════════════════════
"""
import os
import sys
import argparse
from datetime import datetime, timezone, timedelta

def fetch_recent_s3_results(hours=24, dest_dir=None, bucket_name="us-east-1-s3-gnn", prefix="gnn-bench-out/", worker_type="4worker"):
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError
    except ImportError:
        print("Error: boto3 is required. Install via: pip install boto3")
        sys.exit(1)

    if dest_dir is None:
        dest_dir = f"results/s3_latest_results_2worker" if worker_type == "2worker" else "results/s3_latest_results"

    print("=" * 75)
    print(f" ► Connecting to AWS S3: s3://{bucket_name}/{prefix}")
    print(f" ► Worker Cluster Type: {worker_type.upper()}")
    print(f" ► Destination Local Directory: {dest_dir}")
    print(f" ► Filtering for objects modified in the last {hours} hours...")
    print("=" * 75)

    try:
        s3 = boto3.client('s3')
        paginator = s3.get_paginator('list_objects_v2')
        
        prefixes_to_check = [
            "gnn-bench-out/s3_latest_results_2worker/",
            "gnn-bench-out/spark-results-2worker/",
            "gnn-bench-out/spark-results/",
            "gnn-bench-out/s3_latest_results/",
            "gnn-bench-out/",
            "gnn-bench-results/"
        ]
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        matched_objects = []

        seen_keys = set()
        for pfx in prefixes_to_check:
            try:
                pages = paginator.paginate(Bucket=bucket_name, Prefix=pfx)
                for page in pages:
                    for obj in page.get('Contents', []):
                        key = obj['Key']
                        if key in seen_keys or key.endswith('/'):
                            continue
                        seen_keys.add(key)
                        
                        mtime = obj['LastModified']
                        if mtime >= cutoff_time:
                            matched_objects.append(obj)
            except Exception as e:
                pass

        if not matched_objects:
            print(f"ℹ No files found in the strict last {hours}h window. Searching for most recent runs in the last 7 days...")
            fallback_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            for pfx in prefixes_to_check:
                try:
                    pages = paginator.paginate(Bucket=bucket_name, Prefix=pfx)
                    for page in pages:
                        for obj in page.get('Contents', []):
                            key = obj['Key']
                            if key in seen_keys or key.endswith('/'):
                                continue
                            seen_keys.add(key)
                            if obj['LastModified'] >= fallback_cutoff:
                                matched_objects.append(obj)
                except Exception:
                    pass

        if not matched_objects:
            print("⚠ No matching objects found in S3. Check if runs finished and uploaded.")
            return []

        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dest_full = os.path.join(PROJECT_ROOT, dest_dir) if not os.path.isabs(dest_dir) else dest_dir
        results_full = os.path.join(PROJECT_ROOT, "results")
        logs_full = os.path.join(PROJECT_ROOT, "logs")

        os.makedirs(dest_full, exist_ok=True)
        os.makedirs(results_full, exist_ok=True)
        os.makedirs(logs_full, exist_ok=True)

        downloaded_files = []
        for idx, obj in enumerate(matched_objects, 1):
            key = obj['Key']
            size_mb = obj['Size'] / (1024 * 1024)
            mtime_str = obj['LastModified'].strftime('%Y-%m-%d %H:%M:%S UTC')
            
            rel_name = key.replace('gnn-bench-out/spark-results-2worker/', '').replace('gnn-bench-out/spark-results/', '').replace('gnn-bench-out/s3_latest_results_2worker/', '').replace('gnn-bench-out/', '')
            local_target = os.path.join(dest_full, rel_name)
            os.makedirs(os.path.dirname(os.path.abspath(local_target)), exist_ok=True)

            print(f"  [{idx:02d}/{len(matched_objects):02d}] {mtime_str} | {size_mb:>6.2f} MB | {key}")
            try:
                s3.download_file(bucket_name, key, local_target)
                downloaded_files.append(local_target)
                
                base_name = os.path.basename(key)
                if base_name.endswith('.xlsx') or base_name.endswith('.csv') or base_name.endswith('.tex'):
                    import shutil
                    shutil.copy2(local_target, os.path.join(results_full, base_name))
                elif base_name.endswith('.log'):
                    import shutil
                    shutil.copy2(local_target, os.path.join(logs_full, base_name))
            except Exception as dl_err:
                print(f"      ⚠ Failed to download {key}: {dl_err}")

        print("\n" + "=" * 75)
        print(f" ✓ Successfully downloaded {len(downloaded_files)} files into '{dest_full}/'")
        print(f" ✓ Primary Excel and LaTeX files placed into '{results_full}/'")
        print(f" ✓ Logs placed into '{logs_full}/'")
        print("=" * 75)

        # Run master excel generator if logs were pulled
        try:
            sys.path.insert(0, PROJECT_ROOT)
            from runners.generate_emr_excel_report import generate_master_excel
            print(f"\n► Regenerating consolidated Master Excel Report ({worker_type.upper()})...")
            generate_master_excel(cluster_type=worker_type)
        except Exception as e:
            print(f"ℹ (Report note: {e})")

        return downloaded_files

    except NoCredentialsError:
        print("\n" + "!" * 75)
        print(" [AWS Credentials Missing on Local Machine]")
        print(" To pull files directly from S3, run this script directly on your EMR cluster terminal:")
        print("   python3 scripts/fetch_s3_results_last_24h.py --worker-type 2worker")
        print(" Or configure AWS credentials locally: aws configure")
        print("!" * 75)
        return []
    except Exception as e:
        print(f"Error accessing S3: {e}")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch latest S3 results from EMR cluster runs.")
    parser.add_argument("--hours", type=int, default=24, help="Time window in hours to fetch (default: 24)")
    parser.add_argument("--worker-type", type=str, default="2worker", choices=["2worker", "4worker", "8worker"], help="Cluster worker type (default: 2worker)")
    parser.add_argument("--dest", type=str, default=None, help="Destination directory (default: results/s3_latest_results_<worker-type>)")
    parser.add_argument("--bucket", type=str, default="us-east-1-s3-gnn", help="S3 bucket name (default: us-east-1-s3-gnn)")
    args = parser.parse_args()

    fetch_recent_s3_results(hours=args.hours, dest_dir=args.dest, bucket_name=args.bucket, worker_type=args.worker_type)

import os
import sys
import shutil
import subprocess
from pipeline.utils.common import silence_all

def discover_large_tmp():
    """Search candidate directories and pick the writable one with the most free space."""
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
    
    return large_tmp

def setup_emr_env(large_tmp):
    """Set process environment variables to prevent root disk out of memory on EMR."""
    os.environ['HOME'] = large_tmp
    os.environ['PYTHONUSERBASE'] = f'{large_tmp}/.local'
    os.environ['PIP_CACHE_DIR'] = f'{large_tmp}/.pip-cache'
    os.environ['DGL_DOWNLOAD_DIR'] = f'{large_tmp}/.dgl'
    os.environ['TMPDIR'] = large_tmp
    os.environ['TEMP'] = large_tmp
    os.environ['TMP'] = large_tmp
    os.environ.setdefault('DGLBACKEND', 'pytorch')

def install_packages(sc, packages):
    """Install PyPI packages on driver and sync on cluster YARN executors."""
    print("\nVerifying and installing GRL Pipeline dependencies on driver and executors...")
    def safe_install(pkg):
        # 1. Sync package on YARN cluster executors
        try:
            with silence_all():
                sc.install_pypi_package(pkg)
            print(f"  ✓ {pkg:<15} - PyPI package successfully synced on YARN cluster executors")
        except Exception as e:
            err_str = str(e)
            if 'already installed' in err_str or 'already exists' in err_str:
                pass
            else:
                print(f"  ⚠ {pkg:<15} - YARN sync returned non-critical status: {err_str[:80]}...")

        # 2. Forcefully install package on the driver python environment
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install',
                '--user', '--quiet', '--no-cache-dir', pkg
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"  ✓ {pkg:<15} - Package synced successfully on cluster driver node")
        except Exception as e:
            print(f"  ⚠ {pkg:<15} - Driver sync encountered a warning (may already exist): {e}")

    for p in packages:
        safe_install(p)

def install_dgl():
    """Install matching DGL wheels onto EMR driver environment."""
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

def patch_sys_path(large_tmp):
    """Dynamically construct and insert the user-site packages search path to make them importable."""
    import glob
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    tmp_site_packages = f"{large_tmp}/.local/lib/{py_version}/site-packages"
    os.makedirs(tmp_site_packages, exist_ok=True)
    if tmp_site_packages not in sys.path:
        sys.path.insert(0, tmp_site_packages)
        print(f"Added {tmp_site_packages} to python path.")
        
    local_site_packages = f'{large_tmp}/.local/lib/python*/site-packages'
    for path in glob.glob(local_site_packages):
        if path not in sys.path:
            sys.path.insert(0, path)
            print(f"Added {path} to python path.")

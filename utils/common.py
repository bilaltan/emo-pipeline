import os
import sys
import contextlib

@contextlib.contextmanager
def silence_all():
    """A context manager that redirects stdout and stderr at both the Python
    and C/system level (file descriptors 1 and 2) to devnull, silencing everything.
    """
    null_fds = []
    try:
        null_file = open(os.devnull, 'w', encoding='utf-8')
        null_fd = null_file.fileno()
        
        # Save stream pointers
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        # Save low-level fds
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        null_fds.extend([saved_stdout_fd, saved_stderr_fd])
        
        # Redirect sys.stdout and sys.stderr in Python
        sys.stdout = null_file
        sys.stderr = null_file
        
        # Redirect file descriptors 1 and 2 to /dev/null
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

def _patch_torch_load():
    """Monkeypatch torch.load and torch.mps to ensure backward/platform compatibility."""
    try:
        import torch
        import inspect
        if not hasattr(torch, '_orig_load_patched'):
            _orig = torch.load
            torch._orig_load_patched = _orig
            def _patched(*args, **kwargs):
                sig = inspect.signature(_orig)
                if 'weights_only' in sig.parameters:
                    kwargs['weights_only'] = False
                return _orig(*args, **kwargs)
            torch.load = _patched
        if hasattr(torch, 'mps') and not hasattr(torch.mps, 'current_device'):
            torch.mps.current_device = lambda: 0
    except Exception:
        pass

def _delta_exists(spark, path):
    """Check whether a Delta table already exists at the given path.
    Works for both S3 (s3://) and local (file://) paths.
    """
    try:
        # Try reading the Delta log — if it succeeds, the table exists
        spark.read.format('delta').load(path).limit(1).count()
        return True
    except Exception:
        return False

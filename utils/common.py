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
        from delta.tables import DeltaTable

        # Metadata-only probe: avoid a data scan just to test checkpoint reuse.
        return bool(DeltaTable.isDeltaTable(spark, path))
    except Exception:
        try:
            # Fallback for environments where DeltaTable.isDeltaTable is unavailable.
            spark.read.format('delta').load(path).limit(1).count()
            return True
        except Exception:
            return False


def resolve_member_mask(pdf, n_rows, n_local=None):
    """Boolean mask marking which rows are a community's own vertices.

    Phase 2's 1-hop boundary expansion borrows halo vertices from neighbouring
    communities, and those rows carry their real labels and splits. Counting them
    scores the same vertex once per community that borrows it, in communities
    whose local model never trained on its neighbourhood. Phase 3 and Phase 3b
    must mask identically or their metrics are not comparable — this lives in one
    place so the two cannot drift apart again.

    Membership is read from Phase 2's ``is_member`` column, the bundled
    ``_is_member_list``, or a positional ``_n_local`` marker, in that order;
    absent all three every row is treated as a member, which is correct for
    tables written before halo tracking existed.

    ``n_local`` is for frames that append context rows after the community's own
    (Phase 3b appends super-nodes and minor nodes). Those trailing rows are never
    members.
    """
    import numpy as np
    import pandas as pd

    block = n_rows if n_local is None else int(n_local)

    if 'is_member' in getattr(pdf, 'columns', []):
        mask = np.array(
            [bool(v) if not (pd.isna(v) or v is None) else True
             for v in pdf['is_member'].values], dtype=bool)
    elif '_is_member_list' in getattr(pdf, 'columns', []):
        raw = pdf['_is_member_list'].iloc[0]
        mask = np.array([bool(v) for v in (raw if raw is not None else [])], dtype=bool)
    elif '_n_local' in getattr(pdf, 'columns', []):
        mask = np.arange(block) < int(pdf['_n_local'].iloc[0])
    else:
        mask = np.ones(block, dtype=bool)

    if len(mask) != block:
        mask = np.resize(mask, (block,))
    if n_local is None:
        return mask
    return np.concatenate([mask, np.zeros(max(0, n_rows - block), dtype=bool)])

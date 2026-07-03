import os

def get_paths(dataset, alg=None, experiment_name='run-all', s3_bucket=None, local_data_dir=None):
    """Unified S3 and Local path generation helper for GRL experiments."""
    if local_data_dir is not None:
        # Local paths (file:// scheme handled implicitly by spark/os paths)
        root = os.path.join(local_data_dir, "delta-data", dataset)
        p = {
            'root':            root,
            'nodes':           os.path.join(root, 'nodes'),
            'edges':           os.path.join(root, 'edges'),
            'masks':           os.path.join(root, 'masks'),
            'original_nodes':  os.path.join(root, 'original_nodes'),
            'original_edges':  os.path.join(root, 'original_edges'),
            'checkpoints':     os.path.join(local_data_dir, 'checkpoints', dataset),
            'phase4_xlsx':     os.path.join(local_data_dir, 'gnn-bench-out', f'{experiment_name}_{dataset}_phase4.xlsx'),
        }
        if alg:
            tag = f'{experiment_name}_{dataset}_{alg}'
            p.update({
                'communities': os.path.join(root, 'communities', alg),
                'p2_nodes':    os.path.join(root, 'phase2_nodes', tag),
                'p2_edges':    os.path.join(root, 'phase2_edges', tag),
                'phase3_xlsx': os.path.join(local_data_dir, 'gnn-bench-out', f'{tag}_phase3.xlsx'),
                'models':      os.path.join(local_data_dir, 'gnn-bench-out', 'models', tag),
                'tag':         tag,
            })
    else:
        # S3 paths
        bucket = s3_bucket or 'us-east-1-s3-gnn'
        root = f's3://{bucket}/delta-data/{dataset}'
        p = {
            'root':            root,
            'nodes':           f'{root}/nodes/',
            'edges':           f'{root}/edges/',
            'masks':           f'{root}/masks/',
            'original_nodes':  f'{root}/original_nodes/',
            'original_edges':  f'{root}/original_edges/',
            'checkpoints':     f's3://{bucket}/checkpoints/{dataset}/',
            'phase4_xlsx':     (f's3://{bucket}/gnn-bench-out/'
                                f'{experiment_name}_{dataset}_phase4.xlsx'),
        }
        if alg:
            tag = f'{experiment_name}_{dataset}_{alg}'
            p.update({
                'communities': f'{root}/communities/{alg}/',
                'p2_nodes':    f'{root}/phase2_nodes/{tag}/',
                'p2_edges':    f'{root}/phase2_edges/{tag}/',
                'phase3_xlsx': (f's3://{bucket}/gnn-bench-out/'
                                f'{tag}_phase3.xlsx'),
                'models':      f's3://{bucket}/gnn-bench-out/models/{tag}/',
                'tag':         tag,
            })
    return p

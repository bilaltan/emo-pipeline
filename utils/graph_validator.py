import sys
import numpy as np

def validate_graph_properties(n_nodes, src, dst):
    """
    Validate graph integrity before community detection.
    Check for self-loops, duplicate edges, and verify undirected symmetry.
    """
    print("\n" + "="*50)
    print("  GRAPH INTEGRITY VALIDATION")
    print("="*50)
    
    # 1. Basic Counts
    print(f"Nodes count (raw): {n_nodes:,}")
    print(f"Edges count (raw): {len(src):,}")
    
    # 2. Self-loops check
    self_loops = np.sum(src == dst)
    print(f"Self-loops detected: {self_loops:,} ({'WARNING: Should be removed' if self_loops > 0 else 'Clean'})")
    
    # 3. Deduplication and Undirected Check
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    pairs = np.stack([lo, hi], axis=1)
    
    unique_pairs = np.unique(pairs, axis=0)
    print(f"Unique undirected pairs: {len(unique_pairs):,}")
    
    # Expected symmetrized count (excluding self-loops)
    expected_symmetrized = len(unique_pairs) * 2
    actual_symmetrized = len(src)
    
    print(f"Expected symmetrized count: {expected_symmetrized:,}")
    print(f"Actual symmetrized count: {actual_symmetrized:,}")
    
    if actual_symmetrized != expected_symmetrized:
        print("  → STATUS: Graph contains parallel/duplicate edges or is asymmetric.")
    else:
        print("  → STATUS: Graph is a clean, simple, symmetric undirected graph.")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Test with dummy data
    n_nodes = 5
    src = np.array([0, 1, 1, 2, 2, 0, 3, 3])
    dst = np.array([1, 0, 2, 1, 0, 2, 3, 4]) # Contains self-loop (3,3)
    validate_graph_properties(n_nodes, src, dst)

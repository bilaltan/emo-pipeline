import sys
import numpy as np
import torch
import dgl

def main():
    if len(sys.argv) < 4:
        print("Usage: metis_runner.py <edges_npz> <k> <partition_npy>")
        sys.exit(1)
        
    edges_npz = sys.argv[1]
    k = int(sys.argv[2])
    partition_npy = sys.argv[3]
    
    data = np.load(edges_npz)
    src = data['src']
    dst = data['dst']
    num_nodes = int(data['num_nodes'])
    
    g_dgl = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=num_nodes)
    partition_ids = dgl.metis_partition_assignment(g_dgl, k).numpy()
    np.save(partition_npy, partition_ids)

if __name__ == "__main__":
    main()

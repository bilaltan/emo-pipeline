import torch

class DownstreamNodeClassifier(torch.nn.Module):
    def __init__(self, input_dim, classes):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, classes)
        )
    def forward(self, x):
        return self.layers(x)

def run_downstream_classification(embed_np, labels_np, train_mask, val_mask, test_mask, num_classes, num_epochs=10, batch_size=128):
    import torch
    import copy
    import numpy as np
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import train_test_split
    
    input_dim = embed_np.shape[1]
    model = DownstreamNodeClassifier(input_dim, num_classes)
    
    embed_t = torch.tensor(embed_np, dtype=torch.float32)
    labels_t = torch.tensor(labels_np, dtype=torch.long)
    
    indices = np.arange(len(labels_np))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.2, shuffle=True, random_state=45)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, shuffle=True, random_state=45)
    
    if len(train_idx) == 0:
        return 0.0, 0, 0, np.zeros(len(labels_np), dtype=np.int64)
        
    train_embed = embed_t[train_idx]
    train_labels = labels_t[train_idx]
    
    dataset = TensorDataset(train_embed, train_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    best_acc = -1.0
    best_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        model.train()
        for x_b, y_b in loader:
            optimizer.zero_grad()
            y_pred = model(x_b)
            loss = loss_fn(y_pred, y_b)
            loss.backward()
            optimizer.step()
            
        if len(val_idx) > 0:
            model.eval()
            with torch.no_grad():
                val_embed = embed_t[val_idx]
                val_labels = labels_t[val_idx]
                y_pred_val = model(val_embed)
                acc = (y_pred_val.argmax(dim=1) == val_labels).float().mean().item()
                if acc > best_acc:
                    best_acc = acc
                    best_weights = copy.deepcopy(model.state_dict())
        else:
            best_weights = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        y_pred_all = model(embed_t)
        preds = y_pred_all.argmax(dim=1).cpu().numpy()
        
        if len(test_idx) > 0:
            test_embed = embed_t[test_idx]
            test_labels = labels_t[test_idx]
            y_pred_test = model(test_embed)
            correct = (y_pred_test.argmax(dim=1) == test_labels).sum().item()
            total = len(test_idx)
            test_acc = correct / total
        else:
            test_acc = 0.0
            correct = 0
            total = 0
            
    return test_acc, correct, total, preds

def dgl_to_pyg_data(g):
    """Convert a DGL graph `g` to a PyTorch Geometric `Data` object."""
    import torch
    from torch_geometric.data import Data
    
    src, dst = g.edges()
    edge_index = torch.stack([src, dst], dim=0)
    
    x = g.ndata.get('feat', None)
    y = g.ndata.get('label', None)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    for key in g.ndata:
        if 'mask' in key:
            setattr(data, key, g.ndata[key])
    return data

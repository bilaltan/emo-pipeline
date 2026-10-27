#!/usr/bin/env python3
"""
DistDGL step 2: GraphSAGE node classification on the partitions, one process per role (machine.py starts them).
With DGL_ROLE=server this process serves its partition and exits when the trainers finish; otherwise it is a
trainer: a DistGraph client plus one torch.distributed (gloo, CPU) rank.

Model, training and selection match the node baselines and Phase 3b (experiment_config.TRAIN): 2-layer mean SAGE
(DGL SAGEConv, the same lin_l(mean_neigh) + lin_r(x) parametrisation) + linear head, Adam, lr / weight decay /
dropout / hidden size / epochs from TRAIN, the best-validation checkpoint checked every max(1, epochs // 40)
epochs, no early stop. DistDGL trains on sampled mini-batches (its design); validation during training uses the
same sampler, and the chosen checkpoint is scored on validation and test with full neighbourhoods (layer-wise
inference over the whole graph), as the full-graph baseline scores.
"""
import argparse
import copy
import json
import os
import socket
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def build(f, h, c, dropout):
    import dgl.nn as dglnn
    import torch.nn as nn
    import torch.nn.functional as F

    class SAGE(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([dglnn.SAGEConv(f, h, "mean"), dglnn.SAGEConv(h, h, "mean")])
            self.fc, self.dr = nn.Linear(h, c), nn.Dropout(dropout)

        def forward(self, blocks, x):
            h = F.relu(self.layers[0](blocks[0], x))
            h = self.dr(h)
            return self.fc(self.layers[1](blocks[1], h))

    return SAGE()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip-config", required=True)
    ap.add_argument("--part-config", required=True)
    ap.add_argument("--graph-name", default="g")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fanout", default="10,25", help="DistDGL GraphSAGE example's default, input layer first")
    ap.add_argument("--batch-size", type=int, default=1000, help="per trainer (DistDGL example's default)")
    ap.add_argument("--eval-batch-size", type=int, default=10000)
    ap.add_argument("--out", default=None, help="rank 0 appends its JSON record here")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import dgl
    import dgl.distributed
    import torch
    import torch.nn.functional as F

    t_start = time.time()
    dgl.distributed.initialize(args.ip_config)   # a server never returns from here
    torch.distributed.init_process_group("gloo")
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    pb = g.get_partition_book()
    init_s = time.time() - t_start

    from experiment_config import TRAIN
    cfg = dict(TRAIN, epochs=args.epochs or TRAIN["epochs"], lr=args.lr or TRAIN["lr"])
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feat, label = g.ndata["feat"], g.ndata["label"]
    n_classes = int(json.load(open(os.path.join(os.path.dirname(args.part_config), "partition_meta.json")))["n_classes"])
    model = build(feat.shape[1], cfg["hidden_dim"], n_classes, cfg["dropout"])
    ddp = torch.nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.Adam(ddp.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # each trainer takes an equal share of every set, local to its machine where possible
    split = lambda m: dgl.distributed.node_split(g.ndata[m], pb, force_even=True)
    train_nid, val_nid, test_nid = split("train_mask"), split("val_mask"), split("test_mask")
    fanout = [int(f) for f in args.fanout.split(",")]
    sampler = dgl.dataloading.NeighborSampler(fanout)
    train_dl = dgl.distributed.DistNodeDataLoader(g, train_nid, sampler, batch_size=args.batch_size,
                                                  shuffle=True, drop_last=False)
    val_dl = dgl.distributed.DistNodeDataLoader(g, val_nid, sampler, batch_size=args.eval_batch_size,
                                                shuffle=False, drop_last=False)

    def allsum(*v):
        t = torch.tensor(v, dtype=torch.float64)
        torch.distributed.all_reduce(t)
        return t.tolist()

    def sampled_val():
        model.eval()
        right = seen = 0
        with torch.no_grad():
            for inp, seeds, blocks in val_dl:
                pred = model(blocks, feat[inp]).argmax(1)
                right += int((pred == label[seeds].long()).sum())
                seen += len(seeds)
        model.train()
        r, s = allsum(right, seen)
        return r / max(s, 1)

    val_every = max(1, cfg["epochs"] // 40)
    best, state, epoch_s, val_s = -1.0, None, [], 0.0
    torch.distributed.barrier()
    t_train = time.time()
    model.train()
    for ep in range(cfg["epochs"]):
        if ep > 0 and ep % val_every == 0:
            t_v = time.time()
            v = sampled_val()
            val_s += time.time() - t_v
            if v > best:
                best, state = v, copy.deepcopy(model.state_dict())
        t_ep = time.time()
        for inp, seeds, blocks in train_dl:
            loss = F.cross_entropy(ddp(blocks, feat[inp]), label[seeds].long())
            opt.zero_grad()
            loss.backward()
            opt.step()
        epoch_s.append(time.time() - t_ep)
        if rank == 0 and (ep % 10 == 0 or ep == cfg["epochs"] - 1):
            print(f"  epoch {ep:3d}  {epoch_s[-1]:.1f}s  loss {float(loss):.4f}  best val (sampled) {best:.4f}",
                  flush=True)
    t_v = time.time()
    v = sampled_val()
    val_s += time.time() - t_v
    if v > best:
        best, state = v, copy.deepcopy(model.state_dict())
    train_s = time.time() - t_train

    # the chosen checkpoint over every vertex with full neighbourhoods, one layer at a time (DistDGL's inference)
    t_inf = time.time()
    model.load_state_dict(state)
    model.eval()
    nodes = dgl.distributed.node_split(np.arange(g.num_nodes()), pb, force_even=True)
    full = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    x = feat
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            last = i == len(model.layers) - 1
            y = dgl.distributed.DistTensor((g.num_nodes(), n_classes if last else cfg["hidden_dim"]),
                                           torch.float32, f"h{i}_{args.tag}", persistent=True)
            dl = dgl.distributed.DistNodeDataLoader(g, nodes, full, batch_size=args.eval_batch_size,
                                                    shuffle=False, drop_last=False)
            for inp, out_nodes, blocks in dl:
                h = layer(blocks[0], x[inp])
                y[out_nodes] = model.fc(h) if last else F.relu(h)
            g.barrier()
            x = y
    acc = lambda nid: allsum(int((x[nid].argmax(1) == label[nid].long()).sum()), len(nid))
    rv, sv = acc(val_nid)
    rt, st = acc(test_nid)
    infer_s = time.time() - t_inf

    if rank == 0:
        rec = {"dataset": args.tag, "baseline": "distdgl_sage", "task": "node_classification", "seed": args.seed,
               "train": dict(cfg, seed=args.seed), "fanout": fanout, "batch_size_per_trainer": args.batch_size,
               "trainers": world, "machines": pb.num_partitions(),
               "omp_threads": int(os.environ.get("OMP_NUM_THREADS", "0") or 0),
               "val_acc_sampled": best, "val_acc": rv / max(sv, 1), "test_acc": rt / max(st, 1),
               "n_test": int(st), "init_seconds": round(init_s, 1), "train_seconds": round(train_s, 1),
               "val_seconds": round(val_s, 1), "epoch_seconds_mean": round(float(np.mean(epoch_s)), 2),
               "inference_seconds": round(infer_s, 1), "host": socket.gethostname(),
               "dgl": dgl.__version__, "torch": torch.__version__,
               "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        line = json.dumps(rec, default=lambda o: o.item())   # numpy scalars
        print("RESULT " + line, flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(line + "\n")
    torch.distributed.barrier()


if __name__ == "__main__":
    main()

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def read_ids(path):
    xs = []
    with open(path, "r") as f:
        for line in f:
            t = line.strip()
            if t:
                xs.append(t)
    return xs

def read_edges(path):
    es = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip().split("\t")
            if len(s) < 2:
                continue
            u, v = s[0], s[1]
            if u == v:
                continue
            es.append((u, v))
    return es

def l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

class PairMLP(nn.Module):
    def __init__(self, d, hidden, drop):
        super().__init__()
        self.fc1 = nn.Linear(d * 4, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        self.drop = drop

    def forward(self, eu, ev):
        x = torch.cat([eu, ev, torch.abs(eu - ev), eu * ev], dim=1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = self.out(x).squeeze(1)
        return x

def neg_sample_degree_matched(pos_edges, nodes, deg, n_neg, rng):
    nodes_arr = np.array(nodes, dtype=object)
    w = np.array([deg.get(x, 1) for x in nodes_arr], dtype=np.float64)
    w = w / (w.sum() + 1e-12)
    pos_set = set()
    for u, v in pos_edges:
        if u < v:
            pos_set.add((u, v))
        else:
            pos_set.add((v, u))
    neg = []
    tries = 0
    while len(neg) < n_neg and tries < n_neg * 50:
        u = rng.choice(nodes_arr, p=w)
        v = rng.choice(nodes_arr, p=w)
        tries += 1
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in pos_set:
            continue
        neg.append((a, b))
    return neg

def batched_logits(model, E, pairs, bs, device):
    us = [p[0] for p in pairs]
    vs = [p[1] for p in pairs]
    eu = E[us]
    ev = E[vs]
    out = []
    n = len(pairs)
    i = 0
    while i < n:
        j = min(n, i + bs)
        lu = torch.from_numpy(eu[i:j]).to(device)
        lv = torch.from_numpy(ev[i:j]).to(device)
        with torch.no_grad():
            z = model(lu, lv).detach().cpu().numpy()
        out.append(z)
        i = j
    return np.concatenate(out, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_npz", required=True)
    ap.add_argument("--train_pos", required=True)
    ap.add_argument("--val_pos", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--outdir", default="models")
    ap.add_argument("--graph_outdir", default="graphs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--neg_ratio", type=float, default=1.0)
    ap.add_argument("--cand_M", type=int, default=500)
    ap.add_argument("--topm", type=int, default=20)
    ap.add_argument("--score_to_prob", default="sigmoid")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.graph_outdir, exist_ok=True)

    rng = np.random.RandomState(args.seed)

    z = np.load(args.emb_npz, allow_pickle=True)
    ids = list(z["ids"])
    emb = np.array(z["emb"], dtype=np.float32)
    id2i = {k:i for i,k in enumerate(ids)}

    nodes = read_ids(args.nodes)
    nodes = [x for x in nodes if x in id2i]
    emb_nodes = l2norm(emb[[id2i[x] for x in nodes]])

    E = {nodes[i]: emb_nodes[i] for i in range(len(nodes))}

    tr_pos = read_edges(args.train_pos)
    va_pos = read_edges(args.val_pos)
    tr_pos = [(u, v) for (u, v) in tr_pos if u in E and v in E]
    va_pos = [(u, v) for (u, v) in va_pos if u in E and v in E]

    deg = {}
    for u, v in tr_pos:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1

    n_tr_neg = int(len(tr_pos) * args.neg_ratio)
    n_va_neg = int(len(va_pos) * args.neg_ratio)
    tr_neg = neg_sample_degree_matched(tr_pos, nodes, deg, n_tr_neg, rng)
    va_neg = neg_sample_degree_matched(va_pos, nodes, deg, n_va_neg, rng)

    def build_xy(pos, neg):
        pairs = pos + neg
        y = np.concatenate([np.ones(len(pos), dtype=np.float32), np.zeros(len(neg), dtype=np.float32)])
        idx = rng.permutation(len(pairs))
        pairs = [pairs[i] for i in idx]
        y = y[idx]
        return pairs, y

    tr_pairs, tr_y = build_xy(tr_pos, tr_neg)
    va_pairs, va_y = build_xy(va_pos, va_neg)

    d = emb_nodes.shape[1]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = PairMLP(d, args.hidden, args.drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    def iter_batches(pairs, y, bs):
        n = len(pairs)
        for i in range(0, n, bs):
            j = min(n, i + bs)
            ps = pairs[i:j]
            yy = torch.from_numpy(y[i:j]).to(device)
            eu = torch.from_numpy(np.stack([E[u] for u, _ in ps], axis=0)).to(device)
            ev = torch.from_numpy(np.stack([E[v] for _, v in ps], axis=0)).to(device)
            yield eu, ev, yy

    best = 1e18
    for ep in range(args.epochs):
        model.train()
        for eu, ev, yy in iter_batches(tr_pairs, tr_y, args.batch):
            opt.zero_grad(set_to_none=True)
            z = model(eu, ev)
            loss = F.binary_cross_entropy_with_logits(z, yy)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            losses = []
            for eu, ev, yy in iter_batches(va_pairs, va_y, args.batch):
                z = model(eu, ev)
                losses.append(F.binary_cross_entropy_with_logits(z, yy).item())
            vloss = float(np.mean(losses)) if losses else 1e18

        if vloss < best:
            best = vloss
            torch.save({"state_dict": model.state_dict(), "d": d}, os.path.join(args.outdir, "pair_mlp.pt"))

    ckpt = torch.load(os.path.join(args.outdir, "pair_mlp.pt"), map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    X = emb_nodes.astype(np.float32)
    G = X @ X.T
    np.fill_diagonal(G, -1.0)

    N = len(nodes)
    M = min(args.cand_M, N - 1)
    topm = min(args.topm, M)

    edges = []
    for i in range(N):
        idx = np.argpartition(-G[i], M)[:M]
        cand = [nodes[j] for j in idx]
        u = nodes[i]
        pairs = [(u, v) for v in cand]
        logits = batched_logits(model, E, pairs, bs=8192, device=device)

        if args.score_to_prob == "sigmoid":
            scores = 1.0 / (1.0 + np.exp(-logits))
        else:
            scores = logits

        sel = np.argpartition(-scores, topm)[:topm]
        for k in sel:
            v = cand[k]
            w = float(scores[k])
            edges.append((u, v, w))

    dct = {}
    for u, v, w in edges:
        a, b = (u, v) if u < v else (v, u)
        key = (a, b)
        if key not in dct or w > dct[key]:
            dct[key] = w

    out_path = os.path.join(args.graph_outdir, f"fmppi_mlp_M{M}_m{topm}.tsv")
    with open(out_path, "w") as f:
        for (u, v), w in sorted(dct.items()):
            f.write(u + "\t" + v + "\t" + f"{w:.6g}" + "\n")

    meta_path = os.path.join(args.graph_outdir, f"fmppi_mlp_M{M}_m{topm}.meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"emb_npz\t{args.emb_npz}\n")
        f.write(f"train_pos\t{args.train_pos}\n")
        f.write(f"val_pos\t{args.val_pos}\n")
        f.write(f"nodes\t{args.nodes}\n")
        f.write(f"hidden\t{args.hidden}\n")
        f.write(f"drop\t{args.drop}\n")
        f.write(f"lr\t{args.lr}\n")
        f.write(f"wd\t{args.wd}\n")
        f.write(f"epochs\t{args.epochs}\n")
        f.write(f"neg_ratio\t{args.neg_ratio}\n")
        f.write(f"cand_M\t{M}\n")
        f.write(f"topm\t{topm}\n")
        f.write(f"score_to_prob\t{args.score_to_prob}\n")
        f.write(f"val_bce\t{best}\n")

if __name__ == "__main__":
    main()

import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def read_ids(path):
    with open(path) as f:
        return [x.strip() for x in f if x.strip()]

def read_fasta(path):
    seqs = {}
    k, buf = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if k is not None:
                    seqs[k] = "".join(buf)
                k = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if k is not None:
            seqs[k] = "".join(buf)
    return seqs

def l2norm_rows(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

def chunk_seq(s, max_len):
    if len(s) <= max_len:
        return [s]
    out = []
    i = 0
    while i < len(s):
        out.append(s[i:i+max_len])
        i += max_len
    return out

@torch.no_grad()
def embed_seq(model, tok, seq, device, max_len, is_t5):
    parts = chunk_seq(seq, max_len)
    vecs = []
    for p in parts:
        if is_t5:
            p = " ".join(list(p))
        x = tok(p, return_tensors="pt", add_special_tokens=True)
        x = {k:v.to(device) for k,v in x.items()}
        out = model(**x)
        h = out.last_hidden_state[0]
        attn = x["attention_mask"][0].bool()
        h = h[attn]
        v = h.mean(dim=0)
        vecs.append(v)
    return torch.stack(vecs, dim=0).mean(dim=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--node_ids", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_len", type=int, default=1022)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--l2norm", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seqs = read_fasta(args.fasta)
    ids = read_ids(args.node_ids)

    tok = AutoTokenizer.from_pretrained(args.model, do_lower_case=False)
    model = AutoModel.from_pretrained(args.model)
    model.eval().to(device)

    is_t5 = "prot_t5" in args.model.lower() or "t5" in args.model.lower()

    embs = []
    miss = 0
    for nid in ids:
        s = seqs.get(nid)
        if s is None:
            miss += 1
            embs.append(None)
            continue
        v = embed_seq(model, tok, s, device, args.max_len, is_t5)
        if args.fp16:
            v = v.half()
        embs.append(v.detach().cpu())

    if miss:
        raise RuntimeError(f"missing sequences: {miss}/{len(ids)}")

    E = torch.stack(embs, dim=0).numpy()
    if args.l2norm:
        E = l2norm_rows(E.astype(np.float32)).astype(E.dtype, copy=False)
    np.save(args.out, E)

if __name__ == "__main__":
    main()

import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from tqdm import tqdm

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

def slide_seq(s, win, stride):
    if len(s) <= win:
        return [s]
    out, i = [], 0
    while True:
        out.append(s[i:i+win])
        if i + win >= len(s):
            break
        i += stride
    return out

@torch.no_grad()
def embed_windows_esm2(model, tok, parts, device, max_len, bs, use_amp):
    acc = None
    n = 0
    for i in range(0, len(parts), bs):
        b = parts[i:i+bs]
        x = tok(
            b,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
        )
        x = {k: v.to(device) for k, v in x.items()}

        if use_amp and device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                h = model(**x).last_hidden_state
        else:
            h = model(**x).last_hidden_state

        attn = x["attention_mask"].bool()

        keep = attn.clone()
        keep[:, 0] = False
        lens = attn.long().sum(dim=1)
        last = torch.clamp(lens - 1, min=0)
        keep[torch.arange(keep.size(0), device=keep.device), last] = False

        m = keep.unsqueeze(-1).to(h.dtype)
        denom = m.sum(dim=1).clamp_min(1.0)
        v = (h * m).sum(dim=1) / denom

        s = v.sum(dim=0)
        acc = s if acc is None else (acc + s)
        n += v.size(0)

        del h, attn, keep, m, denom, v, s, x
        if device == "cuda":
            torch.cuda.empty_cache()

    return acc / max(n, 1)

@torch.no_grad()
def embed_windows_prott5(model, tok, parts, device, max_len, bs, use_amp):
    acc = None
    n = 0
    for i in range(0, len(parts), bs):
        b = parts[i:i+bs]
        b = [" ".join(list(x)) for x in b]

        x = tok(
            b,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
        )
        x = {k: v.to(device) for k, v in x.items()}

        if use_amp and device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                h = model(**x).last_hidden_state
        else:
            h = model(**x).last_hidden_state

        attn = x["attention_mask"].bool()
        m = attn.unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

        s = v.sum(dim=0)
        acc = s if acc is None else (acc + s)
        n += v.size(0)

        del h, attn, m, v, s, x
        if device == "cuda":
            torch.cuda.empty_cache()

    return acc / max(n, 1)

def build_model(backend, model_name, device):
    if backend == "esm2":
        tok = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        model = AutoModel.from_pretrained(model_name).eval().to(device)
        D = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "d_model", 0))
        if D <= 0:
            raise RuntimeError("cannot infer embedding dim from model config")
        return tok, model, D
    if backend == "prott5":
        tok = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, use_fast=False)
        m = AutoModel.from_pretrained(model_name).eval().to(device)
        model = m.get_encoder()
        D = int(getattr(model.config, "d_model", 0) or getattr(model.config, "hidden_size", 0))
        if D <= 0:
            raise RuntimeError("cannot infer embedding dim from model config")
        return tok, model, D
    raise RuntimeError("backend must be esm2 or prott5")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["esm2", "prott5"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--node_ids", required=True)
    ap.add_argument("--out", required=True)  # npz
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--stride", type=int, default=768)
    ap.add_argument("--bucket", type=int, default=128)
    ap.add_argument("--win_bs", type=int, default=16)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--l2norm", action="store_true")
    ap.add_argument("--cpu_threads", type=int, default=4)
    ap.add_argument("--keep_npy", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(max(1, args.cpu_threads))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(args.fp16 and device == "cuda")

    if args.max_len is None:
        args.max_len = 1022 if args.backend == "esm2" else 1024

    fasta_path = Path(args.fasta)
    ids_path = Path(args.node_ids)
    out_npz = Path(args.out)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    tmp_npy = out_npz.with_suffix(".npy")

    seqs = read_fasta(fasta_path)
    ids = read_ids(ids_path)

    tok, model, D = build_model(args.backend, args.model, device)

    dtype = np.float16 if (args.fp16 and device == "cuda") else np.float32
    mm = np.lib.format.open_memmap(tmp_npy, mode="w+", dtype=dtype, shape=(len(ids), D))

    items = []
    for nid in ids:
        s = seqs.get(nid)
        if s is None:
            raise RuntimeError(f"missing sequence: {nid}")
        items.append((nid, s, len(s)))
    items.sort(key=lambda x: x[2])

    buckets = {}
    for nid, s, L in items:
        k = (L // args.bucket) * args.bucket
        buckets.setdefault(k, []).append((nid, s))

    idx_map = {nid: i for i, nid in enumerate(ids)}
    pbar = tqdm(total=len(ids), desc=f"Embedding ({args.backend})")

    for k in sorted(buckets.keys()):
        for nid, s in buckets[k]:
            parts = slide_seq(s, args.max_len, args.stride)
            if args.backend == "esm2":
                v = embed_windows_esm2(model, tok, parts, device, args.max_len, args.win_bs, use_amp)
            else:
                v = embed_windows_prott5(model, tok, parts, device, args.max_len, args.win_bs, use_amp)

            i = idx_map[nid]
            if dtype == np.float16:
                mm[i, :] = v.float().cpu().numpy().astype(np.float16, copy=False)
            else:
                mm[i, :] = v.float().cpu().numpy()

            if (pbar.n + 1) % 256 == 0:
                mm.flush()
            pbar.update(1)

    pbar.close()
    mm.flush()

    if args.l2norm:
        mm2 = np.lib.format.open_memmap(tmp_npy, mode="r+", dtype=dtype, shape=(len(ids), D))
        bs = 4096
        for i in range(0, len(ids), bs):
            x = np.array(mm2[i:i+bs], dtype=np.float32, copy=True)
            x = l2norm_rows(x)
            if dtype == np.float16:
                mm2[i:i+bs] = x.astype(np.float16, copy=False)
            else:
                mm2[i:i+bs] = x
        mm2.flush()

    emb = np.load(tmp_npy, mmap_mode=None)
    np.savez(out_npz, ids=np.array(ids, dtype=object), emb=emb.astype(np.float32))

    if not args.keep_npy:
        try:
            tmp_npy.unlink()
        except Exception:
            pass

if __name__ == "__main__":
    main()
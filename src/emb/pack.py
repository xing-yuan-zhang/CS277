import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--emb_npy", required=True)
ap.add_argument("--ids_txt", required=True)
ap.add_argument("--out_npz", required=True)
args = ap.parse_args()

emb = np.load(args.emb_npy)
ids = [x.strip() for x in open(args.ids_txt) if x.strip()]
if emb.shape[0] != len(ids):
    raise SystemExit(f"mismatch: emb: {emb.shape[0]} vs ids: {len(ids)}")

np.savez(args.out_npz, ids=np.array(ids, dtype=object), emb=emb.astype(np.float32))
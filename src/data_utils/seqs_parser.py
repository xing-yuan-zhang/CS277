import argparse
from pathlib import Path
import pandas as pd

def read_fa(p):
    m, k, buf = {}, None, []
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line[0] == ">":
                if k and buf:
                    m[k] = "".join(buf)
                k = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if k and buf:
            m[k] = "".join(buf)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--info", default="string_info.tsv")
    ap.add_argument("--seqs", default="9606.protein.sequences.v12.0.fa")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()

    base = Path(__file__).resolve().parent
    P = lambda x: (base / x) if x and not Path(x).is_absolute() else Path(x)
    outdir = P(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    info = pd.read_csv(P(a.info), sep="\t", dtype=str)
    if "string_id" not in info.columns:
        info = info.rename(columns={info.columns[0]: "string_id"})
    sids = info["string_id"].astype(str).tolist()

    seqm = read_fa(P(a.seqs))

    keep = []
    for sid in sids:
        seq = seqm.get(sid)
        if not seq:
            continue
        keep.append((sid, seq))
    
    '''
    with open(outdir / "node_ids.txt", "w", encoding="utf-8") as f:
        for sid, _ in keep:
            f.write(sid + "\n")
    '''
            
    with open(outdir / "seqs.fasta", "w", encoding="utf-8") as f:
        for sid, seq in keep:
            f.write(">" + sid + "\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")

if __name__ == "__main__":
    main()

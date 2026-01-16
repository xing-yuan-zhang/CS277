# src/data_utils/uniprot_mirror.py
import gzip
from pathlib import Path

def iter_sec2pri_from_dat_gz(dat_gz: Path):
    primary = None
    secs = []
    with gzip.open(dat_gz, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("ID   "):
                primary = None
                secs = []
            elif line.startswith("AC   "):
                acs = line[5:].strip()
                for token in acs.split(";"):
                    acc = token.strip()
                    if acc:
                        secs.append(acc)
            elif line.startswith("//"):
                if secs:
                    primary = secs[0]
                    for s in secs[1:]:
                        yield s, primary
                primary = None
                secs = []

def main():
    root = Path(__file__).resolve().parents[2]
    raw = root / "inputs" / "ppi" / "UniProt" / "raw"
    out = root / "inputs" / "ppi" / "UniProt" / "uniprot_sec2pri_9606.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)

    files = [
        raw / "uniprot_sprot_human.dat.gz",
        raw / "uniprot_trembl_human.dat.gz",
    ]
    for p in files:
        if not p.exists():
            raise FileNotFoundError(f"missing: {p}")

    seen = set()
    with open(out, "w", encoding="utf-8") as w:
        w.write("secondary\tprimary\n")
        for p in files:
            for s, pri in iter_sec2pri_from_dat_gz(p):
                key = (s, pri)
                if key in seen:
                    continue
                seen.add(key)
                w.write(f"{s}\t{pri}\n")

    print(f"wrote {out} rows={len(seen):,}")

if __name__ == "__main__":
    main()

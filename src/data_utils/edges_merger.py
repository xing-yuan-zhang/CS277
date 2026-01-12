# src/data_utils/edges_merger.py
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

OUTDIR = ROOT / "inputs/ppi"
OUTDIR.mkdir(parents=True, exist_ok=True)

STRING_EDGES = OUTDIR / "STRING/v12.0/string_edges.tsv"
STRING_ALIASES = OUTDIR / "STRING/v12.0/string_aliases.tsv"
BIOGRID_EDGES = OUTDIR / "BioGRID/biogrid_human_edges.tsv"

OUT_MERGED = OUTDIR / "edges.merged.uniprot.tsv"
OUT_MAP = OUTDIR / "string2uniprot.tsv"

UNIPROT_RE = re.compile(
    r"""
    \b(
        # UniProt accession
        [A-NR-Z][0-9][A-Z0-9]{3}[0-9]
      | [OPQ][0-9][A-Z0-9]{3}[0-9]
        # TrEMBL accession
      | A0A[0-9A-Z]{7}
    )\b
    """,
    re.VERBOSE
)

def looks_like_uniprot(x: str) -> bool:
    return bool(UNIPROT_RE.fullmatch(x))

def build_string_to_uniprot(aliases: pd.DataFrame) -> pd.DataFrame:
    if not {"string_id", "alias", "source"}.issubset(set(aliases.columns)):
        cols = list(aliases.columns)
        aliases = aliases.rename(columns={cols[0]: "string_id", cols[1]: "alias", cols[2]: "source"})

    ali = aliases.copy()
    ali["alias"] = ali["alias"].astype(str).str.strip()

    ali = ali[ali["alias"].map(looks_like_uniprot)].copy()

    def pri(s: str) -> int:
        s = str(s).lower()
        if "uniprot" in s:
            return 0
        if "ensembl" in s:
            return 1
        return 2

    ali["priority"] = ali["source"].map(pri)
    ali = ali.sort_values(["string_id", "priority"])

    best = ali.drop_duplicates(subset=["string_id"], keep="first")[["string_id", "alias", "source"]]
    best = best.rename(columns={"alias": "entry", "source": "source_hint"})
    return best

def extract_uniprot_from_biogrid_field(s: str) -> str | None:
    if s is None:
        return None
    s = str(s)
    parts = s.split("|")
    for p in parts:
        p = p.strip()
        if ":" in p:
            token = p.split(":", 1)[1]
        else:
            token = p
        token = token.strip()
        if looks_like_uniprot(token):
            return token

    m = UNIPROT_RE.search(s)
    return m.group(1) if m else None

def main():
    aliases = pd.read_csv(STRING_ALIASES, sep="\t")
    s2u = build_string_to_uniprot(aliases)
    s2u.to_csv(OUT_MAP, sep="\t", index=False)
    s2u_map = dict(zip(s2u["string_id"], s2u["entry"]))

    se = pd.read_csv(STRING_EDGES, sep="\t")
    if not {"node_a", "node_b"}.issubset(set(se.columns)):
        raise ValueError("string_edges.tsv need node_a/node_b columns")

    se["entry_a"] = se["node_a"].map(s2u_map)
    se["entry_b"] = se["node_b"].map(s2u_map)
    se = se.dropna(subset=["entry_a", "entry_b"]).copy()
    se = se[se["entry_a"] != se["entry_b"]].copy()

    if "combined_score" not in se.columns:
        raise ValueError("string_edges.tsv need combined_score column")

    se["string_score"] = pd.to_numeric(se["combined_score"], errors="coerce")
    se = se.dropna(subset=["string_score"]).copy()
    se["weight"] = se["string_score"] / 1000.0
    se["is_string"] = 1
    se["is_biogrid"] = 0

    keep_cols = ["entry_a", "entry_b", "weight", "string_score", "is_string", "is_biogrid"]
    se = se[keep_cols]

    if BIOGRID_EDGES.exists():
        bg = pd.read_csv(BIOGRID_EDGES, sep="\t")
        if not {"a", "b"}.issubset(set(bg.columns)):
            bg = bg.rename(columns={bg.columns[0]: "a", bg.columns[1]: "b"})

        bg["entry_a"] = bg["a"].map(extract_uniprot_from_biogrid_field)
        bg["entry_b"] = bg["b"].map(extract_uniprot_from_biogrid_field)
        bg = bg.dropna(subset=["entry_a", "entry_b"]).copy()
        bg = bg[bg["entry_a"] != bg["entry_b"]].copy()
        bg["weight"] = 1.0
        bg["string_score"] = pd.NA
        bg["is_string"] = 0
        bg["is_biogrid"] = 1
        bg = bg[keep_cols]
        merged = pd.concat([se, bg], ignore_index=True)
    else:
        merged = se

    # undirected dedup: (min, max)
    a = merged["entry_a"].astype(str)
    b = merged["entry_b"].astype(str)
    merged["u"] = a.where(a < b, b)
    merged["v"] = b.where(a < b, a)

    def max_num(x):
        x = pd.to_numeric(x, errors="coerce")
        return x.max()

    agg = merged.groupby(["u", "v"], as_index=False).agg(
        weight=("weight", "max"),
        string_score=("string_score", max_num),
        is_string=("is_string", "max"),
        is_biogrid=("is_biogrid", "max"),
    )

    agg = agg.rename(columns={"u": "entry_a", "v": "entry_b"})
    agg.to_csv(OUT_MERGED, sep="\t", index=False)
    print(f"[OK] wrote {OUT_MERGED} edges={len(agg):,}")
    print(f"[OK] wrote {OUT_MAP} mappings={len(s2u):,}")

if __name__ == "__main__":
    main()

# src/graph_constructor/merge_motifs_domains_llps.py
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

NODES = ROOT / "inputs/ppi/subgraph/subgraph_nodes.tsv"
ATTR  = ROOT / "inputs/annotations/nodes.attributes.tsv"
ELM   = ROOT / "inputs/annotations/motifs/ELM/elm_instances_human.tsv"
OUT   = ROOT / "inputs/ppi/subgraph/subgraph_node_attributes.tsv"
OUT.parent.mkdir(parents=True, exist_ok=True)

UNIPROT_RE = re.compile(
    r"""
    \b(
        [A-NR-Z][0-9][A-Z0-9]{3}[0-9]
      | [OPQ][0-9][A-Z0-9]{3}[0-9]
      | A0A[0-9A-Z]{7}
    )\b
    """,
    re.VERBOSE,
)

def infer_uniprot_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        cl = c.lower()
        if cl in ["entry", "uniprot", "uniprot_id", "uniprotkb", "primary_accession", "acc"]:
            return c

    for c in df.columns:
        s = df[c].astype(str)
        hit = s.map(lambda x: bool(UNIPROT_RE.match(x))).mean()
        if hit > 0.3:
            return c
    raise ValueError("Cannot infer UniProt column in ELM data.")

def read_elm(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        comment="#",
        engine="python",
        on_bad_lines="skip",
    )

def main():
    nodes = pd.read_csv(NODES, sep="\t")
    attr = pd.read_csv(ATTR, sep="\t")

    nodes["entry"] = nodes["entry"].astype(str).str.strip()
    attr["entry"] = attr["entry"].astype(str).str.strip()

    out = nodes.merge(attr, on="entry", how="left")

    out["elm_total"] = 0
    out["elm_sh3_related"] = 0

    if ELM.exists():
        try:
            elm = read_elm(ELM)

            if "Primary_Acc" in elm.columns:
                ucol = "Primary_Acc"
            else:
                ucol = infer_uniprot_col(elm)

            elm[ucol] = elm[ucol].astype(str).str.strip()

            total = (
                elm.groupby(ucol)
                .size()
                .rename("elm_total")
                .reset_index()
                .rename(columns={ucol: "entry"})
            )

            id_col = None
            for c in elm.columns:
                cl = c.lower()
                if "elm" in cl and "id" in cl:
                    id_col = c
                    break
            if id_col is None:
                for c in elm.columns:
                    if c.lower() in ["elmid", "elmididentifier", "elmidentifier"]:
                        id_col = c
                        break

            out = out.merge(total, on="entry", how="left")

            if id_col is not None:
                sh3 = elm[elm[id_col].astype(str).str.contains("SH3", case=False, na=False)]
                sh3_cnt = (
                    sh3.groupby(ucol)
                    .size()
                    .rename("elm_sh3_related")
                    .reset_index()
                    .rename(columns={ucol: "entry"})
                )
                out = out.merge(sh3_cnt, on="entry", how="left")

        except Exception as e:
            print(f"[WARN] ELM parsing/merge failed: {ELM} ({e}). Continue without ELM features.")

        if "elm_total" not in out.columns:
            out["elm_total"] = 0
        if "elm_sh3_related" not in out.columns:
            out["elm_sh3_related"] = 0

        out["elm_total"] = pd.to_numeric(out["elm_total"], errors="coerce").fillna(0).astype(int)
        out["elm_sh3_related"] = pd.to_numeric(out["elm_sh3_related"], errors="coerce").fillna(0).astype(int)


    out.to_csv(OUT, sep="\t", index=False)
    print(f"[OK] wrote {OUT} rows={len(out):,}")

if __name__ == "__main__":
    main()

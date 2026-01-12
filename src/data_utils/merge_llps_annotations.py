# src/data_utils/merge_llps_annotations.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INTERPRO = ROOT / "inputs/annotations/domains/interproscan_summary.tsv"
LLPSDB_INFILE = ROOT / "inputs/annotations/LLPS/LLPSDB/unambiguous/Phase_separation/protein.xls"
LLPSDB_OUTFILE = ROOT / "inputs/annotations/LLPS/LLPSDB/unambiguous/Phase_separation/llpsdb_positive.tsv"
PHASEPDB_INFILE = ROOT / "inputs/annotations/LLPS/PhaSepDB/PhaSepDB_human.tsv"
PHASEPDB_OUTFILE = ROOT / "inputs/annotations/LLPS/PhaSepDB/phasepdb_positive.tsv"
NODES_OUTFILE = ROOT / "inputs/annotation/nodes.attributes.tsv"

def parse_llpsdb(infile: Path = LLPSDB_INFILE, outfile: Path = LLPSDB_OUTFILE) -> Path:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(infile)

    gene_col = None
    uniprot_col = None

    for c in df.columns:
        if "gene" in c.lower():
            gene_col = c
        if "uniprot" in c.lower():
            uniprot_col = c

    if gene_col is None and uniprot_col is None:
        raise ValueError("No UniProt or Gene entries in LLPSDB protein.xls")

    out = pd.DataFrame()
    out["entry"] = df[uniprot_col] if uniprot_col else None
    out["gene_symbol"] = df[gene_col] if gene_col else None
    out["is_LLPSDB"] = 1

    out = out.dropna(how="all").drop_duplicates()
    out.to_csv(outfile, sep="\t", index=False)
    print(f"[OK] LLPSDB positives: {len(out)}")
    return outfile

def parse_phasepdb(infile: Path = PHASEPDB_INFILE, outfile: Path = PHASEPDB_OUTFILE) -> Path:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(infile, sep="\t")

    entry_col = None
    gene_col = None

    for c in df.columns:
        if c.lower() in ["entry", "uniprot", "uniprot_id", "uniprotkb"]:
            entry_col = c
        if "gene" in c.lower():
            gene_col = c

    if entry_col is None and gene_col is None:
        raise ValueError("No UniProt or Gene entries in PhaSepDB")

    out = pd.DataFrame()
    out["entry"] = df[entry_col] if entry_col else None
    out["gene_symbol"] = df[gene_col] if gene_col else None
    out["is_PhaSepDB"] = 1

    out = out.drop_duplicates()
    out.to_csv(outfile, sep="\t", index=False)
    print(f"[OK] PhaSepDB positives: {len(out)}")
    return outfile

def merge_llps_annotations(
    interpro_file: Path = INTERPRO,
    llpsdb_file: Path = LLPSDB_OUTFILE,
    phasepdb_file: Path = PHASEPDB_OUTFILE,
    outfile: Path = NODES_OUTFILE,
) -> Path:
    outfile.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(interpro_file, sep="\t")

    df = df.rename(columns={
        "Entry": "entry",
        "Gene Names (synonym)": "gene_symbol",
        "InterPro": "interpro_list",
        "Pfam": "pfam_list",
        "Length": "length",
        "Subcellular location [CC]": "subcellular_location",
    })

    df["gene_symbol"] = df["gene_symbol"].fillna("").str.split().str[0]

    df["has_SH3"] = df["interpro_list"].str.contains("SH3", case=False, na=False)
    df["has_PRD"] = df["pfam_list"].str.contains("PRM|PxxP", case=False, na=False)

    df = df[
        [
            "entry",
            "gene_symbol",
            "interpro_list",
            "pfam_list",
            "length",
            "subcellular_location",
            "has_SH3",
            "has_PRD",
        ]
    ]

    llps = None
    if llpsdb_file.exists():
        llps = pd.read_csv(llpsdb_file, sep="\t")
        df = df.merge(llps[["entry", "is_LLPSDB"]], on="entry", how="left")
    else:
        df["is_LLPSDB"] = 0

    if phasepdb_file.exists():
        phase = pd.read_csv(phasepdb_file, sep="\t")
        print("df.entry dtype:", df["entry"].dtype)
        if llps is not None:
            print("llps.entry dtype:", llps["entry"].dtype)
            print("llps entry head:", llps["entry"].head().tolist())
        print("phase.entry dtype:", phase["entry"].dtype)
        df = df.merge(phase[["entry", "is_PhaSepDB"]], on="entry", how="left")
    else:
        df["is_PhaSepDB"] = 0

    df["is_LLPSDB"] = df["is_LLPSDB"].fillna(0).astype(int)
    df["is_PhaSepDB"] = df["is_PhaSepDB"].fillna(0).astype(int)
    df["is_LLPS_any"] = ((df["is_LLPSDB"] + df["is_PhaSepDB"]) > 0).astype(int)
    df = df.drop_duplicates(subset=["entry"])

    df.to_csv(outfile, sep="\t", index=False)
    print(f"[OK] Final nodes.attributes.tsv: {len(df)} nodes")
    return outfile

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["all", "llpsdb", "phasepdb", "merge"], default="all")
    ap.add_argument("--llpsdb_in", default=str(LLPSDB_INFILE))
    ap.add_argument("--llpsdb_out", default=str(LLPSDB_OUTFILE))
    ap.add_argument("--phasepdb_in", default=str(PHASEPDB_INFILE))
    ap.add_argument("--phasepdb_out", default=str(PHASEPDB_OUTFILE))
    ap.add_argument("--interpro", default=str(INTERPRO))
    ap.add_argument("--nodes_out", default=str(NODES_OUTFILE))
    args = ap.parse_args()

    llpsdb_in = Path(args.llpsdb_in)
    llpsdb_out = Path(args.llpsdb_out)
    phasepdb_in = Path(args.phasepdb_in)
    phasepdb_out = Path(args.phasepdb_out)
    interpro = Path(args.interpro)
    nodes_out = Path(args.nodes_out)

    if args.run in ["all", "llpsdb"]:
        parse_llpsdb(llpsdb_in, llpsdb_out)
    if args.run in ["all", "phasepdb"]:
        parse_phasepdb(phasepdb_in, phasepdb_out)
    if args.run in ["all", "merge"]:
        merge_llps_annotations(interpro, llpsdb_out, phasepdb_out, nodes_out)

if __name__ == "__main__":
    main()

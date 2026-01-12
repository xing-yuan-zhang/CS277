# src/data_utils/preprocess_string_biogrid.py
import argparse
import gzip
import os

from pathlib import Path

import pandas as pd

def read_gz_tsv(path, sep="\t", **kwargs):
    import gzip
    import pandas as pd

    default_kwargs = dict(sep=sep, dtype=str, low_memory=False, on_bad_lines="skip")
    default_kwargs.update(kwargs)

    with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        return pd.read_csv(f, **default_kwargs)


def read_gz_space(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt") as f:
        return pd.read_csv(f, sep=" ")


def preprocess_string(root: Path, string_version: str, taxid: str, outdir: Path) -> None:
    string_dir = root / "inputs" / "ppi" / "STRING" / string_version
    detailed = string_dir / f"{taxid}.protein.links.detailed.{string_version}.txt.gz"
    aliases = string_dir / f"{taxid}.protein.aliases.{string_version}.txt.gz"
    info = string_dir / f"{taxid}.protein.info.{string_version}.txt.gz"

    outdir.mkdir(parents=True, exist_ok=True)

    # edges
    df = read_gz_space(detailed)
    p1, p2 = df.columns[0], df.columns[1]
    if "combined_score" not in df.columns:
        raise ValueError(f"combined_score not in columns: {list(df.columns)}")

    keep_cols = [p1, p2, "combined_score"]
    for c in ["neighborhood", "fusion", "cooccurence", "coexpression", "experimental", "database", "textmining"]:
        if c in df.columns:
            keep_cols.append(c)

    edges = df[keep_cols].copy()
    edges.columns = ["node_a", "node_b"] + [c for c in edges.columns[2:]]
    edges.to_csv(outdir / "string_edges.tsv", sep="\t", index=False)

    # aliases id map
    ali = read_gz_tsv(aliases, sep="\t")
    ali = ali.rename(columns={ali.columns[0]: "string_id", ali.columns[1]: "alias", ali.columns[2]: "source"})
    ali.to_csv(outdir / "string_aliases.tsv", sep="\t", index=False)

    # info
    inf = read_gz_tsv(info, sep="\t")
    inf = inf.rename(columns={inf.columns[0]: "string_id"})
    inf.to_csv(outdir / "string_info.tsv", sep="\t", index=False)

    print(f"[OK] STRING: edges={len(edges):,}, aliases={len(ali):,}, info={len(inf):,}")


def preprocess_biogrid(root: Path, outdir: Path) -> None:
    biogrid_zip = root / "inputs" / "ppi" / "BioGRID" / "BIOGRID-ALL-LATEST.mitab.zip"
    if not biogrid_zip.exists():
        print("[skip] BioGRID not found:", biogrid_zip)
        return

    import zipfile
    with zipfile.ZipFile(biogrid_zip, "r") as z:
        names = z.namelist()
        txts = [n for n in names if n.endswith(".txt")]
        if not txts:
            raise ValueError("No .txt found in BioGRID mitab zip.")
        mitab_name = txts[0]
        with z.open(mitab_name) as f:
            df = pd.read_csv(f, sep="\t", header=0, low_memory=False)

    col_alt_a = "Alt IDs Interactor A"
    col_alt_b = "Alt IDs Interactor B"
    col_tax_a = "Taxid Interactor A"
    col_tax_b = "Taxid Interactor B"

    df.columns = [str(c).strip() for c in df.columns]

    col_alt_a = next((c for c in df.columns if c == "Alt IDs Interactor A"), None)
    col_alt_b = next((c for c in df.columns if c == "Alt IDs Interactor B"), None)
    col_tax_a = next((c for c in df.columns if c == "Taxid Interactor A"), None)
    col_tax_b = next((c for c in df.columns if c == "Taxid Interactor B"), None)

    if not all([col_alt_a, col_alt_b, col_tax_a, col_tax_b]):
        raise ValueError(
            "MITAB header columns not found as expected. "
            f"Found columns: {list(df.columns)[:15]} ..."
        )

    tax_a = df[col_tax_a].astype(str)
    tax_b = df[col_tax_b].astype(str)
    is_human = tax_a.str.contains("taxid:9606", na=False) & tax_b.str.contains("taxid:9606", na=False)

    a_alt = df[col_alt_a].astype(str)
    b_alt = df[col_alt_b].astype(str)

    def normalize_uniprot_prefix(s: str) -> str:
        return (
            s.replace("uniprot/swiss-prot:", "uniprotkb:")
             .replace("uniprot/trembl:", "uniprotkb:")
        )

    a = a_alt.map(normalize_uniprot_prefix)
    b = b_alt.map(normalize_uniprot_prefix)

    outdir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"a": a[is_human], "b": b[is_human]}).dropna()

    out = out[out["a"].str.contains("uniprotkb:", na=False) & out["b"].str.contains("uniprotkb:", na=False)]

    out.to_csv(outdir / "biogrid_human_edges.tsv", sep="\t", index=False)
    print(f"[OK] BioGRID human edges={len(out):,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--string-version", default="v12.0")
    ap.add_argument("--taxid", default="9606")
    ap.add_argument("--with-biogrid", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    outdir_string = root / "inputs" / "ppi" / "STRING" / "v12.0"
    outdir_string.mkdir(parents=True, exist_ok=True)
    outdir_biogrid = root / "inputs" / "ppi" / "BioGRID"
    outdir_biogrid.mkdir(parents=True, exist_ok=True)

    preprocess_string(root, args.string_version, args.taxid, outdir_string)

    preprocess_biogrid(root, outdir_biogrid)

    print("[done]")


if __name__ == "__main__":
    main()

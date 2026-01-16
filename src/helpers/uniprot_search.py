# src/helpers/uniprot_search.py
import time
import pandas as pd
import requests
from io import StringIO
from pathlib import Path

API = "https://rest.uniprot.org"


def clean_node_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.split("-").str[0]
        .str.split(".").str[0]
    )


def submit_mapping_job(acc_list):
    r = requests.post(
        f"{API}/idmapping/run",
        data={"from": "UniProtKB_AC-ID", "to": "UniProtKB", "ids": ",".join(acc_list)},
        timeout=60
    )
    r.raise_for_status()
    return r.json()["jobId"]


def wait_for_job(job_id, sleep_s=1.5):
    while True:
        s = requests.get(f"{API}/idmapping/status/{job_id}", timeout=60)
        s.raise_for_status()
        js = s.json()
        if js.get("jobStatus") in ("NEW", "RUNNING"):
            time.sleep(sleep_s)
            continue
        return js


def fetch_all_results_tsv(job_id, fields="accession,gene_primary,protein_name"):
    url = f"{API}/idmapping/uniprotkb/results/{job_id}"
    params = {"format": "tsv", "fields": fields}

    chunks = []
    first_page = True

    while url:
        r = requests.get(url, params=params if first_page else None, timeout=120)
        r.raise_for_status()

        text = r.text
        if first_page:
            chunks.append(text)
            first_page = False
        else:
            lines = text.splitlines()
            if len(lines) > 1:
                chunks.append("\n".join(lines[1:]) + "\n")

        url = r.links.get("next", {}).get("url")

    return "".join(chunks)



def parse_mapping_tsv(tsv_text: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(tsv_text), sep="\t")

    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("entry", "accession"):
            rename[c] = "accession"
        elif "gene names" in lc and "primary" in lc:
            rename[c] = "gene_symbol"
        elif "protein" in lc and "name" in lc:
            rename[c] = "protein_name"

    df = df.rename(columns=rename)

    keep = [c for c in ["accession", "gene_symbol", "protein_name"] if c in df.columns]
    df = df[keep].drop_duplicates("accession")

    return df


def fetch_uniprot_names(accessions, batch_size=500) -> pd.DataFrame:
    acc_raw = pd.Series(accessions)
    acc_clean = clean_node_series(acc_raw)

    acc_list = []
    seen = set()
    for x in acc_clean.tolist():
        if not x or x.lower() == "nan":
            continue
        if x not in seen:
            acc_list.append(x)
            seen.add(x)

    rows = []
    for i in range(0, len(acc_list), batch_size):
        batch = acc_list[i:i + batch_size]

        job_id = submit_mapping_job(batch)
        wait_for_job(job_id)

        tsv = fetch_all_results_tsv(job_id, fields="accession,gene_primary,protein_name")
        df_map = parse_mapping_tsv(tsv)
        rows.append(df_map)

    if not rows:
        return pd.DataFrame(columns=["accession", "gene_symbol", "protein_name"])

    out = pd.concat(rows, ignore_index=True).drop_duplicates("accession")
    return out


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]

    inp_path = ROOT / "outputs/diffusion/diffusion_scores_alpha0.85.csv"
    df = pd.read_csv(inp_path)

    df = df.copy()
    df["node_clean"] = clean_node_series(df["node"])

    mapping = fetch_uniprot_names(df["node_clean"])

    df = df.merge(
        mapping[["accession", "gene_symbol"]],
        left_on="node_clean",
        right_on="accession",
        how="left"
    )

    df["node"] = df["gene_symbol"].fillna(df["node"])

    df = df.sort_values("score", ascending=False)

    out = df[["node", "score", "degree", "weighted_degree"]]

    out_path = ROOT / "outputs/diffusion/diffusion_scores_alpha0.85_gene.tsv"
    out.to_csv(out_path, sep="\t", index=False)

    mapped_n = df["gene_symbol"].notna().sum()
    print(f"Wrote: {out_path}")
    print(f"Mapped to gene symbols: {mapped_n}/{len(df)} ({mapped_n/len(df):.1%})")

# src/data_utils/dataloader.py
import argparse
import os
import sys
import hashlib
import requests
from pathlib import Path

CHUNK = 1024 * 1024


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, out: Path, overwrite: bool = False, timeout: int = 60) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists() and not overwrite:
        return

    tmp = out.with_suffix(out.suffix + ".partial")
    headers = {}

    resume_from = tmp.stat().st_size if tmp.exists() else 0
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    print(f"[dl] {url}")
    with requests.get(url, stream=True, headers=headers, timeout=timeout) as r:
        r.raise_for_status()
        mode = "ab" if resume_from > 0 else "wb"
        with tmp.open(mode) as f:
            for chunk in r.iter_content(chunk_size=CHUNK):
                if chunk:
                    f.write(chunk)

    tmp.replace(out)
    print(f"[ok] -> {out} ({out.stat().st_size/1e6:.2f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="physeval root (default: .)")
    ap.add_argument("--string-version", default="v12.0")
    ap.add_argument("--taxid", default="9606")
    ap.add_argument("--with-biogrid", action="store_true")
    ap.add_argument("--with-elm", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve()

    base = f"https://stringdb-downloads.org/download"

    string_out = root / "inputs" / "ppi" / "STRING" / args.string_version
    files = [
        f"{args.taxid}.protein.links.detailed.{args.string_version}.txt.gz",
        f"{args.taxid}.protein.aliases.{args.string_version}.txt.gz",
        f"{args.taxid}.protein.info.{args.string_version}.txt.gz",
        f"{args.taxid}.protein.physical.links.detailed.{args.string_version}.txt.gz",
        f"{args.taxid}.protein.sequences.{args.string_version}.fa.gz",
    ]

    for fn in files:
        url = f"{base}/{fn}"
        out = string_out / fn
        try:
            download(url, out, overwrite=args.overwrite)
        except Exception as e:
            print(f"[warn] STRING failed: {fn}: {e}")

    if args.with_biogrid:
        biogrid_out = root / "inputs" / "ppi" / "BioGRID"
        biogrid_base = "https://downloads.thebiogrid.org/BioGRID/Latest-Release"
        biogrid_files = [
            "BIOGRID-ALL-LATEST.mitab.zip",
            "BIOGRID-MV-Physical-LATEST.mitab.zip",
        ]
        for fn in biogrid_files:
            url = f"{biogrid_base}/{fn}"
            out = biogrid_out / fn
            try:
                download(url, out, overwrite=args.overwrite)
            except Exception as e:
                print(f"[warn] BioGRID failed: {fn}: {e}")

    if args.with_elm:
        elm_out = root / "inputs" / "annotations" / "motifs" / "ELM"
        elm_url = "http://elm.eu.org/instances.tsv?q=None&taxon=Homo%20sapiens&instance_logic=true%20positive"
        out = elm_out / "ELM_instances_human_true_positive.tsv"
        try:
            download(elm_url, out, overwrite=args.overwrite)
        except Exception as e:
            print(f"[warn] ELM failed: {e}")


if __name__ == "__main__":
    main()

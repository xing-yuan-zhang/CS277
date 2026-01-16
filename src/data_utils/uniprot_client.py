# src/data_utils/uniprot_client.py
from __future__ import annotations
import re
from pathlib import Path
import argparse
import json
import time
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import os
import requests
import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SEEDS = str(ROOT / "inputs/seeds/seeds.init.tsv")
ATTR = Path(ROOT / "inputs/annotations/nodes.attributes.tsv")
OUT = Path(ROOT / "inputs/seeds/seeds.mapped.tsv")

UNIPROT_RE = re.compile(
    r"""
    \b(
        [A-NR-Z][0-9][A-Z0-9]{3}[0-9]
      | [OPQ][0-9][A-Z0-9]{3}[0-9]
      | A0A[0-9A-Z]{7}
    )\b
    """,
    re.VERBOSE
)

def seed_map_main(seeds_path: Optional[str] = None, attr_path: Optional[str] = None, out_path: Optional[str] = None) -> None:
    seeds_path = seeds_path or SEEDS
    attr_path = attr_path or str(ATTR)
    out_path = out_path or str(OUT)

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    s = pd.read_csv(seeds_path, sep="\t")
    if "query" not in s.columns:
        raise ValueError("seeds.tsv need query column")
    s["query"] = s["query"].astype(str).str.strip()

    attr = pd.read_csv(attr_path, sep="\t")
    if not {"entry", "gene_symbol"}.issubset(set(attr.columns)):
        raise ValueError("nodes.attributes.tsv need entry and gene_symbol columns")

    m = attr.dropna(subset=["gene_symbol"]).copy()
    m["gene_symbol"] = m["gene_symbol"].astype(str).str.upper()
    gene2entry = m.groupby("gene_symbol")["entry"].apply(list).to_dict()

    out_rows = []
    for q in s["query"].tolist():
        if UNIPROT_RE.match(q):
            out_rows.append({"query": q, "entry": q, "method": "as_uniprot", "n_candidates": 1})
            continue
        key = q.upper()
        cands = gene2entry.get(key, [])
        if len(cands) == 0:
            out_rows.append({"query": q, "entry": None, "method": "not_found", "n_candidates": 0})
        else:
            out_rows.append({"query": q, "entry": cands[0], "method": "gene_symbol_in_nodes.attributes", "n_candidates": len(cands)})

    out = pd.DataFrame(out_rows)
    out.to_csv(out_p, sep="\t", index=False)
    missing = out["entry"].isna().sum()
    print(f"[OK] wrote {out_p} seeds={len(out)} missing={missing}")
    if missing > 0:
        print("[WARN] some seeds not mapped to UniProt entries.")

@dataclass
class UniProtClientConfig:
    endpoint: str = "https://rest.uniprot.org"
    timeout_sec: int = 30
    max_retries: int = 3
    sleep_sec_between_calls: float = 0.1

class UniProtClient:
    def __init__(self, cfg: UniProtClientConfig, cache_path: Optional[str] = None):
        self.cfg = cfg
        self.cache_path = cache_path
        self.cache: Dict[str, Any] = {}
        if cache_path:
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except FileNotFoundError:
                self.cache = {}

    def _save_cache(self) -> None:
        if not self.cache_path:
            return
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def search_best_hit(
        self,
        query: str,
        taxon_id: int,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        cache_key = f"search_best_hit|q={query}|taxon={taxon_id}|fields={','.join(fields or [])}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        fields = fields or [
            "accession",
            "id",
            "protein_name",
            "gene_primary",
            "gene_names",
            "organism_id",
            "length",
            "reviewed"
        ]

        url = f"{self.cfg.endpoint}/uniprotkb/search"
        params = {
            "query": f"({query}) AND (organism_id:{taxon_id})",
            "format": "json",
            "fields": ",".join(fields),
            "size": 5,
        }

        last_err: Optional[Exception] = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                r = requests.get(url, params=params, timeout=self.cfg.timeout_sec)
                if r.status_code == 400:
                    try:
                        print("[UniProt 400]", r.json())
                    except Exception:
                        print("[UniProt 400 raw]", r.text)
                r.raise_for_status()
                data = r.json()
                self.cache[cache_key] = data
                self._save_cache()
                time.sleep(self.cfg.sleep_sec_between_calls)
                return data
            except Exception as e:
                last_err = e
                time.sleep(0.5 * attempt)

        raise RuntimeError(f"UniProt query failed after retries: {query}") from last_err

@dataclass
class Scope:
    taxon_id: int
    canonical_id: str
    allow_multiple_uniprot_hits: bool
    fail_on_unmapped: bool
    fail_on_ambiguous: bool

def setup_logger(log_path: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("seed_normalize")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def parse_scope(scope_yaml: Dict[str, Any]) -> Scope:
    return Scope(
        taxon_id=int(scope_yaml["organism"]["taxon_id"]),
        canonical_id=scope_yaml["id_policy"]["canonical_id"],
        allow_multiple_uniprot_hits=bool(scope_yaml["id_policy"].get("allow_multiple_uniprot_hits", False)),
        fail_on_unmapped=bool(scope_yaml["quality_gate"].get("fail_on_unmapped", True)),
        fail_on_ambiguous=bool(scope_yaml["quality_gate"].get("fail_on_ambiguous", True)),
    )

def extract_hits(uniprot_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    return uniprot_json.get("results", []) or []

def normalize_one_seed(
    client: UniProtClient,
    query: str,
    taxon_id: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw = client.search_best_hit(query=query, taxon_id=taxon_id)
    hits = extract_hits(raw)
    return hits, raw

def hit_to_row(hit: Dict[str, Any]) -> Dict[str, Any]:
    primary_acc = hit.get("primaryAccession", "")
    uni_id = hit.get("uniProtkbId", "")
    organism = (hit.get("organism") or {}).get("taxonId", "")
    reviewed = hit.get("entryType", "")
    length = (hit.get("sequence") or {}).get("length", "")

    genes = hit.get("genes") or []
    gene_primary = ""
    if genes and isinstance(genes, list):
        gene_obj = genes[0] or {}
        gene_primary = ((gene_obj.get("geneName") or {}).get("value")) or ""

    protein_desc = hit.get("proteinDescription") or {}
    rec_name = ""
    if "recommendedName" in protein_desc:
        rec_name = ((protein_desc["recommendedName"].get("fullName") or {}).get("value")) or ""

    return {
        "uniprot_acc": primary_acc,
        "uniprot_id": uni_id,
        "gene_symbol": gene_primary,
        "protein_name": rec_name,
        "taxon_id": organism,
        "reviewed": reviewed,
        "length": length,
    }

def build_query(seed: str, taxon_id: int, reviewed: bool | None):
    q = f'(gene:{seed}) AND (organism_id:{taxon_id})'
    if reviewed is True:
        q += ' AND (reviewed:true)'
    return q

def pick_best_uniprot_hit(query: str, hits: list[dict]) -> dict | None:
    q = query.upper().strip()

    def is_reviewed(h: dict) -> int:
        et = str(h.get("entryType", "")).lower()
        return 1 if "reviewed" in et else 0

    def gene_primary(h: dict) -> str:
        genes = h.get("genes") or []
        if genes and isinstance(genes, list):
            g0 = genes[0] or {}
            gn = (g0.get("geneName") or {}).get("value")
            if isinstance(gn, str):
                return gn.upper()
        return ""

    def gene_synonyms(h: dict) -> set[str]:
        syns: set[str] = set()
        genes = h.get("genes") or []
        if not genes or not isinstance(genes, list):
            return syns
        g0 = genes[0] or {}
        for s in (g0.get("synonyms") or []):
            v = (s or {}).get("value")
            if isinstance(v, str):
                syns.add(v.upper())
        return syns

    def protein_name(h: dict) -> str:
        pd = h.get("proteinDescription") or {}
        rn = (pd.get("recommendedName") or {}).get("fullName") or {}
        v = rn.get("value")
        if isinstance(v, str):
            return v
        return ""

    def length(h: dict) -> int:
        seq = h.get("sequence") or {}
        try:
            return int(seq.get("length") or 0)
        except Exception:
            return 0

    def score(h: dict) -> tuple:
        reviewed = is_reviewed(h)

        gp = gene_primary(h)
        primary_match = 1 if gp == q else 0

        syn_match = 1 if q in gene_synonyms(h) else 0

        pname = protein_name(h).lower()
        iso_penalty = 1 if "isoform" in pname else 0

        L = length(h)

        return (reviewed, primary_match, syn_match, -iso_penalty, L)

    ranked = sorted(hits, key=score, reverse=True)
    if not ranked:
        return None
    if len(ranked) == 1:
        return ranked[0]

    best = ranked[0]
    s0 = score(best)

    if s0[0] == 1 and s0[1] == 1:
        return best

    if s0[1] == 1:
        return best

    if score(ranked[0]) == score(ranked[1]):
        return None

    return best

def write_tsv(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

def uniprot_main(
    scope_path: str,
    seeds_path: str,
    cache_path: str,
    out_tsv: str,
    out_json: str,
    log_path: str,
) -> None:
    scope_yaml = load_yaml(scope_path)
    seeds_yaml = load_yaml(seeds_path)
    scope = parse_scope(scope_yaml)

    logger = setup_logger(log_path, level=scope_yaml.get("logging", {}).get("level", "INFO"))

    up_cfg = UniProtClientConfig(
        endpoint=scope_yaml.get("uniprot", {}).get("endpoint", "https://rest.uniprot.org"),
        timeout_sec=int(scope_yaml.get("uniprot", {}).get("timeout_sec", 30)),
        max_retries=int(scope_yaml.get("uniprot", {}).get("max_retries", 3)),
        sleep_sec_between_calls=float(scope_yaml.get("uniprot", {}).get("sleep_sec_between_calls", 0.1)),
    )
    client = UniProtClient(cfg=up_cfg, cache_path=cache_path)

    seeds = seeds_yaml.get("seeds", [])
    if not seeds:
        raise ValueError("configs/seeds.yaml: seeds is empty.")

    normalized_rows: List[Dict[str, Any]] = []
    audit_records: List[Dict[str, Any]] = []

    unmapped: List[str] = []
    ambiguous: List[str] = []

    for item in seeds:
        query = str(item["query"]).strip()
        role = item.get("role", "")
        notes = item.get("notes", "")
        evidence = item.get("evidence", [])

        hits, raw = normalize_one_seed(client, query=query, taxon_id=scope.taxon_id)

        if len(hits) == 0:
            logger.error(f"UNMAPPED: {query}")
            unmapped.append(query)
            audit_records.append({"query": query, "status": "unmapped", "raw": raw})
            continue

        if len(hits) > 1 and (not scope.allow_multiple_uniprot_hits):
            best = pick_best_uniprot_hit(query, hits)

            if best is None:
                logger.error(f"AMBIGUOUS: {query} | hits={len(hits)}")
                cand = [(h.get("primaryAccession"), h.get("uniProtkbId"), h.get("entryType")) for h in hits]
                logger.error(f"CANDIDATES: {query} | {cand}")

                ambiguous.append(query)
                audit_records.append({"query": query, "status": "ambiguous", "raw": raw})
                continue

            hits = [best]

        best = hits[0]
        row = hit_to_row(best)

        row.update({
            "query": query,
            "role": role,
            "notes": notes,
            "evidence": json.dumps(evidence, ensure_ascii=False),
        })

        normalized_rows.append(row)
        audit_records.append({"query": query, "status": "ok", "resolved": row})

        logger.info(f"OK: {query} -> {row.get('uniprot_acc')} ({row.get('gene_symbol')})")

    if unmapped and scope.fail_on_unmapped:
        raise SystemExit(f"Fail: unmapped seeds found: {unmapped}")

    if ambiguous and scope.fail_on_ambiguous:
        raise SystemExit(f"Fail: ambiguous seeds found: {ambiguous}")

    write_tsv(out_tsv, normalized_rows)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scope": {
                    "taxon_id": scope.taxon_id,
                    "canonical_id": scope.canonical_id,
                },
                "seeds": normalized_rows,
                "audit": audit_records,
                "unmapped": unmapped,
                "ambiguous": ambiguous,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(f"Wrote: {out_tsv}")
    logger.info(f"Wrote: {out_json}")

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    ap_map = sp.add_parser("map")
    ap_map.add_argument("--seeds", default=SEEDS)
    ap_map.add_argument("--attr", default=str(ATTR))
    ap_map.add_argument("--out", default=str(OUT))

    ap_up = sp.add_parser("uniprot")
    ap_up.add_argument("--scope", required=True, help="configs/scope.yaml")
    ap_up.add_argument("--seeds", required=True, help="configs/seed.init.yaml")
    ap_up.add_argument("--cache", required=True, help="inputs/seeds/uniprot_cache.json")
    ap_up.add_argument("--out_tsv", required=True, help="inputs/seeds/seed_normalized.tsv")
    ap_up.add_argument("--out_json", required=True, help="inputs/seeds/seed_normalized.json")
    ap_up.add_argument("--log", required=True, help="logs/seed_normalize_client.log")

    return ap

def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    if args.cmd == "map":
        seed_map_main(seeds_path=args.seeds, attr_path=args.attr, out_path=args.out)
        return

    if args.cmd == "uniprot":
        uniprot_main(
            scope_path=args.scope,
            seeds_path=args.seeds,
            cache_path=args.cache,
            out_tsv=args.out_tsv,
            out_json=args.out_json,
            log_path=args.log,
        )
        return

    raise SystemExit(f"err: {args.cmd}")

if __name__ == "__main__":
    main()

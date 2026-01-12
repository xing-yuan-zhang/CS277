# src/candidate_diffusion.py
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from pathlib import Path

import networkx as nx
import pandas as pd
import pickle

try:
    import yaml
except ImportError as e:
    raise SystemExit("Missing dependency: pyyaml. Please `pip install pyyaml`.") from e


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("candidate_diffusion")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


@dataclass
class SeedItem:
    query: str
    weight: float = 1.0
    role: str = ""
    evidence: List[str] = field(default_factory=list)


@dataclass
class ControlItem:
    query: str
    role: str = ""
    evidence: List[str] = field(default_factory=list)


def load_graph(path: str):
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G


def load_seeds_yaml(path: str) -> Tuple[List[SeedItem], List[ControlItem]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)

    seeds_raw = obj.get("seeds", []) or []
    ctrls_raw = obj.get("negative_controls", []) or []

    seeds: List[SeedItem] = []
    for x in seeds_raw:
        seeds.append(
            SeedItem(
                query=str(x.get("query", "")).strip(),
                weight=float(x.get("weight", 1.0)),
                role=str(x.get("role", "")).strip(),
                evidence=list(x.get("evidence", []) or []),
            )
        )

    ctrls: List[ControlItem] = []
    for x in ctrls_raw:
        ctrls.append(
            ControlItem(
                query=str(x.get("query", "")).strip(),
                role=str(x.get("role", "")).strip(),
                evidence=list(x.get("evidence", []) or []),
            )
        )

    return seeds, ctrls


def build_personalization(
    G: nx.Graph,
    seeds: List[SeedItem],
    exclude_roles: Set[str],
    logger: logging.Logger,
) -> Tuple[Dict[str, float], List[str], List[str]]:
    missing: List[str] = []
    active: List[str] = []
    weights: Dict[str, float] = {}

    for s in seeds:
        if not s.query:
            continue
        if s.role in exclude_roles:
            missing.append(s.query)
            continue
        if s.query not in G:
            missing.append(s.query)
            continue
        w = float(s.weight) if s.weight is not None else 1.0
        if w <= 0:
            missing.append(s.query)
            continue
        weights[s.query] = weights.get(s.query, 0.0) + w
        active.append(s.query)

    if not weights:
        raise ValueError("No valid seeds found in graph for personalization.")

    total = sum(weights.values())
    personalization = {k: v / total for k, v in weights.items()}

    logger.info(f"Personalization seeds used: {len(personalization)}")
    if missing:
        logger.warning(
            f"Seeds missing or excluded: {len(missing)} -> {missing[:20]}{'...' if len(missing)>20 else ''}"
        )

    return personalization, sorted(set(active)), sorted(set(missing))


def run_ppr(
    G: nx.Graph,
    alpha: float,
    personalization: Dict[str, float],
    weight_attr: Optional[str],
    max_iter: int,
    tol: float,
    logger: logging.Logger,
) -> Dict[str, float]:
    logger.info(
        f"Running PPR with alpha={alpha:.3f}, restart={1-alpha:.3f}, weight_attr={weight_attr}, max_iter={max_iter}, tol={tol}"
    )
    try:
        scores = nx.pagerank(
            G,
            alpha=alpha,
            personalization=personalization,
            weight=weight_attr if weight_attr else None,
            max_iter=max_iter,
            tol=tol,
        )
    except nx.PowerIterationFailedConvergence as e:
        raise RuntimeError(
            f"PageRank did not converge (alpha={alpha}, max_iter={max_iter}, tol={tol}). Try increasing --max_iter or relaxing --tol."
        ) from e
    return scores


def add_degree_features(G: nx.Graph, df: pd.DataFrame, weight_attr: Optional[str]) -> pd.DataFrame:
    deg = dict(G.degree())
    df["degree"] = df["node"].map(deg).fillna(0).astype(int)

    if weight_attr:
        try:
            wdeg = dict(G.degree(weight=weight_attr))
            df["weighted_degree"] = df["node"].map(wdeg).fillna(0.0)
        except Exception:
            pass
    return df


def maybe_merge_node_attributes(
    df: pd.DataFrame,
    node_attr_path: Optional[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    if not node_attr_path:
        return df
    if not os.path.exists(node_attr_path):
        logger.warning(f"Node attributes file not found: {node_attr_path}")
        return df

    na = pd.read_csv(node_attr_path, sep="\t")
    if "entry" in na.columns and "node" not in na.columns:
        na = na.rename(columns={"entry": "node"})
    if "node" not in na.columns:
        logger.warning(f"Node attributes file missing 'entry' or 'node' column: {node_attr_path}")
        return df

    merged = df.merge(na, on="node", how="left")
    logger.info(f"Merged node attributes from {node_attr_path}")
    return merged


def compute_qc(
    df: pd.DataFrame,
    seed_nodes: Set[str],
    negative_nodes: Set[str],
    topk: int = 50,
    llps_col: str = "is_LLPS_any",
) -> Dict:
    df2 = df.copy()
    df2["rank"] = range(1, len(df2) + 1)

    seed_df = df2[df2["node"].isin(seed_nodes)]
    neg_df = df2[df2["node"].isin(negative_nodes)]

    qc = {
        "n_nodes": int(len(df2)),
        "topk": int(topk),
        "n_seeds": int(len(seed_nodes)),
        "n_neg_controls": int(len(negative_nodes)),
        "seeds_in_topk": int((seed_df["rank"] <= topk).sum()) if len(seed_df) else 0,
        "seed_rank_mean": float(seed_df["rank"].mean()) if len(seed_df) else None,
        "seed_rank_median": float(seed_df["rank"].median()) if len(seed_df) else None,
        "neg_rank_mean": float(neg_df["rank"].mean()) if len(neg_df) else None,
        "neg_rank_median": float(neg_df["rank"].median()) if len(neg_df) else None,
        "top10_nodes": df2.head(10)[["node", "score"]].to_dict(orient="records"),
    }

    if len(seed_df):
        qc["seed_ranks"] = seed_df.sort_values("rank")[["node", "rank", "score"]].to_dict(orient="records")
    else:
        qc["seed_ranks"] = []

    if len(neg_df):
        qc["neg_control_ranks"] = neg_df.sort_values("rank")[["node", "rank", "score"]].to_dict(orient="records")
    else:
        qc["neg_control_ranks"] = []

    if llps_col in df2.columns:
        top = df2.head(topk)
        base = df2[llps_col].fillna(0).astype(int)
        topv = top[llps_col].fillna(0).astype(int)
        qc["llps_col"] = llps_col
        qc["llps_base_rate"] = float(base.mean()) if len(base) else None
        qc["llps_in_topk"] = int(topv.sum())
        qc["llps_rate_in_topk"] = float(topv.mean()) if len(topv) else None

    return qc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--graph", required=True, help="Path to NetworkX graph gpickle/pkl")
    p.add_argument("--seeds", required=True, help="Path to seeds.yaml (queries must match graph node IDs)")
    p.add_argument("--outdir", required=True, help="Output directory, e.g., outputs/diffusion")
    p.add_argument("--alpha", nargs="+", type=float, default=[0.7])
    p.add_argument("--topn", type=int, default=100)
    p.add_argument("--weight_attr", type=str, default="weight")
    p.add_argument("--qc_topk", type=int, default=50)
    p.add_argument("--exclude_seed_roles", nargs="*", default=[])
    p.add_argument("--max_iter", type=int, default=500)
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--node_attributes", type=str, default="")
    p.add_argument("--llps_col", type=str, default="is_LLPS_any")
    return p.parse_args()


def main() -> None:
    ROOT = Path(__file__).resolve().parents[1]

    args = parse_args()
    outdir = str(ROOT / args.outdir)
    os.makedirs(outdir, exist_ok=True)

    log_path = os.path.join(outdir, "diffusion.log")
    logger = setup_logger(log_path)

    logger.info("=== Candidate diffusion start ===")
    logger.info(f"Graph: {args.graph}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Outdir: {outdir}")

    G = load_graph(str(ROOT / args.graph))
    logger.info(f"Loaded graph: |V|={G.number_of_nodes():,}, |E|={G.number_of_edges():,}, directed={G.is_directed()}")

    seeds, ctrls = load_seeds_yaml(str(ROOT / args.seeds))

    exclude_roles = set(args.exclude_seed_roles or [])
    personalization, active_seed_nodes, missing_seed_nodes = build_personalization(G, seeds, exclude_roles, logger)

    seed_set = set(active_seed_nodes)
    neg_set = set([c.query for c in ctrls if c.query in G])

    node_attr_path = args.node_attributes.strip()
    if not node_attr_path:
        inferred = os.path.join(os.path.dirname(os.path.abspath(args.graph)), "nodes.final.tsv")
        if os.path.exists(inferred):
            node_attr_path = inferred

    config = {
        "graph": args.graph,
        "seeds_yaml": args.seeds,
        "alphas": args.alpha,
        "topn": args.topn,
        "weight_attr": args.weight_attr if args.weight_attr else None,
        "exclude_seed_roles": sorted(list(exclude_roles)),
        "active_seeds": active_seed_nodes,
        "missing_or_excluded_seeds": missing_seed_nodes,
        "negative_controls_in_graph": sorted(list(neg_set)),
        "max_iter": args.max_iter,
        "tol": args.tol,
        "node_attributes": node_attr_path if node_attr_path else None,
        "llps_col": args.llps_col,
    }
    with open(os.path.join(outdir, "diffusion_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    weight_attr = args.weight_attr.strip() if args.weight_attr else None
    if weight_attr == "":
        weight_attr = None

    for a in args.alpha:
        if not (0.0 < a < 1.0):
            raise ValueError(f"alpha must be in (0,1), got {a}")

        scores = run_ppr(
            G,
            alpha=a,
            personalization=personalization,
            weight_attr=weight_attr,
            max_iter=args.max_iter,
            tol=args.tol,
            logger=logger,
        )

        df = pd.DataFrame({"node": list(scores.keys()), "score": list(scores.values())})
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
        df = add_degree_features(G, df, weight_attr=weight_attr)
        df = maybe_merge_node_attributes(df, node_attr_path, logger)

        full_path = os.path.join(outdir, f"diffusion_scores_alpha{a:.2f}.csv")
        df.to_csv(full_path, index=False)

        cand = df[~df["node"].isin(seed_set)].copy().reset_index(drop=True)
        cand["rank_excluding_seeds"] = range(1, len(cand) + 1)
        cand_top = cand.head(args.topn).copy()

        cand_path = os.path.join(outdir, f"candidates_diffusion_top{args.topn}_alpha{a:.2f}.csv")
        cand_top.to_csv(cand_path, index=False)

        qc = compute_qc(df, seed_nodes=seed_set, negative_nodes=neg_set, topk=args.qc_topk, llps_col=args.llps_col)
        qc_path = os.path.join(outdir, f"qc_alpha{a:.2f}.json")
        with open(qc_path, "w", encoding="utf-8") as f:
            json.dump(qc, f, indent=2, ensure_ascii=False)

        logger.info(f"[alpha={a:.2f}] Wrote full scores: {full_path}")
        logger.info(f"[alpha={a:.2f}] Wrote candidates:  {cand_path}")
        logger.info(
            f"[alpha={a:.2f}] QC seeds_in_top{args.qc_topk}={qc['seeds_in_topk']}/{qc['n_seeds']}, seed_rank_median={qc['seed_rank_median']}"
        )

    logger.info("=== Candidate diffusion done ===")


if __name__ == "__main__":
    main()

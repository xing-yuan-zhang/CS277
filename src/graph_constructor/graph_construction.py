# src/graph_constructor/graph_construction.py
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import pandas as pd

def _infer_sep(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in {".tsv", ".txt"}:
        return "\t"
    return ","


def _read_table(path: Path, sep: Optional[str] = None) -> pd.DataFrame:
    if sep is None:
        sep = _infer_sep(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path, sep=sep, dtype=str)
    return df


def _read_seeds(path: Optional[Path], seeds_csv: Optional[str]) -> List[str]:
    seeds: List[str] = []
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Seed file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            seeds.append(line)
    if seeds_csv:
        seeds.extend([x.strip() for x in seeds_csv.split(",") if x.strip()])

    seen = set()
    uniq = []
    for s in seeds:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


def _to_float_safe(x: str, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _ensure_columns(df: pd.DataFrame, required: Sequence[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}. Present: {list(df.columns)}")


@dataclass(frozen=True)
class GraphBuildConfig:
    edge_score_col: str = "score"
    edge_a_col: str = "protein_a"
    edge_b_col: str = "protein_b"
    node_id_col: str = "protein"
    min_score: float = 0.0
    drop_self_loops: bool = True
    keep_mode: str = "seed_connected"  # {"seed_connected", "largest_cc", "full"}
    make_undirected: bool = True


def build_graph(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    seeds: Sequence[str],
    cfg: GraphBuildConfig,
) -> nx.Graph:
    _ensure_columns(nodes_df, [cfg.node_id_col], "nodes_df")
    _ensure_columns(edges_df, [cfg.edge_a_col, cfg.edge_b_col, cfg.edge_score_col], "edges_df")

    G: nx.Graph = nx.Graph() if cfg.make_undirected else nx.DiGraph()

    node_attr_cols = [c for c in nodes_df.columns if c != cfg.node_id_col]
    for _, row in nodes_df.iterrows():
        pid = str(row[cfg.node_id_col]).strip()
        if not pid:
            continue
        attrs = {c: row[c] for c in node_attr_cols}
        G.add_node(pid, **attrs)

    for s in seeds:
        if s not in G:
            G.add_node(s)
        G.nodes[s]["seed"] = True

    if "role" in nodes_df.columns:
        for s in seeds:
            role_val = G.nodes[s].get("role", None)
            if role_val is not None:
                G.nodes[s]["seed_role"] = role_val

    n_edges_in = 0
    n_edges_added = 0
    for _, row in edges_df.iterrows():
        a = str(row[cfg.edge_a_col]).strip()
        b = str(row[cfg.edge_b_col]).strip()
        if not a or not b:
            continue

        if cfg.drop_self_loops and a == b:
            continue

        score = _to_float_safe(str(row[cfg.edge_score_col]), default=float("nan"))
        n_edges_in += 1
        if score != score:
            continue
        if score < cfg.min_score:
            continue

        if a not in G:
            G.add_node(a)
        if b not in G:
            G.add_node(b)

        if G.has_edge(a, b):
            prev = G[a][b].get("weight", float("-inf"))
            if score > prev:
                G[a][b]["weight"] = score
        else:
            G.add_edge(a, b, weight=score)
        n_edges_added += 1

    if cfg.keep_mode not in {"seed_connected", "largest_cc", "full"}:
        raise ValueError(f"Unknown keep_mode: {cfg.keep_mode}")

    if cfg.keep_mode == "full":
        H = G
    else:
        if cfg.make_undirected:
            components = list(nx.connected_components(G))
        else:
            components = list(nx.weakly_connected_components(G))

        if not components:
            H = G
        elif cfg.keep_mode == "largest_cc":
            largest = max(components, key=len)
            H = G.subgraph(largest).copy()
        else:
            seed_set = set(seeds)
            keep_nodes = set()
            for comp in components:
                if comp & seed_set:
                    keep_nodes |= set(comp)
            H = G.subgraph(keep_nodes).copy()

    for n in H.nodes:
        if "seed" not in H.nodes[n]:
            H.nodes[n]["seed"] = False

    H.graph["n_edges_input_rows"] = n_edges_in
    H.graph["n_edges_added_post_filter"] = n_edges_added
    H.graph["min_score"] = cfg.min_score
    H.graph["keep_mode"] = cfg.keep_mode
    H.graph["seeds"] = list(seeds)

    return H


def save_graph(G: nx.Graph, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suf = out_path.suffix.lower()

    if suf in {".gpickle", ".pkl"}:
        nx.write_gpickle(G, out_path)
    elif suf in {".graphml"}:
        nx.write_graphml(G, out_path)
    elif suf in {".json"}:
        data = nx.node_link_data(G)
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported output suffix: {suf}. Use .gpickle/.graphml/.json")


def print_qc(G: nx.Graph) -> None:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    seeds = [x for x, d in G.nodes(data=True) if d.get("seed") is True]
    w = [d.get("weight") for _, _, d in G.edges(data=True) if d.get("weight") is not None]

    print("=== Graph QC Summary ===")
    print(f"Nodes: {n}")
    print(f"Edges: {m}")
    print(f"Seed nodes present: {len(seeds)} -> {seeds[:20]}{'...' if len(seeds) > 20 else ''}")

    if w:
        w_float = []
        for x in w:
            try:
                w_float.append(float(x))
            except Exception:
                continue
        if w_float:
            print(f"Edge weight stats (n={len(w_float)}): min={min(w_float):.4g}, "
                  f"median={sorted(w_float)[len(w_float)//2]:.4g}, max={max(w_float):.4g}")

    if isinstance(G, nx.DiGraph):
        comps = list(nx.weakly_connected_components(G))
    else:
        comps = list(nx.connected_components(G))
    comp_sizes = sorted([len(c) for c in comps], reverse=True)
    if comp_sizes:
        print(f"Connected components: {len(comp_sizes)}; largest={comp_sizes[0]}; top5={comp_sizes[:5]}")
    print("========================")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Construct weighted PPI graph for CIKA.")
    p.add_argument("--nodes", type=str, required=True, help="Nodes table (csv/tsv). Must contain column 'protein'.")
    p.add_argument("--edges", type=str, required=True, help="Edges table (csv/tsv). Must contain protein_a, protein_b, score.")
    p.add_argument("--out", type=str, required=True, help="Output graph path (.gpickle/.graphml/.json).")

    p.add_argument("--seeds_file", type=str, default=None, help="Seed list file (one symbol per line).")
    p.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds, e.g. SRC,TRIP10,FNBP1")

    p.add_argument("--min_score", type=float, default=0.0, help="Minimum edge score threshold.")
    p.add_argument("--keep_mode", type=str, default="seed_connected",
                   choices=["seed_connected", "largest_cc", "full"],
                   help="Which subgraph to keep.")
    p.add_argument("--directed", action="store_true", help="Build a directed graph (default: undirected).")

    p.add_argument("--node_id_col", type=str, default="protein")
    p.add_argument("--edge_a_col", type=str, default="protein_a")
    p.add_argument("--edge_b_col", type=str, default="protein_b")
    p.add_argument("--edge_score_col", type=str, default="score")

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    nodes_path = Path(args.nodes)
    edges_path = Path(args.edges)
    out_path = Path(args.out)

    seeds = _read_seeds(Path(args.seeds_file) if args.seeds_file else None, args.seeds)
    if not seeds:
        raise ValueError("No seeds provided. Use --seeds_file or --seeds.")

    nodes_df = _read_table(nodes_path)
    edges_df = _read_table(edges_path)

    cfg = GraphBuildConfig(
        edge_score_col=args.edge_score_col,
        edge_a_col=args.edge_a_col,
        edge_b_col=args.edge_b_col,
        node_id_col=args.node_id_col,
        min_score=float(args.min_score),
        drop_self_loops=True,
        keep_mode=args.keep_mode,
        make_undirected=(not args.directed),
    )

    G = build_graph(nodes_df=nodes_df, edges_df=edges_df, seeds=seeds, cfg=cfg)
    print_qc(G)
    save_graph(G, out_path)

    print(f"Saved graph -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

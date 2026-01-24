# src/helpers/spectronaut_alignment.py
import argparse
import os
import re
from typing import List, Set, Tuple, Optional

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path


UNIPROT_RE = re.compile(r"^(?:[A-NR-Z][0-9][A-Z0-9]{3}[0-9]|[OPQ][0-9][A-Z0-9]{3}[0-9]|A0A[A-Z0-9]{7,10})$")
SPLIT_RE = re.compile(r"[;,\s]+")


def normalize_uniprot(x: str) -> str:
    if x is None:
        return ""
    x = str(x).strip()
    if not x:
        return ""
    x = x.split("-")[0]
    return x


def extract_uniprots_from_cell(cell: object) -> List[str]:
    if cell is None:
        return []
    s = str(cell).strip()
    if not s:
        return []
    parts = [p for p in SPLIT_RE.split(s) if p]
    out: List[str] = []
    for p in parts:
        p = normalize_uniprot(p)
        if p and UNIPROT_RE.match(p):
            out.append(p)
    return out


def detect_uniprot_column(df: pd.DataFrame, user_col: Optional[str] = None) -> str:
    if user_col:
        if user_col not in df.columns:
            raise ValueError(f"--ms_id_col '{user_col}' 不在 MS 文件列名中。可用列名：{list(df.columns)}")
        return user_col

    best_col = None
    best_score = -1.0

    for col in df.columns:
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            s = df[col].dropna()
            if len(s) == 0:
                continue
            hit = s.map(lambda v: 1.0 if len(extract_uniprots_from_cell(v)) > 0 else 0.0).mean()
            name_bonus = 0.05 if str(col).lower() in {
                "entry", "accession", "uniprot", "uniprot_id", "protein", "pg.proteinaccessions"
            } else 0.0
            score = hit + name_bonus
            if score > best_score:
                best_score = score
                best_col = col

    if best_col is None or best_score < 0.05:
        raise ValueError(
            "未能可靠检测到 UniProt accession 列。请用 --ms_id_col 显式指定。\n"
            f"MS 文件列名：{list(df.columns)}"
        )
    return best_col


def load_ms_uniprots(ms_path: str, ms_id_col: Optional[str] = None) -> Set[str]:
    df = pd.read_csv(ms_path, sep="\t", low_memory=False)
    col = detect_uniprot_column(df, ms_id_col)
    all_ids: Set[str] = set()
    for cell in df[col].dropna().tolist():
        for acc in extract_uniprots_from_cell(cell):
            all_ids.add(acc)
    return all_ids


def load_edges(edge_path: str, cols: Tuple[str, str] = ("entry_a", "entry_b")) -> pd.DataFrame:
    df = pd.read_csv(edge_path, sep="\t", low_memory=False)
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"边文件缺少列 '{c}'。实际列名：{list(df.columns)}")
    df[cols[0]] = df[cols[0]].astype(str).map(normalize_uniprot)
    df[cols[1]] = df[cols[1]].astype(str).map(normalize_uniprot)
    df = df[df[cols[0]].str.len() > 0]
    df = df[df[cols[1]].str.len() > 0]
    return df


def induced_subgraph_edges(edges: pd.DataFrame, nodes: Set[str], a="entry_a", b="entry_b") -> pd.DataFrame:
    m = edges[a].isin(nodes) & edges[b].isin(nodes)
    return edges.loc[m].copy()


def to_undirected_edge_set(edges: pd.DataFrame, a="entry_a", b="entry_b") -> Set[Tuple[str, str]]:
    s = set()
    for u, v in zip(edges[a].tolist(), edges[b].tolist()):
        if not u or not v:
            continue
        if u == v:
            continue
        s.add(tuple(sorted((u, v))))
    return s


def build_graph_from_edges(edges: pd.DataFrame, a="entry_a", b="entry_b", weight_col: Optional[str] = "weight") -> nx.Graph:
    G = nx.Graph()
    if weight_col and weight_col in edges.columns:
        for u, v, w in zip(edges[a], edges[b], edges[weight_col]):
            if u == v:
                continue
            G.add_edge(u, v, weight=float(w))
    else:
        for u, v in zip(edges[a], edges[b]):
            if u == v:
                continue
            G.add_edge(u, v)
    return G


def take_plot_subgraph(G: nx.Graph, max_nodes: int = 800) -> nx.Graph:
    if G.number_of_nodes() <= max_nodes:
        return G
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    H = G.subgraph(comps[0]).copy()
    if H.number_of_nodes() <= max_nodes:
        return H
    deg = sorted(H.degree, key=lambda x: x[1], reverse=True)
    keep = {n for n, d in deg[:max_nodes]}
    return H.subgraph(keep).copy()


def draw_ms_graph(G_ms: nx.Graph, out_png: str, max_nodes: int = 800, seed: int = 7) -> None:
    H = take_plot_subgraph(G_ms, max_nodes=max_nodes)
    pos = nx.spring_layout(H, seed=seed, k=None)
    plt.figure(figsize=(11, 9))
    nx.draw_networkx_edges(H, pos, alpha=0.25, width=0.7)
    nx.draw_networkx_nodes(H, pos, node_size=18)
    plt.title(f"MS-induced graph on STRING edges (plotted nodes={H.number_of_nodes()}, edges={H.number_of_edges()})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def draw_overlay(
    G_ms: nx.Graph,
    G_diff: nx.Graph,
    overlap_nodes: Set[str],
    overlap_edges: Set[Tuple[str, str]],
    out_png: str,
    max_nodes: int = 1200,
    seed: int = 7,
) -> None:
    U = nx.Graph()
    U.add_nodes_from(G_ms.nodes())
    U.add_nodes_from(G_diff.nodes())
    U.add_edges_from(G_ms.edges())
    U.add_edges_from(G_diff.edges())

    H = take_plot_subgraph(U, max_nodes=max_nodes)
    pos = nx.spring_layout(H, seed=seed)

    ms_nodes = set(G_ms.nodes())
    diff_nodes = set(G_diff.nodes())

    nodes_only_ms = [n for n in H.nodes() if (n in ms_nodes) and (n not in diff_nodes)]
    nodes_only_diff = [n for n in H.nodes() if (n in diff_nodes) and (n not in ms_nodes)]
    nodes_overlap = [n for n in H.nodes() if (n in overlap_nodes)]

    ms_es = {tuple(sorted(e)) for e in G_ms.edges()}
    diff_es = {tuple(sorted(e)) for e in G_diff.edges()}
    H_es = {tuple(sorted(e)) for e in H.edges()}

    edges_overlap = [e for e in H_es if e in overlap_edges]
    edges_only_ms = [e for e in H_es if (e in ms_es) and (e not in diff_es)]
    edges_only_diff = [e for e in H_es if (e in diff_es) and (e not in ms_es)]

    plt.figure(figsize=(12, 10))
    nx.draw_networkx_edges(H, pos, edgelist=edges_only_ms, alpha=0.18, width=0.8)
    nx.draw_networkx_edges(H, pos, edgelist=edges_only_diff, alpha=0.18, width=0.8, style="dashed")
    nx.draw_networkx_edges(H, pos, edgelist=edges_overlap, alpha=0.85, width=1.6)

    nx.draw_networkx_nodes(H, pos, nodelist=nodes_only_ms, node_size=18, alpha=0.65)
    nx.draw_networkx_nodes(H, pos, nodelist=nodes_only_diff, node_size=18, alpha=0.65)
    nx.draw_networkx_nodes(H, pos, nodelist=nodes_overlap, node_size=42, alpha=0.95)

    plt.title(
        "Overlay: MS-induced graph vs diffusion subgraph\n"
        f"Union plotted nodes={H.number_of_nodes()} edges={H.number_of_edges()} | "
        f"Overlap nodes={len(overlap_nodes)} overlap edges={len(overlap_edges)}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms_tsv", required=True)
    ap.add_argument("--string_edges", required=True)
    ap.add_argument("--diff_edges", required=True)
    ap.add_argument("--ms_id_col", default=None)
    ap.add_argument("--outdir", default="ms_vs_diffusion_out")
    ap.add_argument("--max_nodes_ms", type=int, default=800)
    ap.add_argument("--max_nodes_overlay", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)

    args = ap.parse_args()
    ROOT = Path(__file__).resolve().parents[2]
    os.makedirs(str(ROOT / args.outdir), exist_ok=True)

    ms_ids = load_ms_uniprots(str(ROOT / args.ms_tsv), args.ms_id_col)
    if len(ms_ids) == 0:
        raise RuntimeError("No UniProt accession in MS file.")

    string_df = load_edges(str(ROOT / args.string_edges), cols=("entry_a", "entry_b"))
    ms_edges_df = induced_subgraph_edges(string_df, ms_ids, a="entry_a", b="entry_b")

    diff_df = load_edges(str(ROOT / args.diff_edges), cols=("entry_a", "entry_b"))

    G_ms = build_graph_from_edges(ms_edges_df, a="entry_a", b="entry_b", weight_col="weight")
    G_diff = build_graph_from_edges(diff_df, a="entry_a", b="entry_b", weight_col="weight")

    ms_nodes = set(G_ms.nodes())
    diff_nodes = set(G_diff.nodes())
    overlap_nodes = ms_nodes & diff_nodes

    ms_es = to_undirected_edge_set(ms_edges_df, a="entry_a", b="entry_b")
    diff_es = to_undirected_edge_set(diff_df, a="entry_a", b="entry_b")
    overlap_edges = ms_es & diff_es

    stats_path = os.path.join(str(ROOT / args.outdir), "overlap_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"MS proteins extracted: {len(ms_ids)}\n")
        f.write(f"MS-induced graph nodes: {len(ms_nodes)} edges: {G_ms.number_of_edges()}\n")
        f.write(f"Diffusion graph nodes: {len(diff_nodes)} edges: {G_diff.number_of_edges()}\n")
        f.write(f"Overlap nodes (proteins): {len(overlap_nodes)}\n")
        f.write(f"Overlap edges: {len(overlap_edges)}\n")

    pd.Series(sorted(overlap_nodes), name="uniprot").to_csv(
        os.path.join(str(ROOT / args.outdir), "overlap_nodes.tsv"), sep="\t", index=False
    )

    overlap_edges_df = pd.DataFrame(sorted(list(overlap_edges)), columns=["entry_a", "entry_b"])
    overlap_edges_df.to_csv(os.path.join(str(ROOT / args.outdir), "overlap_edges.tsv"), sep="\t", index=False)
    ms_png = os.path.join(str(ROOT / args.outdir), "ms_induced_graph.png")
    overlay_png = os.path.join(str(ROOT / args.outdir), "overlay_ms_vs_diffusion.png")

    draw_ms_graph(G_ms, ms_png, max_nodes=args.max_nodes_ms, seed=args.seed)
    draw_overlay(G_ms, G_diff, overlap_nodes, overlap_edges, overlay_png, max_nodes=args.max_nodes_overlay, seed=args.seed)

if __name__ == "__main__":
    main()

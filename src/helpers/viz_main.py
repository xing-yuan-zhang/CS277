# src/helpers/viz_main.py
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque


def load_graph(edges_tsv: Path, pkl_graph: Path | None) -> nx.Graph:
    if pkl_graph is not None and pkl_graph.exists():
        with open(pkl_graph, "rb") as f:
            g = pickle.load(f)
        if not isinstance(g, nx.Graph):
            raise ValueError("ppi_subgraph.pkl is not a networkx Graph")
        return g
    e = pd.read_csv(edges_tsv, sep="\t", dtype={"entry_a": str, "entry_b": str})
    if "weight" not in e.columns:
        e["weight"] = 1.0
    g = nx.Graph()
    for r in e.itertuples(index=False):
        a = getattr(r, "entry_a")
        b = getattr(r, "entry_b")
        w = getattr(r, "weight")
        if a == b:
            continue
        g.add_edge(a, b, weight=float(w) if pd.notna(w) else 1.0)
    return g


def multi_source_hops(g: nx.Graph, seeds: list[str], cutoff: int | None = None) -> dict[str, int]:
    hops = {}
    dq = deque()
    for s in seeds:
        if s in g:
            hops[s] = 0
            dq.append(s)
    while dq:
        u = dq.popleft()
        if cutoff is not None and hops[u] >= cutoff:
            continue
        for v in g.neighbors(u):
            if v not in hops:
                hops[v] = hops[u] + 1
                dq.append(v)
    return hops

def safe_series(df: pd.DataFrame, col: str, fill=0):
    if col not in df.columns:
        return pd.Series([fill] * len(df), index=df.index)
    s = df[col]
    if s.dtype == object:
        s = s.fillna(fill)
    return s


def ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)


def plot_hop_subgraph_structure(
    g: nx.Graph,
    nodes_df: pd.DataFrame,
    hops: dict[str, int],
    outdir: Path,
    max_hop: int = 2,
    seed_layout: int = 42,
):
    is_seed = dict(zip(nodes_df["entry"].astype(str), nodes_df["is_seed"].astype(int)))
    keep = [n for n, h in hops.items() if h <= max_hop]
    sg = g.subgraph(keep).copy()
    if sg.number_of_nodes() == 0:
        raise ValueError("Empty hop subgraph for the chosen cutoff")
    pos = nx.spring_layout(sg, seed=seed_layout, k=None, weight="weight")
    node_colors = []
    node_sizes = []
    for n in sg.nodes():
        if is_seed.get(n, 0) == 1:
            node_colors.append("tab:red")
            node_sizes.append(220)
        else:
            node_colors.append("0.65")
            node_sizes.append(40)
    weights = np.array([sg[u][v].get("weight", 1.0) for u, v in sg.edges()], dtype=float)
    if len(weights) == 0:
        lw = []
    else:
        wmin, wmax = float(np.min(weights)), float(np.max(weights))
        if wmax > wmin:
            lw = 0.2 + 2.2 * (weights - wmin) / (wmax - wmin)
        else:
            lw = np.full_like(weights, 0.8, dtype=float)
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(sg, pos, alpha=0.25, width=lw)
    nx.draw_networkx_nodes(sg, pos, node_color=node_colors, node_size=node_sizes, linewidths=0.0)
    plt.title(f"Hop subgraph (<= {max_hop} hops), seeds highlighted")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outdir / "fig1_hop_subgraph_seed_colored.png", dpi=300)
    plt.close()


def plot_rank_score_curve(diff_df: pd.DataFrame, outdir: Path):
    d = diff_df.sort_values("score", ascending=False).reset_index(drop=True).copy()
    d["rank"] = np.arange(1, len(d) + 1)
    y = d["score"].to_numpy(dtype=float)
    y = np.clip(y, 1e-18, None)
    plt.figure(figsize=(8, 5.5))
    plt.plot(d["rank"], y, linewidth=1.2)
    plt.yscale("log")
    plt.xlabel("Rank (descending PPR score)")
    plt.ylabel("PPR score (log scale)")
    plt.title("Rank–score curve")
    plt.tight_layout()
    plt.savefig(outdir / "fig2_rank_score_log.png", dpi=300)
    plt.close()


def plot_score_vs_hop_boxplot(diff_df: pd.DataFrame, hops: dict[str, int], outdir: Path, max_hop_show: int = 4):
    d = diff_df.copy()
    d["hop"] = d["node"].astype(str).map(hops)
    d = d.dropna(subset=["hop"]).copy()
    d["hop"] = d["hop"].astype(int)
    d = d[d["hop"] <= max_hop_show].copy()
    if len(d) == 0:
        raise ValueError("No nodes with hop values available for boxplot")
    groups = [d.loc[d["hop"] == h, "score"].astype(float).to_numpy() for h in sorted(d["hop"].unique())]
    labels = [str(h) for h in sorted(d["hop"].unique())]
    groups = [np.clip(g, 1e-18, None) for g in groups]
    plt.figure(figsize=(8, 5.5))
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.yscale("log")
    plt.xlabel("Hop distance to nearest seed")
    plt.ylabel("PPR score (log scale)")
    plt.title("Score vs hop")
    plt.tight_layout()
    plt.savefig(outdir / "fig3_score_vs_hop_boxplot.png", dpi=300)
    plt.close()


def plot_top50_induced_subgraph(
    g: nx.Graph,
    nodes_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    outdir: Path,
    topn: int = 50,
    seed_layout: int = 42,
):
    is_seed = dict(zip(nodes_df["entry"].astype(str), nodes_df["is_seed"].astype(int)))
    d = diff_df.sort_values("score", ascending=False).head(topn).copy()
    top_nodes = d["node"].astype(str).tolist()
    sg = g.subgraph(top_nodes).copy()
    if sg.number_of_nodes() == 0:
        raise ValueError("Empty induced subgraph for top nodes")
    pos = nx.spring_layout(sg, seed=seed_layout, weight="weight")
    score_map = dict(zip(d["node"].astype(str), d["score"].astype(float)))
    scores = np.array([score_map.get(n, 0.0) for n in sg.nodes()], dtype=float)
    smax = float(np.max(scores)) if len(scores) else 1.0
    sizes = 120 + 2600 * (scores / smax if smax > 0 else scores)
    colors = ["tab:red" if is_seed.get(n, 0) == 1 else "tab:blue" for n in sg.nodes()]
    weights = np.array([sg[u][v].get("weight", 1.0) for u, v in sg.edges()], dtype=float)
    if len(weights) == 0:
        lw = []
    else:
        wmin, wmax = float(np.min(weights)), float(np.max(weights))
        if wmax > wmin:
            lw = 0.3 + 2.7 * (weights - wmin) / (wmax - wmin)
        else:
            lw = np.full_like(weights, 0.9, dtype=float)
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_edges(sg, pos, alpha=0.35, width=lw)
    nx.draw_networkx_nodes(sg, pos, node_color=colors, node_size=sizes, linewidths=0.0)
    plt.title(f"Top-{topn} induced subgraph (node size ∝ PPR score)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outdir / "fig4_top50_induced_subgraph.png", dpi=300)
    plt.close()


def plot_topk_annotation_heatmap(
    nodes_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    outdir: Path,
    topk: int = 50,
):
    d = diff_df.sort_values("score", ascending=False).head(topk).copy()
    n = nodes_df.copy()
    n["entry"] = n["entry"].astype(str)
    d["node"] = d["node"].astype(str)
    m = d.merge(n, left_on="node", right_on="entry", how="left")
    def pick_name(row):
        gs = row.get("gene_symbol")
        en = row.get("entry_name")
        if isinstance(gs, str) and gs.strip() != "" and gs != "nan":
            return gs
        if isinstance(en, str) and en.strip() != "" and en != "nan":
            return en
        return row["node"]
    labels = [pick_name(r) for _, r in m.iterrows()]
    cols = [
        ("is_seed", "Seed"),
        ("is_LLPSDB", "LLPSDB"),
        ("is_PhaSepDB", "PhaSepDB"),
        ("is_LLPS_any", "LLPS_any"),
        ("has_SH3", "SH3"),
        ("has_PRD", "PRD"),
        ("elm_sh3_related", "ELM_SH3_related"),
        ("has_any_domain", "Any_domain"),
    ]
    mat = []
    colnames = []
    for c, name in cols:
        if c not in m.columns:
            continue
        v = m[c]
        if v.dtype == object:
            vv = v.fillna(0)
            vv = vv.astype(str).str.lower().isin(["1", "true", "t", "yes", "y"]).astype(int)
        else:
            vv = v.fillna(0).astype(int)
            vv = (vv > 0).astype(int)
        mat.append(vv.to_numpy())
        colnames.append(name)
    if len(mat) == 0:
        raise ValueError("No annotation columns available for heatmap")
    X = np.vstack(mat).T
    plt.figure(figsize=(min(12, 0.8 * len(colnames) + 4), 0.22 * len(labels) + 3))
    plt.imshow(X, aspect="auto", interpolation="nearest")
    plt.xticks(np.arange(len(colnames)), colnames, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)
    plt.title(f"Top-{topk} annotation heatmap")
    cbar = plt.colorbar()
    cbar.set_label("0/1")
    plt.tight_layout()
    plt.savefig(outdir / "fig5_topk_annotation_heatmap.png", dpi=300)
    plt.close()


def plot_ppr_vs_degree_scatter(diff_df: pd.DataFrame, outdir: Path):
    d = diff_df.copy()
    x = d["degree"].astype(float).to_numpy()
    y = np.clip(d["score"].astype(float).to_numpy(), 1e-18, None)
    plt.figure(figsize=(7.5, 5.5))
    plt.scatter(x, y, s=12, alpha=0.6)
    plt.yscale("log")
    plt.xlabel("Degree")
    plt.ylabel("PPR score (log scale)")
    plt.title("PPR score vs degree")
    plt.tight_layout()
    plt.savefig(outdir / "fig6_ppr_vs_degree_scatter.png", dpi=300)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=str, default="inputs/pkl/nodes.final.tsv")
    ap.add_argument("--edges", type=str, default="inputs/pkl/edges.final.tsv")
    ap.add_argument("--diffusion", type=str, default="outputs/diffusion/diffusion_scores_alpha0.85.csv")
    ap.add_argument("--graph-pkl", type=str, default="inputs/pkl/ppi_subgraph.pkl")
    ap.add_argument("--outdir", type=str, default="outputs/figures")
    ap.add_argument("--max-hop", type=int, default=2)
    ap.add_argument("--top50", type=int, default=50)
    ap.add_argument("--topk-heatmap", type=int, default=50)
    ap.add_argument("--max-hop-box", type=int, default=4)
    ap.add_argument("--layout-seed", type=int, default=42)
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[2]
    nodes_path = Path(str(ROOT / args.nodes))
    edges_path = Path(str(ROOT / args.edges))
    diff_path = Path(str(ROOT / args.diffusion))
    graph_pkl = Path(str(ROOT / args.graph_pkl)) if args.graph_pkl else None
    outdir = Path(str(ROOT / args.outdir))
    ensure_outdir(outdir)

    nodes = pd.read_csv(nodes_path, sep="\t")
    edges = pd.read_csv(edges_path, sep="\t")
    diff = pd.read_csv(diff_path)

    nodes["entry"] = nodes["entry"].astype(str)
    diff["node"] = diff["node"].astype(str)

    g = load_graph(edges_path, graph_pkl)
    seeds = nodes.loc[nodes["is_seed"].astype(int) == 1, "entry"].astype(str).tolist()
    hops = multi_source_hops(g, seeds, cutoff=max(args.max_hop, args.max_hop_box))

    plot_hop_subgraph_structure(
        g=g,
        nodes_df=nodes,
        hops=hops,
        outdir=outdir,
        max_hop=args.max_hop,
        seed_layout=args.layout_seed,
    )
    plot_rank_score_curve(diff_df=diff, outdir=outdir)
    plot_score_vs_hop_boxplot(diff_df=diff, hops=hops, outdir=outdir, max_hop_show=args.max_hop_box)
    plot_top50_induced_subgraph(
        g=g,
        nodes_df=nodes,
        diff_df=diff,
        outdir=outdir,
        topn=args.top50,
        seed_layout=args.layout_seed,
    )
    plot_topk_annotation_heatmap(nodes_df=nodes, diff_df=diff, outdir=outdir, topk=args.topk_heatmap)
    plot_ppr_vs_degree_scatter(diff_df=diff, outdir=outdir)

    print(f"wrote figures to: {outdir}")


if __name__ == "__main__":
    main()

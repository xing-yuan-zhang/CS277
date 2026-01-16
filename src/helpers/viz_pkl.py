import argparse
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)


def load_graph(edges_tsv: Path, pkl_graph: Path | None) -> nx.Graph:
    if pkl_graph is not None and pkl_graph.exists():
        with open(pkl_graph, "rb") as f:
            g = pickle.load(f)
        if not isinstance(g, nx.Graph):
            raise ValueError("ppi_subgraph.pkl is not a networkx Graph")
        return g
    e = pd.read_csv(edges_tsv, sep="\t", dtype=str)
    if "weight" not in e.columns:
        e["weight"] = 1.0
    e["weight"] = pd.to_numeric(e["weight"], errors="coerce").fillna(1.0)
    g = nx.Graph()
    for r in e.itertuples(index=False):
        a = getattr(r, "entry_a")
        b = getattr(r, "entry_b")
        w = float(getattr(r, "weight"))
        if a == b:
            continue
        g.add_edge(str(a), str(b), weight=w)
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


def build_rank_map(diff_df: pd.DataFrame) -> dict[str, int]:
    d = diff_df[["node", "score"]].copy()
    d["node"] = d["node"].astype(str).str.strip()
    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    d = d.dropna(subset=["score"]).sort_values("score", ascending=False)
    d["rank"] = range(1, len(d) + 1)
    d["node_key"] = d["node"].str.split("-").str[0]
    return dict(zip(d["node_key"], d["rank"]))


def node_key(x: str) -> str:
    return str(x).strip().split("-")[0]


def viz_colors_by_rank(
    g: nx.Graph,
    rank_map: dict[str, int],
    is_seed_map: dict[str, int],
):
    cmap = plt.cm.Blues
    color_seed = "red"
    color_top50 = cmap(0.90)
    color_50_100 = cmap(0.70)
    color_100_200 = cmap(0.50)
    color_200_plus = cmap(0.30)
    color_unknown = "lightgray"

    node_colors = []
    for n in g.nodes():
        if int(is_seed_map.get(n, 0)) == 1:
            node_colors.append(color_seed)
        else:
            r = rank_map.get(node_key(n))
            if r is None:
                node_colors.append(color_unknown)
            elif r <= 50:
                node_colors.append(color_top50)
            elif r <= 100:
                node_colors.append(color_50_100)
            elif r <= 200:
                node_colors.append(color_100_200)
            else:
                node_colors.append(color_200_plus)
    return node_colors


def plot_hop_subgraph_structure_rank_shaded(
    g: nx.Graph,
    nodes_df: pd.DataFrame,
    hops: dict[str, int],
    rank_map: dict[str, int],
    outdir: Path,
    max_hop: int = 2,
    layout_seed: int = 42,
):
    is_seed_map = dict(zip(nodes_df["entry"].astype(str), nodes_df["is_seed"].astype(int)))
    keep = [n for n, h in hops.items() if h <= max_hop]
    sg = g.subgraph(keep).copy()
    if sg.number_of_nodes() == 0:
        raise ValueError("Empty hop subgraph for chosen cutoff")

    pos = nx.spring_layout(sg, seed=layout_seed, weight="weight", k=0.5)
    node_colors = viz_colors_by_rank(sg, rank_map, is_seed_map)

    w = np.array([sg[u][v].get("weight", 1.0) for u, v in sg.edges()], dtype=float)
    if len(w) == 0:
        widths = []
    else:
        wmin, wmax = float(np.min(w)), float(np.max(w))
        widths = (0.2 + 2.2 * (w - wmin) / (wmax - wmin)) if wmax > wmin else np.full_like(w, 0.8, dtype=float)

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(sg, pos, node_color=node_colors, node_size=30, alpha=0.85, linewidths=0.0)
    nx.draw_networkx_edges(sg, pos, width=widths, alpha=0.25)
    plt.title(f"Hop subgraph (<= {max_hop}) with rank-shaded nodes (viz_scores style)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outdir / "fig1_hop_subgraph_rank_shaded.png", dpi=300)
    plt.close()


def plot_rank_score_curve(diff_df: pd.DataFrame, outdir: Path):
    d = diff_df.sort_values("score", ascending=False).reset_index(drop=True).copy()
    d["rank"] = np.arange(1, len(d) + 1)
    y = pd.to_numeric(d["score"], errors="coerce").to_numpy(dtype=float)
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
    d["node"] = d["node"].astype(str)
    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    d["hop"] = d["node"].map(hops)
    d = d.dropna(subset=["hop", "score"]).copy()
    d["hop"] = d["hop"].astype(int)
    d = d[d["hop"] <= max_hop_show].copy()
    if len(d) == 0:
        raise ValueError("No nodes with hop values available for boxplot")
    hs = sorted(d["hop"].unique())
    groups = [np.clip(d.loc[d["hop"] == h, "score"].to_numpy(dtype=float), 1e-18, None) for h in hs]
    labels = [str(h) for h in hs]
    plt.figure(figsize=(8, 5.5))
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.yscale("log")
    plt.xlabel("Hop distance to nearest seed")
    plt.ylabel("PPR score (log scale)")
    plt.title("Score vs hop")
    plt.tight_layout()
    plt.savefig(outdir / "fig3_score_vs_hop_boxplot.png", dpi=300)
    plt.close()


def plot_top50_induced_with_bold_gene_labels(
    g: nx.Graph,
    nodes_df: pd.DataFrame,
    diff_df: pd.DataFrame,
    outdir: Path,
    topn: int = 50,
    layout_seed: int = 42,
):
    nodes_df = nodes_df.copy()
    nodes_df["entry"] = nodes_df["entry"].astype(str)

    d = diff_df[["node", "score"]].copy()
    d["node"] = d["node"].astype(str)
    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    d = d.dropna(subset=["score"]).sort_values("score", ascending=False).head(topn)

    top_nodes = d["node"].tolist()
    sg = g.subgraph(top_nodes).copy()
    if sg.number_of_nodes() == 0:
        raise ValueError("Empty induced subgraph for top nodes")

    score_map = dict(zip(d["node"], d["score"]))
    scores = np.array([float(score_map.get(n, 0.0)) for n in sg.nodes()], dtype=float)
    smax = float(np.max(scores)) if len(scores) else 1.0
    sizes = 120 + 2600 * (scores / smax if smax > 0 else scores)

    is_seed_map = dict(zip(nodes_df["entry"], nodes_df["is_seed"].astype(int)))
    colors = ["tab:red" if int(is_seed_map.get(n, 0)) == 1 else "tab:blue" for n in sg.nodes()]

    pos = nx.spring_layout(sg, seed=layout_seed, weight="weight")

    w = np.array([sg[u][v].get("weight", 1.0) for u, v in sg.edges()], dtype=float)
    if len(w) == 0:
        widths = []
    else:
        wmin, wmax = float(np.min(w)), float(np.max(w))
        widths = (0.3 + 2.7 * (w - wmin) / (wmax - wmin)) if wmax > wmin else np.full_like(w, 0.9, dtype=float)

    meta = nodes_df.set_index("entry", drop=False)
    labels = {}
    for n in sg.nodes():
        if n in meta.index:
            gs = meta.loc[n].get("gene_symbol", "")
            en = meta.loc[n].get("entry_name", "")
            if isinstance(gs, str) and gs.strip() and gs.lower() != "nan":
                labels[n] = gs.strip()
            elif isinstance(en, str) and en.strip() and en.lower() != "nan":
                labels[n] = en.strip()
            else:
                labels[n] = n
        else:
            labels[n] = n

    plt.figure(figsize=(11, 9))
    nx.draw_networkx_edges(sg, pos, alpha=0.35, width=widths)
    nx.draw_networkx_nodes(sg, pos, node_color=colors, node_size=sizes, linewidths=0.0)
    nx.draw_networkx_labels(sg, pos, labels=labels, font_size=9, font_weight="bold")
    plt.title(f"Top-{topn} induced subgraph (node size ∝ PPR score; bold gene labels)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outdir / "fig4_top50_induced_bold_labels.png", dpi=300)
    plt.close()


def plot_top50_induced_small_nodes_edges_shaded(
    g: nx.Graph,
    diff_df: pd.DataFrame,
    outdir: Path,
    topn: int = 50,
    layout_seed: int = 42,
):
    d = diff_df[["node", "score"]].copy()
    d["node"] = d["node"].astype(str)
    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    d = d.dropna(subset=["score"]).sort_values("score", ascending=False).head(topn)

    top_nodes = d["node"].tolist()
    sg = g.subgraph(top_nodes).copy()
    if sg.number_of_nodes() == 0:
        raise ValueError("Empty induced subgraph for top nodes")

    pos = nx.spring_layout(sg, seed=layout_seed, weight="weight")

    edges = list(sg.edges())
    ew = np.array([sg[u][v].get("weight", 1.0) for u, v in edges], dtype=float)
    if len(edges) > 0:
        emin, emax = float(np.min(ew)), float(np.max(ew))
        norm = (ew - emin) / (emax - emin) if emax > emin else np.zeros_like(ew)
        cmap = plt.cm.Blues
        ecolors = [cmap(0.25 + 0.70 * float(t)) for t in norm]
        order = np.argsort(ew)
        edges = [edges[i] for i in order]
        ecolors = [ecolors[i] for i in order]
        ewidths = [0.3 + 1.8 * float(norm[i]) for i in order]
    else:
        ecolors = []
        ewidths = []

    score_map = dict(zip(d["node"], d["score"]))
    ns = np.array([float(score_map.get(n, 0.0)) for n in sg.nodes()], dtype=float)
    smax = float(np.max(ns)) if len(ns) else 1.0
    node_colors = [plt.cm.Blues(0.25 + 0.70 * float(x / smax)) if smax > 0 else plt.cm.Blues(0.25) for x in ns]

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_edges(sg, pos, edgelist=edges, edge_color=ecolors, width=ewidths, alpha=0.9)
    nx.draw_networkx_nodes(sg, pos, node_color=node_colors, node_size=18, linewidths=0.0, alpha=0.95)
    plt.title(f"Top-{topn} induced subgraph (small nodes; edges shaded by weight)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outdir / "fig4b_top50_induced_small_nodes_edges_shaded.png", dpi=300)
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
        if isinstance(gs, str) and gs.strip() and gs.lower() != "nan":
            return gs.strip()
        if isinstance(en, str) and en.strip() and en.lower() != "nan":
            return en.strip()
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
            vv = v.fillna(0).astype(str).str.lower().isin(["1", "true", "t", "yes", "y"]).astype(int)
        else:
            vv = v.fillna(0).astype(float)
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
    d["degree"] = pd.to_numeric(d["degree"], errors="coerce")
    d["score"] = pd.to_numeric(d["score"], errors="coerce")
    d = d.dropna(subset=["degree", "score"])
    x = d["degree"].to_numpy(dtype=float)
    y = np.clip(d["score"].to_numpy(dtype=float), 1e-18, None)
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
    ap.add_argument("--max-hop-box", type=int, default=4)
    ap.add_argument("--top50", type=int, default=50)
    ap.add_argument("--topk-heatmap", type=int, default=50)
    ap.add_argument("--layout-seed", type=int, default=42)
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[2]
    nodes_path = Path(str(ROOT / args.nodes))
    edges_path = Path(str(ROOT / args.edges))
    diff_path = Path(str(ROOT / args.diffusion))
    graph_pkl = Path(str(ROOT / args.graph_pkl)) if args.graph_pkl else None
    outdir = Path(str(ROOT / args.outdir))
    ensure_outdir(outdir)

    nodes = pd.read_csv(nodes_path, sep="\t", dtype=str)
    nodes["entry"] = nodes["entry"].astype(str)
    if "is_seed" in nodes.columns:
        nodes["is_seed"] = pd.to_numeric(nodes["is_seed"], errors="coerce").fillna(0).astype(int)
    else:
        nodes["is_seed"] = 0

    diff = pd.read_csv(diff_path, dtype=str)
    diff["node"] = diff["node"].astype(str)
    diff["score"] = pd.to_numeric(diff["score"], errors="coerce")
    if "degree" in diff.columns:
        diff["degree"] = pd.to_numeric(diff["degree"], errors="coerce")
    else:
        diff["degree"] = np.nan

    g = load_graph(edges_path, graph_pkl)

    seeds = nodes.loc[nodes["is_seed"] == 1, "entry"].tolist()
    hops = multi_source_hops(g, seeds, cutoff=max(args.max_hop, args.max_hop_box))

    rank_map = build_rank_map(diff)

    plot_hop_subgraph_structure_rank_shaded(
        g=g,
        nodes_df=nodes,
        hops=hops,
        rank_map=rank_map,
        outdir=outdir,
        max_hop=args.max_hop,
        layout_seed=args.layout_seed,
    )
    plot_rank_score_curve(diff_df=diff, outdir=outdir)
    plot_score_vs_hop_boxplot(diff_df=diff, hops=hops, outdir=outdir, max_hop_show=args.max_hop_box)
    plot_top50_induced_with_bold_gene_labels(
        g=g,
        nodes_df=nodes,
        diff_df=diff,
        outdir=outdir,
        topn=args.top50,
        layout_seed=args.layout_seed,
    )
    plot_top50_induced_small_nodes_edges_shaded(
        g=g,
        diff_df=diff,
        outdir=outdir,
        topn=args.top50,
        layout_seed=args.layout_seed,
    )
    plot_topk_annotation_heatmap(nodes_df=nodes, diff_df=diff, outdir=outdir, topk=args.topk_heatmap)
    plot_ppr_vs_degree_scatter(diff_df=diff, outdir=outdir)

    print(f"[OK] wrote figures to: {outdir}")


if __name__ == "__main__":
    main()

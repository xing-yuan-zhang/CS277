# src/graph_constructor/build_graph.py
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import networkx as nx
import pickle


def clean_nan_for_graphml(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if pd.isna(v):
            out[k] = ""
        else:
            out[k] = v
    return out


def choose_seed_component(G: nx.Graph, seeds: set[str]) -> set[str] | None:
    seeds_in_graph = seeds.intersection(G.nodes)
    if not seeds_in_graph:
        return None

    best_comp = None
    best_count = -1
    best_size = -1

    for comp in nx.connected_components(G):
        comp_set = set(comp)
        c = len(comp_set.intersection(seeds_in_graph))
        if c > best_count or (c == best_count and len(comp_set) > best_size):
            best_count = c
            best_size = len(comp_set)
            best_comp = comp_set

    return best_comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default="inputs/ppi/subgraph/subgraph_edges.tsv")
    ap.add_argument("--nodes", default="inputs/ppi/subgraph/subgraph_node_attributes.tsv")
    ap.add_argument("--seeds", default="inputs/ppi/subgraph/subgraph_nodes.tsv",
                    help="need entry and is_seed columns from subgraph_nodes.tsv")
    ap.add_argument("--outdir", default="inputs/pkl")
    ap.add_argument("--keep", choices=["none", "largest", "seed"], default="seed",
                    help="none=connectivity filtering, no cropping; largest=retain the largest component; seed=retain the component containing the most seeds")
    args = ap.parse_args()

    ROOT = Path(__file__).resolve().parents[2]
    edges_path = str(ROOT / args.edges)
    nodes_path = Path(ROOT / args.nodes)
    seeds_path = Path(ROOT / args.seeds)
    outdir = Path(ROOT / args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    e = pd.read_csv(edges_path, sep="\t")
    n = pd.read_csv(nodes_path, sep="\t")

    for c in ["entry_a", "entry_b", "weight"]:
        if c not in e.columns:
            raise ValueError(f"{edges_path} need column {c}")
    if "entry" not in n.columns:
        raise ValueError(f"{nodes_path} need entry column")

    seeds_df = pd.read_csv(seeds_path, sep="\t")
    if not {"entry", "is_seed"}.issubset(set(seeds_df.columns)):
        raise ValueError(f"{seeds_path} need entry and is_seed columns")
    seeds = set(seeds_df.loc[seeds_df["is_seed"].astype(int) == 1, "entry"].astype(str).tolist())

    G = nx.Graph()

    for _, r in n.iterrows():
        d = r.to_dict()
        node = str(d.pop("entry"))
        d = clean_nan_for_graphml(d)
        G.add_node(node, **d)

    for _, r in e.iterrows():
        a = str(r["entry_a"])
        b = str(r["entry_b"])
        attrs = {
            "weight": float(r["weight"]),
            "is_string": int(r.get("is_string", 0)) if not pd.isna(r.get("is_string", 0)) else 0,
            "is_biogrid": int(r.get("is_biogrid", 0)) if not pd.isna(r.get("is_biogrid", 0)) else 0,
        }
        if "string_score" in e.columns and not pd.isna(r.get("string_score", pd.NA)):
            attrs["string_score"] = float(r["string_score"])
        G.add_edge(a, b, **attrs)

    num_cc = nx.number_connected_components(G)
    cc_sizes = sorted([len(c) for c in nx.connected_components(G)], reverse=True)
    largest_cc = cc_sizes[0] if cc_sizes else 0

    seeds_in_graph = len(seeds.intersection(G.nodes))

    stats_lines = []
    stats_lines.append(f"Graph BEFORE filtering: nodes={G.number_of_nodes():,} edges={G.number_of_edges():,}")
    stats_lines.append(f"Connected components={num_cc:,}; largest_component_size={largest_cc:,}")
    stats_lines.append(f"Seeds total={len(seeds):,}; seeds_in_graph={seeds_in_graph:,}")

    kept_nodes = None
    if args.keep == "largest" and G.number_of_nodes() > 0:
        comp = max(nx.connected_components(G), key=len)
        kept_nodes = set(comp)
        G = G.subgraph(kept_nodes).copy()
        stats_lines.append(f"KEEP=largest -> kept_nodes={len(kept_nodes):,}")
    elif args.keep == "seed":
        comp = choose_seed_component(G, seeds)
        if comp is not None:
            kept_nodes = comp
            G = G.subgraph(kept_nodes).copy()
            stats_lines.append(f"KEEP=seed -> kept_nodes={len(kept_nodes):,} (seed-component)")
        else:
            stats_lines.append("KEEP=seed -> no seeds found in graph; no filtering applied")

    num_cc2 = nx.number_connected_components(G) if G.number_of_nodes() > 0 else 0
    cc_sizes2 = sorted([len(c) for c in nx.connected_components(G)], reverse=True) if G.number_of_nodes() > 0 else []
    largest_cc2 = cc_sizes2[0] if cc_sizes2 else 0
    stats_lines.append(f"Graph AFTER filtering: nodes={G.number_of_nodes():,} edges={G.number_of_edges():,}")
    stats_lines.append(f"Connected components={num_cc2:,}; largest_component_size={largest_cc2:,}")
    stats_lines.append(f"Seeds_in_graph_after={len(seeds.intersection(G.nodes)):,}")

    nx.write_graphml(G, outdir / "ppi_subgraph.graphml")
    with open(outdir / "ppi_subgraph.pkl", "wb") as f:
        pickle.dump(G, f)


    final_nodes = pd.DataFrame({"entry": list(G.nodes)})
    final_nodes = final_nodes.merge(n, on="entry", how="left")
    final_edges = nx.to_pandas_edgelist(G)
    final_edges = final_edges.rename(columns={"source": "entry_a", "target": "entry_b"})
    final_nodes.to_csv(outdir / "nodes.final.tsv", sep="\t", index=False)
    final_edges.to_csv(outdir / "edges.final.tsv", sep="\t", index=False)

    (outdir / "graph.stats.txt").write_text("\n".join(stats_lines) + "\n")

    print(f"[OK] graph nodes={G.number_of_nodes():,} edges={G.number_of_edges():,}")
    print(f"[OK] wrote: {outdir/'ppi_subgraph.pkl'}")
    print(f"[OK] wrote: {outdir/'graph.stats.txt'}")


if __name__ == "__main__":
    main()

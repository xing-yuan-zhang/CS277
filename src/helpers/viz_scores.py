# src/helpers/viz_scores.py
from pathlib import Path
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PKL = str(ROOT / "inputs/pkl/ppi_subgraph.pkl")
SCORES_CSV = str(ROOT / "outputs/diffusion/diffusion_scores_alpha0.70.csv")
MAX_NODES = 1000

def main():
    with open(PKL, "rb") as f:
        G = pickle.load(f)

    if G.number_of_nodes() > MAX_NODES:
        comp = max(nx.connected_components(G), key=len)
        G = G.subgraph(list(comp)[:MAX_NODES]).copy()

    df = pd.read_csv(SCORES_CSV)
    df = df[["node", "score"]]
    df["node"] = df["node"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    df = df.sort_values("score", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    df["node_key"] = df["node"].str.split("-").str[0]
    rank_map = dict(zip(df["node_key"], df["rank"]))

    pos = nx.spring_layout(G, seed=42, k=0.5)

    cmap = plt.cm.Blues
    color_seed = "red"
    color_top50 = cmap(0.90)
    color_50_100 = cmap(0.70)
    color_100_200 = cmap(0.50)
    color_200_plus = cmap(0.30)
    color_unknown = "lightgray"

    node_colors = []
    for n, d in G.nodes(data=True):
        if int(d.get("is_seed", 0)) == 1:
            node_colors.append(color_seed)
        else:
            key = str(n).strip().split("-")[0]
            r = rank_map.get(key)
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

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=30, alpha=0.85)
    nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.25)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

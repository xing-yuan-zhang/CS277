# src/helpers/viz_vanilla.py
from pathlib import Path
import pickle
import networkx as nx
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PKL = str(ROOT / "inputs/pkl/ppi_subgraph.pkl")
MAX_NODES = 1000

def main():
    with open(PKL, "rb") as f:
        G = pickle.load(f)

    seed_nodes = [n for n, d in G.nodes(data=True) if d.get("is_seed", 0) == 1]
    print("Seeds in graph:", seed_nodes)
    print(f"[INFO] graph nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    if G.number_of_nodes() > MAX_NODES:
        comp = max(nx.connected_components(G), key=len)
        G = G.subgraph(list(comp)[:MAX_NODES]).copy()
        print(f"[INFO] visualizing subgraph nodes={G.number_of_nodes()}")

    pos = nx.spring_layout(G, seed=42, k=0.5)

    node_colors = []
    for n, d in G.nodes(data=True):
        if int(d.get("is_seed", 0)) == 1:
            node_colors.append("red")
        else:
            node_colors.append("lightgray")

    plt.figure(figsize=(10, 10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=30, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.3)

    plt.title("PPI subgraph (red = seeds)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

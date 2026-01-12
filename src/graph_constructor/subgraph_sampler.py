# src/graph_constructor/subgraph_sampler.py
import argparse
import heapq
from collections import defaultdict
from pathlib import Path
import pandas as pd


def build_keep_mask(chunk: pd.DataFrame, min_string_weight: float, biogrid_physical_only: bool) -> pd.Series:
    has_biogrid = "is_biogrid" in chunk.columns
    has_string = "is_string" in chunk.columns

    keep_string = pd.Series(False, index=chunk.index)
    if has_string:
        keep_string = (chunk["is_string"] == 1) & (chunk["weight"] >= min_string_weight)

    keep_biogrid = pd.Series(False, index=chunk.index)
    if has_biogrid:
        keep_biogrid = (chunk["is_biogrid"] == 1)
        if biogrid_physical_only and ("biogrid_physical" in chunk.columns):
            keep_biogrid = keep_biogrid & (chunk["biogrid_physical"] == 1)

    return keep_string | keep_biogrid


def _heap_push_topk(heap, item, k: int):
    if k <= 0:
        heap.append(item)
        return
    if len(heap) < k:
        heapq.heappush(heap, item)
    else:
        if item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)


def scan_collect_neighbors(
    edges_path: Path,
    active: set[str],
    min_string_weight: float,
    biogrid_physical_only: bool,
    topk_string_per_node: int,
    chunksize: int = 2_000_000,
) -> set[str]:
    neigh = set()
    heaps = defaultdict(list) if topk_string_per_node > 0 else None

    for chunk in pd.read_csv(edges_path, sep="\t", chunksize=chunksize):
        required = {"entry_a", "entry_b"}
        missing_cols = required - set(chunk.columns)
        if missing_cols:
            raise ValueError(f"edges file missing columns: {sorted(missing_cols)}")

        keep = build_keep_mask(chunk, min_string_weight, biogrid_physical_only)
        c = chunk[keep]
        if len(c) == 0:
            continue

        a = c["entry_a"].astype(str)
        b = c["entry_b"].astype(str)

        mask_a = a.isin(active)
        mask_b = b.isin(active)

        if "is_biogrid" in c.columns:
            cb = c[c["is_biogrid"] == 1]
            if len(cb) > 0:
                ab = cb["entry_a"].astype(str)
                bb = cb["entry_b"].astype(str)
                mb_a = ab.isin(active)
                mb_b = bb.isin(active)
                neigh.update(bb[mb_a].tolist())
                neigh.update(ab[mb_b].tolist())

        if "is_string" in c.columns:
            cs = c[c["is_string"] == 1]
            if len(cs) > 0:
                as_ = cs["entry_a"].astype(str)
                bs_ = cs["entry_b"].astype(str)
                ws_ = cs["weight"].astype(float)

                ms_a = as_.isin(active)
                ms_b = bs_.isin(active)

                if topk_string_per_node <= 0:
                    neigh.update(bs_[ms_a].tolist())
                    neigh.update(as_[ms_b].tolist())
                else:
                    for src, dst, w in zip(as_[ms_a], bs_[ms_a], ws_[ms_a]):
                        _heap_push_topk(heaps[src], (float(w), str(dst)), topk_string_per_node)
                    for src, dst, w in zip(bs_[ms_b], as_[ms_b], ws_[ms_b]):
                        _heap_push_topk(heaps[src], (float(w), str(dst)), topk_string_per_node)

    if topk_string_per_node > 0:
        for src, h in heaps.items():
            for w, dst in h:
                neigh.add(dst)

    return neigh


def scan_filter_edges(
    edges_path: Path,
    nodes: set[str],
    min_string_weight: float,
    biogrid_physical_only: bool,
    out_path: Path,
    chunksize: int = 2_000_000,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrote = 0
    first = True

    for chunk in pd.read_csv(edges_path, sep="\t", chunksize=chunksize):
        required = {"entry_a", "entry_b"}
        missing_cols = required - set(chunk.columns)
        if missing_cols:
            raise ValueError(f"edges file missing columns: {sorted(missing_cols)}")

        keep = build_keep_mask(chunk, min_string_weight, biogrid_physical_only)
        c = chunk[keep]
        if len(c) == 0:
            continue

        a = c["entry_a"].astype(str)
        b = c["entry_b"].astype(str)
        sub = c[a.isin(nodes) & b.isin(nodes)].copy()

        if len(sub) == 0:
            continue

        sub.to_csv(out_path, sep="\t", index=False, mode="w" if first else "a", header=first)
        first = False
        wrote += len(sub)

    return wrote


def pick_seed_set(seeds_df: pd.DataFrame, seed_roles: list[str], allow_missing_role: bool) -> set[str]:
    if "entry" not in seeds_df.columns:
        raise ValueError("seeds.mapped.tsv need entry column")

    seeds_df = seeds_df.copy()
    seeds_df["entry"] = seeds_df["entry"].astype(str).str.strip()

    if "role" not in seeds_df.columns:
        if allow_missing_role:
            return set(seeds_df.dropna(subset=["entry"])["entry"].tolist())
        raise ValueError("seeds.mapped.tsv has no role column; pass --allow-missing-role or add role")

    seeds_df["role"] = seeds_df["role"].astype(str).str.strip()

    keep_df = seeds_df[seeds_df["role"].isin(seed_roles)].dropna(subset=["entry"])
    return set(keep_df["entry"].tolist())


def main():
    ROOT = Path(__file__).resolve().parents[2]

    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default=str(ROOT / "inputs/ppi/edges.merged.uniprot.tsv"))
    ap.add_argument("--seeds", default=str(ROOT / "inputs/seeds/seeds.mapped.tsv"))

    ap.add_argument("--hops", type=int, default=1, choices=[1, 2])

    ap.add_argument(
        "--min-string-weight",
        type=float,
        default=0.90,
        help="STRING edge weight=combined_score/1000 threshold; recommended 0.85~0.95",
    )

    ap.add_argument(
        "--biogrid-physical-only",
        action="store_true",
        default=False,
        help="If set, only keep BioGRID edges with biogrid_physical==1 (recommended).",
    )

    ap.add_argument(
        "--topk-string-per-node",
        type=int,
        default=50,
        help="Keep only top-K STRING neighbors per active node when expanding hops. 0 disables top-K.",
    )

    ap.add_argument("--max-nodes", type=int, default=5000)

    ap.add_argument("--outdir", default=str(ROOT / "inputs/ppi/subgraph"))

    ap.add_argument(
        "--seed-roles",
        default="Src-family-kinase,CIP4-TOCA-family,FBP17,TOCA-1",
        help="Comma-separated roles to USE as diffusion seeds (exclude hubs like downstream_signal by default).",
    )

    ap.add_argument(
        "--allow-missing-role",
        action="store_true",
        help="If seeds file has no role column, allow using all entries as seeds.",
    )

    ap.add_argument(
        "--patch-seed",
        default=None,
        help="Optional: do local patch expansion for this seed (UniProt). Empty disables.",
    )

    ap.add_argument(
        "--patch-hops",
        type=int,
        default=2,
        choices=[1, 2],
        help="Local patch hops for --patch-seed (default 2).",
    )

    ap.add_argument(
        "--patch-min-string-weight",
        type=float,
        default=None,
        help="Override min-string-weight for patch only. None uses --min-string-weight.",
    )

    ap.add_argument(
        "--patch-topk-string-per-node",
        type=int,
        default=None,
        help="Override topk-string-per-node for patch only. None uses --topk-string-per-node.",
    )

    args = ap.parse_args()

    edges_path = Path(args.edges)
    seeds_path = Path(args.seeds)
    outdir = Path(args.outdir)

    seeds_df = pd.read_csv(seeds_path, sep="\t")

    seed_roles = [x.strip() for x in str(args.seed_roles).split(",") if x.strip()]
    seed_set = pick_seed_set(seeds_df, seed_roles=seed_roles, allow_missing_role=args.allow_missing_role)

    seed_set = {x for x in seed_set if x and x.lower() != "nan"}

    if len(seed_set) == 0:
        raise ValueError("No usable seed entries after role filtering. Check --seed-roles or seeds.mapped.tsv.")

    n1 = scan_collect_neighbors(
        edges_path=edges_path,
        active=seed_set,
        min_string_weight=args.min_string_weight,
        biogrid_physical_only=args.biogrid_physical_only,
        topk_string_per_node=args.topk_string_per_node,
    )
    nodes = set(seed_set) | set(n1)

    if args.hops == 2:
        n2 = scan_collect_neighbors(
            edges_path=edges_path,
            active=nodes,
            min_string_weight=args.min_string_weight,
            biogrid_physical_only=args.biogrid_physical_only,
            topk_string_per_node=args.topk_string_per_node,
        )
        nodes |= set(n2)

    # local patch for a specific seed with fixed 2-hop expansion
    patch_seed = str(args.patch_seed).strip()
    if patch_seed and patch_seed.lower() != "none":
        patch_minw = args.min_string_weight if args.patch_min_string_weight is None else float(args.patch_min_string_weight)
        patch_topk = args.topk_string_per_node if args.patch_topk_string_per_node is None else int(args.patch_topk_string_per_node)

        # patch only if the seed is in the original seed set and is biologically validated
        if patch_seed in seed_set:
            p1 = scan_collect_neighbors(
                edges_path=edges_path,
                active={patch_seed},
                min_string_weight=patch_minw,
                biogrid_physical_only=args.biogrid_physical_only,
                topk_string_per_node=patch_topk,
            )
            patch_nodes = {patch_seed} | set(p1)

            print(f"[PATCH-DEBUG] p1 neighbors count={len(p1)} example={list(sorted(p1))[:10]}")

            if args.patch_hops == 2:
                p2 = scan_collect_neighbors(
                    edges_path=edges_path,
                    active=patch_nodes,
                    min_string_weight=patch_minw,
                    biogrid_physical_only=args.biogrid_physical_only,
                    topk_string_per_node=patch_topk,
                )
                patch_nodes |= set(p2)

            before = len(nodes)
            nodes |= patch_nodes
            after = len(nodes)
            print(
                f"[PATCH] seed={patch_seed} patch_hops={args.patch_hops} "
                f"minw={patch_minw} topk={patch_topk} added_nodes={after-before:,}"
            )
        else:
            print(f"[PATCH] seed={patch_seed} not in seed_set; skip patch. (If desired, remove this check.)")

    if len(nodes) > args.max_nodes:
        raise ValueError(
            f"Node number {len(nodes)} exceeds max_nodes={args.max_nodes}. "
            f"Try: increase --min-string-weight, set --hops 1, or reduce --topk-string-per-node."
        )

    outdir.mkdir(parents=True, exist_ok=True)
    out_nodes = outdir / "subgraph_nodes.tsv"
    out_edges = outdir / "subgraph_edges.tsv"

    nodes_sorted = sorted(nodes)
    pd.DataFrame(
        {"entry": nodes_sorted, "is_seed": [int(x in seed_set) for x in nodes_sorted]}
    ).to_csv(out_nodes, sep="\t", index=False)

    n_edges = scan_filter_edges(
        edges_path=edges_path,
        nodes=nodes,
        min_string_weight=args.min_string_weight,
        biogrid_physical_only=args.biogrid_physical_only,
        out_path=out_edges,
    )

    print(f"[OK] wrote {out_nodes} nodes={len(nodes):,} seeds={len(seed_set):,}")
    print(f"[OK] wrote {out_edges} edges={n_edges:,}")
    print(f"[INFO] hops={args.hops} min_string_weight={args.min_string_weight} topk_string_per_node={args.topk_string_per_node} biogrid_physical_only={args.biogrid_physical_only}")
    print(f"[INFO] seed_roles={seed_roles}")


if __name__ == "__main__":
    main()

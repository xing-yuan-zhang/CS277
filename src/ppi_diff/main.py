from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

from baselines import compute_degree_scores, evaluate_degree_matched_seed_baseline, evaluate_random_baseline
from build_graph import build_weighted_graph, graph_stats
from data_utils import read_ms_hits, read_string_edges
from metrics import aggregate_numeric_rows, compute_ranking_metrics
from results import maybe_generate_plots
from ppr import rank_nodes_by_score, run_personalized_pagerank
from split_eval import make_split_assignment, write_split_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal STRING + PPR holdout workflow for recovering MS hits."
    )
    parser.add_argument("--string-path", required=True, help="Path to the STRING edge table.")
    parser.add_argument("--ms-hits-path", required=True, help="Path to the MS hits table.")
    parser.add_argument("--output-dir", required=True, help="Directory for workflow outputs.")

    parser.add_argument("--protein1-col", default="protein1")
    parser.add_argument("--protein2-col", default="protein2")
    parser.add_argument("--score-col", default="combined_score")

    parser.add_argument("--protein-col", default="protein_id")
    parser.add_argument("--high-conf-col", default=None)
    parser.add_argument(
        "--high-conf-values",
        default="1,true,yes,high,hc,hit",
        help="Comma-separated truthy values used when --high-conf-col is provided.",
    )
    parser.add_argument("--bait-col", default=None)
    parser.add_argument("--force-baits-in-seed", action="store_true")
    parser.add_argument("--uppercase-ids", action="store_true")

    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--ks", default="10,20,50,100,200")
    parser.add_argument("--degree-mode", choices=("weighted", "unweighted"), default="weighted")
    parser.add_argument("--random-baseline-repeats", type=int, default=100)
    parser.add_argument("--matched-seed-repeats", type=int, default=10)
    parser.add_argument("--matched-candidate-pool-size", type=int, default=25)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def parse_k_values(raw_k_values: str) -> list[int]:
    k_values = []
    for raw_value in raw_k_values.split(","):
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        parsed = int(raw_value)
        if parsed <= 0:
            raise ValueError("All k values must be positive integers.")
        k_values.append(parsed)
    if not k_values:
        raise ValueError("At least one k value is required.")
    return sorted(set(k_values))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_cell(row.get(key)) for key in fieldnames})


def _format_cell(value: object) -> object:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value


def build_score_rows(
    node_ids: list[str],
    scores: list[float],
    ranked: list[tuple[int, float]],
    seed_indices: set[int],
    train_seed_indices: set[int],
    test_indices: set[int],
    bait_indices: set[int],
    score_column: str,
) -> list[dict[str, object]]:
    rank_map = {node_idx: rank for rank, (node_idx, _) in enumerate(ranked, start=1)}
    rows: list[dict[str, object]] = []
    for node_idx, protein_id in enumerate(node_ids):
        rows.append(
            {
                "protein_id": protein_id,
                score_column: scores[node_idx],
                "rank": rank_map.get(node_idx, ""),
                "is_seed": 1 if node_idx in seed_indices else 0,
                "is_train_seed": 1 if node_idx in train_seed_indices else 0,
                "is_test_hit": 1 if node_idx in test_indices else 0,
                "is_bait": 1 if node_idx in bait_indices else 0,
            }
        )
    return rows


def build_metric_row(
    split_index: int,
    split_seed: int,
    method: str,
    evaluation_summary: dict[str, float],
    num_hits_total: int,
    num_hits_in_graph: int,
    num_train_hits: int,
    num_test_hits: int,
    num_seeds: int,
    extra_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "split_index": split_index,
        "split_seed": split_seed,
        "method": method,
        "num_hits_total": num_hits_total,
        "num_hits_in_graph": num_hits_in_graph,
        "num_train_hits": num_train_hits,
        "num_test_hits": num_test_hits,
        "num_seeds": num_seeds,
    }
    row.update(evaluation_summary)
    if extra_fields:
        row.update(extra_fields)
    return row


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    k_values = parse_k_values(args.ks)

    string_data = read_string_edges(
        path=args.string_path,
        protein1_col=args.protein1_col,
        protein2_col=args.protein2_col,
        score_col=args.score_col,
        uppercase_ids=args.uppercase_ids,
    )
    hit_data = read_ms_hits(
        path=args.ms_hits_path,
        protein_col=args.protein_col,
        high_conf_col=args.high_conf_col,
        high_conf_values=[value.strip() for value in args.high_conf_values.split(",") if value.strip()],
        bait_col=args.bait_col,
        uppercase_ids=args.uppercase_ids,
    )

    graph = build_weighted_graph(string_data.edges)
    node_id_set = set(graph.idx_to_node)
    hit_ids_in_graph = set(hit_data.high_conf_hits) & node_id_set
    bait_ids_in_graph = set(hit_data.baits) & node_id_set

    stats = graph_stats(
        graph=graph,
        edge_weights=[weight for _, _, weight in string_data.edges],
        cleaning_stats=string_data.cleaning_stats,
        normalization=string_data.normalization,
    )
    stats["hit_summary"] = {
        "num_high_conf_hits_total": len(hit_data.high_conf_hits),
        "num_high_conf_hits_in_graph": len(hit_ids_in_graph),
        "num_high_conf_hits_missing_from_graph": len(hit_data.high_conf_hits - node_id_set),
        "num_baits_total": len(hit_data.baits),
        "num_baits_in_graph": len(bait_ids_in_graph),
    }
    stats["run_config"] = {
        "alpha": args.alpha,
        "tol": args.tol,
        "max_iter": args.max_iter,
        "train_fraction": args.train_fraction,
        "repeats": args.repeats,
        "random_baseline_repeats": args.random_baseline_repeats,
        "matched_seed_repeats": args.matched_seed_repeats,
        "ks": k_values,
        "degree_mode": args.degree_mode,
        "force_baits_in_seed": args.force_baits_in_seed,
    }
    write_json(output_dir / "graph_stats.json", stats)

    if len(hit_ids_in_graph) == 0:
        raise ValueError("No high-confidence MS hits overlap the STRING graph.")
    if len(hit_ids_in_graph) == 1 and not args.force_baits_in_seed:
        raise ValueError("At least two graph-overlapping hits are needed for a non-trivial holdout split.")

    degree_scores = compute_degree_scores(graph, weighted=(args.degree_mode == "weighted"))
    null_eligible_indices = [
        idx
        for idx, protein_id in enumerate(graph.idx_to_node)
        if protein_id not in hit_ids_in_graph and protein_id not in bait_ids_in_graph
    ]

    metrics_rows: list[dict[str, object]] = []
    first_split_pr_curves: dict[str, list[tuple[float, float]]] = {}
    first_split_score_distribution: dict[str, list[float]] | None = None

    for split_index in range(args.repeats):
        split_seed = args.random_seed + split_index
        split = make_split_assignment(
            hit_ids_in_graph=hit_ids_in_graph,
            bait_ids_in_graph=bait_ids_in_graph,
            split_index=split_index,
            split_seed=split_seed,
            train_fraction=args.train_fraction,
            force_baits_in_seed=args.force_baits_in_seed,
        )

        write_split_table(output_dir / f"split_{split_index:03d}.tsv", graph, split)

        seed_indices = {graph.node_to_idx[protein_id] for protein_id in split.seed_nodes}
        train_seed_indices = {graph.node_to_idx[protein_id] for protein_id in split.train_hits}
        test_indices = {graph.node_to_idx[protein_id] for protein_id in split.test_hits}
        bait_indices = {graph.node_to_idx[protein_id] for protein_id in split.bait_nodes}

        ppr_run = run_personalized_pagerank(
            graph=graph,
            seed_indices=seed_indices,
            alpha=args.alpha,
            tol=args.tol,
            max_iter=args.max_iter,
        )
        ppr_ranked = rank_nodes_by_score(graph, ppr_run.scores, seed_indices)
        ppr_ranked_indices = [node_idx for node_idx, _ in ppr_ranked]
        ppr_evaluation = compute_ranking_metrics(ppr_ranked_indices, test_indices, k_values)

        ppr_rows = build_score_rows(
            node_ids=graph.idx_to_node,
            scores=ppr_run.scores,
            ranked=ppr_ranked,
            seed_indices=seed_indices,
            train_seed_indices=train_seed_indices,
            test_indices=test_indices,
            bait_indices=bait_indices,
            score_column="ppr_score",
        )
        write_tsv(output_dir / f"ppr_scores_{split_index:03d}.tsv", ppr_rows)

        metrics_rows.append(
            build_metric_row(
                split_index=split_index,
                split_seed=split_seed,
                method="ppr",
                evaluation_summary=ppr_evaluation.summary,
                num_hits_total=len(hit_data.high_conf_hits),
                num_hits_in_graph=len(hit_ids_in_graph),
                num_train_hits=len(split.train_hits),
                num_test_hits=len(split.test_hits),
                num_seeds=len(seed_indices),
                extra_fields={
                    "ppr_iterations": ppr_run.iterations,
                    "ppr_residual": ppr_run.residual,
                    "ppr_converged": 1 if ppr_run.converged else 0,
                },
            )
        )

        degree_ranked = rank_nodes_by_score(graph, degree_scores, seed_indices)
        degree_ranked_indices = [node_idx for node_idx, _ in degree_ranked]
        degree_evaluation = compute_ranking_metrics(degree_ranked_indices, test_indices, k_values)

        degree_rows = build_score_rows(
            node_ids=graph.idx_to_node,
            scores=degree_scores,
            ranked=degree_ranked,
            seed_indices=seed_indices,
            train_seed_indices=train_seed_indices,
            test_indices=test_indices,
            bait_indices=bait_indices,
            score_column="degree_score",
        )
        write_tsv(output_dir / f"degree_scores_{split_index:03d}.tsv", degree_rows)

        metrics_rows.append(
            build_metric_row(
                split_index=split_index,
                split_seed=split_seed,
                method=f"{args.degree_mode}_degree",
                evaluation_summary=degree_evaluation.summary,
                num_hits_total=len(hit_data.high_conf_hits),
                num_hits_in_graph=len(hit_ids_in_graph),
                num_train_hits=len(split.train_hits),
                num_test_hits=len(split.test_hits),
                num_seeds=len(seed_indices),
                extra_fields=None,
            )
        )

        random_summary = evaluate_random_baseline(
            candidate_indices=ppr_ranked_indices,
            positive_indices=test_indices,
            k_values=k_values,
            repeats=args.random_baseline_repeats,
            rng=random.Random(split_seed + 10_000_000),
        )
        if random_summary:
            metrics_rows.append(
                build_metric_row(
                    split_index=split_index,
                    split_seed=split_seed,
                    method="random",
                    evaluation_summary=random_summary,
                    num_hits_total=len(hit_data.high_conf_hits),
                    num_hits_in_graph=len(hit_ids_in_graph),
                    num_train_hits=len(split.train_hits),
                    num_test_hits=len(split.test_hits),
                    num_seeds=len(seed_indices),
                    extra_fields={"random_repeats": args.random_baseline_repeats},
                )
            )

        matched_summary = evaluate_degree_matched_seed_baseline(
            graph=graph,
            real_seed_indices=sorted(seed_indices),
            eligible_indices=null_eligible_indices,
            positive_indices=test_indices,
            k_values=k_values,
            repeats=args.matched_seed_repeats,
            alpha=args.alpha,
            tol=args.tol,
            max_iter=args.max_iter,
            rng=random.Random(split_seed + 20_000_000),
            candidate_pool_size=args.matched_candidate_pool_size,
        )
        if matched_summary is not None:
            metrics_rows.append(
                build_metric_row(
                    split_index=split_index,
                    split_seed=split_seed,
                    method="matched_random_seed_ppr",
                    evaluation_summary=matched_summary,
                    num_hits_total=len(hit_data.high_conf_hits),
                    num_hits_in_graph=len(hit_ids_in_graph),
                    num_train_hits=len(split.train_hits),
                    num_test_hits=len(split.test_hits),
                    num_seeds=len(seed_indices),
                    extra_fields={"matched_seed_repeats_requested": args.matched_seed_repeats},
                )
            )

        if split_index == 0:
            first_split_pr_curves = {
                "ppr": ppr_evaluation.pr_curve,
                f"{args.degree_mode}_degree": degree_evaluation.pr_curve,
            }
            positive_score_set = set(test_indices)
            first_split_score_distribution = {
                "test_hits": [ppr_run.scores[idx] for idx in ppr_ranked_indices if idx in positive_score_set],
                "other_candidates": [ppr_run.scores[idx] for idx in ppr_ranked_indices if idx not in positive_score_set],
            }

    write_tsv(output_dir / "metrics_summary.tsv", metrics_rows)
    aggregate_rows = aggregate_numeric_rows(
        metrics_rows,
        group_keys=["method"],
        exclude_keys=["split_index", "split_seed"],
    )
    write_tsv(output_dir / "metrics_aggregate.tsv", aggregate_rows)

    plot_status = {"enabled": False, "reason": "Skipped by --skip-plots."}
    if not args.skip_plots:
        plot_status = maybe_generate_plots(
            output_dir=output_dir,
            aggregate_rows=aggregate_rows,
            pr_curves=first_split_pr_curves,
            score_distribution=first_split_score_distribution,
            k_values=k_values,
        )
    write_json(output_dir / "plot_status.json", plot_status)


if __name__ == "__main__":
    main()

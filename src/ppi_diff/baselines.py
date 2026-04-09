from __future__ import annotations

import random

from build_graph import WeightedUndirectedGraph
from metrics import compute_ranking_metrics
from ppr import rank_nodes_by_score, run_personalized_pagerank
from split_eval import sample_degree_matched_seed_indices


def compute_degree_scores(graph: WeightedUndirectedGraph, weighted: bool = True) -> list[float]:
    return list(graph.weighted_degrees if weighted else graph.degrees)


def _mean_metric_dicts(metric_runs: list[dict[str, float]]) -> dict[str, float]:
    if not metric_runs:
        return {}
    numeric_keys = sorted(metric_runs[0].keys())
    averaged: dict[str, float] = {}
    for key in numeric_keys:
        values = [run[key] for run in metric_runs]
        averaged[key] = sum(values) / len(values)
    averaged["null_repeats_completed"] = float(len(metric_runs))
    return averaged


def evaluate_random_baseline(
    candidate_indices: list[int],
    positive_indices: set[int],
    k_values: list[int],
    repeats: int,
    rng: random.Random,
) -> dict[str, float]:
    if repeats <= 0:
        return {}

    metric_runs: list[dict[str, float]] = []
    for _ in range(repeats):
        shuffled = list(candidate_indices)
        rng.shuffle(shuffled)
        evaluation = compute_ranking_metrics(shuffled, positive_indices, k_values)
        metric_runs.append(evaluation.summary)
    return _mean_metric_dicts(metric_runs)


def evaluate_degree_matched_seed_baseline(
    graph: WeightedUndirectedGraph,
    real_seed_indices: list[int],
    eligible_indices: list[int],
    positive_indices: set[int],
    k_values: list[int],
    repeats: int,
    alpha: float,
    tol: float,
    max_iter: int,
    rng: random.Random,
    candidate_pool_size: int = 25,
) -> dict[str, float] | None:
    if repeats <= 0:
        return None

    metric_runs: list[dict[str, float]] = []
    for _ in range(repeats):
        matched_seed_indices = sample_degree_matched_seed_indices(
            graph=graph,
            real_seed_indices=real_seed_indices,
            eligible_indices=eligible_indices,
            rng=rng,
            candidate_pool_size=candidate_pool_size,
        )
        if matched_seed_indices is None:
            continue

        ppr_run = run_personalized_pagerank(
            graph=graph,
            seed_indices=set(matched_seed_indices),
            alpha=alpha,
            tol=tol,
            max_iter=max_iter,
        )
        ranked = rank_nodes_by_score(graph, ppr_run.scores, set(matched_seed_indices))
        ranked_indices = [idx for idx, _ in ranked]
        evaluation = compute_ranking_metrics(ranked_indices, positive_indices, k_values)
        metric_runs.append(evaluation.summary)

    if not metric_runs:
        return None
    return _mean_metric_dicts(metric_runs)

from __future__ import annotations

from dataclasses import dataclass

from build_graph import WeightedUndirectedGraph


@dataclass
class PPRRun:
    scores: list[float]
    iterations: int
    residual: float
    converged: bool


def run_personalized_pagerank(
    graph: WeightedUndirectedGraph,
    seed_indices: set[int] | list[int],
    alpha: float = 0.85,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> PPRRun:
    seeds = list(dict.fromkeys(seed_indices))
    if not seeds:
        raise ValueError("Seed set is empty. PPR requires at least one seed node.")

    seed_weight = 1.0 / len(seeds)
    num_nodes = graph.num_nodes
    scores = [0.0] * num_nodes
    for seed_idx in seeds:
        scores[seed_idx] = seed_weight

    residual = float("inf")
    converged = False

    for iteration in range(1, max_iter + 1):
        next_scores = [0.0] * num_nodes
        dangling_mass = 0.0

        for node_idx, node_score in enumerate(scores):
            weighted_degree = graph.weighted_degrees[node_idx]
            if weighted_degree <= 0.0:
                dangling_mass += node_score
                continue
            contribution_scale = alpha * node_score / weighted_degree
            for neighbor_idx, weight in graph.neighbors[node_idx]:
                next_scores[neighbor_idx] += contribution_scale * weight

        teleport_mass = (1.0 - alpha) + (alpha * dangling_mass)
        seed_contribution = teleport_mass * seed_weight
        for seed_idx in seeds:
            next_scores[seed_idx] += seed_contribution

        residual = sum(abs(next_scores[node_idx] - scores[node_idx]) for node_idx in range(num_nodes))
        scores = next_scores
        if residual <= tol:
            converged = True
            return PPRRun(scores=scores, iterations=iteration, residual=residual, converged=converged)

    return PPRRun(scores=scores, iterations=max_iter, residual=residual, converged=converged)


def rank_nodes_by_score(
    graph: WeightedUndirectedGraph,
    scores: list[float],
    excluded_indices: set[int] | list[int],
) -> list[tuple[int, float]]:
    excluded = set(excluded_indices)
    ranked = [(idx, scores[idx]) for idx in range(graph.num_nodes) if idx not in excluded]
    ranked.sort(key=lambda item: (-item[1], graph.idx_to_node[item[0]]))
    return ranked

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass
class RankingEvaluation:
    summary: dict[str, float]
    pr_curve: list[tuple[float, float]]


def compute_ranking_metrics(
    ranked_indices: list[int],
    positive_indices: set[int],
    k_values: list[int],
) -> RankingEvaluation:
    positives = set(positive_indices)
    num_candidates = len(ranked_indices)
    num_positives = sum(1 for idx in ranked_indices if idx in positives)
    sorted_k_values = sorted(set(k_values))

    summary: dict[str, float] = {
        "num_candidates": float(num_candidates),
        "num_positives": float(num_positives),
    }

    cumulative_hits = 0
    hits_at_rank: dict[int, int] = {}
    ap_numerator = 0.0
    pr_curve: list[tuple[float, float]] = [(0.0, 1.0)] if num_positives > 0 else []

    for rank, node_idx in enumerate(ranked_indices, start=1):
        if node_idx in positives:
            cumulative_hits += 1
            precision = cumulative_hits / rank
            recall = cumulative_hits / num_positives if num_positives else float("nan")
            ap_numerator += precision
            pr_curve.append((recall, precision))
        hits_at_rank[rank] = cumulative_hits

    summary["auprc"] = (ap_numerator / num_positives) if num_positives else float("nan")
    summary["positive_rate"] = (num_positives / num_candidates) if num_candidates and num_positives else float("nan")

    for k in sorted_k_values:
        cutoff = min(k, num_candidates)
        hits = hits_at_rank.get(cutoff, cumulative_hits if cutoff == num_candidates else 0)
        recall = hits / num_positives if num_positives else float("nan")
        precision = hits / cutoff if cutoff else float("nan")
        if cutoff and num_positives and num_candidates:
            enrichment = (hits / cutoff) / (num_positives / num_candidates)
        else:
            enrichment = float("nan")
        summary[f"recall_at_{k}"] = recall
        summary[f"precision_at_{k}"] = precision
        summary[f"enrichment_at_{k}"] = enrichment

    return RankingEvaluation(summary=summary, pr_curve=pr_curve)


def aggregate_numeric_rows(
    rows: list[dict[str, object]],
    group_keys: list[str],
    exclude_keys: list[str] | None = None,
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    excluded = set(exclude_keys or [])
    for row in rows:
        group_value = tuple(row[key] for key in group_keys)
        grouped.setdefault(group_value, []).append(row)

    aggregated_rows: list[dict[str, object]] = []
    for group_value, group_rows in sorted(grouped.items()):
        aggregate_row: dict[str, object] = {group_keys[idx]: group_value[idx] for idx in range(len(group_keys))}
        aggregate_row["n_rows"] = len(group_rows)

        numeric_keys: set[str] = set()
        for row in group_rows:
            for key, value in row.items():
                if key in group_keys or key in excluded:
                    continue
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    numeric_keys.add(key)

        for key in sorted(numeric_keys):
            values = [float(row[key]) for row in group_rows if isinstance(row.get(key), (int, float))]
            finite_values = [value for value in values if not math.isnan(value)]
            if not finite_values:
                aggregate_row[f"{key}_mean"] = float("nan")
                aggregate_row[f"{key}_std"] = float("nan")
                continue
            aggregate_row[f"{key}_mean"] = statistics.fmean(finite_values)
            aggregate_row[f"{key}_std"] = statistics.stdev(finite_values) if len(finite_values) > 1 else 0.0

        aggregated_rows.append(aggregate_row)

    return aggregated_rows

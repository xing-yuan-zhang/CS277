from __future__ import annotations

import bisect
import random
from dataclasses import dataclass
from pathlib import Path

from build_graph import WeightedUndirectedGraph


@dataclass
class SplitAssignment:
    split_index: int
    split_seed: int
    train_hits: set[str]
    test_hits: set[str]
    seed_nodes: set[str]
    bait_nodes: set[str]


def split_train_test_hits(
    hit_ids_in_graph: set[str],
    bait_ids_in_graph: set[str],
    train_fraction: float,
    rng: random.Random,
    force_baits_in_seed: bool,
) -> tuple[set[str], set[str], set[str]]:
    fixed_train_hits = set(bait_ids_in_graph & hit_ids_in_graph) if force_baits_in_seed else set()
    splittable_hits = sorted(hit_ids_in_graph - fixed_train_hits)
    rng.shuffle(splittable_hits)

    total = len(splittable_hits)
    if total == 0:
        random_train_hits: set[str] = set()
        random_test_hits: set[str] = set()
    elif total == 1:
        random_train_hits = {splittable_hits[0]}
        random_test_hits = set()
    else:
        proposed_train = int(round(total * train_fraction))
        train_size = max(1, min(total - 1, proposed_train))
        random_train_hits = set(splittable_hits[:train_size])
        random_test_hits = set(splittable_hits[train_size:])

    train_hits = fixed_train_hits | random_train_hits
    test_hits = random_test_hits
    seed_nodes = set(train_hits)
    if force_baits_in_seed:
        seed_nodes |= bait_ids_in_graph
    return train_hits, test_hits, seed_nodes


def make_split_assignment(
    hit_ids_in_graph: set[str],
    bait_ids_in_graph: set[str],
    split_index: int,
    split_seed: int,
    train_fraction: float,
    force_baits_in_seed: bool,
) -> SplitAssignment:
    rng = random.Random(split_seed)
    train_hits, test_hits, seed_nodes = split_train_test_hits(
        hit_ids_in_graph=hit_ids_in_graph,
        bait_ids_in_graph=bait_ids_in_graph,
        train_fraction=train_fraction,
        rng=rng,
        force_baits_in_seed=force_baits_in_seed,
    )
    return SplitAssignment(
        split_index=split_index,
        split_seed=split_seed,
        train_hits=train_hits,
        test_hits=test_hits,
        seed_nodes=seed_nodes,
        bait_nodes=set(bait_ids_in_graph),
    )


def write_split_table(path: str | Path, graph: WeightedUndirectedGraph, split: SplitAssignment) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("protein_id\tsplit_label\tis_bait\tis_seed\n")
        for protein_id in graph.idx_to_node:
            if protein_id in split.train_hits:
                split_label = "train_hit"
            elif protein_id in split.test_hits:
                split_label = "test_hit"
            else:
                split_label = "other"
            is_bait = 1 if protein_id in split.bait_nodes else 0
            is_seed = 1 if protein_id in split.seed_nodes else 0
            handle.write(f"{protein_id}\t{split_label}\t{is_bait}\t{is_seed}\n")


def sample_degree_matched_seed_indices(
    graph: WeightedUndirectedGraph,
    real_seed_indices: list[int],
    eligible_indices: list[int],
    rng: random.Random,
    candidate_pool_size: int = 25,
) -> list[int] | None:
    unique_real_seeds = list(dict.fromkeys(real_seed_indices))
    unique_eligible = sorted(set(eligible_indices))
    if len(unique_real_seeds) == 0:
        return []
    if len(unique_eligible) < len(unique_real_seeds):
        return None

    degree_pairs = sorted((graph.weighted_degrees[idx], idx) for idx in unique_eligible)
    sorted_degrees = [degree for degree, _ in degree_pairs]
    chosen_indices: list[int] = []
    used_indices: set[int] = set()
    shuffled_real_seeds = list(unique_real_seeds)
    rng.shuffle(shuffled_real_seeds)

    for seed_idx in shuffled_real_seeds:
        target_degree = graph.weighted_degrees[seed_idx]
        insertion_point = bisect.bisect_left(sorted_degrees, target_degree)
        local_candidates: list[int] = []
        left = insertion_point - 1
        right = insertion_point

        while len(local_candidates) < candidate_pool_size and (left >= 0 or right < len(degree_pairs)):
            left_distance = abs(sorted_degrees[left] - target_degree) if left >= 0 else float("inf")
            right_distance = abs(sorted_degrees[right] - target_degree) if right < len(degree_pairs) else float("inf")

            if left_distance <= right_distance:
                candidate_idx = degree_pairs[left][1]
                left -= 1
            else:
                candidate_idx = degree_pairs[right][1]
                right += 1

            if candidate_idx not in used_indices:
                local_candidates.append(candidate_idx)

        if not local_candidates:
            remaining = [candidate_idx for candidate_idx in unique_eligible if candidate_idx not in used_indices]
            if not remaining:
                return None
            chosen_idx = rng.choice(remaining)
        else:
            chosen_idx = rng.choice(local_candidates)

        chosen_indices.append(chosen_idx)
        used_indices.add(chosen_idx)

    return chosen_indices

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_TRUTHY = {
    "1",
    "true",
    "t",
    "yes",
    "y",
    "high",
    "hc",
    "hit",
}


@dataclass
class StringData:
    edges: list[tuple[str, str, float]]
    normalization: dict[str, object]
    cleaning_stats: dict[str, int]


@dataclass
class HitData:
    high_conf_hits: set[str]
    baits: set[str]
    stats: dict[str, int]


def normalize_protein_id(value: object, uppercase: bool = False) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return normalized.upper() if uppercase else normalized


def split_multi_value_field(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    for separator in (";", "|", ","):
        if separator in text:
            return [part.strip() for part in text.split(separator) if part.strip()]
    return [text]


def _infer_delimiter(path: Path) -> str:
    if path.suffix.lower() == ".csv":
        return ","
    if path.suffix.lower() in {".tsv", ".txt"}:
        return "\t"

    sample = path.read_text(encoding="utf-8-sig")[:4096]
    if not sample:
        return "\t"
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        return dialect.delimiter
    except csv.Error:
        return "\t" if "\t" in sample else ","


def _iter_rows(path: str | Path) -> Iterable[dict[str, str]]:
    table_path = Path(path)
    delimiter = _infer_delimiter(table_path)
    with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"{table_path} does not contain a header row.")
        for row in reader:
            yield row


def _normalize_edge_score(raw_score: float, raw_min: float, raw_max: float) -> tuple[float, str]:
    if 0.0 <= raw_min and raw_max <= 1.0:
        return raw_score, "identity_[0,1]"
    if 0.0 <= raw_min and raw_max <= 1000.0:
        return raw_score / 1000.0, "divide_by_1000"
    if raw_max > raw_min:
        return (raw_score - raw_min) / (raw_max - raw_min), "min_max"
    if raw_score > 0.0:
        return 1.0, "constant_positive"
    return 0.0, "constant_zero"


def read_string_edges(
    path: str | Path,
    protein1_col: str = "protein1",
    protein2_col: str = "protein2",
    score_col: str = "combined_score",
    uppercase_ids: bool = False,
) -> StringData:
    raw_edge_map: dict[tuple[str, str], float] = {}
    raw_rows = 0
    self_loops_removed = 0
    rows_missing_id = 0
    rows_missing_score = 0
    invalid_score_rows = 0
    duplicate_edges_collapsed = 0

    for row in _iter_rows(path):
        raw_rows += 1
        protein1 = normalize_protein_id(row.get(protein1_col), uppercase=uppercase_ids)
        protein2 = normalize_protein_id(row.get(protein2_col), uppercase=uppercase_ids)
        if protein1 is None or protein2 is None:
            rows_missing_id += 1
            continue
        if protein1 == protein2:
            self_loops_removed += 1
            continue

        raw_score_value = row.get(score_col)
        if raw_score_value is None or str(raw_score_value).strip() == "":
            rows_missing_score += 1
            continue
        try:
            raw_score = float(raw_score_value)
        except ValueError:
            invalid_score_rows += 1
            continue

        edge_key = tuple(sorted((protein1, protein2)))
        if edge_key in raw_edge_map:
            duplicate_edges_collapsed += 1
            raw_edge_map[edge_key] = max(raw_edge_map[edge_key], raw_score)
        else:
            raw_edge_map[edge_key] = raw_score

    if not raw_edge_map:
        raise ValueError("No valid STRING edges were loaded after cleaning.")

    deduplicated_raw_scores = list(raw_edge_map.values())
    raw_min = min(deduplicated_raw_scores)
    raw_max = max(deduplicated_raw_scores)

    edges: list[tuple[str, str, float]] = []
    normalization_method = "identity_[0,1]"
    for (protein1, protein2), raw_score in raw_edge_map.items():
        normalized_score, normalization_method = _normalize_edge_score(raw_score, raw_min, raw_max)
        normalized_score = max(0.0, min(1.0, normalized_score))
        edges.append((protein1, protein2, normalized_score))

    return StringData(
        edges=edges,
        normalization={
            "method": normalization_method,
            "raw_min": raw_min,
            "raw_max": raw_max,
            "normalized_min": min(score for _, _, score in edges),
            "normalized_max": max(score for _, _, score in edges),
        },
        cleaning_stats={
            "raw_rows": raw_rows,
            "valid_edges_after_cleaning": len(edges),
            "self_loops_removed": self_loops_removed,
            "duplicate_edges_collapsed": duplicate_edges_collapsed,
            "rows_missing_id": rows_missing_id,
            "rows_missing_score": rows_missing_score,
            "invalid_score_rows": invalid_score_rows,
        },
    )


def _is_high_conf_value(value: object, truthy_values: set[str]) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    if text.lower() in truthy_values:
        return True
    try:
        return float(text) != 0.0
    except ValueError:
        return False


def read_ms_hits(
    path: str | Path,
    protein_col: str = "protein_id",
    high_conf_col: str | None = None,
    high_conf_values: Iterable[str] | None = None,
    bait_col: str | None = None,
    uppercase_ids: bool = False,
) -> HitData:
    truthy_values = {value.strip().lower() for value in (high_conf_values or DEFAULT_TRUTHY) if value.strip()}
    total_rows = 0
    rows_missing_protein = 0
    high_conf_hits: set[str] = set()
    baits: set[str] = set()

    for row in _iter_rows(path):
        total_rows += 1
        protein_id = normalize_protein_id(row.get(protein_col), uppercase=uppercase_ids)
        if protein_id is None:
            rows_missing_protein += 1
            continue

        if high_conf_col is None or _is_high_conf_value(row.get(high_conf_col), truthy_values):
            high_conf_hits.add(protein_id)

        if bait_col:
            for bait_value in split_multi_value_field(row.get(bait_col)):
                bait_id = normalize_protein_id(bait_value, uppercase=uppercase_ids)
                if bait_id is not None:
                    baits.add(bait_id)

    return HitData(
        high_conf_hits=high_conf_hits,
        baits=baits,
        stats={
            "raw_rows": total_rows,
            "rows_missing_protein": rows_missing_protein,
            "high_conf_hit_count": len(high_conf_hits),
            "bait_count": len(baits),
        },
    )

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import fmean, median


@dataclass
class WeightedUndirectedGraph:
    node_to_idx: dict[str, int]
    idx_to_node: list[str]
    neighbors: list[list[tuple[int, float]]]
    degrees: list[int]
    weighted_degrees: list[float]

    @property
    def num_nodes(self) -> int:
        return len(self.idx_to_node)

    @property
    def num_edges(self) -> int:
        return sum(len(adjacent) for adjacent in self.neighbors) // 2


def build_weighted_graph(edges: list[tuple[str, str, float]]) -> WeightedUndirectedGraph:
    nodes = sorted({protein for edge in edges for protein in edge[:2]})
    node_to_idx = {node: index for index, node in enumerate(nodes)}
    neighbors: list[list[tuple[int, float]]] = [[] for _ in nodes]
    degrees = [0 for _ in nodes]
    weighted_degrees = [0.0 for _ in nodes]

    for protein1, protein2, weight in edges:
        idx1 = node_to_idx[protein1]
        idx2 = node_to_idx[protein2]
        neighbors[idx1].append((idx2, weight))
        neighbors[idx2].append((idx1, weight))
        degrees[idx1] += 1
        degrees[idx2] += 1
        weighted_degrees[idx1] += weight
        weighted_degrees[idx2] += weight

    return WeightedUndirectedGraph(
        node_to_idx=node_to_idx,
        idx_to_node=nodes,
        neighbors=neighbors,
        degrees=degrees,
        weighted_degrees=weighted_degrees,
    )


def _summary(values: list[float | int]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    numeric_values = [float(value) for value in values]
    return {
        "min": min(numeric_values),
        "max": max(numeric_values),
        "mean": fmean(numeric_values),
        "median": float(median(numeric_values)),
    }


def connected_component_sizes(graph: WeightedUndirectedGraph) -> list[int]:
    visited = [False] * graph.num_nodes
    component_sizes: list[int] = []

    for start_idx in range(graph.num_nodes):
        if visited[start_idx]:
            continue
        queue = deque([start_idx])
        visited[start_idx] = True
        component_size = 0
        while queue:
            node_idx = queue.popleft()
            component_size += 1
            for neighbor_idx, _ in graph.neighbors[node_idx]:
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    queue.append(neighbor_idx)
        component_sizes.append(component_size)

    component_sizes.sort(reverse=True)
    return component_sizes


def graph_stats(
    graph: WeightedUndirectedGraph,
    edge_weights: list[float],
    cleaning_stats: dict[str, int],
    normalization: dict[str, object],
) -> dict[str, object]:
    component_sizes = connected_component_sizes(graph)
    return {
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "edge_weight_summary": _summary(edge_weights),
        "degree_summary": {
            "degree": _summary(graph.degrees),
            "weighted_degree": _summary(graph.weighted_degrees),
        },
        "connected_components": {
            "count": len(component_sizes),
            "largest_size": component_sizes[0] if component_sizes else 0,
            "sizes_top_10": component_sizes[:10],
        },
        "normalization": normalization,
        "cleaning": cleaning_stats,
    }

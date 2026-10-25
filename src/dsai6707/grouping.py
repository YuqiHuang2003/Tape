"""Deterministic expert grouping by clustering or capacity-two matching."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

CAPACITY_TWO_MATCHING_METHOD = "capacity_two_minimum_weight_matching_output_cosine_v1"
CAPACITY_TWO_MATCHING_NETWORKX_VERSION = "3.4.2"


@dataclass(frozen=True)
class LayerGrouping:
    """A validated expert-to-group assignment for one MoE layer."""

    layer: int
    expert_to_group: tuple[int, ...]
    groups: tuple[tuple[int, ...], ...]

    @classmethod
    def from_assignment(cls, layer: int, assignment: Sequence[int]) -> LayerGrouping:
        if layer < 0:
            raise ValueError("layer must be non-negative")
        if not assignment:
            raise ValueError("assignment must contain at least one expert")
        raw = [int(value) for value in assignment]
        if any(value < 0 for value in raw):
            raise ValueError("group identifiers must be non-negative")

        # Normalize arbitrary labels by first appearance. This gives stable, contiguous ids.
        remap: dict[int, int] = {}
        normalized: list[int] = []
        for value in raw:
            if value not in remap:
                remap[value] = len(remap)
            normalized.append(remap[value])

        members: list[list[int]] = [[] for _ in range(len(remap))]
        for expert, group in enumerate(normalized):
            members[group].append(expert)
        return cls(
            layer=int(layer),
            expert_to_group=tuple(normalized),
            groups=tuple(tuple(group) for group in members),
        )

    @property
    def num_experts(self) -> int:
        return len(self.expert_to_group)

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    @property
    def degrees_of_freedom(self) -> int:
        return self.num_experts - self.num_groups

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "num_experts": self.num_experts,
            "num_groups": self.num_groups,
            "degrees_of_freedom": self.degrees_of_freedom,
            "expert_to_group": list(self.expert_to_group),
            "groups": [list(group) for group in self.groups],
        }


@dataclass(frozen=True)
class GroupingPlan:
    """Fixed grouping for every layer plus provenance needed for reproduction."""

    layers: tuple[LayerGrouping, ...]
    model_id: str
    model_revision: str
    method: str = "average_linkage_output_cosine"
    calibration_fingerprint: str = ""
    seed: int = 0
    schema_version: str = "grouping-v1"

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("grouping plan must contain at least one layer")
        expected = list(range(len(self.layers)))
        actual = [layer.layer for layer in self.layers]
        if actual != expected:
            raise ValueError(
                f"layers must be ordered and contiguous: expected {expected}, got {actual}"
            )
        expert_counts = {layer.num_experts for layer in self.layers}
        if len(expert_counts) != 1:
            raise ValueError("all layers must have the same number of source experts")
        if not self.model_id or not self.model_revision:
            raise ValueError("model_id and model_revision must be recorded")
        if self.method == CAPACITY_TWO_MATCHING_METHOD:
            oversized = [
                (layer.layer, group)
                for layer in self.layers
                for group in layer.groups
                if len(group) > 2
            ]
            if oversized:
                raise ValueError(
                    "capacity-two grouping contains a group with more than two experts"
                )

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def num_experts(self) -> int:
        return self.layers[0].num_experts

    @property
    def total_groups(self) -> int:
        return sum(layer.num_groups for layer in self.layers)

    @property
    def dimension(self) -> int:
        return sum(layer.degrees_of_freedom for layer in self.layers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model": {"id": self.model_id, "revision": self.model_revision},
            "method": self.method,
            "calibration_fingerprint": self.calibration_fingerprint,
            "seed": self.seed,
            "summary": {
                "num_layers": self.num_layers,
                "num_experts_per_layer": self.num_experts,
                "total_groups": self.total_groups,
                "search_dimension": self.dimension,
            },
            "layers": [layer.to_dict() for layer in self.layers],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GroupingPlan:
        if payload.get("schema_version") != "grouping-v1":
            raise ValueError("unsupported or missing grouping schema_version")
        model = payload.get("model", {})
        layers = tuple(
            LayerGrouping.from_assignment(item["layer"], item["expert_to_group"])
            for item in payload["layers"]
        )
        return cls(
            layers=layers,
            model_id=str(model["id"]),
            model_revision=str(model["revision"]),
            method=str(payload.get("method", "average_linkage_output_cosine")),
            calibration_fingerprint=str(payload.get("calibration_fingerprint", "")),
            seed=int(payload.get("seed", 0)),
        )


def cosine_distance_matrix(signatures: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Return a symmetric cosine-distance matrix for rows of ``signatures``."""

    values = np.asarray(signatures, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("signatures must have shape [experts, features]")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= epsilon):
        raise ValueError("expert signatures must have non-zero norm")
    normalized = values / norms
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    return (distance + distance.T) / 2.0


def _validated_distance_matrix(distances: np.ndarray) -> np.ndarray:
    """Return a symmetric non-negative float64 distance matrix."""

    values = np.asarray(distances, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("distances must be a square matrix")
    if not np.isfinite(values).all() or np.any(values < -1e-12):
        raise ValueError("distances must be finite and non-negative")
    if not np.allclose(values, values.T, atol=1e-10):
        raise ValueError("distances must be symmetric")
    values = np.maximum((values + values.T) / 2.0, 0.0)
    np.fill_diagonal(values, 0.0)
    return values


def average_linkage_from_distance(distances: np.ndarray, target_groups: int) -> list[int]:
    """Cluster a precomputed distance matrix with deterministic average linkage.

    With only 32 experts per Granite layer, a transparent O(E^3) implementation is
    preferable to making the experiment depend on a changing third-party clustering API.
    Ties are broken lexicographically by cluster membership.
    """

    distances = _validated_distance_matrix(distances)
    num_experts = distances.shape[0]
    if not 1 <= int(target_groups) <= num_experts:
        raise ValueError("target_groups must be between 1 and the number of experts")

    clusters: list[tuple[int, ...]] = [(expert,) for expert in range(num_experts)]
    while len(clusters) > target_groups:
        best: tuple[float, tuple[int, ...], tuple[int, ...], int, int] | None = None
        for left in range(len(clusters) - 1):
            for right in range(left + 1, len(clusters)):
                left_members = clusters[left]
                right_members = clusters[right]
                score = float(distances[np.ix_(left_members, right_members)].mean())
                candidate = (score, left_members, right_members, left, right)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        assert best is not None
        _, left_members, right_members, left, right = best
        merged = tuple(sorted(left_members + right_members))
        clusters = [cluster for index, cluster in enumerate(clusters) if index not in (left, right)]
        clusters.append(merged)
        clusters.sort()

    clusters.sort(key=lambda members: members[0])
    assignment = [-1] * num_experts
    for group, members in enumerate(clusters):
        for expert in members:
            assignment[expert] = group
    return assignment


def _exact_float64_edge_weights(
    distances: np.ndarray,
) -> tuple[list[tuple[int, int, int]], int]:
    """Encode float64 edge weights as exact integers plus a unique tie objective.

    Every finite float64 has a power-of-two denominator. Scaling all edges to the
    largest denominator therefore preserves the primary sum exactly. A separate
    binary bit per lexicographically ordered edge makes every selected edge set
    unique; multiplying the primary integer by ``2 ** edge_count`` keeps that
    deterministic secondary objective strictly subordinate to one primary unit.
    """

    num_experts = distances.shape[0]
    raw_edges: list[tuple[int, int, int, int]] = []
    common_denominator = 1
    for left in range(num_experts - 1):
        for right in range(left + 1, num_experts):
            numerator, denominator = float(distances[left, right]).as_integer_ratio()
            raw_edges.append((left, right, numerator, denominator))
            common_denominator = max(common_denominator, denominator)

    edge_count = len(raw_edges)
    primary_base = 1 << edge_count
    encoded: list[tuple[int, int, int]] = []
    for edge_rank, (left, right, numerator, denominator) in enumerate(raw_edges):
        if common_denominator % denominator != 0:
            raise RuntimeError("float64 denominators must share a binary multiple")
        primary = numerator * (common_denominator // denominator)
        encoded.append((left, right, primary * primary_base + (1 << edge_rank)))
    return encoded, common_denominator


def capacity_two_matching_from_distance(distances: np.ndarray, target_groups: int) -> list[int]:
    """Return an exact fixed-cardinality minimum-distance capacity-two grouping.

    For ``E`` experts and ``G`` target groups, the grouping must contain exactly
    ``E - G`` disjoint pairs and ``2G - E`` singletons. Singleton dummy vertices
    reduce this fixed-cardinality problem to a minimum-weight perfect matching.
    NetworkX's blossom implementation solves the integer-weight objective exactly.
    The returned identifiers are canonicalized by the first expert in each group.
    """

    distances = _validated_distance_matrix(distances)
    num_experts = distances.shape[0]
    if num_experts < 2:
        raise ValueError("capacity-two matching requires at least two experts")
    if type(target_groups) is not int:
        raise ValueError("target_groups must be an integer")
    minimum_groups = (num_experts + 1) // 2
    if not minimum_groups <= target_groups <= num_experts:
        raise ValueError("capacity-two target_groups must be between ceil(experts/2) and experts")

    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - exercised in minimal installations
        raise RuntimeError(
            "capacity-two matching requires networkx==" f"{CAPACITY_TWO_MATCHING_NETWORKX_VERSION}"
        ) from exc
    if nx.__version__ != CAPACITY_TWO_MATCHING_NETWORKX_VERSION:
        raise RuntimeError(
            "capacity-two matching requires networkx=="
            f"{CAPACITY_TWO_MATCHING_NETWORKX_VERSION}, got {nx.__version__}"
        )

    encoded_edges, _ = _exact_float64_edge_weights(distances)
    graph = nx.Graph()
    graph.add_nodes_from(range(num_experts))
    graph.add_weighted_edges_from(encoded_edges)

    singleton_count = 2 * target_groups - num_experts
    dummy_nodes = tuple(range(num_experts, num_experts + singleton_count))
    graph.add_nodes_from(dummy_nodes)
    graph.add_weighted_edges_from(
        (expert, dummy, 0) for expert in range(num_experts) for dummy in dummy_nodes
    )

    matching = nx.min_weight_matching(graph, weight="weight")
    if len(matching) * 2 != graph.number_of_nodes():
        raise RuntimeError("capacity-two reduction did not produce a perfect matching")

    pairs: list[tuple[int, int]] = []
    matched_singletons: set[int] = set()
    for first, second in matching:
        if first >= num_experts and second >= num_experts:
            raise RuntimeError("invalid dummy-to-dummy edge in capacity-two matching")
        if first < num_experts and second < num_experts:
            pairs.append(tuple(sorted((int(first), int(second)))))
        else:
            matched_singletons.add(int(first if first < num_experts else second))

    expected_pairs = num_experts - target_groups
    if len(pairs) != expected_pairs or len(matched_singletons) != singleton_count:
        raise RuntimeError("capacity-two matching has an invalid pair/singleton count")
    paired_experts = {expert for pair in pairs for expert in pair}
    if paired_experts & matched_singletons:
        raise RuntimeError("capacity-two matching assigned an expert more than once")
    if paired_experts | matched_singletons != set(range(num_experts)):
        raise RuntimeError("capacity-two matching did not cover every expert")

    partner = {left: right for left, right in pairs}
    partner.update({right: left for left, right in pairs})
    assignment = [-1] * num_experts
    group = 0
    for expert in range(num_experts):
        if assignment[expert] >= 0:
            continue
        assignment[expert] = group
        if expert in partner:
            assignment[partner[expert]] = group
        group += 1
    if group != target_groups:
        raise RuntimeError("capacity-two assignment does not have the target group count")
    return assignment


def capacity_two_matching_assignment(signatures: np.ndarray, target_groups: int) -> list[int]:
    """Group signatures using exact capacity-two cosine-distance matching."""

    return capacity_two_matching_from_distance(cosine_distance_matrix(signatures), target_groups)


def average_linkage_assignment(signatures: np.ndarray, target_groups: int) -> list[int]:
    """Cluster expert signatures using cosine distance and average linkage."""

    return average_linkage_from_distance(cosine_distance_matrix(signatures), target_groups)


def plan_from_signatures(
    signatures_by_layer: Iterable[np.ndarray],
    target_groups: int | Sequence[int],
    *,
    model_id: str,
    model_revision: str,
    calibration_fingerprint: str,
    seed: int,
) -> GroupingPlan:
    signatures = list(signatures_by_layer)
    if isinstance(target_groups, int):
        targets = [int(target_groups)] * len(signatures)
    else:
        targets = [int(value) for value in target_groups]
    if len(targets) != len(signatures):
        raise ValueError("one target group count is required per layer")
    layers = tuple(
        LayerGrouping.from_assignment(layer, average_linkage_assignment(values, targets[layer]))
        for layer, values in enumerate(signatures)
    )
    return GroupingPlan(
        layers=layers,
        model_id=model_id,
        model_revision=model_revision,
        calibration_fingerprint=calibration_fingerprint,
        seed=seed,
    )


def plan_from_similarities(
    similarities_by_layer: Iterable[np.ndarray],
    target_groups: int | Sequence[int],
    *,
    model_id: str,
    model_revision: str,
    calibration_fingerprint: str,
    seed: int,
) -> GroupingPlan:
    """Build a plan from exact pairwise cosine similarities of expert outputs."""

    similarities = [np.asarray(values, dtype=np.float64) for values in similarities_by_layer]
    if isinstance(target_groups, int):
        targets = [int(target_groups)] * len(similarities)
    else:
        targets = [int(value) for value in target_groups]
    if len(targets) != len(similarities):
        raise ValueError("one target group count is required per layer")
    layers = tuple(
        LayerGrouping.from_assignment(
            layer,
            average_linkage_from_distance(1.0 - values, targets[layer]),
        )
        for layer, values in enumerate(similarities)
    )
    return GroupingPlan(
        layers=layers,
        model_id=model_id,
        model_revision=model_revision,
        calibration_fingerprint=calibration_fingerprint,
        seed=seed,
    )


def capacity_two_plan_from_similarities(
    similarities_by_layer: Iterable[np.ndarray],
    target_groups: int | Sequence[int],
    *,
    model_id: str,
    model_revision: str,
    calibration_fingerprint: str,
    seed: int,
) -> GroupingPlan:
    """Build a capacity-two plan from pairwise output cosine similarities."""

    similarities = [np.asarray(values, dtype=np.float64) for values in similarities_by_layer]
    if isinstance(target_groups, int):
        targets = [int(target_groups)] * len(similarities)
    else:
        targets = [int(value) for value in target_groups]
    if len(targets) != len(similarities):
        raise ValueError("one target group count is required per layer")
    layers = tuple(
        LayerGrouping.from_assignment(
            layer,
            capacity_two_matching_from_distance(1.0 - values, targets[layer]),
        )
        for layer, values in enumerate(similarities)
    )
    return GroupingPlan(
        layers=layers,
        model_id=model_id,
        model_revision=model_revision,
        method=CAPACITY_TWO_MATCHING_METHOD,
        calibration_fingerprint=calibration_fingerprint,
        seed=seed,
    )

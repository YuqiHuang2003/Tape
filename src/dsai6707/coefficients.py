"""Simplex-safe parameterization of per-group fusion coefficients."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .grouping import GroupingPlan


@dataclass(frozen=True)
class GroupSlice:
    layer: int
    group: int
    members: tuple[int, ...]
    start: int
    stop: int

    @property
    def size(self) -> int:
        return len(self.members)


class CoefficientLayout:
    """Map an unconstrained vector to all group simplexes.

    A group with ``k`` experts uses ``k - 1`` free logits. A fixed zero is
    appended as the final logit before softmax, making the representation
    identifiable. Singleton coefficients are always exactly one.
    """

    def __init__(self, plan: GroupingPlan):
        self.plan = plan
        slices: list[GroupSlice] = []
        layer_ranges: list[tuple[int, int]] = []
        cursor = 0
        for layer in plan.layers:
            layer_start = cursor
            for group, members in enumerate(layer.groups):
                width = max(0, len(members) - 1)
                slices.append(GroupSlice(layer.layer, group, members, cursor, cursor + width))
                cursor += width
            layer_ranges.append((layer_start, cursor))
        self.slices = tuple(slices)
        self.layer_ranges = tuple(layer_ranges)
        self.dimension = cursor
        if self.dimension != plan.dimension:
            raise AssertionError("coefficient layout dimension disagrees with grouping plan")

    def decode(self, vector: Sequence[float] | np.ndarray) -> dict[int, dict[int, np.ndarray]]:
        values = np.asarray(vector, dtype=np.float64)
        if values.shape != (self.dimension,):
            raise ValueError(f"expected vector shape ({self.dimension},), got {values.shape}")
        decoded: dict[int, dict[int, np.ndarray]] = {
            layer.layer: {} for layer in self.plan.layers
        }
        for item in self.slices:
            if item.size == 1:
                weights = np.ones(1, dtype=np.float64)
            else:
                logits = np.concatenate((values[item.start : item.stop], np.zeros(1)))
                logits -= logits.max()
                weights = np.exp(logits)
                weights /= weights.sum()
            decoded[item.layer][item.group] = weights
        return decoded

    def encode(self, coefficients: Mapping[int, Mapping[int, Sequence[float]]]) -> np.ndarray:
        vector = np.empty(self.dimension, dtype=np.float64)
        for item in self.slices:
            weights = np.asarray(coefficients[item.layer][item.group], dtype=np.float64)
            if weights.shape != (item.size,):
                raise ValueError(
                    f"layer {item.layer} group {item.group} expects {item.size} coefficients"
                )
            if np.any(weights <= 0) or not np.isfinite(weights).all():
                raise ValueError(
                    "all coefficients must be finite and strictly positive for encoding"
                )
            weights = weights / weights.sum()
            if item.size > 1:
                vector[item.start : item.stop] = np.log(weights[:-1]) - np.log(weights[-1])
        return vector

    def layer_indices(self, layer: int) -> np.ndarray:
        start, stop = self.layer_ranges[layer]
        return np.arange(start, stop, dtype=np.int64)

    def to_artifact(
        self,
        vector: Sequence[float] | np.ndarray,
        *,
        method: str,
        seed: int | None,
        fitness: float | None = None,
    ) -> dict[str, Any]:
        decoded = self.decode(vector)
        layers: list[dict[str, Any]] = []
        for layer in self.plan.layers:
            groups: list[dict[str, Any]] = []
            for group, members in enumerate(layer.groups):
                groups.append(
                    {
                        "group": group,
                        "members": list(members),
                        "weights": decoded[layer.layer][group].tolist(),
                    }
                )
            layers.append({"layer": layer.layer, "groups": groups})
        return {
            "schema_version": "coefficients-v1",
            "model": {"id": self.plan.model_id, "revision": self.plan.model_revision},
            "method": method,
            "seed": seed,
            "fitness_nll": fitness,
            "dimension": self.dimension,
            "parameterization": "anchored_softmax_last_logit_zero",
            "vector": np.asarray(vector, dtype=float).tolist(),
            "layers": layers,
        }


def frequency_coefficients(
    plan: GroupingPlan,
    counts_by_layer: Sequence[Sequence[float]] | np.ndarray,
    *,
    smoothing: float = 1e-6,
) -> dict[int, dict[int, np.ndarray]]:
    """Normalize routing counts inside each fixed group."""

    counts = np.asarray(counts_by_layer, dtype=np.float64)
    expected = (plan.num_layers, plan.num_experts)
    if counts.shape != expected:
        raise ValueError(f"expected routing counts with shape {expected}, got {counts.shape}")
    if smoothing <= 0:
        raise ValueError("smoothing must be positive so coefficients remain encodable")
    if np.any(counts < 0) or not np.isfinite(counts).all():
        raise ValueError("routing counts must be finite and non-negative")

    output: dict[int, dict[int, np.ndarray]] = {}
    for layer in plan.layers:
        output[layer.layer] = {}
        for group, members in enumerate(layer.groups):
            values = counts[layer.layer, list(members)] + smoothing
            output[layer.layer][group] = values / values.sum()
    return output


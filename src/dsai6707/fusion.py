"""Backend-neutral expert fusion and compression accounting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .grouping import LayerGrouping


def fuse_expert_tensor(
    source: np.ndarray,
    grouping: LayerGrouping,
    coefficients: Mapping[int, Sequence[float]],
) -> np.ndarray:
    """Fuse the first (expert) axis of a weight tensor."""

    weights = np.asarray(source)
    if weights.ndim < 2 or weights.shape[0] != grouping.num_experts:
        raise ValueError("source must have expert id as its first axis")
    fused: list[np.ndarray] = []
    for group, members in enumerate(grouping.groups):
        alpha = np.asarray(coefficients[group], dtype=np.float64)
        if alpha.shape != (len(members),):
            raise ValueError(f"group {group} coefficient count does not match its members")
        if np.any(alpha < 0) or not np.isclose(alpha.sum(), 1.0, atol=1e-8):
            raise ValueError("fusion coefficients must be non-negative and sum to one")
        fused.append(np.tensordot(alpha, weights[list(members)], axes=(0, 0)))
    return np.stack(fused).astype(weights.dtype, copy=False)


def parameter_compression(
    *,
    non_expert_parameters: int,
    expert_parameters_per_expert: int,
    groups_per_layer: Sequence[int],
    source_experts_per_layer: int,
) -> dict[str, float | int]:
    """Compute actual whole-model and expert-only parameter compression."""

    if non_expert_parameters < 0 or expert_parameters_per_expert <= 0:
        raise ValueError("parameter counts must be non-negative")
    if source_experts_per_layer <= 0 or not groups_per_layer:
        raise ValueError("model must contain at least one source expert layer")
    if any(not 1 <= groups <= source_experts_per_layer for groups in groups_per_layer):
        raise ValueError("invalid group count")
    original_expert = (
        expert_parameters_per_expert * source_experts_per_layer * len(groups_per_layer)
    )
    fused_expert = expert_parameters_per_expert * sum(groups_per_layer)
    original_total = non_expert_parameters + original_expert
    fused_total = non_expert_parameters + fused_expert
    return {
        "original_total_parameters": original_total,
        "fused_total_parameters": fused_total,
        "original_expert_parameters": original_expert,
        "fused_expert_parameters": fused_expert,
        "whole_model_reduction_fraction": 1.0 - fused_total / original_total,
        "expert_reduction_fraction": 1.0 - fused_expert / original_expert,
    }


"""Reference implementation of original-router to fused-expert routing."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def aggregate_group_routes(
    top_experts: np.ndarray,
    top_weights: np.ndarray,
    expert_to_group: Sequence[int],
) -> np.ndarray:
    """Sum top-k routing weights that map to the same physical fused expert.

    The original router and its top-k decision are preserved. If two selected
    source experts belong to the same group, the fused expert is evaluated once
    and receives the sum of their routing weights.
    """

    experts = np.asarray(top_experts)
    weights = np.asarray(top_weights, dtype=np.float64)
    mapping = np.asarray(expert_to_group, dtype=np.int64)
    if experts.ndim != 2 or experts.shape != weights.shape:
        raise ValueError("top_experts and top_weights must have the same [tokens, top_k] shape")
    if mapping.ndim != 1 or mapping.size == 0:
        raise ValueError("expert_to_group must be a non-empty vector")
    if np.any(experts < 0) or np.any(experts >= mapping.size):
        raise ValueError("top_experts contains an out-of-range expert id")
    if np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("top_weights must be finite and non-negative")
    if np.any(mapping < 0):
        raise ValueError("expert_to_group contains a negative group id")

    num_groups = int(mapping.max()) + 1
    output = np.zeros((experts.shape[0], num_groups), dtype=np.float64)
    mapped = mapping[experts]
    rows = np.repeat(np.arange(experts.shape[0]), experts.shape[1])
    np.add.at(output, (rows, mapped.reshape(-1)), weights.reshape(-1))
    return output


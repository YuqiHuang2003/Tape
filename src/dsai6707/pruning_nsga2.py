"""Deterministic NSGA-II primitives for fixed-cardinality pruning masks.

The module deliberately contains no model or objective code.  It operates on
boolean masks with shape ``[layers, experts]`` and on dense objective arrays,
so an experiment driver can journal and resume expensive inference separately.
All objectives are minimized.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256

import numpy as np


@dataclass(frozen=True)
class LayerSwap:
    """One fixed-cardinality replacement made within a layer."""

    layer: int
    removed_expert: int
    added_expert: int


@dataclass(frozen=True)
class MultiLayerSwapMutation:
    """A mutated mask and the exact changes needed for a trial journal."""

    mask: np.ndarray
    swaps: tuple[LayerSwap, ...]

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(swap.layer for swap in self.swaps)


def validate_fixed_cardinality_mask(
    mask: np.ndarray,
    *,
    retained_count: int | None = None,
) -> int:
    """Validate a boolean ``[layers, experts]`` mask and return its cardinality."""

    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise ValueError("mask must be a NumPy boolean array")
    if mask.ndim != 2 or mask.shape[0] < 1 or mask.shape[1] < 2:
        raise ValueError("mask must have shape [layers, experts]")

    counts = np.count_nonzero(mask, axis=1)
    expected = int(counts[0]) if retained_count is None else retained_count
    if not isinstance(expected, int | np.integer):
        raise ValueError("retained_count must be an integer")
    expected = int(expected)
    if not 1 <= expected < mask.shape[1]:
        raise ValueError("retained_count must be in [1, num_experts)")
    if not np.all(counts == expected):
        raise ValueError("every layer must retain exactly retained_count experts")
    return expected


def mask_sha256(mask: np.ndarray, *, retained_count: int | None = None) -> str:
    """Return a layout-independent, shape-aware hash of a pruning mask.

    The bit order and integer byte order are explicit, which makes the digest
    stable across C/Fortran array layouts and host architectures.
    """

    validate_fixed_cardinality_mask(mask, retained_count=retained_count)
    shape = np.asarray(mask.shape, dtype="<u8").tobytes()
    bits = np.packbits(
        np.ascontiguousarray(mask).reshape(-1), bitorder="little"
    ).tobytes()
    payload = b"dsai6707-fixed-cardinality-mask-v1\0" + shape + bits
    return sha256(payload).hexdigest()


def balanced_set_crossover(
    first: np.ndarray,
    second: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Create two cardinality-preserving children by splitting parent differences.

    For every layer, experts pruned by both parents remain pruned by both
    children.  The symmetric difference of the two parental pruned sets is
    randomly divided into equal halves.  Each differing expert is therefore
    pruned by exactly one child, and both children retain the parental count.
    """

    if not isinstance(first, np.ndarray) or not isinstance(second, np.ndarray):
        raise ValueError("parents must be NumPy arrays")
    if first.shape != second.shape:
        raise ValueError("parents must have matching shapes")
    retained_count = validate_fixed_cardinality_mask(first)
    validate_fixed_cardinality_mask(second, retained_count=retained_count)

    first_child = np.ones(first.shape, dtype=np.bool_)
    second_child = np.ones(first.shape, dtype=np.bool_)
    for layer in range(first.shape[0]):
        common_pruned = np.flatnonzero(~first[layer] & ~second[layer])
        symmetric_difference = np.flatnonzero(first[layer] ^ second[layer])
        if symmetric_difference.size % 2:
            raise RuntimeError("equal-cardinality parent sets must have an even difference")

        shuffled = rng.permutation(symmetric_difference)
        split = shuffled.size // 2
        first_pruned = np.concatenate((common_pruned, shuffled[:split]))
        second_pruned = np.concatenate((common_pruned, shuffled[split:]))
        first_child[layer, first_pruned] = False
        second_child[layer, second_pruned] = False

    validate_fixed_cardinality_mask(first_child, retained_count=retained_count)
    validate_fixed_cardinality_mask(second_child, retained_count=retained_count)
    return np.ascontiguousarray(first_child), np.ascontiguousarray(second_child)


def mutate_distinct_layer_swaps(
    mask: np.ndarray,
    *,
    rng: np.random.Generator,
    min_layers: int = 1,
    max_layers: int = 3,
) -> MultiLayerSwapMutation:
    """Swap one retained/pruned expert pair in 1--3 distinct layers."""

    retained_count = validate_fixed_cardinality_mask(mask)
    if not isinstance(min_layers, int) or not isinstance(max_layers, int):
        raise ValueError("min_layers and max_layers must be integers")
    if min_layers < 1 or max_layers < min_layers or max_layers > 3:
        raise ValueError("mutation bounds must satisfy 1 <= min_layers <= max_layers <= 3")
    effective_max = min(max_layers, mask.shape[0])
    if min_layers > effective_max:
        raise ValueError("min_layers exceeds the number of available layers")

    mutation_count = int(rng.integers(min_layers, effective_max + 1))
    selected_layers = np.sort(
        rng.choice(mask.shape[0], size=mutation_count, replace=False)
    )
    child = mask.copy()
    swaps: list[LayerSwap] = []
    for raw_layer in selected_layers:
        layer = int(raw_layer)
        retained = np.flatnonzero(child[layer])
        pruned = np.flatnonzero(~child[layer])
        removed = int(rng.choice(retained))
        added = int(rng.choice(pruned))
        child[layer, removed] = False
        child[layer, added] = True
        swaps.append(LayerSwap(layer, removed, added))

    validate_fixed_cardinality_mask(child, retained_count=retained_count)
    return MultiLayerSwapMutation(np.ascontiguousarray(child), tuple(swaps))


def _validate_objectives(objectives: np.ndarray) -> np.ndarray:
    values = np.asarray(objectives, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("objectives must have shape [candidates, objectives]")
    if not np.isfinite(values).all():
        raise ValueError("objectives must contain only finite values")
    return values


def _validate_candidate_ids(
    candidate_ids: Sequence[int] | np.ndarray | None,
    *,
    size: int,
) -> np.ndarray:
    if candidate_ids is None:
        return np.arange(size, dtype=np.int64)
    raw = np.asarray(candidate_ids)
    if raw.ndim != 1 or raw.shape[0] != size or raw.dtype.kind not in "iu":
        raise ValueError("candidate_ids must be a one-dimensional integer sequence")
    ids = raw.astype(np.int64, copy=False)
    if np.unique(ids).size != size:
        raise ValueError("candidate_ids must be unique")
    return ids


def _dominates(first: np.ndarray, second: np.ndarray) -> bool:
    return bool(np.all(first <= second) and np.any(first < second))


def nondominated_sort(
    objectives: np.ndarray,
    *,
    candidate_ids: Sequence[int] | np.ndarray | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Return deterministic Pareto fronts as positional candidate indices."""

    values = _validate_objectives(objectives)
    ids = _validate_candidate_ids(candidate_ids, size=values.shape[0])
    dominated: list[list[int]] = [[] for _ in range(values.shape[0])]
    domination_counts = np.zeros(values.shape[0], dtype=np.int64)

    for first in range(values.shape[0]):
        for second in range(first + 1, values.shape[0]):
            if _dominates(values[first], values[second]):
                dominated[first].append(second)
                domination_counts[second] += 1
            elif _dominates(values[second], values[first]):
                dominated[second].append(first)
                domination_counts[first] += 1

    current = sorted(
        np.flatnonzero(domination_counts == 0).tolist(), key=lambda index: int(ids[index])
    )
    fronts: list[tuple[int, ...]] = []
    while current:
        fronts.append(tuple(current))
        following: list[int] = []
        for first in current:
            for second in dominated[first]:
                domination_counts[second] -= 1
                if domination_counts[second] == 0:
                    following.append(second)
        current = sorted(following, key=lambda index: int(ids[index]))
    return tuple(fronts)


def crowding_distances(
    objectives: np.ndarray,
    front: Iterable[int],
    *,
    candidate_ids: Sequence[int] | np.ndarray | None = None,
) -> dict[int, float]:
    """Compute standard normalized NSGA-II distances for one Pareto front."""

    values = _validate_objectives(objectives)
    ids = _validate_candidate_ids(candidate_ids, size=values.shape[0])
    indices = tuple(int(index) for index in front)
    if not indices:
        return {}
    if len(set(indices)) != len(indices) or any(
        index < 0 or index >= values.shape[0] for index in indices
    ):
        raise ValueError("front must contain unique valid candidate indices")

    ordered_indices = sorted(indices, key=lambda index: int(ids[index]))
    distances = {index: 0.0 for index in ordered_indices}
    if len(indices) <= 2:
        return {index: float("inf") for index in ordered_indices}

    for objective in range(values.shape[1]):
        ordered = sorted(
            indices,
            key=lambda index: (float(values[index, objective]), int(ids[index])),
        )
        low = float(values[ordered[0], objective])
        high = float(values[ordered[-1], objective])
        span = high - low
        if span == 0.0:
            continue
        distances[ordered[0]] = float("inf")
        distances[ordered[-1]] = float("inf")
        for position in range(1, len(ordered) - 1):
            index = ordered[position]
            if np.isinf(distances[index]):
                continue
            previous_value = float(values[ordered[position - 1], objective])
            next_value = float(values[ordered[position + 1], objective])
            distances[index] += (next_value - previous_value) / span
    return distances


def rank_and_crowding(
    objectives: np.ndarray,
    *,
    candidate_ids: Sequence[int] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a Pareto rank and crowding distance for every candidate."""

    values = _validate_objectives(objectives)
    ids = _validate_candidate_ids(candidate_ids, size=values.shape[0])
    ranks = np.empty(values.shape[0], dtype=np.int64)
    distances = np.zeros(values.shape[0], dtype=np.float64)
    for rank, front in enumerate(nondominated_sort(values, candidate_ids=ids)):
        ranks[np.asarray(front, dtype=np.int64)] = rank
        for index, distance in crowding_distances(
            values, front, candidate_ids=ids
        ).items():
            distances[index] = distance
    return ranks, distances


def _crowded_precedes(
    first: int,
    second: int,
    *,
    ranks: np.ndarray,
    distances: np.ndarray,
    candidate_ids: np.ndarray,
) -> bool:
    if ranks[first] != ranks[second]:
        return bool(ranks[first] < ranks[second])
    if distances[first] != distances[second]:
        return bool(distances[first] > distances[second])
    return bool(candidate_ids[first] < candidate_ids[second])


def crowded_tournament(
    ranks: Sequence[int] | np.ndarray,
    distances: Sequence[float] | np.ndarray,
    candidate_ids: Sequence[int] | np.ndarray,
    *,
    rng: np.random.Generator,
    tournament_size: int = 2,
    pool: Iterable[int] | None = None,
) -> int:
    """Select one positional index by crowded comparison and ID tie-break."""

    rank_values = np.asarray(ranks, dtype=np.int64)
    distance_values = np.asarray(distances, dtype=np.float64)
    if rank_values.ndim != 1 or distance_values.shape != rank_values.shape:
        raise ValueError("ranks and distances must be same-length one-dimensional arrays")
    if np.isnan(distance_values).any():
        raise ValueError("distances must not contain NaN")
    ids = _validate_candidate_ids(candidate_ids, size=rank_values.size)
    eligible = (
        np.arange(rank_values.size, dtype=np.int64)
        if pool is None
        else np.asarray(tuple(pool), dtype=np.int64)
    )
    if eligible.ndim != 1 or eligible.size < 1:
        raise ValueError("pool must contain at least one candidate index")
    if np.unique(eligible).size != eligible.size or np.any(eligible < 0) or np.any(
        eligible >= rank_values.size
    ):
        raise ValueError("pool must contain unique valid candidate indices")
    if not isinstance(tournament_size, int) or not 1 <= tournament_size <= eligible.size:
        raise ValueError("tournament_size must be in [1, pool size]")

    contestants = rng.choice(eligible, size=tournament_size, replace=False)
    winner = int(contestants[0])
    for raw_challenger in contestants[1:]:
        challenger = int(raw_challenger)
        if _crowded_precedes(
            challenger,
            winner,
            ranks=rank_values,
            distances=distance_values,
            candidate_ids=ids,
        ):
            winner = challenger
    return winner


def select_survivors(
    objectives: np.ndarray,
    candidate_ids: Sequence[int] | np.ndarray,
    population_size: int,
) -> np.ndarray:
    """Apply deterministic NSGA-II environmental selection.

    Complete fronts are ordered by candidate ID.  The last partial front is
    ordered by decreasing crowding distance and then increasing candidate ID.
    Returned values are positional indices into ``objectives``.
    """

    values = _validate_objectives(objectives)
    ids = _validate_candidate_ids(candidate_ids, size=values.shape[0])
    if not isinstance(population_size, int) or not 1 <= population_size <= values.shape[0]:
        raise ValueError("population_size must be in [1, candidate count]")

    selected: list[int] = []
    for front in nondominated_sort(values, candidate_ids=ids):
        remaining = population_size - len(selected)
        if len(front) <= remaining:
            selected.extend(front)
        else:
            distances = crowding_distances(values, front, candidate_ids=ids)
            ordered = sorted(
                front,
                key=lambda index: (-distances[index], int(ids[index])),
            )
            selected.extend(ordered[:remaining])
        if len(selected) == population_size:
            break
    return np.asarray(selected, dtype=np.int64)

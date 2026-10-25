"""Pure-NumPy operators for globally budgeted expert-pruning masks.

Unlike :mod:`dsai6707.pruning_nsga2`, this module does not require every
layer to retain the same number of experts.  A candidate is valid when its
boolean ``[layers, experts]`` mask retains an exact global number of expert
slots and every layer stays within configured lower and upper bounds.

The module deliberately contains no model or objective code.  All operators
own their returned arrays, leave their inputs untouched, and validate their
results before returning.  This keeps the search representation independently
testable on CPU and prevents malformed candidates from reaching a GPU runner.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np

__all__ = [
    "BudgetTransfer",
    "BudgetTransferMutation",
    "GlobalBudgetCrossover",
    "GlobalBudgetRepair",
    "IdentitySwapMutation",
    "balanced_global_set_crossover",
    "build_mask_from_layer_rankings",
    "crossover_global_budget",
    "global_budget_mask_sha256",
    "mutate_budget_transfer",
    "mutate_identity_swap",
    "repair_global_budget_bounds",
    "validate_global_budget_mask",
]


def _as_index(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _validate_layout_and_constraints(
    mask: np.ndarray,
    *,
    total_retained: int,
    min_retained_per_layer: int,
    max_retained_per_layer: int,
    enforce_layer_bounds: bool,
) -> tuple[int, ...]:
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise ValueError("mask must be a NumPy boolean array")
    if mask.ndim != 2 or mask.shape[0] < 1 or mask.shape[1] < 2:
        raise ValueError("mask must have shape [layers, experts]")

    total = _as_index(total_retained, name="total_retained")
    minimum = _as_index(min_retained_per_layer, name="min_retained_per_layer")
    maximum = _as_index(max_retained_per_layer, name="max_retained_per_layer")
    layers, experts = mask.shape
    if not 1 <= minimum <= maximum <= experts:
        raise ValueError(
            "layer bounds must satisfy 1 <= min_retained_per_layer <= "
            "max_retained_per_layer <= num_experts"
        )
    if not layers * minimum <= total <= layers * maximum:
        raise ValueError("total_retained is infeasible for the layer bounds")

    raw_counts = np.count_nonzero(mask, axis=1)
    counts = tuple(int(value) for value in raw_counts)
    if sum(counts) != total:
        raise ValueError("mask does not retain exactly total_retained expert slots")
    if enforce_layer_bounds and any(value < minimum or value > maximum for value in counts):
        raise ValueError("one or more layer retained counts violate the configured bounds")
    return counts


def validate_global_budget_mask(
    mask: np.ndarray,
    *,
    total_retained: int,
    min_retained_per_layer: int,
    max_retained_per_layer: int,
) -> tuple[int, ...]:
    """Validate a canonical global-budget mask and return its layer counts."""

    return _validate_layout_and_constraints(
        mask,
        total_retained=total_retained,
        min_retained_per_layer=min_retained_per_layer,
        max_retained_per_layer=max_retained_per_layer,
        enforce_layer_bounds=True,
    )


def global_budget_mask_sha256(
    mask: np.ndarray,
    *,
    total_retained: int,
    min_retained_per_layer: int,
    max_retained_per_layer: int,
) -> str:
    """Return a stable, shape-aware hash for a valid global-budget mask.

    Constraint values are used for validation but are not part of the digest:
    the same semantic mask therefore has the same identity if it is admitted by
    two different search configurations.  The domain separator is distinct
    from the legacy fixed-cardinality hash schema.
    """

    validate_global_budget_mask(
        mask,
        total_retained=total_retained,
        min_retained_per_layer=min_retained_per_layer,
        max_retained_per_layer=max_retained_per_layer,
    )
    shape = np.asarray(mask.shape, dtype="<u8").tobytes()
    bits = np.packbits(
        np.ascontiguousarray(mask).reshape(-1), bitorder="little"
    ).tobytes()
    payload = b"dsai6707-global-budget-mask-v1\0" + shape + bits
    return sha256(payload).hexdigest()


@dataclass(frozen=True)
class IdentitySwapMutation:
    """One retained/pruned identity exchange within a single layer."""

    mask: np.ndarray
    layer: int
    removed_expert: int
    added_expert: int


@dataclass(frozen=True)
class BudgetTransfer:
    """One expert slot moved from a donor layer to a receiver layer."""

    donor_layer: int
    receiver_layer: int
    removed_expert: int
    added_expert: int
    donor_count_before: int
    receiver_count_before: int


@dataclass(frozen=True)
class BudgetTransferMutation:
    """A global-budget mutation and its exact cross-layer transfer."""

    mask: np.ndarray
    transfer: BudgetTransfer


@dataclass(frozen=True)
class GlobalBudgetRepair:
    """A bounds-repaired mask and the transfers used to obtain it."""

    mask: np.ndarray
    transfers: tuple[BudgetTransfer, ...]


@dataclass(frozen=True)
class GlobalBudgetCrossover:
    """Two valid children plus any bound-repair transfers."""

    first_child: np.ndarray
    second_child: np.ndarray
    first_repairs: tuple[BudgetTransfer, ...]
    second_repairs: tuple[BudgetTransfer, ...]


def _chosen_index(values: np.ndarray, *, rng: np.random.Generator) -> int:
    if values.ndim != 1 or values.size < 1:
        raise RuntimeError("cannot choose from an empty candidate set")
    return int(values[int(rng.integers(values.size))])


def mutate_identity_swap(
    mask: np.ndarray,
    *,
    rng: np.random.Generator,
    total_retained: int,
    min_retained_per_layer: int,
    max_retained_per_layer: int,
    layer: int | None = None,
) -> IdentitySwapMutation:
    """Uniformly exchange one retained/pruned expert in a single layer."""

    validate_global_budget_mask(
        mask,
        total_retained=total_retained,
        min_retained_per_layer=min_retained_per_layer,
        max_retained_per_layer=max_retained_per_layer,
    )
    eligible = np.flatnonzero(
        (np.count_nonzero(mask, axis=1) > 0) & (~np.all(mask, axis=1))
    )
    if eligible.size == 0:
        raise ValueError("identity swap requires a layer with retained and pruned experts")
    if layer is None:
        selected_layer = _chosen_index(eligible, rng=rng)
    else:
        selected_layer = _as_index(layer, name="layer")
        if selected_layer not in set(int(value) for value in eligible):
            raise ValueError("requested identity-swap layer is not eligible")

    retained = np.flatnonzero(mask[selected_layer])
    pruned = np.flatnonzero(~mask[selected_layer])
    removed = _chosen_index(retained, rng=rng)
    added = _chosen_index(pruned, rng=rng)
    child = np.ascontiguousarray(mask.copy())
    child[selected_layer, removed] = False
    child[selected_layer, added] = True
    validate_global_budget_mask(
        child,
        total_retained=total_retained,
        min_retained_per_layer=min_retained_per_layer,
        max_retained_per_layer=max_retained_per_layer,
    )
    return IdentitySwapMutation(child, selected_layer, removed, added)


def _transfer(
    mask: np.ndarray,
    *,
    donor_layer: int,
    receiver_layer: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, BudgetTransfer]:
    donor_count = int(np.count_nonzero(mask[donor_layer]))
    receiver_count = int(np.count_nonzero(mask[receiver_layer]))
    removed = _chosen_index(np.flatnonzero(mask[donor_layer]), rng=rng)
    added = _chosen_index(np.flatnonzero(~mask[receiver_layer]), rng=rng)
    child = np.ascontiguousarray(mask.copy())
    child[donor_layer, removed] = False
    child[receiver_layer, added] = True
    return child, BudgetTransfer(
        donor_layer=donor_layer,
        receiver_layer=receiver_layer,
        removed_expert=removed,
        added_expert=added,
        donor_count_before=donor_count,
        receiver_count_before=receiver_count,
    )


def mutate_budget_transfer(
    mask: np.ndarray,
    *,
    rng: np.random.Generator,
    total_retained: int,
    min_retained_per_layer: int,
    max_retained_per_layer: int,
    donor_layer: int | None = None,
    receiver_layer: int | None = None,
) -> BudgetTransferMutation:
    """Move one retained slot between two distinct eligible layers."""

    counts = validate_global_budget_mask(
        mask,
        total_retained=total_retained,
        min_retained_per_layer=min_retained_per_layer,
        max_retained_per_layer=max_retained_per_layer,
    )
    if (donor_layer is None) != (receiver_layer is None):
        raise ValueError("donor_layer and receiver_layer must be supplied together")

    minimum = _as_index(min_retained_per_layer, name="min_retained_per_layer")
    maximum = _as_index(max_retained_per_layer, name="max_retained_per_layer")
    pairs = tuple(
        (donor, receiver)
        for donor, donor_count in enumerate(counts)
        if donor_count > minimum
        for receiver, receiver_count in enumerate(counts)
        if receiver != donor and receiver_count < maximum
    )
    if not pairs:
        raise ValueError("no eligible cross-layer budget transfer exists")

    if donor_layer is None:
        selected_donor, selected_receiver = pairs[int(rng.integers(len(pairs)))]
    else:
        selected_donor = _as_index(donor_layer, name="donor_layer")
        selected_receiver = _as_index(receiver_layer, name="receiver_layer")
        if (selected_donor, selected_receiver) not in pairs:
            raise ValueError("requested donor/receiver pair is not eligible")

    child, transfer = _transfer(
        mask,
        donor_layer=selected_donor,
        receiver_layer=selected_receiver,
        rng=rng,
    )
    validate_global_budget_mask(
        child,
        total_retained=total_retained,
        min_retained_per_layer=min_retained_per_layer,
        max_retained_per_layer=max_retained_per_layer,
    )
    return BudgetTransferMutation(child, transfer)


def repair_global_budget_bounds(
    mask: np.ndarray,
    *,
    rng: np.random.Generator,
    total_retained: int,
    min_retained_per_layer: int,
    max_retained_per_layer: int,
) -> GlobalBudgetRepair:
    """Repair layer bounds using only global-budget-preserving transfers.

    The input must already have the exact global retained count.  Each repair
    transfer strictly reduces the total amount of bound violation, so repair is
    finite whenever the declared constraints are feasible.
    """

    counts = np.asarray(
        _validate_layout_and_constraints(
            mask,
            total_retained=total_retained,
            min_retained_per_layer=min_retained_per_layer,
            max_retained_per_layer=max_retained_per_layer,
            enforce_layer_bounds=False,
        ),
        dtype=np.int64,
    )
    minimum = _as_index(min_retained_per_layer, name="min_retained_per_layer")
    maximum = _as_index(max_retained_per_layer, name="max_retained_per_layer")
    child = np.ascontiguousarray(mask.copy())
    transfers: list[BudgetTransfer] = []

    while np.any(counts < minimum) or np.any(counts > maximum):
        underfull = np.flatnonzero(counts < minimum)
        overfull = np.flatnonzero(counts > maximum)
        if overfull.size and underfull.size:
            donor = _chosen_index(overfull, rng=rng)
            receiver = _chosen_index(underfull, rng=rng)
        elif overfull.size:
            donor = _chosen_index(overfull, rng=rng)
            receivers = np.flatnonzero(counts < maximum)
            receivers = receivers[receivers != donor]
            receiver = _chosen_index(receivers, rng=rng)
        else:
            receiver = _chosen_index(underfull, rng=rng)
            donors = np.flatnonzero(counts > minimum)
            donors = donors[donors != receiver]
            donor = _chosen_index(donors, rng=rng)

        child, transfer = _transfer(
            child,
            donor_layer=donor,
            receiver_layer=receiver,
            rng=rng,
        )
        transfers.append(transfer)
        counts[donor] -= 1
        counts[receiver] += 1

    validate_global_budget_mask(
        child,
        total_retained=total_retained,
        min_retained_per_layer=min_retained_per_layer,
        max_retained_per_layer=max_retained_per_layer,
    )
    return GlobalBudgetRepair(child, tuple(transfers))


def crossover_global_budget(
    first: np.ndarray,
    second: np.ndarray,
    *,
    rng: np.random.Generator,
    total_retained: int,
    min_retained_per_layer: int,
    max_retained_per_layer: int,
) -> GlobalBudgetCrossover:
    """Split the parents' global symmetric difference, then repair bounds."""

    if not isinstance(first, np.ndarray) or not isinstance(second, np.ndarray):
        raise ValueError("parents must be NumPy arrays")
    if first.shape != second.shape:
        raise ValueError("parents must have matching shapes")
    validate_global_budget_mask(
        first,
        total_retained=total_retained,
        min_retained_per_layer=min_retained_per_layer,
        max_retained_per_layer=max_retained_per_layer,
    )
    validate_global_budget_mask(
        second,
        total_retained=total_retained,
        min_retained_per_layer=min_retained_per_layer,
        max_retained_per_layer=max_retained_per_layer,
    )

    common = np.ascontiguousarray(first & second)
    difference = np.flatnonzero(np.ravel(first ^ second))
    if difference.size % 2:
        raise RuntimeError("equal-global-budget parents must have an even difference")
    shuffled = rng.permutation(difference)
    split = shuffled.size // 2
    first_child = common.copy().reshape(-1)
    second_child = common.copy().reshape(-1)
    first_child[shuffled[:split]] = True
    second_child[shuffled[split:]] = True
    first_child = np.ascontiguousarray(first_child.reshape(first.shape))
    second_child = np.ascontiguousarray(second_child.reshape(second.shape))

    first_repair = repair_global_budget_bounds(
        first_child,
        rng=rng,
        total_retained=total_retained,
        min_retained_per_layer=min_retained_per_layer,
        max_retained_per_layer=max_retained_per_layer,
    )
    second_repair = repair_global_budget_bounds(
        second_child,
        rng=rng,
        total_retained=total_retained,
        min_retained_per_layer=min_retained_per_layer,
        max_retained_per_layer=max_retained_per_layer,
    )
    return GlobalBudgetCrossover(
        first_child=first_repair.mask,
        second_child=second_repair.mask,
        first_repairs=first_repair.transfers,
        second_repairs=second_repair.transfers,
    )


def balanced_global_set_crossover(
    first: np.ndarray,
    second: np.ndarray,
    *,
    rng: np.random.Generator,
    total_retained: int,
    min_retained_per_layer: int,
    max_retained_per_layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compatibility wrapper returning only the two repaired children."""

    result = crossover_global_budget(
        first,
        second,
        rng=rng,
        total_retained=total_retained,
        min_retained_per_layer=min_retained_per_layer,
        max_retained_per_layer=max_retained_per_layer,
    )
    return result.first_child, result.second_child


def build_mask_from_layer_rankings(
    rankings: np.ndarray,
    retained_counts: Any,
    *,
    total_retained: int,
    min_retained_per_layer: int,
    max_retained_per_layer: int,
) -> np.ndarray:
    """Select each layer's leading ranked experts under a budget vector.

    ``rankings[layer]`` must be a permutation of ``0..num_experts-1`` ordered
    from most to least preferred.  Expert-ID order is used only as supplied;
    callers that derive rankings from tied scores must therefore apply their
    desired deterministic tie-break before calling this function.
    """

    values = np.asarray(rankings)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise ValueError("rankings must have shape [layers, experts]")
    if values.dtype.kind not in "iu":
        raise ValueError("rankings must contain integer expert IDs")
    layers, experts = values.shape
    expected = np.arange(experts, dtype=np.int64)
    normalized = values.astype(np.int64, copy=False)
    for layer, row in enumerate(normalized):
        if not np.array_equal(np.sort(row), expected):
            raise ValueError(f"rankings row {layer} must be a permutation of expert IDs")

    raw_counts = np.asarray(retained_counts)
    if raw_counts.ndim != 1 or raw_counts.shape != (layers,) or raw_counts.dtype.kind not in "iu":
        raise ValueError("retained_counts must be a one-dimensional integer vector per layer")
    counts = raw_counts.astype(np.int64, copy=False)
    mask = np.zeros((layers, experts), dtype=np.bool_)
    for layer, count in enumerate(counts):
        integer_count = int(count)
        if integer_count < 0 or integer_count > experts:
            raise ValueError("retained_counts contains an out-of-range value")
        mask[layer, normalized[layer, :integer_count]] = True

    validate_global_budget_mask(
        mask,
        total_retained=total_retained,
        min_retained_per_layer=min_retained_per_layer,
        max_retained_per_layer=max_retained_per_layer,
    )
    return np.ascontiguousarray(mask)

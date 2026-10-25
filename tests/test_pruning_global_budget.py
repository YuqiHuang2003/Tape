from __future__ import annotations

import numpy as np
import pytest

from dsai6707.pruning_global_budget import (
    balanced_global_set_crossover,
    build_mask_from_layer_rankings,
    crossover_global_budget,
    global_budget_mask_sha256,
    mutate_budget_transfer,
    mutate_identity_swap,
    repair_global_budget_bounds,
    validate_global_budget_mask,
)

LAYERS = 4
EXPERTS = 6
TOTAL = 12
MINIMUM = 2
MAXIMUM = 4


def _mask_with_counts(counts: tuple[int, ...], *, offset: int = 0) -> np.ndarray:
    mask = np.zeros((LAYERS, EXPERTS), dtype=np.bool_)
    for layer, count in enumerate(counts):
        chosen = (np.arange(count) + offset + layer) % EXPERTS
        mask[layer, chosen] = True
    return mask


def _validate(mask: np.ndarray) -> tuple[int, ...]:
    return validate_global_budget_mask(
        mask,
        total_retained=TOTAL,
        min_retained_per_layer=MINIMUM,
        max_retained_per_layer=MAXIMUM,
    )


def test_validator_returns_counts_and_rejects_noncanonical_or_invalid_masks() -> None:
    mask = _mask_with_counts((2, 3, 3, 4))
    assert _validate(mask) == (2, 3, 3, 4)

    with pytest.raises(ValueError, match="boolean"):
        _validate(mask.astype(np.int8))
    with pytest.raises(ValueError, match="total_retained"):
        _validate(mask[:-1])

    below = _mask_with_counts((1, 3, 4, 4))
    with pytest.raises(ValueError, match="bounds"):
        _validate(below)
    above = _mask_with_counts((5, 2, 2, 3))
    with pytest.raises(ValueError, match="bounds"):
        _validate(above)
    with pytest.raises(ValueError, match="infeasible"):
        validate_global_budget_mask(
            mask,
            total_retained=5,
            min_retained_per_layer=2,
            max_retained_per_layer=4,
        )


def test_hash_is_stable_shape_aware_and_sensitive_to_expert_identity() -> None:
    mask = _mask_with_counts((2, 3, 3, 4))
    fortran = np.asfortranarray(mask)
    first = global_budget_mask_sha256(
        mask,
        total_retained=TOTAL,
        min_retained_per_layer=MINIMUM,
        max_retained_per_layer=MAXIMUM,
    )
    repeated = global_budget_mask_sha256(
        fortran,
        total_retained=TOTAL,
        min_retained_per_layer=MINIMUM,
        max_retained_per_layer=MAXIMUM,
    )
    assert first == repeated

    changed = mask.copy()
    removed = int(np.flatnonzero(changed[1])[0])
    added = int(np.flatnonzero(~changed[1])[0])
    changed[1, removed] = False
    changed[1, added] = True
    assert first != global_budget_mask_sha256(
        changed,
        total_retained=TOTAL,
        min_retained_per_layer=MINIMUM,
        max_retained_per_layer=MAXIMUM,
    )

    reshaped = mask.reshape(2, 12)
    assert first != global_budget_mask_sha256(
        reshaped,
        total_retained=TOTAL,
        min_retained_per_layer=2,
        max_retained_per_layer=10,
    )


def test_identity_swap_changes_two_bits_without_changing_layer_budgets() -> None:
    parent = _mask_with_counts((2, 3, 3, 4))
    snapshot = parent.copy()
    first = mutate_identity_swap(
        parent,
        rng=np.random.default_rng(6707),
        total_retained=TOTAL,
        min_retained_per_layer=MINIMUM,
        max_retained_per_layer=MAXIMUM,
        layer=1,
    )
    repeated = mutate_identity_swap(
        parent,
        rng=np.random.default_rng(6707),
        total_retained=TOTAL,
        min_retained_per_layer=MINIMUM,
        max_retained_per_layer=MAXIMUM,
        layer=1,
    )

    np.testing.assert_array_equal(parent, snapshot)
    np.testing.assert_array_equal(first.mask, repeated.mask)
    assert (
        first.layer,
        first.removed_expert,
        first.added_expert,
    ) == (
        repeated.layer,
        repeated.removed_expert,
        repeated.added_expert,
    )
    assert first.layer == 1
    assert parent[first.layer, first.removed_expert]
    assert not first.mask[first.layer, first.removed_expert]
    assert not parent[first.layer, first.added_expert]
    assert first.mask[first.layer, first.added_expert]
    assert int(np.count_nonzero(parent != first.mask)) == 2
    assert _validate(first.mask) == (2, 3, 3, 4)


def test_budget_transfer_moves_exactly_one_slot_between_requested_layers() -> None:
    parent = _mask_with_counts((2, 3, 3, 4))
    snapshot = parent.copy()
    mutation = mutate_budget_transfer(
        parent,
        rng=np.random.default_rng(17),
        total_retained=TOTAL,
        min_retained_per_layer=MINIMUM,
        max_retained_per_layer=MAXIMUM,
        donor_layer=3,
        receiver_layer=0,
    )

    np.testing.assert_array_equal(parent, snapshot)
    assert _validate(mutation.mask) == (3, 3, 3, 3)
    assert mutation.transfer.donor_layer == 3
    assert mutation.transfer.receiver_layer == 0
    assert mutation.transfer.donor_count_before == 4
    assert mutation.transfer.receiver_count_before == 2
    assert int(np.count_nonzero(parent != mutation.mask)) == 2

    uniform = _mask_with_counts((3, 3, 3, 3))
    with pytest.raises(ValueError, match="no eligible"):
        mutate_budget_transfer(
            uniform,
            rng=np.random.default_rng(1),
            total_retained=TOTAL,
            min_retained_per_layer=3,
            max_retained_per_layer=3,
        )


def test_repair_uses_budget_transfers_and_is_deterministic() -> None:
    invalid = _mask_with_counts((1, 5, 3, 3))
    snapshot = invalid.copy()
    first = repair_global_budget_bounds(
        invalid,
        rng=np.random.default_rng(9),
        total_retained=TOTAL,
        min_retained_per_layer=MINIMUM,
        max_retained_per_layer=MAXIMUM,
    )
    repeated = repair_global_budget_bounds(
        invalid,
        rng=np.random.default_rng(9),
        total_retained=TOTAL,
        min_retained_per_layer=MINIMUM,
        max_retained_per_layer=MAXIMUM,
    )

    np.testing.assert_array_equal(invalid, snapshot)
    np.testing.assert_array_equal(first.mask, repeated.mask)
    assert first.transfers == repeated.transfers
    assert _validate(first.mask) == (2, 4, 3, 3)
    assert len(first.transfers) == 1
    transfer = first.transfers[0]
    assert (transfer.donor_layer, transfer.receiver_layer) == (1, 0)


def test_global_crossover_is_reproducible_bounded_and_preserves_total_budget() -> None:
    first_parent = _mask_with_counts((2, 3, 3, 4), offset=0)
    second_parent = _mask_with_counts((4, 3, 3, 2), offset=2)
    first_snapshot = first_parent.copy()
    second_snapshot = second_parent.copy()
    first = crossover_global_budget(
        first_parent,
        second_parent,
        rng=np.random.default_rng(6707),
        total_retained=TOTAL,
        min_retained_per_layer=MINIMUM,
        max_retained_per_layer=MAXIMUM,
    )
    repeated = crossover_global_budget(
        first_parent,
        second_parent,
        rng=np.random.default_rng(6707),
        total_retained=TOTAL,
        min_retained_per_layer=MINIMUM,
        max_retained_per_layer=MAXIMUM,
    )

    np.testing.assert_array_equal(first_parent, first_snapshot)
    np.testing.assert_array_equal(second_parent, second_snapshot)
    np.testing.assert_array_equal(first.first_child, repeated.first_child)
    np.testing.assert_array_equal(first.second_child, repeated.second_child)
    assert first.first_repairs == repeated.first_repairs
    assert first.second_repairs == repeated.second_repairs
    _validate(first.first_child)
    _validate(first.second_child)
    assert int(np.count_nonzero(first.first_child)) == TOTAL
    assert int(np.count_nonzero(first.second_child)) == TOTAL

    wrapped = balanced_global_set_crossover(
        first_parent,
        second_parent,
        rng=np.random.default_rng(6707),
        total_retained=TOTAL,
        min_retained_per_layer=MINIMUM,
        max_retained_per_layer=MAXIMUM,
    )
    np.testing.assert_array_equal(wrapped[0], first.first_child)
    np.testing.assert_array_equal(wrapped[1], first.second_child)


def test_build_mask_selects_prefixes_of_valid_layer_rankings() -> None:
    rankings = np.asarray(
        [
            [5, 4, 3, 2, 1, 0],
            [0, 2, 4, 1, 3, 5],
            [1, 3, 5, 0, 2, 4],
            [2, 1, 0, 5, 4, 3],
        ],
        dtype=np.int64,
    )
    counts = np.asarray([2, 3, 3, 4], dtype=np.int64)
    mask = build_mask_from_layer_rankings(
        rankings,
        counts,
        total_retained=TOTAL,
        min_retained_per_layer=MINIMUM,
        max_retained_per_layer=MAXIMUM,
    )
    assert _validate(mask) == tuple(int(value) for value in counts)
    for layer, count in enumerate(counts):
        assert set(np.flatnonzero(mask[layer])) == set(rankings[layer, :count])

    duplicated = rankings.copy()
    duplicated[0, 0] = duplicated[0, 1]
    with pytest.raises(ValueError, match="permutation"):
        build_mask_from_layer_rankings(
            duplicated,
            counts,
            total_retained=TOTAL,
            min_retained_per_layer=MINIMUM,
            max_retained_per_layer=MAXIMUM,
        )
    with pytest.raises(ValueError, match="integer"):
        build_mask_from_layer_rankings(
            rankings.astype(np.float64),
            counts,
            total_retained=TOTAL,
            min_retained_per_layer=MINIMUM,
            max_retained_per_layer=MAXIMUM,
        )

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256

import numpy as np
import pytest

from dsai6707.pruning_global_budget import global_budget_mask_sha256
from dsai6707.pruning_global_budget_search import (
    GlobalBudgetSearchConfig,
    make_initial_population_from_rankings,
    run_global_budget_nsga2,
    two_objective_shortlist,
)

LAYERS = 24
EXPERTS = 32
TOTAL = 672
MINIMUM = 26
MAXIMUM = 32


def _rankings() -> np.ndarray:
    experts = np.arange(EXPERTS, dtype=np.int64)
    return np.stack([np.roll(experts, -(3 * layer % EXPERTS)) for layer in range(LAYERS)])


def _is_ranking_prefix(mask: np.ndarray, rankings: np.ndarray) -> bool:
    for layer, count in enumerate(np.count_nonzero(mask, axis=1)):
        expected = set(int(value) for value in rankings[layer, : int(count)])
        actual = set(int(value) for value in np.flatnonzero(mask[layer]))
        if actual != expected:
            return False
    return True


def _digest(mask: np.ndarray) -> str:
    return global_budget_mask_sha256(
        mask,
        total_retained=TOTAL,
        min_retained_per_layer=MINIMUM,
        max_retained_per_layer=MAXIMUM,
    )


def _objective(calls: list[int]):
    def objective_many(masks: Sequence[np.ndarray]) -> tuple[tuple[float, float], ...]:
        calls.append(len(masks))
        results: list[tuple[float, float]] = []
        for mask in masks:
            assert mask.dtype == np.bool_
            assert mask.flags.c_contiguous
            pruned = np.argwhere(~mask).astype(np.float64)
            first = 1.0 + float(np.sum((pruned[:, 0] + 1.0) * (pruned[:, 1] + 1.0)))
            second = 1.0 + float(np.sum((LAYERS - pruned[:, 0]) * (EXPERTS - pruned[:, 1])))
            results.append((first, second))
        return tuple(results)

    return objective_many


def test_default_config_locks_128_unique_evaluations() -> None:
    config = GlobalBudgetSearchConfig()
    assert config.evaluation_budget == 128
    assert (config.population_size, config.offspring_size, config.generations) == (16, 16, 7)
    assert (config.total_retained, config.min_retained_per_layer) == (672, 26)
    assert config.max_retained_per_layer == 32
    assert config.crossover_probability == 1.0
    assert config.joint_mutation_pairs is None
    assert config.joint_mutation_probabilities is None

    with pytest.raises(ValueError, match="probabilities"):
        GlobalBudgetSearchConfig(transfer_probabilities=(0.5, 0.3, 0.3))


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), True])
def test_config_rejects_invalid_crossover_probability(probability: object) -> None:
    with pytest.raises(ValueError, match="crossover_probability"):
        GlobalBudgetSearchConfig(crossover_probability=probability)  # type: ignore[arg-type]


def test_config_validates_correlated_joint_mutation_distribution() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        GlobalBudgetSearchConfig(joint_mutation_pairs=((0, 1),))
    with pytest.raises(ValueError, match="supplied together"):
        GlobalBudgetSearchConfig(joint_mutation_probabilities=(1.0,))
    with pytest.raises(ValueError, match=r"cannot be \(0, 0\)"):
        GlobalBudgetSearchConfig(
            joint_mutation_pairs=((0, 0),),
            joint_mutation_probabilities=(1.0,),
        )
    with pytest.raises(ValueError, match="nonnegative integers"):
        GlobalBudgetSearchConfig(
            joint_mutation_pairs=((-1, 1),),
            joint_mutation_probabilities=(1.0,),
        )
    with pytest.raises(ValueError, match="sum to one"):
        GlobalBudgetSearchConfig(
            joint_mutation_pairs=((0, 1), (1, 0)),
            joint_mutation_probabilities=(0.2, 0.2),
        )

    config = GlobalBudgetSearchConfig(
        joint_mutation_pairs=((0, 2), (2, 0), (1, 3)),
        joint_mutation_probabilities=(0.4, 0.3, 0.3),
    )
    assert config.joint_mutation_pairs == ((0, 2), (2, 0), (1, 3))


def test_shared_initial_population_is_deterministic_unique_and_ranking_based() -> None:
    rankings = _rankings()
    first = make_initial_population_from_rankings(rankings)
    second = make_initial_population_from_rankings(rankings)

    assert first.masks.shape == (16, LAYERS, EXPERTS)
    assert first.masks.dtype == np.bool_
    assert first.masks.flags.c_contiguous
    np.testing.assert_array_equal(first.masks, second.masks)
    assert first.contexts == second.contexts
    assert first.population_sha256 == second.population_sha256
    assert len({_digest(mask) for mask in first.masks}) == 16

    counts = np.count_nonzero(first.masks, axis=2)
    np.testing.assert_array_equal(counts[0], np.full(LAYERS, 28))
    assert np.all(np.sum(counts, axis=1) == TOTAL)
    assert np.all((counts >= MINIMUM) & (counts <= MAXIMUM))
    assert all(_is_ranking_prefix(mask, rankings) for mask in first.masks)
    assert {context["kind"] for context in first.contexts} == {
        "uniform_reap_ranking",
        "bounded_pattern",
        "random_transfer",
    }
    assert all(
        context["retained_counts"] == list(counts[index])
        for index, context in enumerate(first.contexts)
    )


def test_allocation_only_runs_128_unique_deterministic_ranking_prefixes() -> None:
    rankings = _rankings()
    initial = make_initial_population_from_rankings(rankings)
    first_calls: list[int] = []
    second_calls: list[int] = []

    first = run_global_budget_nsga2(
        _objective(first_calls),
        rankings,
        mode="allocation_only",
        initial_population=initial,
    )
    second = run_global_budget_nsga2(
        _objective(second_calls),
        rankings,
        mode="allocation_only",
        initial_population=initial,
    )

    assert first_calls == second_calls == [16] * 8
    assert first.masks.shape == (128, LAYERS, EXPERTS)
    assert first.objectives.shape == (128, 2)
    assert first.retained_counts.shape == (128, LAYERS)
    assert len(first.trials) == 128
    assert [trial["candidate_id"] for trial in first.trials] == list(range(128))
    assert len({trial["candidate_sha256"] for trial in first.trials}) == 128
    assert first.proposals == 112 + first.duplicate_proposals
    assert len(first.final_population_ids) == len(set(first.final_population_ids)) == 16
    assert all(_is_ranking_prefix(mask, rankings) for mask in first.masks)
    assert np.all(np.sum(first.retained_counts, axis=1) == TOTAL)
    assert np.all((first.retained_counts >= MINIMUM) & (first.retained_counts <= MAXIMUM))

    offspring = first.trials[16:]
    assert all(len(trial["parent_ids"]) == 2 for trial in offspring)
    assert all(trial["parent_ids"][0] != trial["parent_ids"][1] for trial in offspring)
    assert all(trial["transfer_count"] in (1, 2, 3) for trial in offspring)
    assert all(trial["identity_swap_count"] == 0 for trial in offspring)
    assert all(trial["operations"][0]["operator"] == "global_set_crossover" for trial in offspring)

    np.testing.assert_array_equal(first.masks, second.masks)
    np.testing.assert_array_equal(first.objectives, second.objectives)
    assert first.trials == second.trials
    assert first.final_population_ids == second.final_population_ids
    assert first.proposals == second.proposals
    assert first.duplicate_proposals == second.duplicate_proposals
    assert first.shortlist == second.shortlist


def test_joint_arm_shares_initial_population_and_journals_distinct_identity_swaps() -> None:
    rankings = _rankings()
    initial = make_initial_population_from_rankings(rankings)
    calls: list[int] = []
    result = run_global_budget_nsga2(
        _objective(calls),
        rankings,
        mode="joint",
        initial_population=initial,
    )

    assert calls == [16] * 8
    np.testing.assert_array_equal(result.masks[:16], initial.masks)
    assert result.initial_population_sha256 == initial.population_sha256
    assert len({_digest(mask) for mask in result.masks}) == 128
    assert np.all(np.sum(result.retained_counts, axis=1) == TOTAL)
    assert np.all((result.retained_counts >= MINIMUM) & (result.retained_counts <= MAXIMUM))
    assert any(not _is_ranking_prefix(mask, rankings) for mask in result.masks[16:])
    assert sha256(result.masks.tobytes()).hexdigest() == (
        "aedfbaaef88faf24b7164d6490cb7d64a0cc0dc3684905b2e4b781aa06a4de39"
    )

    for candidate_id, trial in enumerate(result.trials):
        assert trial["retained_counts"] == list(result.retained_counts[candidate_id])
        assert trial["candidate_sha256"] == _digest(result.masks[candidate_id])
    for trial in result.trials[16:]:
        assert trial["transfer_count"] in (1, 2, 3)
        assert trial["identity_swap_count"] in (1, 2, 3)
        swaps = [
            operation
            for operation in trial["operations"]
            if operation["operator"] == "identity_swap"
        ]
        assert len(swaps) == trial["identity_swap_count"]
        assert len({swap["layer"] for swap in swaps}) == len(swaps)


@pytest.mark.parametrize("mutation_pair", [(0, 2), (2, 0)])
def test_joint_correlated_mutations_allow_one_zero_count_with_parent_clones(
    mutation_pair: tuple[int, int],
) -> None:
    rankings = _rankings()
    config = GlobalBudgetSearchConfig(
        population_size=8,
        offspring_size=8,
        generations=2,
        crossover_probability=0.0,
        joint_mutation_pairs=(mutation_pair,),
        joint_mutation_probabilities=(1.0,),
    )
    initial = make_initial_population_from_rankings(rankings, config=config)
    first = run_global_budget_nsga2(
        _objective([]),
        rankings,
        mode="joint",
        config=config,
        initial_population=initial,
    )
    second = run_global_budget_nsga2(
        _objective([]),
        rankings,
        mode="joint",
        config=config,
        initial_population=initial,
    )

    np.testing.assert_array_equal(first.masks, second.masks)
    assert first.trials == second.trials
    assert first.masks.shape == (24, LAYERS, EXPERTS)
    assert np.all(np.sum(first.retained_counts, axis=1) == TOTAL)
    assert np.all((first.retained_counts >= MINIMUM) & (first.retained_counts <= MAXIMUM))

    for trial in first.trials[8:]:
        assert (trial["transfer_count"], trial["identity_swap_count"]) == mutation_pair
        reproduction = trial["operations"][0]
        assert reproduction["operator"] == "parent_clone"
        assert reproduction["source_parent_id"] in trial["parent_ids"]
        assert reproduction["counts_after"] in trial["parent_counts"]
        transfers = [
            operation
            for operation in trial["operations"]
            if operation["operator"] == "budget_transfer"
        ]
        swaps = [
            operation
            for operation in trial["operations"]
            if operation["operator"] == "identity_swap"
        ]
        assert len(transfers) == mutation_pair[0]
        assert len(swaps) == mutation_pair[1]


def test_fractional_crossover_probability_is_deterministic_and_uses_both_paths() -> None:
    rankings = _rankings()
    config = GlobalBudgetSearchConfig(
        population_size=8,
        offspring_size=8,
        generations=3,
        crossover_probability=0.5,
        joint_mutation_pairs=((0, 2), (2, 0), (1, 3)),
        joint_mutation_probabilities=(0.4, 0.3, 0.3),
    )
    initial = make_initial_population_from_rankings(rankings, config=config)
    first = run_global_budget_nsga2(
        _objective([]),
        rankings,
        mode="joint",
        config=config,
        initial_population=initial,
    )
    second = run_global_budget_nsga2(
        _objective([]),
        rankings,
        mode="joint",
        config=config,
        initial_population=initial,
    )

    np.testing.assert_array_equal(first.masks, second.masks)
    assert first.trials == second.trials
    reproduction_operators = {trial["operations"][0]["operator"] for trial in first.trials[8:]}
    assert reproduction_operators == {"global_set_crossover", "parent_clone"}
    observed_pairs = {
        (trial["transfer_count"], trial["identity_swap_count"]) for trial in first.trials[8:]
    }
    assert observed_pairs == {(0, 2), (2, 0), (1, 3)}


def test_allocation_only_parent_clone_journal_names_projection_source() -> None:
    rankings = _rankings()
    config = GlobalBudgetSearchConfig(
        population_size=8,
        offspring_size=8,
        generations=1,
        crossover_probability=0.0,
    )
    initial = make_initial_population_from_rankings(rankings, config=config)
    result = run_global_budget_nsga2(
        _objective([]),
        rankings,
        mode="allocation_only",
        config=config,
        initial_population=initial,
    )

    for trial in result.trials[8:]:
        assert trial["operations"][0]["operator"] == "parent_clone"
        assert trial["operations"][1]["operator"] == "reap_ranking_projection"
        assert trial["operations"][1]["after"] == "parent_clone"
        assert _is_ranking_prefix(result.masks[trial["candidate_id"]], rankings)


def test_shortlist_contains_task_extremes_minimax_and_mean_from_pareto_front() -> None:
    objectives = np.asarray(
        [
            [10.0, 10.0],  # Uniform-REAP reference; dominated but still the scale anchor.
            [4.0, 12.0],  # Task-0 extreme.
            [12.0, 4.0],  # Task-1 extreme.
            [8.0, 8.0],  # Relative minimax.
            [5.0, 10.0],  # Minimum relative mean.
            [9.0, 9.0],  # Dominated.
        ],
        dtype=np.float64,
    )
    shortlist = two_objective_shortlist(
        objectives,
        candidate_ids=np.asarray([0, 10, 11, 12, 13, 14], dtype=np.int64),
    )

    assert shortlist.roles == (
        ("task_0_extreme", 10),
        ("task_1_extreme", 11),
        ("minimax", 12),
        ("mean", 13),
    )
    assert shortlist.candidate_ids == (10, 11, 12, 13)
    assert 14 not in shortlist.candidate_ids

    invalid_reference = objectives.copy()
    invalid_reference[0, 0] = 0.0
    with pytest.raises(ValueError, match="reference objectives"):
        two_objective_shortlist(invalid_reference)

    assert two_objective_shortlist(np.asarray([[1.0, 1.0]])).candidate_ids == ()


def test_search_rejects_bad_objectives_and_degenerate_transfer_space() -> None:
    rankings = _rankings()
    initial = make_initial_population_from_rankings(rankings)

    def bad_objective(masks: Sequence[np.ndarray]) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0,) for _ in masks)

    with pytest.raises(RuntimeError, match="two finite"):
        run_global_budget_nsga2(
            bad_objective,
            rankings,
            mode="allocation_only",
            initial_population=initial,
        )

    fixed = GlobalBudgetSearchConfig(
        total_retained=LAYERS * MINIMUM,
        max_retained_per_layer=MINIMUM,
    )
    with pytest.raises(ValueError, match="legal cross-layer"):
        make_initial_population_from_rankings(rankings, config=fixed)

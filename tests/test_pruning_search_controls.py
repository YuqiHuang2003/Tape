from dataclasses import replace

import numpy as np
import pytest

from dsai6707.pruning_global_budget import global_budget_mask_sha256
from dsai6707.pruning_multistart import initial_population, search_config
from dsai6707.pruning_search_controls import (
    best_train_index,
    ranking_projection,
    run_control,
)


@pytest.fixture
def problem():
    config = replace(
        search_config(layers=6, retained=5),
        population_size=4,
        offspring_size=4,
        generations=3,
        min_retained_per_layer=3,
        max_retained_per_layer=7,
    )
    anchor = np.zeros((6, 8), dtype=bool)
    anchor[:, :5] = True
    initial = initial_population({"anchor": anchor}, config).masks
    return config, initial, np.tile(np.arange(8), (6, 1))


def objective(masks):
    return [(float(mask[:, ::2].sum()), float(mask[::2].sum())) for mask in masks]


@pytest.mark.parametrize("mode", ["random_neighborhood", "hill_climb"])
def test_controls_have_exact_budget_and_shared_initial_masks(problem, mode):
    config, initial, rankings = problem
    original = initial.copy()
    result = run_control(objective, initial, rankings, config=config, mode=mode)
    assert len(result.masks) == config.evaluation_budget == 16
    np.testing.assert_array_equal(result.masks[:4], original)
    np.testing.assert_array_equal(initial, original)
    bounds = {
        name: getattr(config, name)
        for name in ("total_retained", "min_retained_per_layer", "max_retained_per_layer")
    }
    assert len({global_budget_mask_sha256(mask, **bounds) for mask in result.masks}) == 16
    assert np.all(result.masks.sum((1, 2)) == 30)


def test_random_proposals_do_not_depend_on_fitness(problem):
    config, initial, rankings = problem
    first = run_control(objective, initial, rankings, config=config, mode="random_neighborhood")
    second = run_control(
        lambda masks: [(0.0, 0.0)] * len(masks),
        initial,
        rankings,
        config=config,
        mode="random_neighborhood",
    )
    np.testing.assert_array_equal(first.masks, second.masks)
    assert first.trials == second.trials


def test_hill_acceptance_uses_strict_scalar_loss_not_test_or_accuracy(problem):
    config, initial, rankings = problem
    result = run_control(objective, initial, rankings, config=config, mode="hill_climb")
    for row in result.trials[4:]:
        child, parent = row["candidate_id"], row["parent_id"]
        assert row["accepted"] == bool(
            result.objectives[child].mean() < result.objectives[parent].mean()
        )
    ties = run_control(
        lambda masks: [(1.0, 1.0)] * len(masks), initial, rankings, config=config, mode="hill_climb"
    )
    assert all(not row["accepted"] for row in ties.trials[4:])
    assert all(row["parent_id"] < 4 for row in ties.trials[4:])


def test_projection_controls_layer_capacity_and_has_deterministic_ties():
    mask = np.array([[True, False, True], [False, True, False]])
    projected = ranking_projection(mask, np.array([[1.0, 5.0, 3.0], [4.0, 4.0, 1.0]]))
    np.testing.assert_array_equal(projected, [[False, True, True], [True, False, False]])
    np.testing.assert_array_equal(projected.sum(1), mask.sum(1))


def test_best_train_selection_can_be_limited_to_initial_population():
    masks = np.array([[[True, False]], [[False, True]]])
    metrics = [
        {"tasks": {"task": {"acc_norm": a, "objective_choice_cross_entropy": 1.0}}}
        for a in (0.4, 0.6)
    ]
    assert best_train_index(masks, metrics) == 1
    assert best_train_index(masks, metrics, limit=1) == 0
    with pytest.raises(ValueError):
        best_train_index(masks, metrics, limit=3)


def test_controls_reject_invalid_callback_before_search(problem):
    config, initial, rankings = problem
    with pytest.raises(ValueError, match="finite objective pair"):
        run_control(
            lambda masks: [(np.nan, 0)] * len(masks),
            initial,
            rankings,
            config=config,
            mode="hill_climb",
        )

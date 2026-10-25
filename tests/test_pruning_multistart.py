from dataclasses import replace

import numpy as np
import pytest

from dsai6707.pruning_global_budget_search import run_global_budget_nsga2
from dsai6707.pruning_multistart import (
    initial_population,
    mask_digest,
    proxy_selection_key,
    search_config,
    shortlist,
)


@pytest.mark.parametrize("layers,experts,retained", [(24, 32, 24), (24, 32, 16), (27, 64, 32)])
def test_all_requested_budgets_preserve_every_distinct_baseline(layers, experts, retained):
    first = np.zeros((layers, experts), dtype=bool)
    first[:, :retained] = True
    second = np.roll(first, 5, axis=1)
    anchors = {"reap": first, "frequency": second, "duplicate": first.copy()}
    config = search_config(layers=layers, retained=retained)
    initial = initial_population(anchors, config)
    duplicate = initial_population(anchors, config)
    np.testing.assert_array_equal(initial.masks[:2], [first, second])
    np.testing.assert_array_equal(initial.masks, duplicate.masks)
    assert initial.population_sha256 == duplicate.population_sha256
    assert len({mask_digest(mask) for mask in initial.masks}) == 16
    rankings = np.tile(np.arange(experts), (layers, 1))

    def objectives(masks):
        return [(1 + np.sum(mask[:, ::2]) / 100, 1 + np.sum(mask[::2]) / 100) for mask in masks]

    result = run_global_budget_nsga2(
        objectives,
        rankings,
        mode="joint",
        config=replace(config, generations=1),
        initial_population=initial,
    )
    assert len(result.masks) == 32
    np.testing.assert_array_equal(result.masks[:16], initial.masks)
    assert np.all(result.masks.sum(axis=(1, 2)) == layers * retained)
    assert result.masks.sum(axis=2).min() >= retained - 4
    assert result.masks.sum(axis=2).max() <= retained + 4


def _metric(arc, obqa, ce=1.0):
    return {
        "tasks": {
            name: {"acc_norm": acc, "objective_choice_cross_entropy": ce}
            for name, acc in (("arc_easy", arc), ("openbookqa", obqa))
        }
    }


def test_shortlist_keeps_baselines_even_if_search_metrics_are_worse():
    rows = [_metric(0.1, 0.1), _metric(0.2, 0.2), _metric(0.9, 0.9), _metric(0.8, 0.8)]
    assert shortlist(rows, baseline_ids=(0, 1), limit=3) == (0, 1, 2)


def test_holdout_uses_macro_acc_norm_before_ce_and_cannot_drop_frequency():
    baselines = [_metric(0.6, 0.5, 5), _metric(0.7, 0.6, 8)]
    evolved = [_metric(0.6, 0.55, 0.1), _metric(0.8, 0.3, 0.01)]
    rows = baselines + evolved
    winner = min(rows, key=proxy_selection_key)
    assert winner is baselines[1]
    assert proxy_selection_key(winner) <= min(map(proxy_selection_key, baselines))

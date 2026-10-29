"""CPU tests for the fixed-ranking EvoESAP comparison baseline."""

from dataclasses import replace

import numpy as np
import pytest

from dsai6707.evoesap import EvoESAPConfig, level_switch_mutation, run_evoesap


def _config(**changes):
    return replace(
        EvoESAPConfig(
            total_retained=24,
            min_retained_per_layer=2,
            max_retained_per_layer=6,
            population_size=8,
            elite_size=2,
            generations=2,
        ),
        **changes,
    )


def _rankings(layers=6, experts=8):
    return np.stack([np.roll(np.arange(experts), layer) for layer in range(layers)])


def _fitness(masks):
    weights = np.linspace(0.0, 1.0, masks.shape[1] * masks.shape[2]).reshape(masks.shape[1:])
    return (masks * weights).sum(axis=(1, 2)) / masks.shape[1] / masks.shape[2]


def test_default_preserves_original_population_elites_and_matches_256_unique_budget():
    config = EvoESAPConfig(576, 20, 28)
    assert config.population_size == 32
    assert config.elite_size == 4
    assert config.offspring_size == 28
    assert config.generations == 8
    assert config.evaluation_budget == 256
    assert config.max_transfer == 4
    assert config.mutation_cap == 3


@pytest.mark.parametrize(
    "layers,experts,retained",
    [
        (24, 32, 24),
        (24, 32, 16),
        (27, 64, 48),
        (27, 64, 32),
    ],
)
def test_full_budget_fixed_ranking_prefix_conservation_and_uniqueness(layers, experts, retained):
    config = EvoESAPConfig(layers * retained, retained - 4, retained + 4)
    rankings = _rankings(layers, experts)
    batches = []

    def objective(masks):
        batches.append(len(masks))
        return _fitness(masks)

    result = run_evoesap(objective, rankings, config=config)
    assert result.unique_evaluations == 256
    assert batches == [32] + [28] * 8
    assert len(set(map(tuple, result.retained_counts))) == 256
    assert len(set(row["mask_sha256"] for row in result.trials)) == 256
    assert np.all(result.retained_counts.sum(axis=1) == layers * retained)
    assert result.retained_counts.min() >= retained - 4
    assert result.retained_counts.max() <= retained + 4
    assert np.all(result.retained_counts[0] == retained)
    for mask, counts in zip(result.masks, result.retained_counts, strict=True):
        for layer, keep in enumerate(counts):
            np.testing.assert_array_equal(
                np.flatnonzero(mask[layer]), np.sort(rankings[layer, :keep])
            )
    assert result.selection_ids == (result.winner_id,)
    assert result.winner_id == int(np.argmax(result.scores))
    assert len(result.final_population_ids) == 32
    assert len(result.generation_history) == 9
    for current, previous in zip(
        result.generation_history[1:], result.generation_history[:-1], strict=True
    ):
        expected = sorted(previous["population_ids"], key=lambda i: (-result.scores[i], i))[:4]
        assert current["elite_ids"] == expected
        assert current["population_ids"][:4] == expected


def test_seeded_reproducibility_and_different_seed_changes_random_candidates():
    config = _config()
    first = run_evoesap(_fitness, _rankings(), config=config)
    same = run_evoesap(_fitness, _rankings(), config=config)
    other = run_evoesap(_fitness, _rankings(), config=replace(config, seed=6708))
    np.testing.assert_array_equal(first.masks, same.masks)
    np.testing.assert_array_equal(first.scores, same.scores)
    assert first.trials == same.trials
    assert first.generation_history == same.generation_history
    assert not np.array_equal(first.masks, other.masks)


def test_mutation_composes_feasible_integer_level_switches_and_keeps_budget():
    config = _config()
    rng = np.random.default_rng(6707)
    original = np.full(6, 4, dtype=np.int64)
    observed = set()
    for _ in range(200):
        child, steps = level_switch_mutation(original, config=config, rng=rng)
        assert 1 <= len(steps) <= 3
        rebuilt = original.copy()
        for step in steps:
            donor, receiver, delta = (step[k] for k in ("donor_layer", "receiver_layer", "delta"))
            assert donor != receiver
            assert 1 <= delta <= 4
            rebuilt[donor] -= delta
            rebuilt[receiver] += delta
            assert rebuilt.min() >= 2
            assert rebuilt.max() <= 6
        np.testing.assert_array_equal(child, rebuilt)
        assert child.sum() == 24
        observed.add(len(steps))
    assert observed == {1, 2, 3}


def test_equal_scores_use_candidate_id_tie_breaking_and_never_test_labels():
    config = _config()
    result = run_evoesap(lambda masks: np.full(len(masks), 0.5), _rankings(), config=config)
    assert result.winner_id == 0
    assert result.ranked_candidate_ids == tuple(range(config.evaluation_budget))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.01, 1.01])
def test_callback_requires_bounded_finite_scalar_esap(bad):
    with pytest.raises(ValueError, match="ESAP"):
        run_evoesap(lambda masks: np.full(len(masks), bad), _rankings(), config=_config())


def test_callback_shape_is_not_silently_reinterpreted_as_a_two_objective_score():
    with pytest.raises(ValueError, match="one finite scalar"):
        run_evoesap(lambda masks: np.zeros((len(masks), 2)), _rankings(), config=_config())


def test_fail_fast_when_unique_budget_exceeds_feasible_space():
    with pytest.raises(ValueError, match="fewer distinct"):
        run_evoesap(
            _fitness,
            _rankings(2),
            config=_config(
                total_retained=8,
                min_retained_per_layer=3,
                max_retained_per_layer=5,
            ),
        )


def test_nonuniform_budget_can_initialize_a_balanced_integer_allocation():
    result = run_evoesap(_fitness, _rankings(), config=_config(total_retained=25))
    assert result.retained_counts[0].tolist() == [5, 4, 4, 4, 4, 4]


def test_invalid_ranking_and_infeasible_bounds_are_rejected():
    invalid = _rankings()
    invalid[0, 0] = invalid[0, 1]
    with pytest.raises(ValueError, match="permutation"):
        run_evoesap(_fitness, invalid, config=_config())
    with pytest.raises(ValueError, match="infeasible"):
        run_evoesap(_fitness, _rankings(), config=_config(total_retained=100))


@pytest.mark.parametrize(
    "change",
    [
        {"population_size": 4},
        {"elite_size": 0},
        {"generations": -1},
        {"max_transfer": 0},
        {"mutation_cap": 0},
        {"initial_patterns": ("random",)},
        {"seed": True},
        {"total_retained": 24.0},
    ],
)
def test_invalid_config_is_rejected(change):
    with pytest.raises(ValueError):
        _config(**change)

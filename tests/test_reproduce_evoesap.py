from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from dsai6707.evoesap import EvoESAPConfig
from scripts import reproduce_evoesap as subject


def test_search_config_restores_tuple_pattern_setting():
    config = EvoESAPConfig(total_retained=12, min_retained_per_layer=1, max_retained_per_layer=3)
    values = asdict(config)
    values["initial_patterns"] = list(values["initial_patterns"])
    restored = subject.search_config({"search": values})
    assert restored == config
    assert restored.evaluation_budget == 256


def test_shortlist_is_descending_esap_and_always_contains_uniform():
    scores = np.arange(30, dtype=float) / 30
    ids = subject.shortlist(scores)
    assert ids == list(range(29, 6, -1)) + [0]
    assert len(set(ids)) == 24


def test_shortlist_breaks_search_ties_by_candidate_index():
    assert subject.shortlist(np.ones(30)) == list(range(24))
    with pytest.raises(ValueError, match="invalid ESAP"):
        subject.shortlist(np.asarray([np.nan] * 24))


def test_holdout_selection_maximizes_esap_and_uses_mask_ties_not_test_accuracy():
    rows = [
        {"candidate_id": 0, "esap": 0.8, "mask_sha256": "z", "test": 1.0},
        {"candidate_id": 1, "esap": 0.8, "mask_sha256": "a", "test": 0.0},
        {"candidate_id": 2, "esap": 0.7, "mask_sha256": "b", "test": 1.0},
    ]
    assert subject.select_holdout(rows)["candidate_id"] == 1


def test_apply_mask_uses_raw_mask_for_ernie():
    masks = []
    controller = SimpleNamespace(apply=masks.append)
    mask = np.ones((2, 4), dtype=bool)
    subject.apply_mask(controller, "ernie", mask)
    assert masks[0] is mask


def test_apply_mask_builds_granite_routing_plan(monkeypatch):
    from scripts import run_pruning_28of32_taskaware_nsga2 as shared

    mask = np.ones((2, 4), dtype=bool)
    sentinel = object()
    monkeypatch.setattr(
        shared, "_routing_plan", lambda values: sentinel if values is mask else None
    )
    values = []
    subject.apply_mask(SimpleNamespace(apply=values.append), "granite", mask)
    assert values == [sentinel]


def test_gold_examples_only_uses_gold_continuation_and_preserves_task():
    question = SimpleNamespace(
        task_name="arc_easy",
        gold_index=1,
        requests=[
            SimpleNamespace(full_ids=(1, 2, 99), context_ids=(1, 2)),
            SimpleNamespace(full_ids=(1, 2, 3, 4), context_ids=(1, 2)),
        ],
    )
    result = subject.gold_examples(SimpleNamespace(questions={"search": [question]}), "search")
    assert result[0].task == "arc_easy"
    assert result[0].input_ids == (1, 2, 3, 4)
    assert result[0].answer_start == 2


def evaluation_fixture():
    frozen = {"mask_sha256": "mask", "protocol_sha256": "protocol"}
    artifact = {
        "selection_uses_test": False,
        "limit": None,
        "mask_sha256": "mask",
        "freeze_sha256": "freeze",
        "protocol_sha256": "protocol",
        "primary_metric": "acc_norm",
        "is_synthetic": False,
        "dataset_evidence": {task: {"evaluation_split": "test"} for task in subject.TASKS},
        "cost": {
            "sample_counts": {"arc_easy": {"effective": 2376}, "openbookqa": {"effective": 500}}
        },
        "scores": {"tasks": {"arc_easy": 0.4, "openbookqa": 0.6}, "mean": 0.5},
    }
    return artifact, frozen


def test_evaluation_requires_full_train_frozen_test():
    artifact, frozen = evaluation_fixture()
    subject.verify_evaluation(artifact, frozen, "freeze")
    artifact["selection_uses_test"] = True
    with pytest.raises(RuntimeError, match="train-frozen"):
        subject.verify_evaluation(artifact, frozen, "freeze")


def test_evaluation_rejects_partial_test():
    artifact, frozen = evaluation_fixture()
    artifact["cost"]["sample_counts"]["arc_easy"]["effective"] = 100
    with pytest.raises(RuntimeError, match="incomplete"):
        subject.verify_evaluation(artifact, frozen, "freeze")


def test_evaluation_rejects_incorrect_task_average():
    artifact, frozen = evaluation_fixture()
    artifact["scores"]["mean"] = 0.6
    with pytest.raises(RuntimeError, match="mean test"):
        subject.verify_evaluation(artifact, frozen, "freeze")


def test_published_evoesap_statistics_match_twelve_full_tests():
    if not (subject.ARTIFACTS / "cases.json").exists():
        pytest.skip("released artifacts are verified inside the generated repository")
    result = subject.statistics_summary()
    assert len(result["cases"]) == 4
    assert all(
        row["n"] == 3 and row["seeds"] == [6707, 6708, 6709] for row in result["cases"].values()
    )
    assert result["cases"]["ernie_48"]["metrics"]["avg"]["mean_percent"] == pytest.approx(
        59.53, abs=0.005
    )
    assert result["cases"]["ernie_32"]["metrics"]["avg"]["mean_percent"] == pytest.approx(
        48.15, abs=0.005
    )


@pytest.mark.parametrize("base", ["ernie_48", "ernie_32", "granite_24", "granite_16"])
@pytest.mark.parametrize("seed", [6707, 6708, 6709])
def test_cpu_replay_matches_every_released_evoesap_candidate(base, seed):
    if not (subject.ARTIFACTS / "cases.json").exists():
        pytest.skip("released artifacts are verified inside the generated repository")
    result = subject.replay(f"{base}_seed_{seed}")
    assert result["candidates"] == 256
    assert result["selection_matches"] is True

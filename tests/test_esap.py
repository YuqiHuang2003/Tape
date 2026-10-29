from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from dsai6707.esap import (
    ESAPEvaluator,
    TeacherForcedExample,
    aggregate_esap,
    probability_overlap,
)
from dsai6707.task_choice_proxy import canonical_sha256

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _example(task="arc_easy", ids=(1, 2, 3, 4), start=2):
    return TeacherForcedExample(task=task, input_ids=ids, answer_start=start)


def test_overlap_is_full_vocabulary_bounded_identity_and_one_minus_tv():
    teacher = np.array([[0.7, 0.2, 0.1], [0.0, 0.25, 0.75]], dtype=np.float32)
    candidate = np.array([[0.3, 0.4, 0.3], [0.6, 0.2, 0.2]], dtype=np.float32)
    overlap = probability_overlap(teacher, candidate)
    np.testing.assert_allclose(overlap, [0.6, 0.4], atol=3e-8, rtol=0)
    np.testing.assert_allclose(
        overlap, 1 - 0.5 * np.abs(teacher - candidate).sum(axis=1), atol=1e-7
    )
    np.testing.assert_allclose(probability_overlap(teacher, teacher), 1, atol=1e-7)
    np.testing.assert_array_equal(probability_overlap([[1, 0]], [[0, 1]]), [0])
    assert np.all((overlap >= 0) & (overlap <= 1))


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ([[1.0]], [[0.5, 0.5]]),
        ([0.5, 0.5], [0.5, 0.5]),
        (np.empty((0, 2)), np.empty((0, 2))),
        ([[0.5, 0.3]], [[0.5, 0.5]]),
        ([[1.1, -0.1]], [[0.5, 0.5]]),
        ([[float("nan"), 0]], [[0.5, 0.5]]),
        ([[float("inf"), 0]], [[0.5, 0.5]]),
    ],
)
def test_overlap_rejects_invalid_probability_inputs(first, second):
    with pytest.raises(ValueError):
        probability_overlap(first, second)


def test_aggregation_means_positions_then_questions_not_global_tokens():
    examples = (_example(ids=(1, 2), start=1), _example(ids=(1, 2, 3, 4), start=1))
    result = aggregate_esap(examples, [[0], [1, 1, 1]])
    assert result["esap"] == 0.5
    assert result["esap"] != 0.75
    assert result["tasks"]["arc_easy"] == {
        "esap": 0.5,
        "num_questions": 2,
        "answer_tokens": 4,
    }
    assert result["lower_is_better"] is False


def test_explicit_weights_balance_tasks_instead_of_overweighting_larger_task():
    examples = (_example("a"), _example("a"), _example("b"))
    token_values = [[0, 0], [0, 0], [1, 1]]
    assert aggregate_esap(examples, token_values)["fitness"] == pytest.approx(1 / 3)
    result = aggregate_esap(examples, token_values, task_weights={"a": 1, "b": 1})
    assert result["fitness"] == 0.5
    assert result["dataset_esap"] == pytest.approx(1 / 3)
    assert result["task_weights"] == {"a": 0.5, "b": 0.5}
    assert aggregate_esap(examples, token_values, task_weights={"a": 3, "b": 1})["esap"] == 0.25


@pytest.mark.parametrize("weights", [{}, {"a": -1}, {"a": 0}, {"a": float("inf")}])
def test_aggregation_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        aggregate_esap([_example("a")], [[1, 1]], task_weights=weights)


@pytest.mark.parametrize("values", [[], [[1]], [[1, float("nan")]], [[1, -0.1]], [[1, 1.1]]])
def test_aggregation_rejects_missing_or_invalid_answer_positions(values):
    with pytest.raises(ValueError):
        aggregate_esap([_example()], values)


def test_example_is_immutable_integer_tokenization_with_semantic_fingerprint():
    raw = np.array([1, 2, 3])
    example = _example(ids=raw, start=1)
    raw[0] = 9
    assert example.input_ids == (1, 2, 3)
    assert example.sha256 == canonical_sha256(
        {"task": "arc_easy", "input_ids": [1, 2, 3], "answer_start": 1}
    )
    assert example.sha256 != _example("other", ids=(1, 2, 3), start=1).sha256
    assert example.sha256 != _example(ids=(1, 2, 3), start=2).sha256


@pytest.mark.parametrize(
    ("ids", "start"),
    [
        ((1, 2), 0),
        ((1, 2), 2),
        ((1, 2), True),
        ((1, 2), 1.0),
        ((1, -2), 1),
        ((1, 2.0), 1),
        ((1, True), 1),
        (((1, 2),), 1),
    ],
)
def test_example_rejects_invalid_tokens_or_answer_start(ids, start):
    with pytest.raises(ValueError):
        _example(ids=ids, start=start)


def _toy_model():
    import torch

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.arange(25).reshape(5, 5).float() / 7)
            self.calls = []
            self.candidate = False

        def forward(self, *, input_ids, attention_mask, use_cache):
            assert use_cache is False
            assert not torch.is_grad_enabled()
            assert torch.is_inference_mode_enabled()
            self.calls.append((input_ids.clone(), attention_mask.clone()))
            # Depend on each causal token and its absolute time index, so
            # off-by-one answer selection or truncation changes the metric.
            positions = torch.arange(input_ids.shape[1]).reshape(1, -1, 1)
            logits = self.weight[input_ids] + positions * torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5])
            if self.candidate:
                logits = logits + torch.tensor([0.8, -0.4, 0.2, -0.6, 0.0])
            return SimpleNamespace(logits=logits)

    return ToyModel()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch integration test")
def test_evaluator_causal_alignment_left_truncation_sorting_and_frozen_model():
    import torch

    model = _toy_model()
    model.train()
    original = model.weight.detach().clone()
    parameter_ids = tuple(id(value) for value in model.parameters())
    examples = (
        _example("a", ids=(4, 3, 2, 1, 0, 1), start=4),
        _example("b", ids=(1, 3, 2), start=1),
    )
    evaluator = ESAPEvaluator(model, examples, max_length=3, batch_size=2, pad_token_id=4)
    assert evaluator.model is model
    assert not model.training
    assert not model.weight.requires_grad
    info = evaluator.cache_teacher()
    torch.testing.assert_close(model.calls[0][0], torch.tensor([[2, 1, 0], [1, 3, 4]]))
    torch.testing.assert_close(model.calls[0][1], torch.tensor([[1, 1, 1], [1, 1, 0]]))
    assert info["teacher_cache_bytes"] == 4 * 5 * 4
    assert info["probability_dtype"] == "float32"
    assert info["vocabulary_size"] == 5
    assert len(info["example_sha256"]) == 2
    assert evaluator.evaluate()["fitness"] == pytest.approx(1, abs=1e-7)

    model.candidate = True
    result = evaluator.evaluate()
    ids, _ = model.calls[0]
    positions = torch.arange(3).reshape(1, -1, 1)
    logits = original[ids] + positions * torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5])
    teacher = torch.softmax(logits, -1).numpy()
    candidate = torch.softmax(logits + torch.tensor([0.8, -0.4, 0.2, -0.6, 0.0]), -1).numpy()
    expected_a = np.minimum(teacher[0, [1, 2]], candidate[0, [1, 2]]).sum(axis=1).mean()
    expected_b = np.minimum(teacher[1, [0, 1]], candidate[1, [0, 1]]).sum(axis=1).mean()
    assert result["tasks"]["a"]["esap"] == pytest.approx(expected_a)
    assert result["tasks"]["b"]["esap"] == pytest.approx(expected_b)
    assert result["esap"] == pytest.approx((expected_a + expected_b) / 2)
    assert tuple(id(value) for value in model.parameters()) == parameter_ids
    torch.testing.assert_close(model.weight, original, rtol=0, atol=0)
    assert model.weight.grad is None
    assert all(array.dtype == np.float32 for array in evaluator._teacher)
    assert all(not array.flags.writeable for array in evaluator._teacher)
    assert result["evidence"]["examples_sha256"] == info["examples_sha256"]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch integration test")
def test_teacher_cache_required_cannot_be_overwritten_and_can_be_explicitly_cleared():
    evaluator = ESAPEvaluator(_toy_model(), [_example()], max_length=8)
    with pytest.raises(RuntimeError, match="cache_teacher"):
        evaluator.evaluate()
    evaluator.cache_teacher()
    with pytest.raises(RuntimeError, match="already exists"):
        evaluator.cache_teacher()
    evaluator.clear_teacher_cache()
    assert evaluator.evidence()["teacher_cache_bytes"] == 0
    evaluator.cache_teacher()
    assert evaluator.evaluate()["esap"] == pytest.approx(1, abs=1e-7)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch integration test")
def test_teacher_cache_memory_limit_and_long_answer_fail_without_partial_cache():
    evaluator = ESAPEvaluator(_toy_model(), [_example()], max_length=8, max_cache_bytes=39)
    with pytest.raises(MemoryError, match="needs 40 bytes"):
        evaluator.cache_teacher()
    assert evaluator.evidence()["teacher_cache_bytes"] == 0
    with pytest.raises(RuntimeError):
        evaluator.evaluate()
    with pytest.raises(ValueError, match="answer length exceeds"):
        ESAPEvaluator(_toy_model(), [_example()], max_length=1)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="optional PyTorch integration test")
def test_multiple_batches_restore_original_example_order_and_task_means():
    import torch

    model = _toy_model()
    examples = (
        _example("a", ids=(1, 2), start=1),
        _example("b", ids=(2, 1, 3, 4, 0), start=2),
        _example("a", ids=(4, 3, 0), start=1),
    )
    evaluator = ESAPEvaluator(
        model, examples, max_length=8, batch_size=1, task_weights={"a": 1, "b": 1}
    )
    evaluator.cache_teacher()
    model.candidate = True
    result = evaluator.evaluate()
    values = []
    for example in examples:
        ids = torch.tensor([example.input_ids[:-1]])
        positions = torch.arange(ids.shape[1]).reshape(1, -1, 1)
        logits = model.weight[ids] + positions * torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5])
        teacher = torch.softmax(logits, dim=-1)
        candidate = torch.softmax(logits + torch.tensor([0.8, -0.4, 0.2, -0.6, 0.0]), dim=-1)
        overlap = torch.minimum(teacher, candidate).sum(-1)
        values.append(overlap[0, example.answer_start - 1 :].numpy())
    expected = aggregate_esap(examples, values, task_weights={"a": 1, "b": 1})
    assert result["esap"] == pytest.approx(expected["esap"])
    for task in ("a", "b"):
        assert result["tasks"][task] == pytest.approx(expected["tasks"][task])
    assert evaluator.evidence()["num_batches"] == 3

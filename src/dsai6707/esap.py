"""Full-vocabulary, teacher-forced Expected Speculative Acceptance Proxy.

This implements ESAP in EvoESAP (arXiv:2603.06003v1, equations 14--16):
``sum_v min(p_teacher(v), p_candidate(v))`` at gold-answer prediction
positions, first averaged within each example and then across examples.
No decoding, gradients, weight updates, model copies, vocabulary truncation,
temperature changes, or probability-cache quantization are used.

The caller owns the pruning controller.  Cache the teacher while the full
model is active, apply candidate masks externally, and reuse the same resident
model for evaluation.  Full-vocabulary teacher probabilities are retained on
the CPU in float32; candidate probabilities are released after each batch.
PyTorch is optional at import time and required only for model execution.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from dsai6707.task_choice_proxy import canonical_sha256


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


@dataclass(frozen=True)
class TeacherForcedExample:
    """One tokenized context and its gold continuation, before truncation.

    ``answer_start`` is the index of the first answer token in ``input_ids``.
    The causal predictions to score are therefore positions
    ``answer_start - 1`` through ``len(input_ids) - 2``.  Both a non-empty
    context and a non-empty answer are required.
    """

    task: str
    input_ids: tuple[int, ...]
    answer_start: int

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task:
            raise ValueError("task must be a non-empty string")
        values = np.asarray(self.input_ids)
        if values.ndim != 1:
            raise ValueError("input_ids must be a one-dimensional integer sequence")
        # Validate the original values: NumPy coercion would silently turn a
        # mixed (integer, bool) sequence into integers before validation.
        ids = tuple(_integer(value, "input_ids") for value in self.input_ids)
        start = _integer(self.answer_start, "answer_start", minimum=1)
        if start >= len(ids):
            raise ValueError("answer_start must leave a non-empty gold answer")
        object.__setattr__(self, "input_ids", ids)
        object.__setattr__(self, "answer_start", start)

    @property
    def answer_tokens(self) -> int:
        return len(self.input_ids) - self.answer_start

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "input_ids": list(self.input_ids),
            "answer_start": self.answer_start,
        }


def probability_overlap(teacher: Any, candidate: Any) -> np.ndarray:
    """Return ESAP for each row of two normalized full-vocabulary matrices.

    The result equals ``1 - total_variation`` up to float32 roundoff.  Only
    roundoff is clipped; non-probability inputs are rejected, not normalized.
    """

    first = np.asarray(teacher)
    second = np.asarray(candidate)
    if first.ndim != 2 or first.shape != second.shape or 0 in first.shape:
        raise ValueError("teacher and candidate must have equal, non-empty 2D shapes")
    for values in (first, second):
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("distributions must contain finite, non-negative probabilities")
        if not np.allclose(values.sum(axis=1, dtype=np.float64), 1.0, rtol=0, atol=2e-6):
            raise ValueError("each full-vocabulary distribution must sum to one")
    overlap = np.minimum(first, second).sum(axis=1, dtype=np.float64)
    return np.clip(overlap, 0.0, 1.0)


def aggregate_esap(
    examples: Sequence[TeacherForcedExample],
    per_token_scores: Sequence[Sequence[float] | np.ndarray],
    *,
    task_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Average answer positions, then examples; optionally balance tasks.

    With no task weights this is exactly the paper's sample-mean aggregation.
    Explicit equal task weights give task-balanced means even when tasks have
    different example counts.  Long answers never receive more sample weight.
    """

    if not examples or len(examples) != len(per_token_scores):
        raise ValueError("one token-score array is required for each non-empty example")
    by_task: dict[str, list[float]] = {}
    token_counts: dict[str, int] = {}
    question_scores: list[float] = []
    for example, raw in zip(examples, per_token_scores, strict=True):
        values = np.asarray(raw, dtype=np.float64)
        if values.shape != (example.answer_tokens,):
            raise ValueError("token scores must cover every gold-answer position exactly once")
        if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
            raise ValueError("token ESAP scores must be finite values in [0, 1]")
        value = float(values.mean())
        by_task.setdefault(example.task, []).append(value)
        token_counts[example.task] = token_counts.get(example.task, 0) + values.size
        question_scores.append(value)

    tasks = {
        task: {
            "esap": float(np.mean(scores)),
            "num_questions": len(scores),
            "answer_tokens": token_counts[task],
        }
        for task, scores in by_task.items()
    }
    dataset_esap = float(np.mean(question_scores))
    if task_weights is None:
        weights = {task: len(scores) / len(examples) for task, scores in by_task.items()}
        fitness = dataset_esap
        aggregation = "mean_examples_after_mean_answer_positions"
    else:
        if set(task_weights) != set(tasks):
            raise ValueError("task_weights must name exactly the tasks in examples")
        try:
            weights = {task: float(task_weights[task]) for task in tasks}
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("task weights must be finite non-negative numbers") from exc
        if any(not math.isfinite(value) or value < 0 for value in weights.values()):
            raise ValueError("task weights must be finite non-negative numbers")
        total = sum(weights.values())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("task weights must have a positive finite sum")
        weights = {task: value / total for task, value in weights.items()}
        fitness = sum(weights[task] * float(tasks[task]["esap"]) for task in tasks)
        aggregation = "weighted_mean_task_means_after_mean_answer_positions"
    return {
        "schema_version": "full-vocabulary-esap-v1",
        "fitness": fitness,
        "esap": fitness,
        "dataset_esap": dataset_esap,
        "fitness_name": "full_vocabulary_expected_speculative_acceptance_proxy",
        "lower_is_better": False,
        "aggregation": aggregation,
        "task_weights": weights,
        "num_questions": len(examples),
        "answer_tokens": sum(token_counts.values()),
        "tasks": tasks,
    }


@dataclass(frozen=True)
class _Batch:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    question_indices: tuple[int, ...]
    row_indices: np.ndarray
    time_indices: np.ndarray


class ESAPEvaluator:
    """Evaluate masks on one frozen model using a bounded CPU teacher cache.

    ``max_length`` is the maximum model-input length, excluding the final
    target token.  Left truncation matches HFLM 0.4.8's causal likelihood
    request: ``full_ids[-(max_length + 1):][:-1]``.  Answers longer than this
    maximum are rejected rather than partially scored.  Sorting and padding
    are fixed once, so the full model and all masks use identical batches.
    """

    def __init__(
        self,
        model: Any,
        examples: Sequence[TeacherForcedExample],
        *,
        max_length: int,
        batch_size: int = 4,
        pad_token_id: int = 0,
        device: Any = None,
        task_weights: Mapping[str, float] | None = None,
        max_cache_bytes: int = 2 * 1024**3,
    ) -> None:
        if not examples:
            raise ValueError("at least one teacher-forced example is required")
        if not all(isinstance(example, TeacherForcedExample) for example in examples):
            raise TypeError("examples must contain TeacherForcedExample instances")
        self.model = model
        self.examples = tuple(examples)
        self.max_length = _integer(max_length, "max_length", minimum=1)
        self.batch_size = _integer(batch_size, "batch_size", minimum=1)
        self.pad_token_id = _integer(pad_token_id, "pad_token_id")
        self.max_cache_bytes = _integer(max_cache_bytes, "max_cache_bytes", minimum=1)
        self.task_weights = None if task_weights is None else dict(task_weights)
        # Validate weighting before performing any expensive forward passes.
        aggregate_esap(
            self.examples,
            [np.ones(example.answer_tokens) for example in self.examples],
            task_weights=self.task_weights,
        )
        self._batches = self._prepare_batches()
        self._teacher: tuple[np.ndarray, ...] | None = None
        self._vocabulary_size: int | None = None
        self._cache_bytes = 0
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if device is None:
            try:
                device = self.model.get_input_embeddings().weight.device
            except (AttributeError, NotImplementedError):
                try:
                    device = next(self.model.parameters()).device
                except StopIteration:
                    device = "cpu"
        self.device = device

    def _prepare_batches(self) -> tuple[_Batch, ...]:
        prepared = []
        for index, example in enumerate(self.examples):
            if example.answer_tokens > self.max_length:
                raise ValueError("gold answer length exceeds max_length")
            ids = example.input_ids[-(self.max_length + 1) : -1]
            prepared.append((index, example, ids))
        prepared.sort(key=lambda row: (-len(row[1].input_ids), row[1].input_ids))
        batches = []
        for start in range(0, len(prepared), self.batch_size):
            items = prepared[start : start + self.batch_size]
            width = max(len(ids) for _, _, ids in items)
            ids_array = np.full((len(items), width), self.pad_token_id, dtype=np.int64)
            attention_mask = np.zeros_like(ids_array)
            rows: list[int] = []
            times: list[int] = []
            for row, (_, example, ids) in enumerate(items):
                ids_array[row, : len(ids)] = ids
                attention_mask[row, : len(ids)] = 1
                rows.extend([row] * example.answer_tokens)
                times.extend(range(len(ids) - example.answer_tokens, len(ids)))
            batches.append(
                _Batch(
                    input_ids=ids_array,
                    attention_mask=attention_mask,
                    question_indices=tuple(index for index, _, _ in items),
                    row_indices=np.asarray(rows, dtype=np.int64),
                    time_indices=np.asarray(times, dtype=np.int64),
                )
            )
        return tuple(batches)

    def _probabilities(self, batch: _Batch) -> np.ndarray:
        import torch

        with torch.inference_mode():
            ids = torch.as_tensor(batch.input_ids, device=self.device)
            attention = torch.as_tensor(batch.attention_mask, device=self.device)
            output = self.model(input_ids=ids, attention_mask=attention, use_cache=False)
            logits = output.logits
            if (
                logits.ndim != 3
                or tuple(logits.shape[:2]) != tuple(ids.shape)
                or logits.shape[-1] < 1
            ):
                raise ValueError("model logits must have shape [batch, input_length, vocabulary]")
            rows = torch.as_tensor(batch.row_indices, device=logits.device)
            times = torch.as_tensor(batch.time_indices, device=logits.device)
            # Float32 softmax over the complete vocabulary, never over choices
            # or a top-k subset.  Only answer-position probabilities leave GPU.
            selected = logits[rows, times].float()
            probabilities = selected.softmax(dim=-1).cpu().numpy()
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("model produced non-finite answer-token probabilities")
        probabilities.setflags(write=False)
        return probabilities

    def evidence(self) -> dict[str, Any]:
        """Return tokenization/batching fingerprints without raw prompt text."""

        return {
            "source": "https://arxiv.org/html/2603.06003v1#S3.SS4",
            "examples_sha256": canonical_sha256([example.to_dict() for example in self.examples]),
            "example_sha256": [example.sha256 for example in self.examples],
            "answer_positions_only": True,
            "full_vocabulary": True,
            "probability_dtype": "float32",
            "teacher_cache_device": "cpu",
            "teacher_cache_bytes": self._cache_bytes,
            "vocabulary_size": self._vocabulary_size,
            "num_questions": len(self.examples),
            "answer_tokens": sum(example.answer_tokens for example in self.examples),
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "pad_token_id": self.pad_token_id,
            "num_batches": len(self._batches),
            "model_input_positions": sum(
                int(batch.attention_mask.sum()) for batch in self._batches
            ),
            "padded_input_positions": sum(int(batch.input_ids.size) for batch in self._batches),
        }

    def clear_teacher_cache(self) -> None:
        self._teacher = None
        self._vocabulary_size = None
        self._cache_bytes = 0

    def cache_teacher(self) -> dict[str, Any]:
        """Cache full-model probabilities once; the caller must activate it.

        Repeated calls are rejected to avoid accidentally overwriting the
        teacher with a pruned candidate.  Explicitly clear first if needed.
        A failed build does not leave a partially usable teacher cache.
        """

        if self._teacher is not None:
            raise RuntimeError(
                "teacher cache already exists; clear it explicitly before rebuilding"
            )
        self.model.eval()
        values = []
        vocabulary_size = None
        cache_bytes = 0
        for batch in self._batches:
            probabilities = self._probabilities(batch)
            vocabulary = probabilities.shape[1]
            if vocabulary_size is not None and vocabulary != vocabulary_size:
                raise ValueError("teacher vocabulary size changed between batches")
            vocabulary_size = vocabulary
            answer_tokens = sum(example.answer_tokens for example in self.examples)
            required_bytes = answer_tokens * vocabulary * 4
            if required_bytes > self.max_cache_bytes:
                raise MemoryError(
                    f"float32 teacher cache needs {required_bytes} bytes, "
                    f"exceeding max_cache_bytes={self.max_cache_bytes}"
                )
            cache_bytes += probabilities.nbytes
            values.append(probabilities)
        self._teacher = tuple(values)
        self._vocabulary_size = vocabulary_size
        self._cache_bytes = cache_bytes
        return self.evidence()

    def evaluate(self) -> dict[str, Any]:
        """Score the externally activated candidate on the cached contexts."""

        if self._teacher is None:
            raise RuntimeError("cache_teacher must run with the full model before evaluation")
        self.model.eval()
        question_scores: list[np.ndarray | None] = [None] * len(self.examples)
        for batch, teacher in zip(self._batches, self._teacher, strict=True):
            candidate = self._probabilities(batch)
            token_values = probability_overlap(teacher, candidate)
            offset = 0
            for index in batch.question_indices:
                count = self.examples[index].answer_tokens
                question_scores[index] = token_values[offset : offset + count]
                offset += count
            del candidate
        if any(value is None for value in question_scores):  # pragma: no cover
            raise RuntimeError("internal batches did not score every example")
        result = aggregate_esap(self.examples, question_scores, task_weights=self.task_weights)
        result["evidence"] = self.evidence()
        return result

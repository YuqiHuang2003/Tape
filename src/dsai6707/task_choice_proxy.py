"""Deterministic multiple-choice proxy data for expert-pruning search.

This module deliberately stops at the boundary between immutable CPU-side
data preparation and model execution.  It reproduces the causal
log-likelihood request semantics used by lm-eval 0.4.8, packs those requests
for a GPU caller, and reduces returned choice log-likelihoods into a smooth,
task-balanced search objective.

No model is loaded here.  In particular, a parallel search should give each
CUDA lane its own model, pruning controller, and stream, then reuse the packed
arrays produced by :func:`pack_choice_requests` across candidates.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np

ORDERED_CONTENT_HASH_SPEC = "sha256(index<TAB>canonical-json<LF>)"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"cannot encode {type(value).__name__} as canonical JSON")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        default=_json_default,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of a stable, UTF-8 canonical JSON representation."""

    return sha256(_canonical_json(value)).hexdigest()


def ordered_document_content_hash(documents: Sequence[Mapping[str, Any]]) -> str:
    """Hash exact ordered rows using the project's dataset evidence convention."""

    digest = sha256()
    for index, document in enumerate(documents):
        digest.update(str(index).encode("ascii"))
        digest.update(b"\t")
        digest.update(_canonical_json(document))
        digest.update(b"\n")
    return digest.hexdigest()


def _require_nonnegative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if not math.isfinite(numeric) or numeric != result or result < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return result


def _token_tuple(values: Any, *, field: str) -> tuple[int, ...]:
    try:
        tokens = tuple(_require_nonnegative_integer(value, field=field) for value in values)
    except TypeError as exc:
        raise ValueError(f"{field} must be a sequence of token ids") from exc
    return tokens


@dataclass(frozen=True)
class EncodedCausalPair:
    """One context/continuation pair after lm-eval-compatible tokenization."""

    context: str
    continuation: str
    context_ids: tuple[int, ...]
    continuation_ids: tuple[int, ...]

    @property
    def full_ids(self) -> tuple[int, ...]:
        return self.context_ids + self.continuation_ids


def encode_causal_pair(
    tokenizer: Any,
    context: str,
    continuation: str,
    *,
    prefix_token_id: int | None = None,
) -> EncodedCausalPair:
    """Encode a causal log-likelihood pair as lm-eval 0.4.8 does.

    Trailing spaces are moved from the context to the continuation before
    tokenization.  For a non-empty context, the complete string is tokenized
    and the separately tokenized context is removed.  This is important for
    tokenizers whose tokens can cross the context/continuation boundary.

    lm-eval handles an empty context specially by inserting its prefix token.
    Callers using that case must therefore provide ``prefix_token_id``.
    """

    if not isinstance(context, str) or not isinstance(continuation, str):
        raise TypeError("context and continuation must be strings")
    if not continuation:
        raise ValueError("continuation must not be empty")

    original_context = context
    original_continuation = continuation
    if context == "":
        if prefix_token_id is None:
            raise ValueError("an empty context requires prefix_token_id")
        context_ids = (_require_nonnegative_integer(prefix_token_id, field="prefix_token_id"),)
        continuation_ids = _token_tuple(
            tokenizer.encode(continuation, add_special_tokens=False),
            field="continuation_ids",
        )
    else:
        trailing_spaces = len(context) - len(context.rstrip())
        if trailing_spaces:
            continuation = context[-trailing_spaces:] + continuation
            context = context[:-trailing_spaces]
        whole_ids = _token_tuple(
            tokenizer.encode(context + continuation, add_special_tokens=False),
            field="whole_ids",
        )
        context_ids = _token_tuple(
            tokenizer.encode(context, add_special_tokens=False),
            field="context_ids",
        )
        if whole_ids[: len(context_ids)] != context_ids:
            raise ValueError("context tokenization is not a prefix of context+continuation")
        continuation_ids = whole_ids[len(context_ids) :]

    if not context_ids:
        raise ValueError("the encoded context must contain at least one token")
    if not continuation_ids:
        raise ValueError("the encoded continuation must contain at least one token")
    return EncodedCausalPair(
        context=original_context,
        continuation=original_continuation,
        context_ids=context_ids,
        continuation_ids=continuation_ids,
    )


@dataclass(frozen=True)
class ChoiceRequest:
    """Tokenized log-likelihood request for one answer choice."""

    choice_index: int
    choice_text: str
    choice_character_length: int
    context: str
    continuation: str
    context_ids: tuple[int, ...]
    continuation_ids: tuple[int, ...]

    @property
    def full_ids(self) -> tuple[int, ...]:
        return self.context_ids + self.continuation_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "choice_index": self.choice_index,
            "choice_text": self.choice_text,
            "choice_character_length": self.choice_character_length,
            "context": self.context,
            "continuation": self.continuation,
            "context_ids": list(self.context_ids),
            "continuation_ids": list(self.continuation_ids),
        }


@dataclass(frozen=True)
class ChoiceQuestion:
    """One frozen multiple-choice question and all of its requests."""

    task_name: str
    split: str
    document_index: int
    document_sha256: str
    gold_index: int
    context: str
    choices: tuple[str, ...]
    requests: tuple[ChoiceRequest, ...]

    @property
    def question_id(self) -> str:
        return f"{self.task_name}:{self.split}:{self.document_index}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "split": self.split,
            "document_index": self.document_index,
            "document_sha256": self.document_sha256,
            "gold_index": self.gold_index,
            "context": self.context,
            "choices": list(self.choices),
            "requests": [request.to_dict() for request in self.requests],
        }


@dataclass(frozen=True)
class GoldStratifiedSplit:
    """Deterministic, disjoint search and holdout document indices."""

    task_name: str
    seed: int
    required_choice_count: int
    search_per_gold: int
    holdout_per_gold: int
    dataset_rows: int
    eligible_rows: int
    dataset_content_sha256: str
    search_indices: tuple[int, ...]
    holdout_indices: tuple[int, ...]
    search_gold_indices: tuple[int, ...]
    holdout_gold_indices: tuple[int, ...]
    search_document_sha256: tuple[str, ...]
    holdout_document_sha256: tuple[str, ...]
    selection_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "seed": self.seed,
            "required_choice_count": self.required_choice_count,
            "search_per_gold": self.search_per_gold,
            "holdout_per_gold": self.holdout_per_gold,
            "dataset_rows": self.dataset_rows,
            "eligible_rows": self.eligible_rows,
            "dataset_content_sha256": self.dataset_content_sha256,
            "content_hash_spec": ORDERED_CONTENT_HASH_SPEC,
            "search_indices": list(self.search_indices),
            "holdout_indices": list(self.holdout_indices),
            "search_gold_indices": list(self.search_gold_indices),
            "holdout_gold_indices": list(self.holdout_gold_indices),
            "search_document_sha256": list(self.search_document_sha256),
            "holdout_document_sha256": list(self.holdout_document_sha256),
            "selection_sha256": self.selection_sha256,
        }


def select_gold_stratified_documents(
    task: Any,
    documents: Sequence[Mapping[str, Any]],
    *,
    task_name: str,
    seed: int,
    search_per_gold: int,
    holdout_per_gold: int,
    required_choice_count: int = 4,
) -> GoldStratifiedSplit:
    """Freeze label-balanced search/holdout subsets by canonical document hash.

    Within each gold label, rows are ordered by
    ``sha256(seed, task_name, canonical_document)`` with the source index as a
    deterministic tie-breaker.  Search rows are taken first and holdout rows
    immediately afterward, making overlap impossible by construction.
    """

    if not isinstance(task_name, str) or not task_name:
        raise ValueError("task_name must be a non-empty string")
    seed = _require_nonnegative_integer(seed, field="seed")
    search_per_gold = _require_nonnegative_integer(search_per_gold, field="search_per_gold")
    holdout_per_gold = _require_nonnegative_integer(holdout_per_gold, field="holdout_per_gold")
    required_choice_count = _require_nonnegative_integer(
        required_choice_count, field="required_choice_count"
    )
    if required_choice_count < 2:
        raise ValueError("required_choice_count must be at least two")
    if search_per_gold < 1:
        raise ValueError("search_per_gold must be positive")

    bins: dict[int, list[tuple[str, int, str]]] = defaultdict(list)
    dataset_digest = sha256()
    row_count = 0
    eligible_rows = 0
    for index, document in enumerate(documents):
        row = _canonical_json(document)
        document_digest = sha256(row).hexdigest()
        dataset_digest.update(str(index).encode("ascii"))
        dataset_digest.update(b"\t")
        dataset_digest.update(row)
        dataset_digest.update(b"\n")
        row_count += 1

        choices = tuple(task.doc_to_choice(document))
        if len(choices) != required_choice_count:
            continue
        gold = _require_nonnegative_integer(task.doc_to_target(document), field="gold_index")
        if gold >= required_choice_count:
            raise ValueError(f"document {index} has an out-of-range gold index")
        ranking = sha256(
            str(seed).encode("ascii") + b"\0" + task_name.encode("utf-8") + b"\0" + row
        ).hexdigest()
        bins[gold].append((ranking, index, document_digest))
        eligible_rows += 1

    required_per_gold = search_per_gold + holdout_per_gold
    for gold in range(required_choice_count):
        available = len(bins[gold])
        if available < required_per_gold:
            raise ValueError(
                f"gold label {gold} has {available} eligible rows; "
                f"{required_per_gold} are required"
            )
        bins[gold].sort()

    search_rows: list[tuple[int, int, str]] = []
    holdout_rows: list[tuple[int, int, str]] = []
    for gold in range(required_choice_count):
        selected = bins[gold]
        search_rows.extend(
            (index, gold, document_digest)
            for _, index, document_digest in selected[:search_per_gold]
        )
        holdout_rows.extend(
            (index, gold, document_digest)
            for _, index, document_digest in selected[
                search_per_gold : search_per_gold + holdout_per_gold
            ]
        )
    search_rows.sort()
    holdout_rows.sort()
    search_indices = tuple(row[0] for row in search_rows)
    holdout_indices = tuple(row[0] for row in holdout_rows)
    if set(search_indices) & set(holdout_indices):  # pragma: no cover - construction invariant
        raise RuntimeError("search and holdout selections overlap")

    payload = {
        "task_name": task_name,
        "seed": seed,
        "required_choice_count": required_choice_count,
        "search_per_gold": search_per_gold,
        "holdout_per_gold": holdout_per_gold,
        "dataset_content_sha256": dataset_digest.hexdigest(),
        "search": search_rows,
        "holdout": holdout_rows,
    }
    return GoldStratifiedSplit(
        task_name=task_name,
        seed=seed,
        required_choice_count=required_choice_count,
        search_per_gold=search_per_gold,
        holdout_per_gold=holdout_per_gold,
        dataset_rows=row_count,
        eligible_rows=eligible_rows,
        dataset_content_sha256=dataset_digest.hexdigest(),
        search_indices=search_indices,
        holdout_indices=holdout_indices,
        search_gold_indices=tuple(row[1] for row in search_rows),
        holdout_gold_indices=tuple(row[1] for row in holdout_rows),
        search_document_sha256=tuple(row[2] for row in search_rows),
        holdout_document_sha256=tuple(row[2] for row in holdout_rows),
        selection_sha256=canonical_sha256(payload),
    )


def build_choice_questions(
    task: Any,
    documents: Sequence[Mapping[str, Any]],
    document_indices: Sequence[int],
    tokenizer: Any,
    *,
    task_name: str,
    split: str = "train",
    prefix_token_id: int | None = None,
) -> tuple[ChoiceQuestion, ...]:
    """Construct and tokenize zero-shot lm-eval requests for selected rows."""

    if not task_name or not split:
        raise ValueError("task_name and split must be non-empty")
    questions: list[ChoiceQuestion] = []
    seen: set[int] = set()
    for raw_index in document_indices:
        document_index = _require_nonnegative_integer(raw_index, field="document_index")
        if document_index in seen:
            raise ValueError("document_indices must be unique")
        seen.add(document_index)
        if document_index >= len(documents):
            raise ValueError(f"document index {document_index} is out of range")
        document = documents[document_index]
        choices = tuple(str(choice) for choice in task.doc_to_choice(document))
        if len(choices) < 2 or any(not choice for choice in choices):
            raise ValueError(f"document {document_index} has invalid choices")
        gold = _require_nonnegative_integer(task.doc_to_target(document), field="gold_index")
        if gold >= len(choices):
            raise ValueError(f"document {document_index} has an out-of-range gold index")
        context = task.fewshot_context(document, num_fewshot=0)
        if not isinstance(context, str):
            raise TypeError("zero-shot task context must be a string")
        instances = task.construct_requests(
            document,
            context,
            metadata=(task_name, document_index, 1),
            apply_chat_template=False,
            chat_template=None,
        )
        if not isinstance(instances, list | tuple):
            instances = [instances]
        if len(instances) != len(choices):
            raise ValueError("task did not construct exactly one request per choice")
        instances = sorted(instances, key=lambda instance: int(instance.idx))

        requests: list[ChoiceRequest] = []
        for choice_index, (choice, instance) in enumerate(zip(choices, instances, strict=True)):
            if int(instance.idx) != choice_index:
                raise ValueError("choice request indices are not contiguous from zero")
            arguments = instance.args
            if not isinstance(arguments, tuple) or len(arguments) != 2:
                raise ValueError("choice request must contain a context and continuation")
            request_context, continuation = arguments
            pair = encode_causal_pair(
                tokenizer,
                request_context,
                continuation,
                prefix_token_id=prefix_token_id,
            )
            requests.append(
                ChoiceRequest(
                    choice_index=choice_index,
                    choice_text=choice,
                    # lm-eval 0.4.8 normalizes by Python string length, not tokens.
                    choice_character_length=len(choice),
                    context=pair.context,
                    continuation=pair.continuation,
                    context_ids=pair.context_ids,
                    continuation_ids=pair.continuation_ids,
                )
            )
        row = _canonical_json(document)
        questions.append(
            ChoiceQuestion(
                task_name=task_name,
                split=split,
                document_index=document_index,
                document_sha256=sha256(row).hexdigest(),
                gold_index=gold,
                context=context,
                choices=choices,
                requests=tuple(requests),
            )
        )
    return tuple(questions)


def flatten_choice_requests(questions: Sequence[ChoiceQuestion]) -> tuple[ChoiceRequest, ...]:
    """Flatten questions in question-major, then choice-major order."""

    requests: list[ChoiceRequest] = []
    for question in questions:
        if len(question.requests) != len(question.choices):
            raise ValueError("question request count differs from its choices")
        if tuple(request.choice_index for request in question.requests) != tuple(
            range(len(question.choices))
        ):
            raise ValueError("question requests are not in choice order")
        requests.extend(question.requests)
    return tuple(requests)


def choice_proxy_hashes(questions: Sequence[ChoiceQuestion]) -> dict[str, Any]:
    """Hash semantic question data separately from tokenizer-dependent requests."""

    data = [
        {
            "task_name": question.task_name,
            "split": question.split,
            "document_index": question.document_index,
            "document_sha256": question.document_sha256,
            "gold_index": question.gold_index,
            "context": question.context,
            "choices": list(question.choices),
        }
        for question in questions
    ]
    requests = [question.to_dict() for question in questions]
    return {
        "num_questions": len(questions),
        "num_requests": sum(len(question.requests) for question in questions),
        "data_sha256": canonical_sha256(data),
        "requests_sha256": canonical_sha256(requests),
    }


def _readonly_int64(values: Any) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=np.int64)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PackedChoiceBatch:
    """Right-padded causal inputs and sparse continuation scoring indices."""

    input_ids: np.ndarray
    input_lengths: np.ndarray
    request_indices: np.ndarray
    row_indices: np.ndarray
    time_indices: np.ndarray
    target_ids: np.ndarray
    segment_indices: np.ndarray

    @property
    def num_requests(self) -> int:
        return int(self.request_indices.size)

    @property
    def scored_tokens(self) -> int:
        return int(self.target_ids.size)

    @property
    def model_input_positions(self) -> int:
        return int(self.input_lengths.sum(dtype=np.int64))

    @property
    def padded_input_positions(self) -> int:
        return int(self.input_ids.size)


def pack_choice_requests(
    requests: Sequence[ChoiceRequest],
    *,
    batch_size: int,
    max_length: int,
    pad_token_id: int,
) -> tuple[PackedChoiceBatch, ...]:
    """Context-cache, length-sort, and right-pad requests like HFLM 0.4.8.

    HFLM groups requests by ``context_ids + continuation_ids[:-1]`` and runs
    one representative from every group.  This matters for multiple-choice
    tasks whose answers are often one token: all four choices then reuse the
    same context logits and follow the same bfloat16 batch-shaped numeric path
    as the official evaluator.  Sparse score indices may consequently map
    several output requests to one model-input row.
    """

    batch_size = _require_nonnegative_integer(batch_size, field="batch_size")
    max_length = _require_nonnegative_integer(max_length, field="max_length")
    pad_token_id = _require_nonnegative_integer(pad_token_id, field="pad_token_id")
    if batch_size < 1 or max_length < 1:
        raise ValueError("batch_size and max_length must be positive")
    if not requests:
        raise ValueError("at least one request is required")

    grouped: dict[
        tuple[int, ...],
        list[tuple[int, tuple[int, ...], tuple[int, ...]]],
    ] = {}
    for request_index, request in enumerate(requests):
        context_ids = _token_tuple(request.context_ids, field="context_ids")
        continuation_ids = _token_tuple(request.continuation_ids, field="continuation_ids")
        if not context_ids or not continuation_ids:
            raise ValueError("each request needs non-empty context and continuation tokens")
        if len(continuation_ids) > max_length:
            raise ValueError("continuation length exceeds max_length")
        full_ids = context_ids + continuation_ids
        cache_key = context_ids + continuation_ids[:-1]
        grouped.setdefault(cache_key, []).append((request_index, full_ids, continuation_ids))

    prepared: list[
        tuple[
            tuple[int, ...],
            tuple[int, ...],
            int,
            list[tuple[int, tuple[int, ...], tuple[int, ...]]],
        ]
    ] = []
    for members in grouped.values():
        _, representative_full_ids, representative_continuation = members[0]
        model_input = representative_full_ids[-(max_length + 1) : -1]
        continuation_length = len(representative_continuation)
        if len(model_input) < continuation_length:  # pragma: no cover - guarded above
            raise ValueError("truncation removed required continuation context")
        for _, full_ids, continuation_ids in members:
            if len(continuation_ids) != continuation_length:
                raise ValueError("cached continuations must have equal token lengths")
            if full_ids[:-1] != representative_full_ids[:-1]:
                raise RuntimeError("invalid context-cache group")
        prepared.append(
            (
                representative_full_ids,
                model_input,
                continuation_length,
                members,
            )
        )

    # HFLM sorts one representative per cache key by decreasing full length,
    # then lexicographically by the representative's full token sequence.
    prepared.sort(key=lambda row: (-len(row[0]), row[0]))
    batches: list[PackedChoiceBatch] = []
    for start in range(0, len(prepared), batch_size):
        rows = prepared[start : start + batch_size]
        longest = max(len(row[1]) for row in rows)
        input_ids = np.full((len(rows), longest), pad_token_id, dtype=np.int64)
        input_lengths = np.empty(len(rows), dtype=np.int64)
        request_indices: list[int] = []
        scored_rows: list[int] = []
        scored_times: list[int] = []
        targets: list[int] = []
        segments: list[int] = []
        for batch_row, (
            _representative_full_ids,
            model_input,
            continuation_length,
            members,
        ) in enumerate(rows):
            input_length = len(model_input)
            input_ids[batch_row, :input_length] = model_input
            input_lengths[batch_row] = input_length
            first_position = input_length - continuation_length
            for request_index, _full_ids, continuation_ids in members:
                segment = len(request_indices)
                request_indices.append(request_index)
                for offset, target in enumerate(continuation_ids):
                    scored_rows.append(batch_row)
                    scored_times.append(first_position + offset)
                    targets.append(target)
                    segments.append(segment)
        batches.append(
            PackedChoiceBatch(
                input_ids=_readonly_int64(input_ids),
                input_lengths=_readonly_int64(input_lengths),
                request_indices=_readonly_int64(request_indices),
                row_indices=_readonly_int64(scored_rows),
                time_indices=_readonly_int64(scored_times),
                target_ids=_readonly_int64(targets),
                segment_indices=_readonly_int64(segments),
            )
        )
    return tuple(batches)


def restore_choice_log_likelihoods(
    batches: Sequence[PackedChoiceBatch],
    batch_values: Sequence[Sequence[float] | np.ndarray],
    *,
    num_requests: int,
) -> np.ndarray:
    """Restore batch-row scores to original flattened request order."""

    num_requests = _require_nonnegative_integer(num_requests, field="num_requests")
    if num_requests < 1:
        raise ValueError("num_requests must be positive")
    if len(batches) != len(batch_values):
        raise ValueError("one score vector is required for every packed batch")
    restored = np.full(num_requests, np.nan, dtype=np.float64)
    seen = np.zeros(num_requests, dtype=np.bool_)
    for batch, raw_values in zip(batches, batch_values, strict=True):
        values = np.asarray(raw_values, dtype=np.float64)
        if values.ndim != 1 or values.size != batch.num_requests:
            raise ValueError("batch score vector has the wrong shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("choice log-likelihoods must be finite")
        indices = batch.request_indices
        if np.any(indices < 0) or np.any(indices >= num_requests) or np.any(seen[indices]):
            raise ValueError("packed request indices are duplicated or out of range")
        restored[indices] = values
        seen[indices] = True
    if not np.all(seen):
        raise ValueError("packed batches do not cover every request")
    restored.setflags(write=False)
    return restored


def _choice_cross_entropy(scores: np.ndarray, gold: int) -> float:
    maximum = float(np.max(scores))
    log_sum_exp = maximum + math.log(float(np.exp(scores - maximum).sum()))
    return log_sum_exp - float(scores[gold])


def _log_softmax_numpy(scores: np.ndarray) -> np.ndarray:
    maximum = float(np.max(scores))
    shifted = scores - maximum
    return shifted - math.log(float(np.exp(shifted).sum()))


def choice_proxy_metrics(
    questions: Sequence[ChoiceQuestion],
    choice_log_likelihoods: Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Reduce choice log-likelihoods into task-balanced proxy metrics.

    ``fitness`` is the arithmetic mean of per-task raw choice cross-entropies
    and is minimized.  The raw score uses total continuation log-likelihood,
    exactly matching lm-eval's primary ``acc`` argmax.  The normalized score
    divides by answer *character* length, matching lm-eval 0.4.8 ``acc_norm``.
    """

    if not questions:
        raise ValueError("at least one question is required")
    requests = flatten_choice_requests(questions)
    values = np.asarray(choice_log_likelihoods, dtype=np.float64)
    if values.ndim != 1 or values.size != len(requests):
        raise ValueError("choice_log_likelihoods has the wrong shape")
    if not np.all(np.isfinite(values)):
        raise ValueError("choice log-likelihoods must be finite")

    by_task: dict[str, dict[str, list[float]]] = {}
    offset = 0
    for question in questions:
        count = len(question.choices)
        scores = values[offset : offset + count]
        offset += count
        lengths = np.asarray(
            [request.choice_character_length for request in question.requests],
            dtype=np.float64,
        )
        if np.any(lengths <= 0):
            raise ValueError("choice character lengths must be positive")
        normalized_scores = scores / lengths
        task_values = by_task.setdefault(
            question.task_name,
            {"raw_ce": [], "acc": [], "normalized_ce": [], "acc_norm": []},
        )
        task_values["raw_ce"].append(_choice_cross_entropy(scores, question.gold_index))
        task_values["acc"].append(float(int(np.argmax(scores)) == question.gold_index))
        task_values["normalized_ce"].append(
            _choice_cross_entropy(normalized_scores, question.gold_index)
        )
        task_values["acc_norm"].append(
            float(int(np.argmax(normalized_scores)) == question.gold_index)
        )

    task_metrics: dict[str, dict[str, float | int]] = {}
    for task_name, task_values in by_task.items():
        task_metrics[task_name] = {
            "samples": len(task_values["raw_ce"]),
            "raw_choice_cross_entropy": float(np.mean(task_values["raw_ce"])),
            "acc": float(np.mean(task_values["acc"])),
            "normalized_choice_cross_entropy": float(np.mean(task_values["normalized_ce"])),
            "acc_norm": float(np.mean(task_values["acc_norm"])),
        }

    raw_task_ce = [float(metrics["raw_choice_cross_entropy"]) for metrics in task_metrics.values()]
    raw_task_acc = [float(metrics["acc"]) for metrics in task_metrics.values()]
    normalized_task_ce = [
        float(metrics["normalized_choice_cross_entropy"]) for metrics in task_metrics.values()
    ]
    normalized_task_acc = [float(metrics["acc_norm"]) for metrics in task_metrics.values()]
    fitness = float(np.mean(raw_task_ce))
    return {
        "schema_version": "task-choice-proxy-metrics-v1",
        "fitness": fitness,
        "fitness_name": "task_balanced_raw_choice_cross_entropy",
        "lower_is_better": True,
        "num_tasks": len(task_metrics),
        "num_questions": len(questions),
        "num_choice_requests": len(requests),
        "task_order": list(task_metrics),
        "task_balanced_raw_choice_cross_entropy": fitness,
        "task_balanced_acc": float(np.mean(raw_task_acc)),
        "task_balanced_normalized_choice_cross_entropy": float(np.mean(normalized_task_ce)),
        "task_balanced_acc_norm": float(np.mean(normalized_task_acc)),
        "tasks": task_metrics,
    }


def choice_proxy_distillation_metrics(
    questions: Sequence[ChoiceQuestion],
    candidate_choice_log_likelihoods: Sequence[float] | np.ndarray,
    teacher_choice_log_likelihoods: Sequence[float] | np.ndarray,
    *,
    cross_entropy_weight: float = 0.75,
    kl_weight: float = 0.25,
) -> dict[str, Any]:
    """Score labels and preservation of a teacher's choice distribution.

    For every question this computes supervised cross-entropy from the
    candidate's raw total choice log-likelihoods and
    ``KL(teacher || candidate)`` over the corresponding choice softmaxes.
    Values are averaged within each task, then combined with the requested
    weights.  The returned ``task_objectives`` follow first task appearance in
    ``questions`` and ``fitness`` is their arithmetic mean.
    """

    if not questions:
        raise ValueError("at least one question is required")
    try:
        cross_entropy_weight = float(cross_entropy_weight)
        kl_weight = float(kl_weight)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("objective weights must be finite numbers") from exc
    if (
        not math.isfinite(cross_entropy_weight)
        or not math.isfinite(kl_weight)
        or cross_entropy_weight < 0.0
        or kl_weight < 0.0
        or cross_entropy_weight + kl_weight <= 0.0
    ):
        raise ValueError("objective weights must be non-negative with a positive sum")

    requests = flatten_choice_requests(questions)
    candidate = np.asarray(candidate_choice_log_likelihoods, dtype=np.float64)
    teacher = np.asarray(teacher_choice_log_likelihoods, dtype=np.float64)
    expected_shape = (len(requests),)
    if candidate.shape != expected_shape or teacher.shape != expected_shape:
        raise ValueError("candidate and teacher choice log-likelihoods have the wrong shape")
    if not np.all(np.isfinite(candidate)) or not np.all(np.isfinite(teacher)):
        raise ValueError("candidate and teacher choice log-likelihoods must be finite")

    by_task: dict[str, dict[str, list[float]]] = {}
    offset = 0
    for question in questions:
        count = len(question.choices)
        candidate_scores = candidate[offset : offset + count]
        teacher_scores = teacher[offset : offset + count]
        offset += count
        character_lengths = np.asarray(
            [request.choice_character_length for request in question.requests],
            dtype=np.float64,
        )
        if np.any(character_lengths <= 0):
            raise ValueError("choice character lengths must be positive")

        candidate_log_probs = _log_softmax_numpy(candidate_scores)
        teacher_log_probs = _log_softmax_numpy(teacher_scores)
        teacher_probs = np.exp(teacher_log_probs)
        kl = float(np.sum(teacher_probs * (teacher_log_probs - candidate_log_probs)))
        # Roundoff can produce a tiny negative value although KL is non-negative.
        if kl < 0.0 and kl > -1.0e-12:
            kl = 0.0
        raw_ce = _choice_cross_entropy(candidate_scores, question.gold_index)
        normalized_scores = candidate_scores / character_lengths
        task_values = by_task.setdefault(
            question.task_name,
            {"raw_ce": [], "kl": [], "acc": [], "acc_norm": []},
        )
        task_values["raw_ce"].append(raw_ce)
        task_values["kl"].append(kl)
        task_values["acc"].append(float(int(np.argmax(candidate_scores)) == question.gold_index))
        task_values["acc_norm"].append(
            float(int(np.argmax(normalized_scores)) == question.gold_index)
        )

    task_metrics: dict[str, dict[str, float | int]] = {}
    task_objectives: list[float] = []
    for task_name, values in by_task.items():
        raw_ce = float(np.mean(values["raw_ce"]))
        kl = float(np.mean(values["kl"]))
        objective = cross_entropy_weight * raw_ce + kl_weight * kl
        task_objectives.append(objective)
        task_metrics[task_name] = {
            "samples": len(values["raw_ce"]),
            "raw_choice_cross_entropy": raw_ce,
            "teacher_candidate_kl": kl,
            "objective": objective,
            "acc": float(np.mean(values["acc"])),
            "acc_norm": float(np.mean(values["acc_norm"])),
        }

    return {
        "schema_version": "task-choice-proxy-distillation-v1",
        "fitness": float(np.mean(task_objectives)),
        "fitness_name": "task_balanced_weighted_choice_ce_teacher_kl",
        "lower_is_better": True,
        "weights": {
            "cross_entropy": cross_entropy_weight,
            "teacher_candidate_kl": kl_weight,
        },
        "num_tasks": len(task_metrics),
        "num_questions": len(questions),
        "num_choice_requests": len(requests),
        "task_order": list(task_metrics),
        "task_objectives": task_objectives,
        "tasks": task_metrics,
    }

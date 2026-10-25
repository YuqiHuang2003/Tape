from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from dsai6707.pruning_multistart import distillation_metrics as extended_distillation_metrics
from dsai6707.task_choice_proxy import (
    ChoiceQuestion,
    ChoiceRequest,
    build_choice_questions,
    canonical_sha256,
    choice_proxy_distillation_metrics,
    choice_proxy_hashes,
    choice_proxy_metrics,
    encode_causal_pair,
    flatten_choice_requests,
    ordered_document_content_hash,
    pack_choice_requests,
    restore_choice_log_likelihoods,
    select_gold_stratified_documents,
)


class _CharacterTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(character) for character in text]


class _FakeTask:
    @staticmethod
    def doc_to_choice(document):
        return document["choices"]

    @staticmethod
    def doc_to_target(document):
        return document["gold"]

    @staticmethod
    def fewshot_context(document, *, num_fewshot):
        assert num_fewshot == 0
        return document["prompt"]

    @staticmethod
    def construct_requests(
        document,
        context,
        *,
        metadata,
        apply_chat_template,
        chat_template,
    ):
        assert metadata[1] == document["source_index"]
        assert apply_chat_template is False
        assert chat_template is None
        return [
            SimpleNamespace(idx=index, args=(context, f" {choice}"))
            for index, choice in enumerate(document["choices"])
        ]


def test_character_normalized_distillation_matches_acc_norm_not_raw_choice():
    question = _question("arc_easy", 0, 1, ("a", "long-answer"))
    candidate = np.array([-1.0, -2.0])
    teacher = np.array([-2.0, -1.0])
    raw = choice_proxy_distillation_metrics(
        (question,), candidate, teacher, cross_entropy_weight=1, kl_weight=0
    )
    norm = extended_distillation_metrics(
        (question,),
        candidate,
        teacher,
        cross_entropy_weight=1,
        kl_weight=0,
        choice_score_normalization="characters",
        temperature=0.25,
    )
    row = norm["tasks"]["arc_easy"]
    assert row["acc"] == 0
    assert row["acc_norm"] == 1
    expected = np.logaddexp.reduce(candidate / [1, 11] / 0.25) - candidate[1] / 11 / 0.25
    assert row["objective_choice_cross_entropy"] == pytest.approx(expected)
    assert norm["fitness"] < raw["fitness"]
    identical = extended_distillation_metrics(
        (question,), candidate, candidate, choice_score_normalization="characters", temperature=0.25
    )
    assert identical["tasks"]["arc_easy"]["teacher_candidate_kl"] == pytest.approx(0)


@pytest.mark.parametrize("temperature", [0, -1, float("inf"), float("nan")])
def test_distillation_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        extended_distillation_metrics(
            (_question("arc_easy", 0, 0, ("a", "b")),), [-1, -2], [-1, -2], temperature=temperature
        )


def _request(
    choice_index,
    choice_text,
    context_ids,
    continuation_ids,
    *,
    context="context",
):
    return ChoiceRequest(
        choice_index=choice_index,
        choice_text=choice_text,
        choice_character_length=len(choice_text),
        context=context,
        continuation=f" {choice_text}",
        context_ids=tuple(context_ids),
        continuation_ids=tuple(continuation_ids),
    )


def _question(task_name, document_index, gold, choices):
    requests = tuple(
        _request(index, choice, [100 + document_index], [index + 1])
        for index, choice in enumerate(choices)
    )
    return ChoiceQuestion(
        task_name=task_name,
        split="train",
        document_index=document_index,
        document_sha256=f"{document_index:064x}",
        gold_index=gold,
        context=f"question-{document_index}",
        choices=tuple(choices),
        requests=requests,
    )


def test_encode_causal_pair_matches_hflm_boundary_and_empty_context_rules():
    class BoundaryTokenizer:
        table = {
            "A": [1],
            "A B": [1, 7],
            "B": [8],
        }

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return self.table[text]

    pair = encode_causal_pair(BoundaryTokenizer(), "A ", "B")
    assert pair.context == "A "
    assert pair.continuation == "B"
    assert pair.context_ids == (1,)
    # Token 7 comes from tokenizing the complete shifted string "A B"; it is
    # intentionally not the standalone continuation token 8.
    assert pair.continuation_ids == (7,)

    empty = encode_causal_pair(BoundaryTokenizer(), "", "B", prefix_token_id=99)
    assert empty.context_ids == (99,)
    assert empty.continuation_ids == (8,)
    with pytest.raises(ValueError, match="prefix_token_id"):
        encode_causal_pair(BoundaryTokenizer(), "", "B")


def test_gold_stratified_selection_is_balanced_deterministic_and_disjoint():
    task = _FakeTask()
    documents = []
    for gold in range(4):
        for item in range(6):
            index = len(documents)
            documents.append(
                {
                    "source_index": index,
                    "prompt": f"q-{gold}-{item}",
                    "choices": ["a", "b", "c", "d"],
                    "gold": gold,
                }
            )
    # This row is part of the dataset content hash but is ineligible because it
    # does not have the frozen four-choice format.
    documents.append(
        {
            "source_index": len(documents),
            "prompt": "three choices",
            "choices": ["a", "b", "c"],
            "gold": 0,
        }
    )

    first = select_gold_stratified_documents(
        task,
        documents,
        task_name="fixture",
        seed=6707,
        search_per_gold=2,
        holdout_per_gold=2,
    )
    repeated = select_gold_stratified_documents(
        task,
        documents,
        task_name="fixture",
        seed=6707,
        search_per_gold=2,
        holdout_per_gold=2,
    )

    assert first == repeated
    assert len(first.search_indices) == 8
    assert len(first.holdout_indices) == 8
    assert set(first.search_indices).isdisjoint(first.holdout_indices)
    assert sorted(first.search_gold_indices) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert sorted(first.holdout_gold_indices) == [0, 0, 1, 1, 2, 2, 3, 3]
    assert first.dataset_rows == 25
    assert first.eligible_rows == 24
    assert first.dataset_content_sha256 == ordered_document_content_hash(documents)
    assert len(first.selection_sha256) == 64
    assert canonical_sha256(first.to_dict()) == canonical_sha256(repeated.to_dict())


def test_gold_stratified_selection_rejects_insufficient_label_rows():
    documents = [
        {
            "source_index": gold,
            "prompt": str(gold),
            "choices": ["a", "b", "c", "d"],
            "gold": gold,
        }
        for gold in range(4)
    ]
    with pytest.raises(ValueError, match="eligible rows"):
        select_gold_stratified_documents(
            _FakeTask(),
            documents,
            task_name="fixture",
            seed=1,
            search_per_gold=1,
            holdout_per_gold=1,
        )


def test_build_questions_uses_native_requests_and_hashes_tokenized_evidence():
    documents = [
        {
            "source_index": 0,
            "prompt": "Question: sky?\nAnswer:",
            "choices": ["blue", "green", "red", "black"],
            "gold": 0,
        },
        {
            "source_index": 1,
            "prompt": "Question: grass?\nAnswer:",
            "choices": ["blue", "green", "red", "black"],
            "gold": 1,
        },
    ]
    questions = build_choice_questions(
        _FakeTask(),
        documents,
        [1, 0],
        _CharacterTokenizer(),
        task_name="fixture",
    )

    assert [question.document_index for question in questions] == [1, 0]
    assert questions[0].gold_index == 1
    assert questions[0].requests[1].choice_character_length == len("green")
    assert questions[0].requests[1].continuation == " green"
    assert questions[0].requests[1].continuation_ids == tuple(map(ord, " green"))
    assert len(flatten_choice_requests(questions)) == 8

    hashes = choice_proxy_hashes(questions)
    assert hashes["num_questions"] == 2
    assert hashes["num_requests"] == 8
    assert len(hashes["data_sha256"]) == 64
    assert len(hashes["requests_sha256"]) == 64
    assert hashes == choice_proxy_hashes(questions)

    changed = list(questions)
    request = changed[0].requests[0]
    changed_request = ChoiceRequest(
        **{
            **request.__dict__,
            "continuation_ids": request.continuation_ids + (999,),
        }
    )
    changed[0] = ChoiceQuestion(
        **{
            **changed[0].__dict__,
            "requests": (changed_request,) + changed[0].requests[1:],
        }
    )
    changed_hashes = choice_proxy_hashes(changed)
    assert changed_hashes["data_sha256"] == hashes["data_sha256"]
    assert changed_hashes["requests_sha256"] != hashes["requests_sha256"]


def test_pack_choice_requests_matches_causal_shift_truncation_and_restores_order():
    requests = (
        _request(0, "aa", [1, 2, 3], [4, 5]),
        _request(1, "b", [9], [8]),
        _request(2, "c", [6, 7, 8], [9]),
    )
    batches = pack_choice_requests(
        requests,
        batch_size=2,
        max_length=3,
        pad_token_id=0,
    )

    assert len(batches) == 2
    first = batches[0]
    # Requests are sorted by decreasing full token length. Request zero is
    # left-truncated exactly as (ctx + cont)[-(max_length + 1):][:-1].
    np.testing.assert_array_equal(first.request_indices, [0, 2])
    np.testing.assert_array_equal(first.input_ids, [[2, 3, 4], [6, 7, 8]])
    np.testing.assert_array_equal(first.input_lengths, [3, 3])
    np.testing.assert_array_equal(first.row_indices, [0, 0, 1])
    np.testing.assert_array_equal(first.time_indices, [1, 2, 2])
    np.testing.assert_array_equal(first.target_ids, [4, 5, 9])
    np.testing.assert_array_equal(first.segment_indices, [0, 0, 1])
    assert first.model_input_positions == 6
    assert first.padded_input_positions == 6
    assert first.scored_tokens == 3
    assert not first.input_ids.flags.writeable

    np.testing.assert_array_equal(batches[1].request_indices, [1])
    restored = restore_choice_log_likelihoods(
        batches,
        [[-10.0, -30.0], [-20.0]],
        num_requests=3,
    )
    np.testing.assert_array_equal(restored, [-10.0, -20.0, -30.0])
    assert not restored.flags.writeable


def test_pack_choice_requests_reuses_hflm_context_cache_rows():
    # All three choices have the same context and a one-token continuation.
    # HFLM runs one model row, then gathers three different last-token targets.
    requests = (
        _request(0, "a", [10, 11], [21]),
        _request(1, "b", [10, 11], [22]),
        _request(2, "c", [10, 11], [23]),
    )
    (batch,) = pack_choice_requests(
        requests,
        batch_size=8,
        max_length=16,
        pad_token_id=0,
    )

    np.testing.assert_array_equal(batch.input_ids, [[10, 11]])
    np.testing.assert_array_equal(batch.request_indices, [0, 1, 2])
    np.testing.assert_array_equal(batch.row_indices, [0, 0, 0])
    np.testing.assert_array_equal(batch.time_indices, [1, 1, 1])
    np.testing.assert_array_equal(batch.target_ids, [21, 22, 23])
    np.testing.assert_array_equal(batch.segment_indices, [0, 1, 2])
    assert batch.model_input_positions == 2
    assert batch.padded_input_positions == 2


def test_choice_proxy_metrics_are_task_balanced_and_use_character_normalization():
    questions = (
        _question("task_a", 0, 0, ["long", "x"]),
        _question("task_b", 1, 1, ["a", "b"]),
        _question("task_b", 2, 0, ["a", "b"]),
    )
    # task_a raw prediction is wrong (-1 > -2), while character normalization
    # makes choice zero win (-2/4 > -1/1).
    values = np.asarray([-2.0, -1.0, -3.0, -1.0, -1.0, -2.0])
    result = choice_proxy_metrics(questions, values)

    ce_a = math.log(math.exp(-2.0) + math.exp(-1.0)) + 2.0
    ce_b_1 = math.log(math.exp(-3.0) + math.exp(-1.0)) + 1.0
    ce_b_2 = math.log(math.exp(-1.0) + math.exp(-2.0)) + 1.0
    expected_b = (ce_b_1 + ce_b_2) / 2.0
    assert result["task_order"] == ["task_a", "task_b"]
    assert result["tasks"]["task_a"]["raw_choice_cross_entropy"] == pytest.approx(ce_a)
    assert result["tasks"]["task_b"]["raw_choice_cross_entropy"] == pytest.approx(expected_b)
    assert result["fitness"] == pytest.approx((ce_a + expected_b) / 2.0)
    assert result["tasks"]["task_a"]["acc"] == 0.0
    assert result["tasks"]["task_a"]["acc_norm"] == 1.0
    assert result["task_balanced_acc"] == pytest.approx(0.5)
    assert result["task_balanced_acc_norm"] == 1.0


def test_distillation_metrics_are_stable_ordered_and_weighted():
    questions = (
        _question("arc_easy", 0, 0, ["a", "b"]),
        _question("openbookqa", 1, 1, ["a", "b"]),
    )
    # Large common offsets exercise stable softmax/log-softmax evaluation.
    candidate = np.asarray([1000.0, 999.0, -1001.0, -999.0])
    teacher = np.asarray([999.0, 1000.0, -1000.0, -1000.0])
    result = choice_proxy_distillation_metrics(
        questions,
        candidate,
        teacher,
        cross_entropy_weight=0.75,
        kl_weight=0.25,
    )

    assert result["task_order"] == ["arc_easy", "openbookqa"]
    assert len(result["task_objectives"]) == 2
    for task_index, task_name in enumerate(result["task_order"]):
        metrics = result["tasks"][task_name]
        expected = (
            0.75 * metrics["raw_choice_cross_entropy"] + 0.25 * metrics["teacher_candidate_kl"]
        )
        assert metrics["teacher_candidate_kl"] >= 0.0
        assert metrics["objective"] == pytest.approx(expected)
        assert result["task_objectives"][task_index] == pytest.approx(expected)
    assert result["fitness"] == pytest.approx(np.mean(result["task_objectives"]))

    identical = choice_proxy_distillation_metrics(questions, teacher, teacher)
    assert identical["tasks"]["arc_easy"]["teacher_candidate_kl"] == pytest.approx(0.0, abs=1.0e-15)
    assert identical["tasks"]["openbookqa"]["teacher_candidate_kl"] == pytest.approx(
        0.0, abs=1.0e-15
    )


def test_distillation_metrics_validate_shapes_values_and_weights():
    questions = (_question("task", 0, 0, ["a", "b"]),)
    with pytest.raises(ValueError, match="wrong shape"):
        choice_proxy_distillation_metrics(questions, [0.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="must be finite"):
        choice_proxy_distillation_metrics(questions, [0.0, np.inf], [0.0, 1.0])
    with pytest.raises(ValueError, match="weights"):
        choice_proxy_distillation_metrics(
            questions,
            [0.0, 1.0],
            [0.0, 1.0],
            cross_entropy_weight=0.0,
            kl_weight=0.0,
        )

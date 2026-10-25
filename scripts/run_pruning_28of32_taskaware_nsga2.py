"""Run task-aware fixed-cardinality NSGA-II expert selection on one GPU.

The search is deliberately separate from the earlier WikiText-NLL EA.  It
uses frozen ARC-Easy/OpenBookQA train subsets for two task-specific objectives,
keeps validation sealed until a single mask has been frozen, and never updates
model weights.  Two model replicas may overlap inference on two CUDA streams,
but exactly one physical GPU must be visible to the process.
"""

from __future__ import annotations

import argparse
import copy
import math
import shutil
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import yaml

from dsai6707.artifacts import ArtifactRef, verify_stage_manifest, write_stage_manifest
from dsai6707.config import LOCKED_TASK_DATASETS
from dsai6707.data import load_token_bundle
from dsai6707.evaluation import evaluate_zero_shot
from dsai6707.granite import inspect_granite_topology, load_granite_model
from dsai6707.io import environment_snapshot, read_json, sha256_file, write_json, write_jsonl
from dsai6707.pruning import ExpertSelectionPlan as RoutingPlan
from dsai6707.pruning import GranitePruningController
from dsai6707.pruning_metrics import token_weighted_full_model_nll
from dsai6707.pruning_nsga2 import (
    balanced_set_crossover,
    crowded_tournament,
    crowding_distances,
    mask_sha256,
    mutate_distinct_layer_swaps,
    nondominated_sort,
    rank_and_crowding,
    select_survivors,
    validate_fixed_cardinality_mask,
)
from dsai6707.pruning_pilot import PruningPilotError, _development_config
from dsai6707.task_choice_proxy import (
    ChoiceQuestion,
    PackedChoiceBatch,
    build_choice_questions,
    choice_proxy_distillation_metrics,
    choice_proxy_hashes,
    flatten_choice_requests,
    pack_choice_requests,
    restore_choice_log_likelihoods,
    select_gold_stratified_documents,
)

TASK_ORDER = ("arc_easy", "openbookqa")
MASK_NAMESPACE_SEED = 0x4E534741  # ASCII-ish "NSGA"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise PruningPilotError(f"configuration is not a mapping: {path}")
    return value


def _workspace_path(workspace: Path, value: str, *, kind: str) -> Path:
    path = (workspace / value).resolve()
    try:
        path.relative_to(workspace)
    except ValueError as exc:
        raise PruningPilotError(f"{kind} must stay inside the workspace") from exc
    return path


def _ref(path: Path, run_dir: Path) -> ArtifactRef:
    try:
        return ArtifactRef.capture(path, relative_to=run_dir)
    except ValueError:
        return ArtifactRef.capture(path)


def _require_one_visible_gpu() -> None:
    import os

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip() or "," in visible:
        raise PruningPilotError(
            "set CUDA_VISIBLE_DEVICES to exactly one physical GPU before this experiment"
        )
    workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace_config not in {":4096:8", ":16:8"}:
        raise PruningPilotError(
            "CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 before CUDA initialization"
        )
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PruningPilotError("the experiment requires exactly one visible CUDA device")


def _validate_config(config: Mapping[str, Any]) -> None:
    pruning = config["pruning"]
    search = config["search"]
    data = config["data"]
    parallel = config["parallel"]
    fitness = config["fitness"]
    if int(pruning["original_experts_per_layer"]) != 32:
        raise PruningPilotError("this experiment requires the 32-expert Granite checkpoint")
    if int(pruning["retained_experts_per_layer"]) != 28:
        raise PruningPilotError("this experiment is locked to 28/32 experts")
    if int(pruning["top_k"]) != 8:
        raise PruningPilotError("the Granite router top-k must remain eight")
    if str(search["algorithm"]) != "task_aware_fixed_cardinality_nsga2":
        raise PruningPilotError("unexpected search algorithm")
    population = int(search["population_size"])
    offspring = int(search["offspring_size"])
    generations = int(search["generations"])
    budget = int(search["evaluation_budget"])
    if (population, offspring, generations, budget) != (16, 16, 7, 128):
        raise PruningPilotError("NSGA-II budget must be mu=16, lambda=16, seven generations")
    if population + generations * offspring != budget:
        raise PruningPilotError("evaluation budget is inconsistent with generations")
    if float(search["crossover_rate"]) != 1.0:
        raise PruningPilotError("the locked NSGA-II always applies balanced crossover")
    counts = tuple(int(value) for value in search["mutation_swap_counts"])
    probabilities = tuple(float(value) for value in search["mutation_swap_probabilities"])
    if counts != (1, 2, 3) or len(probabilities) != 3:
        raise PruningPilotError("mutation must use one, two, or three distinct layers")
    if any(value < 0.0 for value in probabilities) or not math.isclose(
        sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise PruningPilotError("mutation probabilities must be non-negative and sum to one")
    if int(parallel["replicas"]) != 2 or int(parallel["max_inflight"]) != 2:
        raise PruningPilotError("this run is locked to two replicas on one visible GPU")
    if int(parallel["proxy_batch_size"]) < 1:
        raise PruningPilotError("proxy batch size must be positive")
    if list(fitness["objectives"]) != list(TASK_ORDER):
        raise PruningPilotError("objective order must be ARC-Easy then OpenBookQA")
    ce_weight = float(fitness["gold_ce_weight"])
    kl_weight = float(fitness["original_kl_weight"])
    if (
        ce_weight < 0.0
        or kl_weight < 0.0
        or not math.isclose(ce_weight + kl_weight, 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise PruningPilotError("fitness weights must be non-negative and sum to one")
    for field in ("search_samples_per_task", "holdout_samples_per_task"):
        value = int(data[field])
        if value <= 0 or value % 4:
            raise PruningPilotError(f"data.{field} must be a positive multiple of four")
    if list(config["development_evaluation"]["tasks"]) != list(TASK_ORDER):
        raise PruningPilotError("development task order differs from the locked protocol")
    if config["development_evaluation"].get("split") != "validation":
        raise PruningPilotError("only task validation may be used after winner freezing")


def _routing_plan(mask: np.ndarray) -> RoutingPlan:
    values = np.asarray(mask, dtype=np.bool_)
    if values.ndim != 2:
        raise ValueError("mask must have shape [layers, experts]")
    return RoutingPlan.from_retained(
        [np.flatnonzero(row).astype(int).tolist() for row in values],
        num_experts=int(values.shape[1]),
    )


def _scored_selection_mask(path: Path) -> np.ndarray:
    payload = read_json(path)
    layers = payload.get("layers")
    if not isinstance(layers, list):
        raise PruningPilotError(f"invalid scored selection: {path}")
    mask = np.zeros((len(layers), int(payload["num_experts"])), dtype=np.bool_)
    for expected, layer in enumerate(layers):
        if int(layer["layer"]) != expected:
            raise PruningPilotError(f"unordered layers in {path}")
        mask[expected, [int(value) for value in layer["retained"]]] = True
    return mask


def _routing_selection_mask(path: Path) -> np.ndarray:
    plan = RoutingPlan.from_dict(read_json(path))
    mask = np.zeros((plan.num_layers, plan.num_experts), dtype=np.bool_)
    for layer, retained in enumerate(plan.layers):
        mask[layer, list(retained)] = True
    return mask


def _selection_payload(mask: np.ndarray, *, method: str) -> dict[str, Any]:
    plan = _routing_plan(mask)
    return {
        "schema_version": "expert-selection-v1",
        "method": method,
        "mask_sha256": mask_sha256(mask, retained_count=28),
        **plan.to_dict(),
    }


def _parameter_identity(model: Any) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (name, int(parameter.data_ptr()), int(parameter._version))
        for name, parameter in model.named_parameters()
    )


def _theoretical_parameter_counts(model: Any, *, retained: int) -> dict[str, Any]:
    total = sum(int(parameter.numel()) for parameter in model.parameters())
    original = int(model.config.num_local_experts)
    removed_per_layer = original - retained
    removed = 0
    for layer in model.model.layers:
        moe = layer.block_sparse_moe
        for parameter in (moe.input_linear.weight, moe.output_linear.weight):
            if int(parameter.shape[0]) != original:
                raise RuntimeError("unexpected Granite expert weight layout")
            removed += int(parameter[0].numel()) * removed_per_layer
    return {
        "original_total_parameters": total,
        "theoretical_structurally_retained_parameters": total - removed,
        "theoretical_removed_parameters": removed,
        "theoretical_total_parameter_reduction_fraction": removed / total,
        "expert_capacity_reduction_fraction": removed_per_layer / original,
        "search_implementation": "logical_router_mask_only",
        "search_resident_parameters_per_replica": total,
        "replicas_loaded": 2,
        "physical_checkpoint_exported": False,
    }


@dataclass(frozen=True)
class ProxyData:
    questions: dict[str, tuple[ChoiceQuestion, ...]]
    requests: dict[str, tuple[Any, ...]]
    batches: dict[str, tuple[PackedChoiceBatch, ...]]
    evidence: dict[str, Any]


def _load_proxy_data(config: Mapping[str, Any], tokenizer: Any) -> ProxyData:
    from lm_eval.tasks import TaskManager, get_task_dict

    revisions = config["data"]["task_revisions"]
    specs = [
        {
            "task": task_name,
            "dataset_kwargs": {"revision": str(revisions[task_name])},
        }
        for task_name in TASK_ORDER
    ]
    resolved = get_task_dict(specs, TaskManager())
    if set(resolved) != set(TASK_ORDER):
        raise RuntimeError("lm-eval resolved unexpected proxy tasks")

    search_per_task = int(config["data"]["search_samples_per_task"])
    holdout_per_task = int(config["data"]["holdout_samples_per_task"])
    seed = int(config["search"]["seed"])
    prefix_token_id = (
        tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    )
    all_questions: dict[str, list[ChoiceQuestion]] = {"search": [], "holdout": []}
    task_evidence: dict[str, Any] = {}
    for task_name in TASK_ORDER:
        task = resolved[task_name]
        locked = LOCKED_TASK_DATASETS[task_name]
        expected = {
            "dataset_path": locked["path"],
            "dataset_name": locked["name"],
            "training_split": "train",
            "output_type": "multiple_choice",
        }
        for key, value in expected.items():
            if task.get_config(key) != value:
                raise RuntimeError(
                    f"lm-eval {task_name} resolved {key}={task.get_config(key)!r}; "
                    f"expected {value!r}"
                )
        kwargs = task.get_config("dataset_kwargs")
        if not isinstance(kwargs, Mapping) or kwargs.get("revision") != str(revisions[task_name]):
            raise RuntimeError(f"lm-eval {task_name} did not use the pinned revision")
        documents = task.training_docs()
        fingerprint = getattr(documents, "_fingerprint", None)
        if not isinstance(fingerprint, str) or not fingerprint:
            raise RuntimeError(f"lm-eval {task_name} train split has no fingerprint")
        split = select_gold_stratified_documents(
            task,
            documents,
            task_name=task_name,
            seed=seed,
            search_per_gold=search_per_task // 4,
            holdout_per_gold=holdout_per_task // 4,
            required_choice_count=4,
        )
        search_questions = build_choice_questions(
            task,
            documents,
            split.search_indices,
            tokenizer,
            task_name=task_name,
            split="train-search",
            prefix_token_id=prefix_token_id,
        )
        holdout_questions = build_choice_questions(
            task,
            documents,
            split.holdout_indices,
            tokenizer,
            task_name=task_name,
            split="train-holdout",
            prefix_token_id=prefix_token_id,
        )
        if len(search_questions) != search_per_task or len(holdout_questions) != holdout_per_task:
            raise RuntimeError("proxy selection returned an unexpected number of questions")
        all_questions["search"].extend(search_questions)
        all_questions["holdout"].extend(holdout_questions)
        task_evidence[task_name] = {
            "hub_repository": locked["hub_repository"],
            "dataset_path": locked["path"],
            "dataset_name": locked["name"],
            "revision": str(revisions[task_name]),
            "training_split": "train",
            "huggingface_fingerprint": fingerprint,
            "selection": split.to_dict(),
            "search_proxy": choice_proxy_hashes(search_questions),
            "holdout_proxy": choice_proxy_hashes(holdout_questions),
        }

    questions = {name: tuple(values) for name, values in all_questions.items()}
    requests = {name: flatten_choice_requests(values) for name, values in questions.items()}
    max_length = int(getattr(tokenizer, "model_max_length", 0))
    if max_length <= 0 or max_length > 1_000_000:
        max_length = 4096
    batch_size = int(config["parallel"]["proxy_batch_size"])
    batches = {
        name: pack_choice_requests(
            values,
            batch_size=batch_size,
            max_length=max_length,
            # lm-eval 0.4.8 pad_and_concat right-pads causal requests with zero.
            pad_token_id=0,
        )
        for name, values in requests.items()
    }
    overlap_proof = {
        task_name: not bool(
            set(task_evidence[task_name]["selection"]["search_indices"])
            & set(task_evidence[task_name]["selection"]["holdout_indices"])
        )
        for task_name in TASK_ORDER
    }
    if not all(overlap_proof.values()):
        raise RuntimeError("proxy search and holdout selections overlap")
    evidence = {
        "schema_version": "task-proxy-data-v1",
        "source_split": "train",
        "validation_used": False,
        "test_used": False,
        "seed": seed,
        "task_order": list(TASK_ORDER),
        "search_samples_per_task": search_per_task,
        "holdout_samples_per_task": holdout_per_task,
        "search_holdout_disjoint": overlap_proof,
        "tasks": task_evidence,
        "packing": {
            name: {
                "batch_size": batch_size,
                "num_batches": len(values),
                "num_requests": len(requests[name]),
                "scored_tokens": sum(batch.scored_tokens for batch in values),
                "model_input_positions": sum(batch.model_input_positions for batch in values),
                "padded_input_positions": sum(batch.padded_input_positions for batch in values),
                "right_padding_token_id": 0,
                "max_length": max_length,
            }
            for name, values in batches.items()
        },
    }
    return ProxyData(questions=questions, requests=requests, batches=batches, evidence=evidence)


@dataclass(frozen=True)
class RawProxyScore:
    choice_log_likelihoods: np.ndarray
    lane: int
    wall_seconds: float
    gpu_seconds: float


@dataclass(frozen=True)
class TaskFitness:
    objectives: tuple[float, float]
    metrics: dict[str, Any]
    lane: int
    cost: dict[str, Any]


@dataclass(frozen=True)
class _TorchPackedBatch:
    input_ids: Any
    row_indices: Any
    time_indices: Any
    target_ids: Any
    segment_indices: Any
    num_requests: int


@dataclass(frozen=True)
class _ProxyLane:
    model: Any
    controller: GranitePruningController
    stream: Any
    batches: dict[str, tuple[_TorchPackedBatch, ...]]


class ParallelTaskProxyEvaluator:
    """Two independent model/controller/stream lanes on one CUDA device."""

    def __init__(self, models: Sequence[Any], proxy_data: ProxyData) -> None:
        import torch

        if len(models) != 2:
            raise ValueError("exactly two model replicas are required")
        self.torch = torch
        self.proxy_data = proxy_data
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="task-proxy-lane")
        self._closed = False
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0
        self.parallel_calls = 0
        self.sum_lane_wall_seconds = 0.0
        self.lanes: list[_ProxyLane] = []
        for model in models:
            topology = inspect_granite_topology(model)
            identity = RoutingPlan.all_experts(
                num_layers=int(topology["num_layers"]),
                num_experts=int(topology["num_experts"]),
            )
            device = next(model.parameters()).device
            stream = torch.cuda.Stream(device=device)
            gpu_batches: dict[str, tuple[_TorchPackedBatch, ...]] = {}
            with torch.cuda.stream(stream):
                for split_name, cpu_batches in proxy_data.batches.items():
                    converted = []
                    for batch in cpu_batches:
                        converted.append(
                            _TorchPackedBatch(
                                input_ids=torch.as_tensor(
                                    np.array(batch.input_ids, copy=True),
                                    dtype=torch.long,
                                    device=device,
                                ),
                                row_indices=torch.as_tensor(
                                    np.array(batch.row_indices, copy=True),
                                    dtype=torch.long,
                                    device=device,
                                ),
                                time_indices=torch.as_tensor(
                                    np.array(batch.time_indices, copy=True),
                                    dtype=torch.long,
                                    device=device,
                                ),
                                target_ids=torch.as_tensor(
                                    np.array(batch.target_ids, copy=True),
                                    dtype=torch.long,
                                    device=device,
                                ),
                                segment_indices=torch.as_tensor(
                                    np.array(batch.segment_indices, copy=True),
                                    dtype=torch.long,
                                    device=device,
                                ),
                                num_requests=batch.num_requests,
                            )
                        )
                    gpu_batches[split_name] = tuple(converted)
            self.lanes.append(
                _ProxyLane(
                    model=model,
                    controller=GranitePruningController(model, identity),
                    stream=stream,
                    batches=gpu_batches,
                )
            )
        torch.cuda.synchronize()

    def reset_runtime_counters(self) -> None:
        with self._lock:
            self._active = 0
            self.max_active = 0
            self.parallel_calls = 0
            self.sum_lane_wall_seconds = 0.0

    def _score_lane(self, lane_index: int, mask: np.ndarray, split: str) -> RawProxyScore:
        if split not in self.proxy_data.batches:
            raise ValueError(f"unknown proxy split: {split}")
        torch = self.torch
        lane = self.lanes[lane_index]
        wall_started = perf_counter()
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            gpu_values: list[Any] = []
            with torch.cuda.stream(lane.stream), torch.inference_mode():
                lane.controller.apply(_routing_plan(mask))
                start_event.record(lane.stream)
                for batch in lane.batches[split]:
                    # No attention mask: this reproduces lm-eval 0.4.8's causal
                    # right-padding path. Padding lies strictly after every scored logit.
                    logits = lane.model(batch.input_ids).logits
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    selected = log_probs[
                        batch.row_indices,
                        batch.time_indices,
                        :,
                    ]
                    token_log_likelihood = selected.gather(1, batch.target_ids[:, None]).squeeze(1)
                    # Match lm-eval's per-request reduction in model-logit dtype.
                    request_values = torch.stack(
                        [
                            token_log_likelihood[batch.segment_indices == segment].sum()
                            for segment in range(batch.num_requests)
                        ]
                    )
                    gpu_values.append(request_values)
                concatenated = torch.cat(gpu_values).float()
                end_event.record(lane.stream)
                cpu_values = concatenated.cpu().numpy()
            end_event.synchronize()
            batch_values: list[np.ndarray] = []
            cursor = 0
            for batch in self.proxy_data.batches[split]:
                batch_values.append(cpu_values[cursor : cursor + batch.num_requests])
                cursor += batch.num_requests
            restored = restore_choice_log_likelihoods(
                self.proxy_data.batches[split],
                batch_values,
                num_requests=len(self.proxy_data.requests[split]),
            )
            return RawProxyScore(
                choice_log_likelihoods=restored,
                lane=lane_index,
                wall_seconds=perf_counter() - wall_started,
                gpu_seconds=float(start_event.elapsed_time(end_event) / 1000.0),
            )
        finally:
            with self._lock:
                self._active -= 1

    def score_on_lane(self, lane_index: int, mask: np.ndarray, *, split: str) -> RawProxyScore:
        if self._closed:
            raise RuntimeError("proxy evaluator is closed")
        return self._score_lane(lane_index, np.asarray(mask, dtype=np.bool_).copy(), split)

    def score_many(
        self,
        masks: Sequence[np.ndarray],
        *,
        split: str,
        parallel: bool,
    ) -> tuple[RawProxyScore, ...]:
        if self._closed:
            raise RuntimeError("proxy evaluator is closed")
        if not 1 <= len(masks) <= len(self.lanes):
            raise ValueError("candidate batch must fit the available lanes")
        if parallel and len(masks) > 1:
            with self._lock:
                self.parallel_calls += 1
            futures = [
                self.executor.submit(self._score_lane, lane, mask.copy(), split)
                for lane, mask in enumerate(masks)
            ]
            values = tuple(future.result() for future in futures)
        else:
            values = tuple(self._score_lane(0, mask.copy(), split) for mask in masks)
        with self._lock:
            self.sum_lane_wall_seconds += sum(value.wall_seconds for value in values)
        return values

    def close(self) -> None:
        if self._closed:
            return
        self.executor.shutdown(wait=True, cancel_futures=True)
        for lane in self.lanes:
            lane.controller.close()
        # Release the evaluator's model references so deleting the second
        # replica before full validation actually returns its GPU memory.
        self.lanes.clear()
        self._closed = True


def _task_fitness(
    proxy_data: ProxyData,
    split: str,
    score: RawProxyScore,
    teacher_choice_log_likelihoods: np.ndarray,
    config: Mapping[str, Any],
) -> TaskFitness:
    metrics = choice_proxy_distillation_metrics(
        proxy_data.questions[split],
        score.choice_log_likelihoods,
        teacher_choice_log_likelihoods,
        cross_entropy_weight=float(config["fitness"]["gold_ce_weight"]),
        kl_weight=float(config["fitness"]["original_kl_weight"]),
    )
    if tuple(metrics["task_order"]) != TASK_ORDER:
        raise RuntimeError("proxy metric task order changed")
    objectives = tuple(float(value) for value in metrics["task_objectives"])
    if len(objectives) != 2 or not np.isfinite(objectives).all():
        raise RuntimeError("proxy returned invalid task objectives")
    return TaskFitness(
        objectives=(objectives[0], objectives[1]),
        metrics=metrics,
        lane=score.lane,
        cost={
            "wall_seconds": score.wall_seconds,
            "gpu_seconds": score.gpu_seconds,
        },
    )


def _mutation_dict(mutation: Any) -> list[dict[str, int]]:
    return [
        {
            "layer": int(swap.layer),
            "removed_expert": int(swap.removed_expert),
            "added_expert": int(swap.added_expert),
        }
        for swap in mutation.swaps
    ]


def make_initial_population(
    anchors: Sequence[tuple[str, np.ndarray]],
    *,
    population_size: int,
    seed: int,
    mutation_counts: Sequence[int],
    mutation_probabilities: Sequence[float],
) -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    """Create a unique anchor-seeded population without consulting any score."""

    rng = np.random.default_rng(np.random.SeedSequence([seed, MASK_NAMESPACE_SEED, 0]))
    masks: list[np.ndarray] = []
    contexts: list[dict[str, Any]] = []
    known: set[str] = set()
    normalized_anchors: list[tuple[str, np.ndarray]] = []
    for name, raw_mask in anchors:
        mask = np.ascontiguousarray(raw_mask, dtype=np.bool_)
        validate_fixed_cardinality_mask(mask, retained_count=28)
        digest = mask_sha256(mask, retained_count=28)
        normalized_anchors.append((name, mask))
        if digest in known:
            continue
        known.add(digest)
        masks.append(mask.copy())
        contexts.append({"kind": "anchor", "anchor": name, "mutation": []})
    if not normalized_anchors:
        raise ValueError("at least one initial anchor is required")

    attempts = 0
    while len(masks) < population_size:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError("could not construct a unique initial population")
        anchor_name, anchor = normalized_anchors[(attempts - 1) % len(normalized_anchors)]
        count = int(rng.choice(mutation_counts, p=mutation_probabilities))
        mutation = mutate_distinct_layer_swaps(
            anchor,
            rng=rng,
            min_layers=count,
            max_layers=count,
        )
        digest = mask_sha256(mutation.mask, retained_count=28)
        if digest in known:
            continue
        known.add(digest)
        masks.append(mutation.mask)
        contexts.append(
            {
                "kind": "anchor_mutation",
                "anchor": anchor_name,
                "mutation": _mutation_dict(mutation),
            }
        )
    return np.stack(masks), tuple(contexts)


@dataclass(frozen=True)
class NSGA2SearchResult:
    masks: np.ndarray
    objectives: np.ndarray
    trials: tuple[dict[str, Any], ...]
    final_population_ids: tuple[int, ...]
    proposals: int
    duplicate_proposals: int
    initial_population_sha256: str


def _population_sha256(population: np.ndarray) -> str:
    from hashlib import sha256

    digests = [mask_sha256(mask, retained_count=28) for mask in population]
    return sha256("\n".join(digests).encode("ascii")).hexdigest()


def run_task_nsga2(
    objective_many: Callable[[Sequence[np.ndarray]], Sequence[TaskFitness]],
    initial_population: np.ndarray,
    initial_contexts: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    population_size: int,
    offspring_size: int,
    generations: int,
    tournament_size: int,
    mutation_counts: Sequence[int],
    mutation_probabilities: Sequence[float],
    max_duplicate_attempts: int,
) -> NSGA2SearchResult:
    """Run deterministic two-objective NSGA-II with batched model evaluation."""

    population = np.asarray(initial_population, dtype=np.bool_)
    if population.ndim != 3 or population.shape[0] != population_size:
        raise ValueError("initial_population has the wrong shape")
    if len(initial_contexts) != population_size:
        raise ValueError("one initial context is required per mask")
    for mask in population:
        validate_fixed_cardinality_mask(mask, retained_count=28)
    initial_keys = [mask_sha256(mask, retained_count=28) for mask in population]
    if len(set(initial_keys)) != len(initial_keys):
        raise ValueError("initial population masks must be unique")

    rng = np.random.default_rng(np.random.SeedSequence([seed, MASK_NAMESPACE_SEED, 1]))
    all_masks: list[np.ndarray] = []
    all_objectives: list[tuple[float, float]] = []
    trials: list[dict[str, Any]] = []
    known = set(initial_keys)
    proposals = 0
    duplicates = 0

    def evaluate_generation(
        masks: Sequence[np.ndarray],
        contexts: Sequence[Mapping[str, Any]],
        *,
        phase: str,
        generation: int,
    ) -> list[int]:
        ids: list[int] = []
        for start in range(0, len(masks), 2):
            batch_masks = [np.ascontiguousarray(mask.copy()) for mask in masks[start : start + 2]]
            results = tuple(objective_many(batch_masks))
            if len(results) != len(batch_masks):
                raise RuntimeError("objective returned the wrong number of results")
            for mask, context, value in zip(
                batch_masks, contexts[start : start + 2], results, strict=True
            ):
                candidate_id = len(all_masks)
                objectives = tuple(float(item) for item in value.objectives)
                if len(objectives) != 2 or not np.isfinite(objectives).all():
                    raise RuntimeError("objective returned invalid values")
                all_masks.append(mask)
                all_objectives.append((objectives[0], objectives[1]))
                trial = {
                    "candidate_id": candidate_id,
                    "candidate_sha256": mask_sha256(mask, retained_count=28),
                    "phase": phase,
                    "generation": generation,
                    "objectives": {
                        TASK_ORDER[0]: objectives[0],
                        TASK_ORDER[1]: objectives[1],
                    },
                    "metrics": value.metrics,
                    "lane": int(value.lane),
                    "cost": value.cost,
                    **dict(context),
                }
                trials.append(trial)
                ids.append(candidate_id)
        return ids

    population_ids = np.asarray(
        evaluate_generation(
            list(population),
            initial_contexts,
            phase="initialization",
            generation=0,
        ),
        dtype=np.int64,
    )
    population_objectives = np.asarray(
        [all_objectives[int(candidate_id)] for candidate_id in population_ids],
        dtype=np.float64,
    )

    for generation in range(1, generations + 1):
        ranks, distances = rank_and_crowding(population_objectives, candidate_ids=population_ids)
        child_masks: list[np.ndarray] = []
        child_contexts: list[dict[str, Any]] = []
        duplicate_streak = 0
        mating_event = 0
        while len(child_masks) < offspring_size:
            if duplicate_streak >= max_duplicate_attempts:
                raise RuntimeError("NSGA-II exceeded the duplicate-proposal limit")
            first_position = crowded_tournament(
                ranks,
                distances,
                population_ids,
                rng=rng,
                tournament_size=tournament_size,
            )
            pool = [index for index in range(population_size) if index != first_position]
            second_position = crowded_tournament(
                ranks,
                distances,
                population_ids,
                rng=rng,
                tournament_size=tournament_size,
                pool=pool,
            )
            first_parent_id = int(population_ids[first_position])
            second_parent_id = int(population_ids[second_position])
            first_child, second_child = balanced_set_crossover(
                all_masks[first_parent_id],
                all_masks[second_parent_id],
                rng=rng,
            )
            mating_event += 1
            for sibling, crossed in enumerate((first_child, second_child)):
                proposals += 1
                count = int(rng.choice(mutation_counts, p=mutation_probabilities))
                mutation = mutate_distinct_layer_swaps(
                    crossed,
                    rng=rng,
                    min_layers=count,
                    max_layers=count,
                )
                digest = mask_sha256(mutation.mask, retained_count=28)
                if digest in known:
                    duplicates += 1
                    duplicate_streak += 1
                    continue
                duplicate_streak = 0
                known.add(digest)
                child_masks.append(mutation.mask)
                child_contexts.append(
                    {
                        "kind": "offspring",
                        "parent_ids": [first_parent_id, second_parent_id],
                        "mating_event": mating_event,
                        "sibling": sibling,
                        "crossover": "balanced_pruned_set_complementary",
                        "mutation": _mutation_dict(mutation),
                    }
                )
                if len(child_masks) == offspring_size:
                    break

        child_ids = np.asarray(
            evaluate_generation(
                child_masks,
                child_contexts,
                phase="offspring",
                generation=generation,
            ),
            dtype=np.int64,
        )
        child_objectives = np.asarray(
            [all_objectives[int(candidate_id)] for candidate_id in child_ids],
            dtype=np.float64,
        )
        combined_ids = np.concatenate((population_ids, child_ids))
        combined_objectives = np.concatenate((population_objectives, child_objectives), axis=0)
        survivors = select_survivors(
            combined_objectives,
            combined_ids,
            population_size,
        )
        population_ids = combined_ids[survivors]
        population_objectives = combined_objectives[survivors]

    expected = population_size + generations * offspring_size
    if len(all_masks) != expected or len(known) != expected:
        raise RuntimeError("NSGA-II unique-evaluation accounting failed")
    return NSGA2SearchResult(
        masks=np.stack(all_masks),
        objectives=np.asarray(all_objectives, dtype=np.float64),
        trials=tuple(trials),
        final_population_ids=tuple(int(value) for value in population_ids),
        proposals=proposals,
        duplicate_proposals=duplicates,
        initial_population_sha256=_population_sha256(population),
    )


def _shortlist_candidate_ids(
    result: NSGA2SearchResult,
    *,
    reap_candidate_id: int,
    limit: int,
) -> tuple[int, ...]:
    ids = np.arange(result.objectives.shape[0], dtype=np.int64)
    front = nondominated_sort(result.objectives, candidate_ids=ids)[0]
    evolved = [int(index) for index in front if int(index) != reap_candidate_id]
    if not evolved or limit < 1:
        return ()
    reap = result.objectives[reap_candidate_id]
    if np.any(reap <= 0.0):
        raise RuntimeError("REAP proxy objectives must be positive")
    distances = crowding_distances(result.objectives, front, candidate_ids=ids)
    selectors = [
        lambda index: (float(result.objectives[index, 0]), index),
        lambda index: (float(result.objectives[index, 1]), index),
        lambda index: (
            float(np.max(result.objectives[index] / reap)),
            float(np.mean(result.objectives[index] / reap)),
            index,
        ),
        lambda index: (-float(distances[index]), index),
    ]
    selected: list[int] = []
    for selector in selectors:
        remaining = [index for index in evolved if index not in selected]
        if not remaining or len(selected) >= limit:
            break
        selected.append(min(remaining, key=selector))
    if len(selected) < limit:
        remaining = [index for index in evolved if index not in selected]
        remaining.sort(
            key=lambda index: (
                float(np.max(result.objectives[index] / reap)),
                float(np.mean(result.objectives[index] / reap)),
                index,
            )
        )
        selected.extend(remaining[: limit - len(selected)])
    return tuple(selected)


def _lm_eval_reference_scores(
    model: Any,
    tokenizer: Any,
    requests: Sequence[Any],
    *,
    count: int,
    batch_size: int,
) -> np.ndarray:
    from lm_eval.models.huggingface import HFLM

    wrapped = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=batch_size,
        device=str(next(model.parameters()).device),
    )
    triples = [
        (
            (request.context, request.continuation),
            list(request.context_ids),
            list(request.continuation_ids),
        )
        for request in requests[:count]
    ]
    values = wrapped._loglikelihood_tokens(
        triples,
        disable_tqdm=True,
        override_bs=batch_size,
    )
    return np.asarray([float(value[0]) for value in values], dtype=np.float64)


def _proxy_preflight(
    evaluator: ParallelTaskProxyEvaluator,
    proxy_data: ProxyData,
    model: Any,
    tokenizer: Any,
    identity_mask: np.ndarray,
    reap_mask: np.ndarray,
    frequency_mask: np.ndarray,
    teacher_search: RawProxyScore,
    *,
    tolerance: float,
    batch_size: int,
) -> dict[str, Any]:
    # Compare the complete frozen request set.  Comparing only a prefix would
    # change HFLM's length-sorted batch composition and can legitimately alter
    # bfloat16 scores even when the packing algorithm is identical.
    reference_count = len(proxy_data.requests["search"])
    reference = _lm_eval_reference_scores(
        model,
        tokenizer,
        proxy_data.requests["search"],
        count=reference_count,
        batch_size=batch_size,
    )
    packed_delta = float(
        np.max(np.abs(reference - teacher_search.choice_log_likelihoods[:reference_count]))
    )
    reap_lane0 = evaluator.score_on_lane(0, reap_mask, split="search")
    reap_lane1 = evaluator.score_on_lane(1, reap_mask, split="search")
    frequency_lane0 = evaluator.score_on_lane(0, frequency_mask, split="search")
    frequency_parallel, reap_parallel = evaluator.score_many(
        (frequency_mask, reap_mask), split="search", parallel=True
    )
    reference_predictions: list[int] = []
    packed_predictions: list[int] = []
    offset = 0
    for question in proxy_data.questions["search"]:
        stop = offset + len(question.choices)
        reference_predictions.append(int(np.argmax(reference[offset:stop])))
        packed_predictions.append(
            int(np.argmax(teacher_search.choice_log_likelihoods[offset:stop]))
        )
        offset = stop
    prediction_agreement = reference_predictions == packed_predictions
    deltas = {
        "packed_vs_lm_eval_reference": packed_delta,
        "same_reap_mask_across_replicas": float(
            np.max(np.abs(reap_lane0.choice_log_likelihoods - reap_lane1.choice_log_likelihoods))
        ),
        "frequency_serial_vs_parallel": float(
            np.max(
                np.abs(
                    frequency_lane0.choice_log_likelihoods
                    - frequency_parallel.choice_log_likelihoods
                )
            )
        ),
        "reap_serial_vs_parallel": float(
            np.max(np.abs(reap_lane1.choice_log_likelihoods - reap_parallel.choice_log_likelihoods))
        ),
    }
    overlap = evaluator.max_active >= 2
    passed = max(deltas.values()) <= tolerance and overlap and prediction_agreement
    return {
        "passed": passed,
        "tolerance": tolerance,
        "reference_request_count": reference_count,
        "choice_log_likelihood_max_absolute_deltas": deltas,
        "raw_choice_predictions_identical_to_lm_eval": prediction_agreement,
        "raw_choice_prediction_count": len(reference_predictions),
        "parallel_overlap_observed": overlap,
        "identity_mask_retained_counts": np.count_nonzero(identity_mask, axis=1)
        .astype(int)
        .tolist(),
    }


def _candidate_development_summary(
    artifact: Mapping[str, Any],
    original: Mapping[str, Any],
) -> dict[str, Any]:
    task_accuracy = {task: float(artifact["tasks"][task]["acc"]) for task in TASK_ORDER}
    original_accuracy = {task: float(original["tasks"][task]["acc"]) for task in TASK_ORDER}
    mean = float(artifact["mean_primary_acc"])
    original_mean = float(original["mean_primary_acc"])
    return {
        "mean_accuracy": mean,
        "mean_absolute_change_vs_original": mean - original_mean,
        "mean_raw_retention_vs_original": mean / original_mean,
        "task_accuracy": task_accuracy,
        "task_raw_retention_vs_original": {
            task: task_accuracy[task] / original_accuracy[task] for task in TASK_ORDER
        },
    }


def _copy_source_provenance(
    source_run: Path, source_manifest: Path, run_dir: Path
) -> tuple[Path, Path]:
    provenance = run_dir / "provenance"
    provenance.mkdir()
    snapshot = provenance / "source_root_manifest_snapshot.json"
    shutil.copyfile(source_manifest, snapshot)
    verification = provenance / "verification.json"
    write_json(
        verification,
        {
            "schema_version": "cross-run-provenance-verification-v1",
            "source_run": str(source_run),
            "source_manifest_verified_recursively": True,
            "source_manifest": ArtifactRef.capture(source_manifest).to_dict(),
            "source_manifest_snapshot": _ref(snapshot, run_dir).to_dict(),
        },
    )
    return snapshot, verification


def run(config_path: str | Path) -> Path:
    _require_one_visible_gpu()
    config_path = Path(config_path).resolve()
    workspace = config_path.parent.parent
    config = _load_yaml(config_path)
    _validate_config(config)
    protocol_path = _workspace_path(workspace, str(config["project"]["protocol"]), kind="protocol")
    source_run = _workspace_path(workspace, str(config["source"]["run_dir"]), kind="source run")
    source_manifest = source_run / str(config["source"]["stage"])
    verify_stage_manifest(source_manifest, run_dir=source_run)
    source_config = _load_yaml(source_run / "config.resolved.yaml")
    if source_config["model"] != config["model"]:
        raise PruningPilotError("model configuration differs from the verified source run")
    if source_config["development_evaluation"] != config["development_evaluation"]:
        raise PruningPilotError("development evaluation differs from the verified source run")
    token_path = _workspace_path(
        workspace, str(config["data"]["token_artifact"]), kind="token artifact"
    )
    if sha256_file(token_path) != str(config["data"]["token_artifact_sha256"]):
        raise PruningPilotError("token artifact hash differs from the locked config")
    run_dir = _workspace_path(workspace, str(config["project"]["run_dir"]), kind="run directory")
    if run_dir.exists():
        raise PruningPilotError(f"refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    with (run_dir / "config.resolved.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=True)
    write_json(
        run_dir / "config.source.json",
        {"path": str(config_path), "sha256": sha256_file(config_path)},
    )
    source_snapshot, source_verification = _copy_source_provenance(
        source_run, source_manifest, run_dir
    )

    source_paths = {
        name: (source_run / str(config["source"][field])).resolve()
        for name, field in (
            ("frequency", "frequency_selection"),
            ("reap", "reap_selection"),
            ("old_nll_ea", "old_nll_ea_selection"),
        )
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    frequency_mask = _scored_selection_mask(source_paths["frequency"])
    reap_mask = _scored_selection_mask(source_paths["reap"])
    old_nll_mask = _routing_selection_mask(source_paths["old_nll_ea"])
    for mask in (frequency_mask, reap_mask, old_nll_mask):
        validate_fixed_cardinality_mask(mask, retained_count=28)

    print("loading model replica 1/2 on the single visible GPU ...", flush=True)
    model0, tokenizer0 = load_granite_model(config["model"])
    topology = inspect_granite_topology(model0)
    if (
        int(topology["num_layers"]) != 24
        or int(topology["num_experts"]) != 32
        or int(topology["top_k"]) != 8
    ):
        raise PruningPilotError("Granite topology differs from the locked experiment")
    print("loading model replica 2/2 on the same GPU ...", flush=True)
    model1, tokenizer1 = load_granite_model(config["model"])
    if inspect_granite_topology(model1) != topology:
        raise PruningPilotError("model replicas have different topology")
    identity_before = [_parameter_identity(model0), _parameter_identity(model1)]
    parameter_counts = _theoretical_parameter_counts(model0, retained=28)

    print("freezing task train search/holdout subsets and packing requests ...", flush=True)
    proxy_data = _load_proxy_data(config, tokenizer0)
    data_dir = run_dir / "data"
    data_dir.mkdir()
    subset_path = data_dir / "subsets.json"
    search_questions_path = data_dir / "search_questions.json"
    holdout_questions_path = data_dir / "holdout_questions.json"
    write_json(subset_path, proxy_data.evidence)
    write_json(
        search_questions_path,
        [question.to_dict() for question in proxy_data.questions["search"]],
    )
    write_json(
        holdout_questions_path,
        [question.to_dict() for question in proxy_data.questions["holdout"]],
    )
    write_stage_manifest(
        data_dir / "stage_manifest.json",
        stage="task_supervised_proxy_data",
        inputs={
            "config": ArtifactRef.capture(config_path),
            "protocol": ArtifactRef.capture(protocol_path),
            "task_proxy_implementation": ArtifactRef.capture(
                workspace / "src/dsai6707/task_choice_proxy.py"
            ),
            "script": ArtifactRef.capture(Path(__file__).resolve()),
        },
        outputs={
            "subsets": _ref(subset_path, run_dir),
            "search_questions": _ref(search_questions_path, run_dir),
            "holdout_questions": _ref(holdout_questions_path, run_dir),
        },
        metadata={
            "source_split": "train",
            "validation_used": False,
            "test_used": False,
        },
    )

    models = [model0, model1]
    evaluator = ParallelTaskProxyEvaluator(models, proxy_data)
    identity_mask = np.ones((24, 32), dtype=np.bool_)
    teacher_dir = run_dir / "teacher"
    teacher_dir.mkdir()
    try:
        print("caching Original task-choice distributions ...", flush=True)
        teacher_search = evaluator.score_on_lane(0, identity_mask, split="search")
        teacher_holdout = evaluator.score_on_lane(0, identity_mask, split="holdout")
        teacher_scores_path = teacher_dir / "choice_log_likelihoods.npz"
        np.savez_compressed(
            teacher_scores_path,
            search=teacher_search.choice_log_likelihoods,
            holdout=teacher_holdout.choice_log_likelihoods,
        )
        teacher_summary_path = teacher_dir / "summary.json"
        write_json(
            teacher_summary_path,
            {
                "schema_version": "original-choice-teacher-v1",
                "mask": "32/32 identity",
                "training_performed": False,
                "search_cost": {
                    "wall_seconds": teacher_search.wall_seconds,
                    "gpu_seconds": teacher_search.gpu_seconds,
                },
                "holdout_cost": {
                    "wall_seconds": teacher_holdout.wall_seconds,
                    "gpu_seconds": teacher_holdout.gpu_seconds,
                },
                "search_choice_scores_sha256": sha256_file(teacher_scores_path),
            },
        )
        write_stage_manifest(
            teacher_dir / "stage_manifest.json",
            stage="original_choice_teacher",
            inputs={
                "data_stage": _ref(data_dir / "stage_manifest.json", run_dir),
                "config": ArtifactRef.capture(config_path),
                "script": ArtifactRef.capture(Path(__file__).resolve()),
            },
            outputs={
                "choice_scores": _ref(teacher_scores_path, run_dir),
                "summary": _ref(teacher_summary_path, run_dir),
            },
            metadata={"training_performed": False, "weight_updates": 0},
        )

        print("checking packed scorer and two-lane equivalence ...", flush=True)
        preflight = _proxy_preflight(
            evaluator,
            proxy_data,
            model0,
            tokenizer1,
            identity_mask,
            reap_mask,
            frequency_mask,
            teacher_search,
            tolerance=float(config["parallel"]["score_tolerance"]),
            batch_size=int(config["parallel"]["proxy_batch_size"]),
        )
        if not preflight["passed"]:
            raise RuntimeError(f"task proxy preflight failed: {preflight}")
        print(
            "proxy preflight PASS; max choice-LL delta="
            f"{max(preflight['choice_log_likelihood_max_absolute_deltas'].values()):.3g}",
            flush=True,
        )

        anchors = (
            ("reap", reap_mask),
            ("frequency", frequency_mask),
            ("old_nll_ea", old_nll_mask),
        )
        search = config["search"]
        mutation_counts = tuple(int(value) for value in search["mutation_swap_counts"])
        mutation_probabilities = tuple(
            float(value) for value in search["mutation_swap_probabilities"]
        )
        initial, initial_contexts = make_initial_population(
            anchors,
            population_size=int(search["population_size"]),
            seed=int(search["seed"]),
            mutation_counts=mutation_counts,
            mutation_probabilities=mutation_probabilities,
        )
        search_dir = run_dir / "search"
        search_dir.mkdir()
        initial_path = search_dir / "initial_population.npy"
        np.save(initial_path, initial, allow_pickle=False)

        import torch

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        evaluator.reset_runtime_counters()
        completed = 0

        def objective_many(masks: Sequence[np.ndarray]) -> tuple[TaskFitness, ...]:
            nonlocal completed
            raw_scores = evaluator.score_many(masks, split="search", parallel=True)
            values = tuple(
                _task_fitness(
                    proxy_data,
                    "search",
                    score,
                    teacher_search.choice_log_likelihoods,
                    config,
                )
                for score in raw_scores
            )
            completed += len(values)
            if completed == len(values) or completed % 16 == 0:
                current = ", ".join(
                    f"({value.objectives[0]:.4f},{value.objectives[1]:.4f})" for value in values
                )
                print(
                    f"task-aware NSGA-II: {completed}/{search['evaluation_budget']} "
                    f"unique candidates; objectives={current}",
                    flush=True,
                )
            return values

        search_started = perf_counter()
        result = run_task_nsga2(
            objective_many,
            initial,
            initial_contexts,
            seed=int(search["seed"]),
            population_size=int(search["population_size"]),
            offspring_size=int(search["offspring_size"]),
            generations=int(search["generations"]),
            tournament_size=int(search["tournament_size"]),
            mutation_counts=mutation_counts,
            mutation_probabilities=mutation_probabilities,
            max_duplicate_attempts=int(search["max_duplicate_attempts"]),
        )
        torch.cuda.synchronize()
        search_wall_seconds = perf_counter() - search_started
        search_peak_memory = int(torch.cuda.max_memory_allocated())
        search_max_active_lanes = evaluator.max_active
        search_parallel_calls = evaluator.parallel_calls
        search_sum_lane_wall_seconds = evaluator.sum_lane_wall_seconds
        if completed != int(search["evaluation_budget"]):
            raise RuntimeError("model evaluation count differs from the locked budget")

        reap_hash = mask_sha256(reap_mask, retained_count=28)
        reap_ids = [
            index
            for index, mask in enumerate(result.masks)
            if mask_sha256(mask, retained_count=28) == reap_hash
        ]
        if reap_ids != [0]:
            raise RuntimeError("REAP must be candidate zero exactly once")
        shortlist_ids = _shortlist_candidate_ids(
            result,
            reap_candidate_id=0,
            limit=int(config["rerank"]["evolved_candidates"]),
        )
        holdout_ids = (0, *shortlist_ids)
        evaluator.reset_runtime_counters()
        holdout_started = perf_counter()
        holdout_rows: list[dict[str, Any]] = []
        for start in range(0, len(holdout_ids), 2):
            batch_ids = holdout_ids[start : start + 2]
            raw_values = evaluator.score_many(
                [result.masks[index] for index in batch_ids],
                split="holdout",
                parallel=True,
            )
            for candidate_id, raw_score in zip(batch_ids, raw_values, strict=True):
                value = _task_fitness(
                    proxy_data,
                    "holdout",
                    raw_score,
                    teacher_holdout.choice_log_likelihoods,
                    config,
                )
                holdout_rows.append(
                    {
                        "candidate_id": int(candidate_id),
                        "candidate_sha256": mask_sha256(
                            result.masks[candidate_id], retained_count=28
                        ),
                        "objectives": {
                            TASK_ORDER[0]: value.objectives[0],
                            TASK_ORDER[1]: value.objectives[1],
                        },
                        "metrics": value.metrics,
                        "lane": value.lane,
                        "cost": value.cost,
                    }
                )
        holdout_wall_seconds = perf_counter() - holdout_started
        holdout_runtime = {
            "wall_seconds": holdout_wall_seconds,
            "sum_lane_wall_seconds": evaluator.sum_lane_wall_seconds,
            "parallel_calls": evaluator.parallel_calls,
            "observed_max_active_lanes": evaluator.max_active,
        }
    finally:
        evaluator.close()

    identity_after_search = [_parameter_identity(model0), _parameter_identity(model1)]
    if identity_after_search != identity_before:
        raise RuntimeError("model parameter pointer/version changed during NSGA-II search")

    all_masks_path = search_dir / "candidate_masks.npy"
    objectives_path = search_dir / "objectives.npy"
    trials_path = search_dir / "trials.jsonl"
    summary_path = search_dir / "summary.json"
    shortlist_path = search_dir / "shortlist.json"
    environment_path = search_dir / "environment.json"
    np.save(all_masks_path, result.masks, allow_pickle=False)
    np.save(objectives_path, result.objectives, allow_pickle=False)
    write_jsonl(trials_path, result.trials)
    fronts = nondominated_sort(
        result.objectives,
        candidate_ids=np.arange(result.objectives.shape[0], dtype=np.int64),
    )
    write_json(
        shortlist_path,
        {
            "schema_version": "task-aware-nsga2-shortlist-v1",
            "selection_rule": [
                "minimum_arc_objective",
                "minimum_openbookqa_objective",
                "minimum_taskwise_relative_minimax",
                "maximum_pareto_crowding",
            ],
            "reap_candidate_id": 0,
            "evolved_candidate_ids": list(shortlist_ids),
            "holdout_candidate_ids": list(holdout_ids),
            "validation_consulted": False,
        },
    )
    summary = {
        "schema_version": "task-aware-pruning-nsga2-search-v1",
        "algorithm": "task-aware fixed-cardinality NSGA-II",
        "task_supervised_mask_selection": True,
        "training_performed": False,
        "weight_updates": 0,
        "test_split_used": False,
        "validation_used_for_search": False,
        "seed": int(config["search"]["seed"]),
        "mu": int(config["search"]["population_size"]),
        "lambda": int(config["search"]["offspring_size"]),
        "generations": int(config["search"]["generations"]),
        "unique_evaluations": int(result.masks.shape[0]),
        "proposals": result.proposals,
        "duplicate_proposals": result.duplicate_proposals,
        "initial_population_sha256": result.initial_population_sha256,
        "final_population_ids": list(result.final_population_ids),
        "pareto_fronts": [list(front) for front in fronts],
        "pareto_front_size": len(fronts[0]),
        "shortlist_ids": list(shortlist_ids),
        "preflight": preflight,
        "parallel": {
            "replicas": 2,
            "streams": 2,
            "observed_max_active_lanes": search_max_active_lanes,
            "parallel_calls": search_parallel_calls,
            "sum_lane_wall_seconds": search_sum_lane_wall_seconds,
            "search_wall_seconds": search_wall_seconds,
            "effective_overlap": (
                search_sum_lane_wall_seconds / search_wall_seconds
                if search_wall_seconds > 0.0
                else None
            ),
            "peak_device_memory_bytes": search_peak_memory,
        },
        "parameter_accounting": parameter_counts,
    }
    write_json(summary_path, summary)
    write_json(environment_path, environment_snapshot())
    write_stage_manifest(
        search_dir / "stage_manifest.json",
        stage="task_aware_fixed_cardinality_nsga2",
        inputs={
            "config": ArtifactRef.capture(config_path),
            "protocol": ArtifactRef.capture(protocol_path),
            "source_verification": _ref(source_verification, run_dir),
            "data_stage": _ref(data_dir / "stage_manifest.json", run_dir),
            "teacher_stage": _ref(teacher_dir / "stage_manifest.json", run_dir),
            "frequency_selection": ArtifactRef.capture(source_paths["frequency"]),
            "reap_selection": ArtifactRef.capture(source_paths["reap"]),
            "old_nll_ea_selection": ArtifactRef.capture(source_paths["old_nll_ea"]),
            "nsga2_implementation": ArtifactRef.capture(
                workspace / "src/dsai6707/pruning_nsga2.py"
            ),
            "task_proxy_implementation": ArtifactRef.capture(
                workspace / "src/dsai6707/task_choice_proxy.py"
            ),
            "script": ArtifactRef.capture(Path(__file__).resolve()),
        },
        outputs={
            "initial_population": _ref(initial_path, run_dir),
            "candidate_masks": _ref(all_masks_path, run_dir),
            "objectives": _ref(objectives_path, run_dir),
            "trials": _ref(trials_path, run_dir),
            "summary": _ref(summary_path, run_dir),
            "shortlist": _ref(shortlist_path, run_dir),
            "environment": _ref(environment_path, run_dir),
        },
        metadata={
            "unique_evaluations": int(result.masks.shape[0]),
            "training_performed": False,
            "task_supervised_mask_selection": True,
        },
    )

    print("reranking Pareto candidates on disjoint train holdout and WikiText ...", flush=True)
    bundle = load_token_bundle(token_path)
    source_nll_path = source_run / "baselines" / "nll.json"
    source_nll = read_json(source_nll_path)
    original_validation_nll = float(source_nll["original"]["validation"]["nll"])
    nll_rows: dict[int, dict[str, Any]] = {}
    identity_plan = RoutingPlan.all_experts(num_layers=24, num_experts=32)
    nll_controller = GranitePruningController(model0, identity_plan)
    try:
        for candidate_id in holdout_ids:
            nll_controller.apply(_routing_plan(result.masks[candidate_id]))
            nll = token_weighted_full_model_nll(
                model0,
                bundle.validation,
                loss_tokens=int(config["data"]["validation_tokens"]),
                micro_batch_size=int(config["rerank"]["wikitext_nll_micro_batch_size"]),
            )
            nll_rows[int(candidate_id)] = {
                **nll.to_dict(),
                "ppl_ratio_vs_original": math.exp(float(nll.nll) - original_validation_nll),
            }
    finally:
        nll_controller.close()

    holdout_by_id = {int(row["candidate_id"]): row for row in holdout_rows}
    if set(holdout_by_id) != set(holdout_ids):
        raise RuntimeError("holdout evaluation did not cover the locked shortlist")
    reap_objectives = np.asarray(
        [holdout_by_id[0]["objectives"][task] for task in TASK_ORDER],
        dtype=np.float64,
    )
    eligible: list[tuple[tuple[float, float, float, int], int]] = []
    for candidate_id in shortlist_ids:
        objectives = np.asarray(
            [holdout_by_id[candidate_id]["objectives"][task] for task in TASK_ORDER],
            dtype=np.float64,
        )
        taskwise_no_worse = bool(np.all(objectives <= reap_objectives))
        nll_pass = float(nll_rows[candidate_id]["ppl_ratio_vs_original"]) <= float(
            config["rerank"]["wikitext_ppl_ratio_limit_vs_original"]
        )
        row = holdout_by_id[candidate_id]
        row["wikitext"] = nll_rows[candidate_id]
        row["taskwise_no_worse_than_reap"] = taskwise_no_worse
        row["wikitext_guard_passed"] = nll_pass
        row["eligible"] = taskwise_no_worse and nll_pass
        ratios = objectives / reap_objectives
        task_acc = [float(row["metrics"]["tasks"][task]["acc"]) for task in TASK_ORDER]
        row["relative_objectives_vs_reap"] = {
            task: float(ratio) for task, ratio in zip(TASK_ORDER, ratios, strict=True)
        }
        if row["eligible"]:
            eligible.append(
                (
                    (
                        float(np.max(ratios)),
                        float(np.mean(ratios)),
                        -float(np.mean(task_acc)),
                        int(candidate_id),
                    ),
                    int(candidate_id),
                )
            )
    holdout_by_id[0]["wikitext"] = nll_rows[0]
    holdout_by_id[0]["taskwise_no_worse_than_reap"] = True
    holdout_by_id[0]["wikitext_guard_passed"] = True
    holdout_by_id[0]["eligible"] = True
    holdout_by_id[0]["relative_objectives_vs_reap"] = {task: 1.0 for task in TASK_ORDER}
    if eligible:
        eligible.sort()
        winner_id = eligible[0][1]
        freeze_reason = "evolved_candidate_improved_or_matched_both_holdout_task_losses"
    else:
        winner_id = 0
        freeze_reason = "safe_fallback_to_reap_no_evolved_candidate_passed_both_task_losses"
    winner_mask = result.masks[winner_id]

    rerank_dir = run_dir / "rerank"
    rerank_dir.mkdir()
    holdout_path = rerank_dir / "holdout.json"
    frozen_path = rerank_dir / "frozen_selection.json"
    winner_selection_path = rerank_dir / "winner_selection.json"
    write_json(
        holdout_path,
        {
            "schema_version": "task-aware-nsga2-holdout-v1",
            "candidate_order": list(holdout_ids),
            "candidates": [holdout_by_id[index] for index in holdout_ids],
            "runtime": holdout_runtime,
            "selection_comparator": [
                "both_task_composite_losses_no_worse_than_reap",
                "wikitext_ppl_ratio_vs_original_at_most_1.15",
                "minimum_worst_relative_task_loss",
                "minimum_mean_relative_task_loss",
                "maximum_mean_raw_accuracy",
                "minimum_candidate_id",
            ],
            "validation_consulted": False,
        },
    )
    write_json(winner_selection_path, _selection_payload(winner_mask, method="task_aware_nsga2"))
    write_json(
        frozen_path,
        {
            "schema_version": "frozen-pruning-selection-v1",
            "winner_candidate_id": int(winner_id),
            "winner_candidate_sha256": mask_sha256(winner_mask, retained_count=28),
            "winner_is_direct_reap": winner_id == 0,
            "freeze_reason": freeze_reason,
            "selection_inputs": {
                "search_stage": _ref(search_dir / "stage_manifest.json", run_dir).to_dict(),
                "holdout": _ref(holdout_path, run_dir).to_dict(),
            },
            "task_validation_consulted_before_freeze": False,
            "test_split_used": False,
        },
    )
    write_stage_manifest(
        rerank_dir / "stage_manifest.json",
        stage="task_aware_nsga2_independent_holdout_freeze",
        inputs={
            "search_stage": _ref(search_dir / "stage_manifest.json", run_dir),
            "token_artifact": ArtifactRef.capture(token_path),
            "source_nll": ArtifactRef.capture(source_nll_path),
            "config": ArtifactRef.capture(config_path),
            "script": ArtifactRef.capture(Path(__file__).resolve()),
        },
        outputs={
            "holdout": _ref(holdout_path, run_dir),
            "frozen_selection": _ref(frozen_path, run_dir),
            "winner_selection": _ref(winner_selection_path, run_dir),
        },
        metadata={
            "winner_candidate_id": int(winner_id),
            "winner_is_direct_reap": winner_id == 0,
            "validation_consulted_before_freeze": False,
        },
    )
    # Verify the immutable freeze artifact before any downstream task metric is read.
    verify_stage_manifest(rerank_dir / "stage_manifest.json", run_dir=run_dir)

    frozen_kind = "REAP fallback" if winner_id == 0 else "evolved"
    print(f"winner frozen: candidate {winner_id} ({frozen_kind})", flush=True)
    source_original_path = source_run / str(config["source"]["original_validation"])
    source_reap_path = source_run / str(config["source"]["reap_validation"])
    original_artifact = read_json(source_original_path)
    reap_artifact = read_json(source_reap_path)
    validation_dir = run_dir / "validation"
    validation_dir.mkdir()
    shutil.copyfile(source_original_path, validation_dir / "original.json")
    shutil.copyfile(source_reap_path, validation_dir / "reap.json")

    if winner_id == 0:
        winner_artifact = copy.deepcopy(reap_artifact)
        winner_artifact["method"] = "task_aware_nsga2_safe_reap_fallback"
        winner_artifact["seed"] = int(config["search"]["seed"])
        winner_artifact["reused_from_verified_source"] = True
        write_json(validation_dir / "winner.json", winner_artifact)
        new_validation_executed = False
    else:
        del model1
        models.pop()
        import torch

        torch.cuda.empty_cache()
        controller = GranitePruningController(model0, identity_plan)
        try:
            controller.apply(_routing_plan(winner_mask))
            print(
                "evaluating the frozen NSGA-II winner once on full task validation ...",
                flush=True,
            )
            winner_artifact = evaluate_zero_shot(
                model0,
                tokenizer0,
                _development_config(config),
                output_path=validation_dir / "winner.json",
                method="task_aware_fixed_cardinality_nsga2",
                seed=int(config["search"]["seed"]),
            )
        finally:
            controller.close()
        new_validation_executed = True

    summaries = {
        "original": _candidate_development_summary(original_artifact, original_artifact),
        "reap": _candidate_development_summary(reap_artifact, original_artifact),
        "winner": _candidate_development_summary(winner_artifact, original_artifact),
    }
    validation_analysis = {
        "schema_version": "task-aware-nsga2-validation-v1",
        "split": "validation",
        "tasks": list(TASK_ORDER),
        "winner_candidate_id": int(winner_id),
        "winner_is_direct_reap": winner_id == 0,
        "new_winner_validation_executed_once": new_validation_executed,
        "validation_used_for_selection": False,
        "test_split_used": False,
        "evaluations": summaries,
        "winner_vs_reap": {
            "mean_accuracy_absolute_change": (
                summaries["winner"]["mean_accuracy"] - summaries["reap"]["mean_accuracy"]
            ),
            "task_accuracy_absolute_change": {
                task: (
                    summaries["winner"]["task_accuracy"][task]
                    - summaries["reap"]["task_accuracy"][task]
                )
                for task in TASK_ORDER
            },
        },
    }
    validation_analysis_path = validation_dir / "analysis.json"
    write_json(validation_analysis_path, validation_analysis)
    write_stage_manifest(
        validation_dir / "stage_manifest.json",
        stage="frozen_task_aware_nsga2_validation",
        inputs={
            "rerank_stage": _ref(rerank_dir / "stage_manifest.json", run_dir),
            "frozen_selection": _ref(frozen_path, run_dir),
            "source_manifest_snapshot": _ref(source_snapshot, run_dir),
            "source_verification": _ref(source_verification, run_dir),
            "source_original": ArtifactRef.capture(source_original_path),
            "source_reap": ArtifactRef.capture(source_reap_path),
            "script": ArtifactRef.capture(Path(__file__).resolve()),
        },
        outputs={
            "original": _ref(validation_dir / "original.json", run_dir),
            "reap": _ref(validation_dir / "reap.json", run_dir),
            "winner": _ref(validation_dir / "winner.json", run_dir),
            "analysis": _ref(validation_analysis_path, run_dir),
        },
        metadata={
            "winner_candidate_id": int(winner_id),
            "validation_used_for_selection": False,
            "test_split_used": False,
        },
    )

    identity_model0_final = _parameter_identity(model0)
    if identity_model0_final != identity_before[0]:
        raise RuntimeError("model parameters changed during final validation")
    final_analysis = {
        "schema_version": "task-aware-nsga2-pruning-analysis-v1",
        "model_id": str(config["model"]["id"]),
        "seed": int(config["search"]["seed"]),
        "algorithm": "task-aware fixed-cardinality NSGA-II",
        "task_supervised_mask_selection": True,
        "training_performed": False,
        "weight_updates": 0,
        "single_visible_gpu": True,
        "two_in_gpu_model_replicas_for_search": True,
        "test_split_used": False,
        "validation_used_for_selection": False,
        "retained_experts_per_layer": 28,
        "parameter_accounting": parameter_counts,
        "search": summary,
        "winner_freeze": read_json(frozen_path),
        "holdout": {
            "reap": holdout_by_id[0],
            "winner": holdout_by_id[winner_id],
        },
        "validation": validation_analysis,
        "model_parameter_identity_unchanged": True,
    }
    analysis_path = run_dir / "analysis.json"
    write_json(analysis_path, final_analysis)
    write_stage_manifest(
        run_dir / "stage_manifest.json",
        stage="task_aware_nsga2_pruning_complete",
        inputs={
            "config": ArtifactRef.capture(config_path),
            "protocol": ArtifactRef.capture(protocol_path),
            "source_manifest_snapshot": _ref(source_snapshot, run_dir),
            "source_verification": _ref(source_verification, run_dir),
            "data_stage": _ref(data_dir / "stage_manifest.json", run_dir),
            "teacher_stage": _ref(teacher_dir / "stage_manifest.json", run_dir),
            "search_stage": _ref(search_dir / "stage_manifest.json", run_dir),
            "rerank_stage": _ref(rerank_dir / "stage_manifest.json", run_dir),
            "validation_stage": _ref(validation_dir / "stage_manifest.json", run_dir),
            "script": ArtifactRef.capture(Path(__file__).resolve()),
        },
        outputs={"analysis": _ref(analysis_path, run_dir)},
        metadata={
            "winner_candidate_id": int(winner_id),
            "winner_is_direct_reap": winner_id == 0,
            "training_performed": False,
            "test_split_used": False,
        },
    )
    verify_stage_manifest(run_dir / "stage_manifest.json", run_dir=run_dir)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/expert_pruning_28of32_taskaware_nsga2.yaml",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = run(args.config)
    print(f"task-aware NSGA-II experiment saved in {output}", flush=True)


if __name__ == "__main__":
    main()

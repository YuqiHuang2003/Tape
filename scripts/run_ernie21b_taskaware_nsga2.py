#!/usr/bin/env python3
"""Run a quick task-aware fixed-cardinality NSGA-II search on ERNIE 21B.

The model weights stay frozen.  An individual is a 27x64 routed-expert mask
with a configured, fixed number of retained experts per row.  Candidate fitness
uses disjoint ARC-Easy and OpenBookQA training subsets; task validation is
opened only after the holdout rule has frozen one winner.
"""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.metadata
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "external" / "reap-official" / ".deps"
SOURCE = ROOT / "src"
for bootstrap_path in (SOURCE, DEPS):
    value = str(bootstrap_path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from dsai6707.config import (  # noqa: E402
    FORMAL_EVALUATION_PROTOCOL_ID,
    LOCKED_TASK_DATASETS,
)
from dsai6707.ernie_pruning import (  # noqa: E402
    ErnieExpertMaskController,
    ernie_moe_layers,
)
from dsai6707.evaluation import evaluate_zero_shot  # noqa: E402
from dsai6707.io import (  # noqa: E402
    environment_snapshot,
    read_json,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)
from dsai6707.pruning_nsga2 import (  # noqa: E402
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
from dsai6707.task_choice_proxy import (  # noqa: E402
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
EXPECTED_VERSIONS = {
    "transformers": "4.53.3",
    "datasets": "3.6.0",
    "accelerate": "1.7.0",
    "lm-eval": "0.4.8",
}
RNG_NAMESPACE = 0x45524E49  # "ERNI"


class ExperimentError(RuntimeError):
    """Raised when a locked experiment precondition is violated."""


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ExperimentError(f"configuration is not a mapping: {path}")
    return value


def _workspace_path(value: str, *, kind: str, must_exist: bool = True) -> Path:
    path = (ROOT / value).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ExperimentError(f"{kind} must stay inside the workspace") from exc
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def _require_runtime() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip() or "," in visible:
        raise ExperimentError("set CUDA_VISIBLE_DEVICES to exactly one physical GPU")
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in {":4096:8", ":16:8"}:
        raise ExperimentError("set CUBLAS_WORKSPACE_CONFIG before CUDA initialization")
    for package, expected in EXPECTED_VERSIONS.items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise ExperimentError(f"{package}=={expected} required, found {actual}")


def _validate_config(config: Mapping[str, Any]) -> None:
    pruning = config["pruning"]
    search = config["search"]
    data = config["data"]
    runtime = config["runtime"]
    topology = (
        int(pruning["moe_layers"]),
        int(pruning["original_experts_per_layer"]),
        int(pruning["top_k"]),
    )
    retained = int(pruning["retained_experts_per_layer"])
    if topology != (27, 64, 6):
        raise ExperimentError("this runner requires the ERNIE 27x64, top-6 topology")
    if not 6 <= retained < 64:
        raise ExperimentError("retained experts per layer must be in [6, 64)")
    population = int(search["population_size"])
    offspring = int(search["offspring_size"])
    generations = int(search["generations"])
    budget = int(search["evaluation_budget"])
    if (population, offspring, generations, budget) != (8, 8, 5, 48):
        raise ExperimentError("quick search must use mu=8, lambda=8, five generations")
    if population + generations * offspring != budget:
        raise ExperimentError("inconsistent NSGA-II budget")
    if list(config["fitness"]["objectives"]) != list(TASK_ORDER):
        raise ExperimentError("fitness objectives must be ARC-Easy then OpenBookQA")
    for field in ("search_samples_per_task", "holdout_samples_per_task"):
        count = int(data[field])
        if count <= 0 or count % 4:
            raise ExperimentError(f"data.{field} must be a positive multiple of four")
    if int(runtime["model_replicas"]) != 1 or not bool(runtime["single_visible_gpu"]):
        raise ExperimentError("ERNIE search must use one model replica on one GPU")
    if bool(runtime.get("physical_preflight_required", True)) and not config["model"].get(
        "physical_reap_path"
    ):
        raise ExperimentError("a physical REAP path is required by the configured preflight")


def _configure_torch(torch: Any, *, seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ExperimentError("the process must see exactly one CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise ExperimentError("the selected GPU does not support BF16")


def _load_model(path: Path, *, expected_experts: int) -> tuple[Any, dict[str, Any]]:
    import torch
    import transformers

    loaded = transformers.AutoModelForCausalLM.from_pretrained(
        str(path),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"},
        output_loading_info=True,
        attn_implementation="sdpa",
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise ExperimentError("model loader did not return loading evidence")
    model, loading = loaded
    for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        if list(loading.get(field, []) or []):
            raise ExperimentError(f"non-clean model load: {field}={loading[field][:8]}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    config = model.config
    actual = (
        int(config.num_hidden_layers),
        int(config.moe_num_experts),
        int(config.moe_num_shared_experts),
        int(config.moe_k),
        str(config.moe_gate_act),
    )
    if actual != (28, expected_experts, 2, 6, "softmax"):
        raise ExperimentError(f"unexpected ERNIE topology: {actual}")
    evidence = {
        "path": str(path),
        "class": model.__class__.__name__,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "num_hidden_layers": actual[0],
        "moe_num_experts": actual[1],
        "moe_num_shared_experts": actual[2],
        "moe_k": actual[3],
        "moe_gate_act": actual[4],
        "loading_info_clean": True,
    }
    return model, evidence


def _physical_parameter_accounting(
    model: Any, *, retained_count: int
) -> dict[str, Any]:
    """Count parameters a physically sliced checkpoint would retain."""

    total = int(sum(parameter.numel() for parameter in model.parameters()))
    original_count = 64
    removable_per_expert = 0
    for layer_index, mlp in enumerate(ernie_moe_layers(model)):
        expert_sizes = [
            sum(parameter.numel() for parameter in expert.parameters())
            for expert in mlp.experts
        ]
        if len(expert_sizes) != original_count or len(set(expert_sizes)) != 1:
            raise ExperimentError(f"unexpected expert parameter sizes in layer {layer_index}")
        gate = mlp.gate
        if gate.weight.shape[0] != original_count:
            raise ExperimentError(f"unexpected gate rows in layer {layer_index}")
        gate_row = int(gate.weight[0].numel())
        if gate.bias is not None:
            if gate.bias.shape != (original_count,):
                raise ExperimentError(f"unexpected gate bias in layer {layer_index}")
            gate_row += 1
        statics = getattr(mlp, "moe_statics", None)
        correction = getattr(statics, "e_score_correction_bias", None)
        correction_per_expert = 0
        if correction is not None:
            if correction.shape[-1] != original_count:
                raise ExperimentError(
                    f"unexpected score-correction bias in layer {layer_index}"
                )
            if correction.numel() % original_count:
                raise ExperimentError(
                    f"non-divisible score-correction bias in layer {layer_index}"
                )
            correction_per_expert = int(correction.numel() // original_count)
        removable_per_expert += (
            int(expert_sizes[0]) + gate_row + correction_per_expert
        )
    removed_count = original_count - retained_count
    retained = total - removed_count * removable_per_expert
    return {
        "base_parameters": total,
        "estimated_physical_parameters": retained,
        "estimated_parameter_retention": retained / total,
        "removed_parameters": total - retained,
        "removed_experts_per_moe_layer": removed_count,
        "retained_experts_per_moe_layer": retained_count,
        "counting_method": (
            "exact expert-module, gate-row, and score-correction-entry count "
            "without checkpoint export"
        ),
    }


def _load_tokenizer(path: Path) -> Any:
    import transformers

    return transformers.AutoTokenizer.from_pretrained(
        str(path),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )


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
        {"task": task, "dataset_kwargs": {"revision": str(revisions[task])}}
        for task in TASK_ORDER
    ]
    resolved = get_task_dict(specs, TaskManager())
    if set(resolved) != set(TASK_ORDER):
        raise ExperimentError("lm-eval resolved unexpected tasks")

    search_count = int(config["data"]["search_samples_per_task"])
    holdout_count = int(config["data"]["holdout_samples_per_task"])
    preflight_count = int(config["runtime"]["logical_physical_preflight_questions_per_task"])
    seed = int(config["search"]["seed"])
    prefix = tokenizer.bos_token_id
    if prefix is None:
        prefix = tokenizer.eos_token_id
    by_split: dict[str, list[ChoiceQuestion]] = {
        "search": [],
        "holdout": [],
        "preflight": [],
    }
    task_evidence: dict[str, Any] = {}
    for task_name in TASK_ORDER:
        task = resolved[task_name]
        locked = LOCKED_TASK_DATASETS[task_name]
        for key, expected in {
            "dataset_path": locked["path"],
            "dataset_name": locked["name"],
            "training_split": "train",
            "output_type": "multiple_choice",
        }.items():
            if task.get_config(key) != expected:
                raise ExperimentError(f"unexpected {task_name} {key}")
        kwargs = task.get_config("dataset_kwargs")
        if not isinstance(kwargs, Mapping) or kwargs.get("revision") != str(revisions[task_name]):
            raise ExperimentError(f"{task_name} did not use the pinned revision")
        documents = task.training_docs()
        fingerprint = getattr(documents, "_fingerprint", None)
        split = select_gold_stratified_documents(
            task,
            documents,
            task_name=task_name,
            seed=seed,
            search_per_gold=search_count // 4,
            holdout_per_gold=holdout_count // 4,
            required_choice_count=4,
        )
        search_questions = build_choice_questions(
            task,
            documents,
            split.search_indices,
            tokenizer,
            task_name=task_name,
            split="train-search",
            prefix_token_id=prefix,
        )
        holdout_questions = build_choice_questions(
            task,
            documents,
            split.holdout_indices,
            tokenizer,
            task_name=task_name,
            split="train-holdout",
            prefix_token_id=prefix,
        )
        if preflight_count > len(search_questions):
            raise ExperimentError("preflight subset exceeds search subset")
        by_split["search"].extend(search_questions)
        by_split["holdout"].extend(holdout_questions)
        by_split["preflight"].extend(search_questions[:preflight_count])
        task_evidence[task_name] = {
            "revision": str(revisions[task_name]),
            "training_fingerprint": fingerprint,
            "selection": split.to_dict(),
            "search": choice_proxy_hashes(search_questions),
            "holdout": choice_proxy_hashes(holdout_questions),
        }

    questions = {name: tuple(values) for name, values in by_split.items()}
    requests = {name: flatten_choice_requests(values) for name, values in questions.items()}
    max_length = min(int(getattr(tokenizer, "model_max_length", 4096)), 4096)
    if max_length <= 0:
        max_length = 4096
    batch_size = int(config["runtime"]["proxy_batch_size"])
    batches = {
        name: pack_choice_requests(
            values,
            batch_size=batch_size,
            max_length=max_length,
            pad_token_id=0,
        )
        for name, values in requests.items()
    }
    evidence = {
        "source_split": "train",
        "validation_used": False,
        "test_used": False,
        "seed": seed,
        "task_order": list(TASK_ORDER),
        "tasks": task_evidence,
        "packing": {
            name: {
                "questions": len(questions[name]),
                "requests": len(requests[name]),
                "batches": len(batches[name]),
                "scored_tokens": sum(batch.scored_tokens for batch in batches[name]),
                "padded_input_positions": sum(
                    batch.padded_input_positions for batch in batches[name]
                ),
            }
            for name in questions
        },
    }
    return ProxyData(questions, requests, batches, evidence)


@dataclass(frozen=True)
class GPUChoiceBatch:
    input_ids: Any
    row_indices: Any
    time_indices: Any
    target_ids: Any
    segment_indices: Any
    num_requests: int


@dataclass(frozen=True)
class ProxyScore:
    choice_log_likelihoods: np.ndarray
    wall_seconds: float
    gpu_seconds: float


class SerialChoiceScorer:
    """Reuse packed GPU requests while evaluating one mask at a time."""

    def __init__(self, model: Any, proxy_data: ProxyData) -> None:
        import torch

        self.model = model
        self.proxy_data = proxy_data
        self.device = next(model.parameters()).device
        self.batches: dict[str, tuple[GPUChoiceBatch, ...]] = {}
        for split, cpu_batches in proxy_data.batches.items():
            converted: list[GPUChoiceBatch] = []
            for batch in cpu_batches:
                converted.append(
                    GPUChoiceBatch(
                        input_ids=torch.as_tensor(
                            np.array(batch.input_ids, copy=True),
                            dtype=torch.long,
                            device=self.device,
                        ),
                        row_indices=torch.as_tensor(
                            np.array(batch.row_indices, copy=True),
                            dtype=torch.long,
                            device=self.device,
                        ),
                        time_indices=torch.as_tensor(
                            np.array(batch.time_indices, copy=True),
                            dtype=torch.long,
                            device=self.device,
                        ),
                        target_ids=torch.as_tensor(
                            np.array(batch.target_ids, copy=True),
                            dtype=torch.long,
                            device=self.device,
                        ),
                        segment_indices=torch.as_tensor(
                            np.array(batch.segment_indices, copy=True),
                            dtype=torch.long,
                            device=self.device,
                        ),
                        num_requests=batch.num_requests,
                    )
                )
            self.batches[split] = tuple(converted)

    def score(self, split: str) -> ProxyScore:
        import torch

        if split not in self.batches:
            raise ValueError(f"unknown proxy split: {split}")
        torch.cuda.synchronize(self.device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_started = perf_counter()
        outputs: list[np.ndarray] = []
        start.record()
        with torch.inference_mode():
            for batch in self.batches[split]:
                logits = self.model(batch.input_ids, use_cache=False).logits
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                token_values = log_probs[
                    batch.row_indices,
                    batch.time_indices,
                    :,
                ].gather(1, batch.target_ids[:, None]).squeeze(1)
                request_values = torch.stack(
                    [
                        token_values[batch.segment_indices == segment].sum()
                        for segment in range(batch.num_requests)
                    ]
                )
                outputs.append(request_values.float().cpu().numpy())
                del logits, log_probs, token_values, request_values
        end.record()
        end.synchronize()
        restored = restore_choice_log_likelihoods(
            self.proxy_data.batches[split],
            outputs,
            num_requests=len(self.proxy_data.requests[split]),
        )
        return ProxyScore(
            choice_log_likelihoods=restored,
            wall_seconds=perf_counter() - wall_started,
            gpu_seconds=float(start.elapsed_time(end) / 1000.0),
        )


def _task_metrics(
    proxy: ProxyData,
    split: str,
    score: ProxyScore,
    teacher: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    metrics = choice_proxy_distillation_metrics(
        proxy.questions[split],
        score.choice_log_likelihoods,
        teacher,
        cross_entropy_weight=float(config["fitness"]["gold_ce_weight"]),
        kl_weight=float(config["fitness"]["original_kl_weight"]),
    )
    if tuple(metrics["task_order"]) != TASK_ORDER:
        raise ExperimentError("proxy task order changed")
    objectives = np.asarray(metrics["task_objectives"], dtype=np.float64)
    if objectives.shape != (2,) or not np.isfinite(objectives).all():
        raise ExperimentError("invalid task objectives")
    return objectives, metrics


def _question_predictions(
    questions: Sequence[ChoiceQuestion], values: np.ndarray
) -> tuple[int, ...]:
    predictions: list[int] = []
    cursor = 0
    for question in questions:
        stop = cursor + len(question.choices)
        predictions.append(int(np.argmax(values[cursor:stop])))
        cursor = stop
    if cursor != len(values):
        raise ExperimentError("choice score length disagrees with questions")
    return tuple(predictions)


def _mask_from_lowest(scores: np.ndarray, *, prune_count: int) -> np.ndarray:
    import torch

    values = torch.as_tensor(np.asarray(scores))
    if values.shape != (64,) or not bool(torch.isfinite(values).all()):
        raise ExperimentError("observer score vector must contain 64 finite values")
    _, pruned = torch.topk(values, prune_count, largest=False)
    mask = np.ones(64, dtype=np.bool_)
    mask[pruned.cpu().numpy().astype(np.int64)] = False
    return mask


def _observer_anchors(
    path: Path, *, retained_count: int
) -> tuple[list[tuple[str, np.ndarray]], dict[str, Any]]:
    import torch

    observer = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(observer, dict) or list(observer) != list(range(1, 28)):
        raise ExperimentError("observer must contain ordered MoE layers 1..27")
    rows: dict[str, list[np.ndarray]] = {
        "reap": [],
        "frequency": [],
        "hybrid": [],
        "ean_mean": [],
    }
    evidence_layers: list[dict[str, Any]] = []
    prune_count = 64 - retained_count
    for layer in range(1, 28):
        record = observer[layer]
        reap = np.asarray(record["reap"].cpu(), dtype=np.float64)
        frequency = np.asarray(record["expert_frequency"].cpu(), dtype=np.float64)
        ean_mean = np.asarray(record["ean_mean"].cpu(), dtype=np.float64)
        for name, values in (("reap", reap), ("frequency", frequency), ("ean_mean", ean_mean)):
            if values.shape != (64,) or not np.isfinite(values).all():
                raise ExperimentError(f"invalid {name} observer values in layer {layer}")
        reap_mask = _mask_from_lowest(reap, prune_count=prune_count)
        frequency_mask = _mask_from_lowest(frequency, prune_count=prune_count)
        ean_mask = _mask_from_lowest(ean_mean, prune_count=prune_count)
        log_frequency = np.log1p(frequency)

        def standardize(values: np.ndarray) -> np.ndarray:
            deviation = float(values.std())
            return (values - float(values.mean())) / max(deviation, 1e-12)

        hybrid_score = standardize(reap) + standardize(log_frequency)
        hybrid_mask = _mask_from_lowest(hybrid_score, prune_count=prune_count)
        rows["reap"].append(reap_mask)
        rows["frequency"].append(frequency_mask)
        rows["hybrid"].append(hybrid_mask)
        rows["ean_mean"].append(ean_mask)
        evidence_layers.append(
            {
                "observer_layer": layer,
                "mask_row": layer - 1,
                "reap_retained": np.flatnonzero(reap_mask).astype(int).tolist(),
                "frequency_retained": np.flatnonzero(frequency_mask).astype(int).tolist(),
                "hybrid_retained": np.flatnonzero(hybrid_mask).astype(int).tolist(),
                "ean_mean_retained": np.flatnonzero(ean_mask).astype(int).tolist(),
            }
        )
    anchors = [(name, np.stack(values)) for name, values in rows.items()]
    unique: list[tuple[str, np.ndarray]] = []
    known: set[str] = set()
    for name, mask in anchors:
        validate_fixed_cardinality_mask(mask, retained_count=retained_count)
        digest = mask_sha256(mask, retained_count=retained_count)
        if digest not in known:
            known.add(digest)
            unique.append((name, mask))
    if not unique or unique[0][0] != "reap":
        raise ExperimentError("REAP anchor is missing")
    evidence = {
        "observer_path": str(path),
        "observer_sha256": sha256_file(path),
        "retained_experts_per_layer": retained_count,
        "pruned_experts_per_layer": prune_count,
        "unique_anchor_names": [name for name, _ in unique],
        "layers": evidence_layers,
    }
    return unique, evidence


def _mutation_payload(mutation: Any) -> list[dict[str, int]]:
    return [
        {
            "mask_row": int(swap.layer),
            "decoder_layer": int(swap.layer) + 1,
            "removed_expert": int(swap.removed_expert),
            "added_expert": int(swap.added_expert),
        }
        for swap in mutation.swaps
    ]


def _initial_population(
    anchors: Sequence[tuple[str, np.ndarray]],
    config: Mapping[str, Any],
    *,
    retained_count: int,
) -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    search = config["search"]
    size = int(search["population_size"])
    seed = int(search["seed"])
    counts = tuple(int(value) for value in search["mutation_swap_counts"])
    probabilities = tuple(float(value) for value in search["mutation_swap_probabilities"])
    rng = np.random.default_rng(np.random.SeedSequence([seed, RNG_NAMESPACE, 0]))
    masks: list[np.ndarray] = []
    contexts: list[dict[str, Any]] = []
    known: set[str] = set()
    for name, raw in anchors:
        mask = np.ascontiguousarray(raw, dtype=np.bool_)
        digest = mask_sha256(mask, retained_count=retained_count)
        if digest in known:
            continue
        known.add(digest)
        masks.append(mask.copy())
        contexts.append({"kind": "anchor", "anchor": name, "mutation": []})
        if len(masks) == size:
            break
    attempts = 0
    while len(masks) < size:
        attempts += 1
        if attempts > 10_000:
            raise ExperimentError("could not create a unique initial population")
        anchor_name, anchor = anchors[(attempts - 1) % len(anchors)]
        count = int(rng.choice(counts, p=probabilities))
        mutation = mutate_distinct_layer_swaps(
            anchor,
            rng=rng,
            min_layers=count,
            max_layers=count,
        )
        digest = mask_sha256(mutation.mask, retained_count=retained_count)
        if digest in known:
            continue
        known.add(digest)
        masks.append(mutation.mask)
        contexts.append(
            {
                "kind": "anchor_mutation",
                "anchor": anchor_name,
                "mutation": _mutation_payload(mutation),
            }
        )
    return np.stack(masks), tuple(contexts)


@dataclass(frozen=True)
class SearchResult:
    masks: np.ndarray
    objectives: np.ndarray
    trials: tuple[dict[str, Any], ...]
    final_population_ids: tuple[int, ...]
    proposals: int
    duplicates: int


def _run_nsga2(
    scorer: SerialChoiceScorer,
    controller: ErnieExpertMaskController,
    proxy: ProxyData,
    teacher_search: np.ndarray,
    initial: np.ndarray,
    contexts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    retained_count: int,
) -> SearchResult:
    search = config["search"]
    seed = int(search["seed"])
    population_size = int(search["population_size"])
    offspring_size = int(search["offspring_size"])
    generations = int(search["generations"])
    budget = int(search["evaluation_budget"])
    tournament_size = int(search["tournament_size"])
    counts = tuple(int(value) for value in search["mutation_swap_counts"])
    probabilities = tuple(float(value) for value in search["mutation_swap_probabilities"])
    max_duplicate_attempts = int(search["max_duplicate_attempts"])
    rng = np.random.default_rng(np.random.SeedSequence([seed, RNG_NAMESPACE, 1]))
    masks: list[np.ndarray] = []
    objectives: list[np.ndarray] = []
    trials: list[dict[str, Any]] = []
    known: set[str] = set()
    proposals = 0
    duplicates = 0

    def evaluate(mask: np.ndarray, context: Mapping[str, Any], generation: int) -> int:
        candidate_id = len(masks)
        digest = mask_sha256(mask, retained_count=retained_count)
        if digest in known:
            raise ExperimentError("duplicate mask reached evaluator")
        known.add(digest)
        controller.apply(mask)
        raw = scorer.score("search")
        objective, metrics = _task_metrics(
            proxy,
            "search",
            raw,
            teacher_search,
            config,
        )
        masks.append(mask.copy())
        objectives.append(objective)
        row = {
            "candidate_id": candidate_id,
            "generation": generation,
            "mask_sha256": digest,
            "objectives": {
                TASK_ORDER[0]: float(objective[0]),
                TASK_ORDER[1]: float(objective[1]),
            },
            "metrics": metrics,
            "cost": {
                "wall_seconds": raw.wall_seconds,
                "gpu_seconds": raw.gpu_seconds,
            },
            "proposal": dict(context),
        }
        trials.append(row)
        mean_raw_accuracy = float(
            np.mean([metrics["tasks"][task]["acc"] for task in TASK_ORDER])
        )
        print(
            f"candidate {candidate_id + 1:02d}/{budget}: "
            f"ARC={objective[0]:.5f}, OBQA={objective[1]:.5f}, "
            f"raw-acc={mean_raw_accuracy:.4f}",
            flush=True,
        )
        return candidate_id

    population_ids = np.asarray(
        [evaluate(mask, context, 0) for mask, context in zip(initial, contexts, strict=True)],
        dtype=np.int64,
    )
    population_objectives = np.asarray(
        [objectives[int(candidate)] for candidate in population_ids],
        dtype=np.float64,
    )
    for generation in range(1, generations + 1):
        ranks, distances = rank_and_crowding(
            population_objectives,
            candidate_ids=population_ids,
        )
        children: list[np.ndarray] = []
        child_contexts: list[dict[str, Any]] = []
        duplicate_streak = 0
        mating_event = 0
        while len(children) < offspring_size:
            if duplicate_streak >= max_duplicate_attempts:
                raise ExperimentError("too many duplicate offspring proposals")
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
            first_id = int(population_ids[first_position])
            second_id = int(population_ids[second_position])
            first_child, second_child = balanced_set_crossover(
                masks[first_id], masks[second_id], rng=rng
            )
            mating_event += 1
            for sibling, crossed in enumerate((first_child, second_child)):
                proposals += 1
                count = int(rng.choice(counts, p=probabilities))
                mutation = mutate_distinct_layer_swaps(
                    crossed,
                    rng=rng,
                    min_layers=count,
                    max_layers=count,
                )
                digest = mask_sha256(mutation.mask, retained_count=retained_count)
                if digest in known or any(
                    mask_sha256(value, retained_count=retained_count) == digest
                    for value in children
                ):
                    duplicates += 1
                    duplicate_streak += 1
                    continue
                duplicate_streak = 0
                children.append(mutation.mask)
                child_contexts.append(
                    {
                        "kind": "offspring",
                        "parent_ids": [first_id, second_id],
                        "mating_event": mating_event,
                        "sibling": sibling,
                        "crossover": "balanced_retained_set",
                        "mutation": _mutation_payload(mutation),
                    }
                )
                if len(children) == offspring_size:
                    break
        child_ids = np.asarray(
            [
                evaluate(mask, context, generation)
                for mask, context in zip(children, child_contexts, strict=True)
            ],
            dtype=np.int64,
        )
        child_objectives = np.asarray(
            [objectives[int(candidate)] for candidate in child_ids], dtype=np.float64
        )
        combined_ids = np.concatenate((population_ids, child_ids))
        combined_objectives = np.concatenate(
            (population_objectives, child_objectives), axis=0
        )
        positions = select_survivors(
            combined_objectives,
            combined_ids,
            population_size,
        )
        population_ids = combined_ids[positions]
        population_objectives = combined_objectives[positions]
    if len(masks) != budget or len(known) != budget:
        raise ExperimentError("unique evaluation accounting failed")
    return SearchResult(
        masks=np.stack(masks),
        objectives=np.asarray(objectives, dtype=np.float64),
        trials=tuple(trials),
        final_population_ids=tuple(int(value) for value in population_ids),
        proposals=proposals,
        duplicates=duplicates,
    )


def _shortlist(result: SearchResult, *, limit: int) -> tuple[int, ...]:
    ids = np.arange(result.objectives.shape[0], dtype=np.int64)
    front = nondominated_sort(result.objectives, candidate_ids=ids)[0]
    evolved = [int(index) for index in front if int(index) != 0]
    if not evolved:
        return ()
    reap = result.objectives[0]
    distances = crowding_distances(result.objectives, front, candidate_ids=ids)
    selectors = (
        lambda index: (float(result.objectives[index, 0]), index),
        lambda index: (float(result.objectives[index, 1]), index),
        lambda index: (
            float(np.max(result.objectives[index] / reap)),
            float(np.mean(result.objectives[index] / reap)),
            -float(distances[index]),
            index,
        ),
    )
    chosen: list[int] = []
    for selector in selectors:
        remaining = [index for index in evolved if index not in chosen]
        if not remaining or len(chosen) >= limit:
            break
        chosen.append(min(remaining, key=selector))
    return tuple(chosen)


def _evaluation_config(batch_size: int) -> dict[str, Any]:
    datasets = copy.deepcopy(LOCKED_TASK_DATASETS)
    for task in TASK_ORDER:
        datasets[task]["split"] = "validation"
    return {
        "protocol_id": FORMAL_EVALUATION_PROTOCOL_ID,
        "tasks": list(TASK_ORDER),
        "task_datasets": datasets,
        "num_fewshot": 0,
        "primary_metric": "acc",
        "report_metrics": ["acc", "acc_norm"],
        "batch_size": batch_size,
        "random_seed": 0,
        "numpy_random_seed": 1234,
        "torch_random_seed": 1234,
        "fewshot_random_seed": 1234,
        "log_samples": False,
        "expected_samples": {"arc_easy": 570, "openbookqa": 500},
    }


def _validation_comparison(
    original: Mapping[str, Any],
    reap: Mapping[str, Any],
    winner: Mapping[str, Any],
) -> dict[str, Any]:
    variants = {"original": original, "reap": reap, "winner": winner}
    tasks: dict[str, Any] = {}
    for task in TASK_ORDER:
        tasks[task] = {}
        for metric in ("acc", "acc_norm"):
            values = {
                name: float(artifact["tasks"][task][metric])
                for name, artifact in variants.items()
            }
            tasks[task][metric] = {
                **values,
                "winner_minus_reap": values["winner"] - values["reap"],
                "winner_retention_vs_original": values["winner"] / values["original"],
            }
    means: dict[str, Any] = {}
    for metric in ("acc", "acc_norm"):
        values = {
            name: float(
                np.mean([artifact["tasks"][task][metric] for task in TASK_ORDER])
            )
            for name, artifact in variants.items()
        }
        means[metric] = {
            **values,
            "winner_minus_reap": values["winner"] - values["reap"],
            "winner_retention_vs_original": values["winner"] / values["original"],
        }
    return {"tasks": tasks, "unweighted_mean": means}


def run(config_path: str | Path) -> Path:
    _require_runtime()
    config_path = Path(config_path).resolve()
    config = _load_yaml(config_path)
    _validate_config(config)
    run_dir = _workspace_path(
        str(config["project"]["run_dir"]), kind="run directory", must_exist=False
    )
    if run_dir.exists():
        raise ExperimentError(f"refusing to overwrite existing run: {run_dir}")
    protocol_path = _workspace_path(str(config["project"]["protocol"]), kind="protocol")
    base_path = _workspace_path(str(config["model"]["base_path"]), kind="base model")
    observer_path = _workspace_path(
        str(config["source"]["observer_path"]), kind="observer"
    )
    retained_count = int(config["pruning"]["retained_experts_per_layer"])
    physical_preflight_required = bool(
        config["runtime"].get("physical_preflight_required", True)
    )
    physical_reap_path = None
    if physical_preflight_required:
        physical_reap_path = _workspace_path(
            str(config["model"]["physical_reap_path"]), kind="physical REAP model"
        )

    import torch

    seed = int(config["search"]["seed"])
    _configure_torch(torch, seed=seed)
    run_dir.mkdir(parents=True)
    with (run_dir / "config.resolved.yaml").open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=True)
    write_json(
        run_dir / "run_state.json",
        {"status": "running", "started_at": utc_now(), "training_performed": False},
    )
    write_json(
        run_dir / "provenance.json",
        {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "protocol": {"path": str(protocol_path), "sha256": sha256_file(protocol_path)},
            "script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(__file__)},
        },
    )

    anchors, anchor_evidence = _observer_anchors(
        observer_path, retained_count=retained_count
    )
    write_json(run_dir / "anchors.json", anchor_evidence)
    reap_mask = anchors[0][1]
    tokenizer = _load_tokenizer(base_path)
    print("freezing train search/holdout questions ...", flush=True)
    proxy = _load_proxy_data(config, tokenizer)
    write_json(run_dir / "proxy_data.json", proxy.evidence)

    physical_evidence = None
    physical_preflight = None
    if physical_reap_path is not None:
        print(
            "loading physical REAP checkpoint for mask-equivalence preflight ...",
            flush=True,
        )
        physical_model, physical_evidence = _load_model(
            physical_reap_path, expected_experts=retained_count
        )
        physical_scorer = SerialChoiceScorer(physical_model, proxy)
        physical_preflight = physical_scorer.score("preflight")
        del physical_scorer, physical_model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("loading one original ERNIE model for teacher and EA search ...", flush=True)
    model, base_evidence = _load_model(base_path, expected_experts=64)
    parameter_accounting = _physical_parameter_accounting(
        model, retained_count=retained_count
    )
    identity_before = tuple(
        (name, int(parameter.data_ptr()), int(parameter._version))
        for name, parameter in model.named_parameters()
    )
    identity_mask = np.ones((27, 64), dtype=np.bool_)
    controller = ErnieExpertMaskController(model, identity_mask)
    scorer = SerialChoiceScorer(model, proxy)
    controller.apply(reap_mask)
    logical_preflight = scorer.score("preflight")
    if physical_preflight is not None:
        maximum_delta = float(
            np.max(
                np.abs(
                    physical_preflight.choice_log_likelihoods
                    - logical_preflight.choice_log_likelihoods
                )
            )
        )
        prediction_match = _question_predictions(
            proxy.questions["preflight"], physical_preflight.choice_log_likelihoods
        ) == _question_predictions(
            proxy.questions["preflight"], logical_preflight.choice_log_likelihoods
        )
        tolerance = float(config["runtime"]["logical_physical_choice_ll_tolerance"])
        preflight = {
            "mode": "logical_vs_physical_checkpoint",
            "passed": bool(maximum_delta <= tolerance and prediction_match),
            "questions": len(proxy.questions["preflight"]),
            "choice_requests": len(proxy.requests["preflight"]),
            "choice_ll_max_absolute_delta": maximum_delta,
            "choice_predictions_identical": prediction_match,
            "tolerance": tolerance,
            "physical_cost": {
                "wall_seconds": physical_preflight.wall_seconds,
                "gpu_seconds": physical_preflight.gpu_seconds,
            },
            "logical_cost": {
                "wall_seconds": logical_preflight.wall_seconds,
                "gpu_seconds": logical_preflight.gpu_seconds,
            },
        }
    else:
        finite = bool(np.isfinite(logical_preflight.choice_log_likelihoods).all())
        reference_value = config["runtime"].get("implementation_validation_reference")
        reference: dict[str, Any] | None = None
        if reference_value:
            reference_path = _workspace_path(
                str(reference_value), kind="implementation validation reference"
            )
            reference = {
                "path": str(reference_path),
                "sha256": sha256_file(reference_path),
                "artifact": read_json(reference_path),
            }
        preflight = {
            "mode": "logical_compact_runtime_smoke",
            "passed": finite,
            "retained_experts_per_layer": retained_count,
            "row_counts_valid": bool(
                np.all(np.count_nonzero(reap_mask, axis=1) == retained_count)
            ),
            "choice_log_likelihoods_finite": finite,
            "questions": len(proxy.questions["preflight"]),
            "choice_requests": len(proxy.requests["preflight"]),
            "logical_cost": {
                "wall_seconds": logical_preflight.wall_seconds,
                "gpu_seconds": logical_preflight.gpu_seconds,
            },
            "physical_checkpoint_comparison_run": False,
            "implementation_validation_reference": reference,
        }
    write_json(run_dir / "mask_runtime_preflight.json", preflight)
    if not preflight["passed"]:
        raise ExperimentError(f"REAP mask runtime preflight failed: {preflight}")
    print(f"mask runtime preflight PASS ({preflight['mode']})", flush=True)

    print("caching Original teacher distributions ...", flush=True)
    controller.apply(identity_mask)
    teacher_search_score = scorer.score("search")
    teacher_holdout_score = scorer.score("holdout")
    np.savez_compressed(
        run_dir / "teacher_choice_log_likelihoods.npz",
        search=teacher_search_score.choice_log_likelihoods,
        holdout=teacher_holdout_score.choice_log_likelihoods,
    )

    initial, initial_contexts = _initial_population(
        anchors, config, retained_count=retained_count
    )
    np.save(run_dir / "initial_population.npy", initial, allow_pickle=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    search_started = perf_counter()
    print("starting 48-candidate task-aware NSGA-II ...", flush=True)
    result = _run_nsga2(
        scorer,
        controller,
        proxy,
        teacher_search_score.choice_log_likelihoods,
        initial,
        initial_contexts,
        config,
        retained_count=retained_count,
    )
    search_wall = perf_counter() - search_started
    search_peak = int(torch.cuda.max_memory_allocated())
    np.save(run_dir / "candidate_masks.npy", result.masks, allow_pickle=False)
    np.save(run_dir / "objectives.npy", result.objectives, allow_pickle=False)
    write_jsonl(run_dir / "trials.jsonl", result.trials)

    shortlist = _shortlist(result, limit=int(config["rerank"]["evolved_candidates"]))
    holdout_ids = (0,) + shortlist
    holdout_rows: list[dict[str, Any]] = []
    print(f"reranking candidates {list(holdout_ids)} on disjoint train holdout ...", flush=True)
    for candidate_id in holdout_ids:
        controller.apply(result.masks[candidate_id])
        raw = scorer.score("holdout")
        objective, metrics = _task_metrics(
            proxy,
            "holdout",
            raw,
            teacher_holdout_score.choice_log_likelihoods,
            config,
        )
        holdout_rows.append(
            {
                "candidate_id": int(candidate_id),
                "mask_sha256": mask_sha256(
                    result.masks[candidate_id], retained_count=retained_count
                ),
                "objectives": {
                    TASK_ORDER[0]: float(objective[0]),
                    TASK_ORDER[1]: float(objective[1]),
                },
                "metrics": metrics,
                "cost": {"wall_seconds": raw.wall_seconds, "gpu_seconds": raw.gpu_seconds},
            }
        )
    by_id = {int(row["candidate_id"]): row for row in holdout_rows}
    reap_objectives = np.asarray(
        [by_id[0]["objectives"][task] for task in TASK_ORDER], dtype=np.float64
    )
    eligible: list[tuple[tuple[float, float, float, int], int]] = []
    for candidate_id in shortlist:
        row = by_id[candidate_id]
        values = np.asarray([row["objectives"][task] for task in TASK_ORDER])
        ratios = values / reap_objectives
        passed = bool(np.all(values <= reap_objectives))
        row["relative_objectives_vs_reap"] = {
            task: float(ratio) for task, ratio in zip(TASK_ORDER, ratios, strict=True)
        }
        row["both_task_losses_no_worse_than_reap"] = passed
        if passed:
            task_acc = [float(row["metrics"]["tasks"][task]["acc"]) for task in TASK_ORDER]
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
    by_id[0]["relative_objectives_vs_reap"] = {task: 1.0 for task in TASK_ORDER}
    by_id[0]["both_task_losses_no_worse_than_reap"] = True
    if eligible:
        eligible.sort()
        winner_id = eligible[0][1]
        freeze_reason = "evolved_candidate_no_worse_on_both_holdout_task_losses"
    else:
        winner_id = 0
        freeze_reason = "safe_reap_fallback_no_evolved_candidate_passed_both_holdout_losses"
    winner_mask = result.masks[winner_id]
    frozen = {
        "winner_candidate_id": int(winner_id),
        "winner_is_direct_reap": winner_id == 0,
        "winner_mask_sha256": mask_sha256(
            winner_mask, retained_count=retained_count
        ),
        "freeze_reason": freeze_reason,
        "search_shortlist": list(shortlist),
        "holdout_candidate_ids": list(holdout_ids),
        "validation_consulted_before_freeze": False,
        "test_used": False,
        "frozen_at": utc_now(),
    }
    write_json(run_dir / "holdout.json", {"candidates": holdout_rows})
    write_json(run_dir / "frozen_selection.json", frozen)
    write_json(
        run_dir / "winner_selection.json",
        {
            "candidate_id": int(winner_id),
            "mask_sha256": frozen["winner_mask_sha256"],
            "retained_experts_by_decoder_layer": {
                str(row + 1): np.flatnonzero(winner_mask[row]).astype(int).tolist()
                for row in range(27)
            },
        },
    )
    print(
        f"winner frozen: candidate {winner_id} "
        f"({'REAP fallback' if winner_id == 0 else 'evolved'})",
        flush=True,
    )

    original_path = _workspace_path(
        str(config["source"]["original_validation"]), kind="original validation"
    )
    original = read_json(original_path)
    validation_dir = run_dir / "validation"
    validation_dir.mkdir()
    shutil.copyfile(original_path, validation_dir / "original.json")
    reap_path = validation_dir / "reap.json"
    reap_validation_value = config["source"].get("reap_validation")
    if reap_validation_value:
        reap_validation_path = _workspace_path(
            str(reap_validation_value), kind="REAP validation"
        )
        shutil.copyfile(reap_validation_path, reap_path)
        reap_validation = read_json(reap_path)
        reap_validation_executed = False
    else:
        controller.apply(reap_mask)
        print("evaluating direct REAP mask on full task validation ...", flush=True)
        reap_validation = evaluate_zero_shot(
            model,
            tokenizer,
            _evaluation_config(int(config["runtime"]["final_validation_batch_size"])),
            output_path=reap_path,
            method=f"ernie21b_reap_logical_{retained_count}of64",
            seed=seed,
        )
        reap_validation_executed = True
    winner_path = validation_dir / "winner.json"
    if winner_id == 0:
        shutil.copyfile(reap_path, winner_path)
        winner_validation = read_json(winner_path)
        validation_executed = False
    else:
        controller.apply(winner_mask)
        print("evaluating frozen evolved winner on full task validation ...", flush=True)
        winner_validation = evaluate_zero_shot(
            model,
            tokenizer,
            _evaluation_config(int(config["runtime"]["final_validation_batch_size"])),
            output_path=winner_path,
            method=f"ernie21b_task_aware_nsga2_logical_{retained_count}of64",
            seed=seed,
        )
        validation_executed = True
    comparison = _validation_comparison(original, reap_validation, winner_validation)
    write_json(validation_dir / "comparison.json", comparison)

    identity_after = tuple(
        (name, int(parameter.data_ptr()), int(parameter._version))
        for name, parameter in model.named_parameters()
    )
    if identity_after != identity_before:
        raise ExperimentError("model parameters changed during mask search")
    fronts = nondominated_sort(
        result.objectives,
        candidate_ids=np.arange(result.objectives.shape[0], dtype=np.int64),
    )
    analysis = {
        "schema_version": "ernie-task-aware-nsga2-quick-v2",
        "model": config["model"]["id"],
        "algorithm": "task-aware fixed-cardinality NSGA-II",
        "seed": seed,
        "training_performed": False,
        "weight_updates": 0,
        "single_visible_gpu": True,
        "single_model_replica_during_search": True,
        "task_supervised_mask_selection": True,
        "validation_used_for_selection": False,
        "test_used": False,
        "mask_shape": [27, 64],
        "retained_experts_per_layer": retained_count,
        "same_structural_size_as_direct_reap_baseline": True,
        "physical_parameter_accounting": parameter_accounting,
        "physical_reap_model": physical_evidence,
        "base_model": base_evidence,
        "mask_runtime_preflight": preflight,
        "search": {
            "mu": int(config["search"]["population_size"]),
            "lambda": int(config["search"]["offspring_size"]),
            "generations": int(config["search"]["generations"]),
            "unique_evaluations": int(result.masks.shape[0]),
            "proposals": result.proposals,
            "duplicate_proposals": result.duplicates,
            "wall_seconds": search_wall,
            "peak_device_memory_bytes": search_peak,
            "pareto_fronts": [list(front) for front in fronts],
            "final_population_ids": list(result.final_population_ids),
            "shortlist_ids": list(shortlist),
            "reap_search_objectives": {
                TASK_ORDER[0]: float(result.objectives[0, 0]),
                TASK_ORDER[1]: float(result.objectives[0, 1]),
            },
        },
        "holdout": {"reap": by_id[0], "winner": by_id[winner_id]},
        "winner_freeze": frozen,
        "full_validation_executed_for_direct_reap": reap_validation_executed,
        "full_validation_executed_for_evolved_winner": validation_executed,
        "validation": comparison,
        "model_parameter_identity_unchanged": True,
        "environment": environment_snapshot(),
        "limitations": [
            "single seed and 48-candidate quick search",
            "logical masks keep the full base checkpoint resident during search/evaluation",
            "the frozen winner was not exported as a physical checkpoint",
            "WikiText perplexity guard omitted in this quick run",
        ],
    }
    write_json(run_dir / "analysis.json", analysis)
    controller.close()
    write_json(
        run_dir / "run_state.json",
        {
            "status": "complete",
            "started_at": read_json(run_dir / "run_state.json")["started_at"],
            "completed_at": utc_now(),
            "training_performed": False,
            "winner_candidate_id": int(winner_id),
        },
    )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/ernie21b_reap50_taskaware_nsga2_quick.yaml",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = run(args.config)
    print(f"completed ERNIE task-aware NSGA-II run: {output}", flush=True)


if __name__ == "__main__":
    main()

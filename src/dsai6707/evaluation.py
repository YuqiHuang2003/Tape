"""Pinned lm-evaluation-harness integration and result normalization."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from .config import (
    FORMAL_EVALUATION_PROTOCOL_ID,
    SUPPLEMENTAL_EVALUATION_PROTOCOL_ID,
)
from .io import environment_snapshot, write_json

_CONTENT_HASH_SPEC = "sha256(index<TAB>canonical-json<LF>)"


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, Mapping):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [_json_safe(item) for item in value]
        if hasattr(value, "item"):
            return value.item()
        return str(value)


def _metric(task_result: Mapping[str, Any], name: str) -> float | None:
    for key in (name, f"{name},none"):
        if key in task_result:
            return float(task_result[key])
    return None


def _integer_sample_count(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("sample count must be an integer")
    number = float(value)
    integer = int(value)
    if not math.isfinite(number) or number != integer:
        raise ValueError("sample count must be a finite integer")
    return integer


def _sample_counts(raw: Mapping[str, Any], task: str) -> tuple[int, int]:
    counts = raw.get("n-samples", {})
    if not isinstance(counts, Mapping) or task not in counts:
        raise RuntimeError(f"lm-eval did not report sample counts for {task}")
    value = counts[task]
    try:
        if isinstance(value, Mapping):
            original = _integer_sample_count(value.get("original", value.get("effective", -1)))
            effective = _integer_sample_count(value.get("effective", original))
        else:
            original = effective = _integer_sample_count(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"lm-eval returned non-integer sample counts for {task}: {value}"
        ) from exc
    if original <= 0 or effective <= 0:
        raise RuntimeError(f"lm-eval returned invalid sample counts for {task}: {value}")
    return original, effective


def validate_raw_evaluation_evidence(
    raw: Mapping[str, Any],
    task_names: Sequence[str],
    normalized_tasks: Mapping[str, Any],
    normalized_counts: Mapping[str, Any],
    *,
    metric_names: Sequence[str] = ("acc", "acc_norm"),
) -> None:
    """Require normalized metrics/counts to agree with lm-eval's raw result."""

    raw_results = raw.get("results")
    raw_counts = raw.get("n-samples")
    expected_tasks = set(task_names)
    if (
        not isinstance(raw_results, Mapping)
        or set(raw_results) != expected_tasks
        or not isinstance(raw_counts, Mapping)
        or set(raw_counts) != expected_tasks
    ):
        raise ValueError("raw lm-eval results/counts do not match the requested tasks")
    for task in task_names:
        task_result = raw_results.get(task)
        normalized_result = normalized_tasks.get(task)
        stated_counts = normalized_counts.get(task)
        if not isinstance(task_result, Mapping) or not isinstance(normalized_result, Mapping):
            raise ValueError(f"raw lm-eval metrics are missing for {task}")
        for metric in metric_names:
            try:
                raw_value = _metric(task_result, metric)
                normalized_value = float(normalized_result.get(metric))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"raw or normalized {task}.{metric} is invalid") from exc
            if (
                raw_value is None
                or not math.isfinite(raw_value)
                or not 0.0 <= raw_value <= 1.0
                or not math.isfinite(normalized_value)
                or raw_value != normalized_value
            ):
                raise ValueError(f"normalized {task}.{metric} disagrees with raw lm-eval")
        try:
            original, effective = _sample_counts(raw, task)
        except RuntimeError as exc:
            raise ValueError(f"raw lm-eval sample counts are invalid for {task}") from exc
        try:
            stated_original = _integer_sample_count(stated_counts.get("original", -1))
            stated_effective = _integer_sample_count(stated_counts.get("effective", -1))
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"normalized {task} sample counts are invalid") from exc
        if (
            not isinstance(stated_counts, Mapping)
            or stated_original != original
            or stated_effective != effective
        ):
            raise ValueError(f"normalized {task} sample counts disagree with raw lm-eval")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _dataset_content_evidence(
    task_object: Any,
    *,
    task_name: str,
    dataset_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fingerprint the exact ordered documents exposed to lm-eval for a task."""

    split = str(dataset_config["split"])
    dataset_name = dataset_config["name"]
    if dataset_name is not None:
        dataset_name = str(dataset_name)
    expected = {
        "task": task_name,
        "dataset_path": str(dataset_config["path"]),
        "dataset_name": dataset_name,
        f"{split}_split": split,
    }
    if "output_type" in dataset_config:
        expected["output_type"] = str(dataset_config["output_type"])
    for config_key, expected_value in expected.items():
        actual = task_object.get_config(config_key)
        if actual != expected_value:
            raise RuntimeError(
                f"lm-eval {task_name} resolved {config_key}={actual!r}; "
                f"expected {expected_value!r} from the locked protocol"
            )
    dataset_kwargs = task_object.get_config("dataset_kwargs")
    revision = str(dataset_config["revision"])
    if not isinstance(dataset_kwargs, Mapping) or dataset_kwargs.get("revision") != revision:
        raise RuntimeError(f"lm-eval {task_name} did not load the pinned dataset revision")

    documents = task_object.eval_docs
    fingerprint = getattr(documents, "_fingerprint", None)
    if not isinstance(fingerprint, str) or not fingerprint:
        raise RuntimeError(f"lm-eval {task_name} dataset has no Hugging Face fingerprint")
    content_digest = hashlib.sha256()
    first_row_sha256: str | None = None
    last_row_sha256: str | None = None
    row_count = 0
    for index, document in enumerate(documents):
        row = _canonical_json(document)
        row_digest = hashlib.sha256(row).hexdigest()
        if first_row_sha256 is None:
            first_row_sha256 = row_digest
        last_row_sha256 = row_digest
        content_digest.update(str(index).encode("ascii"))
        content_digest.update(b"\t")
        content_digest.update(row)
        content_digest.update(b"\n")
        row_count += 1
    if row_count == 0:
        raise RuntimeError(f"lm-eval {task_name} resolved an empty evaluation split")

    features = getattr(documents, "features", None)
    if hasattr(features, "to_dict"):
        features = features.to_dict()
    feature_payload = _canonical_json(features)
    return {
        "hub_repository": str(dataset_config["hub_repository"]),
        "lm_eval_task": task_name,
        "dataset_path": expected["dataset_path"],
        "dataset_name": expected["dataset_name"],
        "revision": revision,
        "evaluation_split": split,
        "output_type": task_object.get_config("output_type"),
        "num_rows": row_count,
        "huggingface_fingerprint": fingerprint,
        "content_sha256": content_digest.hexdigest(),
        "content_hash_spec": _CONTENT_HASH_SPEC,
        "first_row_sha256": first_row_sha256,
        "last_row_sha256": last_row_sha256,
        "feature_schema_sha256": hashlib.sha256(feature_payload).hexdigest(),
    }


def _select_task_evaluation_split(
    task_object: Any,
    *,
    task_name: str,
    dataset_config: Mapping[str, Any],
) -> None:
    """Keep the official test split or explicitly retarget evaluation to validation."""

    requested_split = str(dataset_config["split"])
    official_test_split = task_object.get_config("test_split")
    if requested_split == official_test_split:
        return

    validation_split = task_object.get_config("validation_split")
    if not isinstance(validation_split, str) or requested_split != validation_split:
        raise RuntimeError(
            f"lm-eval {task_name} has unsupported evaluation split {requested_split!r}; "
            f"official test_split={official_test_split!r}, "
            f"validation_split={validation_split!r}"
        )

    internal_config = getattr(task_object, "_config", None)
    try:
        if isinstance(internal_config, MutableMapping):
            internal_config["test_split"] = validation_split
        elif internal_config is not None:
            internal_config.test_split = validation_split
        else:
            raise AttributeError("task has no mutable _config")
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            f"lm-eval {task_name} cannot retarget test_split to validation"
        ) from exc
    if task_object.get_config("test_split") != validation_split:
        raise RuntimeError(
            f"lm-eval {task_name} did not retarget test_split to validation"
        )


def _pinned_task_objects(
    task_names: list[str],
    evaluation_config: Mapping[str, Any],
    *,
    task_manager: Any,
    get_task_dict: Any,
) -> tuple[list[Any], dict[str, dict[str, Any]]]:
    """Resolve official tasks while overriding only datasets.load_dataset revision."""

    dataset_configs = evaluation_config["task_datasets"]
    task_specs = [
        {
            "task": task_name,
            "dataset_kwargs": {"revision": str(dataset_configs[task_name]["revision"])},
        }
        for task_name in task_names
    ]
    resolved = get_task_dict(task_specs, task_manager)
    if set(resolved) != set(task_names):
        raise RuntimeError(
            f"lm-eval resolved unexpected tasks: expected {task_names}, got {list(resolved)}"
        )
    objects = [resolved[task_name] for task_name in task_names]
    for task_name, task_object in zip(task_names, objects, strict=False):
        _select_task_evaluation_split(
            task_object,
            task_name=task_name,
            dataset_config=dataset_configs[task_name],
        )
    evidence = {
        task_name: _dataset_content_evidence(
            task_object,
            task_name=task_name,
            dataset_config=dataset_configs[task_name],
        )
        for task_name, task_object in zip(task_names, objects, strict=False)
    }
    return objects, evidence


def _verify_raw_task_revisions(
    raw: Mapping[str, Any],
    task_names: list[str],
    evaluation_config: Mapping[str, Any],
) -> None:
    configs = raw.get("configs")
    if not isinstance(configs, Mapping):
        raise RuntimeError("lm-eval did not return its resolved task configurations")
    for task_name in task_names:
        task_config = configs.get(task_name)
        if not isinstance(task_config, Mapping):
            raise RuntimeError(f"lm-eval did not return the resolved config for {task_name}")
        kwargs = task_config.get("dataset_kwargs")
        expected_revision = str(evaluation_config["task_datasets"][task_name]["revision"])
        if not isinstance(kwargs, Mapping) or kwargs.get("revision") != expected_revision:
            raise RuntimeError(
                f"lm-eval result config does not prove the pinned revision for {task_name}"
            )


def evaluate_zero_shot(
    model: Any,
    tokenizer: Any,
    evaluation_config: Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
    method: str,
    seed: int | None,
    limit: int | float | None = None,
) -> dict[str, Any]:
    """Evaluate one locked zero-shot protocol through lm-eval's native tasks."""

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import TaskManager, get_task_dict
    except ImportError as exc:
        raise RuntimeError("lm-eval==0.4.8 is required for downstream evaluation") from exc
    try:
        lm_eval_version = importlib.metadata.version("lm-eval")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("cannot verify the installed lm-eval distribution") from exc
    if lm_eval_version != "0.4.8":
        raise RuntimeError(
            f"this result protocol is locked to lm-eval==0.4.8, got {lm_eval_version}"
        )

    protocol_id = str(evaluation_config.get("protocol_id", FORMAL_EVALUATION_PROTOCOL_ID))
    expected_protocol_tasks = {
        FORMAL_EVALUATION_PROTOCOL_ID: ["arc_easy", "openbookqa"],
        SUPPLEMENTAL_EVALUATION_PROTOCOL_ID: ["winogrande", "piqa"],
    }
    if protocol_id not in expected_protocol_tasks:
        raise ValueError(f"unknown evaluation protocol: {protocol_id}")
    tasks = list(evaluation_config["tasks"])
    if tasks != expected_protocol_tasks[protocol_id]:
        raise ValueError("task order differs from the locked evaluation protocol")
    default_metrics = (
        ["acc", "acc_norm"] if protocol_id == FORMAL_EVALUATION_PROTOCOL_ID else ["acc"]
    )
    report_metrics = list(evaluation_config.get("report_metrics", default_metrics))
    if not report_metrics or len(set(report_metrics)) != len(report_metrics):
        raise ValueError("evaluation report_metrics must be a non-empty unique list")
    primary_metric = str(evaluation_config.get("primary_metric", "acc"))
    if primary_metric not in report_metrics:
        raise ValueError("the primary metric must be included in report_metrics")
    task_manager = TaskManager()
    task_objects, dataset_evidence = _pinned_task_objects(
        tasks,
        evaluation_config,
        task_manager=task_manager,
        get_task_dict=get_task_dict,
    )
    device = next(model.parameters()).device
    gpu_start = None
    gpu_end = None
    if device.type == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record()
    wall_started = perf_counter()
    wrapped = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=evaluation_config.get("batch_size", 1),
        device=str(device),
    )
    raw = lm_eval.simple_evaluate(
        model=wrapped,
        tasks=task_objects,
        task_manager=task_manager,
        num_fewshot=0,
        limit=limit,
        random_seed=int(evaluation_config.get("random_seed", 0)),
        numpy_random_seed=int(evaluation_config.get("numpy_random_seed", 1234)),
        torch_random_seed=int(evaluation_config.get("torch_random_seed", 1234)),
        fewshot_random_seed=int(evaluation_config.get("fewshot_random_seed", 1234)),
        log_samples=bool(evaluation_config.get("log_samples", False)),
    )
    _verify_raw_task_revisions(raw, tasks, evaluation_config)
    gpu_seconds = None
    peak_device_memory_bytes = None
    if device.type == "cuda":
        gpu_end.record()
        torch.cuda.synchronize(device)
        gpu_seconds = float(gpu_start.elapsed_time(gpu_end) / 1000.0)
        peak_device_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    wall_seconds = perf_counter() - wall_started
    normalized: dict[str, dict[str, float]] = {}
    normalized_counts: dict[str, dict[str, int]] = {}
    primary: list[float] = []
    expected_samples = evaluation_config.get("expected_samples", {})
    for task in tasks:
        task_result = raw["results"][task]
        normalized[task] = {}
        for metric in report_metrics:
            value = _metric(task_result, metric)
            if value is None:
                raise RuntimeError(f"lm-eval did not return the locked {metric} metric for {task}")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RuntimeError(f"lm-eval returned an out-of-range {metric} for {task}")
            normalized[task][metric] = value
        original_count, effective_count = _sample_counts(raw, task)
        expected_count = int(expected_samples.get(task, -1))
        if original_count != expected_count:
            raise RuntimeError(
                f"{task} dataset size changed: expected {expected_count}, got {original_count}"
            )
        if limit is None and effective_count != expected_count:
            raise RuntimeError(
                f"formal {task} evaluation used {effective_count}/{expected_count} samples"
            )
        normalized_counts[task] = {
            "expected": expected_count,
            "original": original_count,
            "effective": effective_count,
        }
        primary.append(normalized[task][primary_metric])
    validate_raw_evaluation_evidence(
        raw,
        tasks,
        normalized,
        normalized_counts,
        metric_names=report_metrics,
    )
    artifact = {
        "schema_version": "downstream-evaluation-v1",
        "is_synthetic": False,
        "method": method,
        "seed": seed,
        "tasks": normalized,
        "mean_primary_acc": sum(primary) / len(primary),
        "primary_metric": primary_metric,
        "num_fewshot": 0,
        "limit": limit,
        "dataset_evidence": dataset_evidence,
        "cost": {
            "wall_seconds": wall_seconds,
            "gpu_seconds": gpu_seconds,
            "peak_device_memory_bytes": peak_device_memory_bytes,
            "sample_counts": normalized_counts,
        },
        "environment": environment_snapshot(),
        "raw_lm_eval": _json_safe(raw),
    }
    if "protocol_id" in evaluation_config:
        artifact["evaluation_protocol_id"] = protocol_id
    chance_accuracy = evaluation_config.get("chance_accuracy")
    if chance_accuracy is not None:
        if not isinstance(chance_accuracy, Mapping) or list(chance_accuracy) != tasks:
            raise ValueError("chance_accuracy must follow the locked task order")
        adjusted: dict[str, float] = {}
        normalized_chance: dict[str, float] = {}
        for task in tasks:
            chance = float(chance_accuracy[task])
            if not math.isfinite(chance) or not 0.0 <= chance < 1.0:
                raise ValueError(f"invalid chance accuracy for {task}")
            normalized_chance[task] = chance
            adjusted[task] = (normalized[task][primary_metric] - chance) / (1.0 - chance)
        artifact["chance_accuracy"] = normalized_chance
        artifact["chance_adjusted_primary"] = adjusted
        artifact["mean_chance_adjusted_primary"] = sum(adjusted.values()) / len(adjusted)
    if output_path is not None:
        write_json(output_path, artifact)
    return artifact

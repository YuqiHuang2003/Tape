"""Verified aggregation of the pre-registered search and evaluation results.

The summarizer deliberately reads only canonical (unlimited) evaluation paths.
Every upstream stage manifest is verified before any output is written, so a
changed or partially copied result cannot silently enter the final tables.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .artifacts import ArtifactRef, verify_stage_manifest, write_stage_manifest
from .config import config_digest
from .evaluation import validate_raw_evaluation_evidence
from .io import read_json, sha256_file, write_json

_METHODS = ("original", "frequency", "coordinate", "layer_block")
_STRATEGIES = ("coordinate", "layer_block")


def _finite_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL record in {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL record in {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSONL record in {path}:{line_number}")
            rows.append(row)
    return rows


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _verify_search(
    run_dir: Path,
    *,
    strategy: str,
    seed: int,
    expected_evaluations: int,
    fitness_tokens: int,
    validation_tokens: int,
    warmup_tokens: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Path],
]:
    directory = run_dir / "search" / strategy / f"seed_{seed}"
    manifest_path = directory / "stage_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing search stage manifest: {manifest_path}")
    manifest = verify_stage_manifest(manifest_path, run_dir=run_dir)
    if manifest.get("stage") != "evolutionary_search":
        raise ValueError(f"unexpected search stage in {manifest_path}")
    metadata = manifest.get("metadata", {})
    if metadata.get("strategy") != strategy or int(metadata.get("seed", -1)) != seed:
        raise ValueError(f"search manifest identity mismatch: {manifest_path}")
    if metadata.get("valid_for_main_results") is not True:
        raise ValueError(f"search is not valid for main results: {strategy}/seed_{seed}")
    required_outputs = {
        "summary",
        "trials",
        "best_coefficients",
        "warmup",
        "execution",
        "environment",
    }
    if not required_outputs.issubset(manifest.get("outputs", {})):
        raise ValueError(f"search manifest omits canonical outputs: {manifest_path}")

    summary_path = directory / "summary.json"
    trials_path = directory / "trials.jsonl"
    best_path = directory / "best_coefficients.json"
    warmup_path = directory / "warmup.json"
    execution_path = directory / "execution.json"
    environment_path = directory / "environment.json"
    for name, expected_path in {
        "summary": summary_path,
        "trials": trials_path,
        "best_coefficients": best_path,
        "warmup": warmup_path,
        "execution": execution_path,
        "environment": environment_path,
    }.items():
        recorded = Path(str(manifest["outputs"][name]["path"]))
        resolved = recorded if recorded.is_absolute() else run_dir / recorded
        if resolved.resolve() != expected_path.resolve():
            raise ValueError(f"search manifest has wrong {name} path: {manifest_path}")
    summary = read_json(summary_path)
    if summary.get("schema_version") != "search-summary-v1":
        raise ValueError(f"unsupported search summary: {summary_path}")
    if summary.get("strategy") != strategy or int(summary.get("seed", -1)) != seed:
        raise ValueError(f"search summary identity mismatch: {summary_path}")
    if summary.get("valid_for_main_results") is not True:
        raise ValueError(f"search summary is not valid for main results: {summary_path}")
    if int(summary.get("evaluations", -1)) != expected_evaluations:
        raise ValueError(f"search did not consume the locked evaluation budget: {summary_path}")

    trials = _read_jsonl(trials_path)
    if len(trials) != expected_evaluations:
        raise ValueError(f"trial count does not match the locked budget: {trials_path}")
    observed_fitness: list[float] = []
    observed_vectors: list[list[float]] = []
    for expected_index, trial in enumerate(trials):
        if trial.get("schema_version") != "search-trial-v1":
            raise ValueError(f"unsupported trial schema at {trials_path}:{expected_index + 1}")
        if int(trial.get("evaluation", -1)) != expected_index:
            raise ValueError(f"non-contiguous evaluation index in {trials_path}")
        if trial.get("strategy") != strategy or int(trial.get("seed", -1)) != seed:
            raise ValueError(f"trial identity mismatch in {trials_path}")
        if trial.get("status") != "ok":
            raise ValueError(f"failed candidate present in formal search: {trials_path}")
        if int(trial.get("loss_tokens", -1)) != fitness_tokens:
            raise ValueError(f"trial used the wrong loss-token count: {trials_path}")
        for field in (
            "wall_seconds",
            "fusion_seconds",
            "forward_seconds",
            "gpu_seconds",
            "fusion_gpu_seconds",
            "forward_gpu_seconds",
        ):
            if _finite_float(trial.get(field), field=f"trial {field}") < 0.0:
                raise ValueError(f"trial {field} must be non-negative: {trials_path}")
        if int(trial.get("peak_device_memory_bytes", 0)) <= 0:
            raise ValueError(f"trial is missing peak CUDA memory: {trials_path}")
        fitness_nll = _finite_float(
            trial.get("fitness_nll"), field="trial fitness_nll"
        )
        nll_sum = _finite_float(trial.get("nll_sum"), field="trial nll_sum")
        loss_tokens = int(trial["loss_tokens"])
        if not math.isclose(
            fitness_nll,
            nll_sum / loss_tokens,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"trial NLL disagrees with its token sum: {trials_path}")
        observed_fitness.append(fitness_nll)
        vector = trial.get("vector")
        if not isinstance(vector, list) or not all(math.isfinite(float(v)) for v in vector):
            raise ValueError(f"trial has an invalid coefficient vector: {trials_path}")
        observed_vectors.append(vector)
    best = min(observed_fitness)
    summary_best = _finite_float(summary.get("best_fitness_nll"), field="best_fitness_nll")
    if not math.isclose(best, summary_best, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"search summary best fitness disagrees with trials: {summary_path}")
    _finite_float(summary.get("validation_nll"), field="validation_nll")
    actual_tokens = sum(int(trial["loss_tokens"]) for trial in trials)
    planned_tokens = expected_evaluations * fitness_tokens
    if int(summary.get("planned_calibration_loss_tokens", -1)) != planned_tokens:
        raise ValueError(f"planned loss-token total is missing or wrong: {summary_path}")
    if int(summary.get("actual_calibration_loss_tokens", -1)) != actual_tokens:
        raise ValueError(f"actual loss-token total disagrees with trials: {summary_path}")
    if int(summary.get("failed_evaluations", -1)) != 0:
        raise ValueError(f"formal search records failed evaluations: {summary_path}")
    validation = summary.get("validation")
    if not isinstance(validation, dict) or validation.get("status") != "ok":
        raise ValueError(f"validation cost record is missing: {summary_path}")
    if int(validation.get("loss_tokens", -1)) != validation_tokens:
        raise ValueError(f"validation used the wrong loss-token count: {summary_path}")
    if not math.isclose(
        _finite_float(validation.get("nll"), field="validation.nll"),
        _finite_float(summary.get("validation_nll"), field="validation_nll"),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(f"validation NLL fields disagree: {summary_path}")
    validation_nll = _finite_float(validation.get("nll"), field="validation.nll")
    validation_nll_sum = _finite_float(
        validation.get("nll_sum"), field="validation.nll_sum"
    )
    if not math.isclose(
        validation_nll,
        validation_nll_sum / validation_tokens,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"validation NLL disagrees with its token sum: {summary_path}"
        )
    for field in (
        "wall_seconds",
        "fusion_seconds",
        "forward_seconds",
        "gpu_seconds",
        "fusion_gpu_seconds",
        "forward_gpu_seconds",
    ):
        if _finite_float(validation.get(field), field=f"validation.{field}") < 0.0:
            raise ValueError(f"validation {field} must be non-negative: {summary_path}")
    if int(validation.get("peak_device_memory_bytes", 0)) <= 0:
        raise ValueError(f"validation peak CUDA memory is missing: {summary_path}")

    best_index = min(
        range(len(observed_fitness)), key=lambda index: (observed_fitness[index], index)
    )
    if int(summary.get("best_evaluation", -1)) != best_index:
        raise ValueError(f"search summary best evaluation disagrees with trials: {summary_path}")
    if summary.get("best_vector") != observed_vectors[best_index]:
        raise ValueError(f"search summary best vector disagrees with trials: {summary_path}")
    if int(summary.get("validation_loss_tokens", -1)) != validation_tokens:
        raise ValueError(f"top-level validation token count is wrong: {summary_path}")
    best_artifact = read_json(best_path)
    if (
        best_artifact.get("schema_version") != "coefficients-v1"
        or best_artifact.get("method") != strategy
        or int(best_artifact.get("seed", -1)) != seed
        or best_artifact.get("vector") != observed_vectors[best_index]
        or not math.isclose(
            _finite_float(best_artifact.get("fitness_nll"), field="best artifact fitness"),
            observed_fitness[best_index],
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError(f"best coefficient artifact disagrees with trials: {best_path}")

    warmup = read_json(warmup_path)
    if (
        warmup.get("schema_version") != "search-warmup-log-v1"
        or warmup.get("strategy") != strategy
        or int(warmup.get("seed", -1)) != seed
        or int(warmup.get("expected_loss_tokens", -1)) != warmup_tokens
        or int(warmup.get("budget_evaluations_consumed", -1)) != 0
    ):
        raise ValueError(f"search warmup log disagrees with the protocol: {warmup_path}")
    warmup_sessions = warmup.get("sessions")
    if not isinstance(warmup_sessions, list) or not warmup_sessions:
        raise ValueError(f"search warmup log has no sessions: {warmup_path}")
    if int(metadata.get("warmup_sessions", -1)) != len(warmup_sessions):
        raise ValueError(f"search manifest warmup count is inconsistent: {manifest_path}")
    warmup_wall_seconds = 0.0
    warmup_gpu_seconds = 0.0
    warmup_peak = 0
    for session_index, session in enumerate(warmup_sessions):
        if (
            not isinstance(session, dict)
            or session.get("schema_version") != "search-warmup-session-v1"
            or int(session.get("session_index", -1)) != session_index
            or int(session.get("budget_evaluations_consumed", -1)) != 0
            or int(session.get("loss_tokens", -1)) != warmup_tokens
        ):
            raise ValueError(f"invalid search warmup session: {warmup_path}")
        _finite_float(session.get("fitness_nll"), field="warmup fitness_nll")
        wall = _finite_float(session.get("wall_seconds"), field="warmup wall_seconds")
        gpu = _finite_float(session.get("gpu_seconds"), field="warmup gpu_seconds")
        peak = int(session.get("peak_device_memory_bytes", 0))
        details = session.get("details")
        if wall < 0.0 or gpu < 0.0 or peak <= 0:
            raise ValueError(f"search warmup has invalid CUDA cost: {warmup_path}")
        if (
            not isinstance(details, dict)
            or details.get("status") != "ok"
            or int(details.get("loss_tokens", -1)) != warmup_tokens
            or not math.isclose(
                _finite_float(details.get("wall_seconds"), field="warmup details wall"),
                wall,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                _finite_float(details.get("gpu_seconds"), field="warmup details GPU"),
                gpu,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError(f"search warmup details are inconsistent: {warmup_path}")
        warmup_wall_seconds += wall
        warmup_gpu_seconds += gpu
        warmup_peak = max(warmup_peak, peak)

    execution = read_json(execution_path)
    if (
        execution.get("schema_version") != "ea-recovery-v1"
        or execution.get("strategy") != strategy
        or int(execution.get("seed", -1)) != seed
        or int(execution.get("evaluations", -1)) != expected_evaluations
        or execution.get("checkpoint_schema_version") != "ea-resume-state-v1"
        or execution.get("checkpoint_granularity")
        != "each_candidate_and_replacement_boundary"
        or execution.get("temporary_checkpoint_removed_after_completion") is not True
    ):
        raise ValueError(f"search recovery record is invalid: {execution_path}")
    for field in (
        "resume_count",
        "checkpoint_writes_this_invocation",
        "checkpoint_writes_total",
    ):
        if int(execution.get(field, -1)) < 0:
            raise ValueError(f"search recovery record has invalid {field}: {execution_path}")
    resumed_at = execution.get("resumed_at_evaluation")
    if resumed_at is not None and not 0 <= int(resumed_at) < expected_evaluations:
        raise ValueError(f"search recovery point is outside the budget: {execution_path}")
    if metadata.get("recovery") != execution:
        raise ValueError(f"search manifest recovery record is inconsistent: {manifest_path}")

    environment = read_json(environment_path)
    fingerprint = str(environment.get("numerical_environment_sha256", ""))
    invalid_digest = any(character not in "0123456789abcdef" for character in fingerprint)
    environment_sessions = environment.get("sessions")
    if (
        environment.get("schema_version") != "search-environment-log-v1"
        or environment.get("strategy") != strategy
        or int(environment.get("seed", -1)) != seed
        or len(fingerprint) != 64
        or invalid_digest
        or not isinstance(environment_sessions, list)
        or len(environment_sessions) != len(warmup_sessions)
        or int(metadata.get("environment_sessions", -1)) != len(environment_sessions)
        or metadata.get("numerical_environment_sha256") != fingerprint
    ):
        raise ValueError(f"search numerical environment log is invalid: {environment_path}")
    for session_index, session in enumerate(environment_sessions):
        snapshot = session.get("snapshot") if isinstance(session, dict) else None
        accelerator = snapshot.get("accelerator") if isinstance(snapshot, dict) else None
        packages = snapshot.get("packages") if isinstance(snapshot, dict) else None
        cublas = accelerator.get("cublas_workspace_config") if isinstance(
            accelerator, dict
        ) else None
        if (
            not isinstance(session, dict)
            or int(session.get("session_index", -1)) != session_index
            or session.get("numerical_environment_sha256") != fingerprint
            or not isinstance(accelerator, dict)
            or accelerator.get("cuda_available") is not True
            or accelerator.get("deterministic_algorithms") is not True
            or accelerator.get("cudnn_deterministic") is not True
            or not isinstance(accelerator.get("devices"), list)
            or len(accelerator["devices"]) != 1
            or not isinstance(cublas, dict)
            or cublas.get("valid") is not True
            or cublas.get("verified_before_cuda_initialization_in_process") is not True
            or not isinstance(packages, dict)
            or packages.get("transformers") != "4.46.3"
        ):
            raise ValueError(f"search CUDA environment is not reproducible: {environment_path}")

    audit = {
        "warmup_sessions": len(warmup_sessions),
        "warmup_loss_tokens": len(warmup_sessions) * warmup_tokens,
        "warmup_wall_seconds": warmup_wall_seconds,
        "warmup_gpu_seconds": warmup_gpu_seconds,
        "warmup_peak_device_memory_bytes": warmup_peak,
        "resumed_from_checkpoint": bool(execution.get("resumed_from_checkpoint")),
        "resume_count": int(execution["resume_count"]),
        "checkpoint_writes_total": int(execution["checkpoint_writes_total"]),
        "numerical_environment_sha256": fingerprint,
    }
    return summary, trials, audit, {
        "summary": summary_path,
        "trials": trials_path,
        "best_coefficients": best_path,
        "warmup": warmup_path,
        "execution": execution_path,
        "environment": environment_path,
        "stage": manifest_path,
    }


def _evaluation_path(run_dir: Path, method: str, seed: int | None) -> Path:
    suffix = "deterministic" if seed is None else f"seed_{seed}"
    return run_dir / "evaluation" / method / f"{suffix}.json"


def _verify_dataset_evidence_consistency(
    evaluations: Mapping[tuple[str, int | None], Mapping[str, Any]],
    tasks: Sequence[str],
) -> None:
    """Require every compared method to use byte-identical dataset evidence."""

    reference_key = ("original", None)
    reference = evaluations[reference_key]["dataset_evidence"]
    for key, artifact in evaluations.items():
        evidence = artifact["dataset_evidence"]
        for task in tasks:
            if evidence[task] != reference[task]:
                method, seed = key
                raise ValueError(
                    "downstream dataset evidence differs across formal evaluations: "
                    f"{method}/seed_{seed} task {task} disagrees with original"
                )


def _verify_evaluation(
    run_dir: Path,
    *,
    method: str,
    seed: int | None,
    tasks: Sequence[str],
    expected_samples: Mapping[str, int],
    task_datasets: Mapping[str, Mapping[str, Any]],
    strategies: Sequence[str],
    seeds: Sequence[int],
) -> tuple[dict[str, Any], Path, Path]:
    path = _evaluation_path(run_dir, method, seed)
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        debug_root = run_dir / "evaluation" / "debug"
        detail = (
            " (debug/limit outputs cannot replace a formal evaluation)"
            if debug_root.exists()
            else ""
        )
        raise FileNotFoundError(f"missing canonical evaluation: {path}{detail}")
    manifest = verify_stage_manifest(manifest_path, run_dir=run_dir)
    if manifest.get("stage") != "downstream_evaluation":
        raise ValueError(f"unexpected evaluation stage in {manifest_path}")
    metadata = manifest.get("metadata", {})
    if metadata.get("method") != method or metadata.get("seed") != seed:
        raise ValueError(f"evaluation manifest identity mismatch: {manifest_path}")
    if metadata.get("limit") is not None:
        raise ValueError(f"limited/debug evaluation is not a formal result: {manifest_path}")
    inputs = manifest.get("inputs", {})
    required_inputs = {
        "resolved_config",
        *(
            f"locked_search_{strategy}_{seed_value}"
            for strategy in strategies
            for seed_value in seeds
        ),
    }
    if method != "original":
        required_inputs.update(
            {"grouping_stage", "grouping", "coefficient_stage", "coefficients"}
        )
    missing_inputs = sorted(required_inputs - set(inputs))
    if missing_inputs:
        raise ValueError(
            f"evaluation manifest is missing locked inputs {missing_inputs}: {manifest_path}"
        )
    expected_input_paths: dict[str, Path] = {
        "resolved_config": run_dir / "config.resolved.yaml",
        **{
            f"locked_search_{strategy}_{seed_value}": run_dir
            / "search"
            / strategy
            / f"seed_{seed_value}"
            / "stage_manifest.json"
            for strategy in strategies
            for seed_value in seeds
        },
    }
    if method != "original":
        expected_input_paths["grouping_stage"] = (
            run_dir / "grouping" / "stage_manifest.json"
        )
        expected_input_paths["grouping"] = run_dir / "grouping" / "grouping.json"
        if method == "frequency":
            coefficient_directory = run_dir / "baselines" / "frequency"
        else:
            coefficient_directory = run_dir / "search" / method / f"seed_{seed}"
        expected_input_paths["coefficient_stage"] = (
            coefficient_directory / "stage_manifest.json"
        )
        filename = "coefficients.json" if method == "frequency" else "best_coefficients.json"
        expected_input_paths["coefficients"] = coefficient_directory / filename
    for name, expected_path in expected_input_paths.items():
        recorded = Path(str(inputs[name]["path"]))
        actual_path = recorded if recorded.is_absolute() else run_dir / recorded
        if actual_path.resolve() != expected_path.resolve():
            raise ValueError(f"evaluation manifest has wrong {name} path: {manifest_path}")
    evaluation_output = manifest.get("outputs", {}).get("evaluation", {})
    recorded_output = Path(str(evaluation_output.get("path", "")))
    resolved_output = (
        recorded_output if recorded_output.is_absolute() else run_dir / recorded_output
    )
    if resolved_output.resolve() != path.resolve():
        raise ValueError(f"evaluation manifest points to the wrong output: {manifest_path}")

    artifact = read_json(path)
    if artifact.get("schema_version") != "downstream-evaluation-v1":
        raise ValueError(f"unsupported downstream evaluation: {path}")
    if artifact.get("is_synthetic") is not False:
        raise ValueError(f"synthetic evaluation cannot enter formal results: {path}")
    if artifact.get("method") != method or artifact.get("seed") != seed:
        raise ValueError(f"evaluation identity mismatch: {path}")
    if artifact.get("limit") is not None:
        raise ValueError(f"limited/debug evaluation cannot enter formal results: {path}")
    if int(artifact.get("num_fewshot", -1)) != 0 or artifact.get("primary_metric") != "acc":
        raise ValueError(f"evaluation protocol mismatch: {path}")
    task_results = artifact.get("tasks")
    if not isinstance(task_results, dict) or list(task_results) != list(tasks):
        raise ValueError(f"evaluation task order mismatch: {path}")
    primary: list[float] = []
    for task in tasks:
        values = task_results[task]
        if not isinstance(values, dict):
            raise ValueError(f"invalid task result for {task}: {path}")
        for metric in ("acc", "acc_norm"):
            score = _finite_float(values.get(metric), field=f"{task}.{metric}")
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"{task}.{metric} is outside [0, 1]: {path}")
            if metric == "acc":
                primary.append(score)
    stated_mean = _finite_float(artifact.get("mean_primary_acc"), field="mean_primary_acc")
    if not math.isclose(stated_mean, float(np.mean(primary)), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"mean_primary_acc disagrees with task metrics: {path}")
    cost = artifact.get("cost")
    if not isinstance(cost, dict):
        raise ValueError(f"evaluation cost record is missing: {path}")
    if _finite_float(cost.get("wall_seconds"), field="downstream wall_seconds") <= 0.0:
        raise ValueError(f"downstream wall time must be positive: {path}")
    if _finite_float(cost.get("gpu_seconds"), field="downstream gpu_seconds") < 0.0:
        raise ValueError(f"downstream GPU time must be non-negative: {path}")
    if int(cost.get("peak_device_memory_bytes", 0)) <= 0:
        raise ValueError(f"downstream peak CUDA memory is missing: {path}")
    counts = cost.get("sample_counts")
    if not isinstance(counts, dict) or list(counts) != list(tasks):
        raise ValueError(f"downstream sample counts are missing or out of order: {path}")
    for task in tasks:
        expected = int(expected_samples[task])
        task_counts = counts.get(task)
        if not isinstance(task_counts, dict) or any(
            int(task_counts.get(field, -1)) != expected
            for field in ("expected", "original", "effective")
        ):
            raise ValueError(f"formal downstream sample count mismatch for {task}: {path}")
    packages = artifact.get("environment", {}).get("packages", {})
    if packages.get("lm_eval") != "0.4.8":
        raise ValueError(f"formal evaluation was not run with lm-eval==0.4.8: {path}")
    evidence = artifact.get("dataset_evidence")
    raw_lm_eval = artifact.get("raw_lm_eval")
    raw_configs = raw_lm_eval.get("configs") if isinstance(raw_lm_eval, dict) else None
    if not isinstance(evidence, dict) or list(evidence) != list(tasks):
        raise ValueError(f"ordered downstream dataset evidence is missing: {path}")
    if not isinstance(raw_configs, dict):
        raise ValueError(f"resolved lm-eval task configs are missing: {path}")
    for task in tasks:
        locked = task_datasets[task]
        task_evidence = evidence.get(task)
        if not isinstance(task_evidence, dict):
            raise ValueError(f"dataset evidence is missing for {task}: {path}")
        expected_fields = {
            "hub_repository": locked["hub_repository"],
            "dataset_path": locked["path"],
            "dataset_name": locked["name"],
            "revision": locked["revision"],
            "evaluation_split": locked["split"],
            "num_rows": int(expected_samples[task]),
            "content_hash_spec": "sha256(index<TAB>canonical-json<LF>)",
        }
        for field, expected in expected_fields.items():
            if task_evidence.get(field) != expected:
                raise ValueError(f"{task} dataset evidence has wrong {field}: {path}")
        if not str(task_evidence.get("huggingface_fingerprint", "")).strip():
            raise ValueError(f"{task} Hugging Face fingerprint is missing: {path}")
        for field in (
            "content_sha256",
            "first_row_sha256",
            "last_row_sha256",
            "feature_schema_sha256",
        ):
            digest = str(task_evidence.get(field, ""))
            invalid = any(
                character not in "0123456789abcdef" for character in digest
            )
            if len(digest) != 64 or invalid:
                raise ValueError(f"{task} dataset evidence has invalid {field}: {path}")
        raw_task = raw_configs.get(task)
        raw_kwargs = raw_task.get("dataset_kwargs") if isinstance(raw_task, dict) else None
        if not isinstance(raw_kwargs, dict) or raw_kwargs.get("revision") != locked[
            "revision"
        ]:
            raise ValueError(f"resolved lm-eval config has wrong revision for {task}: {path}")
    assert isinstance(raw_lm_eval, dict)
    validate_raw_evaluation_evidence(raw_lm_eval, tasks, task_results, counts)
    return artifact, path, manifest_path


def _verify_compression(
    run_dir: Path,
    *,
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    parameter_dtype: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Verify the shared physical fixed-group checkpoint and its accounting."""

    from .granite import _validate_grouped_checkpoint_manifest

    directory = run_dir / "checkpoints" / "frequency" / "deterministic"
    stage_path = directory / "stage_manifest.json"
    if not stage_path.is_file():
        raise FileNotFoundError(
            f"formal summary requires the frequency checkpoint export: {stage_path}"
        )
    stage = verify_stage_manifest(stage_path, run_dir=run_dir)
    if stage.get("stage") != "checkpoint_export":
        raise ValueError(f"unexpected checkpoint-export stage: {stage_path}")
    metadata = stage.get("metadata", {})
    if (
        metadata.get("method") != "frequency"
        or metadata.get("seed") is not None
        or metadata.get("roundtrip_passed") is not True
    ):
        raise ValueError(f"frequency checkpoint was not round-trip validated: {stage_path}")

    compression_path = directory / "compression.json"
    checkpoint_manifest_path = directory / "manifest.json"
    roundtrip_path = directory / "roundtrip_validation.json"
    model_path = directory / "model.safetensors"
    expected_outputs = {
        "checkpoint_manifest": checkpoint_manifest_path,
        "model": model_path,
        "compression": compression_path,
        "roundtrip": roundtrip_path,
    }
    stage_outputs = stage.get("outputs", {})
    for name, expected_path in expected_outputs.items():
        reference = stage_outputs.get(name)
        if not isinstance(reference, dict):
            raise ValueError(f"checkpoint stage omits {name}: {stage_path}")
        recorded = Path(str(reference.get("path", "")))
        resolved = recorded if recorded.is_absolute() else run_dir / recorded
        if resolved.resolve() != expected_path.resolve():
            raise ValueError(f"checkpoint stage has wrong {name} path: {stage_path}")
    compression = read_json(compression_path)
    checkpoint_manifest = _validate_grouped_checkpoint_manifest(directory)
    roundtrip = read_json(roundtrip_path)
    if compression.get("schema_version") != "compression-v1":
        raise ValueError(f"unsupported compression artifact: {compression_path}")
    if checkpoint_manifest.get("schema_version") != "fused-checkpoint-v1":
        raise ValueError(f"unsupported checkpoint manifest: {checkpoint_manifest_path}")
    if checkpoint_manifest.get("base_model") != {
        "id": model_id,
        "revision": model_revision,
    }:
        raise ValueError(f"checkpoint base-model provenance mismatch: {checkpoint_manifest_path}")
    if checkpoint_manifest.get("tokenizer") != {
        "id": model_id,
        "revision": tokenizer_revision,
    }:
        raise ValueError(f"checkpoint tokenizer provenance mismatch: {checkpoint_manifest_path}")
    if checkpoint_manifest.get("parameter_dtype") != parameter_dtype:
        raise ValueError(f"checkpoint parameter dtype mismatch: {checkpoint_manifest_path}")
    if roundtrip.get("schema_version") != "checkpoint-roundtrip-v1" or roundtrip.get(
        "passed"
    ) is not True:
        raise ValueError(f"checkpoint round-trip validation failed: {roundtrip_path}")

    integer_fields = (
        "original_total_parameters",
        "fused_total_parameters",
        "original_expert_parameters",
        "fused_expert_parameters",
        "standalone_checkpoint_bytes",
    )
    values = {name: int(compression.get(name, 0)) for name in integer_fields}
    if any(value <= 0 for value in values.values()):
        raise ValueError(f"compression counts and sizes must be positive: {compression_path}")
    if not (
        values["fused_total_parameters"] < values["original_total_parameters"]
        and values["fused_expert_parameters"] < values["original_expert_parameters"]
    ):
        raise ValueError(
            f"checkpoint does not physically reduce expert parameters: {compression_path}"
        )
    base = compression.get("base_weight_checkpoint")
    if not isinstance(base, dict) or base.get("revision") != model_revision:
        raise ValueError(f"base checkpoint byte provenance is missing: {compression_path}")
    base_bytes = int(base.get("total_bytes", 0))
    base_files = base.get("files")
    if (
        base_bytes <= 0
        or not isinstance(base_files, list)
        or sum(int(item.get("bytes", 0)) for item in base_files) != base_bytes
    ):
        raise ValueError(f"base checkpoint byte total is invalid: {compression_path}")
    if values["standalone_checkpoint_bytes"] != model_path.stat().st_size:
        raise ValueError(
            f"standalone weight byte total disagrees with the file: {compression_path}"
        )
    if int(compression.get("standalone_weight_file_bytes", -1)) != values[
        "standalone_checkpoint_bytes"
    ]:
        raise ValueError(f"standalone weight byte fields disagree: {compression_path}")
    file_entries = checkpoint_manifest.get("files", {})
    required_payload = {
        "model.safetensors",
        "config.json",
        "grouping.json",
        "coefficients.json",
        "compression.json",
    }
    if not isinstance(file_entries, dict) or not required_payload.issubset(file_entries):
        raise ValueError(f"checkpoint manifest omits required payload: {checkpoint_manifest_path}")
    declared_payload_bytes = 0
    for relative, entry in file_entries.items():
        candidate = directory / relative
        if not candidate.is_file():
            raise FileNotFoundError(f"checkpoint payload file is missing: {candidate}")
        if int(entry.get("bytes", -1)) != candidate.stat().st_size:
            raise ValueError(f"checkpoint payload byte mismatch: {candidate}")
        if entry.get("sha256") != sha256_file(candidate):
            raise ValueError(f"checkpoint payload hash mismatch: {candidate}")
        declared_payload_bytes += candidate.stat().st_size
    model_entry = file_entries.get("model.safetensors", {})
    if int(model_entry.get("bytes", -1)) != values["standalone_checkpoint_bytes"]:
        raise ValueError(
            f"checkpoint manifest byte total disagrees: {checkpoint_manifest_path}"
        )
    if int(checkpoint_manifest.get("model_weight_bytes", -1)) != values[
        "standalone_checkpoint_bytes"
    ]:
        raise ValueError(f"checkpoint model_weight_bytes disagrees: {checkpoint_manifest_path}")
    if int(checkpoint_manifest.get("declared_payload_bytes", -1)) != declared_payload_bytes:
        raise ValueError(f"checkpoint payload byte total disagrees: {checkpoint_manifest_path}")

    expected_fractions = {
        "whole_model_reduction_fraction": 1.0
        - values["fused_total_parameters"] / values["original_total_parameters"],
        "expert_reduction_fraction": 1.0
        - values["fused_expert_parameters"] / values["original_expert_parameters"],
        "weight_checkpoint_reduction_fraction": 1.0
        - values["standalone_checkpoint_bytes"] / base_bytes,
    }
    for field, expected in expected_fractions.items():
        actual = _finite_float(compression.get(field), field=field)
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{field} is arithmetically inconsistent: {compression_path}")
    normalized = {
        **values,
        "base_weight_checkpoint_bytes": base_bytes,
        **expected_fractions,
    }
    return normalized, {
        "checkpoint_export_stage": stage_path,
        "checkpoint_manifest": checkpoint_manifest_path,
        "compression": compression_path,
        "roundtrip": roundtrip_path,
        "checkpoint_model": model_path,
    }


def _metric_names(tasks: Sequence[str]) -> list[str]:
    names = [
        "search_best_nll",
        "validation_nll",
        "warmup_sessions",
        "warmup_loss_tokens",
        "planned_calibration_loss_tokens",
        "actual_calibration_loss_tokens",
        "validation_loss_tokens",
        "total_calibration_loss_tokens",
        "total_compute_loss_tokens",
        "failed_evaluations",
        "warmup_wall_seconds",
        "warmup_gpu_seconds",
        "search_wall_seconds",
        "search_fusion_seconds",
        "search_forward_seconds",
        "search_gpu_seconds",
        "validation_wall_seconds",
        "validation_fusion_seconds",
        "validation_forward_seconds",
        "validation_gpu_seconds",
        "downstream_wall_seconds",
        "downstream_gpu_seconds",
        "total_wall_seconds",
        "total_gpu_seconds",
        "warmup_peak_device_memory_bytes",
        "search_peak_device_memory_bytes",
        "validation_peak_device_memory_bytes",
        "downstream_peak_device_memory_bytes",
        "mean_primary_acc",
        "original_total_parameters",
        "deployed_total_parameters",
        "original_expert_parameters",
        "deployed_expert_parameters",
        "whole_model_reduction_fraction",
        "expert_reduction_fraction",
        "base_weight_checkpoint_bytes",
        "deployed_weight_checkpoint_bytes",
        "weight_checkpoint_reduction_fraction",
    ]
    for task in tasks:
        names.extend((f"{task}_acc", f"{task}_acc_norm"))
    return names


def _long_row(
    method: str,
    seed: int | None,
    evaluation: Mapping[str, Any],
    tasks: Sequence[str],
    compression: Mapping[str, Any],
    summary: Mapping[str, Any] | None = None,
    trials: Sequence[Mapping[str, Any]] | None = None,
    search_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    is_original = method == "original"
    validation = {} if summary is None else summary["validation"]
    downstream_cost = evaluation["cost"]
    search_costs = {
        destination: 0.0
        if trials is None
        else sum(float(trial[source]) for trial in trials)
        for source, destination in (
            ("wall_seconds", "search_wall_seconds"),
            ("fusion_seconds", "search_fusion_seconds"),
            ("forward_seconds", "search_forward_seconds"),
            ("gpu_seconds", "search_gpu_seconds"),
        )
    }
    validation_costs = {
        destination: 0.0 if summary is None else float(validation[source])
        for source, destination in (
            ("wall_seconds", "validation_wall_seconds"),
            ("fusion_seconds", "validation_fusion_seconds"),
            ("forward_seconds", "validation_forward_seconds"),
            ("gpu_seconds", "validation_gpu_seconds"),
        )
    }
    actual_tokens = (
        0 if summary is None else int(summary["actual_calibration_loss_tokens"])
    )
    validation_tokens = 0 if summary is None else int(summary["validation_loss_tokens"])
    warmup_loss_tokens = (
        0 if search_audit is None else int(search_audit["warmup_loss_tokens"])
    )
    warmup_wall_seconds = (
        0.0 if search_audit is None else float(search_audit["warmup_wall_seconds"])
    )
    warmup_gpu_seconds = (
        0.0 if search_audit is None else float(search_audit["warmup_gpu_seconds"])
    )
    row: dict[str, Any] = {
        "method": method,
        "seed": "" if seed is None else seed,
        "run_type": "deterministic" if seed is None else "seeded",
        "compression_basis": "original" if is_original else "shared_fixed_group_export",
        "evaluations": 0 if summary is None else int(summary["evaluations"]),
        "best_evaluation_index": "" if summary is None else int(summary["best_evaluation"]),
        "best_evaluation_count": ""
        if summary is None
        else int(summary["best_evaluation"]) + 1,
        "search_best_nll": "" if summary is None else float(summary["best_fitness_nll"]),
        "validation_nll": "" if summary is None else float(summary["validation_nll"]),
        "warmup_sessions": 0
        if search_audit is None
        else int(search_audit["warmup_sessions"]),
        "warmup_loss_tokens": warmup_loss_tokens,
        "planned_calibration_loss_tokens": 0
        if summary is None
        else int(summary["planned_calibration_loss_tokens"]),
        "actual_calibration_loss_tokens": actual_tokens,
        "validation_loss_tokens": validation_tokens,
        "total_calibration_loss_tokens": actual_tokens + validation_tokens,
        "total_compute_loss_tokens": (
            warmup_loss_tokens + actual_tokens + validation_tokens
        ),
        "failed_evaluations": 0
        if summary is None
        else int(summary["failed_evaluations"]),
        "resumed_from_checkpoint": False
        if search_audit is None
        else bool(search_audit["resumed_from_checkpoint"]),
        "resume_count": 0
        if search_audit is None
        else int(search_audit["resume_count"]),
        "checkpoint_writes_total": 0
        if search_audit is None
        else int(search_audit["checkpoint_writes_total"]),
        "numerical_environment_sha256": ""
        if search_audit is None
        else str(search_audit["numerical_environment_sha256"]),
        "warmup_wall_seconds": warmup_wall_seconds,
        "warmup_gpu_seconds": warmup_gpu_seconds,
        **search_costs,
        **validation_costs,
        "downstream_wall_seconds": float(downstream_cost["wall_seconds"]),
        "downstream_gpu_seconds": float(downstream_cost["gpu_seconds"]),
        "total_wall_seconds": warmup_wall_seconds
        + search_costs["search_wall_seconds"]
        + validation_costs["validation_wall_seconds"]
        + float(downstream_cost["wall_seconds"]),
        "total_gpu_seconds": warmup_gpu_seconds
        + search_costs["search_gpu_seconds"]
        + validation_costs["validation_gpu_seconds"]
        + float(downstream_cost["gpu_seconds"]),
        "warmup_peak_device_memory_bytes": ""
        if search_audit is None
        else int(search_audit["warmup_peak_device_memory_bytes"]),
        "search_peak_device_memory_bytes": ""
        if trials is None
        else max(int(trial["peak_device_memory_bytes"]) for trial in trials),
        "validation_peak_device_memory_bytes": ""
        if summary is None
        else int(validation["peak_device_memory_bytes"]),
        "downstream_peak_device_memory_bytes": int(
            downstream_cost["peak_device_memory_bytes"]
        ),
        "mean_primary_acc": float(evaluation["mean_primary_acc"]),
        "original_total_parameters": int(compression["original_total_parameters"]),
        "deployed_total_parameters": int(
            compression[
                "original_total_parameters" if is_original else "fused_total_parameters"
            ]
        ),
        "original_expert_parameters": int(compression["original_expert_parameters"]),
        "deployed_expert_parameters": int(
            compression[
                "original_expert_parameters" if is_original else "fused_expert_parameters"
            ]
        ),
        "whole_model_reduction_fraction": 0.0
        if is_original
        else float(compression["whole_model_reduction_fraction"]),
        "expert_reduction_fraction": 0.0
        if is_original
        else float(compression["expert_reduction_fraction"]),
        "base_weight_checkpoint_bytes": int(
            compression["base_weight_checkpoint_bytes"]
        ),
        "deployed_weight_checkpoint_bytes": int(
            compression[
                "base_weight_checkpoint_bytes"
                if is_original
                else "standalone_checkpoint_bytes"
            ]
        ),
        "weight_checkpoint_reduction_fraction": 0.0
        if is_original
        else float(compression["weight_checkpoint_reduction_fraction"]),
    }
    for task in tasks:
        for metric in ("acc", "acc_norm"):
            value = evaluation["tasks"][task].get(metric)
            row[f"{task}_{metric}"] = "" if value is None else float(value)
    return row


def _sample_summary(
    long_rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in _METHODS:
        method_rows = [row for row in long_rows if row["method"] == method]
        for metric in metrics:
            values = np.asarray(
                [float(row[metric]) for row in method_rows if row.get(metric) != ""],
                dtype=np.float64,
            )
            if not values.size:
                continue
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(np.mean(values)),
                    "sample_sd": "" if values.size < 2 else float(np.std(values, ddof=1)),
                }
            )
    return rows


def _method_totals(
    long_rows: Sequence[Mapping[str, Any]], metrics: Sequence[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in _METHODS:
        method_rows = [row for row in long_rows if row["method"] == method]
        for metric in metrics:
            values = [float(row[metric]) for row in method_rows if row.get(metric) != ""]
            if values:
                rows.append(
                    {
                        "method": method,
                        "metric": metric,
                        "n": len(values),
                        "total": float(sum(values)),
                    }
                )
    return rows


def _paired_differences(
    long_rows: Sequence[Mapping[str, Any]], seeds: Sequence[int], metrics: Sequence[str]
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["method"]), int(row["seed"])): row
        for row in long_rows
        if row["seed"] != ""
    }
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        coordinate = lookup[("coordinate", seed)]
        blocked = lookup[("layer_block", seed)]
        for metric in metrics:
            coordinate_value = coordinate.get(metric, "")
            blocked_value = blocked.get(metric, "")
            if coordinate_value == "" or blocked_value == "":
                continue
            coordinate_number = float(coordinate_value)
            blocked_number = float(blocked_value)
            rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "coordinate": coordinate_number,
                    "layer_block": blocked_number,
                    "layer_block_minus_coordinate": blocked_number - coordinate_number,
                }
            )
    return rows


def _convergence_rows(
    trials_by_run: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    seeds: Sequence[int],
    budget: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy in _STRATEGIES:
        curves: list[np.ndarray] = []
        for seed in seeds:
            values = np.asarray(
                [float(row["fitness_nll"]) for row in trials_by_run[(strategy, seed)]],
                dtype=np.float64,
            )
            curves.append(np.minimum.accumulate(values))
        matrix = np.stack(curves, axis=0)
        if matrix.shape != (len(seeds), budget):
            raise AssertionError("verified trial matrix has an unexpected shape")
        means = np.mean(matrix, axis=0)
        standard_deviations = np.std(matrix, axis=0, ddof=1)
        for evaluation_index in range(budget):
            rows.append(
                {
                    "method": strategy,
                    "evaluation_count": evaluation_index + 1,
                    "n": len(seeds),
                    "best_so_far_mean_nll": float(means[evaluation_index]),
                    "best_so_far_sample_sd_nll": float(standard_deviations[evaluation_index]),
                }
            )
    return rows


def _convergence_svg(
    rows: Sequence[Mapping[str, Any]],
    trials_by_run: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    seeds: Sequence[int],
) -> str:
    width, height = 960, 560
    left, right, top, bottom = 82, 28, 38, 68
    plot_width = width - left - right
    plot_height = height - top - bottom
    grouped = {
        method: [row for row in rows if row["method"] == method]
        for method in _STRATEGIES
    }
    x_values = [int(row["evaluation_count"]) for row in rows]
    upper = [
        float(row["best_so_far_mean_nll"]) + float(row["best_so_far_sample_sd_nll"])
        for row in rows
    ]
    lower = [
        float(row["best_so_far_mean_nll"]) - float(row["best_so_far_sample_sd_nll"])
        for row in rows
    ]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(lower), max(upper)
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        padding = max(abs(y_min) * 0.05, 1.0e-6)
        y_min -= padding
        y_max += padding
    else:
        padding = 0.04 * (y_max - y_min)
        y_min -= padding
        y_max += padding

    def x_pixel(value: float) -> float:
        return left + (value - x_min) * plot_width / (x_max - x_min)

    def y_pixel(value: float) -> float:
        return top + (y_max - value) * plot_height / (y_max - y_min)

    elements = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<style>text{font-family:Arial,sans-serif;fill:#1f2937}'
            '.grid{stroke:#e5e7eb;stroke-width:1}'
            '.axis{stroke:#374151;stroke-width:1.2}</style>'
        ),
        (
            f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" '
            'font-size="17" font-weight="600">'
            'EA convergence (mean ± sample SD over seeds)</text>'
        ),
    ]
    for tick in range(6):
        fraction = tick / 5
        y_value = y_max - fraction * (y_max - y_min)
        y = top + fraction * plot_height
        elements.append(
            f'<line class="grid" x1="{left}" y1="{y:.2f}" '
            f'x2="{left + plot_width}" y2="{y:.2f}"/>'
        )
        elements.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="11">{y_value:.5g}</text>'
        )
    for tick in range(6):
        fraction = tick / 5
        value = x_min + fraction * (x_max - x_min)
        x = left + fraction * plot_width
        elements.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 22}" '
            f'text-anchor="middle" font-size="11">{value:.0f}</text>'
        )
    elements.extend(
        [
            (
                f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" '
                f'y2="{top + plot_height}"/>'
            ),
            (
                f'<line class="axis" x1="{left}" y1="{top + plot_height}" '
                f'x2="{left + plot_width}" y2="{top + plot_height}"/>'
            ),
            (
                f'<text x="{left + plot_width / 2:.1f}" y="{height - 20}" '
                'text-anchor="middle" font-size="13">Completed candidate evaluations</text>'
            ),
            (
                f'<text x="20" y="{top + plot_height / 2:.1f}" text-anchor="middle" '
                f'font-size="13" transform="rotate(-90 20 {top + plot_height / 2:.1f})">'
                'Best-so-far calibration NLL (lower is better)</text>'
            ),
        ]
    )
    colors = {"coordinate": "#2563eb", "layer_block": "#dc2626"}
    for legend_index, method in enumerate(_STRATEGIES):
        method_rows = grouped[method]
        color = colors[method]
        for seed in seeds:
            seed_curve = np.minimum.accumulate(
                np.asarray(
                    [
                        float(trial["fitness_nll"])
                        for trial in trials_by_run[(method, seed)]
                    ],
                    dtype=np.float64,
                )
            )
            seed_line = " ".join(
                f"{x_pixel(index):.2f},{y_pixel(value):.2f}"
                for index, value in enumerate(seed_curve, start=1)
            )
            elements.append(
                f'<polyline points="{seed_line}" fill="none" stroke="{color}" '
                'stroke-width="0.9" stroke-opacity="0.28"/>'
            )
        upper_points = [
            (
                x_pixel(float(row["evaluation_count"])),
                y_pixel(
                    float(row["best_so_far_mean_nll"])
                    + float(row["best_so_far_sample_sd_nll"])
                ),
            )
            for row in method_rows
        ]
        lower_points = [
            (
                x_pixel(float(row["evaluation_count"])),
                y_pixel(
                    float(row["best_so_far_mean_nll"])
                    - float(row["best_so_far_sample_sd_nll"])
                ),
            )
            for row in reversed(method_rows)
        ]
        band = " ".join(f"{x:.2f},{y:.2f}" for x, y in upper_points + lower_points)
        line = " ".join(
            f"{x_pixel(float(row['evaluation_count'])):.2f},"
            f"{y_pixel(float(row['best_so_far_mean_nll'])):.2f}"
            for row in method_rows
        )
        elements.append(
            f'<polygon points="{band}" fill="{color}" fill-opacity="0.12" stroke="none"/>'
        )
        elements.append(
            f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2.2"/>'
        )
        legend_x = left + plot_width - 180
        legend_y = top + 18 + 22 * legend_index
        elements.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="2.5"/>'
        )
        elements.append(
            f'<text x="{legend_x + 36}" y="{legend_y + 4}" font-size="12">{method}</text>'
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def summarize_results(
    config: Mapping[str, Any],
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and aggregate all formal runs from the locked experiment."""

    root = Path(run_dir).resolve()
    resolved_config = root / "config.resolved.yaml"
    if not resolved_config.is_file():
        raise FileNotFoundError(f"missing locked run configuration: {resolved_config}")
    with resolved_config.open("r", encoding="utf-8") as handle:
        locked_config = yaml.safe_load(handle)
    if not isinstance(locked_config, dict) or config_digest(locked_config) != config_digest(
        dict(config)
    ):
        raise ValueError("provided configuration differs from the run's locked configuration")
    strategies = tuple(str(value) for value in config["search"]["strategies"])
    seeds = tuple(int(value) for value in config["search"]["seeds"])
    if strategies != _STRATEGIES or len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError(
            "formal summary requires coordinate/layer_block and exactly three seeds"
        )
    budget = int(config["search"]["evaluation_budget"])
    warmup_tokens = int(config["search"]["warmup_tokens"])
    fitness_tokens = int(config["data"]["fitness_tokens"])
    validation_tokens = int(config["data"]["validation_tokens"])
    tasks = tuple(str(value) for value in config["evaluation"]["tasks"])
    expected_samples = {
        str(name): int(value)
        for name, value in config["evaluation"]["expected_samples"].items()
    }
    task_datasets = config["evaluation"]["task_datasets"]
    if tasks != ("arc_easy", "openbookqa"):
        raise ValueError("formal summary requires the pre-registered downstream tasks")

    destination = (root / "summary") if output_dir is None else Path(output_dir).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            "formal summary output must remain inside the locked run directory"
        ) from exc
    filenames = (
        "runs_long.csv",
        "method_summary.csv",
        "method_totals.csv",
        "compression_summary.csv",
        "paired_differences.csv",
        "convergence.csv",
        "convergence.svg",
        "summary.json",
        "stage_manifest.json",
    )
    if destination.exists():
        raise ValueError(f"refusing to overwrite formal summary directory: {destination}")

    summaries: dict[tuple[str, int], dict[str, Any]] = {}
    trials_by_run: dict[tuple[str, int], list[dict[str, Any]]] = {}
    search_audits: dict[tuple[str, int], dict[str, Any]] = {}
    source_paths: dict[str, Path] = {"resolved_config": resolved_config}
    compression, compression_paths = _verify_compression(
        root,
        model_id=str(config["model"]["id"]),
        model_revision=str(config["model"]["revision"]),
        tokenizer_revision=str(config["model"]["tokenizer_revision"]),
        parameter_dtype=str(config["model"]["dtype"]),
    )
    source_paths.update(compression_paths)
    for strategy in strategies:
        for seed in seeds:
            summary, trials, audit, paths = _verify_search(
                root,
                strategy=strategy,
                seed=seed,
                expected_evaluations=budget,
                fitness_tokens=fitness_tokens,
                validation_tokens=validation_tokens,
                warmup_tokens=warmup_tokens,
            )
            summaries[(strategy, seed)] = summary
            trials_by_run[(strategy, seed)] = trials
            search_audits[(strategy, seed)] = audit
            prefix = f"search_{strategy}_seed_{seed}"
            for name, path in paths.items():
                source_paths[f"{prefix}_{name}"] = path

    evaluations: dict[tuple[str, int | None], dict[str, Any]] = {}
    for method in ("original", "frequency"):
        artifact, path, manifest_path = _verify_evaluation(
            root,
            method=method,
            seed=None,
            tasks=tasks,
            expected_samples=expected_samples,
            task_datasets=task_datasets,
            strategies=strategies,
            seeds=seeds,
        )
        evaluations[(method, None)] = artifact
        source_paths[f"evaluation_{method}_deterministic"] = path
        source_paths[f"evaluation_{method}_deterministic_manifest"] = manifest_path
    for strategy in strategies:
        for seed in seeds:
            artifact, path, manifest_path = _verify_evaluation(
                root,
                method=strategy,
                seed=seed,
                tasks=tasks,
                expected_samples=expected_samples,
                task_datasets=task_datasets,
                strategies=strategies,
                seeds=seeds,
            )
            evaluations[(strategy, seed)] = artifact
            source_paths[f"evaluation_{strategy}_seed_{seed}"] = path
            source_paths[f"evaluation_{strategy}_seed_{seed}_manifest"] = manifest_path
    _verify_dataset_evidence_consistency(evaluations, tasks)

    long_rows: list[dict[str, Any]] = []
    for method in ("original", "frequency"):
        long_rows.append(
            _long_row(
                method,
                None,
                evaluations[(method, None)],
                tasks,
                compression,
            )
        )
    for strategy in strategies:
        for seed in seeds:
            long_rows.append(
                _long_row(
                    strategy,
                    seed,
                    evaluations[(strategy, seed)],
                    tasks,
                    compression,
                    summary=summaries[(strategy, seed)],
                    trials=trials_by_run[(strategy, seed)],
                    search_audit=search_audits[(strategy, seed)],
                )
            )
    metrics = _metric_names(tasks)
    summary_metrics = ["evaluations", "best_evaluation_count", *metrics]
    method_rows = _sample_summary(long_rows, summary_metrics)
    additive_metrics = (
        "evaluations",
        "warmup_sessions",
        "warmup_loss_tokens",
        "planned_calibration_loss_tokens",
        "actual_calibration_loss_tokens",
        "validation_loss_tokens",
        "total_calibration_loss_tokens",
        "total_compute_loss_tokens",
        "failed_evaluations",
        "warmup_wall_seconds",
        "warmup_gpu_seconds",
        "search_wall_seconds",
        "search_fusion_seconds",
        "search_forward_seconds",
        "search_gpu_seconds",
        "validation_wall_seconds",
        "validation_fusion_seconds",
        "validation_forward_seconds",
        "validation_gpu_seconds",
        "downstream_wall_seconds",
        "downstream_gpu_seconds",
        "total_wall_seconds",
        "total_gpu_seconds",
    )
    total_rows = _method_totals(long_rows, additive_metrics)
    paired_metrics = [
        "search_best_nll",
        "validation_nll",
        "mean_primary_acc",
        *(f"{task}_{metric}" for task in tasks for metric in ("acc", "acc_norm")),
    ]
    paired_rows = _paired_differences(long_rows, seeds, paired_metrics)
    convergence_rows = _convergence_rows(trials_by_run, seeds, budget)
    compression_rows = [
        {
            "deployment": "original",
            "total_parameters": compression["original_total_parameters"],
            "expert_parameters": compression["original_expert_parameters"],
            "whole_model_reduction_fraction": 0.0,
            "expert_reduction_fraction": 0.0,
            "weight_checkpoint_bytes": compression["base_weight_checkpoint_bytes"],
            "weight_checkpoint_reduction_fraction": 0.0,
        },
        {
            "deployment": "fixed_group_fused_shared_by_frequency_and_ea",
            "total_parameters": compression["fused_total_parameters"],
            "expert_parameters": compression["fused_expert_parameters"],
            "whole_model_reduction_fraction": compression[
                "whole_model_reduction_fraction"
            ],
            "expert_reduction_fraction": compression["expert_reduction_fraction"],
            "weight_checkpoint_bytes": compression["standalone_checkpoint_bytes"],
            "weight_checkpoint_reduction_fraction": compression[
                "weight_checkpoint_reduction_fraction"
            ],
        },
    ]

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.summary-", dir=destination.parent)
    )
    long_fields = [
        "method",
        "seed",
        "run_type",
        "compression_basis",
        "evaluations",
        "best_evaluation_index",
        "best_evaluation_count",
        "resumed_from_checkpoint",
        "resume_count",
        "checkpoint_writes_total",
        "numerical_environment_sha256",
        *metrics,
    ]
    payload = {
        "schema_version": "formal-result-summary-v1",
        "is_synthetic": False,
        "methods": list(_METHODS),
        "strategies": list(strategies),
        "seeds": list(seeds),
        "evaluation_budget": budget,
        "tasks": list(tasks),
        "run_rows": len(long_rows),
        "method_total_rows": len(total_rows),
        "paired_rows": len(paired_rows),
        "convergence_rows": len(convergence_rows),
        "sample_sd_definition": "Bessel-corrected sample standard deviation (ddof=1)",
        "paired_difference_definition": "layer_block minus coordinate for the same seed",
        "compression_definition": (
            "The frequency checkpoint supplies the shared physical topology and byte counts "
            "for frequency and both EA methods; coefficient values do not change shape."
        ),
    }
    try:
        _write_csv(staging / "runs_long.csv", long_fields, long_rows)
        _write_csv(
            staging / "method_summary.csv",
            ("method", "metric", "n", "mean", "sample_sd"),
            method_rows,
        )
        _write_csv(
            staging / "method_totals.csv",
            ("method", "metric", "n", "total"),
            total_rows,
        )
        _write_csv(
            staging / "compression_summary.csv",
            (
                "deployment",
                "total_parameters",
                "expert_parameters",
                "whole_model_reduction_fraction",
                "expert_reduction_fraction",
                "weight_checkpoint_bytes",
                "weight_checkpoint_reduction_fraction",
            ),
            compression_rows,
        )
        _write_csv(
            staging / "paired_differences.csv",
            (
                "seed",
                "metric",
                "coordinate",
                "layer_block",
                "layer_block_minus_coordinate",
            ),
            paired_rows,
        )
        _write_csv(
            staging / "convergence.csv",
            (
                "method",
                "evaluation_count",
                "n",
                "best_so_far_mean_nll",
                "best_so_far_sample_sd_nll",
            ),
            convergence_rows,
        )
        _write_text(
            staging / "convergence.svg",
            _convergence_svg(convergence_rows, trials_by_run, seeds),
        )
        write_json(staging / "summary.json", payload)
        outputs = {
            name.replace(".", "_"): ArtifactRef(
                path=str((destination / name).relative_to(root)),
                sha256=sha256_file(staging / name),
            )
            for name in filenames
            if name != "stage_manifest.json"
        }
        write_stage_manifest(
            staging / "stage_manifest.json",
            stage="formal_result_summary",
            inputs={
                name: ArtifactRef.capture(path, relative_to=root)
                for name, path in source_paths.items()
            },
            outputs=outputs,
            metadata={
                "is_synthetic": False,
                "strategies": list(strategies),
                "seeds": list(seeds),
                "evaluation_budget": budget,
            },
        )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return payload


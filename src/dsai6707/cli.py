"""Command-line entry points for the reproducible experiment stages."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import ArtifactRef, verify_stage_manifest, write_stage_manifest
from .coefficients import CoefficientLayout, frequency_coefficients
from .config import (
    CAPACITY_TWO_ABLATION_STUDY_TYPE,
    CAPACITY_TWO_EA_PILOT_SCOPE,
    SUPPLEMENTAL_EVALUATION_PROTOCOL_ID,
    load_config,
    load_supplemental_evaluation_config,
    lock_run_config,
    lock_supplemental_evaluation_config,
)
from .evolution import (
    EAConfig,
    make_initial_population,
    population_hash,
    run_evolution,
    run_objective_warmup,
)
from .grouping import (
    GroupingPlan,
    capacity_two_plan_from_similarities,
    plan_from_similarities,
)
from .io import environment_snapshot, read_json, read_jsonl, sha256_file, write_json


def _artifact_finite_float(value: Any, *, field: str) -> float:
    """Parse a finite floating-point artifact field."""

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not np.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _artifact_integer(value: Any, *, field: str) -> int:
    """Parse an integer-valued artifact field without leaking conversion errors."""

    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        number = float(value)
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a finite integer") from exc
    if not np.isfinite(number) or number != integer:
        raise ValueError(f"{field} must be a finite integer")
    return integer


def _target_groups(value: Any, layers: int) -> int | list[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, list) and len(value) == layers:
        return [int(item) for item in value]
    raise ValueError("target_groups_per_layer must be an integer or one integer per layer")


def _capacity_two_invariant_summary(plan: GroupingPlan, *, target_groups: int) -> dict[str, Any]:
    """Validate and summarize the strict pair/singleton structure."""

    expected_pairs = plan.num_experts - target_groups
    expected_singletons = 2 * target_groups - plan.num_experts
    per_layer: list[dict[str, int]] = []
    for layer in plan.layers:
        sizes = [len(group) for group in layer.groups]
        pairs = sizes.count(2)
        singletons = sizes.count(1)
        maximum = max(sizes)
        if (
            layer.num_groups != target_groups
            or pairs != expected_pairs
            or singletons != expected_singletons
            or pairs + singletons != layer.num_groups
            or maximum > 2
        ):
            raise RuntimeError(
                f"capacity-two invariant failed for layer {layer.layer}: "
                f"groups={layer.num_groups}, pairs={pairs}, "
                f"singletons={singletons}, max_size={maximum}"
            )
        per_layer.append(
            {
                "layer": layer.layer,
                "pairs": pairs,
                "singletons": singletons,
                "max_group_size": maximum,
            }
        )
    return {
        "schema_version": "capacity-two-invariants-v1",
        "num_layers": plan.num_layers,
        "num_experts_per_layer": plan.num_experts,
        "target_groups_per_layer": target_groups,
        "pairs_per_layer": expected_pairs,
        "singletons_per_layer": expected_singletons,
        "total_pairs": expected_pairs * plan.num_layers,
        "total_singletons": expected_singletons * plan.num_layers,
        "max_group_size": max(row["max_group_size"] for row in per_layer),
        "per_layer": per_layer,
    }


def _locked_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    config = load_config(args.config)
    run_dir = Path(args.run_dir)
    lock_run_config(config, run_dir)
    return config, run_dir


def _verify(run_dir: Path, relative_manifest: str, *, expected_stage: str) -> dict[str, Any]:
    path = run_dir / relative_manifest
    if not path.is_file():
        raise FileNotFoundError(f"missing upstream stage manifest: {path}")
    manifest = verify_stage_manifest(path, run_dir=run_dir)
    if manifest.get("stage") != expected_stage:
        raise ValueError(
            f"unexpected stage in {path}: expected {expected_stage}, got {manifest.get('stage')}"
        )
    return manifest


def _guard_new_stage(run_dir: Path, relative_directory: str, *, stage: str) -> Path:
    """Refuse to overwrite either finalized or interrupted stage artifacts."""

    output = run_dir / relative_directory
    manifest_path = output / "stage_manifest.json"
    if manifest_path.is_file():
        _verify(
            run_dir,
            str(manifest_path.relative_to(run_dir)),
            expected_stage=stage,
        )
        raise ValueError(f"refusing to overwrite finalized {stage} stage: {output}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(
            f"incomplete {stage} artifacts already exist in {output}; preserve them for "
            "audit and use a new run directory"
        )
    return output


@contextmanager
def _exclusive_search_lock(run_dir: Path, *, strategy: str, seed: int):
    """Hold a process-scoped single-writer lock for one formal search."""

    lock_path = run_dir / "search" / ".locks" / f"{strategy}_seed_{seed}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _assert_all_searches_complete(config: dict[str, Any], run_dir: Path) -> None:
    for strategy in config["search"]["strategies"]:
        for seed in config["search"]["seeds"]:
            relative = f"search/{strategy}/seed_{int(seed)}/stage_manifest.json"
            _verify(run_dir, relative, expected_stage="evolutionary_search")
            summary = read_json(
                run_dir / "search" / strategy / f"seed_{int(seed)}" / "summary.json"
            )
            if not summary.get("valid_for_main_results", False):
                raise ValueError(f"search {strategy}/seed_{seed} is not valid for main results")


def _search_gate_refs(config: dict[str, Any], run_dir: Path) -> dict[str, ArtifactRef]:
    return {
        f"locked_search_{strategy}_{int(seed)}": ArtifactRef.capture(
            run_dir / "search" / strategy / f"seed_{int(seed)}" / "stage_manifest.json",
            relative_to=run_dir,
        )
        for strategy in config["search"]["strategies"]
        for seed in config["search"]["seeds"]
    }


def _capacity_two_ea_reference_refs(
    config: dict[str, Any], run_dir: Path
) -> dict[str, ArtifactRef]:
    """Prove that an isolated EA run exactly reuses its immutable Frequency setup."""

    project = config.get("project", {})
    method_scope = project.get("method_scope")
    if project.get("study_type") != CAPACITY_TWO_ABLATION_STUDY_TYPE or method_scope not in {
        "all_methods",
        CAPACITY_TWO_EA_PILOT_SCOPE,
    }:
        return {}

    protocol_field = (
        "ea_pilot_protocol"
        if method_scope == CAPACITY_TWO_EA_PILOT_SCOPE
        else "ea_extension_protocol"
    )
    protocol_sha_field = f"{protocol_field}_sha256"
    protocol_path = Path(str(project[protocol_field])).resolve()
    if not protocol_path.is_file():
        raise FileNotFoundError(f"missing locked EA protocol: {protocol_path}")
    if sha256_file(protocol_path) != project[protocol_sha_field]:
        raise ValueError("capacity-two EA protocol SHA-256 mismatch")

    reference_run = Path(str(project["frequency_reference_run"])).resolve()
    if reference_run == run_dir.resolve():
        raise ValueError("capacity-two EA run must be distinct from its Frequency reference")
    for relative, stage in (
        ("data/stage_manifest.json", "prepare_data"),
        ("statistics/stage_manifest.json", "collect_statistics"),
        ("grouping/stage_manifest.json", "fixed_grouping"),
        ("baselines/frequency/stage_manifest.json", "frequency_baseline"),
    ):
        _verify(reference_run, relative, expected_stage=stage)

    compared = {
        "tokens": "data/tokens.npz",
        "statistics": "statistics/expert_statistics.npz",
        "grouping": "grouping/grouping.json",
        "coefficients": "baselines/frequency/coefficients.json",
    }
    refs: dict[str, ArtifactRef] = {
        "frequency_reference_config": ArtifactRef.capture(reference_run / "config.resolved.yaml"),
        protocol_field: ArtifactRef.capture(protocol_path),
    }
    for label, relative in compared.items():
        current = run_dir / relative
        reference = reference_run / relative
        if not current.is_file() or not reference.is_file():
            raise FileNotFoundError(
                f"capacity-two EA/reference comparison requires both copies of {relative}"
            )
        if sha256_file(current) != sha256_file(reference):
            raise ValueError(
                f"capacity-two EA run differs from its Frequency reference: {relative}"
            )
        refs[f"frequency_reference_{label}"] = ArtifactRef.capture(reference)
    return refs


def _requires_search_gate(config: dict[str, Any], method: str) -> bool:
    """Return whether an evaluation must wait for all configured searches."""

    project = config.get("project")
    study_type = (
        project.get("study_type", "formal_main") if isinstance(project, dict) else "formal_main"
    )
    frequency_only_studies = {
        "compression_ablation",
        CAPACITY_TWO_ABLATION_STUDY_TYPE,
    }
    return not (study_type in frequency_only_studies and method == "frequency")


def _validate_evaluation_artifact(
    artifact: dict[str, Any],
    *,
    config: dict[str, Any],
    method: str,
    seed: int | None,
    limit: float | None,
) -> None:
    from .evaluation import validate_raw_evaluation_evidence

    if artifact.get("schema_version") != "downstream-evaluation-v1":
        raise ValueError("downstream evaluation has an unsupported schema")
    if artifact.get("is_synthetic") is not False:
        raise ValueError("a synthetic evaluation cannot be finalized as a formal result")
    if artifact.get("method") != method or artifact.get("seed") != seed:
        raise ValueError("downstream evaluation identity mismatch")
    if artifact.get("limit") != limit:
        raise ValueError("downstream evaluation limit mismatch")
    if (
        artifact.get("primary_metric") != "acc"
        or _artifact_integer(artifact.get("num_fewshot", -1), field="downstream num_fewshot") != 0
    ):
        raise ValueError("downstream evaluation protocol mismatch")
    tasks = artifact.get("tasks")
    expected_tasks = config["evaluation"]["tasks"]
    if not isinstance(tasks, dict) or list(tasks) != expected_tasks:
        raise ValueError("downstream task order differs from the locked protocol")
    dataset_evidence = artifact.get("dataset_evidence")
    if not isinstance(dataset_evidence, dict) or list(dataset_evidence) != expected_tasks:
        raise ValueError("downstream evaluation is missing ordered dataset evidence")
    raw_lm_eval = artifact.get("raw_lm_eval")
    raw_configs = raw_lm_eval.get("configs") if isinstance(raw_lm_eval, dict) else None
    if not isinstance(raw_configs, dict):
        raise ValueError("downstream evaluation is missing lm-eval task configs")
    for task in expected_tasks:
        expected_dataset = config["evaluation"]["task_datasets"][task]
        evidence = dataset_evidence.get(task)
        if not isinstance(evidence, dict):
            raise ValueError(f"downstream evaluation is missing dataset evidence for {task}")
        locked_fields = {
            "hub_repository": expected_dataset["hub_repository"],
            "dataset_path": expected_dataset["path"],
            "dataset_name": expected_dataset["name"],
            "revision": expected_dataset["revision"],
            "evaluation_split": expected_dataset["split"],
        }
        for field, expected in locked_fields.items():
            if evidence.get(field) != expected:
                raise ValueError(f"downstream {task} dataset evidence has wrong {field}")
        if _artifact_integer(
            evidence.get("num_rows", -1), field=f"downstream {task} row count"
        ) != int(config["evaluation"]["expected_samples"][task]):
            raise ValueError(f"downstream {task} dataset evidence has wrong row count")
        if not str(evidence.get("huggingface_fingerprint", "")).strip():
            raise ValueError(f"downstream {task} is missing a Hugging Face fingerprint")
        for field in (
            "content_sha256",
            "first_row_sha256",
            "last_row_sha256",
            "feature_schema_sha256",
        ):
            digest = str(evidence.get(field, ""))
            invalid_character = any(character not in "0123456789abcdef" for character in digest)
            if len(digest) != 64 or invalid_character:
                raise ValueError(f"downstream {task} has invalid {field}")
        if evidence.get("content_hash_spec") != "sha256(index<TAB>canonical-json<LF>)":
            raise ValueError(f"downstream {task} has an unknown content hash specification")
        raw_task_config = raw_configs.get(task)
        raw_kwargs = (
            raw_task_config.get("dataset_kwargs") if isinstance(raw_task_config, dict) else None
        )
        if (
            not isinstance(raw_kwargs, dict)
            or raw_kwargs.get("revision") != expected_dataset["revision"]
        ):
            raise ValueError(f"lm-eval raw config does not prove {task} dataset revision")
    primary = []
    for task in expected_tasks:
        for metric in ("acc", "acc_norm"):
            value = _artifact_finite_float(
                tasks[task].get(metric), field=f"downstream {task}.{metric}"
            )
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid {task}.{metric} in downstream evaluation")
            if metric == "acc":
                primary.append(value)
    if not np.isclose(
        _artifact_finite_float(artifact.get("mean_primary_acc"), field="downstream mean accuracy"),
        float(np.mean(primary)),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("downstream mean accuracy is internally inconsistent")
    cost = artifact.get("cost")
    if not isinstance(cost, dict):
        raise ValueError("downstream evaluation is missing cost evidence")
    wall_seconds = _artifact_finite_float(
        cost.get("wall_seconds"), field="downstream wall-time cost"
    )
    gpu_seconds = _artifact_finite_float(
        cost.get("gpu_seconds"), field="downstream CUDA event time"
    )
    if wall_seconds <= 0.0:
        raise ValueError("downstream evaluation is missing wall-time cost")
    if gpu_seconds < 0.0:
        raise ValueError("downstream evaluation is missing CUDA event time")
    if (
        _artifact_integer(
            cost.get("peak_device_memory_bytes"),
            field="downstream peak device memory",
        )
        <= 0
    ):
        raise ValueError("downstream evaluation is missing peak device memory")
    counts = cost.get("sample_counts")
    if not isinstance(counts, dict) or list(counts) != expected_tasks:
        raise ValueError("downstream evaluation is missing task sample counts")
    for task, expected in config["evaluation"]["expected_samples"].items():
        item = counts.get(task, {})
        if (
            not isinstance(item, dict)
            or _artifact_integer(
                item.get("expected", -1), field=f"downstream {task} expected samples"
            )
            != expected
            or _artifact_integer(
                item.get("original", -1), field=f"downstream {task} original samples"
            )
            != expected
        ):
            raise ValueError(f"downstream dataset size mismatch for {task}")
        effective = _artifact_integer(
            item.get("effective", -1), field=f"downstream {task} effective samples"
        )
        if effective <= 0 or (limit is None and effective != expected):
            raise ValueError(f"formal downstream evaluation is incomplete for {task}")
    environment = artifact.get("environment")
    packages = environment.get("packages", {}) if isinstance(environment, dict) else {}
    if not isinstance(packages, dict) or packages.get("lm_eval") != "0.4.8":
        raise ValueError("formal downstream evaluation requires lm-eval==0.4.8")
    assert isinstance(raw_lm_eval, dict)
    validate_raw_evaluation_evidence(raw_lm_eval, expected_tasks, tasks, counts)


def _validate_supplemental_evaluation_artifact(
    artifact: dict[str, Any],
    *,
    supplemental_config: dict[str, Any],
    method: str,
    seed: int | None = None,
) -> None:
    """Validate a complete post-hoc WinoGrande/PIQA evaluation artifact."""

    from .evaluation import validate_raw_evaluation_evidence

    evaluation = supplemental_config["evaluation"]
    expected_tasks = evaluation["tasks"]
    if artifact.get("schema_version") != "downstream-evaluation-v1":
        raise ValueError("supplemental evaluation has an unsupported schema")
    if artifact.get("is_synthetic") is not False:
        raise ValueError("supplemental evaluation cannot be synthetic")
    if artifact.get("evaluation_protocol_id") != SUPPLEMENTAL_EVALUATION_PROTOCOL_ID:
        raise ValueError("supplemental evaluation protocol identity mismatch")
    if (
        artifact.get("method") != method
        or artifact.get("seed") != seed
        or artifact.get("limit") is not None
    ):
        raise ValueError("supplemental evaluation identity or full-data policy mismatch")
    if (
        artifact.get("primary_metric") != "acc"
        or _artifact_integer(artifact.get("num_fewshot", -1), field="supplemental num_fewshot") != 0
    ):
        raise ValueError("supplemental evaluation must be zero-shot accuracy")

    tasks = artifact.get("tasks")
    # JSON artifacts are written with ``sort_keys=True``, so mapping insertion
    # order does not survive a write/read round trip.  The locked config still
    # controls execution order; persisted evidence must contain exactly the
    # locked task set.
    if not isinstance(tasks, dict) or set(tasks) != set(expected_tasks):
        raise ValueError("supplemental task set differs from the locked protocol")
    dataset_evidence = artifact.get("dataset_evidence")
    if not isinstance(dataset_evidence, dict) or set(dataset_evidence) != set(expected_tasks):
        raise ValueError("supplemental evaluation lacks exact dataset evidence")
    raw_lm_eval = artifact.get("raw_lm_eval")
    raw_configs = raw_lm_eval.get("configs") if isinstance(raw_lm_eval, dict) else None
    if not isinstance(raw_configs, dict) or set(raw_configs) != set(expected_tasks):
        raise ValueError("supplemental evaluation lacks exact lm-eval task configs")

    for task in expected_tasks:
        expected_dataset = evaluation["task_datasets"][task]
        evidence = dataset_evidence.get(task)
        if not isinstance(evidence, dict):
            raise ValueError(f"supplemental evaluation lacks dataset evidence for {task}")
        locked_fields = {
            "hub_repository": expected_dataset["hub_repository"],
            "lm_eval_task": task,
            "dataset_path": expected_dataset["path"],
            "dataset_name": expected_dataset["name"],
            "revision": expected_dataset["revision"],
            "evaluation_split": expected_dataset["split"],
            "output_type": expected_dataset["output_type"],
        }
        for field, expected in locked_fields.items():
            if evidence.get(field) != expected:
                raise ValueError(f"supplemental {task} evidence has wrong {field}")
        expected_count = int(evaluation["expected_samples"][task])
        if (
            _artifact_integer(evidence.get("num_rows", -1), field=f"supplemental {task} row count")
            != expected_count
        ):
            raise ValueError(f"supplemental {task} evidence has wrong row count")
        if not str(evidence.get("huggingface_fingerprint", "")).strip():
            raise ValueError(f"supplemental {task} lacks a Hugging Face fingerprint")
        expected_content = evaluation["expected_content_evidence"][task]
        for field, expected in expected_content.items():
            if evidence.get(field) != expected:
                raise ValueError(f"supplemental {task} content evidence has wrong {field}")
        for field in (
            "content_sha256",
            "first_row_sha256",
            "last_row_sha256",
            "feature_schema_sha256",
        ):
            digest = str(evidence.get(field, ""))
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"supplemental {task} has invalid {field}")
        if evidence.get("content_hash_spec") != "sha256(index<TAB>canonical-json<LF>)":
            raise ValueError(f"supplemental {task} has unknown content hash specification")
        raw_task_config = raw_configs[task]
        raw_kwargs = (
            raw_task_config.get("dataset_kwargs") if isinstance(raw_task_config, dict) else None
        )
        if (
            not isinstance(raw_kwargs, dict)
            or raw_kwargs.get("revision") != expected_dataset["revision"]
        ):
            raise ValueError(f"lm-eval raw config does not prove {task} revision")

    primary: list[float] = []
    for task in expected_tasks:
        task_metrics = tasks[task]
        if not isinstance(task_metrics, dict) or list(task_metrics) != ["acc"]:
            raise ValueError(f"supplemental {task} must report exactly raw acc")
        value = _artifact_finite_float(task_metrics["acc"], field=f"supplemental {task}.acc")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid supplemental {task}.acc")
        primary.append(value)
    if not np.isclose(
        _artifact_finite_float(
            artifact.get("mean_primary_acc"), field="supplemental mean accuracy"
        ),
        float(np.mean(primary)),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("supplemental mean accuracy is internally inconsistent")

    expected_chance = evaluation["chance_accuracy"]
    if artifact.get("chance_accuracy") != expected_chance:
        raise ValueError("supplemental chance baseline differs from the locked protocol")
    adjusted = artifact.get("chance_adjusted_primary")
    if not isinstance(adjusted, dict) or set(adjusted) != set(expected_tasks):
        raise ValueError("supplemental evaluation lacks exact chance-adjusted scores")
    expected_adjusted: list[float] = []
    for task, accuracy in zip(expected_tasks, primary, strict=True):
        chance = float(expected_chance[task])
        expected_value = (accuracy - chance) / (1.0 - chance)
        actual_value = _artifact_finite_float(
            adjusted.get(task), field=f"supplemental {task} chance-adjusted accuracy"
        )
        if not np.isclose(actual_value, expected_value, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"supplemental {task} chance-adjusted score is inconsistent")
        expected_adjusted.append(expected_value)
    if not np.isclose(
        _artifact_finite_float(
            artifact.get("mean_chance_adjusted_primary"),
            field="supplemental mean chance-adjusted accuracy",
        ),
        float(np.mean(expected_adjusted)),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("supplemental mean chance-adjusted score is inconsistent")

    cost = artifact.get("cost")
    if not isinstance(cost, dict):
        raise ValueError("supplemental evaluation lacks cost evidence")
    if _artifact_finite_float(cost.get("wall_seconds"), field="supplemental wall time") <= 0.0:
        raise ValueError("supplemental evaluation lacks positive wall time")
    if _artifact_finite_float(cost.get("gpu_seconds"), field="supplemental CUDA event time") < 0.0:
        raise ValueError("supplemental evaluation has invalid CUDA event time")
    if (
        _artifact_integer(
            cost.get("peak_device_memory_bytes"), field="supplemental peak device memory"
        )
        <= 0
    ):
        raise ValueError("supplemental evaluation lacks peak device memory")
    counts = cost.get("sample_counts")
    if not isinstance(counts, dict) or set(counts) != set(expected_tasks):
        raise ValueError("supplemental evaluation lacks exact sample counts")
    for task in expected_tasks:
        expected_count = int(evaluation["expected_samples"][task])
        item = counts.get(task)
        if not isinstance(item, dict):
            raise ValueError(f"supplemental evaluation lacks {task} sample counts")
        for field in ("expected", "original", "effective"):
            if (
                _artifact_integer(item.get(field, -1), field=f"supplemental {task} {field} samples")
                != expected_count
            ):
                raise ValueError(f"supplemental evaluation is incomplete for {task}")

    environment = artifact.get("environment")
    packages = environment.get("packages", {}) if isinstance(environment, dict) else {}
    if not isinstance(packages, dict) or packages.get("lm_eval") != "0.4.8":
        raise ValueError("supplemental evaluation requires lm-eval==0.4.8")
    assert isinstance(environment, dict)
    _validate_search_environment(environment, require_cuda=True)
    assert isinstance(raw_lm_eval, dict)
    validate_raw_evaluation_evidence(
        raw_lm_eval,
        expected_tasks,
        tasks,
        counts,
        metric_names=("acc",),
    )


def command_validate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    print(f"valid: {args.config} ({config['project']['name']})")


def command_smoke(args: argparse.Namespace) -> None:
    from .smoke import run_smoke

    config = load_config(args.config)
    result = run_smoke(args.output, config=config)
    print(
        f"synthetic smoke passed: {len(result['search'])} searches; "
        f"artifacts in {Path(args.output).resolve()}"
    )


def command_prepare_data(args: argparse.Namespace) -> None:
    from .data import prepare_wikitext

    config, run_dir = _locked_config(args)
    output = _guard_new_stage(run_dir, "data", stage="prepare_data")
    manifest = prepare_wikitext(config, output)
    write_json(output / "environment.json", environment_snapshot())
    write_stage_manifest(
        output / "stage_manifest.json",
        stage="prepare_data",
        inputs={
            "resolved_config": ArtifactRef.capture(
                run_dir / "config.resolved.yaml", relative_to=run_dir
            )
        },
        outputs={
            "tokens": ArtifactRef.capture(output / "tokens.npz", relative_to=run_dir),
            "manifest": ArtifactRef.capture(output / "manifest.json", relative_to=run_dir),
            "environment": ArtifactRef.capture(output / "environment.json", relative_to=run_dir),
        },
    )
    print(f"prepared pinned WikiText tokens: {manifest['tokens_sha256']} " f"in {output.resolve()}")


def command_inspect_model(args: argparse.Namespace) -> None:
    from .granite import inspect_granite_topology, load_granite_model

    config, run_dir = _locked_config(args)
    output = run_dir / "model_topology.json"
    if output.exists():
        raise ValueError(f"refusing to overwrite model topology artifact: {output}")
    model, _ = load_granite_model(config["model"])
    topology = inspect_granite_topology(model)
    write_json(output, {"schema_version": "model-topology-v1", **topology})
    print(f"Granite topology validated: {topology}")


def command_collect_stats(args: argparse.Namespace) -> None:
    from .data import load_token_bundle
    from .granite import collect_expert_statistics, load_granite_model

    config, run_dir = _locked_config(args)
    _verify(run_dir, "data/stage_manifest.json", expected_stage="prepare_data")
    output = _guard_new_stage(run_dir, "statistics", stage="collect_statistics")
    token_path = run_dir / "data" / "tokens.npz"
    token_manifest_path = run_dir / "data" / "manifest.json"
    if not token_path.is_file() or not token_manifest_path.is_file():
        raise FileNotFoundError("run prepare-data before collect-stats")
    bundle = load_token_bundle(token_path)
    model, _ = load_granite_model(config["model"])
    stats = collect_expert_statistics(
        model,
        bundle.statistics,
        signature_tokens_per_layer=int(config["data"]["signature_tokens_per_layer"]),
        forward_batch_size=int(config["data"].get("forward_batch_size", 1)),
    )
    stats.save(
        output,
        {
            "model": {
                "id": config["model"]["id"],
                "revision": config["model"]["revision"],
            },
            "token_manifest_sha256": sha256_file(token_manifest_path),
            "token_array_sha256": sha256_file(token_path),
            "signature_tokens_per_layer": int(config["data"]["signature_tokens_per_layer"]),
        },
    )
    write_json(output / "environment.json", environment_snapshot())
    write_stage_manifest(
        output / "stage_manifest.json",
        stage="collect_statistics",
        inputs={
            "resolved_config": ArtifactRef.capture(
                run_dir / "config.resolved.yaml", relative_to=run_dir
            ),
            "data_stage": ArtifactRef.capture(
                run_dir / "data" / "stage_manifest.json", relative_to=run_dir
            ),
            "tokens": ArtifactRef.capture(token_path, relative_to=run_dir),
            "token_manifest": ArtifactRef.capture(token_manifest_path, relative_to=run_dir),
        },
        outputs={
            "statistics": ArtifactRef.capture(
                output / "expert_statistics.npz", relative_to=run_dir
            ),
            "metadata": ArtifactRef.capture(output / "metadata.json", relative_to=run_dir),
            "environment": ArtifactRef.capture(output / "environment.json", relative_to=run_dir),
        },
    )
    print(f"expert statistics saved in {output.resolve()}")


def command_cluster(args: argparse.Namespace) -> None:
    config, run_dir = _locked_config(args)
    _verify(run_dir, "data/stage_manifest.json", expected_stage="prepare_data")
    _verify(
        run_dir,
        "statistics/stage_manifest.json",
        expected_stage="collect_statistics",
    )
    output = _guard_new_stage(run_dir, "grouping", stage="fixed_grouping")
    stats_path = run_dir / "statistics" / "expert_statistics.npz"
    token_manifest_path = run_dir / "data" / "manifest.json"
    if not stats_path.is_file():
        raise FileNotFoundError("run collect-stats before cluster")
    with np.load(stats_path, allow_pickle=False) as payload:
        similarities = payload["output_similarity"].astype(np.float64, copy=False)
    token_manifest = read_json(token_manifest_path)
    target_groups = _target_groups(
        config["grouping"]["target_groups_per_layer"], similarities.shape[0]
    )
    plan_factory = (
        capacity_two_plan_from_similarities
        if config["project"].get("study_type") == CAPACITY_TWO_ABLATION_STUDY_TYPE
        else plan_from_similarities
    )
    plan = plan_factory(
        similarities,
        target_groups,
        model_id=str(config["model"]["id"]),
        model_revision=str(config["model"]["revision"]),
        calibration_fingerprint=str(token_manifest["sampling_fingerprint"]),
        seed=int(config["data"]["seed"]),
    )
    grouping_payload = plan.to_dict()
    grouping_payload["configuration_source"] = {
        "retention_ratio": config["grouping"].get("retention_ratio"),
        "rounding_rule": config["grouping"].get("rounding_rule"),
        "reference": config["grouping"]["source_reference"],
    }
    stage_metadata: dict[str, Any] = {
        "search_dimension": plan.dimension,
        "total_groups": plan.total_groups,
    }
    if config["project"].get("study_type") == CAPACITY_TWO_ABLATION_STUDY_TYPE:
        if type(target_groups) is not int:
            raise RuntimeError("capacity-two grouping requires one integer target")
        invariant_summary = _capacity_two_invariant_summary(plan, target_groups=target_groups)
        implementation_path = Path(__file__).with_name("grouping.py")
        grouping_payload["configuration_source"].update(
            {
                key: config["grouping"][key]
                for key in (
                    "strategy",
                    "pairing_algorithm",
                    "networkx_version",
                    "max_group_size",
                    "distance_encoding",
                    "tie_break",
                    "decision_provenance",
                )
            }
        )
        grouping_payload["configuration_source"].update(
            {
                "input_statistics": {
                    "path": "statistics/expert_statistics.npz",
                    "sha256": sha256_file(stats_path),
                },
                "implementation": {
                    "path": "src/dsai6707/grouping.py",
                    "sha256": sha256_file(implementation_path),
                },
            }
        )
        grouping_payload["capacity_two_invariants"] = invariant_summary
        stage_metadata["capacity_two_invariants"] = invariant_summary
    output.mkdir(parents=True, exist_ok=True)
    grouping_path = output / "grouping.json"
    write_json(grouping_path, grouping_payload)
    write_stage_manifest(
        output / "stage_manifest.json",
        stage="fixed_grouping",
        inputs={
            "resolved_config": ArtifactRef.capture(
                run_dir / "config.resolved.yaml", relative_to=run_dir
            ),
            "data_stage": ArtifactRef.capture(
                run_dir / "data" / "stage_manifest.json", relative_to=run_dir
            ),
            "statistics_stage": ArtifactRef.capture(
                run_dir / "statistics" / "stage_manifest.json", relative_to=run_dir
            ),
            "statistics": ArtifactRef.capture(stats_path, relative_to=run_dir),
            "token_manifest": ArtifactRef.capture(token_manifest_path, relative_to=run_dir),
        },
        outputs={"grouping": ArtifactRef.capture(grouping_path, relative_to=run_dir)},
        metadata=stage_metadata,
    )
    print(
        f"fixed grouping saved: {plan.total_groups} physical experts, "
        f"search dimension D={plan.dimension}"
    )


def _load_plan(run_dir: Path) -> GroupingPlan:
    path = run_dir / "grouping" / "grouping.json"
    if not path.is_file():
        raise FileNotFoundError("run cluster before this stage")
    return GroupingPlan.from_dict(read_json(path))


def command_frequency_baseline(args: argparse.Namespace) -> None:
    config, run_dir = _locked_config(args)
    _verify(
        run_dir,
        "statistics/stage_manifest.json",
        expected_stage="collect_statistics",
    )
    _verify(run_dir, "grouping/stage_manifest.json", expected_stage="fixed_grouping")
    output = _guard_new_stage(run_dir, "baselines/frequency", stage="frequency_baseline")
    plan = _load_plan(run_dir)
    stats_path = run_dir / "statistics" / "expert_statistics.npz"
    with np.load(stats_path, allow_pickle=False) as payload:
        counts = payload["routing_counts"]
    coefficients = frequency_coefficients(
        plan, counts, smoothing=float(config["fusion"]["smoothing"])
    )
    layout = CoefficientLayout(plan)
    vector = layout.encode(coefficients)
    output.mkdir(parents=True, exist_ok=True)
    coefficient_path = output / "coefficients.json"
    write_json(
        coefficient_path,
        layout.to_artifact(vector, method="routing_frequency", seed=None, fitness=None),
    )
    write_stage_manifest(
        output / "stage_manifest.json",
        stage="frequency_baseline",
        inputs={
            "resolved_config": ArtifactRef.capture(
                run_dir / "config.resolved.yaml", relative_to=run_dir
            ),
            "statistics_stage": ArtifactRef.capture(
                run_dir / "statistics" / "stage_manifest.json", relative_to=run_dir
            ),
            "grouping_stage": ArtifactRef.capture(
                run_dir / "grouping" / "stage_manifest.json", relative_to=run_dir
            ),
            "grouping": ArtifactRef.capture(
                run_dir / "grouping" / "grouping.json", relative_to=run_dir
            ),
            "statistics": ArtifactRef.capture(stats_path, relative_to=run_dir),
        },
        outputs={"coefficients": ArtifactRef.capture(coefficient_path, relative_to=run_dir)},
    )
    print(f"frequency baseline saved; D={layout.dimension}")


def _ea_config(config: dict[str, Any]) -> EAConfig:
    search = config["search"]
    return EAConfig(
        population_size=int(search["population_size"]),
        offspring_size=int(search.get("offspring_size", search["population_size"])),
        evaluation_budget=int(search["evaluation_budget"]),
        tournament_size=int(search["tournament_size"]),
        init_sigma=float(search["init_sigma"]),
        mutation_sigma=float(search["mutation_sigma"]),
    )


def _verify_raw_search(
    output: Path,
    *,
    strategy: str,
    seed: int,
    layout: CoefficientLayout,
    population: np.ndarray,
    expected_budget: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, float]:
    """Validate a completed search whose final stage manifest was not yet written."""

    summary = read_json(output / "summary.json")
    trials = read_jsonl(output / "trials.jsonl")
    artifact = read_json(output / "best_coefficients.json")
    if summary.get("schema_version") != "search-summary-v1":
        raise ValueError("raw search summary has an unsupported schema")
    if summary.get("strategy") != strategy or int(summary.get("seed", -1)) != seed:
        raise ValueError("raw search identity does not match the requested finalization")
    evaluations = int(summary.get("evaluations", -1))
    if evaluations != expected_budget or len(trials) != expected_budget:
        raise ValueError("raw search did not consume the locked evaluation budget")
    if summary.get("initial_population_sha256") != population_hash(population):
        raise ValueError("raw search used a different initial population")

    fitnesses: list[float] = []
    vectors: list[np.ndarray] = []
    for expected_index, trial in enumerate(trials):
        if trial.get("schema_version") != "search-trial-v1":
            raise ValueError(f"raw trial {expected_index} has an unsupported schema")
        if int(trial.get("evaluation", -1)) != expected_index:
            raise ValueError("raw trial indices are not contiguous")
        if trial.get("strategy") != strategy or int(trial.get("seed", -1)) != seed:
            raise ValueError("raw trial identity mismatch")
        if trial.get("status") not in ("ok", "failed"):
            raise ValueError("raw trial has no recognized execution status")
        fitness = float(trial.get("fitness_nll", float("nan")))
        vector = np.asarray(trial.get("vector"), dtype=np.float64)
        if not np.isfinite(fitness) or vector.shape != (layout.dimension,):
            raise ValueError("raw trial contains a non-finite fitness or invalid vector")
        if not np.isfinite(vector).all():
            raise ValueError("raw trial contains non-finite coordinates")
        fitnesses.append(fitness)
        vectors.append(vector)

    best_index = min(range(len(fitnesses)), key=lambda index: (fitnesses[index], index))
    best_fitness = fitnesses[best_index]
    best_vector = np.asarray(artifact.get("vector"), dtype=np.float64)
    if int(summary.get("best_evaluation", -1)) != best_index or not np.isclose(
        float(summary.get("best_fitness_nll", float("nan"))),
        best_fitness,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("raw summary best result disagrees with trials")
    if best_vector.shape != (layout.dimension,) or not np.array_equal(
        best_vector, vectors[best_index]
    ):
        raise ValueError("raw best coefficients disagree with the best trial")
    if not np.array_equal(best_vector, np.asarray(summary.get("best_vector"), dtype=np.float64)):
        raise ValueError("raw best coefficients disagree with the search summary")
    expected_artifact = layout.to_artifact(
        best_vector,
        method=strategy,
        seed=seed,
        fitness=best_fitness,
    )
    if artifact != expected_artifact:
        raise ValueError("raw best coefficient artifact is internally inconsistent")
    return summary, trials, best_vector, best_fitness


def _is_capacity_two_ea_pilot(config: dict[str, Any]) -> bool:
    project = config.get("project")
    return bool(
        isinstance(project, dict)
        and project.get("study_type") == CAPACITY_TWO_ABLATION_STUDY_TYPE
        and project.get("method_scope") == CAPACITY_TWO_EA_PILOT_SCOPE
    )


def _capacity_two_pilot_selection_payload(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Recompute the locked winner using only calibration/validation objectives."""

    if not _is_capacity_two_ea_pilot(config):
        raise ValueError("winner selection is only defined for the one-seed capacity-two pilot")
    strategies = [str(value) for value in config["search"]["strategies"]]
    seeds = [int(value) for value in config["search"]["seeds"]]
    selection_rule = config.get("pilot_selection")
    if strategies != ["coordinate", "layer_block"] or seeds != [6707]:
        raise ValueError("the one-seed pilot search identity differs from the lock")
    if not isinstance(selection_rule, dict) or selection_rule.get("strategy_order") != strategies:
        raise ValueError("the one-seed pilot selection rule is missing or inconsistent")

    _assert_all_searches_complete(config, run_dir)
    _capacity_two_ea_reference_refs(config, run_dir)
    plan = _load_plan(run_dir)
    layout = CoefficientLayout(plan)
    if layout.dimension != 384:
        raise ValueError("the G16 pilot must have exactly 384 searchable coefficients")

    frequency_path = run_dir / "baselines" / "frequency" / "coefficients.json"
    frequency_artifact = read_json(frequency_path)
    center = np.asarray(frequency_artifact.get("vector"), dtype=np.float64)
    ea_config = _ea_config(config)
    initial_path = run_dir / "search" / "initial_populations" / "seed_6707.npy"
    if not initial_path.is_file():
        raise FileNotFoundError(f"missing shared pilot initial population: {initial_path}")
    population = np.load(initial_path, allow_pickle=False)
    expected_population = make_initial_population(
        center,
        population_size=ea_config.population_size,
        sigma=ea_config.init_sigma,
        seed=6707,
    )
    if not np.array_equal(population, expected_population):
        raise ValueError("pilot initial population differs from the locked generator")
    logical_population_hash = population_hash(population)
    file_population_hash = sha256_file(initial_path)

    candidates: list[dict[str, Any]] = []
    planned_tokens = ea_config.evaluation_budget * int(config["data"]["fitness_tokens"])
    for strategy_index, strategy in enumerate(strategies):
        output = run_dir / "search" / strategy / "seed_6707"
        summary, trials, _best_vector, best_fitness = _verify_raw_search(
            output,
            strategy=strategy,
            seed=6707,
            layout=layout,
            population=population,
            expected_budget=ea_config.evaluation_budget,
        )
        validation_nll = _artifact_finite_float(
            summary.get("validation_nll"), field=f"{strategy} validation_nll"
        )
        calibration_nll = _artifact_finite_float(
            summary.get("best_fitness_nll"), field=f"{strategy} best_fitness_nll"
        )
        validation = summary.get("validation")
        if (
            summary.get("valid_for_main_results") is not True
            or int(summary.get("evaluations", -1)) != ea_config.evaluation_budget
            or int(summary.get("failed_evaluations", -1)) != 0
            or int(summary.get("planned_calibration_loss_tokens", -1)) != planned_tokens
            or int(summary.get("actual_calibration_loss_tokens", -1)) != planned_tokens
            or int(summary.get("validation_loss_tokens", -1))
            != int(config["data"]["validation_tokens"])
            or not isinstance(validation, dict)
            or validation.get("status") != "ok"
            or int(validation.get("loss_tokens", -1)) != int(config["data"]["validation_tokens"])
            or validation_nll
            != _artifact_finite_float(validation.get("nll"), field=f"{strategy} validation.nll")
            or calibration_nll != best_fitness
            or len(trials) != ea_config.evaluation_budget
        ):
            raise ValueError(f"pilot search {strategy}/seed_6707 is not budget-exact and valid")
        if (
            summary.get("initial_population_path") != "search/initial_populations/seed_6707.npy"
            or summary.get("initial_population_sha256") != logical_population_hash
            or summary.get("initial_population_file_sha256") != file_population_hash
        ):
            raise ValueError(f"pilot search {strategy} did not use the shared initial population")
        coefficient_path = output / "best_coefficients.json"
        candidates.append(
            {
                "strategy": strategy,
                "seed": 6707,
                "validation_nll": validation_nll,
                "best_fitness_nll": calibration_nll,
                "strategy_order": strategy_index,
                "evaluations": ea_config.evaluation_budget,
                "summary_path": str((output / "summary.json").relative_to(run_dir)),
                "summary_sha256": sha256_file(output / "summary.json"),
                "coefficients_path": str(coefficient_path.relative_to(run_dir)),
                "coefficients_sha256": sha256_file(coefficient_path),
            }
        )

    winner = min(
        candidates,
        key=lambda row: (
            float(row["validation_nll"]),
            float(row["best_fitness_nll"]),
            int(row["strategy_order"]),
        ),
    )
    return {
        "schema_version": "capacity-two-ea-pilot-selection-v1",
        "study_type": "post_hoc_one_seed_pilot",
        "target_groups_per_layer": 16,
        "max_group_size": 2,
        "seed": 6707,
        "selection_rule": dict(selection_rule),
        "selection_uses_downstream_metrics": False,
        "shared_initial_population": {
            "path": str(initial_path.relative_to(run_dir)),
            "population_sha256": logical_population_hash,
            "file_sha256": file_population_hash,
        },
        "candidates": candidates,
        "winner": {
            "strategy": winner["strategy"],
            "seed": 6707,
            "validation_nll": winner["validation_nll"],
            "best_fitness_nll": winner["best_fitness_nll"],
            "coefficients_path": winner["coefficients_path"],
            "coefficients_sha256": winner["coefficients_sha256"],
        },
        "downstream_evaluation_scope": "winner_only",
    }


def command_select_pilot_winner(args: argparse.Namespace) -> None:
    """Commit the one-seed pilot winner before any EA downstream evaluation."""

    config, run_dir = _locked_config(args)
    if not _is_capacity_two_ea_pilot(config):
        raise ValueError("select-pilot-winner requires the one-seed capacity-two pilot config")
    selection_dir = _guard_new_stage(
        run_dir,
        "pilot_selection",
        stage="pilot_winner_selection",
    )
    payload = _capacity_two_pilot_selection_payload(config, run_dir)
    selection_path = selection_dir / "winner.json"
    write_json(selection_path, payload)
    search_refs = _search_gate_refs(config, run_dir)
    reference_refs = _capacity_two_ea_reference_refs(config, run_dir)
    seed = int(payload["seed"])
    inputs: dict[str, ArtifactRef] = {
        "resolved_config": ArtifactRef.capture(
            run_dir / "config.resolved.yaml", relative_to=run_dir
        ),
        "shared_initial_population": ArtifactRef.capture(
            run_dir / "search" / "initial_populations" / f"seed_{seed}.npy",
            relative_to=run_dir,
        ),
        **search_refs,
        **reference_refs,
    }
    for candidate in payload["candidates"]:
        strategy = str(candidate["strategy"])
        directory = run_dir / "search" / strategy / f"seed_{seed}"
        inputs[f"{strategy}_summary"] = ArtifactRef.capture(
            directory / "summary.json", relative_to=run_dir
        )
        inputs[f"{strategy}_coefficients"] = ArtifactRef.capture(
            directory / "best_coefficients.json", relative_to=run_dir
        )
    write_stage_manifest(
        selection_dir / "stage_manifest.json",
        stage="pilot_winner_selection",
        inputs=inputs,
        outputs={"winner": ArtifactRef.capture(selection_path, relative_to=run_dir)},
        metadata={
            "winner_strategy": payload["winner"]["strategy"],
            "seed": seed,
            "selection_uses_downstream_metrics": False,
            "downstream_evaluation_scope": "winner_only",
        },
    )
    _verify(
        run_dir,
        "pilot_selection/stage_manifest.json",
        expected_stage="pilot_winner_selection",
    )
    print(
        "one-seed pilot winner selected without downstream metrics: "
        f"{payload['winner']['strategy']} seed {seed}"
    )


def _capacity_two_pilot_winner_refs(
    config: dict[str, Any],
    run_dir: Path,
    *,
    method: str,
    seed: int,
) -> dict[str, ArtifactRef]:
    """Fail closed unless the requested downstream method is the locked winner."""

    if not _is_capacity_two_ea_pilot(config):
        return {}
    _verify(
        run_dir,
        "pilot_selection/stage_manifest.json",
        expected_stage="pilot_winner_selection",
    )
    selection_path = run_dir / "pilot_selection" / "winner.json"
    selection = read_json(selection_path)
    expected = _capacity_two_pilot_selection_payload(config, run_dir)
    if selection != expected:
        raise ValueError("pilot winner artifact differs from the locked objective-only selection")
    winner = selection["winner"]
    if method != winner.get("strategy") or seed != int(winner.get("seed", -1)):
        raise ValueError(
            "the one-seed pilot permits downstream evaluation only for the locked winner "
            f"{winner.get('strategy')}/seed_{winner.get('seed')}"
        )
    return {
        "pilot_selection_stage": ArtifactRef.capture(
            run_dir / "pilot_selection" / "stage_manifest.json", relative_to=run_dir
        ),
        "pilot_winner": ArtifactRef.capture(selection_path, relative_to=run_dir),
    }


def _validate_search_warmup_log(
    payload: dict[str, Any],
    *,
    strategy: str,
    seed: int,
    expected_loss_tokens: int,
    require_cuda_cost: bool,
) -> list[dict[str, Any]]:
    """Validate every budget-exempt warmup session for one search."""

    if payload.get("schema_version") != "search-warmup-log-v1":
        raise ValueError("search warmup log has an unsupported schema")
    if payload.get("strategy") != strategy or int(payload.get("seed", -1)) != seed:
        raise ValueError("search warmup log identity mismatch")
    if int(payload.get("expected_loss_tokens", -1)) != expected_loss_tokens:
        raise ValueError("search warmup token count differs from the locked protocol")
    if int(payload.get("budget_evaluations_consumed", -1)) != 0:
        raise ValueError("search warmup must not consume the EA evaluation budget")
    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError("search warmup log has no session list")
    for expected_index, session in enumerate(sessions):
        if not isinstance(session, dict) or session.get("schema_version") != (
            "search-warmup-session-v1"
        ):
            raise ValueError("search warmup log contains an invalid session")
        if int(session.get("session_index", -1)) != expected_index:
            raise ValueError("search warmup session indices are not contiguous")
        if (
            int(session.get("budget_evaluations_consumed", -1)) != 0
            or int(session.get("loss_tokens", -1)) != expected_loss_tokens
        ):
            raise ValueError("search warmup session has invalid budget or token accounting")
        wall_seconds = float(session.get("wall_seconds", float("nan")))
        if not np.isfinite(wall_seconds) or wall_seconds < 0.0:
            raise ValueError("search warmup session has invalid wall time")
        gpu_seconds = session.get("gpu_seconds")
        peak = session.get("peak_device_memory_bytes")
        if require_cuda_cost and (gpu_seconds is None or peak is None):
            raise ValueError("formal CUDA warmup is missing GPU time or peak memory")
        if gpu_seconds is not None and (
            not np.isfinite(float(gpu_seconds)) or float(gpu_seconds) < 0.0
        ):
            raise ValueError("search warmup session has invalid GPU time")
        if peak is not None and int(peak) <= 0:
            raise ValueError("search warmup session has invalid peak device memory")
    return sessions


def _record_search_warmup(
    output: Path,
    *,
    strategy: str,
    seed: int,
    expected_loss_tokens: int,
    require_cuda_cost: bool,
    session: dict[str, Any],
) -> dict[str, Any]:
    """Append one complete warmup session with an atomic whole-log rewrite."""

    path = output / "warmup.json"
    if path.is_file():
        payload = read_json(path)
        sessions = _validate_search_warmup_log(
            payload,
            strategy=strategy,
            seed=seed,
            expected_loss_tokens=expected_loss_tokens,
            require_cuda_cost=require_cuda_cost,
        )
    else:
        payload = {
            "schema_version": "search-warmup-log-v1",
            "strategy": strategy,
            "seed": seed,
            "expected_loss_tokens": expected_loss_tokens,
            "budget_evaluations_consumed": 0,
            "sessions": [],
        }
        sessions = payload["sessions"]
    normalized = {**session, "session_index": len(sessions)}
    sessions.append(normalized)
    _validate_search_warmup_log(
        payload,
        strategy=strategy,
        seed=seed,
        expected_loss_tokens=expected_loss_tokens,
        require_cuda_cost=require_cuda_cost,
    )
    write_json(path, payload)
    return payload


def _validate_search_environment(payload: dict[str, Any], *, require_cuda: bool) -> None:
    if not isinstance(payload.get("packages"), dict):
        raise ValueError("search environment is missing package versions")
    accelerator = payload.get("accelerator")
    if not isinstance(accelerator, dict):
        raise ValueError("search environment is missing accelerator metadata")
    if not require_cuda:
        return
    cublas = accelerator.get("cublas_workspace_config")
    if (
        accelerator.get("cuda_available") is not True
        or accelerator.get("deterministic_algorithms") is not True
        or accelerator.get("cudnn_deterministic") is not True
        or not isinstance(accelerator.get("devices"), list)
        or len(accelerator["devices"]) != 1
        or not isinstance(cublas, dict)
        or cublas.get("valid") is not True
        or cublas.get("verified_before_cuda_initialization_in_process") is not True
    ):
        raise ValueError(
            "formal search environment does not prove exactly one visible GPU and "
            "deterministic CUDA execution"
        )


def _environment_fingerprint(payload: dict[str, Any]) -> str:
    accelerator = payload["accelerator"]
    cublas = accelerator.get("cublas_workspace_config", {})
    numerical_identity = {
        "python": payload.get("python"),
        "platform": payload.get("platform"),
        "packages": payload.get("packages"),
        "accelerator": {
            key: accelerator.get(key)
            for key in (
                "cuda_available",
                "cuda_runtime",
                "cudnn",
                "driver_version",
                "devices",
                "deterministic_algorithms",
                "cudnn_deterministic",
                "cudnn_benchmark",
            )
        },
        "cublas_workspace_config": {
            "value": cublas.get("value"),
            "valid": cublas.get("valid"),
            "verified_before_cuda_initialization_in_process": cublas.get(
                "verified_before_cuda_initialization_in_process"
            ),
        },
    }
    encoded = json.dumps(
        numerical_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_search_environment(
    output: Path,
    *,
    strategy: str,
    seed: int,
    snapshot: dict[str, Any],
    require_cuda: bool,
) -> dict[str, Any]:
    """Append an invocation environment and reject numerical-environment drift."""

    _validate_search_environment(snapshot, require_cuda=require_cuda)
    fingerprint = _environment_fingerprint(snapshot)
    path = output / "environment.json"
    if path.is_file():
        payload = read_json(path)
        if (
            payload.get("schema_version") != "search-environment-log-v1"
            or payload.get("strategy") != strategy
            or int(payload.get("seed", -1)) != seed
        ):
            raise ValueError("search environment log identity mismatch")
        if payload.get("numerical_environment_sha256") != fingerprint:
            raise ValueError("search cannot resume in a different numerical environment")
        sessions = payload.get("sessions")
        if not isinstance(sessions, list):
            raise ValueError("search environment log has no sessions")
    else:
        payload = {
            "schema_version": "search-environment-log-v1",
            "strategy": strategy,
            "seed": seed,
            "numerical_environment_sha256": fingerprint,
            "sessions": [],
        }
        sessions = payload["sessions"]
    sessions.append(
        {
            "session_index": len(sessions),
            "numerical_environment_sha256": fingerprint,
            "snapshot": snapshot,
        }
    )
    write_json(path, payload)
    return payload


def command_search(args: argparse.Namespace) -> None:
    config, run_dir = _locked_config(args)
    project = config["project"]
    if project.get("study_type") == CAPACITY_TWO_ABLATION_STUDY_TYPE and project.get(
        "method_scope"
    ) not in {"all_methods", CAPACITY_TWO_EA_PILOT_SCOPE}:
        raise ValueError(
            "this capacity-two run is locked to the frequency baseline; use the "
            "separate EA extension or one-seed pilot config/run"
        )
    if _is_capacity_two_ea_pilot(config):
        # The pilot protocol fixes one serial stream, coordinate then layer-block.
        # A shared lock prevents the two strategy processes from occupying the
        # same physical GPU concurrently even if both commands are launched.
        with _exclusive_search_lock(run_dir, strategy="pilot_serial", seed=int(args.seed)):
            if args.strategy == "layer_block":
                _verify(
                    run_dir,
                    f"search/coordinate/seed_{int(args.seed)}/stage_manifest.json",
                    expected_stage="evolutionary_search",
                )
            _command_search_locked(args, config=config, run_dir=run_dir)
        return
    with _exclusive_search_lock(run_dir, strategy=str(args.strategy), seed=int(args.seed)):
        _command_search_locked(args, config=config, run_dir=run_dir)


def _command_search_locked(
    args: argparse.Namespace, *, config: dict[str, Any], run_dir: Path
) -> None:
    reference_refs = _capacity_two_ea_reference_refs(config, run_dir)
    from .data import load_token_bundle
    from .granite import CalibrationNLL, GraniteFusionController, load_granite_model

    _verify(run_dir, "data/stage_manifest.json", expected_stage="prepare_data")
    _verify(run_dir, "grouping/stage_manifest.json", expected_stage="fixed_grouping")
    _verify(
        run_dir,
        "baselines/frequency/stage_manifest.json",
        expected_stage="frequency_baseline",
    )
    if args.strategy not in config["search"]["strategies"]:
        raise ValueError("strategy is not pre-registered in the resolved configuration")
    if int(args.seed) not in [int(value) for value in config["search"]["seeds"]]:
        raise ValueError("seed is not pre-registered in the resolved configuration")
    plan = _load_plan(run_dir)
    layout = CoefficientLayout(plan)
    coefficient_path = run_dir / "baselines" / "frequency" / "coefficients.json"
    if not coefficient_path.is_file():
        raise FileNotFoundError("run frequency-baseline before search")
    frequency_artifact = read_json(coefficient_path)
    center = np.asarray(frequency_artifact["vector"], dtype=np.float64)
    ea_config = _ea_config(config)

    initial_dir = run_dir / "search" / "initial_populations"
    initial_dir.mkdir(parents=True, exist_ok=True)
    initial_path = initial_dir / f"seed_{int(args.seed)}.npy"
    expected = make_initial_population(
        center,
        population_size=ea_config.population_size,
        sigma=ea_config.init_sigma,
        seed=int(args.seed),
    )
    if initial_path.exists():
        population = np.load(initial_path, allow_pickle=False)
        if not np.array_equal(population, expected):
            raise ValueError("saved initial population does not match the locked generator")
    else:
        _write_npy_atomic(initial_path, expected)
        population = expected

    output = run_dir / "search" / args.strategy / f"seed_{int(args.seed)}"
    stage_path = output / "stage_manifest.json"
    if stage_path.exists():
        _verify(
            run_dir,
            str(stage_path.relative_to(run_dir)),
            expected_stage="evolutionary_search",
        )
        raise ValueError(f"refusing to overwrite finalized search: {output}")

    raw_paths = {
        "summary": output / "summary.json",
        "trials": output / "trials.jsonl",
        "best": output / "best_coefficients.json",
        "execution": output / "execution.json",
    }
    resume_state_path = output / "resume_state.npz"
    present = {name: path.is_file() for name, path in raw_paths.items()}
    if any(present.values()) and not all(present.values()) and not resume_state_path.is_file():
        raise ValueError(
            f"incomplete raw search artifacts in {output}; preserve the directory for audit "
            "and use a new run directory"
        )
    resume_finalization = all(present.values())

    bundle = load_token_bundle(run_dir / "data" / "tokens.npz")
    model, _ = load_granite_model(config["model"])
    controller = GraniteFusionController(
        model,
        plan,
        layout.decode(center),
        source_device=str(config["fusion"].get("source_weight_device", "model")),
    )
    require_cuda_cost = str(config["model"].get("device")) == "cuda"
    environment_path = output / "environment.json"
    output.mkdir(parents=True, exist_ok=True)
    search_environment = environment_snapshot()
    environment_log = _record_search_environment(
        output,
        strategy=args.strategy,
        seed=int(args.seed),
        snapshot=search_environment,
        require_cuda=require_cuda_cost,
    )
    warmup_tokens = int(config["search"]["warmup_tokens"])
    warmup_objective = CalibrationNLL(
        controller,
        layout,
        bundle.fitness,
        loss_tokens=warmup_tokens,
        micro_batch_size=int(config["search"].get("micro_batch_size", 1)),
    )
    warmup_session = run_objective_warmup(
        warmup_objective,
        center,
        expected_loss_tokens=warmup_tokens,
    )
    warmup_log = _record_search_warmup(
        output,
        strategy=args.strategy,
        seed=int(args.seed),
        expected_loss_tokens=warmup_tokens,
        require_cuda_cost=require_cuda_cost,
        session=warmup_session,
    )
    if resume_finalization:
        recovery = read_json(output / "execution.json")
        if (
            recovery.get("schema_version") != "ea-recovery-v1"
            or recovery.get("strategy") != args.strategy
            or int(recovery.get("seed", -1)) != int(args.seed)
            or int(recovery.get("evaluations", -1)) != ea_config.evaluation_budget
        ):
            raise ValueError("search execution metadata is missing or inconsistent")
        summary, trials, best_vector, best_fitness = _verify_raw_search(
            output,
            strategy=args.strategy,
            seed=int(args.seed),
            layout=layout,
            population=population,
            expected_budget=ea_config.evaluation_budget,
        )
        evaluations = int(summary["evaluations"])
    else:
        objective = CalibrationNLL(
            controller,
            layout,
            bundle.fitness,
            loss_tokens=int(config["data"]["fitness_tokens"]),
            micro_batch_size=int(config["search"].get("micro_batch_size", 1)),
        )
        result = run_evolution(
            objective,
            layout,
            population,
            strategy=args.strategy,
            config=ea_config,
            seed=int(args.seed),
            output_dir=output,
        )
        trials = list(result.trials)
        evaluations = result.evaluations
        best_vector = result.best_vector
        best_fitness = result.best_fitness
        summary = result.summary()
        recovery = result.recovery

    validation_objective = CalibrationNLL(
        controller,
        layout,
        bundle.validation,
        loss_tokens=int(config["data"]["validation_tokens"]),
        micro_batch_size=int(config["search"].get("micro_batch_size", 1)),
    )
    validation = validation_objective(best_vector)
    failed = sum(row.get("status") != "ok" for row in trials)
    actual_loss_tokens = sum(int(row.get("loss_tokens") or 0) for row in trials)
    planned_loss_tokens = evaluations * int(config["data"]["fitness_tokens"])
    validation_ok = bool(
        validation.metadata.get("status") == "ok" and np.isfinite(validation.value)
    )
    summary.update(
        {
            "validation_nll": validation.value,
            "validation_loss_tokens": validation.metadata["loss_tokens"],
            "validation": {
                "nll": validation.value,
                **validation.metadata,
            },
            "planned_calibration_loss_tokens": planned_loss_tokens,
            "actual_calibration_loss_tokens": actual_loss_tokens,
            "failed_evaluations": failed,
            "valid_for_main_results": failed == 0
            and actual_loss_tokens == planned_loss_tokens
            and validation_ok,
            "initial_population_path": str(initial_path.relative_to(run_dir)),
            "initial_population_file_sha256": sha256_file(initial_path),
        }
    )
    write_json(output / "summary.json", summary)
    if not validation_ok:
        raise RuntimeError(
            "validation evaluation failed; the completed search was preserved and this "
            "command can be rerun to finalize it"
        )
    # The manifest is the final commit marker; no temporary recovery state may
    # remain once downstream stages are allowed to observe it.
    resume_state_path.unlink(missing_ok=True)
    write_stage_manifest(
        output / "stage_manifest.json",
        stage="evolutionary_search",
        inputs={
            "resolved_config": ArtifactRef.capture(
                run_dir / "config.resolved.yaml", relative_to=run_dir
            ),
            "data_stage": ArtifactRef.capture(
                run_dir / "data" / "stage_manifest.json", relative_to=run_dir
            ),
            "grouping_stage": ArtifactRef.capture(
                run_dir / "grouping" / "stage_manifest.json", relative_to=run_dir
            ),
            "frequency_stage": ArtifactRef.capture(
                run_dir / "baselines" / "frequency" / "stage_manifest.json",
                relative_to=run_dir,
            ),
            "tokens": ArtifactRef.capture(run_dir / "data" / "tokens.npz", relative_to=run_dir),
            "grouping": ArtifactRef.capture(
                run_dir / "grouping" / "grouping.json", relative_to=run_dir
            ),
            "frequency_coefficients": ArtifactRef.capture(coefficient_path, relative_to=run_dir),
            "initial_population": ArtifactRef.capture(initial_path, relative_to=run_dir),
            **reference_refs,
        },
        outputs={
            "trials": ArtifactRef.capture(output / "trials.jsonl", relative_to=run_dir),
            "best_coefficients": ArtifactRef.capture(
                output / "best_coefficients.json", relative_to=run_dir
            ),
            "summary": ArtifactRef.capture(output / "summary.json", relative_to=run_dir),
            "warmup": ArtifactRef.capture(output / "warmup.json", relative_to=run_dir),
            "execution": ArtifactRef.capture(output / "execution.json", relative_to=run_dir),
            "environment": ArtifactRef.capture(environment_path, relative_to=run_dir),
        },
        metadata={
            "strategy": args.strategy,
            "seed": int(args.seed),
            "valid_for_main_results": summary["valid_for_main_results"],
            "warmup_sessions": len(warmup_log["sessions"]),
            "environment_sessions": len(environment_log["sessions"]),
            "numerical_environment_sha256": environment_log["numerical_environment_sha256"],
            "warmup_budget_evaluations_consumed": 0,
            "recovery": recovery,
        },
    )
    print(
        f"{args.strategy} seed {args.seed}: {evaluations} evaluations, "
        f"best calibration NLL={best_fitness:.6f}, "
        f"validation NLL={validation.value:.6f}"
    )


def command_evaluate(args: argparse.Namespace) -> None:
    from .evaluation import evaluate_zero_shot
    from .granite import GraniteFusionController, load_granite_model

    config, run_dir = _locked_config(args)
    # Even a limited test subset reveals held-out information, so canonical
    # evaluations remain gated behind all configured finalized searches. The only
    # exceptions are fixed frequency baselines in explicitly post-hoc ablations,
    # whose identities are recorded in their resolved configurations.
    requires_search_gate = _requires_search_gate(config, args.method)
    if requires_search_gate:
        _assert_all_searches_complete(config, run_dir)
    seed: int | None = None
    coefficient_path: Path | None = None
    plan: GroupingPlan | None = None
    layout: CoefficientLayout | None = None
    input_refs: dict[str, ArtifactRef] = {
        "resolved_config": ArtifactRef.capture(
            run_dir / "config.resolved.yaml", relative_to=run_dir
        ),
    }
    if requires_search_gate:
        input_refs.update(_search_gate_refs(config, run_dir))
    if args.method in ("original", "frequency") and args.seed is not None:
        raise ValueError("--seed is only valid for EA methods")
    if args.method != "original":
        _verify(run_dir, "grouping/stage_manifest.json", expected_stage="fixed_grouping")
        plan = _load_plan(run_dir)
        layout = CoefficientLayout(plan)
        input_refs["grouping_stage"] = ArtifactRef.capture(
            run_dir / "grouping" / "stage_manifest.json", relative_to=run_dir
        )
        input_refs["grouping"] = ArtifactRef.capture(
            run_dir / "grouping" / "grouping.json", relative_to=run_dir
        )
        if args.method == "frequency":
            _verify(
                run_dir,
                "baselines/frequency/stage_manifest.json",
                expected_stage="frequency_baseline",
            )
            coefficient_path = run_dir / "baselines" / "frequency" / "coefficients.json"
            input_refs["coefficient_stage"] = ArtifactRef.capture(
                run_dir / "baselines" / "frequency" / "stage_manifest.json",
                relative_to=run_dir,
            )
        else:
            if args.seed is None:
                raise ValueError("--seed is required for EA methods")
            seed = int(args.seed)
            if seed not in [int(value) for value in config["search"]["seeds"]]:
                raise ValueError("seed is not pre-registered in the resolved configuration")
            input_refs.update(
                _capacity_two_pilot_winner_refs(
                    config,
                    run_dir,
                    method=str(args.method),
                    seed=seed,
                )
            )
            _verify(
                run_dir,
                f"search/{args.method}/seed_{seed}/stage_manifest.json",
                expected_stage="evolutionary_search",
            )
            coefficient_path = (
                run_dir / "search" / args.method / f"seed_{seed}" / "best_coefficients.json"
            )
            input_refs["coefficient_stage"] = ArtifactRef.capture(
                run_dir / "search" / args.method / f"seed_{seed}" / "stage_manifest.json",
                relative_to=run_dir,
            )
        artifact = read_json(coefficient_path)
        vector = np.asarray(artifact["vector"], dtype=np.float64)
        input_refs["coefficients"] = ArtifactRef.capture(coefficient_path, relative_to=run_dir)
    suffix = "deterministic" if seed is None else f"seed_{seed}"
    if args.limit is None:
        output = run_dir / "evaluation" / args.method / f"{suffix}.json"
    else:
        limit_label = str(args.limit).replace(".", "_")
        output = (
            run_dir
            / "evaluation"
            / "debug"
            / f"limit_{limit_label}"
            / args.method
            / f"{suffix}.json"
        )
    output_manifest = output.with_suffix(".manifest.json")
    if output_manifest.exists():
        verify_stage_manifest(output_manifest, run_dir=run_dir)
        raise ValueError(f"refusing to overwrite finalized evaluation: {output}")
    if output.exists():
        existing = read_json(output)
        _validate_evaluation_artifact(
            existing,
            config=config,
            method=args.method,
            seed=seed,
            limit=args.limit,
        )
        write_stage_manifest(
            output_manifest,
            stage="downstream_evaluation",
            inputs=input_refs,
            outputs={"evaluation": ArtifactRef.capture(output, relative_to=run_dir)},
            metadata={"method": args.method, "seed": seed, "limit": args.limit},
        )
        print(f"finalized existing downstream evaluation in {output.resolve()}")
        return

    model, tokenizer = load_granite_model(config["model"])
    if coefficient_path is not None:
        assert plan is not None and layout is not None
        GraniteFusionController(
            model,
            plan,
            layout.decode(vector),
            source_device=str(config["fusion"].get("source_weight_device", "model")),
        )
    evaluation = evaluate_zero_shot(
        model,
        tokenizer,
        config["evaluation"],
        output_path=output,
        method=args.method,
        seed=seed,
        limit=args.limit,
    )
    _validate_evaluation_artifact(
        evaluation,
        config=config,
        method=args.method,
        seed=seed,
        limit=args.limit,
    )
    write_stage_manifest(
        output_manifest,
        stage="downstream_evaluation",
        inputs=input_refs,
        outputs={"evaluation": ArtifactRef.capture(output, relative_to=run_dir)},
        metadata={"method": args.method, "seed": seed, "limit": args.limit},
    )
    print(f"downstream evaluation saved in {output.resolve()}")


def command_evaluate_supplemental(args: argparse.Namespace) -> None:
    """Run the isolated post-hoc WinoGrande/PIQA full-validation protocol."""

    from .evaluation import evaluate_zero_shot
    from .granite import GraniteFusionController, load_granite_model

    config, run_dir = _locked_config(args)
    supplemental = load_supplemental_evaluation_config(args.supplemental_config)
    lock_supplemental_evaluation_config(supplemental, run_dir)
    ea_methods = ("coordinate", "layer_block")
    if args.method not in ("original", "frequency", *ea_methods):
        raise ValueError("unsupported supplemental evaluation method")
    requested_seed = getattr(args, "seed", None)
    if args.method in ("original", "frequency") and requested_seed is not None:
        raise ValueError("--seed is only valid for EA methods")
    seed: int | None = None
    if args.method in ea_methods:
        if requested_seed is None:
            raise ValueError("--seed is required for EA methods")
        seed = int(requested_seed)
        if seed not in [int(value) for value in config["search"]["seeds"]]:
            raise ValueError("seed is not pre-registered in the resolved configuration")
        _assert_all_searches_complete(config, run_dir)

    input_refs: dict[str, ArtifactRef] = {
        "resolved_config": ArtifactRef.capture(
            run_dir / "config.resolved.yaml", relative_to=run_dir
        ),
        "supplemental_protocol": ArtifactRef.capture(
            run_dir / "supplemental_evaluation" / "protocol.resolved.yaml",
            relative_to=run_dir,
        ),
        "supplemental_protocol_lock": ArtifactRef.capture(
            run_dir / "supplemental_evaluation" / "protocol.lock.json",
            relative_to=run_dir,
        ),
    }
    if args.method in ea_methods:
        input_refs.update(_search_gate_refs(config, run_dir))
    coefficient_path: Path | None = None
    plan: GroupingPlan | None = None
    layout: CoefficientLayout | None = None
    vector: np.ndarray | None = None
    if args.method != "original":
        _verify(run_dir, "grouping/stage_manifest.json", expected_stage="fixed_grouping")
        plan = _load_plan(run_dir)
        layout = CoefficientLayout(plan)
        if args.method == "frequency":
            _verify(
                run_dir,
                "baselines/frequency/stage_manifest.json",
                expected_stage="frequency_baseline",
            )
            coefficient_path = run_dir / "baselines" / "frequency" / "coefficients.json"
            coefficient_stage_path = run_dir / "baselines" / "frequency" / "stage_manifest.json"
        else:
            assert seed is not None
            input_refs.update(
                _capacity_two_pilot_winner_refs(
                    config,
                    run_dir,
                    method=str(args.method),
                    seed=seed,
                )
            )
            _verify(
                run_dir,
                f"search/{args.method}/seed_{seed}/stage_manifest.json",
                expected_stage="evolutionary_search",
            )
            coefficient_path = (
                run_dir / "search" / args.method / f"seed_{seed}" / "best_coefficients.json"
            )
            coefficient_stage_path = (
                run_dir / "search" / args.method / f"seed_{seed}" / "stage_manifest.json"
            )
        coefficient_artifact = read_json(coefficient_path)
        vector = np.asarray(coefficient_artifact["vector"], dtype=np.float64)
        input_refs.update(
            {
                "grouping_stage": ArtifactRef.capture(
                    run_dir / "grouping" / "stage_manifest.json", relative_to=run_dir
                ),
                "grouping": ArtifactRef.capture(
                    run_dir / "grouping" / "grouping.json", relative_to=run_dir
                ),
                "coefficient_stage": ArtifactRef.capture(
                    coefficient_stage_path,
                    relative_to=run_dir,
                ),
                "coefficients": ArtifactRef.capture(coefficient_path, relative_to=run_dir),
            }
        )

    suffix = "deterministic" if seed is None else f"seed_{seed}"
    output = run_dir / "supplemental_evaluation" / args.method / f"{suffix}.json"
    output_manifest = output.with_suffix(".manifest.json")
    if output_manifest.exists():
        manifest = verify_stage_manifest(output_manifest, run_dir=run_dir)
        if manifest.get("stage") != "supplemental_downstream_evaluation":
            raise ValueError("supplemental evaluation manifest has an unexpected stage")
        raise ValueError(f"refusing to overwrite finalized supplemental evaluation: {output}")
    if output.exists():
        existing = read_json(output)
        _validate_supplemental_evaluation_artifact(
            existing,
            supplemental_config=supplemental,
            method=args.method,
            seed=seed,
        )
        write_stage_manifest(
            output_manifest,
            stage="supplemental_downstream_evaluation",
            inputs=input_refs,
            outputs={"evaluation": ArtifactRef.capture(output, relative_to=run_dir)},
            metadata={
                "method": args.method,
                "seed": seed,
                "limit": None,
                "protocol_id": supplemental["protocol_id"],
                "post_hoc": True,
                "tasks": supplemental["evaluation"]["tasks"],
                "expected_samples": supplemental["evaluation"]["expected_samples"],
            },
        )
        print(f"finalized existing supplemental evaluation in {output.resolve()}")
        return

    model, tokenizer = load_granite_model(config["model"])
    if coefficient_path is not None:
        assert plan is not None and layout is not None and vector is not None
        GraniteFusionController(
            model,
            plan,
            layout.decode(vector),
            source_device=str(config["fusion"].get("source_weight_device", "model")),
        )
    evaluation = evaluate_zero_shot(
        model,
        tokenizer,
        supplemental["evaluation"],
        output_path=output,
        method=args.method,
        seed=seed,
        limit=None,
    )
    _validate_supplemental_evaluation_artifact(
        evaluation,
        supplemental_config=supplemental,
        method=args.method,
        seed=seed,
    )
    write_stage_manifest(
        output_manifest,
        stage="supplemental_downstream_evaluation",
        inputs=input_refs,
        outputs={"evaluation": ArtifactRef.capture(output, relative_to=run_dir)},
        metadata={
            "method": args.method,
            "seed": seed,
            "limit": None,
            "protocol_id": supplemental["protocol_id"],
            "post_hoc": True,
            "tasks": supplemental["evaluation"]["tasks"],
            "expected_samples": supplemental["evaluation"]["expected_samples"],
        },
    )
    print(f"supplemental evaluation saved in {output.resolve()}")


def command_export(args: argparse.Namespace) -> None:
    import gc

    import torch

    from .granite import (
        GraniteFusionController,
        load_granite_model,
        load_grouped_checkpoint,
    )

    config, run_dir = _locked_config(args)
    _verify(run_dir, "grouping/stage_manifest.json", expected_stage="fixed_grouping")
    plan = _load_plan(run_dir)
    layout = CoefficientLayout(plan)
    seed: int | None = None
    coefficient_stage_path: Path
    winner_refs: dict[str, ArtifactRef] = {}
    if args.method == "frequency":
        if args.seed is not None:
            raise ValueError("--seed is only valid for EA methods")
        _verify(
            run_dir,
            "baselines/frequency/stage_manifest.json",
            expected_stage="frequency_baseline",
        )
        coefficient_path = run_dir / "baselines" / "frequency" / "coefficients.json"
        coefficient_stage_path = run_dir / "baselines" / "frequency" / "stage_manifest.json"
    else:
        if args.seed is None:
            raise ValueError("--seed is required for EA methods")
        seed = int(args.seed)
        if seed not in [int(value) for value in config["search"]["seeds"]]:
            raise ValueError("seed is not pre-registered in the resolved configuration")
        winner_refs = _capacity_two_pilot_winner_refs(
            config,
            run_dir,
            method=str(args.method),
            seed=seed,
        )
        _verify(
            run_dir,
            f"search/{args.method}/seed_{seed}/stage_manifest.json",
            expected_stage="evolutionary_search",
        )
        coefficient_path = (
            run_dir / "search" / args.method / f"seed_{seed}" / "best_coefficients.json"
        )
        coefficient_stage_path = (
            run_dir / "search" / args.method / f"seed_{seed}" / "stage_manifest.json"
        )
    coefficient_artifact = read_json(coefficient_path)
    vector = np.asarray(coefficient_artifact["vector"], dtype=np.float64)
    expected_method = "routing_frequency" if args.method == "frequency" else args.method
    expected_artifact = layout.to_artifact(
        vector,
        method=expected_method,
        seed=seed,
        fitness=coefficient_artifact.get("fitness_nll"),
    )
    if coefficient_artifact != expected_artifact:
        raise ValueError("coefficient artifact is internally inconsistent")
    suffix = "deterministic" if seed is None else f"seed_{seed}"
    output = Path(args.output) if args.output else run_dir / "checkpoints" / args.method / suffix
    output = output.resolve()
    export_stage = output / "stage_manifest.json"
    if export_stage.exists():
        manifest = verify_stage_manifest(export_stage, run_dir=run_dir)
        if manifest.get("stage") != "checkpoint_export":
            raise ValueError("unexpected stage in checkpoint export manifest")
        raise ValueError(f"refusing to overwrite finalized checkpoint: {output}")
    if output.exists() and not (output / "manifest.json").is_file():
        raise ValueError(f"incomplete checkpoint directory cannot be resumed: {output}")

    model, tokenizer = load_granite_model(config["model"])
    controller = GraniteFusionController(
        model,
        plan,
        layout.decode(vector),
        source_device=str(config["fusion"].get("source_weight_device", "model")),
    )
    device = next(model.parameters()).device
    vocab_size = int(model.config.vocab_size)
    probe = torch.tensor(
        [
            [int(getattr(model.config, "bos_token_id", 1) or 1)]
            + [index % vocab_size for index in range(2, 9)]
        ],
        dtype=torch.long,
        device=device,
    )
    with torch.inference_mode():
        expected_logits = model(input_ids=probe, use_cache=False).logits.float().cpu()
    if not output.exists():
        controller.export_fused_experts(
            output,
            coefficient_artifact=coefficient_artifact,
            tokenizer=tokenizer,
        )

    del controller
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    loaded, _ = load_grouped_checkpoint(output, device=str(config["model"]["device"]))
    with torch.inference_mode():
        actual_logits = loaded(
            input_ids=probe.to(next(loaded.parameters()).device), use_cache=False
        )
        actual_logits = actual_logits.logits.float().cpu()
    torch.testing.assert_close(actual_logits, expected_logits, rtol=1.0e-5, atol=1.0e-5)
    difference = (actual_logits - expected_logits).abs()
    roundtrip_path = output / "roundtrip_validation.json"
    write_json(
        roundtrip_path,
        {
            "schema_version": "checkpoint-roundtrip-v1",
            "probe_token_ids": probe.cpu().tolist(),
            "max_abs_logit_difference": float(difference.max().item()),
            "mean_abs_logit_difference": float(difference.mean().item()),
            "rtol": 1.0e-5,
            "atol": 1.0e-5,
            "passed": True,
        },
    )
    del loaded
    if device.type == "cuda":
        torch.cuda.empty_cache()

    internal_manifest = read_json(output / "manifest.json")
    model_entry = internal_manifest["files"]["model.safetensors"]
    write_stage_manifest(
        export_stage,
        stage="checkpoint_export",
        inputs={
            "resolved_config": ArtifactRef.capture(
                run_dir / "config.resolved.yaml", relative_to=run_dir
            ),
            "grouping_stage": ArtifactRef.capture(
                run_dir / "grouping" / "stage_manifest.json", relative_to=run_dir
            ),
            "coefficient_stage": ArtifactRef.capture(coefficient_stage_path, relative_to=run_dir),
            "grouping": ArtifactRef.capture(
                run_dir / "grouping" / "grouping.json", relative_to=run_dir
            ),
            "coefficients": ArtifactRef.capture(coefficient_path, relative_to=run_dir),
            **winner_refs,
        },
        outputs={
            "checkpoint_manifest": ArtifactRef.capture(output / "manifest.json"),
            "model": ArtifactRef(
                path=str((output / "model.safetensors").resolve()),
                sha256=str(model_entry["sha256"]),
            ),
            "compression": ArtifactRef.capture(output / "compression.json"),
            "roundtrip": ArtifactRef.capture(roundtrip_path),
        },
        metadata={"method": args.method, "seed": seed, "roundtrip_passed": True},
    )
    print(f"standalone grouped checkpoint saved in {output.resolve()}")


def command_summarize(args: argparse.Namespace) -> None:
    from .summarize import summarize_results

    config, run_dir = _locked_config(args)
    result = summarize_results(config, run_dir)
    print(
        f"formal result summary saved in {(run_dir / 'summary').resolve()}: "
        f"{result['run_rows']} runs, {result['convergence_rows']} convergence rows"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsai6707",
        description="Granite fixed-group expert fusion research pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config", help="validate all locked choices")
    validate.add_argument("--config", default="configs/experiment.yaml")
    validate.set_defaults(function=command_validate)

    smoke = subparsers.add_parser("smoke", help="run the network-free synthetic pipeline")
    smoke.add_argument("--config", default="configs/dry_run.yaml")
    smoke.add_argument("--output", default="results/smoke")
    smoke.set_defaults(function=command_smoke)

    for name, function in (
        ("prepare-data", command_prepare_data),
        ("inspect-model", command_inspect_model),
        ("collect-stats", command_collect_stats),
        ("cluster", command_cluster),
        ("frequency-baseline", command_frequency_baseline),
    ):
        stage = subparsers.add_parser(name)
        stage.add_argument("--config", default="configs/experiment.yaml")
        stage.add_argument("--run-dir", default="results/runs/granite_fixed_group_ea")
        stage.set_defaults(function=function)

    search = subparsers.add_parser("search")
    search.add_argument("--config", default="configs/experiment.yaml")
    search.add_argument("--run-dir", default="results/runs/granite_fixed_group_ea")
    search.add_argument("--strategy", choices=("coordinate", "layer_block"), required=True)
    search.add_argument("--seed", type=int, required=True)
    search.set_defaults(function=command_search)

    select_pilot = subparsers.add_parser(
        "select-pilot-winner",
        help="lock the G16 one-seed EA winner using validation/calibration NLL only",
    )
    select_pilot.add_argument(
        "--config", default="configs/experiment_capacity2_g16_ea_pilot_seed6707.yaml"
    )
    select_pilot.add_argument(
        "--run-dir", default="results/runs/granite_capacity2_g16_ea_pilot_seed6707_v2"
    )
    select_pilot.set_defaults(function=command_select_pilot_winner)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", default="configs/experiment.yaml")
    evaluate.add_argument("--run-dir", default="results/runs/granite_fixed_group_ea")
    evaluate.add_argument(
        "--method",
        choices=("original", "frequency", "coordinate", "layer_block"),
        required=True,
    )
    evaluate.add_argument("--seed", type=int)
    evaluate.add_argument("--limit", type=float)
    evaluate.set_defaults(function=command_evaluate)

    supplemental = subparsers.add_parser(
        "evaluate-supplemental",
        help="run the locked post-hoc WinoGrande/PIQA full-validation protocol",
    )
    supplemental.add_argument("--config", default="configs/experiment.yaml")
    supplemental.add_argument(
        "--supplemental-config", default="configs/supplemental_evaluation.yaml"
    )
    supplemental.add_argument("--run-dir", default="results/runs/granite_fixed_group_ea")
    supplemental.add_argument(
        "--method",
        choices=("original", "frequency", "coordinate", "layer_block"),
        required=True,
    )
    supplemental.add_argument("--seed", type=int)
    supplemental.set_defaults(function=command_evaluate_supplemental)

    export = subparsers.add_parser("export")
    export.add_argument("--config", default="configs/experiment.yaml")
    export.add_argument("--run-dir", default="results/runs/granite_fixed_group_ea")
    export.add_argument(
        "--method", choices=("frequency", "coordinate", "layer_block"), required=True
    )
    export.add_argument("--seed", type=int)
    export.add_argument("--output")
    export.set_defaults(function=command_export)

    summarize = subparsers.add_parser(
        "summarize", help="verify and aggregate all formal search/evaluation results"
    )
    summarize.add_argument("--config", default="configs/experiment.yaml")
    summarize.add_argument("--run-dir", default="results/runs/granite_fixed_group_ea")
    summarize.set_defaults(function=command_summarize)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.function(args)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main(sys.argv[1:])

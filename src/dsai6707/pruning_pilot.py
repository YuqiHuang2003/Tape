"""Auditable, gated single-GPU pilot for post-hoc Granite expert pruning.

The commands in this module deliberately separate the cheap correctness and
language-model checks from downstream evaluation and evolutionary search.  A
later stage may only run after the persisted decision artifact says that every
pre-registered gate needed by that stage passed.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .artifacts import ArtifactRef, verify_stage_manifest, write_stage_manifest
from .config import FORMAL_EVALUATION_PROTOCOL_ID, LOCKED_TASK_DATASETS
from .data import TokenBundle, load_token_bundle
from .evaluation import evaluate_zero_shot
from .granite import inspect_granite_topology, load_granite_model
from .io import (
    CUBLAS_WORKSPACE_CONFIG_ALLOWED,
    CUBLAS_WORKSPACE_CONFIG_NAME,
    environment_snapshot,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from .pruning import ExpertSelectionPlan as RoutingPlan
from .pruning import GranitePruningController
from .pruning_evolution import (
    PruningEAConfig,
    PruningFitnessValue,
    mutate_uniform_swap,
    pruning_population_hash,
    run_pruning_evolution,
)
from .pruning_metrics import (
    ExpertSelectionPlan as ScoredSelectionPlan,
)
from .pruning_metrics import (
    build_expert_selection_plan,
    collect_reap_statistics,
    spearman_rank_correlation,
    token_weighted_full_model_nll,
)


class PruningPilotError(RuntimeError):
    """Raised when a locked pruning-pilot invariant is violated."""


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise PruningPilotError(f"configuration is not a mapping: {path}")
    return value


def _workspace_and_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    config = _load_yaml(config_path)
    workspace = config_path.parent.parent
    if not (workspace / "src" / "dsai6707").is_dir():
        raise PruningPilotError("configuration must live in this DSAI6707 workspace")
    return workspace, config_path, config


def _run_dir(workspace: Path, config: Mapping[str, Any]) -> Path:
    path = (workspace / str(config["project"]["run_dir"])).resolve()
    try:
        path.relative_to(workspace)
    except ValueError as exc:
        raise PruningPilotError("run_dir must stay inside the DSAI6707 workspace") from exc
    return path


def _locked_source_path(workspace: Path, relative: str) -> Path:
    path = (workspace / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _verify_locked_sources(workspace: Path, config: Mapping[str, Any]) -> dict[str, Path]:
    project = config["project"]
    data = config["data"]
    paths = {
        "protocol": _locked_source_path(workspace, str(project["protocol"])),
        "tokens": _locked_source_path(workspace, str(data["token_artifact"])),
        "token_manifest": _locked_source_path(workspace, str(data["token_manifest"])),
        "reference_statistics": _locked_source_path(workspace, str(data["reference_statistics"])),
    }
    expected = {
        "protocol": str(project["protocol_sha256"]),
        "tokens": str(data["token_artifact_sha256"]),
        "token_manifest": str(data["token_manifest_sha256"]),
        "reference_statistics": str(data["reference_statistics_sha256"]),
    }
    for name, path in paths.items():
        actual = sha256_file(path)
        if actual != expected[name]:
            raise PruningPilotError(
                f"locked {name} hash mismatch: expected {expected[name]}, got {actual}"
            )
    return paths


def _implementation_paths(workspace: Path) -> dict[str, Path]:
    """Return the exact implementation files that define a pilot result."""

    names = (
        "pruning_pilot.py",
        "pruning.py",
        "pruning_metrics.py",
        "pruning_evolution.py",
        "evaluation.py",
    )
    return {
        f"implementation_{Path(name).stem}": _locked_source_path(workspace, f"src/dsai6707/{name}")
        for name in names
    }


def _ref(path: Path, run_dir: Path) -> ArtifactRef:
    try:
        return ArtifactRef.capture(path, relative_to=run_dir)
    except ValueError:
        return ArtifactRef.capture(path)


def _verify_stage(run_dir: Path, relative: str, expected: str) -> dict[str, Any]:
    manifest = verify_stage_manifest(run_dir / relative, run_dir=run_dir)
    if manifest.get("stage") != expected:
        raise PruningPilotError(
            f"unexpected stage in {relative}: expected {expected}, got {manifest.get('stage')}"
        )
    return manifest


def _require_new_stage(directory: Path) -> None:
    if directory.exists():
        raise PruningPilotError(f"refusing to overwrite existing stage directory: {directory}")
    directory.mkdir(parents=True)


def _require_one_gpu() -> None:
    workspace_config = os.environ.get(CUBLAS_WORKSPACE_CONFIG_NAME)
    if workspace_config not in CUBLAS_WORKSPACE_CONFIG_ALLOWED:
        raise PruningPilotError(
            f"{CUBLAS_WORKSPACE_CONFIG_NAME} must be set before CUDA probing to one of "
            f"{sorted(CUBLAS_WORKSPACE_CONFIG_ALLOWED)}"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise PruningPilotError(
            "this pilot is locked to CUDA_VISIBLE_DEVICES=0 (exactly one visible GPU)"
        )
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise PruningPilotError("the pilot requires exactly one visible CUDA device")


def _validate_topology(topology: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    pruning = config["pruning"]
    expected_experts = int(pruning["original_experts_per_layer"])
    expected_top_k = int(pruning["top_k"])
    if int(topology["num_experts"]) != expected_experts:
        raise PruningPilotError("checkpoint expert count differs from the locked protocol")
    if int(topology["top_k"]) != expected_top_k:
        raise PruningPilotError("checkpoint top-k differs from the locked protocol")
    retained = int(pruning["retained_experts_per_layer"])
    if not expected_top_k <= retained < expected_experts:
        raise PruningPilotError("invalid retained-expert cardinality")


def _routing_plan_from_mask(mask: np.ndarray) -> RoutingPlan:
    values = np.asarray(mask, dtype=np.bool_)
    if values.ndim != 2:
        raise ValueError("mask must have shape [layers, experts]")
    return RoutingPlan.from_retained(
        [np.flatnonzero(row).tolist() for row in values],
        num_experts=int(values.shape[1]),
    )


def _selection_from_json(path: Path) -> RoutingPlan:
    artifact = read_json(path)
    layers = artifact.get("layers")
    if not isinstance(layers, list):
        raise PruningPilotError(f"invalid selection artifact: {path}")
    retained = [row["retained"] for row in layers]
    return RoutingPlan.from_retained(retained, num_experts=int(artifact["num_experts"]))


def _selection_mask_from_json(path: Path) -> np.ndarray:
    plan = _selection_from_json(path)
    mask = np.zeros((plan.num_layers, plan.num_experts), dtype=np.bool_)
    for layer, retained in enumerate(plan.layers):
        mask[layer, list(retained)] = True
    return mask


def _write_selection(path: Path, selection: ScoredSelectionPlan, *, method: str) -> None:
    write_json(path, selection.to_dict(method=method))


def _forward_identity_snapshot(model: Any, ids: Any) -> dict[str, Any]:
    import torch

    with torch.inference_mode():
        output = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            use_cache=False,
            output_router_logits=True,
        )
    router_logits = tuple(output.router_logits)
    top_k = int(model.config.num_experts_per_tok)
    top_indices = []
    top_gates = []
    for logits in router_logits:
        values, indices = logits.float().topk(top_k, dim=1)
        top_indices.append(indices.detach().cpu())
        top_gates.append(torch.softmax(values, dim=1).detach().cpu())
    return {
        "lm_logits": output.logits.detach().cpu(),
        "router_logits": tuple(value.detach().cpu() for value in router_logits),
        "top_indices": tuple(top_indices),
        "top_gates": tuple(top_gates),
    }


def _compare_identity(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    def sequence_equal(name: str) -> bool:
        return all(
            torch.equal(left, right) for left, right in zip(before[name], after[name], strict=True)
        )

    lm_equal = torch.equal(before["lm_logits"], after["lm_logits"])
    lm_max_abs = float(
        (before["lm_logits"].float() - after["lm_logits"].float()).abs().max().item()
    )
    router_equal = sequence_equal("router_logits")
    router_max_abs = max(
        float((left.float() - right.float()).abs().max().item())
        for left, right in zip(before["router_logits"], after["router_logits"], strict=True)
    )
    return {
        "lm_logits_bit_exact": lm_equal,
        "lm_logits_max_abs": lm_max_abs,
        "router_logits_bit_exact": router_equal,
        "router_logits_max_abs": router_max_abs,
        "router_topk_indices_bit_exact": sequence_equal("top_indices"),
        "router_topk_gates_bit_exact": sequence_equal("top_gates"),
    }


def _parameter_identity(model: Any) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (name, int(parameter.data_ptr()), int(parameter._version))
        for name, parameter in model.named_parameters()
    )


def _nll_pair(
    model: Any,
    bundle: TokenBundle,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    data = config["data"]
    micro = int(config["search"].get("micro_batch_size", 1))
    calibration = token_weighted_full_model_nll(
        model,
        bundle.fitness,
        loss_tokens=int(data["fitness_tokens"]),
        micro_batch_size=micro,
    )
    validation = token_weighted_full_model_nll(
        model,
        bundle.validation,
        loss_tokens=int(data["validation_tokens"]),
        micro_batch_size=micro,
    )
    return {"calibration": calibration.to_dict(), "validation": validation.to_dict()}


def _unique_swap_population(
    frequency: np.ndarray,
    reap: np.ndarray,
    *,
    population_size: int,
    seed: int,
) -> np.ndarray:
    if frequency.shape != reap.shape or frequency.dtype != np.bool_ or reap.dtype != np.bool_:
        raise ValueError("baseline masks must be same-shaped boolean arrays")
    bases = [np.ascontiguousarray(frequency.copy()), np.ascontiguousarray(reap.copy())]
    keys = {mask.tobytes() for mask in bases}
    if len(keys) != 2:
        raise PruningPilotError(
            "Frequency and REAP produced the same mask; the locked two-baseline "
            "initial population cannot be constructed"
        )
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x494E4954]))
    population = list(bases)
    attempts = 0
    while len(population) < population_size:
        base = bases[(len(population) - 2) % 2]
        candidate = mutate_uniform_swap(base, rng=rng).mask
        key = candidate.tobytes()
        attempts += 1
        if key in keys:
            if attempts > 10000:
                raise PruningPilotError("could not construct a unique initial population")
            continue
        population.append(np.ascontiguousarray(candidate))
        keys.add(key)
    return np.ascontiguousarray(np.stack(population), dtype=np.bool_)


def command_init(args: argparse.Namespace) -> None:
    workspace, config_path, config = _workspace_and_config(args.config)
    sources = _verify_locked_sources(workspace, config)
    implementations = _implementation_paths(workspace)
    run_dir = _run_dir(workspace, config)
    if run_dir.exists():
        raise PruningPilotError(f"refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    resolved_path = run_dir / "config.resolved.yaml"
    with resolved_path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=True)
    source_config_path = run_dir / "config.source.json"
    write_json(
        source_config_path,
        {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "protocol_sha256": sha256_file(sources["protocol"]),
        },
    )
    write_stage_manifest(
        run_dir / "stage_manifest.json",
        stage="pruning_pilot_initialized",
        inputs={
            "source_config": ArtifactRef.capture(config_path),
            **{name: ArtifactRef.capture(path) for name, path in sources.items()},
            **{name: ArtifactRef.capture(path) for name, path in implementations.items()},
        },
        outputs={
            "resolved_config": _ref(resolved_path, run_dir),
            "config_source": _ref(source_config_path, run_dir),
        },
        metadata={"post_hoc": True, "training": False, "visible_gpu_limit": 1},
    )
    print(f"initialized locked pruning pilot: {run_dir}", flush=True)


def _locked_run(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    workspace, _, config = _workspace_and_config(args.config)
    sources = _verify_locked_sources(workspace, config)
    run_dir = _run_dir(workspace, config)
    _verify_stage(run_dir, "stage_manifest.json", "pruning_pilot_initialized")
    resolved = _load_yaml(run_dir / "config.resolved.yaml")
    if resolved != config:
        raise PruningPilotError("source configuration changed after run initialization")
    return run_dir, config, sources


def command_baselines(args: argparse.Namespace) -> None:
    _require_one_gpu()
    run_dir, config, sources = _locked_run(args)
    output = run_dir / "baselines"
    _require_new_stage(output)
    model, _ = load_granite_model(config["model"])
    topology = inspect_granite_topology(model)
    _validate_topology(topology, config)
    bundle = load_token_bundle(sources["tokens"])
    import torch

    device = next(model.parameters()).device
    probe_ids = torch.as_tensor(bundle.fitness[:1, :64], dtype=torch.long, device=device)
    parameter_state = _parameter_identity(model)
    before = _forward_identity_snapshot(model, probe_ids)
    original_probe_nll = token_weighted_full_model_nll(
        model, bundle.fitness[:1, :64], loss_tokens=63
    )
    identity_plan = RoutingPlan.all_experts(
        num_layers=int(topology["num_layers"]), num_experts=int(topology["num_experts"])
    )
    with GranitePruningController(model, identity_plan):
        after = _forward_identity_snapshot(model, probe_ids)
        identity_probe_nll = token_weighted_full_model_nll(
            model, bundle.fitness[:1, :64], loss_tokens=63
        )
    comparison = _compare_identity(before, after)
    comparison["nll_original"] = original_probe_nll.nll
    comparison["nll_identity_controller"] = identity_probe_nll.nll
    comparison["nll_abs_delta"] = abs(identity_probe_nll.nll - original_probe_nll.nll)
    comparison["parameters_same_objects_and_versions"] = parameter_state == _parameter_identity(
        model
    )
    pruning = config["pruning"]
    comparison["passed"] = bool(
        comparison["router_topk_indices_bit_exact"]
        and comparison["router_topk_gates_bit_exact"]
        and comparison["parameters_same_objects_and_versions"]
        and comparison["lm_logits_max_abs"] <= float(pruning["identity_logits_max_abs"])
        and comparison["router_logits_max_abs"] <= float(pruning["identity_logits_max_abs"])
        and comparison["nll_abs_delta"] <= float(pruning["identity_nll_abs"])
    )
    identity_path = output / "identity.json"
    write_json(
        identity_path,
        {
            "schema_version": "pruning-identity-gate-v1",
            "probe_tokens": 64,
            "loss_tokens": 63,
            **comparison,
        },
    )
    print(
        f"P0 identity: {'PASS' if comparison['passed'] else 'FAIL'}; "
        f"logits max |delta|={comparison['lm_logits_max_abs']:.3g}, "
        f"NLL |delta|={comparison['nll_abs_delta']:.3g}",
        flush=True,
    )
    if not comparison["passed"]:
        raise PruningPilotError("identity gate failed; pruning baselines were not run")

    print("collecting exact Frequency/REAP statistics on 16,384 tokens ...", flush=True)
    statistics = collect_reap_statistics(
        model,
        bundle.statistics,
        forward_batch_size=int(config["data"]["forward_batch_size"]),
    )
    with np.load(sources["reference_statistics"], allow_pickle=False) as reference:
        reference_counts = reference["routing_counts"].astype(np.int64, copy=False)
    counts_match = np.array_equal(statistics.active_count, reference_counts)
    if not counts_match:
        raise PruningPilotError(
            "new exact router counts disagree with the locked reference statistics"
        )
    statistics_path = output / "statistics.npz"
    np.savez_compressed(
        statistics_path,
        active_count=statistics.active_count,
        routing_mass=statistics.routing_mass,
        weighted_activation_sum=statistics.weighted_activation_sum,
        reap_score=statistics.reap_score,
        tokens_per_layer=statistics.tokens_per_layer,
    )
    statistics_metadata_path = output / "statistics.json"
    write_json(
        statistics_metadata_path,
        {
            "schema_version": "pruning-statistics-v1",
            "reap_formula": str(pruning["reap_formula"]),
            "frequency_signal": str(pruning["frequency_signal"]),
            "top_k": statistics.top_k,
            "num_experts": statistics.num_experts,
            "reference_routing_counts_exact_match": counts_match,
            "cost": dict(statistics.cost),
        },
    )

    retained = int(pruning["retained_experts_per_layer"])
    frequency = build_expert_selection_plan(
        statistics.active_count.astype(np.float64), retained_per_layer=retained
    )
    reap = build_expert_selection_plan(statistics.reap_score, retained_per_layer=retained)
    frequency_path = output / "frequency_selection.json"
    reap_path = output / "reap_selection.json"
    _write_selection(frequency_path, frequency, method="frequency")
    _write_selection(reap_path, reap, method="reap")

    initial_population = _unique_swap_population(
        frequency.mask,
        reap.mask,
        population_size=int(config["search"]["population_size"]),
        seed=int(config["search"]["seed"]),
    )
    initial_path = output / "initial_population.npy"
    np.save(initial_path, initial_population, allow_pickle=False)

    all_plan = identity_plan
    controller = GranitePruningController(model, all_plan)
    nll: dict[str, dict[str, Any]] = {}
    for name, plan in (
        ("original", all_plan),
        ("frequency", _routing_plan_from_mask(frequency.mask)),
        ("reap", _routing_plan_from_mask(reap.mask)),
    ):
        controller.apply(plan)
        print(f"measuring {name} calibration/validation NLL ...", flush=True)
        nll[name] = _nll_pair(model, bundle, config)

    original_validation = float(nll["original"]["validation"]["nll"])
    for name in ("frequency", "reap"):
        delta = float(nll[name]["validation"]["nll"]) - original_validation
        nll[name]["validation_nll_delta"] = delta
        nll[name]["validation_ppl_ratio"] = math.exp(delta)

    diagnostics: list[dict[str, Any]] = []
    cached = {
        initial_population[0].tobytes(): nll["frequency"],
        initial_population[1].tobytes(): nll["reap"],
    }
    diagnostic_count = int(config["search"]["diagnostic_masks"])
    for index, mask in enumerate(initial_population[:diagnostic_count]):
        result = cached.get(mask.tobytes())
        if result is None:
            controller.apply(_routing_plan_from_mask(mask))
            print(f"diagnostic NLL mask {index + 1}/{diagnostic_count} ...", flush=True)
            result = _nll_pair(model, bundle, config)
        diagnostics.append(
            {
                "index": index,
                "source": "frequency" if index == 0 else "reap" if index == 1 else "one_swap",
                "retained_experts": [np.flatnonzero(row).astype(int).tolist() for row in mask],
                **result,
            }
        )
    controller.close()
    correlation = spearman_rank_correlation(
        [float(row["calibration"]["nll"]) for row in diagnostics],
        [float(row["validation"]["nll"]) for row in diagnostics],
    )
    ppl_passed = {
        name: float(nll[name]["validation_ppl_ratio"])
        <= float(config["gates"]["max_validation_ppl_ratio"])
        for name in ("frequency", "reap")
    }
    ppl_gate = any(ppl_passed.values())
    correlation_gate = correlation >= float(config["gates"]["min_calibration_validation_spearman"])
    nll_path = output / "nll.json"
    diagnostics_path = output / "diagnostics.json"
    analysis_path = output / "analysis.json"
    write_json(nll_path, {"schema_version": "pruning-baseline-nll-v1", **nll})
    write_json(
        diagnostics_path,
        {
            "schema_version": "pruning-nll-diagnostic-v1",
            "spearman": correlation,
            "masks": diagnostics,
        },
    )
    write_json(
        analysis_path,
        {
            "schema_version": "pruning-baseline-analysis-v1",
            "identity_gate": True,
            "validation_ppl_gate": ppl_gate,
            "baseline_validation_ppl_passed": ppl_passed,
            "calibration_validation_spearman": correlation,
            "spearman_gate": correlation_gate,
            "run_development_evaluation": ppl_gate,
            "ea_not_yet_authorized": True,
        },
    )
    environment_path = output / "environment.json"
    write_json(environment_path, environment_snapshot())
    write_stage_manifest(
        output / "stage_manifest.json",
        stage="pruning_baselines",
        inputs={
            "initialization": _ref(run_dir / "stage_manifest.json", run_dir),
            **{name: _ref(path, run_dir) for name, path in sources.items()},
        },
        outputs={
            "identity": _ref(identity_path, run_dir),
            "statistics": _ref(statistics_path, run_dir),
            "statistics_metadata": _ref(statistics_metadata_path, run_dir),
            "frequency_selection": _ref(frequency_path, run_dir),
            "reap_selection": _ref(reap_path, run_dir),
            "initial_population": _ref(initial_path, run_dir),
            "nll": _ref(nll_path, run_dir),
            "diagnostics": _ref(diagnostics_path, run_dir),
            "analysis": _ref(analysis_path, run_dir),
            "environment": _ref(environment_path, run_dir),
        },
        metadata={
            "initial_population_sha256": pruning_population_hash(initial_population),
            "diagnostic_masks": diagnostic_count,
        },
    )
    print(
        f"P1 NLL gate: {'PASS' if ppl_gate else 'FAIL'}; "
        f"Spearman={correlation:.4f} ({'PASS' if correlation_gate else 'FAIL'})",
        flush=True,
    )


def _development_config(config: Mapping[str, Any]) -> dict[str, Any]:
    development = copy.deepcopy(dict(config["development_evaluation"]))
    if development.get("split") != "validation":
        raise PruningPilotError("this post-hoc pilot may only use the validation split")
    datasets = copy.deepcopy(LOCKED_TASK_DATASETS)
    for task in development["tasks"]:
        datasets[task]["split"] = "validation"
    development["task_datasets"] = datasets
    development["protocol_id"] = FORMAL_EVALUATION_PROTOCOL_ID
    return development


def command_dev_baselines(args: argparse.Namespace) -> None:
    _require_one_gpu()
    run_dir, config, _ = _locked_run(args)
    _verify_stage(run_dir, "baselines/stage_manifest.json", "pruning_baselines")
    baseline_analysis = read_json(run_dir / "baselines" / "analysis.json")
    if not baseline_analysis.get("run_development_evaluation"):
        raise PruningPilotError("validation perplexity gate failed; dev evaluation is blocked")
    output = run_dir / "development_baselines"
    _require_new_stage(output)
    model, tokenizer = load_granite_model(config["model"])
    topology = inspect_granite_topology(model)
    evaluation_config = _development_config(config)
    all_plan = RoutingPlan.all_experts(
        num_layers=int(topology["num_layers"]), num_experts=int(topology["num_experts"])
    )
    plans = {
        "original": all_plan,
        "frequency": _selection_from_json(run_dir / "baselines" / "frequency_selection.json"),
        "reap": _selection_from_json(run_dir / "baselines" / "reap_selection.json"),
    }
    controller = GranitePruningController(model, all_plan)
    artifacts: dict[str, dict[str, Any]] = {}
    for name, plan in plans.items():
        controller.apply(plan)
        print(f"evaluating {name} on ARC-Easy/OpenBookQA validation ...", flush=True)
        artifacts[name] = evaluate_zero_shot(
            model,
            tokenizer,
            evaluation_config,
            output_path=output / f"{name}.json",
            method=name,
            seed=None,
        )
        print(f"{name} dev mean={artifacts[name]['mean_primary_acc']:.6f}", flush=True)
    controller.close()

    original = artifacts["original"]
    gates = config["gates"]
    baseline_nll = read_json(run_dir / "baselines" / "nll.json")
    candidates: dict[str, Any] = {}
    for name in ("frequency", "reap"):
        task_retention = {
            task: float(artifacts[name]["tasks"][task]["acc"])
            / float(original["tasks"][task]["acc"])
            for task in evaluation_config["tasks"]
        }
        mean_retention = float(artifacts[name]["mean_primary_acc"]) / float(
            original["mean_primary_acc"]
        )
        absolute_floor = all(
            float(artifacts[name]["tasks"][task]["acc"]) >= float(gates["min_dev_task_accuracy"])
            for task in evaluation_config["tasks"]
        )
        passed = bool(
            mean_retention >= float(gates["min_dev_mean_raw_retention"])
            and all(
                value >= float(gates["min_dev_task_raw_retention"])
                for value in task_retention.values()
            )
            and absolute_floor
        )
        candidates[name] = {
            "mean_raw_retention": mean_retention,
            "task_raw_retention": task_retention,
            "absolute_accuracy_floor_passed": absolute_floor,
            "passed": passed,
            "validation_ppl_passed": float(baseline_nll[name]["validation_ppl_ratio"])
            <= float(gates["max_validation_ppl_ratio"]),
            "joint_gate_passed": passed
            and float(baseline_nll[name]["validation_ppl_ratio"])
            <= float(gates["max_validation_ppl_ratio"]),
        }
    analysis_path = output / "analysis.json"
    write_json(
        analysis_path,
        {
            "schema_version": "pruning-development-baselines-v1",
            "split": "validation",
            "original_mean_accuracy": original["mean_primary_acc"],
            "candidates": candidates,
            "development_gate": any(row["passed"] for row in candidates.values()),
            "joint_baseline_gate": any(row["joint_gate_passed"] for row in candidates.values()),
        },
    )
    write_stage_manifest(
        output / "stage_manifest.json",
        stage="pruning_development_baselines",
        inputs={
            "baseline_stage": _ref(run_dir / "baselines" / "stage_manifest.json", run_dir),
            "frequency_selection": _ref(
                run_dir / "baselines" / "frequency_selection.json", run_dir
            ),
            "reap_selection": _ref(run_dir / "baselines" / "reap_selection.json", run_dir),
        },
        outputs={
            **{name: _ref(output / f"{name}.json", run_dir) for name in artifacts},
            "analysis": _ref(analysis_path, run_dir),
        },
        metadata={"split": "validation", "test_split_used": False},
    )
    print(
        "P2 downstream gate: "
        + ("PASS" if any(row["passed"] for row in candidates.values()) else "FAIL"),
        flush=True,
    )


def command_analyze(args: argparse.Namespace) -> None:
    run_dir, config, _ = _locked_run(args)
    _verify_stage(run_dir, "baselines/stage_manifest.json", "pruning_baselines")
    baseline = read_json(run_dir / "baselines" / "analysis.json")
    output = run_dir / "decision"
    _require_new_stage(output)
    development: dict[str, Any] | None = None
    if baseline.get("run_development_evaluation"):
        _verify_stage(
            run_dir,
            "development_baselines/stage_manifest.json",
            "pruning_development_baselines",
        )
        development = read_json(run_dir / "development_baselines" / "analysis.json")
    nll = read_json(run_dir / "baselines" / "nll.json")
    comparator = min(
        ("frequency", "reap"),
        key=lambda name: (float(nll[name]["validation"]["nll"]), name),
    )
    ppl_passed_names = {
        name
        for name in ("frequency", "reap")
        if float(nll[name]["validation_ppl_ratio"])
        <= float(config["gates"]["max_validation_ppl_ratio"])
    }
    development_passed_names = {
        name
        for name in ("frequency", "reap")
        if development and bool(development["candidates"][name]["passed"])
    }
    jointly_passed_names = sorted(ppl_passed_names & development_passed_names)
    conditions = {
        "identity_gate": bool(baseline["identity_gate"]),
        "validation_ppl_gate": bool(baseline["validation_ppl_gate"]),
        "spearman_gate": bool(baseline["spearman_gate"]),
        "development_gate": bool(development and development["development_gate"]),
        "same_baseline_passes_ppl_and_development": bool(jointly_passed_names),
    }
    proceed = all(conditions.values())
    decision_path = output / "decision.json"
    write_json(
        decision_path,
        {
            "schema_version": "pruning-ea-decision-v1",
            "proceed_to_ea": proceed,
            "conditions": conditions,
            "simple_comparator": comparator,
            "baselines_passing_all_quality_gates": jointly_passed_names,
            "comparator_selection_rule": "lowest_independent_validation_nll",
            "test_split_used": False,
            "reason": (
                "all pre-registered gates passed"
                if proceed
                else "EA stopped because at least one pre-registered gate failed"
            ),
        },
    )
    inputs = {"baseline_stage": _ref(run_dir / "baselines" / "stage_manifest.json", run_dir)}
    if development is not None:
        inputs["development_stage"] = _ref(
            run_dir / "development_baselines" / "stage_manifest.json", run_dir
        )
    write_stage_manifest(
        output / "stage_manifest.json",
        stage="pruning_ea_decision",
        inputs=inputs,
        outputs={"decision": _ref(decision_path, run_dir)},
        metadata={"proceed_to_ea": proceed, "simple_comparator": comparator},
    )
    print(
        f"EA decision: {'GO' if proceed else 'STOP'}; comparator={comparator}; "
        f"conditions={conditions}",
        flush=True,
    )


def _ea_strategy(value: str) -> str:
    mapping = {
        "uniform_swap": "uniform",
        "importance_guided_swap": "reap_guided",
        "uniform": "uniform",
        "reap_guided": "reap_guided",
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(f"unknown pruning EA strategy: {value}") from exc


def command_authorize_exploratory(args: argparse.Namespace) -> None:
    """Record an explicit post-gate exploratory branch without changing STOP."""

    run_dir, config, _ = _locked_run(args)
    _verify_stage(run_dir, "decision/stage_manifest.json", "pruning_ea_decision")
    decision = read_json(run_dir / "decision" / "decision.json")
    if decision.get("proceed_to_ea"):
        raise PruningPilotError("the primary protocol already authorizes EA")
    output = run_dir / "exploratory_override"
    _require_new_stage(output)
    authorization_path = output / "authorization.json"
    write_json(
        authorization_path,
        {
            "schema_version": "pruning-exploratory-override-v1",
            "primary_decision_preserved": "STOP",
            "reason": "explicit_user_request_after_reviewing_failed_baseline_gate",
            "scope": "EA search and independent WikiText validation only",
            "strategies": list(config["search"]["strategies"]),
            "single_seed": int(config["search"]["seed"]),
            "training_performed": False,
            "test_split_authorized": False,
            "may_reclassify_primary_result": False,
        },
    )
    write_stage_manifest(
        output / "stage_manifest.json",
        stage="pruning_exploratory_override",
        inputs={
            "primary_decision": _ref(run_dir / "decision" / "stage_manifest.json", run_dir),
            "post_run_audit": _ref(run_dir / "audit" / "stage_manifest.json", run_dir),
        },
        outputs={"authorization": _ref(authorization_path, run_dir)},
        metadata={"primary_decision_preserved": "STOP"},
    )
    print("exploratory EA branch authorized; primary STOP decision preserved", flush=True)


def command_ea(args: argparse.Namespace) -> None:
    _require_one_gpu()
    run_dir, config, sources = _locked_run(args)
    _verify_stage(run_dir, "decision/stage_manifest.json", "pruning_ea_decision")
    decision = read_json(run_dir / "decision" / "decision.json")
    exploratory = bool(getattr(args, "exploratory", False))
    if not decision.get("proceed_to_ea") and not exploratory:
        raise PruningPilotError("pre-registered analysis blocked EA")
    if exploratory:
        _verify_stage(
            run_dir,
            "exploratory_override/stage_manifest.json",
            "pruning_exploratory_override",
        )
    elif not decision.get("proceed_to_ea"):
        raise PruningPilotError("pre-registered analysis blocked EA")
    requested = str(args.strategy)
    if requested not in config["search"]["strategies"]:
        raise PruningPilotError(f"strategy is not in the locked protocol: {requested}")
    strategy = _ea_strategy(requested)
    search_root = "search_exploratory" if exploratory else "search"
    output = run_dir / search_root / requested
    _require_new_stage(output)
    initial_path = run_dir / "baselines" / "initial_population.npy"
    initial = np.load(initial_path, allow_pickle=False)
    if initial.dtype != np.bool_ or initial.ndim != 3:
        raise PruningPilotError("invalid locked initial population")
    with np.load(run_dir / "baselines" / "statistics.npz", allow_pickle=False) as payload:
        reap_score = payload["reap_score"].astype(np.float64, copy=False)
    model, _ = load_granite_model(config["model"])
    bundle = load_token_bundle(sources["tokens"])
    topology = inspect_granite_topology(model)
    all_plan = RoutingPlan.all_experts(
        num_layers=int(topology["num_layers"]), num_experts=int(topology["num_experts"])
    )
    controller = GranitePruningController(model, all_plan)
    search = config["search"]
    ea_config = PruningEAConfig(
        retained_count=int(config["pruning"]["retained_experts_per_layer"]),
        population_size=int(search["population_size"]),
        offspring_size=int(search["offspring_size"]),
        evaluation_budget=int(search["evaluation_budget"]),
        tournament_size=int(search["tournament_size"]),
        epsilon_uniform=float(search["guided_uniform_epsilon"]),
        crossover_rate=0.0,
    )
    journal_path = output / "objective_journal.jsonl"
    evaluation_index = 0

    def objective(mask: np.ndarray) -> PruningFitnessValue:
        nonlocal evaluation_index
        controller.apply(_routing_plan_from_mask(mask))
        result = token_weighted_full_model_nll(
            model,
            bundle.fitness,
            loss_tokens=int(config["data"]["fitness_tokens"]),
            micro_batch_size=int(search["micro_batch_size"]),
        )
        row = {
            "evaluation": evaluation_index,
            "nll": result.nll,
            "nll_sum": result.nll_sum,
            "loss_tokens": result.loss_tokens,
            "retained_experts": [np.flatnonzero(layer).astype(int).tolist() for layer in mask],
            "cost": dict(result.cost),
        }
        with journal_path.open("a", encoding="utf-8") as handle:
            import json

            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
        evaluation_index += 1
        if evaluation_index == 1 or evaluation_index % 25 == 0:
            print(
                f"{requested}: {evaluation_index}/{ea_config.evaluation_budget} "
                f"unique candidates, current NLL={result.nll:.6f}",
                flush=True,
            )
        return PruningFitnessValue(
            result.nll,
            {
                "nll_sum": result.nll_sum,
                "loss_tokens": result.loss_tokens,
                "cost": dict(result.cost),
                "retained_experts": row["retained_experts"],
            },
        )

    result = run_pruning_evolution(
        objective,
        initial,
        strategy=strategy,
        config=ea_config,
        seed=int(search["seed"]),
        importance=reap_score if strategy == "reap_guided" else None,
    )
    if evaluation_index != ea_config.evaluation_budget:
        raise PruningPilotError("EA objective-call count differs from the locked budget")
    controller.apply(_routing_plan_from_mask(result.best_mask))
    validation = token_weighted_full_model_nll(
        model,
        bundle.validation,
        loss_tokens=int(config["data"]["validation_tokens"]),
        micro_batch_size=int(search["micro_batch_size"]),
    )
    controller.close()
    summary = result.summary()
    summary.update(
        {
            "protocol_strategy": requested,
            "validation": validation.to_dict(),
            "validation_nll": validation.nll,
            "test_split_used": False,
            "exploratory_override": exploratory,
            "primary_stop_decision_preserved": exploratory,
        }
    )
    summary_path = output / "summary.json"
    trials_path = output / "trials.jsonl"
    best_path = output / "best_selection.json"
    environment_path = output / "environment.json"
    write_json(summary_path, summary)
    write_jsonl(trials_path, result.trials)
    write_json(
        best_path,
        {
            "schema_version": "expert-selection-v1",
            "method": requested,
            **_routing_plan_from_mask(result.best_mask).to_dict(),
        },
    )
    write_json(environment_path, environment_snapshot())
    write_stage_manifest(
        output / "stage_manifest.json",
        stage=(
            "pruning_exploratory_evolutionary_search"
            if exploratory
            else "pruning_evolutionary_search"
        ),
        inputs={
            "decision_stage": _ref(run_dir / "decision" / "stage_manifest.json", run_dir),
            "baseline_stage": _ref(run_dir / "baselines" / "stage_manifest.json", run_dir),
            "initial_population": _ref(initial_path, run_dir),
            "tokens": _ref(sources["tokens"], run_dir),
            **(
                {
                    "exploratory_override": _ref(
                        run_dir / "exploratory_override" / "stage_manifest.json", run_dir
                    )
                }
                if exploratory
                else {}
            ),
        },
        outputs={
            "summary": _ref(summary_path, run_dir),
            "trials": _ref(trials_path, run_dir),
            "objective_journal": _ref(journal_path, run_dir),
            "best_selection": _ref(best_path, run_dir),
            "environment": _ref(environment_path, run_dir),
        },
        metadata={
            "evaluations": result.evaluations,
            "initial_population_sha256": result.initial_population_hash,
            "exploratory_override": exploratory,
        },
    )
    print(
        f"{requested} complete: calibration NLL={result.best_fitness:.6f}, "
        f"validation NLL={validation.nll:.6f}",
        flush=True,
    )


def command_select(args: argparse.Namespace) -> None:
    run_dir, config, _ = _locked_run(args)
    strategies = list(config["search"]["strategies"])
    exploratory = bool(getattr(args, "exploratory", False))
    search_root = "search_exploratory" if exploratory else "search"
    expected_stage = (
        "pruning_exploratory_evolutionary_search" if exploratory else "pruning_evolutionary_search"
    )
    summaries: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        _verify_stage(
            run_dir,
            f"{search_root}/{strategy}/stage_manifest.json",
            expected_stage,
        )
        summaries[strategy] = read_json(run_dir / search_root / strategy / "summary.json")
    expected_seed = int(config["search"]["seed"])
    expected_budget = int(config["search"]["evaluation_budget"])
    expected_population = int(config["search"]["population_size"])
    initial_hashes = {str(summary["initial_population_sha256"]) for summary in summaries.values()}
    if len(initial_hashes) != 1:
        raise PruningPilotError("EA strategies did not use the same initial population")
    initialization_evidence: list[tuple[tuple[str, float], ...]] = []
    comparable_environments: list[dict[str, Any]] = []
    for name in strategies:
        summary = summaries[name]
        if int(summary["seed"]) != expected_seed or int(summary["evaluations"]) != expected_budget:
            raise PruningPilotError(f"{name} search seed or budget differs from the protocol")
        trials = read_jsonl(run_dir / search_root / name / "trials.jsonl")
        if len(trials) != expected_budget:
            raise PruningPilotError(f"{name} search trial count differs from its summary")
        initialization_evidence.append(
            tuple(
                (str(row["candidate_sha256"]), float(row["fitness"]))
                for row in trials[:expected_population]
            )
        )
        environment = read_json(run_dir / search_root / name / "environment.json")
        comparable_environments.append(
            {
                "python": environment.get("python"),
                "platform": environment.get("platform"),
                "packages": environment.get("packages"),
                "accelerator": environment.get("accelerator"),
            }
        )
    if any(rows != initialization_evidence[0] for rows in initialization_evidence[1:]):
        raise PruningPilotError(
            "EA strategies disagree on candidate IDs or fitnesses in the shared initialization"
        )
    if any(env != comparable_environments[0] for env in comparable_environments[1:]):
        raise PruningPilotError("EA strategies ran under different recorded environments")
    output = run_dir / ("winner_exploratory" if exploratory else "winner")
    _require_new_stage(output)
    strategy = min(
        strategies,
        key=lambda name: (float(summaries[name]["validation_nll"]), name),
    )
    decision = read_json(run_dir / "decision" / "decision.json")
    comparator = str(decision["simple_comparator"])
    baseline_nll = read_json(run_dir / "baselines" / "nll.json")
    comparator_nll = float(baseline_nll[comparator]["validation"]["nll"])
    winner_nll = float(summaries[strategy]["validation_nll"])
    improvement = comparator_nll - winner_nll
    nll_gate = improvement >= float(config["gates"]["min_ea_validation_nll_improvement"])
    selection_path = output / "selection.json"
    write_json(
        selection_path,
        {
            "schema_version": "pruning-winner-selection-v1",
            "strategy": strategy,
            "selection_rule": "lowest_independent_validation_nll",
            "validation_nll": winner_nll,
            "simple_comparator": comparator,
            "comparator_validation_nll": comparator_nll,
            "validation_nll_improvement": improvement,
            "validation_nll_improvement_gate": nll_gate,
            "test_split_used": False,
            "exploratory_override": exploratory,
            "primary_stop_decision_preserved": exploratory,
        },
    )
    write_stage_manifest(
        output / "stage_manifest.json",
        stage=(
            "pruning_exploratory_winner_selection" if exploratory else "pruning_winner_selection"
        ),
        inputs={
            strategy: _ref(run_dir / search_root / strategy / "stage_manifest.json", run_dir)
            for strategy in strategies
        },
        outputs={"selection": _ref(selection_path, run_dir)},
        metadata={
            "strategy": strategy,
            "validation_nll_improvement_gate": nll_gate,
            "exploratory_override": exploratory,
        },
    )
    print(
        f"winner={strategy}; validation NLL improvement vs {comparator}="
        f"{improvement:.6f} ({'PASS' if nll_gate else 'FAIL'})",
        flush=True,
    )


def command_dev_winner(args: argparse.Namespace) -> None:
    _require_one_gpu()
    run_dir, config, _ = _locked_run(args)
    _verify_stage(run_dir, "winner/stage_manifest.json", "pruning_winner_selection")
    selection = read_json(run_dir / "winner" / "selection.json")
    if not selection.get("validation_nll_improvement_gate"):
        raise PruningPilotError(
            "winner failed the validation-NLL improvement gate; downstream evaluation is blocked"
        )
    strategy = str(selection["strategy"])
    output = run_dir / "development_winner"
    _require_new_stage(output)
    model, tokenizer = load_granite_model(config["model"])
    winner_plan = RoutingPlan.from_dict(
        read_json(run_dir / "search" / strategy / "best_selection.json")
    )
    controller = GranitePruningController(model, winner_plan)
    evaluation = evaluate_zero_shot(
        model,
        tokenizer,
        _development_config(config),
        output_path=output / "winner.json",
        method=strategy,
        seed=int(config["search"]["seed"]),
    )
    controller.close()
    comparator = str(selection["simple_comparator"])
    comparator_eval = read_json(run_dir / "development_baselines" / f"{comparator}.json")
    tasks = list(config["development_evaluation"]["tasks"])
    mean_improvement = float(evaluation["mean_primary_acc"]) - float(
        comparator_eval["mean_primary_acc"]
    )
    task_changes = {
        task: float(evaluation["tasks"][task]["acc"]) - float(comparator_eval["tasks"][task]["acc"])
        for task in tasks
    }
    useful = bool(
        selection["validation_nll_improvement_gate"]
        and mean_improvement >= float(config["gates"]["min_ea_dev_mean_absolute_improvement"])
        and all(
            change >= -float(config["gates"]["max_ea_task_absolute_regression"])
            for change in task_changes.values()
        )
    )
    conclusion_path = output / "conclusion.json"
    write_json(
        conclusion_path,
        {
            "schema_version": "pruning-pilot-conclusion-v1",
            "strategy": strategy,
            "simple_comparator": comparator,
            "validation_nll_improvement": selection["validation_nll_improvement"],
            "mean_dev_accuracy_improvement": mean_improvement,
            "task_accuracy_changes": task_changes,
            "ea_has_practical_continuation_value": useful,
            "single_seed_pilot": True,
            "training_performed": False,
            "test_split_used": False,
        },
    )
    write_stage_manifest(
        output / "stage_manifest.json",
        stage="pruning_winner_development_evaluation",
        inputs={
            "winner_stage": _ref(run_dir / "winner" / "stage_manifest.json", run_dir),
            "development_baselines_stage": _ref(
                run_dir / "development_baselines" / "stage_manifest.json", run_dir
            ),
        },
        outputs={
            "evaluation": _ref(output / "winner.json", run_dir),
            "conclusion": _ref(conclusion_path, run_dir),
        },
        metadata={"ea_has_practical_continuation_value": useful},
    )
    print(
        f"final pilot conclusion: {'CONTINUE' if useful else 'STOP'}; "
        f"mean dev delta={mean_improvement:+.6f}, task deltas={task_changes}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/expert_pruning_pilot.yaml", help="locked YAML config"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands = {
        "init": command_init,
        "baselines": command_baselines,
        "dev-baselines": command_dev_baselines,
        "analyze": command_analyze,
        "authorize-exploratory": command_authorize_exploratory,
        "dev-winner": command_dev_winner,
    }
    for name, function in commands.items():
        child = subparsers.add_parser(name)
        child.set_defaults(function=function)
    ea = subparsers.add_parser("ea")
    ea.add_argument("--strategy", required=True)
    ea.add_argument("--exploratory", action="store_true")
    ea.set_defaults(function=command_ea)
    select = subparsers.add_parser("select")
    select.add_argument("--exploratory", action="store_true")
    select.set_defaults(function=command_select)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()

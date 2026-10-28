#!/usr/bin/env python3
"""Portable entry point for the released TAPE artifacts and GPU experiments."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np  # noqa: E402

from dsai6707.io import read_json, sha256_file, utc_now, write_json  # noqa: E402
from dsai6707.pruning_global_budget_search import GlobalBudgetInitialPopulation  # noqa: E402
from dsai6707.pruning_multistart import mask_digest, proxy_selection_key  # noqa: E402
from dsai6707.task_choice_proxy import canonical_sha256  # noqa: E402
from scripts import run_uniform_tape as uniform  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
TASKS = ("arc_easy", "openbookqa")


def case_spec(name: str) -> tuple[Path, dict]:
    catalog = read_json(ARTIFACTS / "cases.json")
    if name not in catalog["cases"]:
        raise ValueError(f"unknown case {name!r}; choose from {list(catalog['cases'])}")
    folder = ARTIFACTS / "cases" / name
    return folder, read_json(folder / "config.json")


def search_config(spec: dict):
    config = replace(
        uniform.config_for(spec["shape"][0], spec["retained_per_layer_reference"], spec["mode"]),
        seed=spec["search"]["seed"],
    )
    if canonical_sha256(asdict(config)) != canonical_sha256(spec["search"]):
        raise RuntimeError("released settings differ from the search implementation")
    return config


def initial_population(folder: Path) -> GlobalBudgetInitialPopulation:
    metadata = read_json(folder / "initial_population.json")
    return GlobalBudgetInitialPopulation(
        np.load(folder / "initial_population.npy", allow_pickle=False),
        tuple(metadata["contexts"]),
        metadata["population_sha256"],
    )


def aggregate_scores(scores: list[dict], original_mean: float) -> dict:
    """Compute sample variability across searches, not across test questions."""
    if len(scores) < 2 or not 0 < original_mean <= 1:
        raise ValueError("at least two runs and a positive original accuracy are required")
    values = {task: [] for task in TASKS}
    values.update(avg=[], performance_retained=[])
    for score in scores:
        tasks = [float(score["tasks"][task]) for task in TASKS]
        if not all(np.isfinite(value) and 0 <= value <= 1 for value in tasks):
            raise ValueError("accuracy must be a finite fraction between zero and one")
        average = mean(tasks)
        if not np.isclose(average, score["mean"], rtol=0, atol=1e-12):
            raise ValueError("saved average differs from the task scores")
        for task, value in zip(TASKS, tasks, strict=True):
            values[task].append(100 * value)
        values["avg"].append(100 * average)
        values["performance_retained"].append(100 * average / original_mean)
    return {
        metric: {"mean_percent": mean(rows), "sample_std_percentage_points": stdev(rows)}
        for metric, rows in values.items()
    }


def statistics_summary() -> dict:
    """Recompute each reported cohort from its individual released evaluations."""
    summary = read_json(ARTIFACTS / "multiseed_summary.json")
    originals = read_json(ARTIFACTS / "originals.json")
    groups = {}
    for name, group in summary["cases"].items():
        _, spec = case_spec(name)
        runs = {}
        for row in group["runs"]:
            folder, run_spec = case_spec(row["case_id"])
            if row["seed"] != run_spec["search"]["seed"] or row["seed"] in runs:
                raise RuntimeError(f"invalid or duplicate seed record in {name}")
            evaluation = read_json(folder / "evaluations.json")["tape_multistart"]
            scores = evaluation["scores"]
            if canonical_sha256(scores) != canonical_sha256(row["scores"]):
                raise RuntimeError(f"summary differs from evaluation for {row['case_id']}")
            if evaluation["source_sha256"] != row["source_evaluation_sha256"]:
                raise RuntimeError(f"summary source hash differs for {row['case_id']}")
            runs[row["seed"]] = scores
        original_mean = originals[spec["family"]]["scores"]["mean"]

        groups[name] = verified_cohort(
            group["reported_seeds"], group["metrics"], runs, original_mean, name
        )
        if group["n"] != groups[name]["n"]:
            raise RuntimeError(f"incorrect reported sample count in {name}")
        for cohort in ("original_cohort", "all_recorded"):
            declared = group[cohort]
            groups[name][cohort] = verified_cohort(
                declared["seeds"], declared["metrics"], runs, original_mean, name
            )
            if declared["n"] != groups[name][cohort]["n"]:
                raise RuntimeError(f"incorrect {cohort} sample count in {name}")
    return {
        "std_definition": "sample standard deviation across search seeds (ddof=1)",
        "late_seed_note": summary["late_seed_note"],
        "cases": groups,
    }


def verified_cohort(
    seeds: list[int], declared: dict, runs: dict, original_mean: float, group_name: str
) -> dict:
    metrics = aggregate_scores([runs[seed] for seed in seeds], original_mean)
    for metric, values in metrics.items():
        for key, value in values.items():
            if not np.isclose(value, declared[metric][key], rtol=0, atol=1e-9):
                raise RuntimeError(f"incorrect {metric}/{key} statistic in {group_name}")
    return {"seeds": seeds, "n": len(seeds), "metrics": metrics}


def verify_release() -> int:
    manifest = read_json(ROOT / "RELEASE_MANIFEST.json")
    for name, entry in manifest["files"].items():
        path = ROOT / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise RuntimeError(f"unsafe release member: {name}")
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"release member missing or modified: {name}")
    catalog = read_json(ARTIFACTS / "cases.json")
    for name in catalog["cases"]:
        folder, spec = case_spec(name)
        config = search_config(spec)
        masks = np.load(folder / "candidate_masks.npy", allow_pickle=False)
        if len(masks) != config.evaluation_budget:
            raise RuntimeError("wrong candidate budget")
        capacities = masks.sum(axis=2)
        if not np.all(masks.sum((1, 2)) == config.total_retained) or not np.all(
            (capacities >= config.min_retained_per_layer)
            & (capacities <= config.max_retained_per_layer)
        ):
            raise RuntimeError("infeasible released candidates")
        if len({mask_digest(mask) for mask in masks}) != len(masks):
            raise RuntimeError("duplicate released candidates")
        frozen = read_json(folder / "frozen_selection.json")
        if frozen["selection_uses_test"] is not False:
            raise RuntimeError("unexpected test-dependent selection")
        final = np.load(folder / "winner_mask.npy", allow_pickle=False)
        if mask_digest(final) != frozen["winner"]["mask_sha256"]:
            raise RuntimeError("frozen selection does not match its mask")
    if (ARTIFACTS / "multiseed_summary.json").exists():
        statistics_summary()
    return len(manifest["files"])


def replay(name: str) -> dict:
    """Re-execute the search with recorded train objectives, without a model."""
    folder, spec = case_spec(name)
    expected = np.load(folder / "candidate_masks.npy", allow_pickle=False)
    archive = read_json(folder / "search.json")
    lookup = {
        mask_digest(mask): tuple(metrics["task_objectives"])
        for mask, metrics in zip(expected, archive["metrics"], strict=True)
    }

    def objective(masks):
        return [lookup[mask_digest(mask)] for mask in masks]

    result = uniform.run_search(
        objective,
        np.load(folder / "rankings.npy", allow_pickle=False),
        search_mode=spec["mode"],
        config=search_config(spec),
        initial_population=initial_population(folder),
    )
    if not np.array_equal(result.masks, expected):
        raise RuntimeError("CPU replay generated a different candidate sequence")
    frozen = read_json(folder / "frozen_selection.json")
    selected = min(
        frozen["holdout_candidates"],
        key=lambda row: proxy_selection_key(row["metrics"], key=row["mask_sha256"]),
    )
    if selected != frozen["winner"]:
        raise RuntimeError("CPU replay selected a different output")
    return {"case": name, "candidates": len(result.masks), "selection_matches": True}


def require_gpu(spec: dict) -> None:
    from scripts.run_pruning_28of32_taskaware_nsga2 import _require_one_visible_gpu

    _require_one_visible_gpu()
    if spec["family"] == "ernie":
        from scripts.run_ernie21b_taskaware_nsga2 import _require_runtime

        _require_runtime()


def load_models(spec: dict, model_path: Path | None, *, replicas: int):
    import torch

    torch.set_num_threads(4)
    if spec["family"] == "granite":
        from dsai6707.granite import load_granite_model

        models, tokenizer = [], None
        for _ in range(replicas):
            model, tokenizer = load_granite_model(spec["model"])
            models.append(model)
        return models, tokenizer
    if model_path is None:
        raise ValueError("ERNIE requires --model-path pointing to the prepared checkpoint")
    from scripts import run_ernie21b_taskaware_nsga2 as ernie

    ernie._configure_torch(torch, seed=6707)
    tokenizer = ernie._load_tokenizer(model_path.resolve())
    model, _ = ernie._load_model(model_path.resolve(), expected_experts=64)
    return [model], tokenizer


def evaluation(model, tokenizer, mask: np.ndarray, output: Path, *, label: str) -> dict:
    from dsai6707.evaluation import evaluate_zero_shot
    from scripts.evaluate_reap_ernie import _evaluation_config

    artifact = evaluate_zero_shot(
        model,
        tokenizer,
        _evaluation_config("test", batch_size=16, log_samples=True),
        method=label,
        seed=6707,
    )
    artifact.update(mask_sha256=mask_digest(mask), selection_uses_test=False)
    write_json(output, artifact)
    return {
        "tasks": {task: artifact["tasks"][task]["acc_norm"] for task in TASKS},
        "mean": float(np.mean([artifact["tasks"][task]["acc_norm"] for task in TASKS])),
    }


def apply_controller(model, family: str, mask: np.ndarray):
    if family == "granite":
        from dsai6707.pruning import GranitePruningController
        from scripts.run_pruning_28of32_taskaware_nsga2 import _routing_plan

        return GranitePruningController(model, _routing_plan(mask))
    from dsai6707.ernie_pruning import ErnieExpertMaskController

    return ErnieExpertMaskController(model, mask)


def evaluation_mask(folder: Path, spec: dict, method: str) -> np.ndarray:
    if method == "original":
        return np.ones(spec["shape"], dtype=bool)
    if method == "tape":
        return np.load(folder / "winner_mask.npy", allow_pickle=False)
    extra = folder / "evaluation_masks" / f"{method}.npy"
    if extra.exists():
        return np.load(extra, allow_pickle=False)
    if method not in spec["anchor_names"]:
        raise ValueError(f"no released mask for {method}")
    return np.load(folder / "anchors.npy", allow_pickle=False)[spec["anchor_names"].index(method)]


def evaluate(name: str, model_path: Path | None, output: Path, method: str) -> dict:
    folder, spec = case_spec(name)
    require_gpu(spec)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an evaluation: {output}")
    models, tokenizer = load_models(spec, model_path, replicas=1)
    mask = evaluation_mask(folder, spec, method)
    controller = apply_controller(models[0], spec["family"], mask)
    try:
        return evaluation(models[0], tokenizer, mask, output, label=f"released_{name}_{method}")
    finally:
        controller.close()


def run(name: str, model_path: Path | None, output: Path, *, test: bool) -> dict:
    """Fresh forward-only search; no scores or tests are reused from the release."""
    import torch

    from scripts import run_pruning_28of32_taskaware_nsga2 as shared
    from scripts.run_tape_multistart_followup import _metrics

    folder, spec = case_spec(name)
    require_gpu(spec)
    if output.exists():
        raise FileExistsError(f"choose a new output directory: {output}")
    output.mkdir(parents=True)
    started = perf_counter()
    models, tokenizer = load_models(
        spec, model_path, replicas=2 if spec["family"] == "granite" else 1
    )
    before = [shared._parameter_identity(model) for model in models]
    if any(model.training or any(p.requires_grad for p in model.parameters()) for model in models):
        raise RuntimeError("the released runner only supports frozen inference")
    proxy = shared._load_proxy_data(
        {
            "data": spec["data"],
            "search": {"seed": 6707},
            "parallel": {"proxy_batch_size": 32 if spec["family"] == "granite" else 8},
        },
        tokenizer,
    )
    expected_proxy = read_json(folder / "proxy_data.json")
    if canonical_sha256(proxy.evidence) != canonical_sha256(expected_proxy):
        raise RuntimeError("downloaded/tokenized train data differs from the released data")
    write_json(output / "proxy_data.json", proxy.evidence)
    variant = uniform.VARIANT
    scores = {"search": {}, "holdout": {}}
    if spec["family"] == "granite":
        parallel = shared.ParallelTaskProxyEvaluator(models, proxy)
        controller = None
    else:
        from scripts.run_ernie21b_taskaware_nsga2 import SerialChoiceScorer

        parallel = None
        controller = apply_controller(models[0], "ernie", np.ones(spec["shape"], dtype=bool))
        scorer = SerialChoiceScorer(models[0], proxy)

    def score_many(masks, split):
        missing = []
        for mask in masks:
            digest = mask_digest(mask)
            if digest not in scores[split] and digest not in {mask_digest(m) for m in missing}:
                missing.append(mask)
        width = 2 if parallel is not None else 1
        for offset in range(0, len(missing), width):
            batch = missing[offset : offset + width]
            if parallel is not None:
                values = parallel.score_many(batch, split=split, parallel=len(batch) == 2)
            else:
                controller.apply(batch[0])
                values = (scorer.score(split),)
            for mask, value in zip(batch, values, strict=True):
                scores[split][mask_digest(mask)] = value.choice_log_likelihoods
        return [scores[split][mask_digest(mask)] for mask in masks]

    try:
        teachers = {
            split: score_many([np.ones(spec["shape"], dtype=bool)], split)[0]
            for split in ("search", "holdout")
        }
        metrics = []

        def objective(masks):
            values = score_many(masks, "search")
            rows = [_metrics(proxy, "search", v, teachers["search"], variant) for v in values]
            metrics.extend(rows)
            print(f"SEARCH {name} {len(metrics)}/256", flush=True)
            return [tuple(row["task_objectives"]) for row in rows]

        search_started = perf_counter()
        result = uniform.run_search(
            objective,
            np.load(folder / "rankings.npy", allow_pickle=False),
            search_mode=spec["mode"],
            config=search_config(spec),
            initial_population=initial_population(folder),
        )
        search_seconds = perf_counter() - search_started
        ids = uniform.shortlist(metrics)
        selection_started = perf_counter()
        values = score_many([result.masks[i] for i in ids], "holdout")
        candidates = [
            {
                "candidate_id": int(index),
                "mask_sha256": mask_digest(result.masks[index]),
                "metrics": _metrics(proxy, "holdout", value, teachers["holdout"], variant),
            }
            for index, value in zip(ids, values, strict=True)
        ]
        winner = min(
            candidates, key=lambda row: proxy_selection_key(row["metrics"], key=row["mask_sha256"])
        )
        selection_seconds = perf_counter() - selection_started
        mask = result.masks[winner["candidate_id"]]
        frozen_path = output / "frozen_selection.json"
        write_json(
            frozen_path,
            {
                "frozen_at": utc_now(),
                "selection_uses_test": False,
                "winner": winner,
                "holdout_candidates": candidates,
            },
        )
        np.save(output / "winner_mask.npy", mask, allow_pickle=False)
        np.save(output / "candidate_masks.npy", result.masks, allow_pickle=False)
        write_json(output / "search.json", {"trials": result.trials, "metrics": metrics})
        if parallel is not None:
            parallel.close()
        else:
            controller.close()
        test_result = None
        if test:
            selected_controller = apply_controller(models[0], spec["family"], mask)
            try:
                test_result = evaluation(
                    models[0], tokenizer, mask, output / "test.json", label=f"fresh_{name}"
                )
            finally:
                selected_controller.close()
        if [shared._parameter_identity(model) for model in models] != before:
            raise RuntimeError("inference changed model weights")
        summary = {
            "case": name,
            "search_wall_seconds": search_seconds,
            "selection_wall_seconds": selection_seconds,
            "total_wall_seconds": perf_counter() - started,
            "training_performed": False,
            "parameters_unchanged": True,
            "peak_allocated_memory_bytes": torch.cuda.max_memory_allocated(),
            "test": test_result,
        }
        write_json(output / "execution.json", summary)
        return summary
    finally:
        if parallel is not None:
            parallel.close()
        elif controller is not None:
            controller.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("check", "tables", "statistics", "replay", "run", "evaluate")
    )
    parser.add_argument("--case", default="all")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", default="tape")
    parser.add_argument(
        "--test", action="store_true", help="test the train-selected output after freezing"
    )
    args = parser.parse_args()
    if args.action == "check":
        print(json.dumps({"verified_files": verify_release()}))
    elif args.action == "tables":
        print((ROOT / "docs/RESULTS.md").read_text())
    elif args.action == "statistics":
        print(json.dumps(statistics_summary(), indent=2))
    elif args.action == "replay":
        names = (
            list(read_json(ARTIFACTS / "cases.json")["cases"])
            if args.case == "all"
            else [args.case]
        )
        for name in names:
            print(json.dumps(replay(name)))
    else:
        if args.case == "all" or args.output is None:
            parser.error("run/evaluate require one --case and a new --output path")
        result = (
            run(args.case, args.model_path, args.output.resolve(), test=args.test)
            if args.action == "run"
            else evaluate(args.case, args.model_path, args.output.resolve(), args.method)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

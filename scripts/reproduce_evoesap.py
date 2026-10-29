#!/usr/bin/env python3
"""Verify, replay, or rerun the released budget-matched EvoESAP baseline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np  # noqa: E402

from dsai6707.evoesap import EvoESAPConfig, run_evoesap  # noqa: E402
from dsai6707.io import read_json, sha256_file, utc_now, write_json  # noqa: E402
from dsai6707.pruning_global_budget import build_mask_from_layer_rankings  # noqa: E402
from dsai6707.pruning_multistart import mask_digest  # noqa: E402
from dsai6707.task_choice_proxy import canonical_sha256  # noqa: E402
from scripts import reproduce_tape as tape  # noqa: E402

ARTIFACTS = ROOT / "artifacts/evoesap"
TASKS = ("arc_easy", "openbookqa")


def case_spec(name: str) -> tuple[Path, dict, Path, dict]:
    catalog = read_json(ARTIFACTS / "cases.json")
    if name not in catalog["cases"]:
        raise ValueError(f"unknown EvoESAP case {name!r}; choose from {list(catalog['cases'])}")
    folder = ARTIFACTS / "cases" / name
    spec = read_json(folder / "config.json")
    reference = ROOT / "artifacts/cases" / spec["reference_case"]
    reference_spec = read_json(reference / "config.json")
    for member in ("rankings.npy", "proxy_data.json"):
        if sha256_file(reference / member) != spec["reference_inputs_sha256"][member]:
            raise RuntimeError(f"reference input changed for {name}: {member}")
    return folder, spec, reference, reference_spec


def search_config(spec: dict) -> EvoESAPConfig:
    values = dict(spec["search"])
    values["initial_patterns"] = tuple(values["initial_patterns"])
    return EvoESAPConfig(**values)


def shortlist(scores: np.ndarray, *, size: int = 24) -> list[int]:
    """Freeze top search-ESAP candidates, retaining the uniform control."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or not 1 <= size <= len(values):
        raise ValueError("invalid ESAP shortlist")
    indices = np.lexsort((np.arange(len(values)), -values))[:size].astype(int).tolist()
    if 0 not in indices:
        indices[-1] = 0
    return indices


def select_holdout(candidates: list[dict]) -> dict:
    if not candidates:
        raise ValueError("a non-empty holdout shortlist is required")
    return min(candidates, key=lambda row: (-row["esap"], row["mask_sha256"]))


def verify_evaluation(evaluation: dict, frozen: dict, freeze_sha256: str) -> None:
    """Check full test coverage and its link to the train-frozen output."""
    if (
        evaluation["selection_uses_test"] is not False
        or evaluation["limit"] is not None
        or evaluation["mask_sha256"] != frozen["mask_sha256"]
        or evaluation["freeze_sha256"] != freeze_sha256
        or evaluation["protocol_sha256"] != frozen["protocol_sha256"]
        or evaluation["primary_metric"] != "acc_norm"
        or evaluation["is_synthetic"] is not False
    ):
        raise RuntimeError("test evidence does not match the train-frozen EvoESAP output")
    for task, expected in (("arc_easy", 2376), ("openbookqa", 500)):
        if (
            evaluation["dataset_evidence"][task]["evaluation_split"] != "test"
            or evaluation["cost"]["sample_counts"][task]["effective"] != expected
        ):
            raise RuntimeError("incomplete or incorrect EvoESAP test split")
        value = evaluation["scores"]["tasks"][task]
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise RuntimeError("invalid EvoESAP test accuracy")
        if "tasks" in evaluation and value != evaluation["tasks"][task]["acc_norm"]:
            raise RuntimeError("reported EvoESAP score is not its source acc_norm")
    if not np.isclose(
        evaluation["scores"]["mean"],
        np.mean([evaluation["scores"]["tasks"][task] for task in TASKS]),
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError("incorrect EvoESAP mean test accuracy")


def verify_metric(row: dict, teacher: dict, questions_per_task: int) -> None:
    metrics = row["metrics"]
    if (
        teacher["full_vocabulary"] is not True
        or teacher["answer_positions_only"] is not True
        or metrics["lower_is_better"] is not False
        or metrics["num_questions"] != 2 * questions_per_task
        or metrics["answer_tokens"] != teacher["answer_tokens"]
        or metrics["aggregation"] != "mean_examples_after_mean_answer_positions"
        or metrics["fitness"] != metrics["esap"]
        or metrics["dataset_esap"] != metrics["esap"]
        or not 0 <= metrics["esap"] <= 1
    ):
        raise RuntimeError("invalid full-vocabulary EvoESAP fitness evidence")
    for task in TASKS:
        if metrics["tasks"][task]["num_questions"] != questions_per_task:
            raise RuntimeError("wrong number of EvoESAP calibration questions")
    if not np.isclose(
        np.mean([metrics["tasks"][task]["esap"] for task in TASKS]),
        metrics["esap"],
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError("incorrect task-balanced EvoESAP fitness")


def verify_case(name: str) -> dict:
    folder, spec, reference, _ = case_spec(name)
    config = search_config(spec)
    masks = np.load(folder / "candidate_masks.npy", allow_pickle=False)
    rankings = np.load(reference / "rankings.npy", allow_pickle=False)
    if masks.dtype != np.bool_ or masks.shape != (config.evaluation_budget, *spec["shape"]):
        raise RuntimeError("wrong EvoESAP candidate shape or evaluation budget")
    capacities = masks.sum(axis=2)
    if (
        not np.all(masks.sum(axis=(1, 2)) == config.total_retained)
        or np.any(capacities < config.min_retained_per_layer)
        or np.any(capacities > config.max_retained_per_layer)
        or len({mask_digest(mask) for mask in masks}) != len(masks)
    ):
        raise RuntimeError("infeasible or duplicate EvoESAP candidates")
    for mask, counts in zip(masks, capacities, strict=True):
        expected_mask = build_mask_from_layer_rankings(
            rankings,
            counts,
            total_retained=config.total_retained,
            min_retained_per_layer=config.min_retained_per_layer,
            max_retained_per_layer=config.max_retained_per_layer,
        )
        if not np.array_equal(mask, expected_mask):
            raise RuntimeError("EvoESAP changed the fixed within-layer REAP ranking")
    archive = read_json(folder / "search.json")
    if canonical_sha256(archive["config"]) != canonical_sha256(spec["search"]):
        raise RuntimeError("EvoESAP search configuration differs from its evidence")
    if len(archive["esap"]) != len(masks) or len(archive["trials"]) != len(masks):
        raise RuntimeError("incomplete EvoESAP search journal")
    metrics = read_json(folder / "metrics.json")
    teacher_search = read_json(folder / "teacher_search.json")
    teacher_holdout = read_json(folder / "teacher_holdout.json")
    if len(metrics["search"]) != len(masks):
        raise RuntimeError("missing EvoESAP search metrics")
    for index, (mask, score, row) in enumerate(
        zip(masks, archive["esap"], metrics["search"], strict=True)
    ):
        verify_metric(row, teacher_search, 64)
        digest = mask_digest(mask)
        if (
            row["candidate_id"] != index
            or row["mask_sha256"] != digest
            or archive["trials"][index]["mask_sha256"] != digest
            or row["metrics"]["esap"] != score
            or archive["trials"][index]["esap"] != score
        ):
            raise RuntimeError("EvoESAP search scores do not match their candidate masks")
    frozen = read_json(folder / "frozen_selection.json")
    if frozen["selection_uses_test"] is not False or frozen["selection_split"] != "train-holdout":
        raise RuntimeError("EvoESAP output was not selected on train holdout")
    if frozen["protocol_sha256"] != read_json(ARTIFACTS / "protocol.json")["source_sha256"]:
        raise RuntimeError("EvoESAP frozen output has the wrong source protocol")
    ids = shortlist(np.asarray(archive["esap"]))
    if [row["candidate_id"] for row in frozen["holdout_candidates"]] != ids:
        raise RuntimeError("incorrect EvoESAP search-to-holdout shortlist")
    if len(metrics["holdout"]) != len(ids):
        raise RuntimeError("missing EvoESAP holdout metrics")
    for frozen_row, metric_row in zip(
        frozen["holdout_candidates"], metrics["holdout"], strict=True
    ):
        verify_metric(metric_row, teacher_holdout, 256)
        index = frozen_row["candidate_id"]
        if (
            frozen_row["mask_sha256"] != mask_digest(masks[index])
            or metric_row["candidate_id"] != index
            or metric_row["mask_sha256"] != frozen_row["mask_sha256"]
            or metric_row["metrics"]["esap"] != frozen_row["esap"]
        ):
            raise RuntimeError("EvoESAP holdout scores do not match their masks")
    if select_holdout(frozen["holdout_candidates"]) != frozen["winner"]:
        raise RuntimeError("incorrect EvoESAP holdout winner")
    winner_mask = np.load(folder / "winner_mask.npy", allow_pickle=False)
    if mask_digest(winner_mask) != frozen["mask_sha256"] or not np.array_equal(
        winner_mask, masks[frozen["winner"]["candidate_id"]]
    ):
        raise RuntimeError("EvoESAP frozen winner mask changed")
    evaluation = read_json(folder / "evaluation.json")
    verify_evaluation(evaluation, frozen, sha256_file(folder / "frozen_selection.json"))
    return {"case": name, "candidates": len(masks), "selection_matches": True}


def replay(name: str) -> dict:
    """Re-execute operators with recorded ESAP, without model or test access."""
    verified = verify_case(name)
    folder, spec, reference, _ = case_spec(name)
    expected = np.load(folder / "candidate_masks.npy", allow_pickle=False)
    archive = read_json(folder / "search.json")
    scores = dict(zip((mask_digest(mask) for mask in expected), archive["esap"], strict=True))
    result = run_evoesap(
        lambda masks: [scores[mask_digest(mask)] for mask in masks],
        np.load(reference / "rankings.npy", allow_pickle=False),
        config=search_config(spec),
    )
    if not np.array_equal(result.masks, expected):
        raise RuntimeError("CPU replay generated a different EvoESAP candidate sequence")
    if (
        canonical_sha256(result.trials) != canonical_sha256(archive["trials"])
        or canonical_sha256(result.generation_history)
        != canonical_sha256(archive["generation_history"])
        or result.winner_id != archive["search_winner_id"]
    ):
        raise RuntimeError("CPU replay generated a different EvoESAP search journal")
    return verified


def statistics_summary() -> dict:
    """Recompute means and sample SD from all twelve individual full tests."""
    summary = read_json(ARTIFACTS / "summary.json")
    originals = read_json(ROOT / "artifacts/originals.json")
    groups = {}
    for name, group in summary["cases"].items():
        rows = []
        for run in group["runs"]:
            folder, spec, _, reference_spec = case_spec(run["case_id"])
            if spec["reference_case"] != name or run["seed"] != spec["search"]["seed"]:
                raise RuntimeError("inconsistent EvoESAP seed or reference case")
            evaluation = read_json(folder / "evaluation.json")
            if evaluation["source_sha256"] != run["source_evaluation_sha256"]:
                raise RuntimeError("EvoESAP statistics source hash differs")
            if canonical_sha256(evaluation["scores"]) != canonical_sha256(run["scores"]):
                raise RuntimeError("EvoESAP statistics differ from individual test scores")
            rows.append(evaluation["scores"])
        metrics = tape.aggregate_scores(rows, originals[reference_spec["family"]]["scores"]["mean"])
        if group["n"] != len(rows) or group["seeds"] != [row["seed"] for row in group["runs"]]:
            raise RuntimeError("incorrect EvoESAP cohort size or seed list")
        for metric, values in metrics.items():
            for key, value in values.items():
                if not np.isclose(value, group["metrics"][metric][key], rtol=0, atol=1e-9):
                    raise RuntimeError(f"incorrect EvoESAP statistic: {name}/{metric}/{key}")
        groups[name] = {"n": len(rows), "seeds": group["seeds"], "metrics": metrics}
    return {"std_definition": summary["std_definition"], "cases": groups}


def verify_release() -> dict:
    catalog = read_json(ARTIFACTS / "cases.json")
    checked = [verify_case(name) for name in catalog["cases"]]
    statistics_summary()
    return {
        "verified_runs": len(checked),
        "verified_candidates": sum(r["candidates"] for r in checked),
    }


def apply_mask(controller, family: str, mask: np.ndarray) -> None:
    if family == "granite":
        from scripts.run_pruning_28of32_taskaware_nsga2 import _routing_plan

        controller.apply(_routing_plan(mask))
    elif family == "ernie":
        controller.apply(mask)
    else:
        raise ValueError("unknown pruning-controller family")


def gold_examples(proxy, split: str) -> list:
    from dsai6707.esap import TeacherForcedExample

    return [
        TeacherForcedExample(
            question.task_name,
            question.requests[question.gold_index].full_ids,
            len(question.requests[question.gold_index].context_ids),
        )
        for question in proxy.questions[split]
    ]


def evaluate(name: str, model_path: Path | None, output: Path) -> dict:
    folder, _, _, spec = case_spec(name)
    tape.require_gpu(spec)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite an evaluation: {output}")
    models, tokenizer = tape.load_models(spec, model_path, replicas=1)
    mask = np.load(folder / "winner_mask.npy", allow_pickle=False)
    controller = tape.apply_controller(models[0], spec["family"], mask)
    try:
        return tape.evaluation(models[0], tokenizer, mask, output, label=f"released_{name}_evoesap")
    finally:
        controller.close()


def run(name: str, model_path: Path | None, output: Path, *, test: bool) -> dict:
    """Fresh inference-only search with the recorded adaptation and data split."""
    from dsai6707.esap import ESAPEvaluator
    from scripts import run_pruning_28of32_taskaware_nsga2 as shared

    _, spec, reference, reference_spec = case_spec(name)
    tape.require_gpu(reference_spec)
    if output.exists():
        raise FileExistsError(f"choose a new output directory: {output}")
    output.mkdir(parents=True)
    models, tokenizer = tape.load_models(reference_spec, model_path, replicas=1)
    model = models[0]
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("EvoESAP supports only frozen model inference")
    before = shared._parameter_identity(model)
    proxy = shared._load_proxy_data(
        {
            "data": reference_spec["data"],
            "search": {"seed": 6707},
            "parallel": {"proxy_batch_size": 32 if reference_spec["family"] == "granite" else 8},
        },
        tokenizer,
    )
    if canonical_sha256(proxy.evidence) != canonical_sha256(
        read_json(reference / "proxy_data.json")
    ):
        raise RuntimeError("downloaded/tokenized training data differs from the release")
    write_json(output / "proxy_data.json", proxy.evidence)
    family = reference_spec["family"]
    full_mask = np.ones(spec["shape"], dtype=bool)
    controller = tape.apply_controller(model, family, full_mask)
    started = perf_counter()
    try:
        evaluator = ESAPEvaluator(
            model,
            gold_examples(proxy, "search"),
            max_length=proxy.evidence["packing"]["search"]["max_length"],
            batch_size=8 if family == "granite" else 4,
        )
        write_json(output / "teacher_search.json", evaluator.cache_teacher())

        def objective(masks, metric=evaluator):
            values = []
            for mask in masks:
                apply_mask(controller, family, mask)
                metrics = metric.evaluate()
                write_json(output / "search_scores" / f"{mask_digest(mask)}.json", metrics)
                values.append(metrics["esap"])
            return values

        search_started = perf_counter()
        result = run_evoesap(
            objective,
            np.load(reference / "rankings.npy", allow_pickle=False),
            config=search_config(spec),
        )
        search_seconds = perf_counter() - search_started
        np.save(output / "candidate_masks.npy", result.masks, allow_pickle=False)
        write_json(
            output / "search.json",
            {
                "config": asdict(search_config(spec)),
                "esap": result.scores.tolist(),
                "trials": result.trials,
                "generation_history": result.generation_history,
                "search_winner_id": result.winner_id,
                "wall_seconds": search_seconds,
            },
        )
        del objective
        del evaluator
        apply_mask(controller, family, full_mask)
        evaluator = ESAPEvaluator(
            model,
            gold_examples(proxy, "holdout"),
            max_length=proxy.evidence["packing"]["holdout"]["max_length"],
            batch_size=8 if family == "granite" else 4,
        )
        write_json(output / "teacher_holdout.json", evaluator.cache_teacher())
        candidates = []
        for index in shortlist(result.scores):
            mask = result.masks[index]
            apply_mask(controller, family, mask)
            metrics = evaluator.evaluate()
            write_json(output / "holdout_scores" / f"{mask_digest(mask)}.json", metrics)
            candidates.append(
                {"candidate_id": index, "mask_sha256": mask_digest(mask), "esap": metrics["esap"]}
            )
        winner = select_holdout(candidates)
        mask = result.masks[winner["candidate_id"]]
        frozen = {
            "method": "evoesap_budget_matched",
            "frozen_at": utc_now(),
            "mask_sha256": mask_digest(mask),
            "protocol_sha256": sha256_file(ARTIFACTS / "protocol.json"),
            "search_seed": spec["search"]["seed"],
            "selection_split": "train-holdout",
            "selection_uses_test": False,
            "winner": winner,
            "holdout_candidates": candidates,
        }
        write_json(output / "frozen_selection.json", frozen)
        np.save(output / "winner_mask.npy", mask, allow_pickle=False)
        del evaluator
        apply_mask(controller, family, mask)
        scores = None
        if test:
            scores = tape.evaluation(
                model, tokenizer, mask, output / "test.json", label=f"fresh_{name}"
            )
            artifact = read_json(output / "test.json")
            artifact.update(
                freeze_sha256=sha256_file(output / "frozen_selection.json"),
                protocol_sha256=frozen["protocol_sha256"],
            )
            write_json(output / "test.json", artifact)
        if before != shared._parameter_identity(model):
            raise RuntimeError("EvoESAP inference changed model weights")
        execution = {
            "case": name,
            "search_wall_seconds": search_seconds,
            "total_wall_seconds": perf_counter() - started,
            "training_performed": False,
            "parameters_unchanged": True,
            "test": scores,
        }
        write_json(output / "execution.json", execution)
        return execution
    finally:
        controller.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "statistics", "replay", "run", "evaluate"))
    parser.add_argument("--case", default="all")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--test", action="store_true", help="test only after freezing train selection"
    )
    args = parser.parse_args()
    if args.action == "check":
        print(json.dumps(verify_release(), indent=2))
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
            else evaluate(args.case, args.model_path, args.output.resolve())
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

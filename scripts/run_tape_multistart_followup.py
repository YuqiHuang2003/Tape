#!/usr/bin/env python3
"""Post-hoc multi-start TAPE follow-up: Granite 75/50%, ERNIE 50%.

Only train-search enters evolution, only disjoint train-holdout selects masks.
Every baseline remains eligible. Official test is read after a freeze barrier.
Existing test results motivated this follow-up; it is explicitly exploratory.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import numpy as np

from dsai6707.io import read_json, sha256_file, utc_now, write_json
from dsai6707.pruning_global_budget_search import run_global_budget_nsga2
from dsai6707.pruning_multistart import (
    VARIANTS,
    distillation_metrics,
    initial_population,
    mask_digest,
    proxy_selection_key,
    search_config,
    shortlist,
)
from dsai6707.task_choice_proxy import canonical_sha256

DEFAULT_OUTPUT = ROOT / "results/runs/tape_multistart_followup_20261021"
TASKS = ("arc_easy", "openbookqa")


def _anchors(family: str, retained: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if family == "ernie":
        from scripts.run_report_review_evaluation import _family_methods

        rows = _family_methods("ernie")
        names = ("reap", "frequency", "ean_mean", "hybrid", "identity", "tape")
        anchors = {
            name: next(row["mask"] for row in rows if row["key"] == f"{name}_50") for name in names
        }
        import csv

        scores = np.zeros((27, 64))
        with (
            ROOT / "results/analysis/ernie_pruned_expert_choices/50pct_expert_status.csv"
        ).open() as f:
            for row in csv.DictReader(f):
                scores[int(row["layer"]) - 1, int(row["expert"])] = float(row["reap_score"])
    else:
        source = ROOT / "results/runs/granite_expert_pruning_28of32_seed6707_parallel2/baselines"
        scores = np.asarray(
            [row["scores"] for row in read_json(source / "reap_selection.json")["layers"]]
        )
        frequency = np.asarray(
            [row["scores"] for row in read_json(source / "frequency_selection.json")["layers"]]
        )

        def standardize(values: np.ndarray) -> np.ndarray:
            return (values - values.mean(axis=1, keepdims=True)) / np.maximum(
                values.std(axis=1, keepdims=True), 1e-12
            )

        hybrid = standardize(scores) + standardize(np.log1p(frequency))
        anchors = {}
        for name, values in (("reap", scores), ("frequency", frequency), ("hybrid", hybrid)):
            mask = np.zeros_like(values, dtype=bool)
            for layer, row in enumerate(values):
                order = np.lexsort((np.arange(len(row)), -row))
                mask[layer, order[:retained]] = True
            anchors[name] = mask
    rankings = np.stack([np.lexsort((np.arange(len(row)), -row)) for row in scores])
    return anchors, rankings


def prepare(output: Path) -> None:
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        return
    if (output / "evaluations").exists():
        raise RuntimeError("cannot change protocol after test evaluation")
    cases = {}
    for family, retained in (("granite", 24), ("granite", 16), ("ernie", 32)):
        key = f"{family}_{retained}"
        anchors, rankings = _anchors(family, retained)
        folder = output / key
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "anchors.npy", np.stack(list(anchors.values())), allow_pickle=False)
        np.save(folder / "rankings.npy", rankings, allow_pickle=False)
        config = search_config(layers=rankings.shape[0], retained=retained)
        initial = initial_population(anchors, config)
        np.save(folder / "initial_population.npy", initial.masks, allow_pickle=False)
        write_json(
            folder / "initial_population.json",
            {"contexts": initial.contexts, "population_sha256": initial.population_sha256},
        )
        cases[key] = {
            "family": family,
            "retained_per_layer_reference": retained,
            "shape": list(rankings.shape),
            "anchor_names": list(anchors),
            "anchor_digests": [mask_digest(m) for m in anchors.values()],
            "variants": [
                {
                    **variant,
                    "search": asdict(
                        search_config(
                            layers=rankings.shape[0], retained=retained, wide=variant["wide"]
                        )
                    ),
                }
                for variant in VARIANTS
            ],
        }
    write_json(
        protocol_path,
        {
            "schema_version": "tape-multistart-followup-v1",
            "created_at": utc_now(),
            "posthoc_exploratory": True,
            "training_performed": False,
            "test_previously_seen": True,
            "selection_uses_test": False,
            "primary_metric": "acc_norm",
            "cases": cases,
            "data": {
                "task_revisions": {
                    "arc_easy": "210d026faf9955653af8916fad021475a3f00453",
                    "openbookqa": "388097ea7776314e93a529163e0fea805b8a6454",
                },
                "search_samples_per_task": 64,
                "holdout_samples_per_task": 256,
            },
            "selection": (
                "macro acc_norm on disjoint train-holdout; "
                "ties min task acc_norm, normalized CE, digest"
            ),
            "baseline_protection": (
                "every direct baseline is included in holdout, " "even if dominated in search"
            ),
            "shortlist": (
                "each variant includes every initial anchor and "
                "four further masks by search acc_norm"
            ),
            "notes": [
                "Same train splits and exact initial masks across variants.",
                "Raw vs normalized-standard changes only score normalization/temperature.",
                (
                    "Wide variant jointly changes CE/KL weights, temperature and mutation; "
                    "not a factorial ablation."
                ),
                (
                    "Historical masks may have used some train rows previously; "
                    "current search/holdout are disjoint."
                ),
                "Granite retention is expert-slot retention, not total-model parameter retention.",
            ],
        },
    )
    print(f"PREPARED {protocol_path}", flush=True)


def _metrics(
    proxy: Any, split: str, candidate: np.ndarray, teacher: np.ndarray, variant: dict[str, Any]
) -> dict[str, Any]:
    return distillation_metrics(
        proxy.questions[split],
        candidate,
        teacher,
        cross_entropy_weight=variant["ce_weight"],
        kl_weight=variant["kl_weight"],
        choice_score_normalization=variant["normalization"],
        temperature=variant["temperature"],
    )


def _reuse_existing_test(
    family: str, mask: np.ndarray, *, name: str, frozen_path: Path
) -> dict[str, Any] | None:
    """Reuse identical, audited test masks only after the new winner is frozen."""
    if not frozen_path.is_file():
        raise RuntimeError("test access is forbidden before the freeze barrier")
    if family != "ernie":
        return None
    from scripts.run_report_review_evaluation import _compact_samples, _load_plan

    source = ROOT / "results/runs/report_review_test_20261021"
    manifest, masks, plan_sha = _load_plan(source, family)
    for row, existing in zip(manifest["methods"], masks, strict=True):
        if not np.array_equal(mask, existing):
            continue
        path = source / "evaluations" / family / f"{row['key']}.json"
        artifact = read_json(path)
        if (
            artifact["mask_sha256"] != mask_digest(mask)
            or artifact["frozen_plan_sha256"] != plan_sha
        ):
            raise RuntimeError("existing evaluation does not match its audited mask")
        _compact_samples(artifact)
        historical_role = artifact.pop("frozen_role", None)
        artifact.update(
            method_key=name,
            freeze_sha256=sha256_file(frozen_path),
            historical_frozen_role=historical_role,
            reused_test_source=str(path.relative_to(ROOT)),
            reused_test_source_sha256=sha256_file(path),
            selection_uses_test=False,
            posthoc_exploratory=True,
        )
        return artifact
    return None


def run(
    output: Path,
    key: str,
    *,
    test: bool,
    variants: tuple[dict[str, Any], ...] | None = None,
    config_factory: Any = search_config,
    shortlist_extra: int = 4,
    include_pareto_shortlist: bool = False,
    score_cache_source: Path | None = None,
) -> None:
    protocol = read_json(output / "protocol.json")
    case = protocol["cases"][key]
    family, retained = case["family"], int(case["retained_per_layer_reference"])
    folder = output / key
    experiment_variants = tuple(VARIANTS) if variants is None else tuple(variants)
    if not experiment_variants or shortlist_extra < 1:
        raise ValueError("at least one search variant and one extra shortlist slot are required")
    if family == "ernie":
        sys.path.insert(0, str(ROOT / "external/reap-official/.deps"))
    import torch
    import yaml

    from scripts import run_pruning_28of32_taskaware_nsga2 as shared

    shared._require_one_visible_gpu()
    torch.set_num_threads(4)
    if family == "granite":
        from dsai6707.granite import load_granite_model

        config = yaml.safe_load(
            (ROOT / "configs/expert_pruning_28of32_taskaware_nsga2.yaml").read_text()
        )
        model, tokenizer = load_granite_model(config["model"])
        replica, _ = load_granite_model(config["model"])
    else:
        from dsai6707.ernie_pruning import ErnieExpertMaskController
        from scripts import run_ernie21b_taskaware_nsga2 as ernie

        ernie._require_runtime()
        ernie._configure_torch(torch, seed=6707)
        path = ROOT / "external/reap-official/artifacts/models/ERNIE-4.5-21B-A3B-PT"
        tokenizer = ernie._load_tokenizer(path)
        model, _ = ernie._load_model(path, expected_experts=64)
    proxy_config = {
        "data": protocol["data"],
        "search": {"seed": 6707},
        "parallel": {"proxy_batch_size": 32 if family == "granite" else 8},
    }
    proxy = shared._load_proxy_data(proxy_config, tokenizer)
    data_digest = canonical_sha256(proxy.evidence)
    if score_cache_source is not None:
        source_proxy = read_json(score_cache_source / "proxy_data.json")
        if canonical_sha256(source_proxy) != data_digest:
            raise RuntimeError("refusing external scores from a different proxy")
    if (folder / "proxy_data.json").exists():
        if canonical_sha256(read_json(folder / "proxy_data.json")) != data_digest:
            raise RuntimeError("refusing cached scores from a changed proxy")
    write_json(folder / "proxy_data.json", proxy.evidence)
    if family == "granite":
        evaluator = shared.ParallelTaskProxyEvaluator((model, replica), proxy)

        def apply(mask: np.ndarray) -> None:
            evaluator.lanes[0].controller.apply(shared._routing_plan(mask))

        models = (model, replica)
    else:
        controller = ErnieExpertMaskController(model, np.ones(tuple(case["shape"]), dtype=bool))
        scorer = ernie.SerialChoiceScorer(model, proxy)
        apply = controller.apply
        models = (model,)
    before = [shared._parameter_identity(m) for m in models]
    if any(p.requires_grad for m in models for p in m.parameters()):
        raise RuntimeError("inference-only models unexpectedly require gradients")
    state = {
        "status": "searching",
        "started_at": utc_now(),
        "case": key,
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "protocol_sha256": sha256_file(output / "protocol.json"),
        "data_sha256": data_digest,
        "new_physical_scores": 0,
        "cache_hits": 0,
        "external_cache_hits": 0,
        "external_cache_source": str(score_cache_source) if score_cache_source else None,
    }
    anchors = dict(
        zip(case["anchor_names"], np.load(folder / "anchors.npy", allow_pickle=False), strict=True)
    )
    if [mask_digest(mask) for mask in anchors.values()] != case["anchor_digests"]:
        raise RuntimeError("frozen baseline masks changed")
    rankings = np.load(folder / "rankings.npy", allow_pickle=False)

    def score_many(masks: list[np.ndarray], split: str) -> list[np.ndarray]:
        cache = folder / "scores" / split
        cache.mkdir(parents=True, exist_ok=True)
        values: dict[str, np.ndarray] = {}
        missing = []
        for mask in masks:
            digest = mask_digest(mask)
            path = cache / f"{digest}.npy"
            if path.exists():
                values[digest] = np.load(path, allow_pickle=False)
                state["cache_hits"] += 1
            elif (
                score_cache_source is not None
                and (score_cache_source / "scores" / split / f"{digest}.npy").exists()
            ):
                source_path = score_cache_source / "scores" / split / f"{digest}.npy"
                values[digest] = np.load(source_path, allow_pickle=False)
                state["external_cache_hits"] += 1
            elif digest not in {mask_digest(m) for m in missing}:
                missing.append(mask)
            if digest in values and (
                values[digest].shape != (len(proxy.requests[split]),)
                or not np.isfinite(values[digest]).all()
            ):
                raise RuntimeError("invalid cached choice log-likelihoods")
        width = 2 if family == "granite" else 1
        for start in range(0, len(missing), width):
            chunk = missing[start : start + width]
            if family == "granite":
                raw = evaluator.score_many(chunk, split=split, parallel=len(chunk) == 2)
            else:
                apply(chunk[0])
                raw = (scorer.score(split),)
            for mask, result in zip(chunk, raw, strict=True):
                digest = mask_digest(mask)
                values[digest] = result.choice_log_likelihoods
                np.save(cache / f"{digest}.npy", result.choice_log_likelihoods, allow_pickle=False)
                state["new_physical_scores"] += 1
            state.update(last_split=split, updated_at=utc_now())
            write_json(folder / "state.json", state)
        return [values[mask_digest(mask)] for mask in masks]

    teachers = {
        split: score_many([np.ones(tuple(case["shape"]), dtype=bool)], split)[0]
        for split in ("search", "holdout")
    }
    selected: dict[str, dict[str, Any]] = {}
    mask_lookup: dict[str, np.ndarray] = {}
    for variant in experiment_variants:
        name = variant["name"]
        arm = folder / name
        arm.mkdir(parents=True, exist_ok=True)
        config = config_factory(layers=rankings.shape[0], retained=retained, wide=variant["wide"])
        initial = initial_population(anchors, config)
        prepared_population = folder / "initial_population.npy"
        if prepared_population.exists() and not np.array_equal(
            initial.masks, np.load(prepared_population, allow_pickle=False)
        ):
            raise RuntimeError("runtime initial population differs from the prepared population")
        search_metrics = []

        def objective_many(
            masks: Any,
            *,
            variant: dict[str, Any] = variant,
            search_metrics: list[Any] = search_metrics,
            name: str = name,
            config: Any = config,
        ) -> list[tuple[float, float]]:
            scores = score_many(list(masks), "search")
            metrics = [
                _metrics(proxy, "search", row, teachers["search"], variant) for row in scores
            ]
            search_metrics.extend(metrics)
            best_acc = max(
                np.mean([t["acc_norm"] for t in m["tasks"].values()]) for m in search_metrics
            )
            print(
                f"SEARCH {key} {name} {len(search_metrics)}/{config.evaluation_budget} "
                f"best_acc_norm={best_acc:.6f}",
                flush=True,
            )
            return [tuple(row["task_objectives"]) for row in metrics]

        started = perf_counter()
        result = run_global_budget_nsga2(
            objective_many, rankings, mode="joint", config=config, initial_population=initial
        )
        np.save(arm / "candidate_masks.npy", result.masks, allow_pickle=False)
        write_json(
            arm / "search.json",
            {
                "trials": result.trials,
                "metrics": search_metrics,
                "final_population_ids": result.final_population_ids,
                "wall_seconds": perf_counter() - started,
                "evaluation_budget": config.evaluation_budget,
            },
        )
        ids = shortlist(
            search_metrics,
            baseline_ids=range(len(set(case["anchor_digests"]))),
            limit=len(set(case["anchor_digests"])) + shortlist_extra,
        )
        if include_pareto_shortlist:
            ids = tuple(dict.fromkeys((*ids, *result.shortlist.candidate_ids)))
        for index in ids:
            digest = mask_digest(result.masks[index])
            mask_lookup[digest] = result.masks[index]
            row = selected.setdefault(digest, {"mask_sha256": digest, "sources": []})
            row["sources"].append({"variant": name, "candidate_id": int(index)})
    # One common, metric-aligned holdout rule is applied across all search arms.
    state["status"] = "holdout_selection"
    write_json(folder / "state.json", state)
    print(f"HOLDOUT_START {key} candidates={len(selected)}", flush=True)
    digests = list(selected)
    held_scores = score_many([mask_lookup[d] for d in digests], "holdout")
    selection_variant = experiment_variants[1 if len(experiment_variants) > 1 else 0]
    for digest, scores in zip(digests, held_scores, strict=True):
        selected[digest]["metrics"] = _metrics(
            proxy, "holdout", scores, teachers["holdout"], selection_variant
        )
    winner = min(
        selected.values(),
        key=lambda row: proxy_selection_key(row["metrics"], key=row["mask_sha256"]),
    )
    baselines = {name: selected[mask_digest(mask)] for name, mask in anchors.items()}
    best_baseline = min(
        baselines.values(),
        key=lambda row: proxy_selection_key(row["metrics"], key=row["mask_sha256"]),
    )
    if proxy_selection_key(winner["metrics"]) > proxy_selection_key(best_baseline["metrics"]):
        raise RuntimeError("baseline protection failed")
    frozen = {
        "schema_version": "tape-multistart-winner-v1",
        "frozen_at": utc_now(),
        "protocol_sha256": state["protocol_sha256"],
        "primary_metric": "acc_norm",
        "selection_split": "train-holdout",
        "selection_uses_test": False,
        "posthoc_exploratory": True,
        "winner": winner,
        "winner_equals_baselines": [
            n for n, m in anchors.items() if mask_digest(m) == winner["mask_sha256"]
        ],
        "baselines": baselines,
        "holdout_candidates": list(selected.values()),
    }
    frozen_path = folder / "frozen_selection.json"
    np.save(folder / "winner_mask.npy", mask_lookup[winner["mask_sha256"]], allow_pickle=False)
    if frozen_path.exists():
        old = read_json(frozen_path)
        if old["winner"]["mask_sha256"] != winner["mask_sha256"]:
            raise RuntimeError("cannot change an already frozen winner")
    else:
        write_json(frozen_path, frozen)
    print(
        f"FROZEN {key} winner={winner['mask_sha256'][:12]} "
        f"holdout_acc_norm={-proxy_selection_key(winner['metrics'])[0]:.6f} "
        f"baseline_equal={frozen['winner_equals_baselines']}",
        flush=True,
    )
    state["status"] = "winner_frozen"
    write_json(folder / "state.json", state)
    if test:
        from dsai6707.evaluation import evaluate_zero_shot
        from scripts.evaluate_reap_ernie import _evaluation_config
        from scripts.run_report_review_evaluation import _compact_samples

        fixed = {**anchors, "tape_multistart": mask_lookup[winner["mask_sha256"]]}
        completed: dict[str, Any] = {}
        for name, mask in fixed.items():
            target = folder / "evaluations" / f"{name}.json"
            digest = mask_digest(mask)
            if target.exists():
                artifact = read_json(target)
                if artifact["mask_sha256"] != digest or artifact["freeze_sha256"] != sha256_file(
                    frozen_path
                ):
                    raise RuntimeError("test artifact does not match the frozen selection")
            elif digest in completed:
                artifact = {**completed[digest], "method_key": name}
                write_json(target, artifact)
            else:
                state.update(status="test_evaluation", test_method=name)
                write_json(folder / "state.json", state)
                artifact = _reuse_existing_test(family, mask, name=name, frozen_path=frozen_path)
                if artifact is None:
                    print(f"TEST_START {key} {name}", flush=True)
                    apply(mask)
                    artifact = evaluate_zero_shot(
                        model,
                        tokenizer,
                        _evaluation_config("test", batch_size=16, log_samples=True),
                        method=f"multistart_{key}_{name}",
                        seed=6707,
                    )
                else:
                    print(f"TEST_REUSE {key} {name}", flush=True)
                artifact.update(
                    mask_sha256=digest,
                    freeze_sha256=sha256_file(frozen_path),
                    method_key=name,
                    selection_uses_test=False,
                    posthoc_exploratory=True,
                )
                write_json(target, artifact)
            completed[digest] = artifact
            write_json(folder / "evaluations" / f"{name}_samples.json", _compact_samples(artifact))
            mean = np.mean([artifact["tasks"][task]["acc_norm"] for task in TASKS])
            print(f"TEST_DONE {key} {name} acc_norm={mean:.8f}", flush=True)
    if [shared._parameter_identity(m) for m in models] != before:
        raise RuntimeError("model weights changed during inference-only experiment")
    state.update(
        status="complete" if test else "winner_frozen",
        completed_at=utc_now(),
        parameters_unchanged=True,
        training_performed=False,
    )
    write_json(folder / "state.json", state)
    if family == "granite":
        evaluator.close()
    else:
        controller.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case", choices=("granite_24", "granite_16", "ernie_32"))
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output.resolve())
    else:
        if args.case is None:
            parser.error("run requires --case")
        run(args.output.resolve(), args.case, test=args.test)


if __name__ == "__main__":
    main()

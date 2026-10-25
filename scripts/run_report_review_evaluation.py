#!/usr/bin/env python3
"""Freeze existing masks, then evaluate them once on official test splits.

This supplement performs no search or training.  A CPU-only preparation stage
fixes all method roles and masks before any new test score is computed.  Each
family is evaluated serially using one model on one visible GPU.  Per-question
correctness is retained for paired comparisons and sampling uncertainty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results/runs/report_review_test_20261021"
for folder in (ROOT, ROOT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mask_digest(mask: Any) -> str:
    return hashlib.sha256(str(mask.shape).encode() + mask.tobytes()).hexdigest()


def _selection_mask(path: Path, *, experts: int) -> Any:
    import numpy as np

    payload = _read(path)
    rows = payload["layers"]
    mask = np.zeros((len(rows), experts), dtype=bool)
    for index, row in enumerate(rows):
        if row["layer"] != index:
            raise ValueError(f"Unordered mask layers: {path}")
        retained = row.get("retained_experts", row.get("retained"))
        if not isinstance(retained, list) or len(set(retained)) != len(retained):
            raise ValueError(f"Invalid retained experts: {path}")
        mask[index, retained] = True
    return mask


def _projection(mask: Any, scores: Any) -> Any:
    import numpy as np

    projected = np.zeros_like(mask)
    for layer, row in enumerate(scores):
        order = np.lexsort((np.arange(len(row)), -np.asarray(row)))
        projected[layer, order[: int(mask[layer].sum())]] = True
    return projected


def _family_methods(family: str) -> list[dict[str, Any]]:
    import numpy as np

    runs = ROOT / "results/runs"
    methods: list[dict[str, Any]] = []

    def add(key: str, label: str, role: str, mask: Any, source: str, *, retention: float) -> None:
        values = np.ascontiguousarray(mask, dtype=bool)
        expected_shape = (27, 64) if family == "ernie" else (24, 32)
        if values.shape != expected_shape:
            raise ValueError(f"Unexpected {key} mask shape: {values.shape}")
        expected_total = int(round(values.size * retention))
        if int(values.sum()) != expected_total:
            raise ValueError(f"Invalid {key} global budget")
        if int(values.sum(axis=1).min()) < (6 if family == "ernie" else 8):
            raise ValueError(f"Insufficient experts for top-k: {key}")
        methods.append(
            {
                "key": key,
                "label": label,
                "role": role,
                "retention": retention,
                "source": source,
                "mask": values,
            }
        )

    if family == "ernie":
        add("original", "Original", "reference", np.ones((27, 64)), "original", retention=1)
        for retained, final_id, role in ((48, 251, "primary"), (32, 231, "diagnostic")):
            rate = retained / 64
            suffix = "75" if retained == 48 else "50"
            historical = runs / (f"ernie45_21b_a3b_reap{suffix}_taskaware_nsga2_quick_seed6707")
            anchor_rows = _read(historical / "anchors.json")["layers"]
            reap = None
            for key, label in (
                ("reap", "REAP"),
                ("frequency", "Frequency"),
                ("ean_mean", "EAN"),
                ("hybrid", "REAP+Frequency"),
            ):
                mask = np.zeros((27, 64), dtype=bool)
                for layer, row in enumerate(anchor_rows):
                    if row["mask_row"] != layer:
                        raise ValueError("Invalid historical anchor order")
                    mask[layer, row[f"{key}_retained"]] = True
                add(
                    f"{key}_{suffix}",
                    label,
                    "baseline",
                    mask,
                    str((historical / "anchors.json").relative_to(ROOT)),
                    retention=rate,
                )
                if key == "reap":
                    reap = mask
            run = runs / f"ernie45_21b_a3b_anchored_joint_{retained}of64_seed6707"
            candidates = np.load(run / "search/candidate_masks.npy", allow_pickle=False)
            if not np.array_equal(candidates[1], reap):
                raise ValueError("Historical and anchored REAP masks disagree")
            add(
                f"tape_{suffix}",
                "TAPE",
                role,
                candidates[final_id],
                f"{run.relative_to(ROOT)}/search/candidate_masks.npy[{final_id}]",
                retention=rate,
            )
            add(
                f"identity_{suffix}",
                "Identity-only",
                "ablation",
                candidates[0],
                f"{run.relative_to(ROOT)}/search/candidate_masks.npy[0]",
                retention=rate,
            )
            if retained == 48:
                first = runs / "ernie45_21b_a3b_global_budget_joint_48of64_seed6707"
                selection = _read(first / "allocation_only/diagnostic_selection.json")
                file = first / "search/allocation_only/candidate_masks.npy"
                masks = np.load(file, allow_pickle=False)
                add(
                    "allocation_75",
                    "Allocation-only",
                    "diagnostic_ablation",
                    masks[int(selection["candidate_id"])],
                    f"{file.relative_to(ROOT)}[{selection['candidate_id']}]",
                    retention=rate,
                )
            # This fixed projection isolates identity at the final layer capacities.
            # The scores below are read from the already audited expert-status CSV.
            import csv

            expert_csv = ROOT / (
                "results/analysis/ernie_pruned_expert_choices/" f"{suffix}pct_expert_status.csv"
            )
            scores = np.zeros((27, 64), dtype=float)
            with expert_csv.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    scores[int(row["layer"]) - 1, int(row["expert"])] = float(row["reap_score"])
            add(
                f"projection_{suffix}",
                "REAP at TAPE layer capacities",
                "ablation",
                _projection(candidates[final_id], scores),
                str(expert_csv.relative_to(ROOT)),
                retention=rate,
            )
    else:
        add("original", "Original", "reference", np.ones((24, 32)), "original", retention=1)
        old = runs / "granite_expert_pruning_28of32_seed6707_parallel2"
        for key, label in (("reap", "REAP"), ("frequency", "Frequency")):
            file = old / f"baselines/{key}_selection.json"
            add(
                key,
                label,
                "baseline",
                _selection_mask(file, experts=32),
                str(file.relative_to(ROOT)),
                retention=0.875,
            )
        sources = (
            (
                "nll_ea",
                "NLL-EA",
                "baseline",
                old / "search/importance_guided_swap/best_selection.json",
            ),
            (
                "scalar_ea",
                "Scalar-EA",
                "algorithm_ablation",
                runs
                / "granite_expert_pruning_28of32_taskaware_scalar_ea_seed6707"
                / "rerank/winner_selection.json",
            ),
            (
                "identity",
                "Identity-only / NSGA-II",
                "ablation",
                runs
                / "granite_expert_pruning_28of32_taskaware_nsga2_seed6707"
                / "rerank/winner_selection.json",
            ),
            (
                "tape",
                "TAPE",
                "primary",
                runs
                / "granite_1b_global_budget_joint_pilot_seed6707"
                / "joint/winner_selection.json",
            ),
            (
                "allocation",
                "Allocation-only",
                "ablation",
                runs
                / "granite_1b_global_budget_joint_pilot_seed6707"
                / "allocation_only/winner_selection.json",
            ),
        )
        for key, label, role, file in sources:
            add(
                key,
                label,
                role,
                _selection_mask(file, experts=32),
                str(file.relative_to(ROOT)),
                retention=0.875,
            )
    return methods


def prepare(output: Path) -> None:
    import numpy as np

    from dsai6707.io import utc_now, write_json

    manifest_path = output / "frozen_plan.json"
    if manifest_path.exists():
        print(f"Frozen plan already exists: {manifest_path}", flush=True)
        return
    if (output / "evaluations").exists():
        raise ValueError("Cannot freeze a new plan after evaluations exist")
    families = {}
    for family in ("ernie", "granite"):
        rows = _family_methods(family)
        masks_path = output / f"{family}_masks.npy"
        output.mkdir(parents=True, exist_ok=True)
        np.save(masks_path, np.stack([row["mask"] for row in rows]), allow_pickle=False)
        families[family] = {
            "masks_file": masks_path.name,
            "masks_file_sha256": _digest(masks_path),
            "methods": [
                {
                    **{k: v for k, v in row.items() if k != "mask"},
                    "mask_sha256": _mask_digest(row["mask"]),
                }
                for row in rows
            ],
        }
    write_json(
        manifest_path,
        {
            "schema_version": "report-review-test-plan-v1",
            "frozen_at": utc_now(),
            "selection_uses_test": False,
            "training_performed": False,
            "samples": {"arc_easy": 2376, "openbookqa": 500},
            "primary_metric": "acc",
            "aggregation": "unweighted_mean_of_task_accuracies",
            "families": families,
            "notes": [
                "All existing primary/diagnostic roles are preserved on test.",
                "No mask, CE/KL weight, or candidate is chosen from test results.",
                "Sampling intervals quantify test-question variation, not search-run variance.",
                "Same-capacity REAP projections are fixed before test evaluation.",
            ],
        },
    )
    print(f"Frozen {sum(len(v['methods']) for v in families.values())} masks: {manifest_path}")


def _load_plan(output: Path, family: str) -> tuple[dict[str, Any], Any, str]:
    import numpy as np

    path = output / "frozen_plan.json"
    plan = _read(path)
    manifest = plan["families"][family]
    masks_path = output / manifest["masks_file"]
    if _digest(masks_path) != manifest["masks_file_sha256"]:
        raise ValueError("Frozen masks file changed")
    masks = np.load(masks_path, allow_pickle=False)
    if len(masks) != len(manifest["methods"]):
        raise ValueError("Frozen mask count changed")
    for row, mask in zip(manifest["methods"], masks, strict=True):
        if _mask_digest(mask) != row["mask_sha256"]:
            raise ValueError("Frozen mask digest changed")
    return manifest, masks, _digest(path)


def _compact_samples(artifact: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for task in ("arc_easy", "openbookqa"):
        samples = artifact["raw_lm_eval"]["samples"][task]
        rows = [
            {
                "doc_id": int(row["doc_id"]),
                "doc_hash": row["doc_hash"],
                "acc": int(row["acc"]),
                "acc_norm": int(row["acc_norm"]),
            }
            for row in samples
        ]
        rows.sort(key=lambda row: row["doc_id"])
        if len(rows) != (2376 if task == "arc_easy" else 500):
            raise ValueError(f"Incomplete per-question results: {task}")
        for metric in ("acc", "acc_norm"):
            if not math.isclose(
                sum(row[metric] for row in rows) / len(rows),
                artifact["tasks"][task][metric],
                abs_tol=1e-12,
            ):
                raise ValueError(f"Per-question and aggregate {metric} disagree")
        result[task] = rows
    return result


def run(output: Path, family: str, *, batch_size: int, selected_keys: list[str] | None) -> None:
    from dsai6707.io import utc_now, write_json

    manifest, masks, plan_sha = _load_plan(output, family)
    methods = manifest["methods"]
    completed = output / "evaluations" / family
    completed.mkdir(parents=True, exist_ok=True)
    if selected_keys is not None and not set(selected_keys) <= {r["key"] for r in methods}:
        raise ValueError("Unknown requested method key")
    pending = []
    for row, mask in zip(methods, masks, strict=True):
        if selected_keys is not None and row["key"] not in selected_keys:
            continue
        target = completed / f"{row['key']}.json"
        if target.exists():
            old = _read(target)
            if old["frozen_plan_sha256"] != plan_sha or old["mask_sha256"] != row["mask_sha256"]:
                raise ValueError("Refusing to resume an evaluation from a different frozen plan")
        else:
            pending.append((row, mask, target))
    if not pending:
        print(f"All requested {family} test evaluations are already complete", flush=True)
        summarize(output)
        return

    if family == "ernie":
        deps = ROOT / "external/reap-official/.deps"
        sys.path.insert(0, str(deps))
    import torch

    from scripts.evaluate_reap_ernie import _evaluation_config

    if family == "ernie":
        from dsai6707.ernie_pruning import ErnieExpertMaskController
        from scripts import run_ernie21b_taskaware_nsga2 as experiment

        experiment._require_runtime()
        experiment._configure_torch(torch, seed=6707)
        path = ROOT / "external/reap-official/artifacts/models/ERNIE-4.5-21B-A3B-PT"
        tokenizer = experiment._load_tokenizer(path)
        model, _ = experiment._load_model(path, expected_experts=64)
        import numpy as np

        controller = ErnieExpertMaskController(model, np.ones((27, 64), dtype=bool))
        apply = controller.apply
    else:
        import yaml

        from dsai6707.granite import load_granite_model
        from dsai6707.pruning import ExpertSelectionPlan, GranitePruningController
        from scripts.run_pruning_28of32_taskaware_nsga2 import _require_one_visible_gpu

        _require_one_visible_gpu()
        config = yaml.safe_load(
            (ROOT / "configs/expert_pruning_28of32_taskaware_nsga2.yaml").read_text()
        )
        model, tokenizer = load_granite_model(config["model"])
        controller = GranitePruningController(
            model, ExpertSelectionPlan.all_experts(num_layers=24, num_experts=32)
        )

        def apply(mask: Any) -> None:
            import numpy as np

            controller.apply(
                ExpertSelectionPlan.from_retained(
                    [np.flatnonzero(row).tolist() for row in mask], num_experts=32
                )
            )

    torch.set_num_threads(4)
    from dsai6707.evaluation import evaluate_zero_shot

    config = _evaluation_config("test", batch_size=batch_size, log_samples=True)
    write_json(
        output / f"{family}_state.json",
        {
            "status": "running",
            "started_at": utc_now(),
            "frozen_plan_sha256": plan_sha,
            "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pending_keys": [row["key"] for row, _, _ in pending],
        },
    )
    try:
        for row, mask, target in pending:
            print(f"TEST_START {family} {row['key']} role={row['role']}", flush=True)
            apply(mask)
            artifact = evaluate_zero_shot(
                model, tokenizer, config, method=f"report_test_{family}_{row['key']}", seed=6707
            )
            artifact.update(
                {
                    "frozen_plan_sha256": plan_sha,
                    "mask_sha256": row["mask_sha256"],
                    "frozen_role": row["role"],
                    "method_key": row["key"],
                    "selection_uses_test": False,
                    "evaluation_split": "test",
                    "completed_at": utc_now(),
                }
            )
            write_json(completed / f"{row['key']}_samples.json", _compact_samples(artifact))
            # Preserve the full native result too, including responses and dataset evidence.
            write_json(target, artifact)
            print(
                f"TEST_DONE {family} {row['key']} mean={artifact['mean_primary_acc']:.8f}",
                flush=True,
            )
            summarize(output)
    finally:
        controller.close()
    write_json(
        output / f"{family}_state.json",
        {
            "status": "requested_evaluations_complete",
            "completed_at": utc_now(),
            "frozen_plan_sha256": plan_sha,
        },
    )


def paired_comparison(
    candidate: dict[str, Any], reference: dict[str, Any], *, metric: str = "acc"
) -> dict[str, Any]:
    import numpy as np

    if metric not in ("acc", "acc_norm"):
        raise ValueError(f"Unsupported accuracy metric: {metric}")
    diffs = []
    tasks = {}
    for task in ("arc_easy", "openbookqa"):
        left, right = candidate[task], reference[task]
        if [(r["doc_id"], r["doc_hash"]) for r in left] != [
            (r["doc_id"], r["doc_hash"]) for r in right
        ]:
            raise ValueError("Paired comparison uses different test questions")
        diff = np.asarray([a[metric] - b[metric] for a, b in zip(left, right, strict=True)])
        wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
        n, smaller = wins + losses, min(wins, losses)
        if n:
            tail = sum(
                math.exp(
                    math.lgamma(n + 1)
                    - math.lgamma(k + 1)
                    - math.lgamma(n - k + 1)
                    - n * math.log(2)
                )
                for k in range(smaller + 1)
            )
            p = min(1.0, 2.0 * tail)
        else:
            p = 1.0
        tasks[task] = {
            "accuracy_difference": float(diff.mean()),
            "candidate_only_correct": wins,
            "reference_only_correct": losses,
            "mcnemar_exact_p": p,
        }
        diffs.append(diff)
    rng = np.random.default_rng(6707)
    boot = np.zeros(5000)
    for values in diffs:
        for start in range(0, len(boot), 100):
            indices = rng.integers(0, len(values), size=(100, len(values)))
            boot[start : start + 100] += values[indices].mean(axis=1) / 2
    return {
        "metric": metric,
        "mean_accuracy_difference": float(np.mean([d.mean() for d in diffs])),
        "paired_stratified_bootstrap_95pct": np.quantile(boot, [0.025, 0.975]).tolist(),
        "bootstrap_replicates": 5000,
        "uncertainty_scope": "test_question_sampling_only_not_search_run_variance",
        "tasks": tasks,
    }


def _load_scoring_amendment(output: Path) -> dict[str, Any]:
    amendment = _read(output / "scoring_amendment.json")
    if amendment["frozen_plan_sha256"] != _digest(output / "frozen_plan.json"):
        raise ValueError("Scoring amendment refers to a different frozen plan")
    if amendment["primary_metric"] != "acc_norm":
        raise ValueError("Scoring amendment does not specify the paper-aligned metric")
    for field in (
        "masks_unchanged",
        "method_roles_unchanged",
        "search_and_selection_unchanged",
        "raw_acc_preserved",
        "frozen_plan_preserved",
    ):
        if amendment.get(field) is not True:
            raise ValueError(f"Scoring amendment does not preserve {field}")
    if amendment.get("selection_uses_test") is not False:
        raise ValueError("Scoring amendment must not enable test-based selection")
    return amendment


def summarize(output: Path, *, primary_metric: str = "acc") -> dict[str, Any]:
    from dsai6707.io import utc_now, write_json

    if not (output / "frozen_plan.json").exists():
        raise ValueError("Prepare the frozen plan first")
    if primary_metric not in ("acc", "acc_norm"):
        raise ValueError(f"Unsupported accuracy metric: {primary_metric}")
    amendment = _load_scoring_amendment(output) if primary_metric == "acc_norm" else None
    families = {}
    for family in ("ernie", "granite"):
        manifest, _, plan_sha = _load_plan(output, family)
        folder = output / "evaluations" / family
        original_path = folder / "original.json"
        original = _read(original_path) if original_path.exists() else None
        original_mean = (
            sum(original["tasks"][task][primary_metric] for task in ("arc_easy", "openbookqa")) / 2
            if original
            else None
        )
        rows, pairs = [], {}
        for method in manifest["methods"]:
            file = folder / f"{method['key']}.json"
            if not file.exists():
                continue
            value = _read(file)
            if value["frozen_plan_sha256"] != plan_sha:
                raise ValueError("Evaluation and frozen plan disagree")
            if value["mask_sha256"] != method["mask_sha256"]:
                raise ValueError("Evaluation and frozen mask disagree")
            if value["frozen_role"] != method["role"]:
                raise ValueError("Evaluation and frozen role disagree")
            accuracy = {
                task: value["tasks"][task][primary_metric] for task in ("arc_easy", "openbookqa")
            }
            mean_accuracy = sum(accuracy.values()) / 2
            rows.append(
                {
                    **method,
                    "metric": primary_metric,
                    "accuracy": accuracy,
                    "mean_accuracy": mean_accuracy,
                    "mean_accuracy_retention": mean_accuracy / original_mean
                    if original_mean
                    else None,
                    "raw_accuracy": {
                        task: value["tasks"][task]["acc"] for task in ("arc_easy", "openbookqa")
                    },
                    "raw_mean_accuracy": value["mean_primary_acc"],
                    "artifact": str(file.relative_to(output)),
                }
            )
        comparisons = (
            [
                (f"tape_{s}", f"{r}_{s}")
                for s in ("75", "50")
                for r in ("reap", "frequency", "ean_mean", "hybrid", "identity", "projection")
            ]
            if family == "ernie"
            else [("tape", r) for r in ("reap", "frequency", "nll_ea", "allocation", "identity")]
            + [("identity", "scalar_ea")]
        )
        if family == "ernie":
            comparisons += [("identity_50", "reap_50"), ("tape_75", "allocation_75")]
            comparisons += [(f"projection_{s}", f"reap_{s}") for s in ("75", "50")]
        for left, right in comparisons:
            left_path, right_path = (
                folder / f"{left}_samples.json",
                folder / f"{right}_samples.json",
            )
            if left_path.exists() and right_path.exists():
                pairs[f"{left}_vs_{right}"] = paired_comparison(
                    _read(left_path), _read(right_path), metric=primary_metric
                )
        families[family] = {
            "evaluations": rows,
            "planned_count": len(manifest["methods"]),
            "completed_count": len(rows),
            "paired_comparisons": pairs,
        }
    result = {
        "schema_version": "report-review-test-summary-v1",
        "updated_at": utc_now(),
        "split": "test",
        "primary_metric": primary_metric,
        "aggregation": "unweighted_mean_of_task_accuracies",
        "selection_uses_test": False,
        "families": families,
    }
    if amendment is not None:
        result["scoring_amendment"] = amendment
    file_name = "paper_aligned_analysis.json" if primary_metric == "acc_norm" else "analysis.json"
    write_json(output / file_name, result)
    print(
        f"TEST_SUMMARY metric={primary_metric} "
        + " ".join(f"{k}={v['completed_count']}/{v['planned_count']}" for k, v in families.items()),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "summarize"))
    parser.add_argument("--family", choices=("ernie", "granite"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument(
        "--paper-aligned",
        action="store_true",
        help="summarize acc_norm with the recorded user-requested scoring amendment",
    )
    args = parser.parse_args()
    if args.paper_aligned and args.stage != "summarize":
        parser.error("--paper-aligned is a reporting option for summarize only")
    output = args.output_dir.resolve()
    if args.stage == "prepare":
        prepare(output)
    elif args.stage == "run":
        if args.family is None or args.batch_size < 1:
            parser.error("run requires --family and a positive batch size")
        run(output, args.family, batch_size=args.batch_size, selected_keys=args.methods)
    else:
        summarize(output, primary_metric="acc_norm" if args.paper_aligned else "acc")


if __name__ == "__main__":
    main()

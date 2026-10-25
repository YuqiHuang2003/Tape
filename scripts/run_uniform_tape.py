#!/usr/bin/env python3
"""Unified, inference-only TAPE reruns; historical experiments remain immutable."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np

from dsai6707.io import read_json, sha256_file, utc_now, write_json
from dsai6707.pruning_global_budget import (
    GlobalBudgetCrossover,
    build_mask_from_layer_rankings,
    global_budget_mask_sha256,
    mutate_budget_transfer,
    mutate_identity_swap,
    validate_global_budget_mask,
)
from dsai6707.pruning_global_budget_search import (
    GlobalBudgetInitialPopulation,
    run_global_budget_nsga2,
)
from dsai6707.pruning_multistart import mask_digest, proxy_selection_key, search_config
from scripts import run_tape_multistart_followup as base

OUTPUT = ROOT / "results/runs/uniform_tape_20261023"
MAIN_CASES = (("granite", 24), ("granite", 16), ("ernie", 48), ("ernie", 32))
VARIANT = {
    "name": "uniform_norm",
    "normalization": "characters",
    "temperature": 0.25,
    "ce_weight": 0.75,
    "kl_weight": 0.25,
    "wide": False,
}


def config_for(layers: int, retained: int, mode: str = "joint") -> Any:
    config = replace(search_config(layers=layers, retained=retained), generations=15)
    if mode == "identity_only":
        config = replace(
            config,
            joint_mutation_pairs=((0, 1), (0, 2), (0, 4)),
            joint_mutation_probabilities=(0.50, 0.30, 0.20),
        )
    elif mode == "allocation_only":
        config = replace(
            config,
            transfer_counts=(1, 2, 4),
            transfer_probabilities=(6 / 11, 3 / 11, 2 / 11),
        )
    elif mode != "joint":
        raise ValueError("unknown search mode")
    return config


def bounds_for(config: Any) -> dict[str, int]:
    return {
        key: getattr(config, key)
        for key in ("total_retained", "min_retained_per_layer", "max_retained_per_layer")
    }


def initial_population(
    anchors: dict[str, np.ndarray], config: Any, mode: str, *, rankings: np.ndarray | None = None
) -> GlobalBudgetInitialPopulation:
    bounds = bounds_for(config)
    masks, contexts, known = [], [], set()
    for name, mask in anchors.items():
        validate_global_budget_mask(mask, **bounds)
        digest = mask_digest(mask)
        if digest not in known:
            masks.append(mask.copy())
            contexts.append({"initial_source": name, "exact_baseline": True})
            known.add(digest)
    parents = [(c["initial_source"], m) for c, m in zip(contexts, masks, strict=True)]
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0x554E4946]))
    for attempt in range(4096):
        if len(masks) == config.population_size:
            break
        name, parent = parents[attempt % len(parents)]
        count = (1, 2, 4)[(attempt // len(parents)) % 3]
        child = parent.copy()
        for _ in range(count):
            if mode == "allocation_only":
                child = mutate_budget_transfer(child, rng=rng, **bounds).mask
            else:
                child = mutate_identity_swap(child, rng=rng, **bounds).mask
        if mode == "joint" and (attempt // len(parents)) % 2:
            child = mutate_budget_transfer(child, rng=rng, **bounds).mask
        if mode == "allocation_only":
            # Every anchor is already a REAP prefix; recover its fixed ranking
            # from the supplied preparation context, never from task scores.
            if rankings is None:
                raise ValueError("allocation-only initialization requires the fixed ranking")
            child = project(child, rankings, config)
        digest = mask_digest(child)
        if digest in known:
            continue
        known.add(digest)
        masks.append(child)
        contexts.append({"initial_source": name, "exact_baseline": False, "steps": count})
    if len(masks) != config.population_size:
        raise RuntimeError("cannot construct the frozen initial population")
    digests = [global_budget_mask_sha256(mask, **bounds) for mask in masks]
    digest = sha256(
        b"dsai6707-global-budget-population-v1\0" + "\n".join(digests).encode("ascii")
    ).hexdigest()
    return GlobalBudgetInitialPopulation(np.stack(masks), tuple(contexts), digest)


def project(mask: np.ndarray, rankings: np.ndarray, config: Any) -> np.ndarray:
    return build_mask_from_layer_rankings(rankings, mask.sum(axis=1), **bounds_for(config))


def identity_crossover(first: np.ndarray, second: np.ndarray, *, rng: Any, **kwargs: Any) -> Any:
    """Recombine within each layer; never redistribute its expert slots."""
    del kwargs
    if not np.array_equal(first.sum(axis=1), second.sum(axis=1)):
        raise ValueError("identity-only parents must have identical layer capacities")
    a, b = first & second, first & second
    for layer in range(len(first)):
        different = rng.permutation(np.flatnonzero(first[layer] ^ second[layer]))
        half = len(different) // 2
        a[layer, different[:half]] = True
        b[layer, different[half:]] = True
    return GlobalBudgetCrossover(a, b, (), ())


def run_search(objective: Any, rankings: np.ndarray, *, search_mode: str, **kwargs: Any) -> Any:
    from dsai6707 import pruning_global_budget_search as engine

    kwargs["mode"] = "allocation_only" if search_mode == "allocation_only" else "joint"
    original_crossover = engine.crossover_global_budget
    if search_mode == "identity_only":
        engine.crossover_global_budget = identity_crossover
    try:
        result = run_global_budget_nsga2(objective, rankings, **kwargs)
    finally:
        engine.crossover_global_budget = original_crossover
    if search_mode == "identity_only":
        reference = kwargs["config"].total_retained // len(rankings)
        if not np.all(result.masks.sum(axis=2) == reference):
            raise RuntimeError("identity-only search unexpectedly redistributed layer capacity")
    return result


def shortlist(metrics: list[Any], **kwargs: Any) -> tuple[int, ...]:
    """Equal selection budget: all 16 starts plus eight generated candidates."""
    del kwargs
    order = sorted(
        range(16, len(metrics)), key=lambda i: proxy_selection_key(metrics[i], key=f"{i:04d}")
    )
    return tuple(range(16)) + tuple(order[:8])


def _main_anchors(family: str, retained: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if family == "ernie" and retained == 48:
        from scripts.run_report_review_evaluation import _family_methods

        rows = {row["key"]: row["mask"] for row in _family_methods("ernie")}
        anchors = {name: rows[f"{name}_75"] for name in ("reap", "frequency", "hybrid")}
        _, rankings = base._anchors("ernie", 32)
        incumbent = rows["tape_75"]
    else:
        all_anchors, rankings = base._anchors(family, retained)
        anchors = {name: all_anchors[name] for name in ("reap", "frequency", "hybrid")}
        if family == "ernie":
            source = ROOT / "results/runs/ernie21b_ean_train_only_50_20261022"
        else:
            source = ROOT / f"results/runs/tape_multistart_followup_20261021/granite_{retained}"
        frozen = read_json(source / "frozen_selection.json")
        expected = frozen.get("winner", frozen)["mask_sha256"]
        incumbent = np.load(source / "winner_mask.npy", allow_pickle=False)
        if mask_digest(incumbent) != expected:
            raise RuntimeError("historical incumbent differs from its frozen selection")
    return {**anchors, "historical_tape": incumbent}, rankings


def _ablation_anchors(mode: str, config: Any, rankings: np.ndarray) -> dict[str, np.ndarray]:
    anchors, _ = _main_anchors("ernie", 48)
    if mode == "identity_only":
        from scripts.run_report_review_evaluation import _family_methods

        previous = next(r["mask"] for r in _family_methods("ernie") if r["key"] == "identity_75")
        return {
            **{n: m for n, m in anchors.items() if n != "historical_tape"},
            "historical_identity": previous,
        }
    uniform = anchors["reap"]
    # Use explicit feasible capacity patterns, not identity-only perturbations.
    front_counts = np.full(27, 48, dtype=np.int64)
    front_counts[:9] += 1
    front_counts[-9:] -= 1
    front = build_mask_from_layer_rankings(rankings, front_counts, **bounds_for(config))
    back = build_mask_from_layer_rankings(rankings, 96 - front_counts, **bounds_for(config))
    return {
        "reap": uniform,
        "historical_projection": project(anchors["historical_tape"], rankings, config),
        "front_allocation": front,
        "back_allocation": back,
    }


def _code_paths() -> list[Path]:
    return [
        Path(__file__),
        ROOT / "scripts/run_tape_multistart_followup.py",
        ROOT / "src/dsai6707/pruning_global_budget_search.py",
        ROOT / "src/dsai6707/pruning_global_budget.py",
        ROOT / "src/dsai6707/pruning_multistart.py",
        ROOT / "src/dsai6707/task_choice_proxy.py",
    ]


def prepare(output: Path) -> None:
    if (output / "protocol.json").exists():
        protocol(output)
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("refusing to prepare in a nonempty experiment directory")
    cases = {}
    specs = [(f"{f}_{k}", f, k, "joint") for f, k in MAIN_CASES]
    specs += [(f"ernie_48_{m}", "ernie", 48, m) for m in ("identity_only", "allocation_only")]
    for key, family, retained, mode in specs:
        anchors, rankings = _main_anchors(family, retained)
        config = config_for(len(rankings), retained, mode)
        if mode != "joint":
            anchors = _ablation_anchors(mode, config, rankings)
        initial = initial_population(anchors, config, mode, rankings=rankings)
        folder = output / key
        folder.mkdir(parents=True, exist_ok=True)
        files = {
            "anchors.npy": np.stack(list(anchors.values())),
            "rankings.npy": rankings,
            "initial_population.npy": initial.masks,
        }
        for name, array in files.items():
            np.save(folder / name, array, allow_pickle=False)
        write_json(
            folder / "initial_population.json",
            {"contexts": initial.contexts, "population_sha256": initial.population_sha256},
        )
        cases[key] = {
            "family": family,
            "mode": mode,
            "retained_per_layer_reference": retained,
            "shape": list(rankings.shape),
            "anchor_names": list(anchors),
            "anchor_digests": [mask_digest(m) for m in anchors.values()],
            "variants": [{**VARIANT, "search": asdict(config)}],
            "files": {name: sha256_file(folder / name) for name in files},
        }
    source_data = read_json(base.DEFAULT_OUTPUT / "protocol.json")["data"]
    write_json(
        output / "protocol.json",
        {
            "schema_version": "uniform-tape-v1",
            "created_at": utc_now(),
            "posthoc_exploratory": True,
            "test_previously_seen": True,
            "selection_uses_test": False,
            "training_performed": False,
            "data": source_data,
            "common": {
                "search_samples_per_task": 64,
                "selection_samples_per_task": 256,
                "population_size": 16,
                "generations": 15,
                "candidate_budget": 256,
                "selection_pool_size": 24,
                "normalization": "characters",
                "temperature": 0.25,
                "ce_weight": 0.75,
                "kl_weight": 0.25,
                "selection_rule": (
                    "extra train macro acc_norm; ties min task acc_norm, normalized CE, digest"
                ),
                "initialization": (
                    "three direct scoring baselines, corresponding historical TAPE, "
                    "and local neighbors"
                ),
                "shortlist": (
                    "all sixteen initial candidates and eight generated candidates "
                    "by search acc_norm"
                ),
            },
            "historical_results_preserved": True,
            "diagnostics": {
                "ernie75_same_capacity_reap": (
                    "freeze the joint winner layer counts and select REAP prefixes; "
                    "evaluate only after recording this fixed projection"
                ),
                "ernie75_initial_reference": (
                    "select among all sixteen initial masks by the same extra-train "
                    "acc_norm rule; never select the initial reference using test"
                ),
                "expert_distribution": (
                    "update comparisons from the new ERNIE75 frozen masks; "
                    "do not reuse old mask counts or old expert associations"
                ),
            },
            "cases": cases,
            "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in _code_paths()},
        },
    )
    print(f"PREPARED {output}: four common-protocol runs plus two ERNIE75 ablations", flush=True)


def protocol(output: Path) -> dict[str, Any]:
    result = read_json(output / "protocol.json")
    for name, digest in result["code_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise RuntimeError(f"frozen implementation changed: {name}")
    for key, case in result["cases"].items():
        for name, digest in case["files"].items():
            if sha256_file(output / key / name) != digest:
                raise RuntimeError(f"frozen artifact changed: {key}/{name}")
    return result


def run(output: Path, key: str) -> None:
    frozen = protocol(output)
    case = frozen["cases"][key]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("this experiment is restricted to physical GPU zero")
    state_path = output / key / "state.json"
    if state_path.exists() and read_json(state_path)["status"] == "complete":
        print(f"COMPLETE {key}: preserving finished results", flush=True)
        return
    mode = case["mode"]
    rankings = np.load(output / key / "rankings.npy", allow_pickle=False)
    base.initial_population = lambda anchors, config: initial_population(
        anchors, config, mode, rankings=rankings
    )
    base.shortlist = shortlist

    def search(objective: Any, rankings: np.ndarray, **kwargs: Any) -> Any:
        return run_search(objective, rankings, search_mode=mode, **kwargs)

    base.run_global_budget_nsga2 = search
    base.run(
        output,
        key,
        test=True,
        variants=(VARIANT,),
        config_factory=lambda layers, retained, wide=False: config_for(layers, retained, mode),
        shortlist_extra=8,
    )
    print(f"UNIFORM_DONE {key}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "check"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--case")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "prepare":
        prepare(output)
    elif args.action == "check":
        protocol(output)
        print("FROZEN_PROTOCOL_OK", flush=True)
    elif args.case is None:
        parser.error("run requires --case")
    else:
        run(output, args.case)


if __name__ == "__main__":
    main()

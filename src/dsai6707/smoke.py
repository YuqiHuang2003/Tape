"""Network-free end-to-end dry run for the research pipeline."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .coefficients import CoefficientLayout, frequency_coefficients
from .config import lock_run_config
from .evolution import EAConfig, make_initial_population, run_evolution
from .grouping import GroupingPlan, LayerGrouping
from .io import environment_snapshot, write_json

_DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"name": "synthetic_smoke", "run_dir": "results/smoke"},
    "model": {
        "id": "synthetic/toy-moe",
        "revision": "local-v1",
        "tokenizer_revision": "local-v1",
        "transformers_version": "4.46.3",
    },
    "data": {
        "dataset": "synthetic",
        "config": "fixed-v1",
        "revision": "local-v1",
        "sequence_length": 8,
        "statistics_segments": 4,
        "fitness_tokens": 16,
        "validation_tokens": 16,
    },
    "grouping": {
        "target_groups_per_layer": 2,
        "source_reference": "local deterministic fixture",
    },
    "fusion": {
        "baseline": "routing_frequency",
        "smoothing": 1.0e-6,
        "accumulation_dtype": "float32",
    },
    "search": {
        "warmup_tokens": 16,
        "population_size": 4,
        "offspring_size": 4,
        "evaluation_budget": 12,
        "tournament_size": 2,
        "init_sigma": 0.04,
        "mutation_sigma": 0.2,
        "seeds": [6707, 6708, 6709],
        "strategies": ["coordinate", "layer_block"],
    },
    "evaluation": {
        "tasks": ["arc_easy", "openbookqa"],
        "task_datasets": {
            "arc_easy": {
                "hub_repository": "allenai/ai2_arc",
                "path": "allenai/ai2_arc",
                "name": "ARC-Easy",
                "split": "test",
                "revision": "210d026faf9955653af8916fad021475a3f00453",
            },
            "openbookqa": {
                "hub_repository": "allenai/openbookqa",
                "path": "openbookqa",
                "name": "main",
                "split": "test",
                "revision": "388097ea7776314e93a529163e0fea805b8a6454",
            },
        },
        "num_fewshot": 0,
        "primary_metric": "acc",
    },
}


def run_smoke(
    output_dir: str | Path, *, config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Run both EA variants on a deterministic two-layer synthetic objective."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    resolved = deepcopy(_DEFAULT_CONFIG if config is None else dict(config))
    if resolved["data"]["dataset"] != "synthetic":
        raise ValueError("the smoke runner accepts only a synthetic configuration")
    if int(resolved["grouping"]["target_groups_per_layer"]) != 2:
        raise ValueError("the fixed smoke fixture has exactly two groups per layer")
    locked_digest = lock_run_config(resolved, destination)
    plan = GroupingPlan(
        layers=(
            LayerGrouping.from_assignment(0, [0, 0, 1, 1]),
            LayerGrouping.from_assignment(1, [0, 0, 1, 1]),
        ),
        model_id="synthetic/toy-moe",
        model_revision="local-v1",
        method="fixed_smoke_grouping",
        calibration_fingerprint="synthetic-v1",
        seed=6707,
    )
    write_json(destination / "grouping.json", plan.to_dict())
    counts = np.asarray([[8, 2, 3, 7], [6, 4, 9, 1]], dtype=np.float64)
    layout = CoefficientLayout(plan)
    baseline = frequency_coefficients(plan, counts)
    center = layout.encode(baseline)
    target = np.asarray([0.25, -0.35, 0.45, -0.15], dtype=np.float64)
    coupling = np.asarray(
        [
            [1.3, 0.2, 0.05, 0.0],
            [0.2, 0.9, 0.0, 0.1],
            [0.05, 0.0, 1.1, -0.15],
            [0.0, 0.1, -0.15, 1.4],
        ],
        dtype=np.float64,
    )

    def objective(vector: np.ndarray) -> float:
        delta = vector - target
        return float(2.0 + delta @ coupling @ delta)

    search = resolved["search"]
    ea_config = EAConfig(
        population_size=int(search["population_size"]),
        offspring_size=int(search.get("offspring_size", search["population_size"])),
        evaluation_budget=int(search["evaluation_budget"]),
        tournament_size=int(search.get("tournament_size", 2)),
        init_sigma=float(search.get("init_sigma", 0.04)),
        mutation_sigma=float(search.get("mutation_sigma", 0.2)),
    )
    summaries: list[dict[str, Any]] = []
    strategies = tuple(search.get("strategies", ("coordinate", "layer_block")))
    if strategies != ("coordinate", "layer_block"):
        raise ValueError("the smoke comparison requires coordinate and layer_block in order")
    for seed_value in search["seeds"]:
        seed = int(seed_value)
        population = make_initial_population(
            center,
            population_size=ea_config.population_size,
            sigma=ea_config.init_sigma,
            seed=seed,
        )
        for strategy in strategies:
            result = run_evolution(
                objective,
                layout,
                population,
                strategy=strategy,
                config=ea_config,
                seed=seed,
                output_dir=destination / "search" / strategy / f"seed_{seed}",
            )
            summaries.append(
                {
                    "method": strategy,
                    "seed": seed,
                    "best_nll": result.best_fitness,
                    "evaluations": result.evaluations,
                    "initial_population_sha256": result.initial_population_hash,
                }
            )

    frequency_nll = objective(center)
    original_nll = objective(np.zeros(layout.dimension))
    synthetic_metrics = {
        "schema_version": "smoke-results-v1",
        "is_synthetic": True,
        "warning": "These values validate plumbing only and are not research results.",
        "resolved_config_sha256": locked_digest,
        "original": {"calibration_nll": original_nll},
        "frequency": {"calibration_nll": frequency_nll},
        "search": summaries,
        "environment": environment_snapshot(),
    }
    write_json(destination / "smoke_summary.json", synthetic_metrics)
    with (destination / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "seed",
                "best_nll",
                "evaluations",
                "initial_population_sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(summaries)
    return synthetic_metrics


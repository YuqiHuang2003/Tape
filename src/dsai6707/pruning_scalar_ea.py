"""Deterministic scalar EA for fixed-cardinality expert-pruning masks.

This module contains no model or task implementation.  It consumes the same
``[population, layers, experts]`` boolean initial population used by the
task-aware NSGA-II experiment and objective values exposing the four fields
``objectives``, ``metrics``, ``lane``, and ``cost``.  The two task objectives
are retained verbatim while their arithmetic mean is minimized by an elitist
``(mu + lambda)`` evolutionary algorithm.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Protocol

import numpy as np

from dsai6707.pruning_nsga2 import (
    balanced_set_crossover,
    mask_sha256,
    mutate_distinct_layer_swaps,
    validate_fixed_cardinality_mask,
)

# Match the task-aware NSGA-II runner's search RNG stream. Parent selection
# may choose different winners, but tournament, crossover, and mutation draws
# start from the same deterministic stream. Algorithm-dependent duplicate
# retries may cause the streams to diverge later.
PAIRED_SEARCH_NAMESPACE_SEED = 0x4E534741  # ASCII-ish "NSGA"


class TaskFitnessLike(Protocol):
    """Structural type returned by a batched task objective."""

    objectives: Sequence[float]
    metrics: Mapping[str, Any]
    lane: int
    cost: Mapping[str, Any]


BatchObjective = Callable[[Sequence[np.ndarray]], Sequence[TaskFitnessLike]]


@dataclass(frozen=True)
class ScalarEASearchResult:
    """Complete journal-ready result of a scalar fixed-cardinality search."""

    masks: np.ndarray
    objectives: np.ndarray
    scalar_fitness: np.ndarray
    trials: tuple[dict[str, Any], ...]
    final_population_ids: tuple[int, ...]
    best_candidate_id: int
    proposals: int
    duplicate_proposals: int
    initial_population_sha256: str

    @property
    def unique_evaluations(self) -> int:
        return int(self.masks.shape[0])


def _validate_candidate_ids(
    candidate_ids: Sequence[int] | np.ndarray,
    *,
    size: int,
) -> np.ndarray:
    raw = np.asarray(candidate_ids)
    if raw.ndim != 1 or raw.shape[0] != size or raw.dtype.kind not in "iu":
        raise ValueError("candidate_ids must be a one-dimensional integer sequence")
    values = raw.astype(np.int64, copy=False)
    if np.unique(values).size != size:
        raise ValueError("candidate_ids must be unique")
    return values


def _validate_scalar_fitness(
    scalar_fitness: Sequence[float] | np.ndarray,
) -> np.ndarray:
    values = np.asarray(scalar_fitness, dtype=np.float64)
    if values.ndim != 1 or values.size < 1:
        raise ValueError("scalar_fitness must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all():
        raise ValueError("scalar_fitness must contain only finite values")
    return values


def scalar_tournament(
    scalar_fitness: Sequence[float] | np.ndarray,
    candidate_ids: Sequence[int] | np.ndarray,
    *,
    rng: np.random.Generator,
    tournament_size: int = 2,
    pool: Sequence[int] | np.ndarray | None = None,
) -> int:
    """Return a positional index using fitness then candidate-ID tie-breaking."""

    fitness = _validate_scalar_fitness(scalar_fitness)
    ids = _validate_candidate_ids(candidate_ids, size=fitness.size)
    eligible = (
        np.arange(fitness.size, dtype=np.int64)
        if pool is None
        else np.asarray(pool, dtype=np.int64)
    )
    if eligible.ndim != 1 or eligible.size < 1:
        raise ValueError("pool must contain at least one candidate position")
    if (
        np.unique(eligible).size != eligible.size
        or np.any(eligible < 0)
        or np.any(eligible >= fitness.size)
    ):
        raise ValueError("pool must contain unique valid candidate positions")
    if not isinstance(tournament_size, int) or not 1 <= tournament_size <= eligible.size:
        raise ValueError("tournament_size must be in [1, pool size]")

    contestants = rng.choice(eligible, size=tournament_size, replace=False)
    return int(
        min(
            (int(position) for position in contestants),
            key=lambda position: (float(fitness[position]), int(ids[position])),
        )
    )


def select_scalar_survivors(
    scalar_fitness: Sequence[float] | np.ndarray,
    candidate_ids: Sequence[int] | np.ndarray,
    population_size: int,
) -> np.ndarray:
    """Return positional indices for deterministic elitist ``(mu + lambda)`` selection."""

    fitness = _validate_scalar_fitness(scalar_fitness)
    ids = _validate_candidate_ids(candidate_ids, size=fitness.size)
    if not isinstance(population_size, int) or not 1 <= population_size <= fitness.size:
        raise ValueError("population_size must be in [1, candidate count]")
    order = np.lexsort((ids, fitness))
    return order[:population_size].astype(np.int64, copy=False)


def _population_sha256(population: np.ndarray, *, retained_count: int) -> str:
    digests = [mask_sha256(mask, retained_count=retained_count) for mask in population]
    return sha256("\n".join(digests).encode("ascii")).hexdigest()


def _mutation_dict(mutation: Any) -> list[dict[str, int]]:
    return [
        {
            "layer": int(swap.layer),
            "removed_expert": int(swap.removed_expert),
            "added_expert": int(swap.added_expert),
        }
        for swap in mutation.swaps
    ]


def _normalize_objective_result(
    value: TaskFitnessLike,
) -> tuple[tuple[float, float], float, dict[str, Any], int, dict[str, Any]]:
    try:
        objective_values = np.asarray(value.objectives, dtype=np.float64)
        metrics = dict(value.metrics)
        lane = int(value.lane)
        cost = dict(value.cost)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise TypeError("objective values must expose objectives, metrics, lane, and cost") from exc
    if objective_values.shape != (2,) or not np.isfinite(objective_values).all():
        raise ValueError("each objective result must contain two finite task objectives")
    objectives = (float(objective_values[0]), float(objective_values[1]))
    scalar = float(np.mean(objective_values))
    return objectives, scalar, metrics, lane, cost


def run_task_scalar_ea(
    objective_many: BatchObjective,
    initial_population: np.ndarray,
    initial_contexts: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    population_size: int,
    offspring_size: int,
    generations: int,
    tournament_size: int,
    mutation_counts: Sequence[int],
    mutation_probabilities: Sequence[float],
    max_duplicate_attempts: int,
    evaluation_batch_size: int = 2,
    evaluation_budget: int | None = None,
) -> ScalarEASearchResult:
    """Minimize the arithmetic mean of two task objectives.

    Balanced-set crossover and one-swap mutations in one, two, or three
    distinct layers preserve the exact retained-expert count in every layer.
    Only unique masks consume the evaluation budget.
    """

    population = np.asarray(initial_population, dtype=np.bool_)
    if population.ndim != 3 or population.shape[0] != population_size:
        raise ValueError("initial_population has the wrong shape")
    if len(initial_contexts) != population_size:
        raise ValueError("one initial context is required per mask")
    if population_size < 3:
        raise ValueError("population_size must be at least three")
    if offspring_size < 1 or generations < 0:
        raise ValueError("offspring_size must be positive and generations non-negative")
    if not isinstance(tournament_size, int) or not 1 <= tournament_size < population_size:
        raise ValueError("tournament_size must be in [1, population_size)")
    if not isinstance(evaluation_batch_size, int) or evaluation_batch_size < 1:
        raise ValueError("evaluation_batch_size must be positive")
    if not isinstance(max_duplicate_attempts, int) or max_duplicate_attempts < 1:
        raise ValueError("max_duplicate_attempts must be positive")

    counts = tuple(int(value) for value in mutation_counts)
    probabilities = np.asarray(tuple(mutation_probabilities), dtype=np.float64)
    if (
        not counts
        or len(counts) != probabilities.size
        or len(set(counts)) != len(counts)
        or any(count not in (1, 2, 3) for count in counts)
    ):
        raise ValueError("mutation counts must be unique values drawn from 1, 2, and 3")
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or not np.isclose(float(probabilities.sum()), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("mutation probabilities must be finite, non-negative, and sum to one")

    expected_evaluations = population_size + generations * offspring_size
    if evaluation_budget is not None and evaluation_budget != expected_evaluations:
        raise ValueError("evaluation_budget must equal mu + generations * lambda")

    retained_count = validate_fixed_cardinality_mask(population[0])
    for mask in population:
        validate_fixed_cardinality_mask(mask, retained_count=retained_count)
    initial_keys = [mask_sha256(mask, retained_count=retained_count) for mask in population]
    if len(set(initial_keys)) != len(initial_keys):
        raise ValueError("initial population masks must be unique")

    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), PAIRED_SEARCH_NAMESPACE_SEED, 1])
    )
    all_masks: list[np.ndarray] = []
    all_objectives: list[tuple[float, float]] = []
    all_scalar_fitness: list[float] = []
    trials: list[dict[str, Any]] = []
    known = set(initial_keys)
    proposals = 0
    duplicate_proposals = 0

    def evaluate_generation(
        masks: Sequence[np.ndarray],
        contexts: Sequence[Mapping[str, Any]],
        *,
        phase: str,
        generation: int,
    ) -> list[int]:
        if len(masks) != len(contexts):
            raise ValueError("one context is required per evaluated mask")
        candidate_ids: list[int] = []
        for start in range(0, len(masks), evaluation_batch_size):
            batch_masks = tuple(
                np.ascontiguousarray(mask.copy())
                for mask in masks[start : start + evaluation_batch_size]
            )
            raw_results = objective_many(batch_masks)
            if isinstance(raw_results, str | bytes) or not isinstance(raw_results, Sequence):
                raise TypeError("objective_many must return a sequence")
            if len(raw_results) != len(batch_masks):
                raise RuntimeError("objective returned the wrong number of results")
            for mask, context, raw_value in zip(
                batch_masks,
                contexts[start : start + evaluation_batch_size],
                raw_results,
                strict=True,
            ):
                objectives, scalar, metrics, lane, cost = _normalize_objective_result(raw_value)
                candidate_id = len(all_masks)
                all_masks.append(mask)
                all_objectives.append(objectives)
                all_scalar_fitness.append(scalar)
                trials.append(
                    {
                        **dict(context),
                        "candidate_id": candidate_id,
                        "candidate_sha256": mask_sha256(mask, retained_count=retained_count),
                        "phase": phase,
                        "generation": generation,
                        "objectives": list(objectives),
                        "scalar_fitness": scalar,
                        "metrics": metrics,
                        "lane": lane,
                        "cost": cost,
                    }
                )
                candidate_ids.append(candidate_id)
        return candidate_ids

    population_ids = np.asarray(
        evaluate_generation(
            list(population),
            initial_contexts,
            phase="initialization",
            generation=0,
        ),
        dtype=np.int64,
    )
    population_scalar = np.asarray(
        [all_scalar_fitness[int(candidate_id)] for candidate_id in population_ids],
        dtype=np.float64,
    )

    for generation in range(1, generations + 1):
        child_masks: list[np.ndarray] = []
        child_contexts: list[dict[str, Any]] = []
        duplicate_streak = 0
        mating_event = 0
        while len(child_masks) < offspring_size:
            if duplicate_streak >= max_duplicate_attempts:
                raise RuntimeError("scalar EA exceeded the duplicate-proposal limit")
            first_position = scalar_tournament(
                population_scalar,
                population_ids,
                rng=rng,
                tournament_size=tournament_size,
            )
            second_pool = np.asarray(
                [position for position in range(population_size) if position != first_position],
                dtype=np.int64,
            )
            second_position = scalar_tournament(
                population_scalar,
                population_ids,
                rng=rng,
                tournament_size=tournament_size,
                pool=second_pool,
            )
            first_parent_id = int(population_ids[first_position])
            second_parent_id = int(population_ids[second_position])
            first_child, second_child = balanced_set_crossover(
                all_masks[first_parent_id],
                all_masks[second_parent_id],
                rng=rng,
            )
            mating_event += 1
            for sibling, crossed in enumerate((first_child, second_child)):
                proposals += 1
                mutation_count = int(rng.choice(counts, p=probabilities))
                mutation = mutate_distinct_layer_swaps(
                    crossed,
                    rng=rng,
                    min_layers=mutation_count,
                    max_layers=mutation_count,
                )
                digest = mask_sha256(mutation.mask, retained_count=retained_count)
                if digest in known:
                    duplicate_proposals += 1
                    duplicate_streak += 1
                    continue
                duplicate_streak = 0
                known.add(digest)
                child_masks.append(mutation.mask)
                child_contexts.append(
                    {
                        "kind": "offspring",
                        "parent_ids": [first_parent_id, second_parent_id],
                        "mating_event": mating_event,
                        "sibling": sibling,
                        "crossover": "balanced_pruned_set_complementary",
                        "mutation": _mutation_dict(mutation),
                    }
                )
                if len(child_masks) == offspring_size:
                    break

        child_ids = np.asarray(
            evaluate_generation(
                child_masks,
                child_contexts,
                phase="offspring",
                generation=generation,
            ),
            dtype=np.int64,
        )
        child_scalar = np.asarray(
            [all_scalar_fitness[int(candidate_id)] for candidate_id in child_ids],
            dtype=np.float64,
        )
        combined_ids = np.concatenate((population_ids, child_ids))
        combined_scalar = np.concatenate((population_scalar, child_scalar))
        survivors = select_scalar_survivors(
            combined_scalar,
            combined_ids,
            population_size,
        )
        population_ids = combined_ids[survivors]
        population_scalar = combined_scalar[survivors]

    if len(all_masks) != expected_evaluations or len(known) != expected_evaluations:
        raise RuntimeError("scalar EA unique-evaluation accounting failed")
    scalar_array = np.asarray(all_scalar_fitness, dtype=np.float64)
    best_candidate_id = int(
        min(
            range(scalar_array.size),
            key=lambda candidate_id: (
                float(scalar_array[candidate_id]),
                candidate_id,
            ),
        )
    )
    return ScalarEASearchResult(
        masks=np.stack(all_masks),
        objectives=np.asarray(all_objectives, dtype=np.float64),
        scalar_fitness=scalar_array,
        trials=tuple(trials),
        final_population_ids=tuple(int(value) for value in population_ids),
        best_candidate_id=best_candidate_id,
        proposals=proposals,
        duplicate_proposals=duplicate_proposals,
        initial_population_sha256=_population_sha256(population, retained_count=retained_count),
    )


def scalar_shortlist_candidate_ids(
    result: ScalarEASearchResult,
    *,
    reap_candidate_id: int,
    limit: int,
) -> tuple[int, ...]:
    """Return the lowest-fitness candidates, excluding the direct REAP mask."""

    fitness = _validate_scalar_fitness(result.scalar_fitness)
    if result.objectives.shape != (fitness.size, 2) or result.masks.shape[0] != fitness.size:
        raise ValueError("search result arrays have inconsistent candidate counts")
    if not isinstance(reap_candidate_id, int) or not 0 <= reap_candidate_id < fitness.size:
        raise ValueError("reap_candidate_id is outside the candidate range")
    if not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    candidates = [
        candidate_id for candidate_id in range(fitness.size) if candidate_id != reap_candidate_id
    ]
    candidates.sort(
        key=lambda candidate_id: (
            float(fitness[candidate_id]),
            candidate_id,
        )
    )
    return tuple(candidates[:limit])

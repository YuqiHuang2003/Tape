"""Budget-exact evolutionary search with matched coordinate and layer mutations."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .coefficients import CoefficientLayout
from .io import write_json, write_jsonl

MutationStrategy = Literal["coordinate", "layer_block"]


@dataclass(frozen=True)
class EAConfig:
    population_size: int = 16
    offspring_size: int = 16
    evaluation_budget: int = 300
    tournament_size: int = 2
    init_sigma: float = 0.02
    mutation_sigma: float = 0.15

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.offspring_size < 1:
            raise ValueError("offspring_size must be positive")
        if self.evaluation_budget < self.population_size:
            raise ValueError("evaluation_budget must include the full initial population")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("invalid tournament_size")
        if self.init_sigma < 0 or self.mutation_sigma <= 0:
            raise ValueError("sigma values must be non-negative, with positive mutation_sigma")


@dataclass(frozen=True)
class FitnessValue:
    value: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SearchResult:
    strategy: MutationStrategy
    seed: int
    best_vector: np.ndarray
    best_fitness: float
    best_evaluation: int
    evaluations: int
    initial_population_hash: str
    trials: tuple[dict[str, Any], ...]
    recovery: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": "search-summary-v1",
            "strategy": self.strategy,
            "seed": self.seed,
            "evaluations": self.evaluations,
            "best_evaluation": self.best_evaluation,
            "best_fitness_nll": self.best_fitness,
            "best_vector": self.best_vector.tolist(),
            "initial_population_sha256": self.initial_population_hash,
        }


def population_hash(population: np.ndarray) -> str:
    values = np.ascontiguousarray(population, dtype="<f8")
    return sha256(values.tobytes()).hexdigest()


def run_objective_warmup(
    objective: Callable[[np.ndarray], float | FitnessValue],
    vector: np.ndarray,
    *,
    expected_loss_tokens: int,
) -> dict[str, Any]:
    """Run one budget-exempt warmup and normalize its cost record.

    This helper deliberately lives outside :func:`run_evolution`: calling it
    cannot increment an EA evaluation counter or influence either EA random
    stream.  The real Granite objective supplies CUDA timing and memory data;
    CPU-only fixtures may use ``None`` for those two accelerator fields.
    """

    if expected_loss_tokens <= 0:
        raise ValueError("expected_loss_tokens must be positive")
    result = _evaluate(objective, np.asarray(vector, dtype=np.float64))
    metadata = dict(result.metadata)
    if metadata.get("status") != "ok":
        raise RuntimeError("warmup objective did not complete successfully")
    if int(metadata.get("loss_tokens", -1)) != expected_loss_tokens:
        raise ValueError("warmup objective consumed the wrong number of loss tokens")
    wall_seconds = float(metadata.get("wall_seconds", float("nan")))
    if not np.isfinite(wall_seconds) or wall_seconds < 0.0:
        raise ValueError("warmup objective did not report a valid wall time")
    gpu_value = metadata.get("gpu_seconds")
    if gpu_value is not None and (
        not np.isfinite(float(gpu_value)) or float(gpu_value) < 0.0
    ):
        raise ValueError("warmup objective reported an invalid GPU time")
    peak_value = metadata.get("peak_device_memory_bytes")
    if peak_value is not None and int(peak_value) < 0:
        raise ValueError("warmup objective reported invalid peak device memory")
    return {
        "schema_version": "search-warmup-session-v1",
        "budget_evaluations_consumed": 0,
        "fitness_nll": float(result.value),
        "loss_tokens": expected_loss_tokens,
        "wall_seconds": wall_seconds,
        "gpu_seconds": None if gpu_value is None else float(gpu_value),
        "peak_device_memory_bytes": None if peak_value is None else int(peak_value),
        "details": metadata,
    }


def make_initial_population(
    center: Sequence[float] | np.ndarray,
    *,
    population_size: int,
    sigma: float,
    seed: int,
) -> np.ndarray:
    """Create a shared population with the exact baseline as candidate zero."""

    center_array = np.asarray(center, dtype=np.float64)
    if center_array.ndim != 1:
        raise ValueError("center must be a vector")
    if population_size < 1 or sigma < 0:
        raise ValueError("invalid population settings")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x1A17]))
    population = np.repeat(center_array[None, :], population_size, axis=0)
    if population_size > 1 and center_array.size > 0:
        population[1:] += rng.normal(0.0, sigma, size=population[1:].shape)
    return population


def _evaluate(
    objective: Callable[[np.ndarray], float | FitnessValue], vector: np.ndarray
) -> FitnessValue:
    result = objective(vector.copy())
    if isinstance(result, FitnessValue):
        fitness = result
    else:
        fitness = FitnessValue(float(result), {})
    if not np.isfinite(fitness.value):
        raise ValueError("objective returned a non-finite fitness")
    return fitness


_RESUME_ARRAYS = frozenset(
    {
        "metadata",
        "population",
        "fitnesses",
        "candidate_ids",
        "pending_children",
        "pending_fitnesses",
        "pending_ids",
        "best_vector",
    }
)


def _write_resume_state(
    path: Path,
    *,
    strategy: MutationStrategy,
    seed: int,
    config: EAConfig,
    layout: CoefficientLayout,
    initial_hash: str,
    population: np.ndarray,
    fitnesses: np.ndarray,
    candidate_ids: np.ndarray,
    initialized_count: int,
    generation: int,
    evaluation: int,
    pending_target: int,
    pending_children: list[np.ndarray],
    pending_fitnesses: list[float],
    pending_ids: list[int],
    best_fitness: float,
    best_vector: np.ndarray,
    best_evaluation: int,
    trials: list[dict[str, Any]],
    selection_rng: np.random.Generator,
    mutation_rng: np.random.Generator,
    checkpoint_writes: int,
    resume_count: int,
) -> None:
    """Atomically persist every value needed for an exact next candidate."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pending_array = (
        np.stack(pending_children).astype(np.float64, copy=False)
        if pending_children
        else np.empty((0, layout.dimension), dtype=np.float64)
    )
    metadata = {
        "schema_version": "ea-resume-state-v1",
        "strategy": strategy,
        "seed": seed,
        "config": asdict(config),
        "layout_dimension": layout.dimension,
        "layer_ranges": [list(bounds) for bounds in layout.layer_ranges],
        "initial_population_sha256": initial_hash,
        "initialized_count": initialized_count,
        "generation": generation,
        "evaluation": evaluation,
        "pending_target": pending_target,
        "best_fitness_nll": best_fitness,
        "best_evaluation": best_evaluation,
        "trials": trials,
        "selection_rng_state": selection_rng.bit_generator.state,
        "mutation_rng_state": mutation_rng.bit_generator.state,
        "checkpoint_writes": checkpoint_writes,
        "resume_count": resume_count,
    }
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        np.savez(
            handle,
            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
            population=np.ascontiguousarray(population, dtype=np.float64),
            fitnesses=np.ascontiguousarray(fitnesses, dtype=np.float64),
            candidate_ids=np.ascontiguousarray(candidate_ids, dtype=np.int64),
            pending_children=np.ascontiguousarray(pending_array, dtype=np.float64),
            pending_fitnesses=np.asarray(pending_fitnesses, dtype=np.float64),
            pending_ids=np.asarray(pending_ids, dtype=np.int64),
            best_vector=np.ascontiguousarray(best_vector, dtype=np.float64),
        )
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _load_resume_state(
    path: Path,
    *,
    strategy: MutationStrategy,
    seed: int,
    config: EAConfig,
    layout: CoefficientLayout,
    initial_population: np.ndarray,
    initial_hash: str,
) -> dict[str, Any]:
    """Load and strictly validate an EA resume checkpoint."""

    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != _RESUME_ARRAYS:
                raise ValueError("resume checkpoint has an unexpected array inventory")
            metadata = json.loads(str(payload["metadata"].item()))
            state: dict[str, Any] = {
                "metadata": metadata,
                "population": payload["population"].astype(np.float64, copy=True),
                "fitnesses": payload["fitnesses"].astype(np.float64, copy=True),
                "candidate_ids": payload["candidate_ids"].astype(np.int64, copy=True),
                "pending_children": payload["pending_children"].astype(
                    np.float64, copy=True
                ),
                "pending_fitnesses": payload["pending_fitnesses"].astype(
                    np.float64, copy=True
                ),
                "pending_ids": payload["pending_ids"].astype(np.int64, copy=True),
                "best_vector": payload["best_vector"].astype(np.float64, copy=True),
            }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid EA resume checkpoint: {path}") from exc

    if not isinstance(metadata, dict) or metadata.get("schema_version") != (
        "ea-resume-state-v1"
    ):
        raise ValueError("unsupported EA resume checkpoint schema")
    identity = (
        metadata.get("strategy") == strategy
        and int(metadata.get("seed", -1)) == seed
        and metadata.get("config") == asdict(config)
        and int(metadata.get("layout_dimension", -1)) == layout.dimension
        and metadata.get("layer_ranges")
        == [list(bounds) for bounds in layout.layer_ranges]
        and metadata.get("initial_population_sha256") == initial_hash
    )
    if not identity:
        raise ValueError("EA resume checkpoint does not match this locked search")

    expected_population_shape = (config.population_size, layout.dimension)
    population = state["population"]
    fitnesses = state["fitnesses"]
    candidate_ids = state["candidate_ids"]
    pending_children = state["pending_children"]
    pending_fitnesses = state["pending_fitnesses"]
    pending_ids = state["pending_ids"]
    best_vector = state["best_vector"]
    if population.shape != expected_population_shape or not np.isfinite(population).all():
        raise ValueError("resume checkpoint has an invalid population")
    if fitnesses.shape != (config.population_size,) or candidate_ids.shape != (
        config.population_size,
    ):
        raise ValueError("resume checkpoint has invalid survivor arrays")
    if pending_children.ndim != 2 or pending_children.shape[1:] != (layout.dimension,):
        raise ValueError("resume checkpoint has invalid pending children")
    if (
        pending_fitnesses.shape != (pending_children.shape[0],)
        or pending_ids.shape != (pending_children.shape[0],)
        or not np.isfinite(pending_children).all()
        or not np.isfinite(pending_fitnesses).all()
    ):
        raise ValueError("resume checkpoint has inconsistent pending arrays")
    if best_vector.shape != (layout.dimension,) or not np.isfinite(best_vector).all():
        raise ValueError("resume checkpoint has an invalid best vector")

    initialized_count = int(metadata.get("initialized_count", -1))
    generation = int(metadata.get("generation", -1))
    evaluation = int(metadata.get("evaluation", -1))
    pending_target = int(metadata.get("pending_target", -1))
    trials = metadata.get("trials")
    if not isinstance(trials, list) or len(trials) != evaluation:
        raise ValueError("resume checkpoint trial count is inconsistent")
    if not 1 <= evaluation <= config.evaluation_budget:
        raise ValueError("resume checkpoint has an invalid evaluation counter")
    if not 0 <= initialized_count <= config.population_size or generation < 0:
        raise ValueError("resume checkpoint has invalid progress counters")
    if initialized_count != min(evaluation, config.population_size):
        raise ValueError("resume checkpoint initialization progress is inconsistent")
    if not 0 <= pending_children.shape[0] <= pending_target <= config.offspring_size:
        raise ValueError("resume checkpoint pending progress is inconsistent")
    if initialized_count < config.population_size:
        if generation != 0 or pending_target != 0 or pending_children.shape[0] != 0:
            raise ValueError("resume checkpoint mixes initialization and offspring state")
        if not np.array_equal(population, initial_population):
            raise ValueError("resume checkpoint changed the initial population prematurely")
        if not np.isfinite(fitnesses[:initialized_count]).all() or not np.isnan(
            fitnesses[initialized_count:]
        ).all():
            raise ValueError("resume checkpoint has invalid initialization fitnesses")
        if not np.array_equal(
            candidate_ids[:initialized_count], np.arange(initialized_count)
        ) or not np.all(candidate_ids[initialized_count:] == -1):
            raise ValueError("resume checkpoint has invalid initialization candidate IDs")
    else:
        if not np.isfinite(fitnesses).all():
            raise ValueError("resume checkpoint contains non-finite survivor fitness")
        if len(set(candidate_ids.tolist())) != config.population_size:
            raise ValueError("resume checkpoint survivor candidate IDs are not unique")

    trial_fitnesses: list[float] = []
    for expected_id, row in enumerate(trials):
        if not isinstance(row, dict) or int(row.get("evaluation", -1)) != expected_id:
            raise ValueError("resume checkpoint trial IDs are not contiguous")
        if row.get("strategy") != strategy or int(row.get("seed", -1)) != seed:
            raise ValueError("resume checkpoint contains a trial from another search")
        vector = np.asarray(row.get("vector"), dtype=np.float64)
        fitness = float(row.get("fitness_nll", float("nan")))
        if vector.shape != (layout.dimension,) or not np.isfinite(vector).all():
            raise ValueError("resume checkpoint contains an invalid trial vector")
        if not np.isfinite(fitness):
            raise ValueError("resume checkpoint contains an invalid trial fitness")
        trial_fitnesses.append(fitness)
    best_id = min(range(evaluation), key=lambda index: (trial_fitnesses[index], index))
    if (
        int(metadata.get("best_evaluation", -1)) != best_id
        or float(metadata.get("best_fitness_nll", float("nan")))
        != trial_fitnesses[best_id]
        or not np.array_equal(
            best_vector, np.asarray(trials[best_id]["vector"], dtype=np.float64)
        )
    ):
        raise ValueError("resume checkpoint best result is inconsistent with its trials")
    for index, candidate_id in enumerate(candidate_ids[:initialized_count]):
        if not 0 <= int(candidate_id) < evaluation:
            raise ValueError("resume checkpoint references an unknown survivor")
        trial = trials[int(candidate_id)]
        if fitnesses[index] != float(trial["fitness_nll"]) or not np.array_equal(
            population[index], np.asarray(trial["vector"], dtype=np.float64)
        ):
            raise ValueError("resume checkpoint survivor state disagrees with its trial")
    if pending_children.shape[0]:
        if pending_target <= 0 or generation <= 0:
            raise ValueError("resume checkpoint has pending children outside a generation")
        expected_pending_ids = list(
            range(evaluation - pending_children.shape[0], evaluation)
        )
        if pending_ids.tolist() != expected_pending_ids:
            raise ValueError("resume checkpoint pending candidate IDs are not contiguous")
        for index, candidate_id in enumerate(expected_pending_ids):
            trial = trials[candidate_id]
            if int(trial.get("generation", -1)) != generation:
                raise ValueError("resume checkpoint pending child has the wrong generation")
            if pending_fitnesses[index] != float(trial["fitness_nll"]) or not np.array_equal(
                pending_children[index], np.asarray(trial["vector"], dtype=np.float64)
            ):
                raise ValueError("resume checkpoint pending child disagrees with its trial")
    state["trials"] = trials
    state["initialized_count"] = initialized_count
    state["generation"] = generation
    state["evaluation"] = evaluation
    state["pending_target"] = pending_target
    return state


def run_evolution(
    objective: Callable[[np.ndarray], float | FitnessValue],
    layout: CoefficientLayout,
    initial_population: np.ndarray,
    *,
    strategy: MutationStrategy,
    config: EAConfig,
    seed: int,
    output_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> SearchResult:
    """Run a deterministic (mu + lambda) EA and consume exactly the given budget.

    The two strategies share initialization, tournament selection, replacement,
    sigma and budget. The mutation mask is the only algorithmic difference:

    * ``coordinate`` independently mutates each coordinate with probability
      ``1 / active_layers``;
    * ``layer_block`` uniformly chooses one active layer and mutates all of its
      coordinates.

    The expected changed-coordinate count and expected squared perturbation are
    therefore identical. An empty coordinate mask is deliberately retained and
    evaluated; resampling it would silently alter that distribution.
    """

    config.validate()
    if strategy not in ("coordinate", "layer_block"):
        raise ValueError(f"unknown mutation strategy: {strategy}")
    population = np.asarray(initial_population, dtype=np.float64).copy()
    expected_shape = (config.population_size, layout.dimension)
    if population.shape != expected_shape:
        raise ValueError(
            f"expected initial population shape {expected_shape}, got {population.shape}"
        )

    active_layers = [
        layer for layer, (start, stop) in enumerate(layout.layer_ranges) if stop > start
    ]
    if layout.dimension > 0 and not active_layers:
        raise AssertionError("non-empty coefficient vector has no active layers")
    if layout.dimension == 0:
        raise ValueError("search is undefined when every group is a singleton")

    # Separate streams prevent one mutation implementation from changing how many
    # random numbers the shared selection process consumes.  Both states are part
    # of the atomic resume checkpoint.
    selection_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5E1EC7]))
    mutation_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x6D07]))
    initial_hash = population_hash(population)
    destination = Path(output_dir) if output_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
    state_path = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else destination / "resume_state.npz"
        if destination is not None
        else None
    )

    trials: list[dict[str, Any]]
    fitnesses: np.ndarray
    candidate_ids: np.ndarray
    pending_children: list[np.ndarray]
    pending_fitnesses: list[float]
    pending_ids: list[int]
    resumed = state_path is not None and state_path.is_file()
    resumed_at_evaluation: int | None = None
    if resumed:
        state = _load_resume_state(
            state_path,
            strategy=strategy,
            seed=seed,
            config=config,
            layout=layout,
            initial_population=population,
            initial_hash=initial_hash,
        )
        population = state["population"]
        fitnesses = state["fitnesses"]
        candidate_ids = state["candidate_ids"]
        initialized_count = state["initialized_count"]
        generation = state["generation"]
        evaluation = state["evaluation"]
        pending_target = state["pending_target"]
        pending_children = [row.copy() for row in state["pending_children"]]
        pending_fitnesses = state["pending_fitnesses"].tolist()
        pending_ids = state["pending_ids"].tolist()
        best_fitness = float(state["metadata"]["best_fitness_nll"])
        best_vector = state["best_vector"]
        best_evaluation = int(state["metadata"]["best_evaluation"])
        trials = list(state["trials"])
        checkpoint_writes = int(state["metadata"].get("checkpoint_writes", 0))
        resume_count = int(state["metadata"].get("resume_count", 0)) + 1
        try:
            selection_rng.bit_generator.state = state["metadata"]["selection_rng_state"]
            mutation_rng.bit_generator.state = state["metadata"]["mutation_rng_state"]
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("resume checkpoint contains an invalid RNG state") from exc
        resumed_at_evaluation = evaluation
    else:
        trials = []
        fitnesses = np.full(config.population_size, np.nan, dtype=np.float64)
        candidate_ids = np.full(config.population_size, -1, dtype=np.int64)
        initialized_count = 0
        generation = 0
        evaluation = 0
        pending_target = 0
        pending_children = []
        pending_fitnesses = []
        pending_ids = []
        best_fitness = float("inf")
        best_vector = population[0].copy()
        best_evaluation = -1
        checkpoint_writes = 0
        resume_count = 0
    writes_at_invocation_start = checkpoint_writes

    def checkpoint() -> None:
        nonlocal checkpoint_writes
        if state_path is None:
            return
        next_write = checkpoint_writes + 1
        _write_resume_state(
            state_path,
            strategy=strategy,
            seed=seed,
            config=config,
            layout=layout,
            initial_hash=initial_hash,
            population=population,
            fitnesses=fitnesses,
            candidate_ids=candidate_ids,
            initialized_count=initialized_count,
            generation=generation,
            evaluation=evaluation,
            pending_target=pending_target,
            pending_children=pending_children,
            pending_fitnesses=pending_fitnesses,
            pending_ids=pending_ids,
            best_fitness=best_fitness,
            best_vector=best_vector,
            best_evaluation=best_evaluation,
            trials=trials,
            selection_rng=selection_rng,
            mutation_rng=mutation_rng,
            checkpoint_writes=next_write,
            resume_count=resume_count,
        )
        checkpoint_writes = next_write

    # Persist the invocation itself before evaluating another candidate.  This
    # keeps resume_count correct even if repeated restarts fail inside the first
    # candidate after recovery.
    if resumed:
        checkpoint()

    def record(
        vector: np.ndarray,
        result: FitnessValue,
        *,
        generation: int,
        parent_id: int | None,
        changed: np.ndarray,
        block_layer: int | None,
        phase: str,
    ) -> int:
        nonlocal evaluation, best_fitness, best_vector, best_evaluation
        candidate_id = evaluation
        evaluation += 1
        if result.value < best_fitness:
            best_fitness = result.value
            best_vector = vector.copy()
            best_evaluation = candidate_id
        row: dict[str, Any] = {
            "schema_version": "search-trial-v1",
            "evaluation": candidate_id,
            "generation": generation,
            "phase": phase,
            "strategy": strategy,
            "seed": seed,
            "parent_id": parent_id,
            "changed_coordinate_count": int(changed.size),
            "changed_coordinates": changed.astype(int).tolist(),
            "block_layer": block_layer,
            "fitness_nll": float(result.value),
            "best_fitness_nll": float(best_fitness),
            "vector": vector.tolist(),
        }
        row.update(result.metadata)
        trials.append(row)
        return candidate_id

    for index in range(initialized_count, config.population_size):
        result = _evaluate(objective, population[index])
        fitnesses[index] = result.value
        candidate_ids[index] = record(
            population[index],
            result,
            generation=0,
            parent_id=None,
            changed=np.empty(0, dtype=np.int64),
            block_layer=None,
            phase="initialization",
        )
        initialized_count = index + 1
        checkpoint()

    while evaluation < config.evaluation_budget:
        if pending_target == 0:
            generation += 1
            pending_target = min(
                config.offspring_size, config.evaluation_budget - evaluation
            )
            pending_children = []
            pending_fitnesses = []
            pending_ids = []

        for _ in range(len(pending_children), pending_target):
            tournament = selection_rng.integers(
                0, config.population_size, size=config.tournament_size
            )
            # Stable tuple tie-break makes exact ties reproducible.
            parent_index = min(
                tournament.tolist(), key=lambda idx: (fitnesses[idx], int(candidate_ids[idx]))
            )
            child = population[parent_index].copy()
            block_layer: int | None = None
            if strategy == "coordinate":
                probability = 1.0 / len(active_layers)
                mask = mutation_rng.random(layout.dimension) < probability
                changed = np.flatnonzero(mask)
            else:
                block_layer = int(mutation_rng.choice(active_layers))
                changed = layout.layer_indices(block_layer)
            if changed.size:
                child[changed] += mutation_rng.normal(
                    0.0, config.mutation_sigma, size=changed.size
                )

            result = _evaluate(objective, child)
            child_id = record(
                child,
                result,
                generation=generation,
                parent_id=int(candidate_ids[parent_index]),
                changed=changed,
                block_layer=block_layer,
                phase="offspring",
            )
            pending_children.append(child)
            pending_fitnesses.append(result.value)
            pending_ids.append(child_id)
            checkpoint()

        children = np.stack(pending_children).astype(np.float64, copy=False)
        child_fitness = np.asarray(pending_fitnesses, dtype=np.float64)
        child_ids = np.asarray(pending_ids, dtype=np.int64)
        combined_population = np.concatenate((population, children), axis=0)
        combined_fitness = np.concatenate((fitnesses, child_fitness), axis=0)
        combined_ids = np.concatenate((candidate_ids, child_ids), axis=0)
        order = np.lexsort((combined_ids, combined_fitness))[: config.population_size]
        population = combined_population[order].copy()
        fitnesses = combined_fitness[order].copy()
        candidate_ids = combined_ids[order].copy()
        pending_target = 0
        pending_children = []
        pending_fitnesses = []
        pending_ids = []
        checkpoint()

    recovery = {
        "schema_version": "ea-recovery-v1",
        "strategy": strategy,
        "seed": seed,
        "evaluations": evaluation,
        "checkpoint_schema_version": "ea-resume-state-v1",
        "checkpoint_granularity": "each_candidate_and_replacement_boundary",
        "resumed_from_checkpoint": resumed,
        "resume_count": resume_count,
        "resumed_at_evaluation": resumed_at_evaluation,
        "checkpoint_writes_this_invocation": checkpoint_writes
        - writes_at_invocation_start,
        "checkpoint_writes_total": checkpoint_writes,
        "temporary_checkpoint_removed_after_completion": state_path is not None,
    }

    result = SearchResult(
        strategy=strategy,
        seed=seed,
        best_vector=best_vector,
        best_fitness=best_fitness,
        best_evaluation=best_evaluation,
        evaluations=evaluation,
        initial_population_hash=initial_hash,
        trials=tuple(trials),
        recovery=recovery,
    )
    if destination is not None:
        np.save(destination / "initial_population.npy", initial_population, allow_pickle=False)
        write_jsonl(destination / "trials.jsonl", result.trials)
        write_json(destination / "summary.json", result.summary())
        write_json(
            destination / "best_coefficients.json",
            layout.to_artifact(
                result.best_vector,
                method=strategy,
                seed=seed,
                fitness=result.best_fitness,
            ),
        )
        write_json(destination / "execution.json", recovery)
    if state_path is not None:
        state_path.unlink(missing_ok=True)
    return result


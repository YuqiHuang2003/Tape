"""Deterministic evolutionary search over fixed-cardinality expert masks.

The search budget is defined as the number of *unique* candidates passed to
the objective.  A duplicate proposal is a cache hit: it is recorded in the
result, but it neither calls the objective nor consumes the evaluation budget.
This convention makes a nominal budget such as 300 comparable across mutation
strategies even when one strategy proposes more duplicates than another.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal

import numpy as np

PruningStrategy = Literal["uniform", "reap_guided"]


@dataclass(frozen=True)
class PruningEAConfig:
    """Settings shared by the uniform and REAP-guided searches."""

    retained_count: int
    population_size: int = 16
    offspring_size: int = 16
    evaluation_budget: int = 300
    tournament_size: int = 2
    epsilon_uniform: float = 0.1
    crossover_rate: float = 0.0
    max_duplicate_attempts: int = 128

    def validate(self, *, num_layers: int, num_experts: int) -> None:
        if num_layers < 1 or num_experts < 2:
            raise ValueError("the search requires at least one layer and two experts")
        if not 1 <= self.retained_count < num_experts:
            raise ValueError("retained_count must be in [1, num_experts)")
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.offspring_size < 1:
            raise ValueError("offspring_size must be positive")
        if self.evaluation_budget < self.population_size:
            raise ValueError("evaluation_budget must include the full initial population")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("invalid tournament_size")
        if not 0.0 <= self.epsilon_uniform <= 1.0:
            raise ValueError("epsilon_uniform must be in [0, 1]")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        if self.max_duplicate_attempts < 1:
            raise ValueError("max_duplicate_attempts must be positive")

        search_space_size = math.comb(num_experts, self.retained_count) ** num_layers
        if self.population_size > search_space_size:
            raise ValueError("population_size exceeds the finite mask search space")
        if self.evaluation_budget > search_space_size:
            raise ValueError("evaluation_budget exceeds the finite mask search space")


@dataclass(frozen=True)
class PruningFitnessValue:
    value: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SwapMutation:
    mask: np.ndarray
    layer: int
    removed_expert: int
    added_expert: int
    used_uniform: bool


@dataclass(frozen=True)
class PruningSearchResult:
    strategy: PruningStrategy
    seed: int
    best_mask: np.ndarray
    best_fitness: float
    best_evaluation: int
    evaluations: int
    proposals: int
    cache_hits: int
    fallback_candidates: int
    initial_population_hash: str
    trials: tuple[dict[str, Any], ...]

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": "pruning-search-summary-v1",
            "strategy": self.strategy,
            "seed": self.seed,
            "evaluations": self.evaluations,
            "proposals": self.proposals,
            "cache_hits": self.cache_hits,
            "fallback_candidates": self.fallback_candidates,
            "best_evaluation": self.best_evaluation,
            "best_fitness_nll": self.best_fitness,
            "best_mask": self.best_mask.tolist(),
            "initial_population_sha256": self.initial_population_hash,
        }


def _mask_key(mask: np.ndarray) -> bytes:
    return np.ascontiguousarray(mask, dtype=np.bool_).tobytes()


def pruning_population_hash(population: np.ndarray) -> str:
    """Hash the exact boolean bytes and shape of an initial population."""

    values = np.ascontiguousarray(population, dtype=np.bool_)
    shape = np.asarray(values.shape, dtype="<i8")
    return sha256(shape.tobytes() + values.tobytes()).hexdigest()


def validate_pruning_mask(mask: np.ndarray, *, retained_count: int) -> None:
    """Validate the canonical ``[layers, experts]`` boolean representation."""

    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise ValueError("a pruning mask must be a NumPy boolean array")
    if mask.ndim != 2 or mask.shape[0] < 1 or mask.shape[1] < 2:
        raise ValueError("a pruning mask must have shape [layers, experts]")
    if not 1 <= retained_count < mask.shape[1]:
        raise ValueError("retained_count must be in [1, num_experts)")
    counts = np.count_nonzero(mask, axis=1)
    if not np.all(counts == retained_count):
        raise ValueError("every layer must retain exactly retained_count experts")


def _random_mask(
    rng: np.random.Generator,
    *,
    num_layers: int,
    num_experts: int,
    retained_count: int,
) -> np.ndarray:
    mask = np.zeros((num_layers, num_experts), dtype=np.bool_)
    for layer in range(num_layers):
        selected = rng.choice(num_experts, size=retained_count, replace=False)
        mask[layer, selected] = True
    return mask


def _first_unseen_mask(
    *,
    num_layers: int,
    num_experts: int,
    retained_count: int,
    known: set[bytes],
) -> np.ndarray:
    """Return the first unseen mask in deterministic lexicographic order."""

    choices_per_layer = math.comb(num_experts, retained_count)
    search_space_size = choices_per_layer**num_layers
    for state_index in range(search_space_size):
        mask = np.zeros((num_layers, num_experts), dtype=np.bool_)
        remainder = state_index
        for layer in range(num_layers - 1, -1, -1):
            combination_index = remainder % choices_per_layer
            remainder //= choices_per_layer
            mask[layer, _unrank_combination(combination_index, num_experts, retained_count)] = True
        if _mask_key(mask) not in known:
            return mask
    raise RuntimeError("the finite pruning search space is exhausted")


def _unrank_combination(rank: int, size: int, selected_count: int) -> np.ndarray:
    """Decode a zero-based lexicographic combination without materializing all of them."""

    total = math.comb(size, selected_count)
    if not 0 <= rank < total:
        raise ValueError("combination rank is outside the finite choice space")
    selected: list[int] = []
    lower_bound = 0
    remaining = selected_count
    while remaining:
        for value in range(lower_bound, size - remaining + 1):
            suffix_count = math.comb(size - value - 1, remaining - 1)
            if rank < suffix_count:
                selected.append(value)
                lower_bound = value + 1
                remaining -= 1
                break
            rank -= suffix_count
    return np.asarray(selected, dtype=np.int64)


def make_initial_pruning_population(
    *,
    num_layers: int,
    num_experts: int,
    retained_count: int,
    population_size: int,
    seed: int,
    baseline_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build a strategy-independent, unique initial population.

    When supplied, ``baseline_mask`` is byte-for-byte candidate zero.  The RNG
    stream deliberately has no strategy input, so callers can construct this
    array once and pass the same bytes to both search variants.
    """

    probe = PruningEAConfig(
        retained_count=retained_count,
        population_size=population_size,
        evaluation_budget=population_size,
    )
    probe.validate(num_layers=num_layers, num_experts=num_experts)

    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x5052554E]))
    masks: list[np.ndarray] = []
    known: set[bytes] = set()
    if baseline_mask is not None:
        validate_pruning_mask(baseline_mask, retained_count=retained_count)
        if baseline_mask.shape != (num_layers, num_experts):
            raise ValueError("baseline_mask shape does not match the requested layout")
        first = np.ascontiguousarray(baseline_mask.copy())
        masks.append(first)
        known.add(_mask_key(first))

    duplicate_attempts = 0
    while len(masks) < population_size:
        candidate = _random_mask(
            rng,
            num_layers=num_layers,
            num_experts=num_experts,
            retained_count=retained_count,
        )
        key = _mask_key(candidate)
        if key in known:
            duplicate_attempts += 1
            if duplicate_attempts >= 128:
                candidate = _first_unseen_mask(
                    num_layers=num_layers,
                    num_experts=num_experts,
                    retained_count=retained_count,
                    known=known,
                )
                key = _mask_key(candidate)
                duplicate_attempts = 0
            else:
                continue
        masks.append(candidate)
        known.add(key)

    return np.ascontiguousarray(np.stack(masks), dtype=np.bool_)


def _uniform_swap(mask: np.ndarray, rng: np.random.Generator) -> SwapMutation:
    layer = int(rng.integers(mask.shape[0]))
    retained = np.flatnonzero(mask[layer])
    pruned = np.flatnonzero(~mask[layer])
    removed = int(rng.choice(retained))
    added = int(rng.choice(pruned))
    child = mask.copy()
    child[layer, removed] = False
    child[layer, added] = True
    return SwapMutation(child, layer, removed, added, True)


def mutate_uniform_swap(mask: np.ndarray, *, rng: np.random.Generator) -> SwapMutation:
    """Uniformly choose a layer, one retained expert, and one pruned expert."""

    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError("mask must be a [layers, experts] boolean array")
    if np.any(np.count_nonzero(mask, axis=1) == 0) or np.any(np.all(mask, axis=1)):
        raise ValueError("each layer must contain retained and pruned experts")
    return _uniform_swap(mask, rng)


def _ordinal_weights(values: np.ndarray, *, prefer_high: bool) -> np.ndarray:
    """Return tie-aware rank weights without assuming positive importances."""

    _, inverse = np.unique(values, return_inverse=True)
    if prefer_high:
        weights = inverse + 1
    else:
        weights = int(inverse.max()) + 1 - inverse
    return weights.astype(np.float64)


def mutate_reap_guided_swap(
    mask: np.ndarray,
    importance: np.ndarray,
    *,
    rng: np.random.Generator,
    epsilon_uniform: float,
) -> SwapMutation:
    """Swap experts using REAP scores, with epsilon-uniform exploration.

    The guided branch preferentially removes low-importance retained experts
    and inserts high-importance pruned experts.  Tie-aware ordinal weights are
    used so the operator remains well-defined for zero or negative scores.
    ``importance`` is expected to contain externally computed REAP scores; this
    module intentionally does not estimate them.
    """

    if not 0.0 <= epsilon_uniform <= 1.0:
        raise ValueError("epsilon_uniform must be in [0, 1]")
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype != np.bool_
        or mask.ndim != 2
        or importance.shape != mask.shape
    ):
        raise ValueError("mask and importance must have matching [layers, experts] shapes")
    if not np.isfinite(importance).all():
        raise ValueError("importance must contain only finite values")
    if np.any(np.count_nonzero(mask, axis=1) == 0) or np.any(np.all(mask, axis=1)):
        raise ValueError("each layer must contain retained and pruned experts")

    if rng.random() < epsilon_uniform:
        return _uniform_swap(mask, rng)

    layer = int(rng.integers(mask.shape[0]))
    retained = np.flatnonzero(mask[layer])
    pruned = np.flatnonzero(~mask[layer])
    remove_weights = _ordinal_weights(importance[layer, retained], prefer_high=False)
    add_weights = _ordinal_weights(importance[layer, pruned], prefer_high=True)
    removed = int(rng.choice(retained, p=remove_weights / remove_weights.sum()))
    added = int(rng.choice(pruned, p=add_weights / add_weights.sum()))
    child = mask.copy()
    child[layer, removed] = False
    child[layer, added] = True
    return SwapMutation(child, layer, removed, added, False)


def crossover_whole_layers(
    first: np.ndarray,
    second: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Copy complete layer masks from either parent, preserving cardinality."""

    if (
        not isinstance(first, np.ndarray)
        or not isinstance(second, np.ndarray)
        or first.dtype != np.bool_
        or second.dtype != np.bool_
        or first.ndim != 2
        or first.shape != second.shape
    ):
        raise ValueError("parents must be same-shaped [layers, experts] boolean arrays")
    take_second = rng.random(first.shape[0]) < 0.5
    child = first.copy()
    child[take_second] = second[take_second]
    return child, tuple(int(index) for index in np.flatnonzero(take_second))


def _evaluate(
    objective: Callable[[np.ndarray], float | PruningFitnessValue], mask: np.ndarray
) -> PruningFitnessValue:
    result = objective(mask.copy())
    fitness = (
        result
        if isinstance(result, PruningFitnessValue)
        else PruningFitnessValue(float(result), {})
    )
    if not np.isfinite(fitness.value):
        raise ValueError("objective returned a non-finite fitness")
    return fitness


def _tournament_index(
    fitnesses: np.ndarray,
    candidate_ids: np.ndarray,
    *,
    tournament_size: int,
    rng: np.random.Generator,
) -> int:
    entrants = rng.choice(fitnesses.size, size=tournament_size, replace=False)
    return int(min(entrants, key=lambda index: (fitnesses[index], candidate_ids[index])))


def _select_survivors(
    population: np.ndarray,
    fitnesses: np.ndarray,
    candidate_ids: np.ndarray,
    *,
    population_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.lexsort((candidate_ids, fitnesses))[:population_size]
    return population[order].copy(), fitnesses[order].copy(), candidate_ids[order].copy()


def run_pruning_evolution(
    objective: Callable[[np.ndarray], float | PruningFitnessValue],
    initial_population: np.ndarray,
    *,
    strategy: PruningStrategy,
    config: PruningEAConfig,
    seed: int,
    importance: np.ndarray | None = None,
) -> PruningSearchResult:
    """Minimize an objective under an exact unique-candidate budget.

    Duplicate proposals are looked up in the run-local deterministic cache and
    retried.  After ``max_duplicate_attempts`` consecutive hits, a deterministic
    unseen mask is injected.  Consequently, a valid budget is always consumed
    exactly, even after the evolving population converges.
    """

    if strategy not in ("uniform", "reap_guided"):
        raise ValueError(f"unknown pruning strategy: {strategy}")
    if (
        not isinstance(initial_population, np.ndarray)
        or initial_population.dtype != np.bool_
        or initial_population.ndim != 3
    ):
        raise ValueError("initial_population must be [population, layers, experts] boolean")
    if initial_population.shape[0] != config.population_size:
        raise ValueError("initial_population size does not match config.population_size")
    num_layers, num_experts = initial_population.shape[1:]
    config.validate(num_layers=num_layers, num_experts=num_experts)
    for mask in initial_population:
        validate_pruning_mask(mask, retained_count=config.retained_count)
    keys = [_mask_key(mask) for mask in initial_population]
    if len(set(keys)) != len(keys):
        raise ValueError("initial_population candidates must be unique")

    if strategy == "reap_guided":
        if importance is None or importance.shape != (num_layers, num_experts):
            raise ValueError("reap_guided search requires matching importance scores")
        if not np.isfinite(importance).all():
            raise ValueError("importance must contain only finite values")

    population = np.ascontiguousarray(initial_population.copy())
    initial_hash = pruning_population_hash(population)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x45415052]))
    cache: dict[bytes, float] = {}
    trials: list[dict[str, Any]] = []
    population_fitness = np.empty(config.population_size, dtype=np.float64)
    population_ids = np.arange(config.population_size, dtype=np.int64)

    best_fitness = float("inf")
    best_mask = population[0].copy()
    best_evaluation = -1
    for evaluation, mask in enumerate(population):
        fitness = _evaluate(objective, mask)
        value = float(fitness.value)
        cache[_mask_key(mask)] = value
        population_fitness[evaluation] = value
        trials.append(
            {
                "evaluation": evaluation,
                "phase": "initialization",
                "generation": 0,
                "fitness": value,
                "candidate_sha256": sha256(_mask_key(mask)).hexdigest(),
                "metadata": dict(fitness.metadata),
            }
        )
        if (value, evaluation) < (best_fitness, best_evaluation):
            best_fitness = value
            best_mask = mask.copy()
            best_evaluation = evaluation

    evaluations = config.population_size
    proposals = 0
    cache_hits = 0
    fallback_candidates = 0
    duplicate_streak = 0
    generation = 0

    while evaluations < config.evaluation_budget:
        generation += 1
        target = min(config.offspring_size, config.evaluation_budget - evaluations)
        children: list[np.ndarray] = []
        child_fitnesses: list[float] = []
        child_ids: list[int] = []

        while len(children) < target:
            first_index = _tournament_index(
                population_fitness,
                population_ids,
                tournament_size=config.tournament_size,
                rng=rng,
            )
            parent = population[first_index].copy()
            second_parent_id: int | None = None
            crossover_layers: tuple[int, ...] = ()
            if config.crossover_rate > 0.0 and rng.random() < config.crossover_rate:
                second_index = _tournament_index(
                    population_fitness,
                    population_ids,
                    tournament_size=config.tournament_size,
                    rng=rng,
                )
                second_parent_id = int(population_ids[second_index])
                parent, crossover_layers = crossover_whole_layers(
                    parent, population[second_index], rng=rng
                )

            if strategy == "uniform":
                mutation = _uniform_swap(parent, rng)
            else:
                assert importance is not None
                mutation = mutate_reap_guided_swap(
                    parent,
                    importance,
                    rng=rng,
                    epsilon_uniform=config.epsilon_uniform,
                )
            candidate = mutation.mask
            proposals += 1
            key = _mask_key(candidate)
            if key in cache:
                cache_hits += 1
                duplicate_streak += 1
                if duplicate_streak < config.max_duplicate_attempts:
                    continue
                candidate = _first_unseen_mask(
                    num_layers=num_layers,
                    num_experts=num_experts,
                    retained_count=config.retained_count,
                    known=set(cache),
                )
                key = _mask_key(candidate)
                proposals += 1
                fallback_candidates += 1
                duplicate_streak = 0
                mutation_details: dict[str, Any] = {
                    "mutation_layer": None,
                    "removed_expert": None,
                    "added_expert": None,
                    "used_uniform": None,
                    "fallback_unseen": True,
                }
            else:
                duplicate_streak = 0
                mutation_details = {
                    "mutation_layer": mutation.layer,
                    "removed_expert": mutation.removed_expert,
                    "added_expert": mutation.added_expert,
                    "used_uniform": mutation.used_uniform,
                    "fallback_unseen": False,
                }

            fitness = _evaluate(objective, candidate)
            value = float(fitness.value)
            evaluation = evaluations
            evaluations += 1
            cache[key] = value
            children.append(candidate)
            child_fitnesses.append(value)
            child_ids.append(evaluation)
            row = {
                "evaluation": evaluation,
                "phase": "offspring",
                "generation": generation,
                "fitness": value,
                "candidate_sha256": sha256(key).hexdigest(),
                "parent_evaluation": int(population_ids[first_index]),
                "second_parent_evaluation": second_parent_id,
                "crossover_layers": list(crossover_layers),
                "metadata": dict(fitness.metadata),
            }
            row.update(mutation_details)
            trials.append(row)
            if (value, evaluation) < (best_fitness, best_evaluation):
                best_fitness = value
                best_mask = candidate.copy()
                best_evaluation = evaluation

        combined_population = np.concatenate((population, np.stack(children)), axis=0)
        combined_fitness = np.concatenate(
            (population_fitness, np.asarray(child_fitnesses, dtype=np.float64))
        )
        combined_ids = np.concatenate(
            (population_ids, np.asarray(child_ids, dtype=np.int64))
        )
        population, population_fitness, population_ids = _select_survivors(
            combined_population,
            combined_fitness,
            combined_ids,
            population_size=config.population_size,
        )

    return PruningSearchResult(
        strategy=strategy,
        seed=seed,
        best_mask=best_mask,
        best_fitness=best_fitness,
        best_evaluation=best_evaluation,
        evaluations=evaluations,
        proposals=proposals,
        cache_hits=cache_hits,
        fallback_candidates=fallback_candidates,
        initial_population_hash=initial_hash,
        trials=tuple(trials),
    )

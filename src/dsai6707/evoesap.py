"""Inference-only EvoESAP allocation search with a fixed expert ordering.

This is an independent NumPy implementation of Algorithm 1 and Equations
5--7 of ``EvoESAP: Non-Uniform Expert Pruning for Sparse MoE``
(https://arxiv.org/abs/2603.06003). The public implementation was consulted at
https://github.com/ZongfangLiu/EvoESAP/blob/main/src/reap/search_utils.py.

The fitness callback must return the *full-vocabulary*, teacher-forced ESAP
defined in Equations 14--16, not option-level overlap or answer accuracy.
Higher fitness is better. Rankings contain expert IDs in descending importance
order; a candidate always retains a prefix of that immutable order per layer.

The default controlled-comparison budget is 256 distinct evaluations:
32 initial candidates plus eight generations of 28 offspring, with four
elites carried without re-evaluation. This retains the paper's population,
elite, transfer, and composed-mutation settings but reduces its generation
budget. Global duplicate rejection is an explicit evaluation-budget adaptation.
Layer bounds are supplied by the caller: using TAPE's narrower bounds is a
controlled-space adaptation, not the original paper's full [routing-k, n]
retained-count search space. Every integer count inside the supplied interval
is allowed; there are no invented coarse capacity levels.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np

from .pruning_global_budget import build_mask_from_layer_rankings

__all__ = [
    "EvoESAPConfig",
    "EvoESAPResult",
    "level_switch_mutation",
    "run_evoesap",
]


@dataclass(frozen=True)
class EvoESAPConfig:
    """Explicit bounds and original-operator, budget-matched search controls."""

    total_retained: int
    min_retained_per_layer: int
    max_retained_per_layer: int
    seed: int = 6707
    population_size: int = 32
    elite_size: int = 4
    generations: int = 8
    max_transfer: int = 4
    mutation_cap: int = 3
    max_duplicate_attempts: int = 8192
    max_mutation_attempts: int = 4096
    initial_patterns: tuple[str, ...] = ("uniform", "up", "down", "mid", "mid-neg")

    def __post_init__(self) -> None:
        integer_fields = (
            "total_retained",
            "min_retained_per_layer",
            "max_retained_per_layer",
            "seed",
            "population_size",
            "elite_size",
            "generations",
            "max_transfer",
            "mutation_cap",
            "max_duplicate_attempts",
            "max_mutation_attempts",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if not 1 <= self.min_retained_per_layer <= self.max_retained_per_layer:
            raise ValueError("invalid retained-count bounds")
        if self.total_retained < 1 or self.seed < 0 or self.generations < 0:
            raise ValueError("invalid total_retained, seed, or generations")
        if not 1 <= self.elite_size < self.population_size:
            raise ValueError("elite_size must be positive and smaller than population_size")
        if (
            min(
                self.max_transfer,
                self.mutation_cap,
                self.max_duplicate_attempts,
                self.max_mutation_attempts,
            )
            < 1
        ):
            raise ValueError("mutation and retry settings must be positive")
        supported = {"uniform", "up", "down", "mid", "mid-neg"}
        if any(name not in supported for name in self.initial_patterns):
            raise ValueError("unsupported initial pattern")
        if not self.initial_patterns or self.initial_patterns[0] != "uniform":
            raise ValueError("initial_patterns must start with the uniform REAP candidate")
        if len(self.initial_patterns) > self.population_size:
            raise ValueError("initial patterns exceed the population size")

    @property
    def offspring_size(self) -> int:
        return self.population_size - self.elite_size

    @property
    def evaluation_budget(self) -> int:
        return self.population_size + self.generations * self.offspring_size


@dataclass(frozen=True)
class EvoESAPResult:
    """Candidate masks, scalar fitness, and JSON-compatible construction records."""

    masks: np.ndarray
    retained_counts: np.ndarray
    scores: np.ndarray
    trials: tuple[dict[str, Any], ...]
    generation_history: tuple[dict[str, Any], ...]
    final_population_ids: tuple[int, ...]
    ranked_candidate_ids: tuple[int, ...]
    proposals: int
    duplicate_proposals: int

    @property
    def unique_evaluations(self) -> int:
        return len(self.masks)

    @property
    def winner_id(self) -> int:
        return self.ranked_candidate_ids[0]

    @property
    def selection_ids(self) -> tuple[int, ...]:
        """Original EvoESAP selects the globally best ESAP, without test access."""
        return (self.winner_id,)


def _validate_counts(counts: np.ndarray, config: EvoESAPConfig) -> np.ndarray:
    values = np.asarray(counts)
    if values.ndim != 1 or len(values) < 2 or values.dtype.kind not in "iu":
        raise ValueError("retained counts must be an integer vector with at least two layers")
    if int(values.sum()) != config.total_retained:
        raise ValueError("retained counts do not conserve the exact global budget")
    if np.any(values < config.min_retained_per_layer) or np.any(
        values > config.max_retained_per_layer
    ):
        raise ValueError("retained counts violate the declared layer bounds")
    return values.astype(np.int64, copy=True)


def level_switch_mutation(
    counts: np.ndarray,
    *,
    config: EvoESAPConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[dict[str, int], ...]]:
    """Compose min(U[1,tau], U[1,tau]) feasible budget-preserving transfers.

    Distinct layer pairs and delta in 1..max_transfer are sampled uniformly,
    with infeasible proposals rejected, as in the paper's pseudocode. We use
    retained rather than removed counts; reversing the transfer gives the
    identical level-switch operator. Exhausted retries raise, never silently
    return an unmutated or partially mutated candidate.
    """
    child = _validate_counts(counts, config)
    target = int(min(rng.integers(1, config.mutation_cap + 1, size=2)))
    steps: list[dict[str, int]] = []
    for _ in range(target):
        for _attempt in range(config.max_mutation_attempts):
            donor, receiver = (int(v) for v in rng.choice(len(child), size=2, replace=False))
            delta = int(rng.integers(1, config.max_transfer + 1))
            if child[donor] - delta < config.min_retained_per_layer:
                continue
            if child[receiver] + delta > config.max_retained_per_layer:
                continue
            child[donor] -= delta
            child[receiver] += delta
            steps.append({"donor_layer": donor, "receiver_layer": receiver, "delta": delta})
            break
        else:
            raise RuntimeError("could not draw a feasible level-switch mutation")
    return _validate_counts(child, config), tuple(steps)


def _completion_counts(layers: int, width: int) -> list[list[int]]:
    """Integer dynamic program for exact uniform feasible-allocation sampling."""
    ways = [[1]]
    for count in range(1, layers + 1):
        previous = ways[-1]
        current = []
        for total in range(count * width + 1):
            current.append(
                sum(
                    previous[total - value]
                    for value in range(width + 1)
                    if 0 <= total - value < len(previous)
                )
            )
        ways.append(current)
    return ways


def _randbelow(rng: np.random.Generator, upper: int) -> int:
    """Draw an exact uniform Python integer, including counts above uint64."""
    if upper < 1:
        raise ValueError("random upper bound must be positive")
    byte_count = (upper.bit_length() + 7) // 8
    mask = (1 << upper.bit_length()) - 1
    while True:
        value = int.from_bytes(rng.bytes(byte_count), "little") & mask
        if value < upper:
            return value


def _sample_feasible(
    layers: int, config: EvoESAPConfig, ways: list[list[int]], rng: np.random.Generator
) -> np.ndarray:
    width = config.max_retained_per_layer - config.min_retained_per_layer
    remaining = config.total_retained - layers * config.min_retained_per_layer
    counts = np.full(layers, config.min_retained_per_layer, dtype=np.int64)
    for layer in range(layers):
        suffix_layers = layers - layer - 1
        choices = [
            value for value in range(width + 1) if 0 <= remaining - value < len(ways[suffix_layers])
        ]
        weights = [ways[suffix_layers][remaining - value] for value in choices]
        draw = _randbelow(rng, sum(weights))
        for value, weight in zip(choices, weights, strict=True):
            if draw < weight:
                counts[layer] += value
                remaining -= value
                break
            draw -= weight
    return _validate_counts(counts, config)


def _pattern_allocation(name: str, layers: int, config: EvoESAPConfig) -> np.ndarray:
    """Allocate removable budget by the official code's structured weights.

    A mandatory lower removal bound is subtracted first when the caller uses
    a restricted comparison space. Fractional residuals and round-robin fills
    preserve the exact global budget despite layer-capacity saturation.
    """
    if name == "uniform":
        counts = np.full(layers, config.total_retained // layers, dtype=np.int64)
        counts[: config.total_retained % layers] += 1
        return _validate_counts(counts, config)
    indices = np.arange(layers, dtype=np.float64)
    middle = (layers - 1) / 2
    weights = {
        "up": indices + 1,
        "down": layers - indices,
        "mid": middle - abs(indices - middle) + 1,
        "mid-neg": abs(indices - middle) + 1,
    }[name]
    width = config.max_retained_per_layer - config.min_retained_per_layer
    budget = layers * config.max_retained_per_layer - config.total_retained
    desired = budget * weights / weights.sum()
    removed = np.minimum(desired.astype(np.int64), width)
    remaining = budget - int(removed.sum())
    order = sorted(range(layers), key=lambda i: (-(desired[i] - removed[i]), i))
    for layer in order:
        if remaining and removed[layer] < width:
            removed[layer] += 1
            remaining -= 1
    while remaining:
        for layer in range(layers):
            if remaining and removed[layer] < width:
                removed[layer] += 1
                remaining -= 1
    return _validate_counts(config.max_retained_per_layer - removed, config)


def run_evoesap(
    objective: Callable[[np.ndarray], Sequence[float] | np.ndarray],
    rankings: np.ndarray,
    *,
    config: EvoESAPConfig,
) -> EvoESAPResult:
    """Run fixed-ranking elite search; call ``objective`` only on unique masks.

    The initial population combines uniform and patterned seeds, then samples
    uniformly from feasible integer allocations using exact completion counts.
    This follows the paper's initialization description; the official code
    instead uses random weights to fill its remaining slots. Both this choice
    and global duplicate rejection are recorded in the generation journal.
    """
    order = np.asarray(rankings)
    if order.ndim != 2 or min(order.shape) < 2 or order.dtype.kind not in "iu":
        raise ValueError("rankings must be an integer [layers, experts] array")
    layers, experts = order.shape
    if not np.all(np.sort(order, axis=1) == np.arange(experts)):
        raise ValueError("every ranking row must be a permutation of expert IDs")
    if config.max_retained_per_layer > experts:
        raise ValueError("maximum retained count exceeds the number of experts")
    width = config.max_retained_per_layer - config.min_retained_per_layer
    residual = config.total_retained - layers * config.min_retained_per_layer
    if not 0 <= residual <= layers * width:
        raise ValueError("global budget is infeasible for the declared layer bounds")
    ways = _completion_counts(layers, width)
    if ways[layers][residual] < config.evaluation_budget:
        raise ValueError("feasible search space contains fewer distinct allocations than budget")
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0x45534150]))
    bounds = {
        "total_retained": config.total_retained,
        "min_retained_per_layer": config.min_retained_per_layer,
        "max_retained_per_layer": config.max_retained_per_layer,
    }
    all_masks: list[np.ndarray] = []
    all_counts: list[np.ndarray] = []
    scores: list[float] = []
    trials: list[dict[str, Any]] = []
    known: set[tuple[int, ...]] = set()
    history: list[dict[str, Any]] = []
    proposals = duplicates = 0

    def register(counts: np.ndarray, context: dict[str, Any]) -> int | None:
        nonlocal proposals, duplicates
        proposals += 1
        counts = _validate_counts(counts, config)
        key = tuple(int(value) for value in counts)
        if key in known:
            duplicates += 1
            return None
        known.add(key)
        mask = build_mask_from_layer_rankings(order, counts, **bounds)
        index = len(all_masks)
        all_masks.append(mask)
        all_counts.append(counts)
        trials.append(
            {
                "candidate_id": index,
                "retained_counts": list(key),
                "mask_sha256": sha256(str(mask.shape).encode() + mask.tobytes()).hexdigest(),
                **context,
            }
        )
        return index

    def evaluate(ids: list[int]) -> None:
        values = np.asarray(
            objective(np.stack([all_masks[index] for index in ids])), dtype=np.float64
        )
        if values.shape != (len(ids),) or not np.isfinite(values).all():
            raise ValueError("ESAP callback must return one finite scalar per candidate")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("ESAP values must lie in [0, 1]")
        for index, value in zip(ids, values, strict=True):
            if index != len(scores):
                raise RuntimeError("candidate score journal is out of order")
            scores.append(float(value))
            trials[index]["esap"] = float(value)

    population: list[int] = []
    for name in config.initial_patterns:
        index = register(
            _pattern_allocation(name, layers, config),
            {
                "generation": 0,
                "source": f"pattern:{name}",
                "parent_id": None,
                "mutation_steps": [],
            },
        )
        if index is not None:
            population.append(index)
    for _attempt in range(config.max_duplicate_attempts):
        if len(population) == config.population_size:
            break
        index = register(
            _sample_feasible(layers, config, ways, rng),
            {
                "generation": 0,
                "source": "uniform_feasible_allocation",
                "parent_id": None,
                "mutation_steps": [],
            },
        )
        if index is not None:
            population.append(index)
    if len(population) != config.population_size:
        raise RuntimeError("could not construct the distinct initial population")
    evaluate(population)

    def record(generation: int, elites: tuple[int, ...]) -> None:
        history.append(
            {
                "generation": generation,
                "population_ids": list(population),
                "elite_ids": list(elites),
                "best_candidate_id": min(population, key=lambda i: (-scores[i], i)),
                "unique_evaluations": len(scores),
            }
        )

    record(0, ())
    for generation in range(1, config.generations + 1):
        elites = tuple(sorted(population, key=lambda i: (-scores[i], i))[: config.elite_size])
        offspring: list[int] = []
        for _attempt in range(config.max_duplicate_attempts):
            if len(offspring) == config.offspring_size:
                break
            parent_id = elites[int(rng.integers(len(elites)))]
            counts, steps = level_switch_mutation(all_counts[parent_id], config=config, rng=rng)
            index = register(
                counts,
                {
                    "generation": generation,
                    "source": "level_switch",
                    "parent_id": parent_id,
                    "mutation_steps": list(steps),
                    "mutation_count": len(steps),
                },
            )
            if index is not None:
                offspring.append(index)
        if len(offspring) != config.offspring_size:
            raise RuntimeError("could not draw enough distinct level-switch offspring")
        evaluate(offspring)
        population = list(elites) + offspring
        record(generation, elites)
    ranked = tuple(sorted(range(len(scores)), key=lambda i: (-scores[i], i)))
    return EvoESAPResult(
        masks=np.stack(all_masks),
        retained_counts=np.stack(all_counts),
        scores=np.asarray(scores, dtype=np.float64),
        trials=tuple(trials),
        generation_history=tuple(history),
        final_population_ids=tuple(population),
        ranked_candidate_ids=ranked,
        proposals=proposals,
        duplicate_proposals=duplicates,
    )

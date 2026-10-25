"""Deterministic two-objective search for globally budgeted MoE pruning.

The search has two deliberately comparable modes.  ``allocation_only``
searches layer budgets while rebuilding every intermediate mask from a fixed
per-layer REAP ranking.  ``joint`` keeps the identities produced by global-set
crossover and additionally mutates expert identities.  Both modes share the
same initial population, NSGA-II selection, evaluation budget, and budget
transfer distribution.

This module contains no model code.  An experiment driver supplies a batched
two-objective callback; consequently all representation, selection, and
journaling behaviour can be tested on CPU before any GPU evaluation starts.
All objectives are minimized.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal

import numpy as np

from .pruning_global_budget import (
    BudgetTransfer,
    build_mask_from_layer_rankings,
    crossover_global_budget,
    global_budget_mask_sha256,
    mutate_budget_transfer,
    mutate_identity_swap,
    validate_global_budget_mask,
)
from .pruning_nsga2 import (
    crowded_tournament,
    nondominated_sort,
    rank_and_crowding,
    select_survivors,
)

SearchMode = Literal["allocation_only", "joint"]

__all__ = [
    "GlobalBudgetInitialPopulation",
    "GlobalBudgetNSGA2Result",
    "GlobalBudgetSearchConfig",
    "SearchMode",
    "TwoObjectiveShortlist",
    "make_initial_population_from_rankings",
    "run_global_budget_nsga2",
    "two_objective_shortlist",
]

_INITIAL_RNG_NAMESPACE = 0x4742494E
_SEARCH_RNG_NAMESPACE = 0x47425345


@dataclass(frozen=True)
class GlobalBudgetSearchConfig:
    """Search controls for the 24-by-32, 672-slot experiment.

    Defaults encode the locked pilot: 16 initial candidates followed by seven
    generations of 16 unique offspring, for 128 unique evaluations per arm.
    Geometry is configurable so the pure-NumPy implementation remains useful
    for small tests and follow-up experiments.
    """

    population_size: int = 16
    offspring_size: int = 16
    generations: int = 7
    tournament_size: int = 2
    seed: int = 6707
    total_retained: int = 672
    min_retained_per_layer: int = 26
    max_retained_per_layer: int = 32
    transfer_counts: tuple[int, ...] = (1, 2, 3)
    transfer_probabilities: tuple[float, ...] = (0.5, 0.3, 0.2)
    identity_swap_counts: tuple[int, ...] = (1, 2, 3)
    identity_swap_probabilities: tuple[float, ...] = (0.5, 0.3, 0.2)
    crossover_probability: float = 1.0
    joint_mutation_pairs: tuple[tuple[int, int], ...] | None = None
    joint_mutation_probabilities: tuple[float, ...] | None = None
    max_duplicate_attempts: int = 512

    def __post_init__(self) -> None:
        integer_fields = {
            "population_size": self.population_size,
            "offspring_size": self.offspring_size,
            "generations": self.generations,
            "tournament_size": self.tournament_size,
            "seed": self.seed,
            "total_retained": self.total_retained,
            "min_retained_per_layer": self.min_retained_per_layer,
            "max_retained_per_layer": self.max_retained_per_layer,
            "max_duplicate_attempts": self.max_duplicate_attempts,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_fields.values()
        ):
            raise ValueError("integer search settings must be integers")
        if self.population_size < 2 or self.offspring_size < 1 or self.generations < 0:
            raise ValueError("population_size >= 2, offspring_size >= 1, and generations >= 0")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if not 1 <= self.tournament_size <= self.population_size - 1:
            raise ValueError("tournament_size must fit the parent pool")
        if self.total_retained < 1:
            raise ValueError("total_retained must be positive")
        if not 1 <= self.min_retained_per_layer <= self.max_retained_per_layer:
            raise ValueError("invalid per-layer retained bounds")
        if self.max_duplicate_attempts < 1:
            raise ValueError("max_duplicate_attempts must be positive")
        _validate_count_distribution(
            self.transfer_counts,
            self.transfer_probabilities,
            name="transfer",
        )
        _validate_count_distribution(
            self.identity_swap_counts,
            self.identity_swap_probabilities,
            name="identity swap",
        )
        if (
            isinstance(self.crossover_probability, bool | np.bool_)
            or not isinstance(
                self.crossover_probability,
                int | float | np.integer | np.floating,
            )
            or not np.isfinite(self.crossover_probability)
            or not 0.0 <= float(self.crossover_probability) <= 1.0
        ):
            raise ValueError("crossover_probability must be a finite number in [0, 1]")
        if (self.joint_mutation_pairs is None) != (self.joint_mutation_probabilities is None):
            raise ValueError(
                "joint_mutation_pairs and joint_mutation_probabilities must be supplied together"
            )
        if self.joint_mutation_pairs is not None:
            _validate_joint_mutation_distribution(
                self.joint_mutation_pairs,
                self.joint_mutation_probabilities,
            )

    @property
    def evaluation_budget(self) -> int:
        """Number of unique candidates evaluated by one complete run."""

        return self.population_size + self.generations * self.offspring_size


@dataclass(frozen=True)
class GlobalBudgetInitialPopulation:
    """Shared, score-independent population used to start both search arms."""

    masks: np.ndarray
    contexts: tuple[dict[str, Any], ...]
    population_sha256: str

    @property
    def journal(self) -> tuple[dict[str, Any], ...]:
        """Alias emphasizing that contexts are serializable construction records."""

        return self.contexts


@dataclass(frozen=True)
class TwoObjectiveShortlist:
    """Named task extremes and balanced selections from the first Pareto front."""

    roles: tuple[tuple[str, int], ...]
    candidate_ids: tuple[int, ...]


@dataclass(frozen=True)
class GlobalBudgetNSGA2Result:
    """Complete in-memory evidence from one search arm."""

    mode: SearchMode
    masks: np.ndarray
    retained_counts: np.ndarray
    objectives: np.ndarray
    trials: tuple[dict[str, Any], ...]
    final_population_ids: tuple[int, ...]
    proposals: int
    duplicate_proposals: int
    initial_population_sha256: str
    shortlist: TwoObjectiveShortlist

    @property
    def unique_evaluations(self) -> int:
        """Number of recorded unique masks."""

        return int(self.masks.shape[0])


def _validate_count_distribution(
    counts: Sequence[int], probabilities: Sequence[float], *, name: str
) -> None:
    if len(counts) < 1 or len(counts) != len(probabilities):
        raise ValueError(f"{name} counts and probabilities must have equal nonzero length")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in counts):
        raise ValueError(f"{name} counts must be positive integers")
    values = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError(f"{name} probabilities must be finite and nonnegative")
    if not np.isclose(float(np.sum(values)), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name} probabilities must sum to one")


def _validate_joint_mutation_distribution(
    pairs: Sequence[Sequence[int]], probabilities: Sequence[float] | None
) -> None:
    if probabilities is None or len(pairs) < 1 or len(pairs) != len(probabilities):
        raise ValueError("joint mutation pairs and probabilities must have equal nonzero length")
    for pair in pairs:
        if not isinstance(pair, Sequence) or isinstance(pair, str | bytes) or len(pair) != 2:
            raise ValueError(
                "each joint mutation pair must contain transfer and identity-swap counts"
            )
        transfer_count, identity_swap_count = pair
        if any(
            isinstance(value, bool | np.bool_)
            or not isinstance(value, int | np.integer)
            or int(value) < 0
            for value in pair
        ):
            raise ValueError("joint mutation counts must be nonnegative integers")
        if int(transfer_count) == 0 and int(identity_swap_count) == 0:
            raise ValueError("a joint mutation pair cannot be (0, 0)")
    values = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("joint mutation probabilities must be finite and nonnegative")
    if not np.isclose(float(np.sum(values)), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("joint mutation probabilities must sum to one")


def _problem_shape(rankings: np.ndarray, config: GlobalBudgetSearchConfig) -> tuple[int, int]:
    values = np.asarray(rankings)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("rankings must have shape [layers, experts]")
    layers, experts = values.shape
    if config.max_retained_per_layer > experts:
        raise ValueError("max_retained_per_layer exceeds the expert count")
    if not (
        layers * config.min_retained_per_layer
        <= config.total_retained
        <= layers * config.max_retained_per_layer
    ):
        raise ValueError("total_retained is infeasible for rankings and layer bounds")
    if config.total_retained in (
        layers * config.min_retained_per_layer,
        layers * config.max_retained_per_layer,
    ):
        raise ValueError("search requires at least one legal cross-layer budget transfer")
    if config.total_retained % layers:
        raise ValueError("initialization requires an integer uniform retained count")
    uniform = config.total_retained // layers
    if not config.min_retained_per_layer <= uniform <= config.max_retained_per_layer:
        raise ValueError("uniform retained count violates the layer bounds")

    # This call validates integer dtype and every ranking row's permutation.
    build_mask_from_layer_rankings(
        values,
        np.full(layers, uniform, dtype=np.int64),
        total_retained=config.total_retained,
        min_retained_per_layer=config.min_retained_per_layer,
        max_retained_per_layer=config.max_retained_per_layer,
    )
    return layers, experts


def _counts(mask: np.ndarray) -> list[int]:
    return [int(value) for value in np.count_nonzero(mask, axis=1)]


def _transfer_dict(transfer: BudgetTransfer) -> dict[str, int]:
    return {
        "donor_layer": int(transfer.donor_layer),
        "receiver_layer": int(transfer.receiver_layer),
        "removed_expert": int(transfer.removed_expert),
        "added_expert": int(transfer.added_expert),
        "donor_count_before": int(transfer.donor_count_before),
        "receiver_count_before": int(transfer.receiver_count_before),
    }


def _identity_delta(before: np.ndarray, after: np.ndarray) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for layer in range(before.shape[0]):
        removed = [int(value) for value in np.flatnonzero(before[layer] & ~after[layer])]
        added = [int(value) for value in np.flatnonzero(~before[layer] & after[layer])]
        if removed or added:
            changes.append({"layer": layer, "removed": removed, "added": added})
    return changes


def _mask_from_counts(
    rankings: np.ndarray,
    counts: Sequence[int],
    config: GlobalBudgetSearchConfig,
) -> np.ndarray:
    return build_mask_from_layer_rankings(
        rankings,
        np.asarray(counts, dtype=np.int64),
        total_retained=config.total_retained,
        min_retained_per_layer=config.min_retained_per_layer,
        max_retained_per_layer=config.max_retained_per_layer,
    )


def _population_sha256(masks: Sequence[np.ndarray], config: GlobalBudgetSearchConfig) -> str:
    digests = [
        global_budget_mask_sha256(
            mask,
            total_retained=config.total_retained,
            min_retained_per_layer=config.min_retained_per_layer,
            max_retained_per_layer=config.max_retained_per_layer,
        )
        for mask in masks
    ]
    payload = b"dsai6707-global-budget-population-v1\0" + "\n".join(digests).encode("ascii")
    return sha256(payload).hexdigest()


def _bounded_patterns(
    layers: int,
    *,
    uniform: int,
    minimum: int,
    maximum: int,
) -> tuple[tuple[str, np.ndarray], ...]:
    delta = min(2, maximum - uniform, uniform - minimum)
    if delta < 1:
        return ()

    signs: list[tuple[str, np.ndarray]] = []
    third = layers // 3
    if third:
        front = np.zeros(layers, dtype=np.int64)
        front[:third] = 1
        front[-third:] = -1
        signs.extend((("front_loaded", front), ("back_loaded", -front)))

    alternating = np.where(np.arange(layers) % 2 == 0, 1, -1).astype(np.int64)
    if int(np.sum(alternating)) == 0:
        signs.extend((("even_loaded", alternating), ("odd_loaded", -alternating)))

    if layers % 2 == 0:
        quarter = layers // 4
        if quarter:
            outer = np.full(layers, -1, dtype=np.int64)
            outer[:quarter] = 1
            outer[-quarter:] = 1
            signs.extend((("outer_loaded", outer), ("middle_loaded", -outer)))

    patterns: list[tuple[str, np.ndarray]] = []
    seen: set[bytes] = set()
    for name, sign in signs:
        if int(np.sum(sign)) != 0:
            continue
        counts = np.full(layers, uniform, dtype=np.int64) + delta * sign
        key = counts.tobytes()
        if key in seen or np.any(counts < minimum) or np.any(counts > maximum):
            continue
        seen.add(key)
        patterns.append((name, counts))
    return tuple(patterns)


def make_initial_population_from_rankings(
    rankings: np.ndarray,
    *,
    config: GlobalBudgetSearchConfig | None = None,
) -> GlobalBudgetInitialPopulation:
    """Build deterministic, unique, REAP-ranking-prefix initial masks.

    The first candidate is uniform.  It is followed by bounded non-uniform
    layer patterns and, when needed, random legal budget-transfer walks from
    the uniform anchor.  No objective value is consulted.
    """

    if config is None:
        config = GlobalBudgetSearchConfig()
    ranking_values = np.asarray(rankings)
    layers, _ = _problem_shape(ranking_values, config)
    uniform = config.total_retained // layers
    uniform_counts = np.full(layers, uniform, dtype=np.int64)
    uniform_mask = _mask_from_counts(ranking_values, uniform_counts, config)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, _INITIAL_RNG_NAMESPACE]))

    masks: list[np.ndarray] = []
    contexts: list[dict[str, Any]] = []
    known: set[str] = set()

    def add(mask: np.ndarray, context: dict[str, Any]) -> bool:
        digest = global_budget_mask_sha256(
            mask,
            total_retained=config.total_retained,
            min_retained_per_layer=config.min_retained_per_layer,
            max_retained_per_layer=config.max_retained_per_layer,
        )
        if digest in known:
            return False
        known.add(digest)
        masks.append(np.ascontiguousarray(mask.copy()))
        contexts.append(context)
        return True

    add(
        uniform_mask,
        {
            "kind": "uniform_reap_ranking",
            "pattern": "uniform",
            "parent_ids": [],
            "operations": [],
            "retained_counts": [uniform] * layers,
        },
    )
    for name, pattern_counts in _bounded_patterns(
        layers,
        uniform=uniform,
        minimum=config.min_retained_per_layer,
        maximum=config.max_retained_per_layer,
    ):
        if len(masks) >= config.population_size:
            break
        mask = _mask_from_counts(ranking_values, pattern_counts, config)
        add(
            mask,
            {
                "kind": "bounded_pattern",
                "pattern": name,
                "parent_ids": [],
                "operations": [
                    {
                        "operator": "bounded_count_pattern",
                        "counts_before": [uniform] * layers,
                        "counts_after": [int(value) for value in pattern_counts],
                    }
                ],
                "retained_counts": [int(value) for value in pattern_counts],
            },
        )

    attempts = 0
    while len(masks) < config.population_size:
        attempts += 1
        if attempts > 10_000:
            raise RuntimeError("could not construct a unique initial population")
        working = uniform_mask.copy()
        operations: list[dict[str, Any]] = []
        step_count = 1 + (attempts - 1) % max(1, layers // 2)
        for step in range(step_count):
            before = working
            mutation = mutate_budget_transfer(
                before,
                rng=rng,
                total_retained=config.total_retained,
                min_retained_per_layer=config.min_retained_per_layer,
                max_retained_per_layer=config.max_retained_per_layer,
            )
            working = _mask_from_counts(ranking_values, _counts(mutation.mask), config)
            operations.append(
                {
                    "operator": "random_budget_transfer_then_ranking_projection",
                    "step": step + 1,
                    "transfer": _transfer_dict(mutation.transfer),
                    "counts_before": _counts(before),
                    "counts_after": _counts(working),
                    "identity_delta": _identity_delta(before, working),
                }
            )
        add(
            working,
            {
                "kind": "random_transfer",
                "pattern": None,
                "parent_ids": [],
                "operations": operations,
                "retained_counts": _counts(working),
            },
        )

    population = np.stack(masks)
    return GlobalBudgetInitialPopulation(
        masks=population,
        contexts=tuple(contexts),
        population_sha256=_population_sha256(population, config),
    )


def _objective_pair(value: Any) -> tuple[float, float]:
    raw = value.objectives if hasattr(value, "objectives") else value
    pair = np.asarray(raw, dtype=np.float64)
    if pair.shape != (2,) or not np.isfinite(pair).all():
        raise RuntimeError("objective callback must return exactly two finite objectives")
    return float(pair[0]), float(pair[1])


def _validate_initial_population(
    initial: GlobalBudgetInitialPopulation,
    *,
    rankings: np.ndarray,
    config: GlobalBudgetSearchConfig,
) -> tuple[np.ndarray, tuple[dict[str, Any], ...], list[str]]:
    population = np.asarray(initial.masks)
    expected_shape = (config.population_size, *rankings.shape)
    if population.dtype != np.bool_ or population.shape != expected_shape:
        raise ValueError("initial population has the wrong shape or dtype")
    if len(initial.contexts) != config.population_size:
        raise ValueError("one initial context is required per mask")
    keys: list[str] = []
    for mask in population:
        validate_global_budget_mask(
            mask,
            total_retained=config.total_retained,
            min_retained_per_layer=config.min_retained_per_layer,
            max_retained_per_layer=config.max_retained_per_layer,
        )
        keys.append(
            global_budget_mask_sha256(
                mask,
                total_retained=config.total_retained,
                min_retained_per_layer=config.min_retained_per_layer,
                max_retained_per_layer=config.max_retained_per_layer,
            )
        )
    if len(set(keys)) != len(keys):
        raise ValueError("initial population masks must be unique")
    actual_population_hash = _population_sha256(population, config)
    if actual_population_hash != initial.population_sha256:
        raise ValueError("initial population hash does not match its masks")
    return (
        np.ascontiguousarray(population.copy()),
        tuple(dict(context) for context in initial.contexts),
        keys,
    )


def _project_to_rankings(
    mask: np.ndarray,
    *,
    rankings: np.ndarray,
    config: GlobalBudgetSearchConfig,
) -> np.ndarray:
    return _mask_from_counts(rankings, _counts(mask), config)


def _offspring_candidate(
    crossed: np.ndarray,
    *,
    repairs: Sequence[BudgetTransfer],
    rankings: np.ndarray,
    mode: SearchMode,
    rng: np.random.Generator,
    config: GlobalBudgetSearchConfig,
    reproduction_operation: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], int, int]:
    operations: list[dict[str, Any]] = [
        dict(reproduction_operation)
        if reproduction_operation is not None
        else {
            "operator": "global_set_crossover",
            "bound_repairs": [_transfer_dict(transfer) for transfer in repairs],
            "counts_after": _counts(crossed),
        }
    ]
    working = np.ascontiguousarray(crossed.copy())
    if mode == "allocation_only":
        projected = _project_to_rankings(working, rankings=rankings, config=config)
        operations.append(
            {
                "operator": "reap_ranking_projection",
                "after": operations[0]["operator"],
                "counts_before": _counts(working),
                "counts_after": _counts(projected),
                "identity_delta": _identity_delta(working, projected),
            }
        )
        working = projected

    paired_identity_swap_count: int | None = None
    if mode == "joint" and config.joint_mutation_pairs is not None:
        assert config.joint_mutation_probabilities is not None
        pair_index = int(
            rng.choice(
                len(config.joint_mutation_pairs),
                p=config.joint_mutation_probabilities,
            )
        )
        raw_transfer_count, raw_identity_swap_count = config.joint_mutation_pairs[pair_index]
        transfer_count = int(raw_transfer_count)
        paired_identity_swap_count = int(raw_identity_swap_count)
    else:
        transfer_count = int(rng.choice(config.transfer_counts, p=config.transfer_probabilities))
    for step in range(transfer_count):
        before = working
        mutation = mutate_budget_transfer(
            before,
            rng=rng,
            total_retained=config.total_retained,
            min_retained_per_layer=config.min_retained_per_layer,
            max_retained_per_layer=config.max_retained_per_layer,
        )
        if mode == "allocation_only":
            working = _project_to_rankings(mutation.mask, rankings=rankings, config=config)
            operator = "budget_transfer_then_reap_ranking_projection"
        else:
            working = mutation.mask
            operator = "budget_transfer"
        operations.append(
            {
                "operator": operator,
                "step": step + 1,
                "transfer": _transfer_dict(mutation.transfer),
                "counts_before": _counts(before),
                "counts_after": _counts(working),
                "identity_delta": _identity_delta(before, working),
            }
        )

    identity_swap_count = 0
    if mode == "joint":
        identity_swap_count = (
            int(
                rng.choice(
                    config.identity_swap_counts,
                    p=config.identity_swap_probabilities,
                )
            )
            if paired_identity_swap_count is None
            else paired_identity_swap_count
        )
        eligible_layers = np.flatnonzero(
            (np.count_nonzero(working, axis=1) > 0) & (~np.all(working, axis=1))
        )
        if identity_swap_count > eligible_layers.size:
            raise ValueError("identity swap count exceeds the number of eligible layers")
        swap_layers = (
            rng.choice(
                eligible_layers,
                size=identity_swap_count,
                replace=False,
            )
            if identity_swap_count
            else np.empty(0, dtype=np.int64)
        )
        for step, raw_layer in enumerate(swap_layers):
            before = working
            mutation = mutate_identity_swap(
                before,
                rng=rng,
                total_retained=config.total_retained,
                min_retained_per_layer=config.min_retained_per_layer,
                max_retained_per_layer=config.max_retained_per_layer,
                layer=int(raw_layer),
            )
            working = mutation.mask
            operations.append(
                {
                    "operator": "identity_swap",
                    "step": step + 1,
                    "layer": int(mutation.layer),
                    "removed_expert": int(mutation.removed_expert),
                    "added_expert": int(mutation.added_expert),
                    "counts_before": _counts(before),
                    "counts_after": _counts(working),
                }
            )

    validate_global_budget_mask(
        working,
        total_retained=config.total_retained,
        min_retained_per_layer=config.min_retained_per_layer,
        max_retained_per_layer=config.max_retained_per_layer,
    )
    return np.ascontiguousarray(working), operations, transfer_count, identity_swap_count


def two_objective_shortlist(
    objectives: np.ndarray,
    *,
    candidate_ids: Sequence[int] | np.ndarray | None = None,
    reference_candidate_id: int = 0,
) -> TwoObjectiveShortlist:
    """Select evolved task extremes plus minimax and mean balanced candidates.

    Selection is restricted to the first Pareto front and excludes the
    reference candidate, which a runner can evaluate separately.  Balanced
    roles use objective ratios relative to that reference.  Role selections
    may coincide; ``candidate_ids`` contains their stable first-occurrence
    de-duplication in role order.  If no evolved candidate remains on the
    first front, both fields are empty.
    """

    values = np.asarray(objectives, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != 2:
        raise ValueError("objectives must have shape [candidates, 2]")
    if not np.isfinite(values).all():
        raise ValueError("objectives must be finite")
    if candidate_ids is None:
        ids = np.arange(values.shape[0], dtype=np.int64)
    else:
        raw_ids = np.asarray(candidate_ids)
        if (
            raw_ids.ndim != 1
            or raw_ids.shape[0] != values.shape[0]
            or raw_ids.dtype.kind not in "iu"
        ):
            raise ValueError("candidate_ids must be one integer ID per objective row")
        ids = raw_ids.astype(np.int64, copy=False)
        if np.unique(ids).size != ids.size:
            raise ValueError("candidate_ids must be unique")

    reference_matches = np.flatnonzero(ids == reference_candidate_id)
    if reference_matches.size != 1:
        raise ValueError("reference_candidate_id is not present exactly once")
    reference = values[int(reference_matches[0])]
    if np.any(reference <= 0.0):
        raise ValueError("reference objectives must be positive")
    relative = values / reference
    first_front = nondominated_sort(values, candidate_ids=ids)[0]
    front = tuple(index for index in first_front if int(ids[index]) != reference_candidate_id)
    if not front:
        return TwoObjectiveShortlist(roles=(), candidate_ids=())

    def select(key: Callable[[int], tuple[float, ...]]) -> int:
        position = min(front, key=key)
        return int(ids[position])

    roles = (
        (
            "task_0_extreme",
            select(lambda index: (float(values[index, 0]), float(ids[index]))),
        ),
        (
            "task_1_extreme",
            select(lambda index: (float(values[index, 1]), float(ids[index]))),
        ),
        (
            "minimax",
            select(
                lambda index: (
                    float(np.max(relative[index])),
                    float(np.mean(relative[index])),
                    float(ids[index]),
                )
            ),
        ),
        (
            "mean",
            select(
                lambda index: (
                    float(np.mean(relative[index])),
                    float(np.max(relative[index])),
                    float(ids[index]),
                )
            ),
        ),
    )
    unique: list[int] = []
    for _, candidate_id in roles:
        if candidate_id not in unique:
            unique.append(candidate_id)
    return TwoObjectiveShortlist(roles=roles, candidate_ids=tuple(unique))


def run_global_budget_nsga2(
    objective_many: Callable[[Sequence[np.ndarray]], Sequence[Any]],
    rankings: np.ndarray,
    *,
    mode: SearchMode,
    config: GlobalBudgetSearchConfig | None = None,
    initial_population: GlobalBudgetInitialPopulation | None = None,
) -> GlobalBudgetNSGA2Result:
    """Run one deterministic allocation-only or joint NSGA-II arm."""

    if config is None:
        config = GlobalBudgetSearchConfig()
    if mode not in ("allocation_only", "joint"):
        raise ValueError("mode must be 'allocation_only' or 'joint'")
    ranking_values = np.asarray(rankings)
    _problem_shape(ranking_values, config)
    initial = (
        make_initial_population_from_rankings(ranking_values, config=config)
        if initial_population is None
        else initial_population
    )
    population, initial_contexts, initial_keys = _validate_initial_population(
        initial,
        rankings=ranking_values,
        config=config,
    )
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, _SEARCH_RNG_NAMESPACE]))

    all_masks: list[np.ndarray] = []
    all_objectives: list[tuple[float, float]] = []
    trials: list[dict[str, Any]] = []
    known = set(initial_keys)
    proposals = 0
    duplicate_proposals = 0

    def evaluate(
        masks: Sequence[np.ndarray],
        contexts: Sequence[Mapping[str, Any]],
        *,
        phase: str,
        generation: int,
    ) -> np.ndarray:
        evaluation_masks = tuple(np.ascontiguousarray(mask.copy()) for mask in masks)
        callback_masks = tuple(mask.copy() for mask in evaluation_masks)
        returned = tuple(objective_many(callback_masks))
        if len(returned) != len(callback_masks):
            raise RuntimeError("objective callback returned the wrong number of values")
        candidate_ids: list[int] = []
        for mask, context, raw_value in zip(evaluation_masks, contexts, returned, strict=True):
            objective = _objective_pair(raw_value)
            candidate_id = len(all_masks)
            digest = global_budget_mask_sha256(
                mask,
                total_retained=config.total_retained,
                min_retained_per_layer=config.min_retained_per_layer,
                max_retained_per_layer=config.max_retained_per_layer,
            )
            counts = _counts(mask)
            all_masks.append(mask)
            all_objectives.append(objective)
            trial = {
                **dict(context),
                "candidate_id": candidate_id,
                "candidate_sha256": digest,
                "mode": mode,
                "phase": phase,
                "generation": generation,
                "objectives": [objective[0], objective[1]],
                "retained_counts": counts,
            }
            trials.append(trial)
            candidate_ids.append(candidate_id)
        return np.asarray(candidate_ids, dtype=np.int64)

    population_ids = evaluate(
        list(population),
        initial_contexts,
        phase="initialization",
        generation=0,
    )
    population_objectives = np.asarray(
        [all_objectives[int(candidate_id)] for candidate_id in population_ids],
        dtype=np.float64,
    )

    for generation in range(1, config.generations + 1):
        ranks, distances = rank_and_crowding(
            population_objectives,
            candidate_ids=population_ids,
        )
        child_masks: list[np.ndarray] = []
        child_contexts: list[dict[str, Any]] = []
        duplicate_streak = 0
        mating_event = 0
        while len(child_masks) < config.offspring_size:
            if duplicate_streak >= config.max_duplicate_attempts:
                raise RuntimeError("NSGA-II exceeded the duplicate-proposal limit")
            first_position = crowded_tournament(
                ranks,
                distances,
                population_ids,
                rng=rng,
                tournament_size=config.tournament_size,
            )
            second_pool = [
                index for index in range(config.population_size) if index != first_position
            ]
            second_position = crowded_tournament(
                ranks,
                distances,
                population_ids,
                rng=rng,
                tournament_size=config.tournament_size,
                pool=second_pool,
            )
            first_parent_id = int(population_ids[first_position])
            second_parent_id = int(population_ids[second_position])
            first_parent = all_masks[first_parent_id]
            second_parent = all_masks[second_parent_id]
            crossover_probability = float(config.crossover_probability)
            use_crossover = crossover_probability == 1.0 or (
                crossover_probability > 0.0 and float(rng.random()) < crossover_probability
            )
            mating_event += 1
            if use_crossover:
                crossover = crossover_global_budget(
                    first_parent,
                    second_parent,
                    rng=rng,
                    total_retained=config.total_retained,
                    min_retained_per_layer=config.min_retained_per_layer,
                    max_retained_per_layer=config.max_retained_per_layer,
                )
                siblings = (
                    (crossover.first_child, crossover.first_repairs, None),
                    (crossover.second_child, crossover.second_repairs, None),
                )
            else:
                siblings = (
                    (
                        first_parent,
                        (),
                        {
                            "operator": "parent_clone",
                            "source_parent_id": first_parent_id,
                            "counts_after": _counts(first_parent),
                        },
                    ),
                    (
                        second_parent,
                        (),
                        {
                            "operator": "parent_clone",
                            "source_parent_id": second_parent_id,
                            "counts_after": _counts(second_parent),
                        },
                    ),
                )
            for sibling, (crossed, repairs, reproduction_operation) in enumerate(siblings):
                proposals += 1
                child, operations, transfer_count, identity_swap_count = _offspring_candidate(
                    crossed,
                    repairs=repairs,
                    rankings=ranking_values,
                    mode=mode,
                    rng=rng,
                    config=config,
                    reproduction_operation=reproduction_operation,
                )
                digest = global_budget_mask_sha256(
                    child,
                    total_retained=config.total_retained,
                    min_retained_per_layer=config.min_retained_per_layer,
                    max_retained_per_layer=config.max_retained_per_layer,
                )
                if digest in known:
                    duplicate_proposals += 1
                    duplicate_streak += 1
                    continue
                duplicate_streak = 0
                known.add(digest)
                child_masks.append(child)
                child_contexts.append(
                    {
                        "kind": "offspring",
                        "parent_ids": [first_parent_id, second_parent_id],
                        "parent_counts": [_counts(first_parent), _counts(second_parent)],
                        "mating_event": mating_event,
                        "sibling": sibling,
                        "transfer_count": transfer_count,
                        "identity_swap_count": identity_swap_count,
                        "operations": operations,
                    }
                )
                if len(child_masks) == config.offspring_size:
                    break

        child_ids = evaluate(
            child_masks,
            child_contexts,
            phase="offspring",
            generation=generation,
        )
        child_objectives = np.asarray(
            [all_objectives[int(candidate_id)] for candidate_id in child_ids],
            dtype=np.float64,
        )
        combined_ids = np.concatenate((population_ids, child_ids))
        combined_objectives = np.concatenate((population_objectives, child_objectives), axis=0)
        survivor_positions = select_survivors(
            combined_objectives,
            combined_ids,
            config.population_size,
        )
        population_ids = combined_ids[survivor_positions]
        population_objectives = combined_objectives[survivor_positions]

    if len(all_masks) != config.evaluation_budget or len(known) != config.evaluation_budget:
        raise RuntimeError("NSGA-II unique-evaluation accounting failed")
    mask_array = np.stack(all_masks)
    objective_array = np.asarray(all_objectives, dtype=np.float64)
    return GlobalBudgetNSGA2Result(
        mode=mode,
        masks=mask_array,
        retained_counts=np.count_nonzero(mask_array, axis=2).astype(np.int64),
        objectives=objective_array,
        trials=tuple(trials),
        final_population_ids=tuple(int(value) for value in population_ids),
        proposals=proposals,
        duplicate_proposals=duplicate_proposals,
        initial_population_sha256=initial.population_sha256,
        shortlist=two_objective_shortlist(objective_array),
    )

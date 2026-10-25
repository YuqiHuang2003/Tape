"""Inference-independent, matched-budget non-evolutionary search controls."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .pruning_global_budget import global_budget_mask_sha256
from .pruning_global_budget_search import GlobalBudgetSearchConfig, _offspring_candidate
from .pruning_multistart import mask_digest, proxy_selection_key


@dataclass(frozen=True)
class ControlResult:
    masks: np.ndarray
    objectives: np.ndarray
    trials: tuple[dict[str, Any], ...]


def best_train_index(
    masks: np.ndarray, metrics: Sequence[Mapping[str, Any]], *, limit: int | None = None
) -> int:
    if len(masks) != len(metrics) or len(masks) < 1:
        raise ValueError("one metric row is required per mask")
    count = len(masks) if limit is None else int(limit)
    if not 1 <= count <= len(masks):
        raise ValueError("selection limit is outside the candidate archive")
    return min(
        range(count), key=lambda i: proxy_selection_key(metrics[i], key=mask_digest(masks[i]))
    )


def ranking_projection(mask: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Keep exactly mask's layer capacities, replacing identities by score rank."""
    if mask.dtype != np.bool_ or mask.ndim != 2 or scores.shape != mask.shape:
        raise ValueError("mask and scores must have matching two-dimensional geometry")
    if not np.isfinite(scores).all():
        raise ValueError("ranking scores must be finite")
    projected = np.zeros_like(mask)
    for layer, row in enumerate(scores):
        order = np.lexsort((np.arange(len(row)), -row))
        projected[layer, order[: int(mask[layer].sum())]] = True
    return projected


def run_control(
    objective_many: Callable[[Sequence[np.ndarray]], Sequence[Sequence[float]]],
    initial_masks: np.ndarray,
    rankings: np.ndarray,
    *,
    config: GlobalBudgetSearchConfig,
    mode: Literal["random_neighborhood", "hill_climb"],
) -> ControlResult:
    """Same starts/budget/mutation kernel as NSGA-II; no population crossover.

    Random search draws an initial parent and 1..generations mutation steps
    independently for every proposal; fitness never changes this distribution.
    Hill climbing uses one chain per start, accepting only a strictly lower
    equal-task-average CE/KL objective. Both output ALL evaluated candidates,
    so their final train-accuracy selection is identical to the EA's rule.
    """
    if mode not in ("random_neighborhood", "hill_climb"):
        raise ValueError("unknown control mode")
    if config.population_size != config.offspring_size:
        raise ValueError("matched controls require one offspring per initial chain")
    if initial_masks.dtype != np.bool_ or initial_masks.shape != (
        config.population_size,
        *rankings.shape,
    ):
        raise ValueError("initial population geometry changed")
    bounds = {
        name: getattr(config, name)
        for name in ("total_retained", "min_retained_per_layer", "max_retained_per_layer")
    }
    masks = [mask.copy() for mask in initial_masks]
    known = {global_budget_mask_sha256(mask, **bounds) for mask in masks}
    if len(known) != len(masks):
        raise ValueError("initial masks must be unique")
    namespace = 0x524E444D if mode == "random_neighborhood" else 0x48494C4C
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, namespace]))

    def evaluate(values: Sequence[np.ndarray]) -> list[tuple[float, float]]:
        result = np.asarray(objective_many(values), dtype=np.float64)
        if result.shape != (len(values), 2) or not np.isfinite(result).all():
            raise ValueError("one finite objective pair is required per candidate")
        return [(float(a), float(b)) for a, b in result]

    objectives = evaluate(masks)
    trials: list[dict[str, Any]] = [
        {
            "candidate_id": i,
            "generation": 0,
            "candidate_sha256": global_budget_mask_sha256(mask, **bounds),
            "parent_id": None,
            "depth": 0,
        }
        for i, mask in enumerate(masks)
    ]
    chains = list(range(config.population_size))
    for generation in range(1, config.generations + 1):
        proposals, contexts = [], []
        for chain in range(config.population_size):
            for _ in range(config.max_duplicate_attempts):
                parent_id = (
                    int(rng.integers(config.population_size))
                    if mode == "random_neighborhood"
                    else chains[chain]
                )
                depth = (
                    int(rng.integers(1, config.generations + 1))
                    if mode == "random_neighborhood"
                    else 1
                )
                child = masks[parent_id].copy()
                for _step in range(depth):
                    child, _, _, _ = _offspring_candidate(
                        child,
                        repairs=(),
                        rankings=rankings,
                        mode="joint",
                        rng=rng,
                        config=config,
                        reproduction_operation={"operator": "non_evolutionary_mutation"},
                    )
                digest = global_budget_mask_sha256(child, **bounds)
                if digest not in known:
                    known.add(digest)
                    proposals.append(child)
                    contexts.append(
                        {
                            "parent_id": parent_id,
                            "chain": chain,
                            "depth": depth,
                            "candidate_sha256": digest,
                        }
                    )
                    break
            else:
                raise RuntimeError("unique control candidate budget cannot be filled")
        values = evaluate(proposals)
        for mask, pair, context in zip(proposals, values, contexts, strict=True):
            index = len(masks)
            if mode == "hill_climb":
                accepted = float(np.mean(pair)) < float(np.mean(objectives[context["parent_id"]]))
                context["accepted"] = accepted
                if accepted:
                    chains[context["chain"]] = index
            masks.append(mask.copy())
            objectives.append(pair)
            trials.append({"candidate_id": index, "generation": generation, **context})
    if len(masks) != config.evaluation_budget:
        raise RuntimeError("control did not use the matched evaluation budget")
    return ControlResult(np.stack(masks), np.asarray(objectives), tuple(trials))

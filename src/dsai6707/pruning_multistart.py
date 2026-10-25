"""Multi-start follow-up controls, independent of any model or test data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any

import numpy as np

from .pruning_global_budget import (
    global_budget_mask_sha256,
    mutate_budget_transfer,
    mutate_identity_swap,
    validate_global_budget_mask,
)
from .pruning_global_budget_search import GlobalBudgetInitialPopulation, GlobalBudgetSearchConfig
from .task_choice_proxy import (
    ChoiceQuestion,
    choice_proxy_distillation_metrics,
    choice_proxy_metrics,
    flatten_choice_requests,
)


def distillation_metrics(
    questions: Sequence[ChoiceQuestion],
    candidate_choice_log_likelihoods: Sequence[float] | np.ndarray,
    teacher_choice_log_likelihoods: Sequence[float] | np.ndarray,
    *,
    cross_entropy_weight: float = 0.75,
    kl_weight: float = 0.25,
    choice_score_normalization: str = "none",
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Optional normalization without altering the hash-locked historical scorer.

    Only CE/KL use transformed scores. Acc and acc_norm always retain native
    lm-eval semantics, with normalization applied exactly once for acc_norm.
    """
    if choice_score_normalization not in ("none", "characters"):
        raise ValueError("choice_score_normalization must be none or characters")
    if not np.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    requests = flatten_choice_requests(questions)
    candidate = np.asarray(candidate_choice_log_likelihoods, dtype=np.float64)
    teacher = np.asarray(teacher_choice_log_likelihoods, dtype=np.float64)
    if candidate.shape != (len(requests),) or teacher.shape != candidate.shape:
        raise ValueError("candidate and teacher choice log-likelihoods have the wrong shape")
    lengths = np.asarray([r.choice_character_length for r in requests], dtype=np.float64)
    if np.any(lengths <= 0):
        raise ValueError("choice character lengths must be positive")
    divisor = lengths if choice_score_normalization == "characters" else np.ones_like(lengths)
    # Separate divisions reproduce the original opt-in scorer's rounding.
    result = choice_proxy_distillation_metrics(
        questions,
        candidate / divisor / temperature,
        teacher / divisor / temperature,
        cross_entropy_weight=cross_entropy_weight,
        kl_weight=kl_weight,
    )
    native = choice_proxy_metrics(questions, candidate)
    for name, row in result["tasks"].items():
        row["objective_choice_cross_entropy"] = row["raw_choice_cross_entropy"]
        for field in ("acc", "acc_norm", "raw_choice_cross_entropy"):
            row[field] = native["tasks"][name][field]
    result.update(
        choice_score_normalization=choice_score_normalization, temperature=float(temperature)
    )
    return result


VARIANTS = (
    {
        "name": "raw_multistart",
        "normalization": "none",
        "temperature": 1.0,
        "ce_weight": 0.75,
        "kl_weight": 0.25,
        "wide": False,
    },
    {
        "name": "norm_multistart",
        "normalization": "characters",
        "temperature": 0.25,
        "ce_weight": 0.75,
        "kl_weight": 0.25,
        "wide": False,
    },
    {
        "name": "norm_label_wide",
        "normalization": "characters",
        "temperature": 0.10,
        "ce_weight": 1.0,
        "kl_weight": 0.0,
        "wide": True,
    },
)


def mask_digest(mask: np.ndarray) -> str:
    values = np.ascontiguousarray(mask, dtype=bool)
    return sha256(str(values.shape).encode() + values.tobytes()).hexdigest()


def search_config(*, layers: int, retained: int, wide: bool = False) -> GlobalBudgetSearchConfig:
    pairs = ((0, 1), (0, 2), (1, 0), (1, 1), (2, 2), (4, 4))
    if wide:
        pairs = ((0, 1), (0, 4), (0, 8), (1, 1), (2, 4), (4, 8))
    return GlobalBudgetSearchConfig(
        population_size=16,
        offspring_size=16,
        generations=7,
        seed=6707,
        total_retained=layers * retained,
        min_retained_per_layer=retained - 4,
        max_retained_per_layer=retained + 4,
        crossover_probability=0.25 if wide else 0.5,
        joint_mutation_pairs=pairs,
        joint_mutation_probabilities=(0.30, 0.15, 0.15, 0.15, 0.15, 0.10),
        max_duplicate_attempts=1024,
    )


def initial_population(
    anchors: Mapping[str, np.ndarray], config: GlobalBudgetSearchConfig
) -> GlobalBudgetInitialPopulation:
    """Include every distinct baseline verbatim, then perturb each in turn."""
    if not anchors or len(anchors) > config.population_size:
        raise ValueError("one to population_size anchors are required")
    bounds = dict(
        total_retained=config.total_retained,
        min_retained_per_layer=config.min_retained_per_layer,
        max_retained_per_layer=config.max_retained_per_layer,
    )
    masks, contexts, known = [], [], set()
    for name, original in anchors.items():
        mask = np.ascontiguousarray(original, dtype=bool)
        validate_global_budget_mask(mask, **bounds)
        key = mask_digest(mask)
        if key not in known:
            known.add(key)
            masks.append(mask.copy())
            contexts.append({"initial_source": name, "exact_baseline": True})
    direct = tuple(zip([row["initial_source"] for row in contexts], masks, strict=True))
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0x4D554C54]))
    attempt = 0
    while len(masks) < config.population_size:
        if attempt > 4096:
            raise RuntimeError("could not construct a unique multi-start population")
        name, parent = direct[attempt % len(direct)]
        child = parent.copy()
        count = (1, 2, 4)[(attempt // len(direct)) % 3]
        for _ in range(count):
            child = mutate_identity_swap(child, rng=rng, **bounds).mask
        if (attempt // len(direct)) % 2:
            child = mutate_budget_transfer(child, rng=rng, **bounds).mask
        attempt += 1
        key = mask_digest(child)
        if key in known:
            continue
        known.add(key)
        masks.append(child)
        contexts.append({"initial_source": name, "exact_baseline": False, "identity_swaps": count})
    digests = [global_budget_mask_sha256(mask, **bounds) for mask in masks]
    digest = sha256(
        b"dsai6707-global-budget-population-v1\0" + "\n".join(digests).encode("ascii")
    ).hexdigest()
    return GlobalBudgetInitialPopulation(np.stack(masks), tuple(contexts), digest)


def proxy_selection_key(metrics: Mapping[str, Any], *, key: str = "") -> tuple[Any, ...]:
    """Use macro acc_norm, not raw CE, for the independent train holdout."""
    tasks = list(metrics["tasks"].values())
    accuracies = [float(row["acc_norm"]) for row in tasks]
    ce = [float(row["objective_choice_cross_entropy"]) for row in tasks]
    return (-float(np.mean(accuracies)), -min(accuracies), float(np.mean(ce)), key)


def shortlist(
    metrics: Sequence[Mapping[str, Any]], *, baseline_ids: Sequence[int], limit: int = 8
) -> tuple[int, ...]:
    """All baselines survive shortlisting; remaining slots use search acc_norm."""
    selected = list(dict.fromkeys(int(value) for value in baseline_ids))
    if any(value < 0 or value >= len(metrics) for value in selected):
        raise ValueError("baseline candidate ID out of range")
    order = sorted(range(len(metrics)), key=lambda i: proxy_selection_key(metrics[i], key=str(i)))
    for index in order:
        if index not in selected and len(selected) < max(limit, len(baseline_ids)):
            selected.append(index)
    return tuple(selected)

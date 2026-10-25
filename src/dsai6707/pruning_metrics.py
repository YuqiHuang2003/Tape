"""Inference-only metrics and deterministic expert selection for pruning.

This module deliberately has no dependency on the fusion implementation.  It
contains the small, auditable pieces shared by pruning baselines and a later
discrete search: exact token-weighted language-model NLL, REAP statistics,
fixed-cardinality expert selection, and Spearman rank correlation.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

__all__ = [
    "ExpertSelectionPlan",
    "ReapStatistics",
    "TokenNLLResult",
    "build_expert_selection_plan",
    "collect_reap_statistics",
    "spearman_rank_correlation",
    "token_weighted_full_model_nll",
]


@dataclass(frozen=True)
class TokenNLLResult:
    """An exact token-weighted causal-language-model NLL measurement."""

    nll: float
    nll_sum: float
    loss_tokens: int
    cost: Mapping[str, float | int | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nll": self.nll,
            "nll_sum": self.nll_sum,
            "loss_tokens": self.loss_tokens,
            "cost": dict(self.cost),
        }


@dataclass(frozen=True)
class ReapStatistics:
    """Per-layer statistics for Router-weighted Expert Activation Pruning.

    ``reap_score[layer, expert]`` is the conditional mean

    ``sum(gate * ||expert_output||_2) / active_count``

    over tokens for which that expert appeared in the original router top-k.
    An expert that is never active receives score zero and remains visible via
    its zero ``active_count``.
    """

    active_count: np.ndarray
    routing_mass: np.ndarray
    weighted_activation_sum: np.ndarray
    reap_score: np.ndarray
    tokens_per_layer: np.ndarray
    top_k: int
    num_experts: int
    cost: Mapping[str, float | int | None]


@dataclass(frozen=True)
class ExpertSelectionPlan:
    """A validated, fixed-cardinality expert-retention plan."""

    scores: np.ndarray
    mask: np.ndarray
    retained_ids: tuple[tuple[int, ...], ...]
    pruned_ids: tuple[tuple[int, ...], ...]
    retained_per_layer: int

    def __post_init__(self) -> None:
        scores = np.asarray(self.scores, dtype=np.float64)
        mask = np.asarray(self.mask, dtype=np.bool_)
        if scores.ndim != 2 or mask.shape != scores.shape:
            raise ValueError("scores and mask must have the same [layers, experts] shape")
        if not np.isfinite(scores).all():
            raise ValueError("expert scores must be finite")
        layers, experts = scores.shape
        if not 1 <= int(self.retained_per_layer) <= experts:
            raise ValueError("retained_per_layer must be in 1..num_experts")
        if len(self.retained_ids) != layers or len(self.pruned_ids) != layers:
            raise ValueError("expert ID lists must contain one entry per layer")
        for layer in range(layers):
            retained = tuple(int(value) for value in self.retained_ids[layer])
            pruned = tuple(int(value) for value in self.pruned_ids[layer])
            if retained != tuple(sorted(retained)) or pruned != tuple(sorted(pruned)):
                raise ValueError("retained and pruned expert IDs must be sorted")
            if len(retained) != self.retained_per_layer:
                raise ValueError("every layer must retain the fixed expert cardinality")
            if set(retained).intersection(pruned) or sorted((*retained, *pruned)) != list(
                range(experts)
            ):
                raise ValueError("retained and pruned expert IDs must partition each layer")
            expected = np.zeros(experts, dtype=np.bool_)
            expected[list(retained)] = True
            if not np.array_equal(mask[layer], expected):
                raise ValueError("mask disagrees with retained expert IDs")

        # Own immutable copies so callers cannot silently change a locked plan.
        scores = scores.copy()
        mask = mask.copy()
        scores.flags.writeable = False
        mask.flags.writeable = False
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "mask", mask)

    @property
    def num_layers(self) -> int:
        return int(self.mask.shape[0])

    @property
    def num_experts(self) -> int:
        return int(self.mask.shape[1])

    def to_dict(self, *, method: str | None = None) -> dict[str, Any]:
        return {
            "schema_version": "expert-selection-plan-v1",
            "method": method,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "retained_per_layer": self.retained_per_layer,
            "tie_break": "descending_score_then_ascending_expert_id",
            "layers": [
                {
                    "layer": layer,
                    "scores": self.scores[layer].tolist(),
                    "retained": list(self.retained_ids[layer]),
                    "pruned": list(self.pruned_ids[layer]),
                }
                for layer in range(self.num_layers)
            ],
        }


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without experiment deps
        raise RuntimeError("PyTorch is required for pruning inference metrics") from exc
    return torch


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as exc:
        raise ValueError("model must expose at least one parameter") from exc


def _validated_token_batches(token_batches: np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
    tokens = np.asarray(token_batches, dtype=np.int64)
    if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] < 2:
        raise ValueError("token_batches must have shape [segments, sequence_length>=2]")
    return tokens


def token_weighted_full_model_nll(
    model: Any,
    token_batches: np.ndarray | Sequence[Sequence[int]],
    *,
    loss_tokens: int,
    micro_batch_size: int = 1,
) -> TokenNLLResult:
    """Evaluate exact next-token NLL with no gradients or padding positions.

    The first ``loss_tokens`` causal positions in row-major block order are
    consumed.  The returned mean is therefore weighted by the exact number of
    token losses, including when the final micro-batch is only partially used.
    """

    torch = _torch()
    tokens = _validated_token_batches(token_batches)
    if int(loss_tokens) != loss_tokens or loss_tokens <= 0:
        raise ValueError("loss_tokens must be a positive integer")
    if int(micro_batch_size) != micro_batch_size or micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be a positive integer")
    loss_tokens = int(loss_tokens)
    micro_batch_size = int(micro_batch_size)
    available = int(tokens.shape[0] * (tokens.shape[1] - 1))
    if available < loss_tokens:
        raise ValueError(f"only {available} loss positions available, need {loss_tokens}")
    if bool(getattr(model, "training", False)):
        raise ValueError("model must be in eval mode for deterministic inference metrics")

    device = _model_device(model)
    wall_started = perf_counter()
    gpu_start = None
    gpu_end = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record()

    nll_sum = 0.0
    consumed = 0
    model_forward_tokens = 0
    with torch.inference_mode():
        for start in range(0, tokens.shape[0], micro_batch_size):
            if consumed >= loss_tokens:
                break
            ids = torch.as_tensor(
                tokens[start : start + micro_batch_size],
                dtype=torch.long,
                device=device,
            )
            logits = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                use_cache=False,
            ).logits
            if logits.ndim != 3 or logits.shape[:2] != ids.shape:
                raise RuntimeError("model returned logits with an unexpected shape")
            losses = torch.nn.functional.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                ids[:, 1:].reshape(-1),
                reduction="none",
            )
            take = min(int(losses.numel()), loss_tokens - consumed)
            nll_sum += float(losses[:take].double().sum().item())
            consumed += take
            model_forward_tokens += int(ids.numel())

    if consumed != loss_tokens:
        raise RuntimeError(f"NLL consumed {consumed} loss positions; expected {loss_tokens}")

    gpu_seconds = None
    peak_device_memory_bytes = None
    if device.type == "cuda":
        assert gpu_start is not None and gpu_end is not None
        gpu_end.record()
        torch.cuda.synchronize(device)
        gpu_seconds = float(gpu_start.elapsed_time(gpu_end) / 1000.0)
        peak_device_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    return TokenNLLResult(
        nll=nll_sum / consumed,
        nll_sum=nll_sum,
        loss_tokens=consumed,
        cost={
            "wall_seconds": perf_counter() - wall_started,
            "gpu_seconds": gpu_seconds,
            "peak_device_memory_bytes": peak_device_memory_bytes,
            "model_forward_tokens": model_forward_tokens,
        },
    )


@dataclass(frozen=True)
class _PendingRouterBatch:
    gates: np.ndarray
    expert_size: np.ndarray
    token_count: int


def _integer_expert_sizes(value: Any, *, num_experts: int) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().to(device="cpu").numpy()
    try:
        sizes_as_float = np.asarray(value, dtype=np.float64)
        sizes = np.asarray(value, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("router returned invalid expert_size") from exc
    if (
        sizes.shape != (num_experts,)
        or not np.isfinite(sizes_as_float).all()
        or not np.array_equal(sizes_as_float, sizes)
        or np.any(sizes < 0)
    ):
        raise RuntimeError("router returned invalid expert_size")
    return sizes


def collect_reap_statistics(
    model: Any,
    token_batches: np.ndarray | Sequence[Sequence[int]],
    *,
    forward_batch_size: int = 1,
) -> ReapStatistics:
    """Collect exact REAP statistics from an unmodified Granite-style MoE.

    Hooks pair each router's expert-sorted ``batch_gates`` and ``expert_size``
    with that layer's output-linear result.  The latter is the expert output
    immediately before Granite multiplies it by the router gate.  Accumulation
    is performed on CPU in explicit row order using float64 scalars.
    """

    torch = _torch()
    tokens = _validated_token_batches(token_batches)
    if int(forward_batch_size) != forward_batch_size or forward_batch_size <= 0:
        raise ValueError("forward_batch_size must be a positive integer")
    forward_batch_size = int(forward_batch_size)
    if bool(getattr(model, "training", False)):
        raise ValueError("model must be in eval mode for deterministic inference metrics")

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("model does not expose model.layers")

    routers: list[Any] = []
    output_linears: list[Any] = []
    layer_top_k: list[int] = []
    layer_num_experts: list[int] = []
    for layer_index, layer in enumerate(layers):
        moe = getattr(layer, "block_sparse_moe", None)
        router = getattr(moe, "router", None)
        output_linear = getattr(moe, "output_linear", None)
        try:
            top_k = int(router.top_k)
            num_experts = int(router.num_experts)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"layer {layer_index} has no compatible Granite router") from exc
        if top_k <= 0 or num_experts < top_k:
            raise ValueError(f"layer {layer_index} has invalid expert/top-k counts")
        if output_linear is None or not hasattr(output_linear, "register_forward_hook"):
            raise ValueError(f"layer {layer_index} has no hookable output_linear")
        routers.append(router)
        output_linears.append(output_linear)
        layer_top_k.append(top_k)
        layer_num_experts.append(num_experts)

    if len(set(layer_top_k)) != 1 or len(set(layer_num_experts)) != 1:
        raise ValueError("all layers must use the same expert count and top-k")
    top_k = layer_top_k[0]
    num_experts = layer_num_experts[0]
    num_layers = len(layers)
    active_count = np.zeros((num_layers, num_experts), dtype=np.int64)
    routing_mass = np.zeros((num_layers, num_experts), dtype=np.float64)
    weighted_activation_sum = np.zeros((num_layers, num_experts), dtype=np.float64)
    tokens_per_layer = np.zeros(num_layers, dtype=np.int64)
    pending: list[deque[_PendingRouterBatch]] = [deque() for _ in range(num_layers)]
    handles: list[Any] = []

    def make_router_hook(layer_index: int) -> Any:
        def hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
            del module, args
            if not isinstance(output, tuple | list) or len(output) < 5:
                raise RuntimeError("router output does not expose REAP routing evidence")
            batch_gates, expert_size, router_logits = output[2], output[3], output[4]
            if not hasattr(batch_gates, "detach") or not hasattr(router_logits, "shape"):
                raise RuntimeError("router output contains invalid gate/logit tensors")
            gates = (
                batch_gates.detach()
                .reshape(-1)
                .to(device="cpu", dtype=torch.float64)
                .numpy()
                .copy()
            )
            if not np.isfinite(gates).all() or np.any(gates < 0.0):
                raise RuntimeError("router returned non-finite or negative gates")
            sizes = _integer_expert_sizes(expert_size, num_experts=num_experts)
            if len(router_logits.shape) != 2 or int(router_logits.shape[1]) != num_experts:
                raise RuntimeError("router logits have an unexpected shape")
            token_count = int(router_logits.shape[0])
            expected_rows = token_count * top_k
            if gates.shape != (expected_rows,) or int(sizes.sum()) != expected_rows:
                raise RuntimeError("router assignments do not equal tokens * top_k")
            pending[layer_index].append(
                _PendingRouterBatch(
                    gates=gates,
                    expert_size=sizes,
                    token_count=token_count,
                )
            )
            tokens_per_layer[layer_index] += token_count

        return hook

    def make_output_hook(layer_index: int) -> Any:
        def hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
            del module, args
            if not pending[layer_index]:
                raise RuntimeError("output_linear ran without a matching router batch")
            evidence = pending[layer_index].popleft()
            if not hasattr(output, "detach") or output.ndim < 2:
                raise RuntimeError("output_linear returned an invalid expert output")
            rows = int(output.shape[0])
            expected_rows = evidence.token_count * top_k
            if rows != expected_rows:
                raise RuntimeError("expert outputs do not equal tokens * top_k")
            norms = (
                output.detach()
                .reshape(rows, -1)
                .float()
                .norm(p=2, dim=1)
                .to(device="cpu", dtype=torch.float64)
                .numpy()
            )
            if not np.isfinite(norms).all():
                raise RuntimeError("expert output norms are non-finite")
            offset = 0
            for expert_id, expert_rows in enumerate(evidence.expert_size.tolist()):
                for _ in range(expert_rows):
                    gate = float(evidence.gates[offset])
                    norm = float(norms[offset])
                    active_count[layer_index, expert_id] += 1
                    routing_mass[layer_index, expert_id] += gate
                    weighted_activation_sum[layer_index, expert_id] += gate * norm
                    offset += 1
            if offset != expected_rows:
                raise RuntimeError("expert_size did not consume every routed expert output")

        return hook

    for layer_index, (router, output_linear) in enumerate(
        zip(routers, output_linears, strict=False)
    ):
        handles.append(router.register_forward_hook(make_router_hook(layer_index)))
        handles.append(output_linear.register_forward_hook(make_output_hook(layer_index)))

    device = _model_device(model)
    wall_started = perf_counter()
    gpu_start = None
    gpu_end = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record()
    try:
        with torch.inference_mode():
            for start in range(0, tokens.shape[0], forward_batch_size):
                ids = torch.as_tensor(
                    tokens[start : start + forward_batch_size],
                    dtype=torch.long,
                    device=device,
                )
                model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    use_cache=False,
                )
    finally:
        for handle in handles:
            handle.remove()

    if any(queue for queue in pending):
        raise RuntimeError("one or more router batches had no matching output_linear result")
    expected_tokens = int(tokens.size)
    if not np.all(tokens_per_layer == expected_tokens):
        raise RuntimeError("not every layer routed every input token")
    expected_assignments = expected_tokens * top_k
    assignment_totals = active_count.sum(axis=1)
    if not np.all(assignment_totals == expected_assignments):
        raise RuntimeError("per-layer active counts do not equal tokens * top_k")

    reap_score = np.zeros_like(weighted_activation_sum)
    np.divide(
        weighted_activation_sum,
        active_count,
        out=reap_score,
        where=active_count != 0,
    )
    gpu_seconds = None
    peak_device_memory_bytes = None
    if device.type == "cuda":
        assert gpu_start is not None and gpu_end is not None
        gpu_end.record()
        torch.cuda.synchronize(device)
        gpu_seconds = float(gpu_start.elapsed_time(gpu_end) / 1000.0)
        peak_device_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    return ReapStatistics(
        active_count=active_count,
        routing_mass=routing_mass,
        weighted_activation_sum=weighted_activation_sum,
        reap_score=reap_score,
        tokens_per_layer=tokens_per_layer,
        top_k=top_k,
        num_experts=num_experts,
        cost={
            "wall_seconds": perf_counter() - wall_started,
            "gpu_seconds": gpu_seconds,
            "peak_device_memory_bytes": peak_device_memory_bytes,
            "model_forward_tokens": expected_tokens,
            "expert_output_rows": int(num_layers * expected_assignments),
        },
    )


def build_expert_selection_plan(
    scores: np.ndarray | Sequence[Sequence[float]],
    *,
    retained_per_layer: int,
) -> ExpertSelectionPlan:
    """Retain the highest-scoring experts with deterministic expert-ID ties."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("scores must have non-empty [layers, experts] shape")
    if not np.isfinite(values).all():
        raise ValueError("expert scores must be finite")
    experts = int(values.shape[1])
    if int(retained_per_layer) != retained_per_layer or not 1 <= retained_per_layer <= experts:
        raise ValueError("retained_per_layer must be an integer in 1..num_experts")
    retained_per_layer = int(retained_per_layer)
    expert_ids = np.arange(experts, dtype=np.int64)
    mask = np.zeros(values.shape, dtype=np.bool_)
    retained: list[tuple[int, ...]] = []
    pruned: list[tuple[int, ...]] = []
    for layer_scores in values:
        ranking = np.lexsort((expert_ids, -layer_scores))
        chosen = tuple(sorted(int(value) for value in ranking[:retained_per_layer]))
        removed = tuple(sorted(set(range(experts)).difference(chosen)))
        mask[len(retained), list(chosen)] = True
        retained.append(chosen)
        pruned.append(removed)
    return ExpertSelectionPlan(
        scores=values,
        mask=mask,
        retained_ids=tuple(retained),
        pruned_ids=tuple(pruned),
        retained_per_layer=retained_per_layer,
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        # Ranks are one-based; the tied interval is [start + 1, stop].
        average = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = average
        start = stop
    return ranks


def spearman_rank_correlation(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> float:
    """Return tie-aware Spearman correlation without SciPy.

    Constant inputs have undefined rank correlation and are rejected rather
    than silently producing ``nan`` in a persisted experiment artifact.
    """

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError("Spearman inputs must be equal-length 1D arrays of size at least two")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Spearman inputs must be finite")
    x_rank = _average_ranks(x)
    y_rank = _average_ranks(y)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = float(
        np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    )
    if denominator == 0.0:
        raise ValueError("Spearman correlation is undefined for a constant input")
    value = float(np.dot(x_centered, y_centered) / denominator)
    return float(np.clip(value, -1.0, 1.0))

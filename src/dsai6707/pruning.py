"""Inference-only expert selection for Granite MoE search.

The controller in this module deliberately keeps the original Granite expert
and router parameters in place.  It masks deleted router logits immediately
after the router's linear projection, so Transformers' own top-k selection and
softmax remain responsible for routing.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Granite expert pruning") from exc
    return torch


def _as_index(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class ExpertSelectionPlan:
    """A canonical, fixed retained-expert set for every MoE layer."""

    layers: tuple[tuple[int, ...], ...]
    num_experts: int
    schema_version: str = "expert-selection-v1"

    def __post_init__(self) -> None:
        num_experts = _as_index(self.num_experts, name="num_experts")
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        if self.schema_version != "expert-selection-v1":
            raise ValueError("unsupported expert-selection schema_version")
        if not self.layers:
            raise ValueError("an expert-selection plan must contain at least one layer")

        normalized_layers: list[tuple[int, ...]] = []
        for layer_index, retained in enumerate(self.layers):
            expert_ids = tuple(
                _as_index(expert, name=f"layer {layer_index} expert id")
                for expert in retained
            )
            if not expert_ids:
                raise ValueError(f"layer {layer_index} must retain at least one expert")
            if len(set(expert_ids)) != len(expert_ids):
                raise ValueError(f"layer {layer_index} contains duplicate expert ids")
            if any(expert < 0 or expert >= num_experts for expert in expert_ids):
                raise ValueError(
                    f"layer {layer_index} expert ids must be in [0, {num_experts})"
                )
            normalized_layers.append(tuple(sorted(expert_ids)))

        object.__setattr__(self, "num_experts", num_experts)
        object.__setattr__(self, "layers", tuple(normalized_layers))

    @classmethod
    def from_retained(
        cls,
        retained_by_layer: Sequence[Sequence[int]],
        *,
        num_experts: int,
    ) -> ExpertSelectionPlan:
        """Build a plan from arbitrary per-layer expert-id sequences."""

        return cls(
            layers=tuple(tuple(retained) for retained in retained_by_layer),
            num_experts=num_experts,
        )

    @classmethod
    def all_experts(cls, *, num_layers: int, num_experts: int) -> ExpertSelectionPlan:
        """Build the strict identity plan that retains every expert."""

        layer_count = _as_index(num_layers, name="num_layers")
        expert_count = _as_index(num_experts, name="num_experts")
        if layer_count < 1:
            raise ValueError("num_layers must be positive")
        if expert_count < 1:
            raise ValueError("num_experts must be positive")
        retained = tuple(range(expert_count))
        return cls(
            layers=tuple(retained for _ in range(layer_count)),
            num_experts=expert_count,
        )

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def retained_counts(self) -> tuple[int, ...]:
        return tuple(len(retained) for retained in self.layers)

    @property
    def is_identity(self) -> bool:
        return all(len(retained) == self.num_experts for retained in self.layers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "num_experts": self.num_experts,
            "layers": [
                {"layer": layer, "retained_experts": list(retained)}
                for layer, retained in enumerate(self.layers)
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExpertSelectionPlan:
        if payload.get("schema_version") != "expert-selection-v1":
            raise ValueError("unsupported or missing expert-selection schema_version")
        raw_layers = payload.get("layers")
        if not isinstance(raw_layers, Sequence) or isinstance(raw_layers, str | bytes):
            raise ValueError("expert-selection layers must be a sequence")
        selections: list[Sequence[int]] = []
        for expected_layer, item in enumerate(raw_layers):
            if not isinstance(item, Mapping) or item.get("layer") != expected_layer:
                raise ValueError("expert-selection layers must be ordered and contiguous")
            retained = item.get("retained_experts")
            if not isinstance(retained, Sequence) or isinstance(retained, str | bytes):
                raise ValueError(f"layer {expected_layer} retained_experts must be a sequence")
            selections.append(retained)
        return cls.from_retained(
            selections,
            num_experts=_as_index(payload.get("num_experts"), name="num_experts"),
        )


@dataclass(frozen=True)
class _RouterSpec:
    layer: Any
    num_experts: int
    top_k: int


@dataclass
class _MaskState:
    mask: Any
    passthrough: bool = True


def _router_specs(model: Any) -> tuple[_RouterSpec, ...]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("model does not expose model.layers")
    specs: list[_RouterSpec] = []
    for layer_index, decoder_layer in enumerate(layers):
        moe = getattr(decoder_layer, "block_sparse_moe", None)
        router = getattr(moe, "router", None)
        linear = getattr(router, "layer", None)
        weight = getattr(linear, "weight", None)
        if linear is None or weight is None or not hasattr(linear, "register_forward_hook"):
            raise ValueError(
                f"layer {layer_index} does not expose the Granite router linear layer"
            )
        num_experts = _as_index(
            getattr(router, "num_experts", weight.shape[0]),
            name=f"layer {layer_index} router.num_experts",
        )
        top_k = _as_index(
            getattr(router, "top_k", None),
            name=f"layer {layer_index} router.top_k",
        )
        if tuple(weight.shape[:1]) != (num_experts,):
            raise ValueError(f"layer {layer_index} router weight shape is inconsistent")
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"layer {layer_index} has an invalid router top_k")
        specs.append(_RouterSpec(layer=linear, num_experts=num_experts, top_k=top_k))
    if not specs:
        raise ValueError("model must contain at least one Granite MoE layer")
    return tuple(specs)


def _make_mask_hook(state: _MaskState) -> Any:
    def mask_router_logits(_module: Any, _inputs: Any, output: Any) -> Any:
        # Returning None is PyTorch's exact no-op hook path.  The 32/32 plan
        # therefore executes the unmodified Transformers router bit-for-bit.
        if state.passthrough:
            return None
        if output.shape[-1] != state.mask.numel():
            raise RuntimeError("router output width changed while pruning was active")
        if output.device != state.mask.device:
            raise RuntimeError(
                "model device changed after installing the pruning controller; "
                "install the controller after final device placement"
            )
        return output.masked_fill(~state.mask, float("-inf"))

    return mask_router_logits


class GranitePruningController:
    """Apply retained-expert masks without copying or changing model parameters.

    This search-time controller is inference-only and is intentionally not
    serialized.  Install it after the model has been moved to its final device.
    Candidate changes are not thread-safe with concurrent model forwards.
    """

    _TOKEN_ATTRIBUTE = "_dsai6707_pruning_hook_token"

    def __init__(self, model: Any, plan: ExpertSelectionPlan) -> None:
        if bool(getattr(model, "training", False)):
            raise ValueError("expert pruning is inference-only; call model.eval() first")
        self.model = model
        self.plan = plan
        self._specs = _router_specs(model)
        self._states: list[_MaskState] = []
        self._handles: list[Any] = []
        self._token = object()
        self._closed = False
        self._validate_plan(plan)

        torch = _torch()
        try:
            for layer_index, spec in enumerate(self._specs):
                if getattr(spec.layer, self._TOKEN_ATTRIBUTE, None) is not None:
                    raise RuntimeError(
                        f"layer {layer_index} already has an expert-pruning controller"
                    )
                state = _MaskState(
                    mask=torch.ones(
                        spec.num_experts,
                        dtype=torch.bool,
                        device=spec.layer.weight.device,
                    )
                )
                setattr(spec.layer, self._TOKEN_ATTRIBUTE, self._token)
                handle = spec.layer.register_forward_hook(_make_mask_hook(state))
                self._states.append(state)
                self._handles.append(handle)
            self.apply(plan)
        except Exception:
            self.close()
            raise

    def _validate_plan(self, plan: ExpertSelectionPlan) -> None:
        if not isinstance(plan, ExpertSelectionPlan):
            raise TypeError("plan must be an ExpertSelectionPlan")
        if plan.num_layers != len(self._specs):
            raise ValueError(
                "expert-selection layer count does not match the Granite checkpoint"
            )
        expected_counts = {spec.num_experts for spec in self._specs}
        if expected_counts != {plan.num_experts}:
            raise ValueError(
                "expert-selection expert count does not match the Granite checkpoint"
            )
        for layer_index, (spec, retained) in enumerate(
            zip(self._specs, plan.layers, strict=True)
        ):
            if len(retained) < spec.top_k:
                raise ValueError(
                    f"layer {layer_index} retains {len(retained)} experts but top_k is "
                    f"{spec.top_k}"
                )

    def apply(self, plan: ExpertSelectionPlan) -> None:
        """Atomically select a new candidate for subsequent forward passes."""

        if self._closed:
            raise RuntimeError("the pruning controller is closed")
        self._validate_plan(plan)
        torch = _torch()
        prepared_masks = []
        for state, retained in zip(self._states, plan.layers, strict=True):
            mask = torch.zeros_like(state.mask)
            mask[list(retained)] = True
            prepared_masks.append(mask)
        with torch.inference_mode():
            for spec, state, mask, retained in zip(
                self._specs,
                self._states,
                prepared_masks,
                plan.layers,
                strict=True,
            ):
                state.mask.copy_(mask)
                state.passthrough = len(retained) == spec.num_experts
        self.plan = plan

    def close(self) -> None:
        """Remove every router hook and restore ordinary unpruned forwards."""

        if self._closed:
            return
        for handle in self._handles:
            handle.remove()
        for spec in self._specs:
            if getattr(spec.layer, self._TOKEN_ATTRIBUTE, None) is self._token:
                delattr(spec.layer, self._TOKEN_ATTRIBUTE)
        self._handles.clear()
        self._closed = True

    def __enter__(self) -> GranitePruningController:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


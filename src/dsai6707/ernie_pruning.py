"""Runtime expert selection for inference-only ERNIE MoE search.

The controller keeps every router and expert parameter intact.  A temporary
gate forward computes only the retained weight rows before ERNIE applies its
configured gate activation, and a temporary ``forward_experts`` method executes
the same retained modules in ascending original-index order.  This reproduces
the tensor shapes and operation order of a physically sliced checkpoint without
building a checkpoint for every search candidate.
"""

from __future__ import annotations

import operator
import types
from dataclasses import dataclass
from typing import Any

import numpy as np


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for ERNIE expert masking") from exc
    return torch


def _as_index(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def ernie_moe_layers(model: Any) -> tuple[Any, ...]:
    """Return routed ERNIE MLPs in decoder-layer order.

    ERNIE-4.5 has an initial dense MLP followed by routed MoE MLPs, so dense
    layers are intentionally omitted from the returned tuple.
    """

    decoder_layers = getattr(getattr(model, "model", None), "layers", None)
    if decoder_layers is None:
        raise ValueError("model does not expose model.layers")

    moe_layers: list[Any] = []
    for decoder_index, decoder_layer in enumerate(decoder_layers):
        mlp = getattr(decoder_layer, "mlp", None)
        gate = getattr(mlp, "gate", None)
        experts = getattr(mlp, "experts", None)
        if gate is None and experts is None:
            continue
        if gate is None or experts is None:
            raise ValueError(
                f"decoder layer {decoder_index} exposes an incomplete ERNIE MoE MLP"
            )
        weight = getattr(gate, "weight", None)
        if weight is None or not hasattr(gate, "register_forward_hook"):
            raise ValueError(
                f"decoder layer {decoder_index} does not expose the ERNIE gate linear layer"
            )
        moe_layers.append(mlp)

    if not moe_layers:
        raise ValueError("model must contain at least one ERNIE MoE layer")
    return tuple(moe_layers)


@dataclass(frozen=True)
class _GateSpec:
    mlp: Any
    gate: Any
    num_experts: int
    top_k: int
    original_gate_forward: Any
    original_forward_experts: Any


@dataclass
class _MaskState:
    gate_weight: Any
    gate_bias: Any
    expert_indices: tuple[int, ...]
    passthrough: bool = True


def _gate_specs(model: Any) -> tuple[_GateSpec, ...]:
    specs: list[_GateSpec] = []
    for moe_index, mlp in enumerate(ernie_moe_layers(model)):
        gate = mlp.gate
        weight = gate.weight
        if len(weight.shape) != 2:
            raise ValueError(f"MoE layer {moe_index} gate weight must be two-dimensional")
        num_experts = _as_index(
            getattr(mlp, "num_local_experts", weight.shape[0]),
            name=f"MoE layer {moe_index} num_local_experts",
        )
        top_k = _as_index(getattr(mlp, "k", None), name=f"MoE layer {moe_index} k")
        if weight.shape[0] != num_experts or len(mlp.experts) != num_experts:
            raise ValueError(f"MoE layer {moe_index} expert count is inconsistent")
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"MoE layer {moe_index} has an invalid top-k value")
        original_forward_experts = getattr(mlp, "forward_experts", None)
        if not callable(original_forward_experts):
            raise ValueError(
                f"MoE layer {moe_index} does not expose a callable forward_experts"
            )
        specs.append(
            _GateSpec(
                mlp=mlp,
                gate=gate,
                num_experts=num_experts,
                top_k=top_k,
                original_gate_forward=gate.forward,
                original_forward_experts=original_forward_experts,
            )
        )

    expert_counts = {spec.num_experts for spec in specs}
    if len(expert_counts) != 1:
        raise ValueError("all ERNIE MoE layers must have the same expert count")
    return tuple(specs)


def _validate_mask(mask: np.ndarray, specs: tuple[_GateSpec, ...]) -> tuple[int, ...]:
    if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
        raise ValueError("mask must be a NumPy boolean array")
    expected_shape = (len(specs), specs[0].num_experts)
    if mask.ndim != 2 or mask.shape != expected_shape:
        raise ValueError(f"mask must have shape {expected_shape}")

    retained_counts = tuple(int(value) for value in np.count_nonzero(mask, axis=1))
    for layer_index, (spec, retained_count) in enumerate(
        zip(specs, retained_counts, strict=True)
    ):
        if retained_count < spec.top_k:
            raise ValueError(
                f"MoE layer {layer_index} retains {retained_count} experts but top-k is "
                f"{spec.top_k}"
            )
    return retained_counts


def _make_gate_forward(spec: _GateSpec, state: _MaskState) -> Any:
    def gate_forward(_module: Any, inputs: Any) -> Any:
        if state.passthrough:
            return spec.original_gate_forward(inputs)
        return _torch().nn.functional.linear(inputs, state.gate_weight, state.gate_bias)

    return types.MethodType(gate_forward, spec.gate)


def _make_forward_experts(spec: _GateSpec, state: _MaskState) -> Any:
    """Build the compact expert path used by a physically sliced checkpoint."""

    def forward_experts(module: Any, dispatched_input: Any) -> Any:
        if state.passthrough:
            return spec.original_forward_experts(dispatched_input)
        retained_count = len(state.expert_indices)
        dispatched_input = dispatched_input.reshape(
            1,
            retained_count,
            -1,
            dispatched_input.shape[-1],
        )
        chunks = dispatched_input.permute(1, 0, 2, 3).contiguous().unbind(0)
        if len(chunks) != retained_count:
            raise RuntimeError("compact ERNIE dispatch produced the wrong expert count")
        outputs = [
            module.experts[expert_index](chunk)
            for expert_index, chunk in zip(state.expert_indices, chunks, strict=True)
        ]
        return _torch().stack(outputs, dim=1)

    return types.MethodType(forward_experts, spec.mlp)


class ErnieExpertMaskController:
    """Apply ERNIE expert masks without changing parameters.

    Install the controller after final model device placement.  Candidate
    changes are inference-only and are not thread-safe with concurrent forwards.
    Retained cardinality may change both between calls and across layers within
    one candidate, provided every layer keeps at least its router top-k experts.
    """

    _TOKEN_ATTRIBUTE = "_dsai6707_ernie_expert_mask_hook_token"

    def __init__(self, model: Any, identity_mask: np.ndarray) -> None:
        if bool(getattr(model, "training", False)):
            raise ValueError("ERNIE expert masking is inference-only; call model.eval() first")
        self.model = model
        self._specs = _gate_specs(model)
        self._states: list[_MaskState] = []
        self._patched_gates: list[tuple[Any, bool, Any]] = []
        self._patched_mlps: list[tuple[Any, bool, Any]] = []
        self._token = object()
        self._closed = False
        self._mask = np.empty((0, 0), dtype=np.bool_)
        _validate_mask(identity_mask, self._specs)

        try:
            for layer_index, spec in enumerate(self._specs):
                if getattr(spec.gate, self._TOKEN_ATTRIBUTE, None) is not None:
                    raise RuntimeError(
                        f"MoE layer {layer_index} already has an ERNIE mask controller"
                    )
                state = _MaskState(
                    gate_weight=spec.gate.weight,
                    gate_bias=spec.gate.bias,
                    expert_indices=tuple(range(spec.num_experts)),
                )
                setattr(spec.gate, self._TOKEN_ATTRIBUTE, self._token)
                had_instance_gate_forward = "forward" in spec.gate.__dict__
                previous_instance_gate_forward = spec.gate.__dict__.get("forward")
                spec.gate.forward = _make_gate_forward(spec, state)
                had_instance_method = "forward_experts" in spec.mlp.__dict__
                previous_instance_method = spec.mlp.__dict__.get("forward_experts")
                spec.mlp.forward_experts = _make_forward_experts(spec, state)
                self._states.append(state)
                self._patched_gates.append(
                    (spec.gate, had_instance_gate_forward, previous_instance_gate_forward)
                )
                self._patched_mlps.append(
                    (spec.mlp, had_instance_method, previous_instance_method)
                )
            self.apply(identity_mask)
        except Exception:
            self.close()
            raise

    @property
    def mask(self) -> np.ndarray:
        """Return a defensive copy of the active retained-expert mask."""

        return self._mask.copy()

    def apply(self, mask: np.ndarray) -> None:
        """Atomically select a mask for subsequent model forwards."""

        if self._closed:
            raise RuntimeError("the ERNIE mask controller is closed")
        retained_counts = _validate_mask(mask, self._specs)
        # Own the NumPy storage so a caller cannot race a candidate mutation
        # with mask preparation or alter the controller's reported state.
        contiguous_mask = np.array(mask, dtype=np.bool_, order="C", copy=True)
        torch = _torch()
        retained_indices = [
            tuple(int(index) for index in np.flatnonzero(layer_mask))
            for layer_mask in contiguous_mask
        ]
        prepared_indices = [
            torch.as_tensor(indices, dtype=torch.long, device=spec.gate.weight.device)
            for spec, indices in zip(self._specs, retained_indices, strict=True)
        ]
        with torch.inference_mode():
            for spec, state, prepared, indices, retained_count in zip(
                self._specs,
                self._states,
                prepared_indices,
                retained_indices,
                retained_counts,
                strict=True,
            ):
                state.gate_weight = spec.gate.weight.index_select(0, prepared).contiguous()
                state.gate_bias = (
                    None
                    if spec.gate.bias is None
                    else spec.gate.bias.index_select(0, prepared).contiguous()
                )
                state.expert_indices = indices
                state.passthrough = retained_count == spec.num_experts
        self._mask = contiguous_mask.copy()

    def close(self) -> None:
        """Remove all gate hooks and restore ordinary unmasked forwards."""

        if self._closed:
            return
        for gate, had_instance_method, previous_instance_method in self._patched_gates:
            if had_instance_method:
                gate.forward = previous_instance_method
            elif "forward" in gate.__dict__:
                delattr(gate, "forward")
        for mlp, had_instance_method, previous_instance_method in self._patched_mlps:
            if had_instance_method:
                mlp.forward_experts = previous_instance_method
            elif "forward_experts" in mlp.__dict__:
                delattr(mlp, "forward_experts")
        for spec in self._specs:
            if getattr(spec.gate, self._TOKEN_ATTRIBUTE, None) is self._token:
                delattr(spec.gate, self._TOKEN_ATTRIBUTE)
        self._patched_gates.clear()
        self._patched_mlps.clear()
        self._closed = True

    def __enter__(self) -> ErnieExpertMaskController:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()


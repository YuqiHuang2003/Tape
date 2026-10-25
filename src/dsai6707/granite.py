"""Transformers 4.46.3 adapter for Granite-3.1 MoE expert fusion.

This module intentionally imports heavyweight dependencies lazily. The core
grouping/search tests therefore run on CPU-only machines without PyTorch.
"""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .grouping import GroupingPlan, LayerGrouping
from .io import (
    ensure_cublas_workspace_config,
    environment_snapshot,
    read_json,
    sha256_file,
    write_json,
)

_CHECKPOINT_REQUIRED_FILES = frozenset(
    {
        "model.safetensors",
        "config.json",
        "grouping.json",
        "coefficients.json",
        "compression.json",
    }
)
_CHECKPOINT_ALLOWED_UNDECLARED_FILES = frozenset(
    {"manifest.json", "stage_manifest.json", "roundtrip_validation.json"}
)
_CHECKPOINT_ATTENTION_IMPLEMENTATIONS = frozenset(
    {"eager", "sdpa", "flash_attention_2"}
)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the Granite experiment") from exc
    return torch


def _transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError("Transformers 4.46.3 is required for the Granite experiment") from exc
    if transformers.__version__ != "4.46.3":
        raise RuntimeError(
            f"unsupported transformers version {transformers.__version__}; expected exactly 4.46.3"
        )
    return transformers


def _configure_torch_determinism(torch: Any) -> None:
    """Apply the shared deterministic inference settings for every loader."""

    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def _cached_weight_checkpoint_stats(model_id: str, revision: str) -> dict[str, Any]:
    """Resolve the exact cached base weight files and their on-disk byte sizes."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to inspect checkpoint bytes") from exc

    filenames: list[str] = []
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        try:
            index_path = hf_hub_download(
                model_id,
                filename=index_name,
                revision=revision,
                local_files_only=True,
            )
        except Exception:
            continue
        index = read_json(index_path)
        filenames = sorted(set(index.get("weight_map", {}).values()))
        if filenames:
            break
    if not filenames:
        for single_name in ("model.safetensors", "pytorch_model.bin"):
            try:
                hf_hub_download(
                    model_id,
                    filename=single_name,
                    revision=revision,
                    local_files_only=True,
                )
            except Exception:
                continue
            filenames = [single_name]
            break
    if not filenames:
        raise RuntimeError("could not resolve the cached base checkpoint weight files")

    files = []
    for filename in filenames:
        path = Path(
            hf_hub_download(
                model_id,
                filename=filename,
                revision=revision,
                local_files_only=True,
            )
        )
        files.append({"name": filename, "bytes": path.stat().st_size})
    return {
        "revision": revision,
        "files": files,
        "total_bytes": sum(int(item["bytes"]) for item in files),
    }


def load_granite_model(model_config: Mapping[str, Any]) -> tuple[Any, Any]:
    """Load the pinned tokenizer/model and validate the expected MoE topology."""

    torch = _torch()
    model_id = str(model_config["id"])
    revision = str(model_config["revision"])
    tokenizer_revision = str(model_config.get("tokenizer_revision", revision))
    device = str(model_config.get("device", "cuda"))
    if device.startswith("cuda"):
        ensure_cublas_workspace_config(
            cuda_initialized=bool(torch.cuda.is_initialized())
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available")
    _configure_torch_determinism(torch)
    transformers = _transformers()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        revision=tokenizer_revision,
        use_fast=True,
    )
    tokenizer._dsai6707_tokenizer_id = model_id
    tokenizer._dsai6707_tokenizer_revision = tokenizer_revision
    dtype_name = str(model_config.get("dtype", "bfloat16"))
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype_name)
    if dtype is None:
        raise ValueError(f"unsupported model dtype: {dtype_name}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    inspect_granite_topology(model)
    model._dsai6707_model_id = model_id
    model._dsai6707_model_revision = revision
    model._dsai6707_base_checkpoint = _cached_weight_checkpoint_stats(model_id, revision)
    return model, tokenizer


def inspect_granite_topology(model: Any) -> dict[str, Any]:
    """Fail early if a checkpoint/library change invalidates weight assumptions."""

    config = model.config
    if getattr(config, "model_type", None) != "granitemoe":
        raise ValueError(f"expected model_type='granitemoe', got {config.model_type!r}")
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or len(layers) != int(config.num_hidden_layers):
        raise ValueError("model.model.layers does not match num_hidden_layers")

    expected = {
        "router": (int(config.num_local_experts), int(config.hidden_size)),
        "gate_up": (
            int(config.num_local_experts),
            2 * int(config.intermediate_size),
            int(config.hidden_size),
        ),
        "down": (
            int(config.num_local_experts),
            int(config.hidden_size),
            int(config.intermediate_size),
        ),
    }
    for layer_index, decoder_layer in enumerate(layers):
        moe = getattr(decoder_layer, "block_sparse_moe", None)
        try:
            actual = {
                "router": tuple(moe.router.layer.weight.shape),
                "gate_up": tuple(moe.input_linear.weight.shape),
                "down": tuple(moe.output_linear.weight.shape),
            }
        except AttributeError as exc:
            raise ValueError(
                f"layer {layer_index} does not expose the Transformers 4.46.3 Granite API"
            ) from exc
        if actual != expected:
            raise ValueError(
                f"layer {layer_index} weight layout mismatch: expected {expected}, got {actual}"
            )
    return {
        "model_type": config.model_type,
        "num_layers": int(config.num_hidden_layers),
        "num_experts": int(config.num_local_experts),
        "top_k": int(config.num_experts_per_tok),
        "hidden_size": int(config.hidden_size),
        "intermediate_size": int(config.intermediate_size),
        "weight_shapes": expected,
    }


@dataclass
class ExpertStatistics:
    routing_counts: np.ndarray
    routing_mass: np.ndarray
    output_similarity: np.ndarray
    sampled_tokens_per_layer: np.ndarray
    runtime: dict[str, Any]

    def save(self, directory: str | Path, metadata: Mapping[str, Any]) -> None:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination / "expert_statistics.npz",
            routing_counts=self.routing_counts,
            routing_mass=self.routing_mass,
            output_similarity=self.output_similarity,
            sampled_tokens_per_layer=self.sampled_tokens_per_layer,
        )
        write_json(
            destination / "metadata.json",
            {
                "schema_version": "expert-statistics-v1",
                "similarity": "cosine_of_concatenated_expert_outputs",
                "routing_count": "original_router_topk_selection_count",
                "routing_mass": "original_router_post_topk_softmax_weight_sum",
                "runtime": self.runtime,
                **dict(metadata),
            },
        )


def collect_expert_statistics(
    model: Any,
    token_batches: np.ndarray,
    *,
    signature_tokens_per_layer: int,
    forward_batch_size: int = 1,
) -> ExpertStatistics:
    """Collect router usage and exact output-cosine statistics.

    Layer inputs are captured from a fixed token sample. Every source expert is
    then evaluated on the same captured inputs. Pairwise Gram matrices are
    accumulated in chunks, avoiding storage of a huge flattened signature.
    """

    torch = _torch()
    topology = inspect_granite_topology(model)
    if token_batches.ndim != 2:
        raise ValueError("token_batches must have shape [segments, sequence_length]")
    if signature_tokens_per_layer <= 0 or forward_batch_size <= 0:
        raise ValueError("sample and batch sizes must be positive")

    layers = model.model.layers
    device = next(model.parameters()).device
    wall_started = perf_counter()
    gpu_started = None
    gpu_finished = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        gpu_started = torch.cuda.Event(enable_timing=True)
        gpu_finished = torch.cuda.Event(enable_timing=True)
        gpu_started.record()
    num_layers = topology["num_layers"]
    num_experts = topology["num_experts"]
    routing_counts = np.zeros((num_layers, num_experts), dtype=np.int64)
    routing_mass = np.zeros((num_layers, num_experts), dtype=np.float64)
    captured: list[list[Any]] = [[] for _ in range(num_layers)]
    captured_count = np.zeros(num_layers, dtype=np.int64)
    hooks = []

    def make_hook(layer_index: int) -> Any:
        def hook(module: Any, args: tuple[Any, ...]) -> None:
            hidden = args[0].detach().reshape(-1, topology["hidden_size"])
            with torch.inference_mode():
                logits = module.router.layer(hidden).float()
                top_logits, top_indices = logits.topk(topology["top_k"], dim=1)
                top_weights = torch.softmax(top_logits, dim=1)
            # Accumulate in fixed top-k/token order on CPU. CUDA scatter_add uses
            # atomic floating-point additions when expert IDs repeat and is not
            # bitwise deterministic.
            indices = top_indices.cpu().numpy()
            weights = top_weights.cpu().numpy().astype(np.float64)
            for slot in range(topology["top_k"]):
                routing_counts[layer_index] += np.bincount(
                    indices[:, slot], minlength=num_experts
                ).astype(np.int64)
                np.add.at(routing_mass[layer_index], indices[:, slot], weights[:, slot])

            remaining = signature_tokens_per_layer - int(captured_count[layer_index])
            if remaining > 0:
                sample = hidden[:remaining].to(device="cpu", dtype=torch.float32)
                captured[layer_index].append(sample)
                captured_count[layer_index] += sample.shape[0]

        return hook

    for layer_index, layer in enumerate(layers):
        hooks.append(layer.block_sparse_moe.register_forward_pre_hook(make_hook(layer_index)))

    try:
        with torch.inference_mode():
            for start in range(0, token_batches.shape[0], forward_batch_size):
                ids = torch.as_tensor(
                    token_batches[start : start + forward_batch_size],
                    dtype=torch.long,
                    device=device,
                )
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    finally:
        for handle in hooks:
            handle.remove()

    similarities = np.empty((num_layers, num_experts, num_experts), dtype=np.float64)
    for layer_index, layer in enumerate(layers):
        if int(captured_count[layer_index]) != signature_tokens_per_layer:
            raise RuntimeError(
                f"layer {layer_index} captured {captured_count[layer_index]} signature tokens; "
                f"expected exactly {signature_tokens_per_layer}"
            )
        inputs = torch.cat(captured[layer_index], dim=0)[:signature_tokens_per_layer]
        moe = layer.block_sparse_moe
        source_device = moe.input_linear.weight.device
        source_dtype = moe.input_linear.weight.dtype
        gram = torch.zeros((num_experts, num_experts), dtype=torch.float64, device="cpu")
        # A small token chunk bounds the [experts, tokens, hidden] temporary.
        for chunk in inputs.split(16, dim=0):
            values = chunk.to(device=source_device, dtype=source_dtype)
            with torch.inference_mode():
                gate_up = torch.einsum("th,eoh->eto", values, moe.input_linear.weight)
                gate, up = gate_up.chunk(2, dim=-1)
                intermediate = moe.activation(gate) * up
                outputs = torch.einsum(
                    "eti,ehi->eth", intermediate, moe.output_linear.weight
                )
                flattened = outputs.float().reshape(num_experts, -1)
                gram += (flattened @ flattened.T).double().cpu()
        diagonal = torch.diagonal(gram).clamp_min(1e-24).sqrt()
        cosine = gram / (diagonal[:, None] * diagonal[None, :])
        cosine = cosine.clamp(-1.0, 1.0)
        cosine.fill_diagonal_(1.0)
        similarities[layer_index] = cosine.numpy()

    gpu_seconds = None
    peak_device_memory_bytes = None
    if device.type == "cuda":
        gpu_finished.record()
        torch.cuda.synchronize(device)
        gpu_seconds = float(gpu_started.elapsed_time(gpu_finished) / 1000.0)
        peak_device_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    runtime = {
        "wall_seconds": perf_counter() - wall_started,
        "gpu_seconds": gpu_seconds,
        "peak_device_memory_bytes": peak_device_memory_bytes,
        "model_forward_tokens": int(token_batches.size),
        "signature_tokens_per_layer": int(signature_tokens_per_layer),
        "source_expert_output_tokens": int(
            num_layers * num_experts * signature_tokens_per_layer
        ),
    }
    return ExpertStatistics(
        routing_counts=routing_counts,
        routing_mass=routing_mass,
        output_similarity=similarities,
        sampled_tokens_per_layer=captured_count,
        runtime=runtime,
    )


@dataclass
class _LayerSource:
    grouping: LayerGrouping
    gate_up: Any
    down: Any


def _new_grouped_moe(original: Any, grouping: LayerGrouping, dtype: Any, device: Any) -> Any:
    torch = _torch()
    nn = torch.nn
    functional = torch.nn.functional

    class GroupedGraniteMoe(nn.Module):
        def __init__(
            self,
            *,
            input_size: int,
            hidden_size: int,
            activation: Any,
            router: Any,
            layer_grouping: LayerGrouping,
            parameter_dtype: Any,
            parameter_device: Any,
        ) -> None:
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.activation = activation
            self.router = router  # Original 32-way router remains intact.
            self.num_groups = layer_grouping.num_groups
            self.register_buffer(
                "expert_to_group",
                torch.tensor(
                    layer_grouping.expert_to_group,
                    dtype=torch.long,
                    device=parameter_device,
                ),
                persistent=True,
            )
            self.gate_up = nn.Parameter(
                torch.empty(
                    layer_grouping.num_groups,
                    2 * self.hidden_size,
                    self.input_size,
                    dtype=parameter_dtype,
                    device=parameter_device,
                ),
                requires_grad=False,
            )
            self.down = nn.Parameter(
                torch.empty(
                    layer_grouping.num_groups,
                    self.input_size,
                    self.hidden_size,
                    dtype=parameter_dtype,
                    device=parameter_device,
                ),
                requires_grad=False,
            )

        def forward(self, layer_input: Any) -> tuple[Any, Any]:
            batch, length, width = layer_input.shape
            hidden = layer_input.reshape(-1, width)
            router_logits = self.router.layer(hidden).float()
            top_logits, top_experts = router_logits.topk(self.router.top_k, dim=1)
            top_weights = torch.softmax(top_logits, dim=1).type_as(hidden)
            mapped = self.expert_to_group[top_experts]
            group_weights = torch.zeros(
                (hidden.shape[0], self.num_groups), dtype=hidden.dtype, device=hidden.device
            )
            row_ids = torch.arange(hidden.shape[0], device=hidden.device)
            for slot in range(self.router.top_k):
                group_ids = mapped[:, slot]
                group_weights[row_ids, group_ids] = (
                    group_weights[row_ids, group_ids] + top_weights[:, slot]
                )

            output = torch.zeros_like(hidden)
            for group in range(self.num_groups):
                token_ids = torch.nonzero(group_weights[:, group] != 0, as_tuple=False).flatten()
                if token_ids.numel() == 0:
                    continue
                selected = hidden[token_ids]
                gate, up = functional.linear(selected, self.gate_up[group]).chunk(2, dim=-1)
                expert_output = functional.linear(self.activation(gate) * up, self.down[group])
                expert_output = expert_output * group_weights[token_ids, group, None]
                output[token_ids] = output[token_ids] + expert_output.to(output.dtype)
            return output.view(batch, length, width), router_logits

    # Pass every value explicitly. In particular, the local class must not close
    # over ``original``: doing so would keep the full source expert tensors alive
    # after the layer is replaced and defeat CPU offloading of source weights.
    return GroupedGraniteMoe(
        input_size=int(original.input_size),
        hidden_size=int(original.hidden_size),
        activation=original.activation,
        router=original.router,
        layer_grouping=grouping,
        parameter_dtype=dtype,
        parameter_device=device,
    )


class GraniteFusionController:
    """Install physical fused experts and update them for coefficient candidates."""

    def __init__(
        self,
        model: Any,
        plan: GroupingPlan,
        coefficients: Mapping[int, Mapping[int, Sequence[float]]],
        *,
        source_device: str = "model",
    ) -> None:
        topology = inspect_granite_topology(model)
        if plan.num_layers != topology["num_layers"] or plan.num_experts != topology["num_experts"]:
            raise ValueError("grouping topology does not match the Granite checkpoint")
        actual_model_id = getattr(model, "_dsai6707_model_id", None)
        actual_revision = getattr(model, "_dsai6707_model_revision", None)
        if actual_model_id != plan.model_id or actual_revision != plan.model_revision:
            raise ValueError(
                "loaded model provenance does not match grouping plan: "
                f"expected {plan.model_id}@{plan.model_revision}, "
                f"got {actual_model_id}@{actual_revision}"
            )
        self.model = model
        self.plan = plan
        self.sources: list[_LayerSource] = []
        self.modules: list[Any] = []

        for layer_index, layer in enumerate(model.model.layers):
            original = layer.block_sparse_moe
            storage_device = (
                original.input_linear.weight.device
                if source_device == "model"
                else source_device
            )
            source = _LayerSource(
                grouping=plan.layers[layer_index],
                gate_up=original.input_linear.weight.detach().to(
                    device=storage_device, copy=True
                ),
                down=original.output_linear.weight.detach().to(
                    device=storage_device, copy=True
                ),
            )
            grouped = _new_grouped_moe(
                original,
                source.grouping,
                dtype=original.input_linear.weight.dtype,
                device=original.input_linear.weight.device,
            )
            layer.block_sparse_moe = grouped
            self.sources.append(source)
            self.modules.append(grouped)
        self.apply(coefficients)

    def apply(
        self,
        coefficients: Mapping[int, Mapping[int, Sequence[float]]],
        layers: Sequence[int] | None = None,
    ) -> None:
        torch = _torch()
        selected_layers = range(self.plan.num_layers) if layers is None else layers
        with torch.inference_mode():
            for layer_index in selected_layers:
                source = self.sources[layer_index]
                target = self.modules[layer_index]
                mixing = torch.zeros(
                    (source.grouping.num_groups, source.grouping.num_experts),
                    dtype=torch.float32,
                    device=source.gate_up.device,
                )
                for group, members in enumerate(source.grouping.groups):
                    alpha = torch.as_tensor(
                        coefficients[layer_index][group],
                        dtype=torch.float32,
                        device=mixing.device,
                    )
                    if alpha.shape != (len(members),):
                        raise ValueError(
                            f"layer {layer_index} group {group}: coefficient shape mismatch"
                        )
                    if bool((alpha < 0).any()) or not torch.isclose(
                        alpha.sum(), torch.tensor(1.0, device=alpha.device), atol=1e-6
                    ):
                        raise ValueError("coefficients must be non-negative and sum to one")
                    mixing[group, list(members)] = alpha

                fused_gate_up = torch.einsum("ge,eoh->goh", mixing, source.gate_up.float())
                fused_down = torch.einsum("ge,ehi->ghi", mixing, source.down.float())
                target.gate_up.copy_(
                    fused_gate_up.to(device=target.gate_up.device, dtype=target.gate_up.dtype)
                )
                target.down.copy_(fused_down.to(device=target.down.device, dtype=target.down.dtype))

    def compression_stats(self) -> dict[str, Any]:
        torch = _torch()
        source_expert_parameters = sum(
            int(source.gate_up.numel() + source.down.numel()) for source in self.sources
        )
        fused_expert_parameters = sum(
            int(module.gate_up.numel() + module.down.numel()) for module in self.modules
        )
        current_total = sum(int(parameter.numel()) for parameter in self.model.parameters())
        original_total = current_total - fused_expert_parameters + source_expert_parameters
        return {
            "schema_version": "compression-v1",
            "original_total_parameters": original_total,
            "fused_total_parameters": current_total,
            "original_expert_parameters": source_expert_parameters,
            "fused_expert_parameters": fused_expert_parameters,
            "whole_model_reduction_fraction": 1.0 - current_total / original_total,
            "expert_reduction_fraction": 1.0
            - fused_expert_parameters / source_expert_parameters,
            "groups_per_layer": [layer.num_groups for layer in self.plan.layers],
            "source_experts_per_layer": self.plan.num_experts,
            "parameter_dtype_bytes": torch.tensor(
                [], dtype=next(self.model.parameters()).dtype
            ).element_size(),
        }

    def export_fused_experts(
        self,
        directory: str | Path,
        *,
        coefficient_artifact: Mapping[str, Any],
        tokenizer: Any,
    ) -> None:
        """Atomically export a complete grouped checkpoint into a new directory."""

        target = Path(directory)
        if target.exists():
            raise ValueError(f"refusing to overwrite checkpoint target: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.export-", dir=target.parent)
        )
        try:
            self._export_fused_experts_into(
                temporary,
                coefficient_artifact=coefficient_artifact,
                tokenizer=tokenizer,
            )
            os.replace(temporary, target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _export_fused_experts_into(
        self,
        directory: str | Path,
        *,
        coefficient_artifact: Mapping[str, Any],
        tokenizer: Any,
    ) -> None:
        try:
            from safetensors.torch import save_model
        except ImportError as exc:
            raise RuntimeError("safetensors is required to export fused weights") from exc
        destination = Path(directory)
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(
                f"refusing to overwrite non-empty checkpoint directory: {destination}"
            )
        destination.mkdir(parents=True, exist_ok=True)
        # Save the full physical model, not merely an expert delta. The state dict
        # contains G_l fused experts, the original 32-way routers, and all non-expert
        # weights. save_model safely handles tied embedding/lm_head storage.
        weight_path = destination / "model.safetensors"
        save_model(self.model, str(weight_path), metadata={"format": "pt"})
        attention_implementation = str(self.model.config._attn_implementation)
        if attention_implementation not in _CHECKPOINT_ATTENTION_IMPLEMENTATIONS:
            raise ValueError(
                "cannot export an unsupported attention implementation: "
                f"{attention_implementation!r}"
            )
        # Transformers deliberately omits ``_attn_implementation_internal`` from
        # serialized configs.  Its save_pretrained path also clears the autoset
        # marker so a reload can select an implementation again.  Mirror that
        # behavior here; the manifest below records the exact implementation.
        export_config = copy.deepcopy(self.model.config)
        export_config._attn_implementation_autoset = False
        export_config.to_json_file(destination / "config.json", use_diff=False)
        tokenizer.save_pretrained(destination)
        write_json(destination / "grouping.json", self.plan.to_dict())
        write_json(destination / "coefficients.json", dict(coefficient_artifact))
        compression = self.compression_stats()
        compression["standalone_checkpoint_bytes"] = weight_path.stat().st_size
        compression["standalone_weight_file_bytes"] = weight_path.stat().st_size
        base_checkpoint = getattr(self.model, "_dsai6707_base_checkpoint", None)
        compression["base_weight_checkpoint"] = base_checkpoint
        if base_checkpoint is not None:
            base_bytes = int(base_checkpoint["total_bytes"])
            compression["weight_checkpoint_reduction_fraction"] = (
                1.0 - weight_path.stat().st_size / base_bytes
            )
        else:
            compression["weight_checkpoint_reduction_fraction"] = None
        write_json(destination / "compression.json", compression)
        required_names = _CHECKPOINT_REQUIRED_FILES
        present_names = {path.name for path in destination.iterdir() if path.is_file()}
        missing = sorted(required_names - present_names)
        if missing:
            raise RuntimeError(f"checkpoint export is missing required files: {missing}")
        file_hashes = {
            str(path.relative_to(destination)): {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(
                destination.rglob("*"), key=lambda item: str(item.relative_to(destination))
            )
            if path.is_file() and path.name != "manifest.json"
        }
        tokenizer_id = getattr(tokenizer, "_dsai6707_tokenizer_id", None)
        tokenizer_revision = getattr(tokenizer, "_dsai6707_tokenizer_revision", None)
        if not isinstance(tokenizer_id, str) or not tokenizer_id:
            raise ValueError("tokenizer provenance is missing its pinned id")
        if not isinstance(tokenizer_revision, str) or not tokenizer_revision:
            raise ValueError("tokenizer provenance is missing its pinned revision")
        weight_bytes = int(file_hashes["model.safetensors"]["bytes"])
        payload_bytes = sum(int(entry["bytes"]) for entry in file_hashes.values())
        write_json(
            destination / "manifest.json",
            {
                "schema_version": "fused-checkpoint-v1",
                "base_model": {"id": self.plan.model_id, "revision": self.plan.model_revision},
                "tokenizer": {"id": tokenizer_id, "revision": tokenizer_revision},
                "transformers_version": "4.46.3",
                "format": "standalone_grouped_granite_safetensors",
                "attention_implementation": attention_implementation,
                "parameter_dtype": str(next(self.model.parameters()).dtype).removeprefix(
                    "torch."
                ),
                "environment": environment_snapshot(),
                "model_weight_bytes": weight_bytes,
                "declared_payload_bytes": payload_bytes,
                "payload_definition": (
                    "sum of every file in files; excludes manifest.json and permitted "
                    "post-export CLI metadata"
                ),
                "files": file_hashes,
                "note": (
                    "Use dsai6707.granite.load_grouped_checkpoint; the stock Granite "
                    "class has different expert keys."
                ),
            },
        )


def _validate_grouped_checkpoint_manifest(source: Path) -> dict[str, Any]:
    """Validate a grouped checkpoint's closed, hashed payload inventory."""

    manifest_path = source / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"grouped checkpoint manifest is missing: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "fused-checkpoint-v1":
        raise ValueError("unsupported grouped checkpoint manifest schema")
    if manifest.get("format") != "standalone_grouped_granite_safetensors":
        raise ValueError("unsupported grouped checkpoint format")
    if manifest.get("transformers_version") != "4.46.3":
        raise ValueError("grouped checkpoint was not exported with transformers 4.46.3")
    if manifest.get("attention_implementation") not in (
        _CHECKPOINT_ATTENTION_IMPLEMENTATIONS
    ):
        raise ValueError("grouped checkpoint has an unsupported attention implementation")
    if manifest.get("parameter_dtype") not in {"bfloat16", "float16", "float32"}:
        raise ValueError("grouped checkpoint has an unsupported parameter dtype")
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict) or not all(
        isinstance(tokenizer.get(key), str) and tokenizer[key]
        for key in ("id", "revision")
    ):
        raise ValueError("grouped checkpoint manifest has no pinned tokenizer provenance")
    file_hashes = manifest.get("files")
    if not isinstance(file_hashes, dict) or not file_hashes:
        raise ValueError("grouped checkpoint manifest has no file hash inventory")
    missing_required = sorted(_CHECKPOINT_REQUIRED_FILES - set(file_hashes))
    if missing_required:
        raise ValueError(
            f"grouped checkpoint manifest omits required files: {missing_required}"
        )

    declared: set[str] = set()
    payload_bytes = 0
    for relative, expected in file_hashes.items():
        relative_path = Path(str(relative))
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or str(relative_path) in ("", ".")
        ):
            raise ValueError(f"unsafe path in checkpoint manifest: {relative}")
        normalized = relative_path.as_posix()
        if normalized != str(relative):
            raise ValueError(f"non-canonical path in checkpoint manifest: {relative}")
        if not isinstance(expected, dict) or set(expected) != {"sha256", "bytes"}:
            raise ValueError(f"malformed checkpoint file entry: {relative}")
        declared.add(normalized)
        path = source / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint file declared by manifest is missing: {path}")
        expected_bytes = int(expected["bytes"])
        if expected_bytes < 0 or path.stat().st_size != expected_bytes:
            raise ValueError(f"checkpoint byte-size mismatch: {relative}")
        if sha256_file(path) != str(expected["sha256"]):
            raise ValueError(f"checkpoint SHA-256 mismatch: {relative}")
        payload_bytes += expected_bytes

    actual = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file()
    }
    unexpected = sorted(actual - declared - _CHECKPOINT_ALLOWED_UNDECLARED_FILES)
    if unexpected:
        raise ValueError(f"undeclared files in grouped checkpoint: {unexpected}")
    weight_bytes = int(file_hashes["model.safetensors"]["bytes"])
    if int(manifest.get("model_weight_bytes", -1)) != weight_bytes:
        raise ValueError("manifest model_weight_bytes disagrees with model.safetensors")
    if int(manifest.get("declared_payload_bytes", -1)) != payload_bytes:
        raise ValueError("manifest declared_payload_bytes disagrees with file inventory")
    return manifest


def _install_grouped_modules_without_sources(model: Any, plan: GroupingPlan) -> None:
    """Replace standard expert containers without retaining the source tensors."""

    topology = inspect_granite_topology(model)
    if plan.num_layers != topology["num_layers"] or plan.num_experts != topology["num_experts"]:
        raise ValueError("grouping topology does not match checkpoint config")
    for layer_index, layer in enumerate(model.model.layers):
        original = layer.block_sparse_moe
        layer.block_sparse_moe = _new_grouped_moe(
            original,
            plan.layers[layer_index],
            dtype=original.input_linear.weight.dtype,
            device=original.input_linear.weight.device,
        )


def _capture_nonpersistent_buffers(model: Any) -> list[tuple[Any, str, Any, tuple[str, ...]]]:
    """Preserve initialized buffers that a safetensors state dict cannot contain."""

    snapshots = []
    for module in model.modules():
        for name in sorted(module._non_persistent_buffers_set):
            buffer = module._buffers.get(name)
            if buffer is None:
                continue
            aliases = tuple(
                attribute
                for attribute, value in vars(module).items()
                if value is buffer
            )
            snapshots.append((module, name, buffer.detach().clone(), aliases))
    return snapshots


def _restore_nonpersistent_buffers(
    snapshots: Sequence[tuple[Any, str, Any, tuple[str, ...]]], *, device: str
) -> None:
    """Restore buffers emptied by ``Module.to_empty`` and their direct aliases."""

    for module, name, value, aliases in snapshots:
        restored = value.to(device=device)
        module._buffers[name] = restored
        for attribute in aliases:
            setattr(module, attribute, restored)


def load_grouped_checkpoint(directory: str | Path, *, device: str = "cuda") -> tuple[Any, Any]:
    """Load a standalone grouped checkpoint without materializing source experts."""

    source = Path(directory)
    manifest = _validate_grouped_checkpoint_manifest(source)
    torch = _torch()
    if str(device).startswith("cuda"):
        ensure_cublas_workspace_config(
            cuda_initialized=bool(torch.cuda.is_initialized())
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available")
    _configure_torch_determinism(torch)
    transformers = _transformers()
    try:
        from accelerate import init_empty_weights
        from safetensors.torch import load_model
    except ImportError as exc:
        raise RuntimeError(
            "accelerate and safetensors are required to load grouped checkpoints"
        ) from exc
    file_hashes = manifest.get("files")
    assert isinstance(file_hashes, dict)
    dtype_name = str(manifest.get("parameter_dtype", ""))
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype_name)
    if dtype is None:
        raise ValueError(f"unsupported or missing checkpoint dtype: {dtype_name!r}")
    plan = GroupingPlan.from_dict(read_json(source / "grouping.json"))
    if manifest.get("base_model") != {
        "id": plan.model_id,
        "revision": plan.model_revision,
    }:
        raise ValueError("manifest base model disagrees with grouping metadata")
    config = transformers.AutoConfig.from_pretrained(source, local_files_only=True)
    attention_implementation = str(manifest["attention_implementation"])
    config._attn_implementation = attention_implementation
    config._attn_implementation_autoset = False
    with init_empty_weights(include_buffers=False):
        model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
        _install_grouped_modules_without_sources(model, plan)
    if model.config._attn_implementation != attention_implementation:
        raise RuntimeError(
            "loaded attention implementation disagrees with checkpoint manifest"
        )
    nonpersistent_buffers = _capture_nonpersistent_buffers(model)
    model.to_empty(device=device)
    # ``to_empty`` breaks shared Parameter identities. Re-tie before strict
    # safetensors loading because save_model stores only one key per shared tensor.
    model.tie_weights()
    load_model(model, str(source / "model.safetensors"), strict=True, device=device)
    model.tie_weights()
    _restore_nonpersistent_buffers(nonpersistent_buffers, device=device)
    model.eval()
    model._dsai6707_model_id = plan.model_id
    model._dsai6707_model_revision = plan.model_revision
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokenizer._dsai6707_tokenizer_id = manifest["tokenizer"]["id"]
    tokenizer._dsai6707_tokenizer_revision = manifest["tokenizer"]["revision"]
    # A tiny structural check that does not assume the original expert container.
    if len(model.model.layers) != plan.num_layers:
        raise RuntimeError("loaded grouped checkpoint has the wrong layer count")
    if any(
        layer.block_sparse_moe.gate_up.shape[0] != grouping.num_groups
        for layer, grouping in zip(model.model.layers, plan.layers, strict=False)
    ):
        raise RuntimeError("loaded physical expert count disagrees with grouping.json")
    for layer, grouping in zip(model.model.layers, plan.layers, strict=False):
        actual_mapping = layer.block_sparse_moe.expert_to_group.detach().cpu().tolist()
        if actual_mapping != list(grouping.expert_to_group):
            raise RuntimeError("loaded routing map disagrees with grouping.json")
    floating_dtypes = {
        parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
    }
    if floating_dtypes != {dtype}:
        raise RuntimeError(
            f"loaded parameter dtypes {floating_dtypes} disagree with manifest dtype {dtype}"
        )
    return model, tokenizer


class CalibrationNLL:
    """Full-model, token-weighted next-token NLL objective."""

    def __init__(
        self,
        controller: GraniteFusionController,
        layout: Any,
        token_batches: np.ndarray,
        *,
        loss_tokens: int,
        micro_batch_size: int = 1,
    ) -> None:
        if token_batches.ndim != 2 or loss_tokens <= 0 or micro_batch_size <= 0:
            raise ValueError("invalid calibration objective settings")
        available = token_batches.shape[0] * (token_batches.shape[1] - 1)
        if available < loss_tokens:
            raise ValueError(f"only {available} loss positions available, need {loss_tokens}")
        self.controller = controller
        self.layout = layout
        self.token_batches = token_batches
        self.loss_tokens = int(loss_tokens)
        self.micro_batch_size = int(micro_batch_size)
        self.previous: np.ndarray | None = None

    def __call__(self, vector: np.ndarray) -> Any:
        from .evolution import FitnessValue

        torch = _torch()
        started = perf_counter()
        model = self.controller.model
        device = next(model.parameters()).device
        nll_sum = 0.0
        consumed = 0
        changed_layers: list[int] | None = None
        total_gpu_start = None
        fusion_gpu_start = None
        fusion_gpu_end = None
        forward_gpu_start = None
        forward_gpu_end = None
        total_gpu_end = None
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                total_gpu_start = torch.cuda.Event(enable_timing=True)
                fusion_gpu_start = torch.cuda.Event(enable_timing=True)
                fusion_gpu_end = torch.cuda.Event(enable_timing=True)
                forward_gpu_start = torch.cuda.Event(enable_timing=True)
                forward_gpu_end = torch.cuda.Event(enable_timing=True)
                total_gpu_end = torch.cuda.Event(enable_timing=True)
                total_gpu_start.record()
                fusion_gpu_start.record()
            fusion_started = perf_counter()
            decoded = self.layout.decode(vector)
            if self.previous is not None:
                changed_layers = []
                for layer, (start, stop) in enumerate(self.layout.layer_ranges):
                    if not np.array_equal(vector[start:stop], self.previous[start:stop]):
                        changed_layers.append(layer)
            self.controller.apply(decoded, layers=changed_layers)
            self.previous = vector.copy()
            if device.type == "cuda":
                fusion_gpu_end.record()
                torch.cuda.synchronize(device)
            fusion_seconds = perf_counter() - fusion_started

            if device.type == "cuda":
                forward_gpu_start.record()
            forward_started = perf_counter()
            with torch.inference_mode():
                for start in range(0, self.token_batches.shape[0], self.micro_batch_size):
                    if consumed >= self.loss_tokens:
                        break
                    ids = torch.as_tensor(
                        self.token_batches[start : start + self.micro_batch_size],
                        dtype=torch.long,
                        device=device,
                    )
                    logits = model(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        use_cache=False,
                    ).logits
                    losses = torch.nn.functional.cross_entropy(
                        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                        ids[:, 1:].reshape(-1),
                        reduction="none",
                    )
                    take = min(losses.numel(), self.loss_tokens - consumed)
                    nll_sum += float(losses[:take].double().sum().item())
                    consumed += take
            if device.type == "cuda":
                forward_gpu_end.record()
                total_gpu_end.record()
                torch.cuda.synchronize(device)
            elapsed = perf_counter() - started
            forward_seconds = perf_counter() - forward_started
            gpu_seconds = (
                float(total_gpu_start.elapsed_time(total_gpu_end) / 1000.0)
                if device.type == "cuda"
                else None
            )
            fusion_gpu_seconds = (
                float(fusion_gpu_start.elapsed_time(fusion_gpu_end) / 1000.0)
                if device.type == "cuda"
                else None
            )
            forward_gpu_seconds = (
                float(forward_gpu_start.elapsed_time(forward_gpu_end) / 1000.0)
                if device.type == "cuda"
                else None
            )
            peak = (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            )
            return FitnessValue(
                nll_sum / consumed,
                {
                    "status": "ok",
                    "nll_sum": nll_sum,
                    "loss_tokens": consumed,
                    "wall_seconds": elapsed,
                    "fusion_seconds": fusion_seconds,
                    "forward_seconds": forward_seconds,
                    "gpu_seconds": gpu_seconds,
                    "fusion_gpu_seconds": fusion_gpu_seconds,
                    "forward_gpu_seconds": forward_gpu_seconds,
                    "peak_device_memory_bytes": peak,
                    "updated_layers": changed_layers,
                },
            )
        except RuntimeError as exc:
            # A started candidate consumes one evaluation by protocol. Force a full
            # rebuild next time in case the failed update was only partially applied.
            self.previous = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return FitnessValue(
                1.0e30,
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "nll_sum": None,
                    "loss_tokens": consumed,
                    "wall_seconds": perf_counter() - started,
                    "updated_layers": changed_layers,
                },
            )


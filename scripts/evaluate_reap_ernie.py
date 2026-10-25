#!/usr/bin/env python3
"""Evaluate one local REAP ERNIE checkpoint on pinned ARC-Easy/OpenBookQA.

The runner intentionally evaluates exactly one checkpoint per process.  Invoke
it once for the original model and once for the pruned model so that two copies
of the 21.9B model can never coexist on the GPU.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ISOLATED_DEPENDENCIES = ROOT / "external" / "reap-official" / ".deps"
PROJECT_SOURCE = ROOT / "src"
REQUIRED_LOCAL_CODE = (
    "configuration_ernie4_5_moe.py",
    "modeling_ernie4_5_moe.py",
)
EXPECTED_SAMPLES = {
    "test": {"arc_easy": 2376, "openbookqa": 500},
    "validation": {"arc_easy": 570, "openbookqa": 500},
}
EXPECTED_ISOLATED_VERSIONS = {
    "transformers": "4.53.3",
    "datasets": "3.6.0",
    "accelerate": "1.7.0",
    "lm-eval": "0.4.8",
}
ALLOWED_CUBLAS_WORKSPACE_CONFIGS = frozenset({":16:8", ":4096:8"})


class ReapErnieEvaluationError(RuntimeError):
    """Raised when a precondition or evidence check fails."""


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="local original/pruned checkpoint")
    parser.add_argument(
        "--tokenizer-path",
        required=True,
        help="local patched base ERNIE directory used for tokenizer and code provenance",
    )
    parser.add_argument("--label", required=True, help="result label, e.g. original or reap50")
    parser.add_argument("--output", required=True, help="new JSON evidence file")
    parser.add_argument("--split", choices=sorted(EXPECTED_SAMPLES), default="validation")
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=1,
        help="explicit lm-eval request batch size (default: 1)",
    )
    parser.add_argument(
        "--log-samples",
        action="store_true",
        help="retain lm-eval per-sample requests, responses, and metrics in the evidence JSON",
    )
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--seed", type=int, default=6707)
    parser.add_argument(
        "--smoke-prompt-only",
        action="store_true",
        help="load and validate the checkpoint, then run one prompt instead of lm-eval",
    )
    return parser


def _bootstrap_local_dependencies() -> None:
    """Put the REAP-compatible packages and this project's source first."""

    if not ISOLATED_DEPENDENCIES.is_dir():
        raise ReapErnieEvaluationError(
            f"isolated REAP dependencies are missing: {ISOLATED_DEPENDENCIES}"
        )
    for path in (PROJECT_SOURCE, ISOLATED_DEPENDENCIES):
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)


def _require_isolated_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution, expected in EXPECTED_ISOLATED_VERSIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ReapErnieEvaluationError(f"required package is missing: {distribution}") from exc
        if actual != expected:
            raise ReapErnieEvaluationError(
                f"{distribution}=={expected} is required, found {actual}"
            )
        versions[distribution] = actual
    return versions


def _require_one_visible_gpu_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    values = os.environ if environ is None else environ
    visible = values.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip() or len(visible.split(",")) != 1:
        raise ReapErnieEvaluationError(
            "set CUDA_VISIBLE_DEVICES to exactly one physical GPU before launching"
        )
    if visible != visible.strip():
        raise ReapErnieEvaluationError("CUDA_VISIBLE_DEVICES must not contain whitespace")
    workspace = values.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in ALLOWED_CUBLAS_WORKSPACE_CONFIGS:
        raise ReapErnieEvaluationError(
            "set CUBLAS_WORKSPACE_CONFIG to :4096:8 or :16:8 before launching"
        )
    return {
        "CUDA_VISIBLE_DEVICES": visible,
        "CUBLAS_WORKSPACE_CONFIG": str(workspace),
    }


def _evaluation_config(
    split: str,
    *,
    batch_size: int = 1,
    log_samples: bool = False,
) -> dict[str, Any]:
    if split not in EXPECTED_SAMPLES:
        raise ValueError(f"unsupported split: {split}")
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer")
    from dsai6707.config import FORMAL_EVALUATION_PROTOCOL_ID, LOCKED_TASK_DATASETS

    datasets = copy.deepcopy(LOCKED_TASK_DATASETS)
    for task in ("arc_easy", "openbookqa"):
        datasets[task]["split"] = split
    return {
        "protocol_id": FORMAL_EVALUATION_PROTOCOL_ID,
        "tasks": ["arc_easy", "openbookqa"],
        "task_datasets": datasets,
        "num_fewshot": 0,
        "primary_metric": "acc",
        "report_metrics": ["acc", "acc_norm"],
        "batch_size": int(batch_size),
        "random_seed": 0,
        "numpy_random_seed": 1234,
        "torch_random_seed": 1234,
        "fewshot_random_seed": 1234,
        "log_samples": bool(log_samples),
        "expected_samples": copy.deepcopy(EXPECTED_SAMPLES[split]),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _small_file_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReapErnieEvaluationError(f"required local file is missing: {path}")
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _weight_files(checkpoint: Path) -> tuple[list[Path], dict[str, Any] | None]:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            names = sorted(set(index["weight_map"].values()))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ReapErnieEvaluationError(f"invalid safetensors index: {index_path}") from exc
        if not names or not all(isinstance(name, str) for name in names):
            raise ReapErnieEvaluationError(f"empty safetensors index: {index_path}")
        return [checkpoint / name for name in names], _small_file_evidence(index_path)
    single = checkpoint / "model.safetensors"
    if single.is_file():
        return [single], None
    raise ReapErnieEvaluationError(f"no safetensors checkpoint found in {checkpoint}")


def _checkpoint_evidence(model_path: Path, tokenizer_path: Path) -> dict[str, Any]:
    checkpoint = model_path.expanduser().resolve(strict=True)
    tokenizer = tokenizer_path.expanduser().resolve(strict=True)
    if not checkpoint.is_dir() or not tokenizer.is_dir():
        raise ReapErnieEvaluationError("model and tokenizer paths must be local directories")

    code: list[dict[str, Any]] = []
    for name in REQUIRED_LOCAL_CODE:
        model_file = checkpoint / name
        reference_file = tokenizer / name
        model_record = _small_file_evidence(model_file)
        reference_record = _small_file_evidence(reference_file)
        if model_record["sha256"] != reference_record["sha256"]:
            raise ReapErnieEvaluationError(
                f"checkpoint local code differs from patched base code: {name}"
            )
        model_record["matches_tokenizer_base"] = True
        code.append(model_record)

    weights, index_evidence = _weight_files(checkpoint)
    missing = [str(path) for path in weights if not path.is_file()]
    if missing:
        raise ReapErnieEvaluationError(f"checkpoint weight shards are missing: {missing}")
    config_path = checkpoint / "config.json"
    try:
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReapErnieEvaluationError(f"invalid model config: {config_path}") from exc
    return {
        "model_path": str(checkpoint),
        "tokenizer_path": str(tokenizer),
        "config": _small_file_evidence(config_path),
        "declared_topology": {
            key: raw_config.get(key)
            for key in (
                "model_type",
                "num_hidden_layers",
                "moe_num_experts",
                "moe_num_shared_experts",
                "moe_k",
                "torch_dtype",
            )
        },
        "local_remote_code": code,
        "weight_index": index_evidence,
        "weight_shards": [{"name": path.name, "bytes": path.stat().st_size} for path in weights],
        "total_weight_bytes": sum(path.stat().st_size for path in weights),
        "tokenizer_config": _small_file_evidence(tokenizer / "tokenizer_config.json"),
        "tokenizer_model": _small_file_evidence(tokenizer / "tokenizer.model"),
    }


def _assert_clean_loading_info(loading_info: Any) -> dict[str, list[Any]]:
    if not isinstance(loading_info, Mapping):
        raise ReapErnieEvaluationError("Transformers did not return weight-loading evidence")
    checked: dict[str, list[Any]] = {}
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        raw = loading_info.get(key, [])
        values = list(raw) if raw is not None else []
        checked[key] = values
        if values:
            preview = values[:8]
            raise ReapErnieEvaluationError(f"non-clean checkpoint load ({key}): {preview}")
    return checked


def _configure_runtime(torch: Any, *, seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


def _runtime_model_evidence(model: Any) -> dict[str, Any]:
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    parameter_devices = sorted({str(parameter.device) for parameter in parameters})
    buffer_devices = sorted({str(buffer.device) for buffer in buffers})
    if parameter_devices != ["cuda:0"]:
        raise ReapErnieEvaluationError(
            f"all parameters must be on cuda:0, found {parameter_devices}"
        )
    non_cuda_buffers = [device for device in buffer_devices if device != "cuda:0"]
    if non_cuda_buffers:
        raise ReapErnieEvaluationError(f"model buffers escaped cuda:0: {non_cuda_buffers}")
    dtype_elements: Counter[str] = Counter()
    for parameter in parameters:
        dtype_elements[str(parameter.dtype)] += parameter.numel()
    if dtype_elements.get("torch.bfloat16", 0) == 0:
        raise ReapErnieEvaluationError("checkpoint did not load BF16 model weights")
    trainable = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    if trainable:
        raise ReapErnieEvaluationError("inference model unexpectedly has trainable parameters")
    config = model.config
    return {
        "class": model.__class__.__name__,
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_count": trainable,
        "parameter_devices": parameter_devices,
        "buffer_devices": buffer_devices,
        "parameter_dtype_elements": dict(sorted(dtype_elements.items())),
        "model_type": getattr(config, "model_type", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "moe_num_experts": getattr(config, "moe_num_experts", None),
        "moe_num_shared_experts": getattr(config, "moe_num_shared_experts", None),
        "moe_k": getattr(config, "moe_k", None),
        "attention_implementation": getattr(config, "_attn_implementation", None),
        "hf_device_map": getattr(model, "hf_device_map", None),
    }


def _load_local_model(
    model_path: Path,
    tokenizer_path: Path,
    *,
    device: str,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    loaded = transformers.AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": device},
        output_loading_info=True,
        attn_implementation="sdpa",
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise ReapErnieEvaluationError("Transformers did not return (model, loading_info)")
    model, loading_info = loaded
    clean_loading_info = _assert_clean_loading_info(loading_info)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    evidence = _runtime_model_evidence(model)
    evidence["loading_info"] = clean_loading_info
    evidence["transformers_version"] = transformers.__version__
    evidence["requested_dtype"] = "torch.bfloat16"
    evidence["requested_device_map"] = {"": device}
    return model, tokenizer, evidence


def _smoke_prompt(model: Any, tokenizer: Any, *, device: str) -> dict[str, Any]:
    import torch

    prompt = "The capital of France is"
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    torch.cuda.reset_peak_memory_stats(torch.device(device))
    with torch.inference_mode():
        output = model(**encoded, use_cache=False)
    logits = output.logits[:, -1, :].float()
    finite = bool(torch.isfinite(logits).all().item())
    if not finite:
        raise ReapErnieEvaluationError("smoke prompt produced non-finite logits")
    token_id = int(logits.argmax(dim=-1).item())
    return {
        "prompt": prompt,
        "input_token_count": int(encoded["input_ids"].shape[-1]),
        "logits_shape": list(output.logits.shape),
        "all_logits_finite": finite,
        "greedy_next_token_id": token_id,
        "greedy_next_token_text": tokenizer.decode([token_id]),
        "peak_device_memory_bytes": int(torch.cuda.max_memory_allocated(torch.device(device))),
    }


def run(args: argparse.Namespace) -> Path:
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise ReapErnieEvaluationError(f"refusing to overwrite existing output: {output}")
    if not str(args.label).strip():
        raise ReapErnieEvaluationError("--label cannot be empty")
    environment = _require_one_visible_gpu_environment()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _bootstrap_local_dependencies()
    versions = _require_isolated_versions()

    # Import torch only after checking the pre-CUDA environment variables.
    import torch

    _configure_runtime(torch, seed=int(args.seed))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ReapErnieEvaluationError("the process must see exactly one CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise ReapErnieEvaluationError("the selected GPU does not support BF16")

    model_path = Path(args.model_path)
    tokenizer_path = Path(args.tokenizer_path)
    checkpoint = _checkpoint_evidence(model_path, tokenizer_path)
    model, tokenizer, model_evidence = _load_local_model(
        model_path.resolve(), tokenizer_path.resolve(), device=args.device
    )
    common = {
        "schema_version": "reap-ernie-single-model-evaluation-v1",
        "label": str(args.label),
        "split": str(args.split),
        "limit": args.limit,
        "seed": int(args.seed),
        "training_performed": False,
        "single_model_process": True,
        "requested_device": str(args.device),
        "gpu_environment": environment,
        "isolated_dependency_versions": versions,
        "evaluation_settings": {
            "batch_size": int(args.batch_size),
            "log_samples": bool(args.log_samples),
        },
        "checkpoint": checkpoint,
        "model_load": model_evidence,
    }
    if args.smoke_prompt_only:
        payload = {
            **common,
            "mode": "smoke_prompt_only",
            "smoke_prompt": _smoke_prompt(model, tokenizer, device=args.device),
        }
    else:
        from dsai6707.evaluation import evaluate_zero_shot

        evaluation = evaluate_zero_shot(
            model,
            tokenizer,
            _evaluation_config(
                args.split,
                batch_size=int(args.batch_size),
                log_samples=bool(args.log_samples),
            ),
            method=str(args.label),
            seed=int(args.seed),
            limit=args.limit,
        )
        evaluation["reap_ernie_runner"] = common
        payload = evaluation

    from dsai6707.io import write_json

    write_json(output, payload)
    return output


def main() -> None:
    args = build_parser().parse_args()
    output = run(args)
    print(f"saved ERNIE evaluation evidence to {output}", flush=True)


if __name__ == "__main__":
    main()

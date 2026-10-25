"""Small, dependency-free helpers for reproducible artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CUBLAS_WORKSPACE_CONFIG_NAME = "CUBLAS_WORKSPACE_CONFIG"
CUBLAS_WORKSPACE_CONFIG_DEFAULT = ":4096:8"
CUBLAS_WORKSPACE_CONFIG_ALLOWED = frozenset({":4096:8", ":16:8"})
_cublas_pre_cuda_verification: str | None = None


def utc_now() -> str:
    """Return a stable ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    """Atomically write UTF-8 JSON so an interrupted run cannot leave half a file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL record is not an object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def ensure_cublas_workspace_config(*, cuda_initialized: bool) -> dict[str, Any]:
    """Set and validate deterministic cuBLAS workspace configuration.

    The variable is read by cuBLAS while a CUDA context is initialized.  It is
    therefore unsafe to fill in a missing value after CUDA has already been
    initialized.  Keeping this helper independent of PyTorch makes that timing
    rule directly testable on CPU-only machines.
    """

    global _cublas_pre_cuda_verification

    value = os.environ.get(CUBLAS_WORKSPACE_CONFIG_NAME)
    configured_now = False
    if value is None:
        if cuda_initialized:
            raise RuntimeError(
                "CUBLAS_WORKSPACE_CONFIG is missing after CUDA initialization; "
                "deterministic cuBLAS execution cannot be guaranteed"
            )
        value = CUBLAS_WORKSPACE_CONFIG_DEFAULT
        os.environ[CUBLAS_WORKSPACE_CONFIG_NAME] = value
        configured_now = True
    if value not in CUBLAS_WORKSPACE_CONFIG_ALLOWED:
        allowed = ", ".join(sorted(CUBLAS_WORKSPACE_CONFIG_ALLOWED))
        raise RuntimeError(
            f"unsupported CUBLAS_WORKSPACE_CONFIG={value!r}; expected one of {allowed}"
        )
    if not cuda_initialized:
        _cublas_pre_cuda_verification = value
    return {
        "value": value,
        "valid": True,
        "configured_now": configured_now,
        "cuda_initialized_at_check": bool(cuda_initialized),
        "verified_before_cuda_initialization_in_process": (
            _cublas_pre_cuda_verification == value
        ),
    }


def environment_snapshot() -> dict[str, Any]:
    """Capture software and accelerator facts needed to reproduce a timed run."""

    packages = {
        label: _distribution_version(distribution)
        for label, distribution in {
            "numpy": "numpy",
            "yaml": "PyYAML",
            "torch": "torch",
            "transformers": "transformers",
            "datasets": "datasets",
            "lm_eval": "lm-eval",
            "networkx": "networkx",
            "accelerate": "accelerate",
            "safetensors": "safetensors",
        }.items()
    }
    accelerator: dict[str, Any] = {
        "cuda_available": False,
        "cuda_runtime": None,
        "cudnn": None,
        "driver_version": None,
        "devices": [],
        "cublas_workspace_config": {
            "value": os.environ.get(CUBLAS_WORKSPACE_CONFIG_NAME),
            "valid": os.environ.get(CUBLAS_WORKSPACE_CONFIG_NAME)
            in CUBLAS_WORKSPACE_CONFIG_ALLOWED,
            "configured_now": False,
            "cuda_initialized_at_check": None,
            "verified_before_cuda_initialization_in_process": (
                _cublas_pre_cuda_verification
                == os.environ.get(CUBLAS_WORKSPACE_CONFIG_NAME)
                and _cublas_pre_cuda_verification is not None
            ),
        },
    }
    try:
        import torch

        # is_initialized() is a read-only state query and does not create a CUDA
        # context.  Configure cuBLAS before is_available(), device enumeration,
        # or any other CUDA operation below.
        cuda_initialized = bool(torch.cuda.is_initialized())
        accelerator["cublas_workspace_config"] = ensure_cublas_workspace_config(
            cuda_initialized=cuda_initialized
        )
        accelerator["cuda_available"] = bool(torch.cuda.is_available())
        accelerator["cuda_runtime"] = torch.version.cuda
        accelerator["cudnn"] = torch.backends.cudnn.version()
        accelerator["deterministic_algorithms"] = bool(
            torch.are_deterministic_algorithms_enabled()
        )
        accelerator["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        accelerator["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        if torch.cuda.is_available():
            accelerator["devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(
                        torch.cuda.get_device_properties(index).total_memory
                    ),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        accelerator["probe_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"

    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        versions = sorted({line.strip() for line in query.stdout.splitlines() if line.strip()})
        accelerator["driver_version"] = versions
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    return {
        "captured_at": utc_now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "accelerator": accelerator,
    }


"""Artifact references and tamper/staleness checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import read_json, sha256_file, utc_now, write_json


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    sha256: str

    @classmethod
    def capture(cls, path: str | Path, *, relative_to: str | Path | None = None) -> ArtifactRef:
        source = Path(path).resolve()
        recorded = source
        if relative_to is not None:
            recorded = source.relative_to(Path(relative_to).resolve())
        return cls(path=str(recorded), sha256=sha256_file(source))

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


def verify_ref(reference: Mapping[str, Any], *, relative_to: str | Path | None = None) -> Path:
    path = Path(str(reference["path"]))
    if not path.is_absolute() and relative_to is not None:
        path = Path(relative_to) / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    expected = str(reference["sha256"])
    if actual != expected:
        raise ValueError(f"artifact hash mismatch for {path}: expected {expected}, got {actual}")
    return path


def write_stage_manifest(
    path: str | Path,
    *,
    stage: str,
    inputs: Mapping[str, ArtifactRef],
    outputs: Mapping[str, ArtifactRef],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    write_json(
        path,
        {
            "schema_version": "stage-manifest-v1",
            "stage": stage,
            "created_at": utc_now(),
            "inputs": {name: ref.to_dict() for name, ref in inputs.items()},
            "outputs": {name: ref.to_dict() for name, ref in outputs.items()},
            "metadata": dict(metadata or {}),
        },
    )


def verify_stage_manifest(
    path: str | Path,
    *,
    run_dir: str | Path,
    _visited: set[Path] | None = None,
) -> dict[str, Any]:
    """Verify every declared input/output hash before a downstream stage proceeds."""

    manifest_path = Path(path).resolve()
    visited = set() if _visited is None else _visited
    if manifest_path in visited:
        raise ValueError(f"cyclic stage-manifest dependency: {manifest_path}")
    visited.add(manifest_path)
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "stage-manifest-v1":
        raise ValueError(f"unsupported stage manifest: {manifest_path}")
    for collection in ("inputs", "outputs"):
        references = manifest.get(collection)
        if not isinstance(references, dict):
            raise ValueError(f"stage manifest has no {collection}: {manifest_path}")
        for reference in references.values():
            verified = verify_ref(reference, relative_to=run_dir)
            if collection == "inputs" and verified.name == "stage_manifest.json":
                verify_stage_manifest(verified, run_dir=run_dir, _visited=visited)
    visited.remove(manifest_path)
    return manifest


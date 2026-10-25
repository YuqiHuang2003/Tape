#!/usr/bin/env python3
"""Download the pinned ERNIE checkpoint and install the recorded compatibility code."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from dsai6707.io import read_json, sha256_file, write_json  # noqa: E402

REVISION = "23c6cb79af127ec966c9c03506dc80a3ff9fa953"
OVERLAY = (
    "config.json",
    "configuration_ernie4_5_moe.py",
    "modeling_ernie4_5_moe.py",
    "tokenization_ernie4_5.py",
    "tokenizer_config.json",
)


def prepare(destination: Path) -> None:
    from huggingface_hub import snapshot_download

    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("choose an empty checkpoint directory; existing files are preserved")
    snapshot_download(
        repo_id="baidu/ERNIE-4.5-21B-A3B-PT",
        revision=REVISION,
        local_dir=str(destination),
        allow_patterns=["*.safetensors", "*.json", "*.py", "*.model", "LICENSE"],
    )
    vendor = ROOT / "vendor/ernie"
    index = read_json(destination / "model.safetensors.index.json")
    for name in set(index["weight_map"].values()):
        if not (destination / name).is_file():
            raise FileNotFoundError(f"missing downloaded weight shard: {name}")
    original = destination / "upstream_compatibility_sources"
    original.mkdir(exist_ok=False)
    for name in OVERLAY:
        path = destination / name
        if path.is_file():
            shutil.copyfile(path, original / name)
        shutil.copyfile(vendor / name, path)
    write_json(
        destination / "tape_compatibility.json",
        {
            "upstream_revision": REVISION,
            "overlay_sha256": {name: sha256_file(destination / name) for name in OVERLAY},
            "weights_modified": False,
        },
    )
    print(f"Prepared checkpoint: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.destination.resolve())


if __name__ == "__main__":
    main()

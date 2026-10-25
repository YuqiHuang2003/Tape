"""Deterministic, non-overlapping WikiText-2 calibration preparation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from .io import sha256_file, write_json


@dataclass(frozen=True)
class TokenBundle:
    statistics: np.ndarray
    fitness: np.ndarray
    validation: np.ndarray


def non_overlapping_blocks(
    token_ids: Sequence[int] | np.ndarray,
    *,
    block_length: int,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose full, non-overlapping blocks and return blocks plus source starts."""

    tokens = np.asarray(token_ids, dtype=np.int64)
    if tokens.ndim != 1 or block_length < 2 or count < 1:
        raise ValueError("invalid token block request")
    available = tokens.size // block_length
    if count > available:
        raise ValueError(f"requested {count} blocks but only {available} are available")
    rng = np.random.default_rng(seed)
    block_ids = rng.permutation(available)[:count]
    starts = block_ids * block_length
    blocks = np.stack([tokens[start : start + block_length] for start in starts])
    return blocks, starts


def _dataset_tokens(tokenizer: Any, rows: Sequence[str]) -> np.ndarray:
    separator = tokenizer.eos_token or "\n"
    text = separator.join(value for value in rows if value and value.strip())
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    return np.asarray(ids, dtype=np.int64)


def loss_blocks_required(loss_tokens: int, sequence_length: int) -> int:
    """Return full causal blocks needed for an exact number of next-token losses."""

    if loss_tokens <= 0 or sequence_length < 2:
        raise ValueError("loss_tokens must be positive and sequence_length at least two")
    return int(np.ceil(loss_tokens / (sequence_length - 1)))


def prepare_wikitext(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Download pinned data/tokenizer and save immutable token arrays plus provenance."""

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install the 'experiment' dependencies before preparing data") from exc

    model_cfg = config["model"]
    data_cfg = config["data"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["id"], revision=model_cfg["tokenizer_revision"], use_fast=True
    )
    common = {
        "path": data_cfg["dataset"],
        "name": data_cfg["config"],
        "revision": data_cfg["revision"],
    }
    train = load_dataset(**common, split="train")
    validation = load_dataset(**common, split="validation")
    train_tokens = _dataset_tokens(tokenizer, train["text"])
    validation_tokens = _dataset_tokens(tokenizer, validation["text"])

    sequence_length = int(data_cfg["sequence_length"])
    statistics_count = int(data_cfg["statistics_segments"])
    # A length-S causal block contributes S-1 next-token loss positions.
    fitness_count = loss_blocks_required(
        int(data_cfg["fitness_tokens"]), sequence_length
    )
    train_blocks, train_starts = non_overlapping_blocks(
        train_tokens,
        block_length=sequence_length,
        count=statistics_count + fitness_count,
        seed=int(data_cfg["seed"]),
    )
    validation_count = loss_blocks_required(
        int(data_cfg["validation_tokens"]), sequence_length
    )
    validation_blocks, validation_starts = non_overlapping_blocks(
        validation_tokens,
        block_length=sequence_length,
        count=validation_count,
        seed=int(data_cfg["seed"]) + 1,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    token_path = destination / "tokens.npz"
    np.savez_compressed(
        token_path,
        statistics=train_blocks[:statistics_count],
        fitness=train_blocks[statistics_count:],
        validation=validation_blocks,
        statistics_starts=train_starts[:statistics_count],
        fitness_starts=train_starts[statistics_count:],
        validation_starts=validation_starts,
    )
    manifest = {
        "schema_version": "token-data-v1",
        "dataset": {
            "id": data_cfg["dataset"],
            "config": data_cfg["config"],
            "revision": data_cfg["revision"],
            "train_split": "train",
            "validation_split": "validation",
        },
        "tokenizer": {
            "id": model_cfg["id"],
            "revision": model_cfg["tokenizer_revision"],
        },
        "seed": int(data_cfg["seed"]),
        "sequence_length": sequence_length,
        "statistics_segments": statistics_count,
        "fitness_loss_tokens": int(data_cfg["fitness_tokens"]),
        "validation_loss_tokens": int(data_cfg["validation_tokens"]),
        "statistics_starts": train_starts[:statistics_count].tolist(),
        "fitness_starts": train_starts[statistics_count:].tolist(),
        "validation_starts": validation_starts.tolist(),
        "tokens_sha256": sha256_file(token_path),
        "sampling_fingerprint": sha256(
            np.concatenate((train_starts, validation_starts)).astype("<i8").tobytes()
        ).hexdigest(),
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def load_token_bundle(path: str | Path) -> TokenBundle:
    with np.load(path, allow_pickle=False) as payload:
        return TokenBundle(
            statistics=payload["statistics"].astype(np.int64, copy=False),
            fitness=payload["fitness"].astype(np.int64, copy=False),
            validation=payload["validation"].astype(np.int64, copy=False),
        )


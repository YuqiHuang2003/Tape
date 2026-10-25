"""Validate the local vectorized ERNIE MoE dispatch against its reference loop."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = (
    ROOT
    / "external/reap-official/artifacts/models/ERNIE-4.5-21B-A3B-PT"
    / "modeling_ernie4_5_moe.py"
)
SOURCE_FILE = (
    ROOT / "external/reap-official/src/reap/models/modeling_ernie4_5_moe.py"
)


def _load_patched_method():
    tree = ast.parse(MODEL_FILE.read_text(encoding="utf-8"))
    method = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Ernie4_5_MoeMLP":
            method = next(
                child
                for child in node.body
                if isinstance(child, ast.FunctionDef)
                and child.name == "moe_gate_dispatch"
            )
            break
    if method is None:
        raise RuntimeError("Could not find Ernie4_5_MoeMLP.moe_gate_dispatch")
    holder = ast.ClassDef(
        name="DispatchHolder",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[holder], type_ignores=[]))
    namespace = {"torch": torch, "Optional": Optional, "Tuple": Tuple}
    exec(compile(module, str(MODEL_FILE), "exec"), namespace)
    return namespace["DispatchHolder"].moe_gate_dispatch


def _reference(x: torch.Tensor, logits: torch.Tensor, k: int, capacity: int):
    tokens, hidden = x.shape
    experts = logits.shape[1]
    topk_prob, expert_id = torch.topk(logits, k, dim=-1)
    combine_weights = topk_prob.clone()
    dispatched = x.new_zeros((experts, capacity, hidden))
    scatter = x.new_full((k, tokens), -1, dtype=torch.int32)
    counters = torch.zeros(experts, dtype=torch.int32, device=x.device)
    for token in range(tokens):
        for route in range(k):
            expert = expert_id[token, route].item()
            slot = counters[expert].item()
            if slot >= capacity:
                combine_weights[token, route] = 0.0
                continue
            scatter[route, token] = expert * capacity + slot
            dispatched[expert, slot] = x[token]
            counters[expert] += 1
    offsets = torch.cumsum(counters, dim=0, dtype=torch.int64)
    return dispatched, combine_weights, scatter, offsets, expert_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tokens", type=int, default=257)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=7)
    args = parser.parse_args()

    if _sha256(MODEL_FILE) != _sha256(SOURCE_FILE):
        raise AssertionError("Patched model sources differ")

    device = torch.device(args.device)
    patched = _load_patched_method()
    results = []
    for seed in range(5):
        generator = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(
            args.tokens,
            args.hidden,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        logits = torch.randn(
            args.tokens,
            args.experts,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        for capacity in (max(1, args.tokens // 32), args.tokens):
            expected = _reference(x, logits, args.top_k, capacity)
            actual = patched(None, x, logits, args.top_k, capacity)
            equality = [torch.equal(left, right) for left, right in zip(expected, actual)]
            if not all(equality):
                raise AssertionError(
                    f"dispatch mismatch: seed={seed}, capacity={capacity}, "
                    f"outputs={equality}"
                )
            results.append(
                {"seed": seed, "capacity": capacity, "outputs_equal": equality}
            )

    print(
        json.dumps(
            {
                "device": str(device),
                "tokens": args.tokens,
                "experts": args.experts,
                "top_k": args.top_k,
                "cases": len(results),
                "all_equal": True,
                "model_source_sha256": _sha256(MODEL_FILE),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
# Adapter contract adapted from Pika-Labs/Pika-H3-Flex, revision
# 0ddf55ad7067ca1e76812253bd7b4600df87f759.
"""Pika H3 Flex adapter loading on the canonical MiniMax-H3 DiT.

Pika publishes PEFT LoRA adapters against Diffusers' unfused H3 names.  The
canonical WorldFoundry model uses the already-converted fused QKV and SwiGLU
projections, so this loader converts and merges the adapter deltas without
copying the transformer or introducing a second H3 implementation.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_LORA_SUFFIXES = (
    (".lora_A.default.weight", "A"),
    (".lora_B.default.weight", "B"),
    (".lora_A.weight", "A"),
    (".lora_B.weight", "B"),
)


def _strip_adapter_prefix(key: str) -> str:
    """Normalize the PEFT wrapper prefixes used by released adapter files."""

    for prefix in ("base_model.model.", "model.", "transformer."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _target_spec(source: str) -> tuple[str, str | None, bool] | None:
    """Map one Diffusers linear name to a canonical target and optional QKV slice."""

    source = source.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
    source = source.replace("transformer_blocks.", "blocks.")
    qkv_targets = {
        ".attn.to_q": "q",
        ".attn.to_k": "k",
        ".attn.to_v": "v",
    }
    for suffix, logical in qkv_targets.items():
        if source.endswith(suffix):
            return source[: -len(suffix)] + ".attn.qkv_proj", logical, False
    if source.endswith(".attn.to_out.0"):
        return source[: -len(".attn.to_out.0")] + ".attn.out_proj", None, False
    if source.endswith(".ff.net.0.proj"):
        return source[: -len(".ff.net.0.proj")] + ".mlp.fc1", None, True
    if source.endswith(".ff.net.2"):
        return source[: -len(".ff.net.2")] + ".mlp.fc2", None, False
    return None


def _read_lora_groups(path: Path) -> dict[str, dict[str, torch.Tensor]]:
    """Read adapter tensors grouped by their Diffusers target module."""

    from safetensors.torch import load_file

    groups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for raw_key, value in load_file(str(path), device="cpu").items():
        key = _strip_adapter_prefix(raw_key)
        for suffix, side in _LORA_SUFFIXES:
            if key.endswith(suffix):
                groups[key[: -len(suffix)]][side] = value
                break
        else:
            raise ValueError(f"Unsupported Pika H3 Flex adapter tensor: {raw_key}")
    return dict(groups)


def load_pika_h3_flex_adapter(
    model: nn.Module,
    path: str | Path,
    *,
    rank: int = 128,
    alpha: int | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Merge one released Pika H3 Flex LoRA adapter into a canonical DiT.

    Args:
        model: Loaded :class:`MiniMaxH3DiTModel` instance.
        path: ``adapter.safetensors`` from the ``fl2va`` or ``ref2va`` release.
        rank: Published LoRA rank.
        alpha: Published LoRA alpha; defaults to ``rank``.
        strict: Reject unsupported targets and incomplete LoRA pairs.

    Returns:
        A small loading report containing the merged target count and names.
    """

    adapter_path = Path(path).expanduser().resolve()
    if not adapter_path.is_file():
        raise FileNotFoundError(f"Pika H3 Flex adapter not found: {adapter_path}")
    if isinstance(rank, bool) or int(rank) <= 0:
        raise ValueError("Pika H3 Flex LoRA rank must be positive")
    rank = int(rank)
    alpha = rank if alpha is None else int(alpha)
    if alpha <= 0:
        raise ValueError("Pika H3 Flex LoRA alpha must be positive")

    merged: list[str] = []
    for source, pair in _read_lora_groups(adapter_path).items():
        target_spec = _target_spec(source)
        if target_spec is None:
            if strict:
                raise ValueError(f"Unsupported Pika H3 Flex LoRA target: {source}")
            continue
        if set(pair) != {"A", "B"}:
            raise ValueError(f"Incomplete Pika H3 Flex LoRA pair for {source}: {sorted(pair)}")
        target, logical, swap_halves = target_spec
        module = model.get_submodule(target)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Pika H3 Flex LoRA target is not linear: {target}")
        matrix_a, matrix_b = pair["A"], pair["B"]
        if matrix_a.ndim != 2 or matrix_b.ndim != 2:
            raise ValueError(f"Pika H3 Flex LoRA tensors must be matrices: {source}")
        if int(matrix_a.shape[0]) != rank or int(matrix_b.shape[1]) != rank:
            raise ValueError(
                f"Pika H3 Flex LoRA rank mismatch for {source}: "
                f"A={tuple(matrix_a.shape)} B={tuple(matrix_b.shape)} rank={rank}"
            )
        delta = (matrix_b.float() @ matrix_a.float()) * (float(alpha) / float(rank))
        if swap_halves:
            if delta.shape[0] % 2:
                raise ValueError(f"Fused SwiGLU adapter output is not divisible by two: {source}")
            delta = torch.cat(delta.chunk(2, dim=0)[::-1], dim=0)

        if logical is None:
            expected = tuple(module.weight.shape)
            if tuple(delta.shape) != expected:
                raise ValueError(f"Pika H3 Flex delta shape {tuple(delta.shape)} != {expected} for {target}")
            with torch.no_grad():
                module.weight.add_(delta.to(device=module.weight.device, dtype=module.weight.dtype))
        else:
            inner = int(module.weight.shape[0]) // 3
            if tuple(delta.shape) != (inner, int(module.weight.shape[1])):
                raise ValueError(f"Pika H3 Flex QKV delta shape {tuple(delta.shape)} is incompatible with {target}")
            offsets = {"q": 0, "k": inner, "v": 2 * inner}
            start = offsets[logical]
            with torch.no_grad():
                module.weight[start : start + inner].add_(
                    delta.to(device=module.weight.device, dtype=module.weight.dtype)
                )
        merged.append(source)

    if not merged:
        raise ValueError(f"Pika H3 Flex adapter contains no supported LoRA targets: {adapter_path}")
    return {"path": str(adapter_path), "rank": rank, "alpha": alpha, "merged_targets": tuple(merged)}


__all__ = ["load_pika_h3_flex_adapter"]

"""Reproduce H-RDT GQA and LongCat cached-query component comparisons.

Legacy forwards freeze the pre-optimization equations. Each CLI invocation
measures one real attention module with seeded random weights in a fresh process.
The mode explicitly switches the module optimization off or on, overriding env defaults.
This is not a pretrained-model quality test or an end-to-end FPS benchmark.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def legacy_hrdt(layer, values, condition=None, mask=None):
    """H-RDT's original explicit-head expansion, including output projection."""
    from worldfoundry.synthesis.action_generation.h_rdt.modeling import (
        _attention_backends,
        _repeat_kv,
        scaled_dot_product_attention,
    )

    batch, sequence, _ = values.shape
    condition = values if condition is None else condition
    key_length = condition.shape[1]
    query = layer.wq(values).view(batch, sequence, layer.n_heads, layer.head_size)
    key_value = layer.wkv(condition).view(batch, key_length, layer.n_kv_heads, layer.head_size, 2)
    key, value = key_value.unbind(-1)
    query = layer.norm_q(query).transpose(1, 2)
    key = _repeat_kv(layer.norm_k(key), layer.n_rep).transpose(1, 2)
    value = _repeat_kv(value, layer.n_rep).transpose(1, 2)
    attention_mask = None
    if mask is not None:
        attention_mask = mask.to(torch.bool).reshape(batch, 1, 1, key_length).expand(-1, -1, sequence, -1)
    output = scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=layer.attn_scale,
        backends=_attention_backends(layer.attention_backend, query.device, has_mask=mask is not None),
    )
    return layer.wo(output.transpose(1, 2).contiguous().view(batch, sequence, -1))


def legacy_longcat(layer, hidden, shape, num_cond_latents, kv_cache):
    """LongCat's original batch repeats and full-length padded-query RoPE."""
    batch, sequence, dim = hidden.shape
    qkv = layer.qkv(hidden).view(batch, sequence, 3, layer.num_heads, layer.head_dim).permute(2, 0, 3, 1, 4)
    query, key, value = qkv.unbind(0)
    query, key = layer.q_norm(query), layer.k_norm(key)
    history_key, history_value = kv_cache
    if history_key.shape[0] == 1:
        history_key = history_key.repeat(batch, 1, 1, 1)
        history_value = history_value.repeat(batch, 1, 1, 1)
    full_key = torch.cat((history_key, key), dim=2).contiguous()
    full_value = torch.cat((history_value, value), dim=2).contiguous()
    padded_query = torch.cat((torch.empty_like(history_key), query), dim=2).contiguous()
    frames, height, width = shape
    padded_query, full_key = layer.rope_3d(padded_query, full_key, (frames + num_cond_latents, height, width))
    query = padded_query[:, :, -sequence:].contiguous()
    output = layer._process_attn(query, full_key, full_value, shape)
    return layer.proj(output.transpose(1, 2).reshape(batch, sequence, dim))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("h-rdt-self", "h-rdt-cross", "longcat"), required=True)
    parser.add_argument("--mode", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--backend", choices=("auto", "flash", "efficient", "math"), default="efficient")
    parser.add_argument("--mask", choices=("none", "padding"), default="none")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--query-tokens", type=int, default=256)
    parser.add_argument("--history-tokens", type=int, default=4096)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.batch, args.heads, args.kv_heads, args.head_dim, args.query_tokens, args.history_tokens, args.rounds)
        < 1
    ):
        parser.error("dimensions and rounds must be positive")
    if args.heads % args.kv_heads:
        parser.error("heads must be divisible by kv-heads")
    if args.mask != "none" and args.model != "h-rdt-cross":
        parser.error("padding masks require h-rdt-cross")
    if args.model == "longcat" and (args.head_dim % 8 or args.history_tokens % args.query_tokens):
        parser.error("LongCat requires head-dim divisible by 8 and history an integer number of query-sized frames")
    torch.manual_seed(29)
    device, dtype = "cuda", getattr(torch, args.dtype)
    dim = args.heads * args.head_dim
    hidden = torch.randn(args.batch, args.query_tokens, dim, device=device, dtype=dtype)
    if args.model == "longcat":
        from worldfoundry.synthesis.visual_generation.longcat_video.longcat_video_runtime.longcat_video.modules.attention import (
            Attention,
        )

        layer = (
            Attention(
                dim,
                args.heads,
                enable_flashattn3=True,
                cp_split_hw=(1, 1),
                enable_kv_optimization=args.mode == "optimized",
            )
            .to(device, dtype)
            .eval()
        )
        # Benchmark the existing FA3-unavailable SDPA branch in this environment.
        # Verify the actual selected kernel using the profiler metadata below.
        cache = tuple(
            torch.randn(1, args.heads, args.history_tokens, args.head_dim, device=device, dtype=dtype) for _ in range(2)
        )
        shape = (1, 1, args.query_tokens)
        frames = args.history_tokens // args.query_tokens

        def operation():
            return layer.forward_with_kv_cache(hidden, shape=shape, num_cond_latents=frames, kv_cache=cache)

    else:
        from worldfoundry.synthesis.action_generation.h_rdt.modeling import Attention, CrossAttention

        config = dict(hidden_size=dim, num_heads=args.heads, num_kv_heads=args.kv_heads, norm_eps=1e-6)
        cls = Attention if args.model == "h-rdt-self" else CrossAttention
        layer = (
            cls(config, attention_backend=args.backend, enable_gqa_optimization=args.mode == "optimized")
            .to(device, dtype)
            .eval()
        )
        condition = (
            torch.randn(args.batch, args.history_tokens, dim, device=device, dtype=dtype)
            if args.model == "h-rdt-cross"
            else None
        )
        mask = None
        if args.mask == "padding":
            mask = torch.ones(args.batch, args.history_tokens, device=device, dtype=torch.bool)
            mask[:, args.history_tokens * 3 // 4 :] = False

        def operation():
            return layer(hidden) if condition is None else layer(hidden, condition, mask)

    for _ in range(10):
        operation()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    steady = torch.cuda.memory_allocated()
    cuda_ms, wall_ms = [], []
    for _ in range(args.rounds):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        operation()
        end.record()
        end.synchronize()
        cuda_ms.append(start.elapsed_time(end))
        wall_ms.append((time.perf_counter() - wall) * 1000)
    peak = torch.cuda.max_memory_allocated()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        operation()
        torch.cuda.synchronize()
    events = {item.key: item.count for item in profile.key_averages()}
    result = {
        "model": args.model,
        "mode": args.mode,
        "optimization_enabled": (
            layer.enable_kv_optimization if args.model == "longcat" else layer.enable_gqa_optimization
        ),
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "real attention module with random weights; no pretrained model or end-to-end FPS",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": args.dtype,
        "mask": args.mask,
        "batch": args.batch,
        "heads": args.heads,
        "kv_heads": args.heads if args.model == "longcat" else args.kv_heads,
        "head_dim": args.head_dim,
        "query_tokens": args.query_tokens,
        "history_tokens": args.history_tokens,
        "warmups": 10,
        "rounds": args.rounds,
        "median_cuda_ms": statistics.median(cuda_ms),
        "p90_cuda_ms": sorted(cuda_ms)[int(0.9 * (len(cuda_ms) - 1))],
        "median_wall_ms": statistics.median(wall_ms),
        "steady_allocated_bytes": steady,
        "peak_allocated_bytes": peak,
        "transient_peak_bytes": peak - steady,
        "cat_calls_per_step": events.get("aten::cat", 0),
        "repeat_calls_per_step": events.get("aten::repeat", 0) + events.get("aten::repeat_interleave", 0),
        "attention_ops": [key for key in events if "attention" in key],
        "cuda_samples_ms": cuda_ms,
        "wall_samples_ms": wall_ms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("samples_ms")}, indent=2))


if __name__ == "__main__":
    main()

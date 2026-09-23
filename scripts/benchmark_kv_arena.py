"""Compare concatenation with KV arena in fresh CUDA processes.

The Evoke cases execute the real attention module with seeded random weights,
including projections, normalization, RoPE, cached history and output projection.
They are component benchmarks, not pretrained-model quality or rollout FPS.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from worldfoundry.core.attention.kv_arena import KVSegmentArena


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("cat", "arena", "evoke-cat", "evoke-arena"), required=True)
    parser.add_argument("--history", type=int, default=4096)
    parser.add_argument("--current", type=int, default=256)
    parser.add_argument("--memory", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--layers", type=int, default=1, help="Number of sequential Evoke attention layers")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--phase", choices=("cached", "cache-build"), default="cached")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(args.history, args.current, args.heads, args.head_dim, args.batch, args.layers, args.rounds) < 1
        or args.memory < 0
    ):
        parser.error("dimensions and rounds must be positive; memory may be zero")
    if args.head_dim % 2:
        parser.error("head-dim must be even for RoPE")
    if args.phase == "cache-build" and not args.case.startswith("evoke"):
        parser.error("cache-build phase requires an Evoke case")
    if args.layers != 1 and not args.case.startswith("evoke"):
        parser.error("multiple layers require an Evoke case")
    torch.manual_seed(19)
    device, dtype = "cuda", getattr(torch, args.dtype)
    arena = None
    if args.case.startswith("evoke"):
        from worldfoundry.base_models.diffusion_model.models.networks.evoke.model import (
            EvokeAttention,
            EvokeAttnProcessor,
        )

        models = []
        for _ in range(args.layers):
            model = (
                EvokeAttention(
                    dim=args.heads * args.head_dim,
                    heads=args.heads,
                    dim_head=args.head_dim,
                    processor=EvokeAttnProcessor(),
                    restrict_self_attn=True,
                )
                .to(device=device, dtype=dtype)
                .eval()
            )
            model.processor.enable_cache(use_arena=args.case == "evoke-arena")
            models.append(model)
        length = args.history + args.current
        x = torch.randn(args.batch, length, args.heads * args.head_dim, device=device, dtype=dtype)
        phase = torch.arange(length, device=device)[:, None] * torch.linspace(
            0.01, 0.2, args.head_dim // 2, device=device
        )
        phase = phase.repeat_interleave(2, -1)
        rope = torch.cat((phase.cos(), phase.sin()), -1)[None].expand(args.batch, -1, -1)
        options = dict(
            rotary_emb=rope, original_context_length=args.current, original_context_length_list=[args.current]
        )

        def forward(first):
            hidden = x
            for model in models:
                hidden = model(hidden, is_first_denoising_step=first, **options)
            return hidden

        forward(True)
        arena_bytes = sum(
            model.processor.kv_cache["arena"].nbytes for model in models if "arena" in model.processor.kv_cache
        )

        def operation():
            return forward(args.phase == "cache-build")

        scope = (
            f"{args.layers} real Evoke attention layer(s); random weights; {args.phase} step; no MLP/residual blocks"
        )
    else:

        def pair(tokens):
            return tuple(
                torch.randn(args.batch, args.heads, tokens, args.head_dim, device=device, dtype=dtype) for _ in range(2)
            )

        static, current, memory = pair(args.history), pair(args.current), pair(args.memory)
        query = torch.randn_like(current[0])
        if args.case == "arena":
            arena = KVSegmentArena(static, current_tokens=args.current, memory=memory)
            # Transfer ownership, so arena doesn't duplicate persistent KV storage.
            static, memory = arena.static, arena.memory
        arena_bytes = arena.nbytes if arena else 0

        def operation():
            if arena is None:
                kv = tuple(torch.cat(parts, dim=2) for parts in zip(static, current, memory))
            else:
                arena.validate_persistent(static, memory)
                kv = arena.stage_current(current)
            return F.scaled_dot_product_attention(query, *kv)

        scope = "segmented KV assembly plus dense SDPA"
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
        wall_ms.append((time.perf_counter() - wall) * 1000)
        cuda_ms.append(start.elapsed_time(end))
    peak = torch.cuda.max_memory_allocated()
    # Profile separately so instrumentation allocations don't contaminate timing/memory.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        operation()
        torch.cuda.synchronize()
    cats = sum(item.count for item in profile.key_averages() if item.key == "aten::cat")
    payload = {
        "case": args.case,
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": scope,
        "phase": args.phase,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": args.dtype,
        "batch": args.batch,
        "layers": args.layers,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "history_tokens": args.history,
        "current_tokens": args.current,
        "memory_tokens": 0 if args.case.startswith("evoke") else args.memory,
        "warmups": 10,
        "rounds": args.rounds,
        "median_cuda_ms": statistics.median(cuda_ms),
        "p90_cuda_ms": sorted(cuda_ms)[int(0.9 * (len(cuda_ms) - 1))],
        "median_wall_ms": statistics.median(wall_ms),
        "steady_allocated_bytes": steady,
        "peak_allocated_bytes": peak,
        "transient_peak_bytes": peak - steady,
        "cat_calls_per_step": cats,
        "arena_bytes": arena_bytes,
        "cuda_samples_ms": cuda_ms,
        "wall_samples_ms": wall_ms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if not k.endswith("samples_ms")}, indent=2))


if __name__ == "__main__":
    main()

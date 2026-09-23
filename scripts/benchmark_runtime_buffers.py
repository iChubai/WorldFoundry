"""Small CUDA A/B probes; run each case in a fresh process, not as model FPS.

python -m scripts.benchmark_runtime_buffers --case offload-packed
python -m scripts.benchmark_runtime_buffers --case offload-legacy
python -m scripts.benchmark_runtime_buffers --case gqa-view
python -m scripts.benchmark_runtime_buffers --case gqa-copy
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch import nn

from worldfoundry.core.attention.native import scaled_dot_product_attention
from worldfoundry.core.vram.layerwise_offload import enable_layerwise_cpu_offload


class _Stack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(1024, 1024) for _ in range(8)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x).tanh()
        return x


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=("offload-packed", "offload-legacy", "gqa-view", "gqa-copy"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rounds", type=int, default=50)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    torch.manual_seed(7)
    handle = None
    if args.case.startswith("offload"):
        model = _Stack().to(device="cuda", dtype=torch.float16).eval()
        x = torch.randn(64, 1024, device="cuda", dtype=torch.float16)
        handle = enable_layerwise_cpu_offload(model, reuse_buffers=args.case == "offload-packed")

        def operation():
            return model(x)

        workload = {"layers": 8, "width": 1024, "batch": 64}
    else:
        q = torch.randn(1, 32, 256, 64, device="cuda", dtype=torch.float16)
        k = torch.randn(1, 8, 8192, 64, device="cuda", dtype=torch.float16)
        v = torch.randn_like(k)

        def operation():
            if args.case == "gqa-view":
                return scaled_dot_product_attention(q, k, v, enable_gqa=True, backend="efficient")
            return scaled_dot_product_attention(
                q, k.repeat_interleave(4, 1), v.repeat_interleave(4, 1), backend="efficient"
            )

        workload = {"q": list(q.shape), "kv": list(k.shape)}
    try:
        for _ in range(10):
            operation()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(args.rounds):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            operation()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        payload = {
            "case": args.case,
            "scope": "synthetic component CUDA timing; excludes setup and teardown",
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": "float16",
            "workload": workload,
            "warmups": 10,
            "rounds": args.rounds,
            "median_ms": statistics.median(samples),
            "p90_ms": sorted(samples)[int(0.9 * (len(samples) - 1))],
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "samples_ms": samples,
        }
        if handle is not None:
            payload["offload"] = handle.report()
        rendered = json.dumps(payload, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n")
        print(rendered)
    finally:
        if handle is not None:
            handle.disable()


if __name__ == "__main__":
    main()

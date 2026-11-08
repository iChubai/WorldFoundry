from typing import Optional

import torch


def log_normal_sample(
    x: torch.Tensor, generator: torch.Generator | None = None, m: float = 0.0, s: float = 1.0
) -> torch.Tensor:
    bs = x.shape[0]
    s = torch.randn(bs, device=x.device, generator=generator) * s + m
    return torch.sigmoid(s)

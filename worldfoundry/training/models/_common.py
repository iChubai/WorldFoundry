"""Shared building blocks for native diffusion training adapters.

Wan, SANA, and the flow-velocity family adapters all freeze their codec and
conditioner, translate effective flow sigmas onto a model timestep scale, and
call the ordinary denoiser graph with autograd enabled.  The small, byte
identical helpers that support that pattern live here so each family adapter
keeps only its own conditioning, latent geometry, and metadata logic.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


def component_module(component: object, *names: str) -> nn.Module | None:
    """Return the first ``nn.Module`` a component exposes under ``names``."""

    if isinstance(component, nn.Module):
        return component
    for name in names:
        value = getattr(component, name, None)
        if isinstance(value, nn.Module):
            return value
    return None


def module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    """Return the module's parameter device and floating-point compute dtype."""

    reference = next(module.parameters(), None)
    if reference is None:
        reference = next(module.buffers(), None)
    if reference is None:
        return torch.device("cpu"), torch.float32
    dtype = reference.dtype if reference.is_floating_point() else torch.float32
    return reference.device, dtype


def freeze(module: nn.Module | None) -> None:
    """Disable gradients and place a frozen role module in eval mode."""

    if module is None:
        return
    module.requires_grad_(False)
    module.eval()


def merge_without_overwrite(
    destination: dict[str, object],
    source: Mapping[str, object],
    *,
    source_name: str,
    family: str,
) -> None:
    """Merge ``source`` into ``destination`` and reject key collisions."""

    overlap = sorted(set(destination) & set(source))
    if overlap:
        raise ValueError(f"{source_name} collides with encoded {family} conditioning keys: {overlap}")
    destination.update(source)


def positive_int(value: object, *, field_name: str) -> int:
    """Coerce ``value`` to a strictly positive integer, rejecting bools."""

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, not bool")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{field_name} must be positive")
    return resolved


def normalize_effective_sigmas(
    sigmas: object,
    *,
    batch_size: int,
    device: torch.device,
    family: str,
) -> torch.Tensor:
    """Validate and broadcast one effective flow sigma per sample.

    Effective sigmas parameterize the flow-matching corruption and must stay in
    ``[0, 1]``; a scalar broadcasts across the batch.
    """

    if not isinstance(sigmas, torch.Tensor):
        raise TypeError(f"{family} objective sigmas must be a torch.Tensor")
    resolved = sigmas.to(device=device, dtype=torch.float32)
    if resolved.ndim == 0:
        resolved = resolved.expand(batch_size)
    elif resolved.numel() == batch_size:
        resolved = resolved.reshape(batch_size)
    else:
        raise ValueError(f"{family} requires one effective sigma per sample")
    if not bool(torch.isfinite(resolved).all()) or not bool(((0 <= resolved) & (resolved <= 1)).all()):
        raise ValueError(f"{family} effective sigmas must be finite values in [0, 1]")
    return resolved


__all__ = [
    "component_module",
    "freeze",
    "merge_without_overwrite",
    "module_device_dtype",
    "normalize_effective_sigmas",
    "positive_int",
]

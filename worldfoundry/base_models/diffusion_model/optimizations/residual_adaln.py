"""Opt-in Wan cross-attention residual/AdaLN fusion with execution receipts.

Enable on a loaded denoiser using ``enable_residual_adaln_fusion(model)``.
The original path remains the default until workload-level validation is
complete. Partial/phase-cache, training and mutating residual calls retain
their original paths. This module does not change global kernel settings.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from worldfoundry.core.kernels import residual_layer_norm_scale_shift
from worldfoundry.core.kernels.registry import kernel_dispatch_receipt_scope


@dataclass
class ResidualAdaLNRuntime:
    """Request-resettable counts distinguish fused execution from fallbacks."""

    blocks: int = 0
    calls: int = 0
    accelerated_calls: int = 0
    fallback_calls: int = 0

    def reset(self):
        self.calls = self.accelerated_calls = self.fallback_calls = 0

    def report(self):
        return {"blocks": self.blocks, "calls": self.calls, "accelerated_calls": self.accelerated_calls, "fallback_calls": self.fallback_calls}

    def apply(self, residual, update, scale, shift, eps):
        receipt = {}
        with kernel_dispatch_receipt_scope(receipt):
            outputs = residual_layer_norm_scale_shift(residual, update, scale, shift, eps)
        self.calls += 1
        accelerated = any(row["op"] == "residual_layer_norm_scale_shift" and row["accelerated"] for row in receipt.get("dispatches", []))
        self.accelerated_calls += int(accelerated)
        self.fallback_calls += int(not accelerated)
        return outputs


def enable_residual_adaln_fusion(model: nn.Module, enabled: bool = True) -> ResidualAdaLNRuntime:
    """Enable/disable supported Wan blocks, returning a shared runtime counter.

    Unsupported models fail before modifying any module. Re-enabling replaces
    the counters; disable restores the original block path.
    """
    from worldfoundry.base_models.diffusion_model.models.networks.wan.model import DiTBlock

    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a bool")
    blocks = [module for module in model.modules() if type(module) is DiTBlock]
    if not blocks:
        raise ValueError("residual AdaLN fusion requires native Wan DiTBlock modules")
    runtime = ResidualAdaLNRuntime(blocks=len(blocks))
    for block in blocks:
        block._worldfoundry_residual_adaln_runtime = runtime if enabled else None
    model._worldfoundry_residual_adaln_runtime = runtime if enabled else None
    return runtime


__all__ = ["ResidualAdaLNRuntime", "enable_residual_adaln_fusion"]

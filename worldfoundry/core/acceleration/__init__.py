"""Opt-in inference approximations and acceleration reporting.

- cache: cross-step residual/prediction reuse.
- token_pruning: token selection and reconstruction.
- quantization: low-precision Linear execution and FP8/NVFP4 packing.
- technology: inventory of active acceleration families.
- mha_schedule: projection fusion and precision schedules for resident attention.

CUDA graph dispatch, frame prefetch, overlap and prewarm are owned by
core.execution. One-shot encoder placement is owned by core.vram.
"""

# ──────────────────────────────────────────────────────────────────────────
# Re-exports — keep ``from worldfoundry.core.acceleration import X`` stable
# ──────────────────────────────────────────────────────────────────────────

from worldfoundry.core.acceleration.cache import (
    AdaCacheResidualCache,
    AdaptiveResidualCache,
    BlockTaylorSeerCache,
    CustomTaylorResidualCache,
    FixedStepCache,
    MagCacheResidualCache,
    TaylorSeerResidualCache,
    TeaCacheResidualCache,
)
from worldfoundry.core.acceleration.quantization.linear import (
    Float8Linear,
    WeightOnlyLinear,
    quantization_runtime_report,
    replace_linear_with_float8,
    replace_linear_with_weight_only,
    reset_quantization_runtime_window,
    set_low_precision_enabled,
)
from worldfoundry.core.acceleration.quantization.nvfp4 import (
    NVFP4Linear,
    dequantize_nvfp4,
    quantize_nvfp4,
    replace_linear_with_nvfp4,
)
from worldfoundry.core.acceleration.technology import (
    AccelerationTechnology,
    acceleration_technology_report,
)
from worldfoundry.core.acceleration.token_pruning import (
    TokenPruner,
    TokenPruneState,
    prune_tokens,
    restore_tokens,
    select_token_indices,
)

__all__ = [
    "AccelerationTechnology",
    "AdaCacheResidualCache",
    "AdaptiveResidualCache",
    "BlockTaylorSeerCache",
    "CustomTaylorResidualCache",
    "FixedStepCache",
    "Float8Linear",
    "MagCacheResidualCache",
    "NVFP4Linear",
    "TaylorSeerResidualCache",
    "TeaCacheResidualCache",
    "TokenPruneState",
    "TokenPruner",
    "WeightOnlyLinear",
    "acceleration_technology_report",
    "dequantize_nvfp4",
    "prune_tokens",
    "quantization_runtime_report",
    "quantize_nvfp4",
    "replace_linear_with_float8",
    "replace_linear_with_nvfp4",
    "replace_linear_with_weight_only",
    "reset_quantization_runtime_window",
    "restore_tokens",
    "select_token_indices",
    "set_low_precision_enabled",
]

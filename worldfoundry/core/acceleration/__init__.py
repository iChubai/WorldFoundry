"""Composable in-tree diffusion accelerations that sit *above* kernels and expose approximation policy.

Responsibility: own *algorithmic* speed/quality trade-offs that must be
disableable, reportable, and comparable per model. Unlike ``kernels/``,
which swaps numerically equivalent (or bit-close) operators, these
paths can skip DiT layers, drop tokens, or quantize GEMMs.

This package is not a kernel catalog, a model-specific integration, or a
quality gate. Callers opt in per model and own the visual/numeric check.
Approximations stay off until a replace/setup helper is invoked.

Public surface (re-exported here; Triton kernels stay submodule-private):

- ``cache``: TeaCache-style reuse. ``FixedStepCache`` skips residuals on a fixed
  cadence; ``AdaptiveResidualCache`` reuses the previous prediction from residual
  magnitude. This skips whole DiT layers, not a single GEMM.
- ``cuda_graph_dispatch``: capture/replay CUDA Graphs keyed by autoregressive
  index so dynamic control flow stays outside the graph.
- ``encoder_lifecycle``: run a one-shot encoder (text / VAE), then offload and
  drop references so peak VRAM goes to the DiT.
- ``overlap`` / ``frame_prefetch`` / ``prewarm``: overlap compute with H2D or
  decode, host-pinned prefetch, and Graph/kernel warmup (with timeouts).
- ``quantization`` / ``nvfp4`` / ``triton_fp8``: replace Linear with quantized
  weights. The low-precision switch is centralized in
  ``set_low_precision_enabled``.
- ``token_pruning``: spatial/temporal token prune and restore. Shapes change, so
  this generally cannot enter a CUDA Graph.
- ``technology``: report which accelerations are active in this process for
  logs and regression diffs.
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
from worldfoundry.core.acceleration.cuda_graph_dispatch import (
    CUDAGraphDispatch,
    cuda_graph_capture_ar_index,
)
from worldfoundry.core.acceleration.encoder_lifecycle import (
    collect_and_release_cuda_memory,
    ensure_one_shot_encoder,
    move_tensors_to_cpu,
    offload_module_to_cpu,
    release_one_shot_encoder_references,
    run_one_shot_encoder_stage,
    setup_one_shot_encoder,
)
from worldfoundry.core.acceleration.frame_prefetch import (
    CudaHostPrefetch,
    LazyCudaFrame,
    prefetch_to_numpy,
)
from worldfoundry.core.acceleration.nvfp4 import (
    NVFP4Linear,
    dequantize_nvfp4,
    quantize_nvfp4,
    replace_linear_with_nvfp4,
)
from worldfoundry.core.acceleration.overlap import (
    CudaStreamOverlap,
    HostThreadOverlap,
    SynchronousOverlap,
)
from worldfoundry.core.acceleration.prewarm import (
    PrewarmDeadline,
    PrewarmSequenceTiming,
    PrewarmTimeoutError,
    PrewarmTiming,
    cuda_graph_prewarm_steps,
    run_async_prewarm_sequence,
    run_prewarm_sequence,
    run_timed_prewarm,
)
from worldfoundry.core.acceleration.quantization import (
    Float8Linear,
    WeightOnlyLinear,
    quantization_runtime_report,
    replace_linear_with_float8,
    replace_linear_with_weight_only,
    reset_quantization_runtime_window,
    set_low_precision_enabled,
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
    "AdaCacheResidualCache",
    "AdaptiveResidualCache",
    "AccelerationTechnology",
    "BlockTaylorSeerCache",
    "CUDAGraphDispatch",
    "CudaHostPrefetch",
    "CudaStreamOverlap",
    "CustomTaylorResidualCache",
    "FixedStepCache",
    "Float8Linear",
    "HostThreadOverlap",
    "LazyCudaFrame",
    "MagCacheResidualCache",
    "NVFP4Linear",
    "PrewarmDeadline",
    "PrewarmSequenceTiming",
    "PrewarmTimeoutError",
    "PrewarmTiming",
    "SynchronousOverlap",
    "TaylorSeerResidualCache",
    "TeaCacheResidualCache",
    "TokenPruneState",
    "TokenPruner",
    "WeightOnlyLinear",
    "collect_and_release_cuda_memory",
    "cuda_graph_capture_ar_index",
    "cuda_graph_prewarm_steps",
    "ensure_one_shot_encoder",
    "move_tensors_to_cpu",
    "offload_module_to_cpu",
    "prefetch_to_numpy",
    "prune_tokens",
    "dequantize_nvfp4",
    "quantize_nvfp4",
    "quantization_runtime_report",
    "reset_quantization_runtime_window",
    "replace_linear_with_float8",
    "replace_linear_with_nvfp4",
    "replace_linear_with_weight_only",
    "release_one_shot_encoder_references",
    "restore_tokens",
    "run_async_prewarm_sequence",
    "run_one_shot_encoder_stage",
    "run_prewarm_sequence",
    "run_timed_prewarm",
    "select_token_indices",
    "set_low_precision_enabled",
    "setup_one_shot_encoder",
    "acceleration_technology_report",
]

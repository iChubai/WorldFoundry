"""Attention subsystem: backend probing, layout-agnostic dispatch, RoPE, KV cache, and Context Parallel.

This package separates *which Attention kernel to run* from *how a model writes QKV*:

- ``backends``: zero-cost probing via ``find_spec`` (does not import crash-prone
  extensions). Distinguishes *installed* from *usable* (compute capability, HIP vs
  CUDA). ``auto`` deliberately resolves to in-tree PyTorch SDPA (the
  no-external-repo contract). FlashAttention, SageAttention, and xFormers require
  an explicit env opt-in and fall back to SDPA when unusable.
- ``dispatch``: normalizes einops layouts, then picks a backend from the workload
  signature. Short sequences, masks, and non-half dtypes always take exact SDPA
  to avoid fused-kernel launch cost or mismatched mask contracts. Failed
  signatures are quarantined; ``torch_sdpa`` is always retained as the last path.
- ``native``: exact SDPA wrapper, GQA compatibility, all-false-mask row
  normalization, and ``NativeAttention`` that can attach Context Parallel.
- ``varlen``: variable-length / packed path. Dense unpadded batches are *not*
  packed into varlen (slower for the small batches diffusion uses). External
  FlashAttention is tried only when ``version=2/3``; otherwise jagged Flash or
  per-sample SDPA.
- ``rope`` / ``rope_*``: 2D / 3D / ND / complex RoPE. The 3D path is CP-aware.
  The KV-cache-relative variant is for sink+window caches (do not rotate K on
  write; rotate on read).
- ``kvcache``: ``BlockKVCache`` is a fixed-size rolling buffer (CUDA Graph
  friendly) vs ``CompactingKVCache`` with pluggable policy/quantization (dynamic
  gather; not Graph-capturable).
- Ulysses / xDiT CP / sequence-parallel modules handle long sequences across
  ranks: all-to-all head dim into sequence shards, compute locally, then swap back.

Public symbols load lazily through ``__getattr__`` so
``import worldfoundry.core.attention`` does not pull in every Triton or
distributed dependency.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# ──────────────────────────────────────────────────────────────────────────
# Lazy export table — names resolve on first access so import stays cheap
# ──────────────────────────────────────────────────────────────────────────

_EXPORT_MODULES = {
    "AttentionKernelCapability": "worldfoundry.core.attention.backends",
    "ModelSpecificAttentionBackendError": "worldfoundry.core.attention.backends",
    "apply_complex_rotary_embedding": "worldfoundry.core.attention.complex_rope",
    "AttentionCallable": "worldfoundry.core.attention.model_backends",
    "AttentionFunction": "worldfoundry.core.attention.model_backends",
    "AttentionBackendInfo": "worldfoundry.core.attention.native",
    "BlockKVCache": "worldfoundry.core.attention.kvcache",
    "InferenceParams": "worldfoundry.core.attention.inference_state",
    "CausalVideoCacheGeometry": "worldfoundry.core.attention.causal_cache",
    "allocate_causal_video_cache": "worldfoundry.core.attention.causal_cache",
    "begin_causal_video_cache_block": "worldfoundry.core.attention.causal_cache",
    "causal_video_cache_geometry": "worldfoundry.core.attention.causal_cache",
    "causal_video_cache_state": "worldfoundry.core.attention.causal_cache",
    "commit_causal_video_cache_block": "worldfoundry.core.attention.causal_cache",
    "finish_causal_video_cache_call": "worldfoundry.core.attention.causal_cache",
    "ContextParallelAttention": "worldfoundry.core.attention.cp",
    "KVCacheRelativeRotaryPositionEmbedding3D": "worldfoundry.core.attention.rope",
    "ModelMetaArgs": "worldfoundry.core.attention.packed_sequence",
    "MaskedAttentionCallable": "worldfoundry.core.attention.model_backends",
    "MaskedAttentionFunction": "worldfoundry.core.attention.model_backends",
    "NativeAttention": "worldfoundry.core.attention.native",
    "PackedCoreAttnParams": "worldfoundry.core.attention.packed_sequence",
    "PackedCrossAttnParams": "worldfoundry.core.attention.packed_sequence",
    "PositionGetter": "worldfoundry.core.attention.rope_2d",
    "RotaryPositionEmbedding3D": "worldfoundry.core.attention.rope",
    "RotaryPositionEmbedding2D": "worldfoundry.core.attention.rope_2d",
    "attention_backend_capability": "worldfoundry.core.attention.backends",
    "attention_backend_from_env": "worldfoundry.core.attention.backends",
    "attention_dispatch_report": "worldfoundry.core.attention.dispatch",
    "attention_compile_receipt_scope": "worldfoundry.core.attention.dispatch",
    "attention_forward": "worldfoundry.core.attention.dispatch",
    "attention_provider_runtime_report": "worldfoundry.core.attention.dispatch",
    "clear_attention_dispatch_cache": "worldfoundry.core.attention.dispatch",
    "complex_rotary_frequencies": "worldfoundry.core.attention.complex_rope",
    "complex_rotary_frequencies_3d": "worldfoundry.core.attention.complex_rope",
    "flash_attention": "worldfoundry.core.attention.varlen",
    "flattened_attention": "worldfoundry.core.attention.hybrid",
    "masked_attention": "worldfoundry.core.attention.varlen",
    "hybrid_provider_attention": "worldfoundry.core.attention.hybrid",
    "flattened_multihead_attention": "worldfoundry.core.attention.native",
    "apply_nd_rotary_embedding": "worldfoundry.core.attention.rope_nd",
    "apply_rope_freqs": "worldfoundry.core.attention.rope",
    "apply_rotary_embedding": "worldfoundry.core.attention.rope",
    "apply_sequence_parallel_rope": "worldfoundry.core.attention.sequence_parallel_rope",
    "attention_backend_report": "worldfoundry.core.attention.backends",
    "attention_backend_info": "worldfoundry.core.attention.native",
    "attention_backend_context": "worldfoundry.core.attention.native",
    "get_1d_rotary_pos_embed": "worldfoundry.core.attention.rope_nd",
    "get_cu_seqlens": "worldfoundry.core.attention.sequence_metadata",
    "get_meshgrid_nd": "worldfoundry.core.attention.rope_nd",
    "get_nd_rotary_pos_embed": "worldfoundry.core.attention.rope_nd",
    "gpu_supports_flash_attention": "worldfoundry.core.attention.backends",
    "normalize_attention_backend": "worldfoundry.core.attention.backends",
    "native_sdpa_priority": "worldfoundry.core.attention.native",
    "normalize_fully_masked_rows": "worldfoundry.core.attention.native",
    "piecewise_attention": "worldfoundry.core.attention.piecewise",
    "piecewise_attention_available": "worldfoundry.core.attention.piecewise",
    "prope_dot_product_attention": "worldfoundry.core.attention.projective_rope",
    "pad_freqs": "worldfoundry.core.attention.sequence_parallel_rope",
    "packed_sequence_attention": "worldfoundry.core.attention.dispatch",
    "attention": "worldfoundry.core.attention.varlen",
    "probe_attention_backends": "worldfoundry.core.attention.backends",
    "require_generic_attention_backend": "worldfoundry.core.attention.backends",
    "reset_attention_provider_runtime": "worldfoundry.core.attention.dispatch",
    "reshape_rotary_for_broadcast": "worldfoundry.core.attention.rope_nd",
    "invert_camera_intrinsics": "worldfoundry.core.attention.projective_rope",
    "invert_se3": "worldfoundry.core.attention.projective_rope",
    "lift_camera_intrinsics": "worldfoundry.core.attention.projective_rope",
    "rotary_frequencies": "worldfoundry.core.attention.rope",
    "rotate_half": "worldfoundry.core.attention.rope",
    "sequence_parallel_attention_forward": "worldfoundry.core.attention.sequence_parallel_rope",
    "QKVSelfAttention": "worldfoundry.core.attention.vit_qkv",
    "QKNormRopeSelfAttention": "worldfoundry.core.attention.vit_qkv",
    "CSOHelper": "worldfoundry.core.attention.context_parallel_runtime",
    "UlyssesScheduler": "worldfoundry.core.attention.context_parallel_runtime",
    "cp_post_process": "worldfoundry.core.attention.context_parallel_runtime",
    "cp_pre_process": "worldfoundry.core.attention.context_parallel_runtime",
    "cso_communication": "worldfoundry.core.attention.context_parallel_runtime",
    "scaled_dot_product_attention": "worldfoundry.core.attention.native",
    "resolve_attention_backend": "worldfoundry.core.attention.backends",
    "resolve_transformers_attention_implementation": "worldfoundry.core.attention.backends",
    "varlen_scaled_dot_product_attention": "worldfoundry.core.attention.varlen",
}


def __getattr__(name: str) -> Any:
    """Resolve a public symbol from ``_EXPORT_MODULES`` on first access.

    The result is written back into ``globals()`` so later lookups skip
    ``import_module``. Unknown names raise :class:`AttributeError`.
    """
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Merge materialized globals with ``__all__`` so completion sees every export."""
    return sorted({*globals(), *__all__})


__all__ = [
    "AttentionKernelCapability",
    "ModelSpecificAttentionBackendError",
    "AttentionCallable",
    "AttentionFunction",
    "BlockKVCache",
    "CausalVideoCacheGeometry",
    "ContextParallelAttention",
    "InferenceParams",
    "AttentionBackendInfo",
    "KVCacheRelativeRotaryPositionEmbedding3D",
    "ModelMetaArgs",
    "MaskedAttentionCallable",
    "MaskedAttentionFunction",
    "NativeAttention",
    "PackedCoreAttnParams",
    "PackedCrossAttnParams",
    "PositionGetter",
    "RotaryPositionEmbedding2D",
    "RotaryPositionEmbedding3D",
    "attention_backend_capability",
    "attention_backend_from_env",
    "attention_backend_report",
    "attention_dispatch_report",
    "attention_compile_receipt_scope",
    "attention_forward",
    "attention_provider_runtime_report",
    "clear_attention_dispatch_cache",
    "complex_rotary_frequencies",
    "complex_rotary_frequencies_3d",
    "attention",
    "apply_complex_rotary_embedding",
    "allocate_causal_video_cache",
    "begin_causal_video_cache_block",
    "causal_video_cache_geometry",
    "causal_video_cache_state",
    "commit_causal_video_cache_block",
    "finish_causal_video_cache_call",
    "apply_nd_rotary_embedding",
    "apply_rope_freqs",
    "apply_rotary_embedding",
    "apply_sequence_parallel_rope",
    "attention_backend_info",
    "attention_backend_context",
    "gpu_supports_flash_attention",
    "flash_attention",
    "flattened_attention",
    "masked_attention",
    "hybrid_provider_attention",
    "invert_camera_intrinsics",
    "invert_se3",
    "flattened_multihead_attention",
    "get_1d_rotary_pos_embed",
    "get_cu_seqlens",
    "get_meshgrid_nd",
    "get_nd_rotary_pos_embed",
    "normalize_attention_backend",
    "native_sdpa_priority",
    "normalize_fully_masked_rows",
    "piecewise_attention",
    "piecewise_attention_available",
    "prope_dot_product_attention",
    "pad_freqs",
    "packed_sequence_attention",
    "probe_attention_backends",
    "require_generic_attention_backend",
    "reset_attention_provider_runtime",
    "QKVSelfAttention",
    "QKNormRopeSelfAttention",
    "CSOHelper",
    "UlyssesScheduler",
    "cp_post_process",
    "cp_pre_process",
    "cso_communication",
    "reshape_rotary_for_broadcast",
    "lift_camera_intrinsics",
    "rotary_frequencies",
    "rotate_half",
    "sequence_parallel_attention_forward",
    "scaled_dot_product_attention",
    "resolve_attention_backend",
    "resolve_transformers_attention_implementation",
    "varlen_scaled_dot_product_attention",
]

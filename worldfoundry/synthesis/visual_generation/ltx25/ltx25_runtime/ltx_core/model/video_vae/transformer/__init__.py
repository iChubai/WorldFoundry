"""Neighborhood-Attention transformer building blocks for the diffusion video VAE."""

from .apply import (
    apply_diffvae_config,
    apply_diffvae_mode,
    build_compile_diffusion_decoder_op,
    build_cutlass_fna_diffusion_decoder_op,
    build_diffvae_mode_op,
    resolve_attention_for_host,
)
from .attention import (
    NAAttentionCallable,
    NattenAttention,
    NeighborhoodAttention3D,
    configure_w_chunks,
    natten_available,
)
from .blocks import DiffusionNABlock, NABlock
from .chunked.block import ChunkedDiffusionNABlock
from .combined.block import CombinedDiffusionNABlock
from .compiling import compile_diffusion_decoder, configure_cutlass_fna_diffusion_decoder, configure_natten_backend
from .config import DiffVAEBlockKind, DiffVAEConfig, DiffVAEMode, NAttentionKind
from .dsl_kernels import (
    NA_SOFTMAX_BOUND_BUFFER,
    DSLAttention,
    DSLDiffusionNABlock,
    enable_blackwell_dsl,
    na_dsl_available,
)
from .fallback_na import EagerSdpaAttention, TritonNaAttention, triton_na_available
from .layers import AdaLNZero, ChannelLinear, LinearPixelShuffleUpsample, modulate
from .qkv import QKVProjections
from .rope import apply_abs_rope, apply_abs_rope_slab, configure_abs_rope, default_rope_dim_split, rope_inv_freqs
from .swiglu import SwiGLU, SwiGLUTileSpec, configure_swiglu_tile

__all__ = [
    "NA_SOFTMAX_BOUND_BUFFER",
    "AdaLNZero",
    "ChannelLinear",
    "ChunkedDiffusionNABlock",
    "CombinedDiffusionNABlock",
    "DSLAttention",
    "DSLDiffusionNABlock",
    "DiffVAEBlockKind",
    "DiffVAEConfig",
    "DiffVAEMode",
    "DiffusionNABlock",
    "EagerSdpaAttention",
    "LinearPixelShuffleUpsample",
    "NAAttentionCallable",
    "NABlock",
    "NAttentionKind",
    "NattenAttention",
    "NeighborhoodAttention3D",
    "QKVProjections",
    "SwiGLU",
    "SwiGLUTileSpec",
    "TritonNaAttention",
    "apply_abs_rope",
    "apply_abs_rope_slab",
    "apply_diffvae_config",
    "apply_diffvae_mode",
    "build_compile_diffusion_decoder_op",
    "build_cutlass_fna_diffusion_decoder_op",
    "build_diffvae_mode_op",
    "compile_diffusion_decoder",
    "configure_abs_rope",
    "configure_cutlass_fna_diffusion_decoder",
    "configure_natten_backend",
    "configure_swiglu_tile",
    "configure_w_chunks",
    "default_rope_dim_split",
    "enable_blackwell_dsl",
    "modulate",
    "na_dsl_available",
    "natten_available",
    "resolve_attention_for_host",
    "rope_inv_freqs",
    "triton_na_available",
]

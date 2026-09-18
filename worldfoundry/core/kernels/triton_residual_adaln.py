"""Inference residual addition and affine-free AdaLN in one Triton launch.

The public capability/gradient/shape gates live in ``diffusion.py``. Outputs
are functional: the residual input is never overwritten. BF16 expression
boundaries match the existing two-kernel path instead of removing rounding
as a side effect of fusion. An optional broadcast gate can be materialized
in the same launch for consumers that require a third output.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from worldfoundry.core.kernels.triton_diffusion import (
    _broadcast_row_offset,
    _num_warps,
    _padded_outer_shape_and_strides,
    _round_float32_to_bfloat16,
)


@triton.jit
def _round_dtype(value, dtype: tl.constexpr):
    if dtype == tl.bfloat16:
        return tl.where(value == value, _round_float32_to_bfloat16(value), value)
    return value.to(dtype).to(tl.float32)


@triton.jit
def _fused_residual_adaln(
    residual_ptr, update_ptr, gate_ptr, scale_ptr, shift_ptr, output_gate_ptr,
    norm_out_ptr, residual_out_ptr, gate_out_ptr,
    features: tl.constexpr, eps: tl.constexpr,
    dims: tl.constexpr, gs: tl.constexpr, ss: tl.constexpr, hs: tl.constexpr, os: tl.constexpr,
    has_gate: tl.constexpr, emit_gate: tl.constexpr,
    product_dtype: tl.constexpr, modulation_dtype: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, block)
    mask = col < features
    offsets = row * features + col
    residual = tl.load(residual_ptr + offsets, mask, other=0).to(tl.float32)
    update = tl.load(update_ptr + offsets, mask, other=0).to(tl.float32)
    if has_gate:
        gate_base = _broadcast_row_offset(row, dims[0], dims[1], dims[2], dims[3], gs[0], gs[1], gs[2], gs[3])
        gate = tl.load(gate_ptr + gate_base + col * gs[4], mask, other=0).to(tl.float32)
        update = _round_dtype(update * gate, product_dtype)
    summed = _round_dtype(residual + update, residual_out_ptr.dtype.element_ty)
    tl.store(residual_out_ptr + offsets, summed, mask)
    # Padded columns must stay zero after residual rounding and centering.
    summed = tl.where(mask, summed, 0.0)
    mean = tl.sum(summed, 0) / features
    centered = tl.where(mask, summed - mean, 0.0)
    variance = tl.sum(centered * centered, 0) / features
    normed = (centered * tl.rsqrt(variance + eps)).to(residual_out_ptr.dtype.element_ty)
    scale_base = _broadcast_row_offset(row, dims[0], dims[1], dims[2], dims[3], ss[0], ss[1], ss[2], ss[3])
    shift_base = _broadcast_row_offset(row, dims[0], dims[1], dims[2], dims[3], hs[0], hs[1], hs[2], hs[3])
    scale_input = tl.load(scale_ptr + scale_base + col * ss[4], mask, other=0)
    shift = tl.load(shift_ptr + shift_base + col * hs[4], mask, other=0).to(tl.float32)
    factor = (1.0 + scale_input.to(tl.float32)).to(scale_input.dtype)
    modulated = (normed.to(tl.float32) * factor.to(tl.float32)).to(modulation_dtype).to(tl.float32)
    tl.store(norm_out_ptr + offsets, modulated + shift, mask)
    if emit_gate:
        out_base = _broadcast_row_offset(row, dims[0], dims[1], dims[2], dims[3], os[0], os[1], os[2], os[3])
        out_gate = tl.load(output_gate_ptr + out_base + col * os[4], mask, other=0)
        tl.store(gate_out_ptr + offsets, out_gate, mask)


def residual_layer_norm_scale_shift(
    residual: torch.Tensor,
    update: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    gate: torch.Tensor | None,
    output_gate: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    """Return normalized/modulated output, residual sum, and optional gate."""
    types = {torch.float32: tl.float32, torch.float16: tl.float16, torch.bfloat16: tl.bfloat16}
    product_dtype = update.dtype if gate is None else torch.promote_types(update.dtype, gate.dtype)
    residual_dtype = torch.promote_types(residual.dtype, product_dtype)
    modulation_dtype = torch.promote_types(residual_dtype, scale.dtype)
    out_dtype = torch.promote_types(modulation_dtype, shift.dtype)
    residual_out = torch.empty_like(residual, dtype=residual_dtype)
    normalized = torch.empty_like(residual, dtype=out_dtype)
    gate_out = torch.empty_like(residual, dtype=output_gate.dtype) if output_gate is not None else None
    dims, ss = _padded_outer_shape_and_strides(residual, scale)
    _, hs = _padded_outer_shape_and_strides(residual, shift)
    _, gs = _padded_outer_shape_and_strides(residual, gate if gate is not None else scale)
    _, os = _padded_outer_shape_and_strides(residual, output_gate if output_gate is not None else scale)
    features = residual.shape[-1]
    _fused_residual_adaln[(residual.numel() // features,)](
        residual, update, gate, scale, shift, output_gate, normalized, residual_out, gate_out,
        features, float(eps), dims, gs, ss, hs, os,
        gate is not None, output_gate is not None,
        types[product_dtype], types[modulation_dtype], triton.next_power_of_2(features),
        num_warps=_num_warps(triton.next_power_of_2(features)),
    )
    return (normalized, residual_out) if gate_out is None else (normalized, residual_out, gate_out)

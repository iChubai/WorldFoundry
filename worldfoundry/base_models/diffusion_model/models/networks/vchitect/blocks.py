"""Native checkpoint-compatible Vchitect transformer blocks."""

from __future__ import annotations

import torch
from torch import nn

from worldfoundry.core.kernels import layer_norm_scale_shift, residual_gate_add

from .attention import Attention


class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.proj(hidden_states), approximate="tanh")


class FeedForward(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.ModuleList((GELU(dim, dim * 4), nn.Dropout(0.0), nn.Linear(dim * 4, dim)))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor, embedding: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.linear(
            self.silu(embedding)
        ).chunk(6, dim=1)
        normalized = layer_norm_scale_shift(
            hidden_states, scale_msa[:, None], shift_msa[:, None], eps=self.norm.eps
        )
        return normalized, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(self.silu(embedding)).chunk(2, dim=1)
        return layer_norm_scale_shift(
            hidden_states, scale[:, None], shift[:, None], eps=self.norm.eps
        )


class JointTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, *, context_pre_only: bool) -> None:
        super().__init__()
        self.context_pre_only = bool(context_pre_only)
        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormContinuous(dim) if context_pre_only else AdaLayerNormZero(dim)
        self.attn = Attention(dim=dim, heads=heads, context_pre_only=context_pre_only)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim)
        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embedding: torch.Tensor,
        *,
        frequencies: torch.Tensor,
        frames: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        normalized, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, embedding)
        if self.context_pre_only:
            normalized_context = self.norm1_context(encoder_hidden_states, embedding)
        else:
            normalized_context, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, embedding
            )
        attention, context_attention = self.attn(
            normalized,
            normalized_context,
            frequencies=frequencies,
            frames=frames,
        )
        hidden_states = residual_gate_add(hidden_states, attention, gate_msa[:, None])
        normalized = layer_norm_scale_shift(
            hidden_states, scale_mlp[:, None], shift_mlp[:, None], eps=self.norm2.eps
        )
        hidden_states = residual_gate_add(hidden_states, self.ff(normalized), gate_mlp[:, None])
        if self.context_pre_only:
            return None, hidden_states
        encoder_hidden_states = residual_gate_add(
            encoder_hidden_states, context_attention, c_gate_msa[:, None]
        )
        normalized_context = layer_norm_scale_shift(
            encoder_hidden_states, c_scale_mlp[:, None], c_shift_mlp[:, None], eps=self.norm2_context.eps
        )
        encoder_hidden_states = residual_gate_add(
            encoder_hidden_states, self.ff_context(normalized_context), c_gate_mlp[:, None]
        )
        return encoder_hidden_states, hidden_states


__all__ = [
    "AdaLayerNormContinuous",
    "AdaLayerNormZero",
    "FeedForward",
    "GELU",
    "JointTransformerBlock",
]

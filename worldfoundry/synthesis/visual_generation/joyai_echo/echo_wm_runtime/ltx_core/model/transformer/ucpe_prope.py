# MIT License
#
# Copyright (c) Authors of
# "Cameras as Relative Positional Encoding" https://arxiv.org/pdf/2507.10496
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# How to use PRoPE attention for self-attention:
#
# 1. Easiest way (fast):
#    attn = PropeDotProductAttention(...)
#    o = attn(q, k, v, viewmats, Ks)
#
# 2. More flexible way (fast):
#    attn = PropeDotProductAttention(...)
#    attn._precompute_and_cache_apply_fns(viewmats, Ks)
#    q = attn._apply_to_q(q)
#    k = attn._apply_to_kv(k)
#    v = attn._apply_to_kv(v)
#    o = F.scaled_dot_product_attention(q, k, v, **kwargs)
#    o = attn._apply_to_o(o)
#
# 3. The most flexible way (but slower because repeated computation of RoPE coefficients):
#    o = prope_dot_product_attention(q, k, v, ...)
#
# How to use PRoPE attention for cross-attention:
#
#    attn_src = PropeDotProductAttention(...)
#    attn_tgt = PropeDotProductAttention(...)
#    attn_src._precompute_and_cache_apply_fns(viewmats_src, Ks_src)
#    attn_tgt._precompute_and_cache_apply_fns(viewmats_tgt, Ks_tgt)
#    q_src = attn_src._apply_to_q(q_src)
#    k_tgt = attn_tgt._apply_to_kv(k_tgt)
#    v_tgt = attn_tgt._apply_to_kv(v_tgt)
#    o_src = F.scaled_dot_product_attention(q_src, k_tgt, v_tgt, **kwargs)
#    o_src = attn_src._apply_to_o(o_src)

from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F

from worldfoundry.base_models.diffusion_model.models.networks.worldcrafter.ucpe.prope import (
    _apply_tiled_projmat as _project_camera_features,
)
from worldfoundry.base_models.diffusion_model.models.networks.worldcrafter.ucpe.prope import (
    _lift_K,
    _rope_precompute_coeffs,
)
from worldfoundry.base_models.three_dimensions.point_clouds.pi3_inference.models.layers.prope import (
    _apply_block_diagonal,
    _invert_K,
    _invert_SE3,
    _rope_apply_coeffs,
)


class PropeDotProductAttention(torch.nn.Module):
    """PRoPE attention with precomputed RoPE coefficients."""

    coeffs_x_0: torch.Tensor
    coeffs_x_1: torch.Tensor
    coeffs_y_0: torch.Tensor
    coeffs_y_1: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        patches_x: int,
        patches_y: int,
        image_width: int,
        image_height: int,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
        precompute_coeffs: bool = True,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.image_width = image_width
        self.image_height = image_height

        if precompute_coeffs:
            coeffs_x: tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
                torch.tile(torch.arange(patches_x), (patches_y,)),
                freq_base=freq_base,
                freq_scale=freq_scale,
                feat_dim=head_dim // 4,
            )
            coeffs_y: tuple[torch.Tensor, torch.Tensor] = _rope_precompute_coeffs(
                torch.repeat_interleave(torch.arange(patches_y), patches_x),
                freq_base=freq_base,
                freq_scale=freq_scale,
                feat_dim=head_dim // 4,
            )
            # Model builders instantiate modules on the meta device before
            # loading weights. Meta buffers cannot later be materialized by a
            # state dict because these coefficients are intentionally not
            # persisted, so generate them lazily on the first real forward.
            if coeffs_x[0].is_meta:
                coeffs_x = coeffs_y = (None, None)
            self.register_buffer("coeffs_x_0", coeffs_x[0], persistent=False)
            self.register_buffer("coeffs_x_1", coeffs_x[1], persistent=False)
            self.register_buffer("coeffs_y_0", coeffs_y[0], persistent=False)
            self.register_buffer("coeffs_y_1", coeffs_y[1], persistent=False)
        else:
            self.coeffs_x_0 = None
            self.coeffs_x_1 = None
            self.coeffs_y_0 = None
            self.coeffs_y_1 = None

    # override load_state_dict to not load coeffs if they exist (for backward compatibility)
    def load_state_dict(self, state_dict, strict=True):
        # remove coeffs from state_dict
        state_dict.pop("coeffs_x_0", None)
        state_dict.pop("coeffs_x_1", None)
        state_dict.pop("coeffs_y_0", None)
        state_dict.pop("coeffs_y_1", None)
        super().load_state_dict(state_dict, strict)

    def forward(
        self,
        q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
        viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
        Ks: torch.Tensor | None,  # (batch, cameras, 3, 3)
        **kwargs,
    ) -> torch.Tensor:
        return prope_dot_product_attention(
            q,
            k,
            v,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=self.patches_x,
            patches_y=self.patches_y,
            image_width=self.image_width,
            image_height=self.image_height,
            coeffs_x=(self.coeffs_x_0, self.coeffs_x_1),
            coeffs_y=(self.coeffs_y_0, self.coeffs_y_1),
            **kwargs,
        )

    def _precompute_and_cache_apply_fns(
        self,
        viewmats: torch.Tensor,
        Ks: torch.Tensor | None,
        coeffs_x: tuple[torch.Tensor, torch.Tensor] | None = None,
        coeffs_y: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        (batch, cameras, _, _) = viewmats.shape
        assert viewmats.shape == (batch, cameras, 4, 4)
        assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
        # self.cameras = cameras

        cached_x = None if self.coeffs_x_0 is None else (self.coeffs_x_0, self.coeffs_x_1)
        cached_y = None if self.coeffs_y_0 is None else (self.coeffs_y_0, self.coeffs_y_1)
        self.apply_fn_q, self.apply_fn_kv, self.apply_fn_o = _prepare_apply_fns(
            head_dim=self.head_dim,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=self.patches_x,
            patches_y=self.patches_y,
            image_width=self.image_width,
            image_height=self.image_height,
            coeffs_x=cached_x if coeffs_x is None else coeffs_x,
            coeffs_y=cached_y if coeffs_y is None else coeffs_y,
        )

    def _apply_to_q(self, q: torch.Tensor) -> torch.Tensor:
        (batch, num_heads, seqlen, head_dim) = q.shape
        # assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert q.shape == (batch, num_heads, seqlen, head_dim)
        assert self.apply_fn_q is not None
        return self.apply_fn_q(q)

    def _apply_to_kv(self, kv: torch.Tensor) -> torch.Tensor:
        (batch, num_heads, seqlen, head_dim) = kv.shape
        # assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert kv.shape == (batch, num_heads, seqlen, head_dim)
        assert self.apply_fn_kv is not None
        return self.apply_fn_kv(kv)

    def _apply_to_o(self, o: torch.Tensor) -> torch.Tensor:
        (batch, num_heads, seqlen, head_dim) = o.shape
        # assert seqlen == self.cameras * self.patches_x * self.patches_y
        assert head_dim == self.head_dim
        assert o.shape == (batch, num_heads, seqlen, head_dim)
        assert self.apply_fn_o is not None
        return self.apply_fn_o(o)


def prope_dot_product_attention(
    q: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    k: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    v: torch.Tensor,  # (batch, num_heads, seqlen, head_dim)
    *,
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: torch.Tensor | None,  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    coeffs_x: tuple[torch.Tensor, torch.Tensor] | None = None,
    coeffs_y: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
) -> torch.Tensor:
    """Similar to torch.nn.functional.scaled_dot_product_attention, but applies PRoPE-style
    positional encoding.

    Currently, we assume that the sequence length is equal to:

        cameras * patches_x * patches_y

    And token ordering allows the `(seqlen,)` axis to be reshaped into
    `(cameras, patches_x, patches_y)`.
    """
    # We're going to assume self-attention: all inputs are the same shape.
    (batch, num_heads, seqlen, head_dim) = q.shape
    cameras = viewmats.shape[1]
    assert q.shape == k.shape == v.shape
    assert viewmats.shape == (batch, cameras, 4, 4)
    assert Ks is None or Ks.shape == (batch, cameras, 3, 3)
    assert seqlen == cameras * patches_x * patches_y

    apply_fn_q, apply_fn_kv, apply_fn_o = _prepare_apply_fns(
        head_dim=head_dim,
        viewmats=viewmats,
        Ks=Ks,
        patches_x=patches_x,
        patches_y=patches_y,
        image_width=image_width,
        image_height=image_height,
        coeffs_x=coeffs_x,
        coeffs_y=coeffs_y,
    )

    out = F.scaled_dot_product_attention(
        query=apply_fn_q(q),
        key=apply_fn_kv(k),
        value=apply_fn_kv(v),
        **kwargs,
    )
    out = apply_fn_o(out)
    assert out.shape == (batch, num_heads, seqlen, head_dim)
    return out


def _prepare_apply_fns(
    head_dim: int,  # Q/K/V will have this last dimension
    viewmats: torch.Tensor,  # (batch, cameras, 4, 4)
    Ks: torch.Tensor | None,  # (batch, cameras, 3, 3)
    patches_x: int,  # How many patches wide is each image?
    patches_y: int,  # How many patches tall is each image?
    image_width: int,  # Width of the image. Used to normalize intrinsics.
    image_height: int,  # Height of the image. Used to normalize intrinsics.
    coeffs_x: torch.Tensor | None = None,
    coeffs_y: torch.Tensor | None = None,
) -> tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Prepare transforms for PRoPE-style positional encoding."""
    device = viewmats.device
    (batch, cameras, _, _) = viewmats.shape
    dtype = viewmats.dtype

    # Normalize camera intrinsics.
    if Ks is not None:
        Ks_norm = torch.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0] / image_width
        Ks_norm[..., 1, 1] = Ks[..., 1, 1] / image_height
        Ks_norm[..., 0, 2] = Ks[..., 0, 2] / image_width - 0.5
        Ks_norm[..., 1, 2] = Ks[..., 1, 2] / image_height - 0.5
        Ks_norm[..., 2, 2] = 1.0
        del Ks

        # Compute the camera projection matrices we use in PRoPE.
        # - K is an `image<-camera` transform.
        # - viewmats is a `camera<-world` transform.
        # - P = lift(K) @ viewmats is an `image<-world` transform.
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2)
        P_inv = torch.einsum(
            "...ij,...jk->...ik",
            _invert_SE3(viewmats),
            _lift_K(_invert_K(Ks_norm)),
        )

    else:
        # GTA formula. P is `camera<-world` transform.
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = _invert_SE3(viewmats)

    assert P.shape == P_inv.shape == (batch, cameras, 4, 4)

    # Precompute cos/sin terms for RoPE. We use tiles/repeats for 'row-major'
    # broadcasting.
    if coeffs_x is None:
        coeffs_x = _rope_precompute_coeffs(
            torch.tile(torch.arange(patches_x, device=device), (patches_y * cameras,)),
            freq_base=100.0,
            freq_scale=1.0,
            feat_dim=head_dim // 4,
            dtype=dtype,
        )
    if coeffs_y is None:
        coeffs_y = _rope_precompute_coeffs(
            torch.tile(
                torch.repeat_interleave(torch.arange(patches_y, device=device), patches_x),
                (cameras,),
            ),
            freq_base=100.0,
            freq_scale=1.0,
            feat_dim=head_dim // 4,
            dtype=dtype,
        )

    # Block-diagonal transforms to the inputs and outputs of the attention operator.
    assert head_dim % 4 == 0
    transforms_q = [
        (partial(_apply_tiled_projmat, matrix=P_T), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ]
    transforms_kv = [
        (partial(_apply_tiled_projmat, matrix=P_inv), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y), head_dim // 4),
    ]
    transforms_o = [
        (partial(_apply_tiled_projmat, matrix=P), head_dim // 2),
        (partial(_rope_apply_coeffs, coeffs=coeffs_x, inverse=True), head_dim // 4),
        (partial(_rope_apply_coeffs, coeffs=coeffs_y, inverse=True), head_dim // 4),
    ]

    apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    return apply_fn_q, apply_fn_kv, apply_fn_o


def _apply_tiled_projmat(feats: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Cast camera matrices to the attention feature precision before projection."""
    return _project_camera_features(feats, matrix.to(device=feats.device, dtype=feats.dtype))

# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.forcing import model as forcing
from worldfoundry.base_models.diffusion_model.models.networks.wan.variants.minwm import action, prope
from worldfoundry.base_models.diffusion_model.models.networks.wan.varlen_cross_attention import (
    CachedTextCrossAttentionMixin,
    ImageTextCrossAttentionMixin,
)

pytestmark = pytest.mark.unit


def test_wan_cross_forwards_are_shared_without_collapsing_native_ancestry() -> None:
    for family in (forcing, action):
        assert family.WanT2VCrossAttention.forward is CachedTextCrossAttentionMixin.forward
        assert family.WanI2VCrossAttention.forward is ImageTextCrossAttentionMixin.forward
        assert issubclass(family.WanT2VCrossAttention, family.WanSelfAttention)
        assert issubclass(family.WanI2VCrossAttention, family.WanSelfAttention)
    assert forcing.WanSelfAttention is not action.WanSelfAttention
    assert forcing.WanRMSNorm is not action.WanRMSNorm


@pytest.mark.parametrize("zero_init", [False, True])
def test_minwm_prope_installer_selects_self_attention_and_excludes_native_cross_types(zero_init: bool) -> None:
    layers = nn.ModuleDict(
        {
            "self_attn": action.WanSelfAttention(32, 4),
            "text": action.WanT2VCrossAttention(32, 4),
            "image": action.WanI2VCrossAttention(32, 4),
            "gan": action.WanGanCrossAttention(32, 4),
            "forcing_self": forcing.WanSelfAttention(32, 4),
        }
    )
    prope.add_prope_parameters(layers, zero_init=zero_init)
    assert [name for name, layer in layers.items() if hasattr(layer, "prope_o")] == ["self_attn"]
    parameter = layers["self_attn"].prope_o.weight
    if zero_init:
        assert torch.count_nonzero(parameter).item() == 0
    prope.add_prope_parameters(layers, zero_init=not zero_init)
    assert layers["self_attn"].prope_o.weight is parameter
    assert type(layers["text"].norm_q) is action.WanRMSNorm

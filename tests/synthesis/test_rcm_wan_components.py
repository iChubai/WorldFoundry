# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""rCM reuses Wan layers without reintroducing FP32 interpolation buffers."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan import reference_21
from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.variants.rcm import Decoder3d
from worldfoundry.core.vram.initialization import init_weights_on_device
from worldfoundry.synthesis.visual_generation.rcm.rcm_runtime.tokenizers.wan2pt1 import WanVAE_

pytestmark = pytest.mark.unit


def test_rcm_vae_uses_reference_graph_and_native_dtype_upsampling() -> None:
    with torch.device("meta"):
        model = WanVAE_(dim=8, z_dim=4, num_res_blocks=1)
        reference = reference_21.WanVAE_(dim=8, z_dim=4, num_res_blocks=1)
    assert type(model.encoder) is reference_21.Encoder3d
    assert type(model.decoder) is Decoder3d
    assert model.state_dict().keys() == reference.state_dict().keys()
    for key, value in model.state_dict().items():
        assert value.shape == reference.state_dict()[key].shape
    for layer in model.decoder.upsamples:
        if isinstance(layer, reference_21.Resample):
            assert type(layer.resample[0]) is nn.Upsample
            pixels = torch.randn(1, 8, 2, 2, dtype=torch.bfloat16)
            assert layer.resample[0](pixels).dtype == torch.bfloat16


@pytest.mark.parametrize("include_buffers", [False, True])
def test_canonical_weight_placement_restores_registration_hooks(include_buffers: bool) -> None:
    parameter_hook = nn.Module.register_parameter
    buffer_hook = nn.Module.register_buffer
    with init_weights_on_device(include_buffers=include_buffers):
        model = nn.Linear(3, 4)
        model.register_buffer("buffer", torch.ones(2))
    assert model.weight.device.type == "meta"
    assert model.buffer.device.type == ("meta" if include_buffers else "cpu")
    assert nn.Module.register_parameter is parameter_hook
    assert nn.Module.register_buffer is buffer_hook

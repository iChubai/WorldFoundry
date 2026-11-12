# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Family loaders retain Wan checkpoint layouts while sharing the codec graph."""

from __future__ import annotations

import importlib

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan import model as wan_vae
from worldfoundry.base_models.diffusion_model.models.encoders.wan.model import WanTextEncoder

pytestmark = pytest.mark.unit

VAE_MODULES = (
    "worldfoundry.synthesis.action_generation.ahawam.wan_video_vae",
    "worldfoundry.synthesis.action_generation.fastwam.wan_video_vae",
    "worldfoundry.synthesis.visual_generation.kairos.kairos_runtime.kairos.modules.vaes.wan_video_vae",
    "worldfoundry.synthesis.visual_generation.worldcam.worldcam_runtime.models.wan_video_vae",
)


@pytest.mark.parametrize("module_name", VAE_MODULES)
@pytest.mark.parametrize(
    "wrapper_name,model_class", [("WanVideoVAE", wan_vae.VideoVAE_), ("WanVideoVAE38", wan_vae.VideoVAE38_)]
)
def test_family_vae_builds_canonical_checkpoint_graph(module_name: str, wrapper_name: str, model_class: type) -> None:
    module = importlib.import_module(module_name)
    with torch.device("meta"):
        wrapper = getattr(module, wrapper_name)()
        canonical = getattr(wan_vae, wrapper_name)()
    assert type(wrapper.model) is model_class
    assert wrapper.state_dict().keys() == canonical.state_dict().keys()
    for key, tensor in wrapper.state_dict().items():
        assert tensor.shape == canonical.state_dict()[key].shape, key


def test_ahawam_loader_builds_canonical_text_encoder() -> None:
    from worldfoundry.synthesis.action_generation.ahawam import modeling

    assert modeling.WanTextEncoder is WanTextEncoder


def test_worldcam_offload_classes_match_canonical_codec() -> None:
    from worldfoundry.synthesis.visual_generation.worldcam.worldcam_runtime.pipelines import wan_video_new

    assert wan_video_new.CausalConv3d is wan_vae.CausalConv3d
    assert wan_video_new.RMS_norm is wan_vae.RMS_norm
    assert wan_video_new.Upsample is wan_vae.Upsample

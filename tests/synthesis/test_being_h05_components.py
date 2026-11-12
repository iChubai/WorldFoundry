# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Being-H0.5 constructs and loads its canonical foundation components."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("norm_type", ["rms_norm", "layer_norm"])
def test_being_vision_registry_loads_canonical_checkpoint(tmp_path: Path, norm_type: str) -> None:
    from worldfoundry.base_models.llm_mllm_core.mllm.intern_vit.configuration import InternVisionConfig
    from worldfoundry.base_models.llm_mllm_core.mllm.intern_vit.model import InternVisionModel
    from worldfoundry.synthesis.action_generation.being_h05.preprocessing.constants import VIT_MODEL_ARCH

    config_class, model_class = VIT_MODEL_ARCH["InternVisionModel"]
    assert config_class is InternVisionConfig
    assert model_class is InternVisionModel
    config = config_class(
        image_size=8,
        patch_size=2,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        norm_type=norm_type,
        use_flash_attn=False,
    )
    model = model_class(config).eval()
    state = model.state_dict()
    assert state["embeddings.patch_embedding.weight"].shape == (16, 3, 2, 2)
    assert state["encoder.layers.0.attn.q_norm.weight"].shape == (16,)
    model.save_pretrained(tmp_path, safe_serialization=True)
    reloaded = model_class.from_pretrained(tmp_path, config=config_class.from_pretrained(tmp_path)).eval()
    assert reloaded.state_dict().keys() == state.keys()
    for key, value in reloaded.state_dict().items():
        torch.testing.assert_close(value, state[key], atol=0, rtol=0)
    with torch.no_grad():
        pixels = torch.randn(1, 3, 12, 12)
        expected = model(pixels).last_hidden_state
        actual = reloaded(pixels).last_hidden_state
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.shape == (1, 37, 16)

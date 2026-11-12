# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""SolarWM loads its Wan codecs through the canonical checkpoint graphs."""

from pathlib import Path
from typing import Any

import pytest
import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.wan import reference_22
from worldfoundry.base_models.diffusion_model.models.encoders.wan import model as wan_text
from worldfoundry.base_models.diffusion_model.models.encoders.wan import reference as wan_t5
from worldfoundry.synthesis.visual_generation.solarwm.solarwm_runtime.solarwm.backends.wan22.runtime import components

pytestmark = pytest.mark.unit


def test_solarwm_wan_text_loads_canonical_weights_and_masks_padding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    factory = wan_t5.umt5_xxl

    def tiny_encoder(**kwargs: Any) -> torch.nn.Module:
        return factory(
            **kwargs,
            vocab_size=31,
            dim=16,
            dim_attn=16,
            dim_ffn=32,
            num_heads=4,
            encoder_layers=1,
            decoder_layers=1,
            num_buckets=8,
            dropout=0.0,
        )

    class Tokenizer:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs == {"name": "tokenizer", "seq_len": 512, "clean": "whitespace"}

        def __call__(self, prompts: list[str], **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            assert prompts == ["a", "b"]
            assert kwargs == {"return_mask": True, "add_special_tokens": True}
            ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
            return ids, ids != 0

    reference = tiny_encoder(encoder_only=True).eval()
    weights = tmp_path / "umt5.pth"
    torch.save(reference.state_dict(), weights)
    monkeypatch.setattr(wan_t5, "umt5_xxl", tiny_encoder)
    monkeypatch.setattr(wan_text, "HuggingfaceTokenizer", Tokenizer)
    encoder = components.WanTextEncoder(weights, "tokenizer")
    assert isinstance(encoder.module, wan_t5.T5Encoder)
    assert encoder.module.state_dict().keys() == reference.state_dict().keys()
    result = encoder(["a", "b"])["prompt_embeds"]
    ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    expected = reference(ids, ids != 0)
    expected[0, 2:] = 0
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_solarwm_wan_vae_loads_canonical_weights_and_streams(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    factory = reference_22._video_vae
    small = {"dim": 4, "dec_dim": 4, "dim_mult": [1, 2, 4, 4], "num_res_blocks": 1}
    reference = factory(z_dim=48, temperal_downsample=[False, True, True], **small).eval()
    weights = tmp_path / "wan.pth"
    torch.save(reference.state_dict(), weights)

    def tiny_vae(**kwargs: Any) -> torch.nn.Module:
        assert kwargs == {
            "pretrained_path": str(weights),
            "z_dim": 48,
            "temperal_downsample": [False, True, True],
        }
        return factory(**kwargs, **small)

    monkeypatch.setattr(reference_22, "_video_vae", tiny_vae)
    codec = components.Wan5BVAE(weights)
    assert isinstance(codec.module, reference_22.WanVAE_)
    assert codec.module.state_dict().keys() == reference.state_dict().keys()
    assert codec.mean == reference_22.WAN_2P2_VAE_MEAN
    assert codec.std == reference_22.WAN_2P2_VAE_STD
    pixels = torch.randn(1, 3, 5, 32, 32)
    with torch.no_grad():
        latents = codec.encode(pixels)
        expected = reference.encode(pixels, codec._scale(pixels)).permute(0, 2, 1, 3, 4)
        torch.testing.assert_close(latents, expected, rtol=0, atol=0)
        decoded = codec.decode(latents)
        codec.module.clear_cache()
        streamed = torch.cat([codec.decode(chunk, use_cache=True) for chunk in latents.split(1, 1)], dim=1)
    torch.testing.assert_close(streamed, decoded, rtol=0, atol=0)

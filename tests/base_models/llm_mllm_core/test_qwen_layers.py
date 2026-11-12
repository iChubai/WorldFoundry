# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Shared Qwen leaves preserve checkpoint names and low-precision boundaries."""

import pytest
import torch
from transformers import PretrainedConfig

from worldfoundry.base_models.llm_mllm_core.mllm.qwen.layers import QwenMLP, QwenRMSNorm, repeat_kv

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64])
@pytest.mark.parametrize("affine_dtype", [torch.float32, torch.bfloat16])
def test_qwen_rms_norm_casts_before_affine(dtype: torch.dtype, affine_dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    norm = QwenRMSNorm(8, eps=1e-5).to(affine_dtype)
    state = {"weight": torch.randn(8).to(affine_dtype)}
    norm.load_state_dict(state, strict=True)
    x = torch.randn(2, 3, 8).to(dtype)
    normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)
    expected = state["weight"] * normalized.to(dtype)
    actual = norm(x)
    assert actual.dtype == expected.dtype
    assert list(norm.state_dict()) == ["weight"]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_qwen_mlp_keeps_gated_projection_checkpoint_names() -> None:
    config = PretrainedConfig(hidden_size=8, intermediate_size=20, hidden_act="silu")
    model = QwenMLP(config).eval()
    assert {key: tuple(value.shape) for key, value in model.state_dict().items()} == {
        "gate_proj.weight": (20, 8),
        "up_proj.weight": (20, 8),
        "down_proj.weight": (8, 20),
    }
    x = torch.randn(2, 3, 8)
    expected = model.down_proj(torch.nn.functional.silu(model.gate_proj(x)) * model.up_proj(x))
    torch.testing.assert_close(model(x), expected, atol=0, rtol=0)


@pytest.mark.parametrize("repetitions", [1, 2, 3])
def test_qwen_repeat_kv_keeps_head_order(repetitions: int) -> None:
    x = torch.arange(48).reshape(1, 2, 3, 8)
    torch.testing.assert_close(repeat_kv(x, repetitions), x.repeat_interleave(repetitions, dim=1), atol=0, rtol=0)
    if repetitions == 1:
        assert repeat_kv(x, repetitions) is x

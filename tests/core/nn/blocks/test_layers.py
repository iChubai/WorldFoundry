# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""Activation helpers retain decomposed rounding and clamped detector odds."""

import pytest
import torch
import torch.nn.functional as F

from worldfoundry.core.nn.blocks.layers import Mish, inverse_sigmoid
from worldfoundry.core.nn.diffusion.convolution import zero_module

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mish_retains_decomposed_low_precision_arithmetic(dtype: torch.dtype) -> None:
    torch.manual_seed(31)
    value = torch.randn(1000, dtype=dtype) * 4
    actual = Mish()(value)
    torch.testing.assert_close(actual, value * torch.tanh(F.softplus(value)), atol=0, rtol=0)
    assert Mish().state_dict() == {}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_inverse_sigmoid_clamps_before_forming_odds(dtype: torch.dtype) -> None:
    value = torch.tensor([-1.0, 0.0, 0.1, 0.5, 0.9, 1.0, 2.0], dtype=dtype)
    clamped = value.clamp(min=0, max=1)
    expected = torch.log(clamped.clamp(min=1e-3) / (1 - clamped).clamp(min=1e-3))
    torch.testing.assert_close(inverse_sigmoid(value), expected, rtol=0, atol=0)


def test_zero_module_preserves_buffers_and_parameter_grad_flags() -> None:
    module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
    module[1].running_mean.fill_(3)
    assert zero_module(module) is module
    assert all(torch.count_nonzero(p).item() == 0 for p in module.parameters())
    assert all(p.requires_grad for p in module.parameters())
    assert torch.equal(module[1].running_mean, torch.full((4,), 3.0))

# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""NumPy schedule extraction preserves device indexing and default-dtype broadcasting."""

import numpy as np
import pytest
import torch

from worldfoundry.core.nn.checkpointing.gradient import set_grad_checkpoint
from worldfoundry.core.nn.diffusion.utils import append_zero, extract_numpy_into_tensor

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_extract_numpy_into_tensor_adds_default_dtype_zeros(dtype: torch.dtype) -> None:
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        schedule = np.arange(12, dtype=np.float64) / 7
        timesteps = torch.tensor([0, 4, 9])
        expected = torch.from_numpy(schedule)[timesteps].float()[:, None, None] + torch.zeros(3, 2, 4)
        actual = extract_numpy_into_tensor(schedule, timesteps, (3, 2, 4))
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_append_zero_preserves_empty_and_nonempty_schedule_dtype(dtype: torch.dtype) -> None:
    for count in (0, 4):
        schedule = torch.arange(count, dtype=dtype)
        actual = append_zero(schedule)
        assert actual.dtype == schedule.dtype
        assert actual.device == schedule.device
        torch.testing.assert_close(actual[:-1], schedule, rtol=0, atol=0)
        assert actual[-1].item() == 0


def test_set_grad_checkpoint_sets_the_same_attributes_on_every_child() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Sequential(torch.nn.Linear(4, 2)))
    keys = list(model.state_dict())
    set_grad_checkpoint(model, use_fp32_attention=True, gc_step=3)
    assert list(model.state_dict()) == keys
    assert all(m.grad_checkpointing and m.fp32_attention and m.grad_checkpointing_step == 3 for m in model.modules())

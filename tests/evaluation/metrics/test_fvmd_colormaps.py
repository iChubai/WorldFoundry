# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""FVMD visualizations resolve colormaps through the installed matplotlib."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest
import torch

from worldfoundry.evaluation.tasks.metrics import fvmd

pytestmark = pytest.mark.unit


@pytest.fixture
def improc(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(fvmd.__file__).resolve().parent / "vendor"))
    return importlib.import_module("fvmd.utils.improc")


def test_colorize_maps_values_to_rgb(improc: ModuleType) -> None:
    values = torch.linspace(0.0, 1.0, 12).reshape(1, 3, 4)

    color = improc.colorize(values)

    assert tuple(color.shape) == (3, 3, 4)
    assert float(color.min()) >= 0.0
    assert float(color.max()) <= 255.0


def test_trajectory_summary_draws_on_every_frame(improc: ModuleType) -> None:
    writer = improc.Summ_writer(writer=None, global_step=0, log_freq=1, fps=4, just_gif=True)
    trajs = 4 + torch.rand(1, 4, 3, 2, generator=torch.Generator().manual_seed(0)) * 24
    rgbs = torch.rand(1, 4, 3, 32, 32, generator=torch.Generator().manual_seed(1)) - 0.5

    frames = writer.summ_traj2ds_on_rgbs(
        "trajectories", trajs, rgbs, cmap="hot", linewidth=1, show_dots=False, only_return=True
    )

    assert tuple(frames.shape) == (1, 4, 3, 32, 32)

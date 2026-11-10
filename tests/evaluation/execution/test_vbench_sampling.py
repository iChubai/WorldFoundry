# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""VBench frame selection retains the upstream sampling and NumPy RNG contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytestmark = pytest.mark.unit

_RUNTIMES = (
    "vbench/runtime/vbench",
    "vbench_plus_plus/runtime/vbench2_beta_i2v",
    "vbench_plus_plus/runtime/vbench2_beta_trustworthiness",
)


@pytest.fixture(params=_RUNTIMES, ids=("vbench", "i2v", "trustworthiness"))
def sampling(request: pytest.FixtureRequest) -> ModuleType:
    source = Path(__file__).resolve().parents[3] / (
        f"worldfoundry/evals/tasks/execution/runners/{request.param}/utils.py"
    )
    spec = importlib.util.spec_from_file_location("test_vbench_sampling_runtime", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("num_frames,vlen", [(4, 40), (6, 3)])
@pytest.mark.parametrize("seed", [0, 42])
def test_random_frames_preserve_upstream_numpy_fallback(
    sampling: ModuleType, monkeypatch: pytest.MonkeyPatch, num_frames: int, vlen: int, seed: int
) -> None:
    actual_rng = np.random.RandomState(seed)
    expected_rng = np.random.RandomState(seed)
    monkeypatch.setattr(sampling, "np", SimpleNamespace(linspace=np.linspace, random=actual_rng))
    expected = sorted(expected_rng.permutation(vlen)[: min(num_frames, vlen)])
    expected += [expected[-1]] * (num_frames - len(expected))

    assert sampling.get_frame_indices(num_frames, vlen) == expected
    np.testing.assert_equal(actual_rng.get_state(), expected_rng.get_state())


def test_upstream_fallback_catches_base_exceptions(sampling: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    chooser = Mock(side_effect=KeyboardInterrupt)
    monkeypatch.setattr(sampling, "random", SimpleNamespace(choice=chooser), raising=False)
    monkeypatch.setattr(sampling, "np", SimpleNamespace(linspace=np.linspace, random=np.random.RandomState(42)))

    assert sampling.get_frame_indices(4, 40) == [15, 16, 19, 26]
    chooser.assert_called_once()


@pytest.mark.parametrize(
    "options,expected",
    [
        ({"sample": "middle"}, [4, 14, 24, 34]),
        ({"sample": "middle", "fix_start": 2}, [2, 12, 22, 32]),
        ({"sample": "fps0.5", "input_fps": 2, "max_num_frames": 3}, [2, 6, 10]),
    ],
)
def test_deterministic_frame_selection(
    sampling: ModuleType, options: dict[str, str | int], expected: list[int]
) -> None:
    assert sampling.get_frame_indices(4, 40, **options) == expected

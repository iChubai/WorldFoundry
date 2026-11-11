# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from worldfoundry.synthesis.visual_generation.show_o.show_o_runtime.models import MAGVITv2
from worldfoundry.synthesis.visual_generation.show_o.show_o_runtime.models.modeling_utils import (
    ConfigMixin,
    ModelMixin,
    register_to_config,
)
from worldfoundry.synthesis.visual_generation.show_o.worldfoundry_runtime import ShowORuntime

pytestmark = pytest.mark.unit


class _TinyModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, width: int = 2) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(width, width)


def test_vq_factory_uses_the_package_class_with_foreign_models_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    foreign = ModuleType("models")
    monkeypatch.setitem(sys.modules, "models", foreign)

    assert ShowORuntime._get_vq_model_class("magvitv2") is MAGVITv2
    assert sys.modules["models"] is foreign
    with pytest.raises(ValueError, match="not supported"):
        ShowORuntime._get_vq_model_class("unknown")


def test_from_flax_uses_diffusers_converter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _TinyModel().save_config(tmp_path)
    checkpoint = tmp_path / "diffusion_flax_model.msgpack"
    checkpoint.touch()
    calls = []

    def convert(model: _TinyModel, path: str) -> _TinyModel:
        calls.append(Path(path))
        with torch.no_grad():
            model.projection.weight.fill_(0.25)
        return model

    converter = ModuleType("diffusers.models.modeling_pytorch_flax_utils")
    monkeypatch.setattr(converter, "load_flax_checkpoint_in_pytorch_model", convert, raising=False)
    monkeypatch.setitem(sys.modules, converter.__name__, converter)

    loaded = _TinyModel.from_pretrained(tmp_path, from_flax=True, local_files_only=True, low_cpu_mem_usage=False)

    assert calls == [checkpoint]
    assert not loaded.training
    torch.testing.assert_close(loaded.projection.weight, torch.full((2, 2), 0.25), rtol=0, atol=0)

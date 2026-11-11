# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from worldfoundry.core.configuration.factory import get_obj_from_str
from worldfoundry.core.io.paths import resolve_data_path
from worldfoundry.synthesis.visual_generation.mineworld.lvm import (
    LlamaForCausalLM,
    LlamaLVM,
)
from worldfoundry.synthesis.visual_generation.mineworld.vae import VAE

pytestmark = pytest.mark.unit
CONFIG_DIR = resolve_data_path("models", "runtime", "configs", "mineworld")


@pytest.mark.parametrize("config_path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda path: path.stem)
def test_packaged_config_resolves_owned_model_classes(config_path: Path) -> None:
    config = OmegaConf.load(config_path)

    assert get_obj_from_str(config.model.target) is LlamaLVM
    assert get_obj_from_str(config.model.params.model_class) is LlamaForCausalLM
    assert get_obj_from_str(config.model.params.tokenizer_config.target) is VAE

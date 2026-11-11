# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from worldfoundry.core.configuration.factory import get_obj_from_str
from worldfoundry.synthesis.visual_generation.dino_wm import worldfoundry_runtime as runtime

pytestmark = pytest.mark.unit
PACKAGE = "worldfoundry.synthesis.visual_generation.dino_wm"


@pytest.mark.parametrize(
    "relative_path",
    [
        "encoder/dino.yaml",
        "encoder/dino_cls.yaml",
        "encoder/identity.yaml",
        "encoder/r3m.yaml",
        "encoder/resnet.yaml",
        "decoder/transposed_conv.yaml",
        "decoder/vqvae.yaml",
        "action_encoder/proprio.yaml",
        "action_encoder/repeat.yaml",
        "proprio_encoder/proprio.yaml",
        "proprio_encoder/repeat.yaml",
        "predictor/vit.yaml",
    ],
)
def test_packaged_model_config_targets_resolve_to_owned_classes(relative_path: str) -> None:
    config = OmegaConf.load(runtime.DEFAULT_CONFIG.parent / relative_path)

    assert config._target_.startswith(f"{PACKAGE}.models.")
    target = get_obj_from_str(config._target_)
    assert callable(target)
    assert target.__module__.startswith(f"{PACKAGE}.models.")


def test_r3m_config_cleanup_resolves_the_owned_class() -> None:
    from worldfoundry.synthesis.visual_generation.dino_wm.models.encoder.r3m import R3M, cleanup_config

    config = OmegaConf.create({"agent": {"_target_": "r3m.R3M", "size": 18, "langweight": 1.0}})
    cleaned = cleanup_config(config)

    assert get_obj_from_str(cleaned._target_) is R3M
    assert cleaned.langweight == 0
    assert config.agent._target_ == "r3m.R3M"


@pytest.mark.parametrize("relative_path", ["env/__init__.py", "env/pointmaze/__init__.py"])
def test_gym_registration_targets_resolve_inside_the_env_package(relative_path: str) -> None:
    tree = ast.parse((runtime.RUNTIME_DIR / relative_path).read_text())
    targets = [
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "register"
        for keyword in node.keywords
        if keyword.arg == "entry_point"
    ]
    assert targets
    repo = Path(__file__).resolve().parents[2]
    for target in targets:
        assert target.startswith(f"{PACKAGE}.env.")
        module, name = target.split(":")
        source = repo.joinpath(*module.split(".")).with_suffix(".py")
        if not source.is_file():
            source = repo.joinpath(*module.split("."), "__init__.py")
        definitions = ast.parse(source.read_text()).body
        assert any(
            (isinstance(node, ast.ClassDef) and node.name == name)
            or (isinstance(node, ast.ImportFrom) and any((alias.asname or alias.name) == name for alias in node.names))
            for node in definitions
        )

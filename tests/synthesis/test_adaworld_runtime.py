# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from omegaconf import OmegaConf

from worldfoundry.core.configuration.factory import get_obj_from_str
from worldfoundry.core.io.paths import resolve_data_path

pytestmark = pytest.mark.unit
PACKAGE = "worldfoundry.synthesis.visual_generation.adaworld.worldmodel"


def _targets(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "target":
                yield item
            else:
                yield from _targets(item)
    elif isinstance(value, list):
        for item in value:
            yield from _targets(item)


@pytest.mark.parametrize("name", ["adaworld", "adaworld_adapt_discrete_action", "adaworld_adapt_continuous_action"])
def test_packaged_model_targets_resolve_to_owned_classes(name: str) -> None:
    path = resolve_data_path("models", "runtime", "configs", "adaworld", f"worldmodel/inference/{name}.yaml")
    config = OmegaConf.to_container(OmegaConf.load(path))
    targets = list(_targets(config))
    assert targets
    for target in targets:
        assert target.startswith("worldfoundry.synthesis.visual_generation.adaworld.")
        assert callable(get_obj_from_str(target))


def test_sampler_resolves_its_guider_and_discretization_through_the_package() -> None:
    from worldfoundry.synthesis.visual_generation.adaworld.worldmodel import sample_utils
    from worldfoundry.synthesis.visual_generation.adaworld.worldmodel.vwm.modules.diffusionmodules.discretizer import (
        EDMShiftDiscretization,
    )
    from worldfoundry.synthesis.visual_generation.adaworld.worldmodel.vwm.modules.diffusionmodules.guiders import (
        VanillaCFG,
    )

    sampler = sample_utils.init_sampling(steps=5, cfg_scale=1.1, n_context_frames=6)
    assert type(sampler.discretization) is EDMShiftDiscretization
    assert type(sampler.guider) is VanillaCFG
    assert sampler.num_steps == 5
    assert sampler.num_frames == 7

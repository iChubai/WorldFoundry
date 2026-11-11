# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import ModuleType

import pytest
from packaging.version import parse

pytestmark = pytest.mark.unit


def test_load_easyanimate_components_returns_the_owned_classes(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = ModuleType("pkg_resources")
    monkeypatch.setattr(metadata, "parse_version", parse, raising=False)
    monkeypatch.setitem(sys.modules, "pkg_resources", metadata)
    from worldfoundry.synthesis.visual_generation.easyanimate.easyanimate_runtime import load_easyanimate_components
    from worldfoundry.synthesis.visual_generation.easyanimate.easyanimate_runtime.easyanimate.models import (
        name_to_autoencoder_magvit,
        name_to_transformer3d,
    )
    from worldfoundry.synthesis.visual_generation.easyanimate.easyanimate_runtime.easyanimate.pipeline.pipeline_easyanimate_inpaint import (
        EasyAnimateInpaintPipeline,
    )
    from worldfoundry.synthesis.visual_generation.easyanimate.easyanimate_runtime.easyanimate.utils.fp8_optimization import (
        convert_weight_dtype_wrapper,
    )
    from worldfoundry.synthesis.visual_generation.easyanimate.easyanimate_runtime.easyanimate.utils.utils import (
        get_image_to_video_latent,
    )

    components = load_easyanimate_components()
    assert components.name_to_autoencoder_magvit is name_to_autoencoder_magvit
    assert components.name_to_transformer3d is name_to_transformer3d
    assert components.inpaint_pipeline_cls is EasyAnimateInpaintPipeline
    assert components.multi_text_encoder_inpaint_pipeline_cls is EasyAnimateInpaintPipeline
    assert components.convert_weight_dtype_wrapper is convert_weight_dtype_wrapper
    assert components.get_image_to_video_latent is get_image_to_video_latent

# SPDX-FileCopyrightText: 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0

"""NexusScore forwards explicit and default model paths without loading weights."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from worldfoundry.evaluation.tasks.metrics.opens2v import nexus

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("source", ["defaults", "environment", "explicit"])
def test_nexus_forwards_model_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str) -> None:
    runtime = tmp_path / "runtime"
    monkeypatch.setattr(nexus, "ensure_opens2v_eval_path", lambda: runtime)
    for key in ("WORLDFOUNDRY_YOLO_WORLD_CKPT", "WORLDFOUNDRY_YOLO_CLIP_MODEL", "WORLDFOUNDRY_GME_MODEL_PATH"):
        monkeypatch.delenv(key, raising=False)
    if source in {"environment", "explicit"}:
        monkeypatch.setenv("WORLDFOUNDRY_YOLO_WORLD_CKPT", "env-yolo")
        monkeypatch.setenv("WORLDFOUNDRY_YOLO_CLIP_MODEL", "env-clip")
        monkeypatch.setenv("WORLDFOUNDRY_GME_MODEL_PATH", "env-gme")
    options: dict[str, Any] = {}
    if source == "explicit":
        options = {
            "model_config": "chosen-config.py",
            "yolo_model_path": "chosen-yolo",
            "yolo_clip_model_path": "chosen-clip",
            "gme_model_path": "chosen-gme",
        }
    loader = Mock(return_value=(object(), object(), object(), object()))
    frames = Mock(return_value=[])
    gme = Mock()
    monkeypatch.setitem(
        sys.modules,
        "get_nexusscore",
        SimpleNamespace(load_model_and_config=loader, sample_video_frames=frames, yoloworld_inference=Mock()),
    )
    monkeypatch.setitem(sys.modules, "utils.gme.gme_model", SimpleNamespace(GmeQwen2VL=gme))
    manifest = tmp_path / "inputs.json"
    manifest.write_text("{}", encoding="utf-8")
    (tmp_path / "sample.mp4").touch()

    result = nexus.compute_nexus_score(
        input_video_folder=tmp_path,
        input_image_folder=tmp_path,
        input_json_file=manifest,
        device="cpu",
        **options,
    )

    args, clip_path, device = loader.call_args.args
    assert loader.call_count == 1
    assert args.model_config == options.get(
        "model_config",
        str(runtime / "utils/yoloworld/configs/yolo_world_v2_l_vlpan_bn_2e-4_80e_8gpus_image_prompt_demo.py"),
    )
    expected_yolo = {
        "defaults": "BestWishYsh/OpenS2V-Weight/yolo_world_v2_l_image_prompt_adapter-719a7afb.pth",
        "environment": "env-yolo",
        "explicit": "chosen-yolo",
    }
    assert args.yolo_model_path == expected_yolo[source]
    assert args.work_dir is None and args.cfg_options is None
    assert (
        clip_path
        == {"defaults": "openai/clip-vit-base-patch32", "environment": "env-clip", "explicit": "chosen-clip"}[source]
    )
    assert device == "cpu"
    gme.assert_called_once_with(
        {"defaults": "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct/", "environment": "env-gme", "explicit": "chosen-gme"}[
            source
        ],
        attn_model="flash_attention_2",
        device="cpu",
    )
    frames.assert_called_once_with(str(tmp_path / "sample.mp4"), num_frames=32)
    assert result == {"mean_nexus_score": 0.0, "videos": {"sample": {"nexus_score": 0.0}}}

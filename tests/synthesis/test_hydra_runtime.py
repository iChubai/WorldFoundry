# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from worldfoundry.synthesis.visual_generation.hydra import worldfoundry_runtime as runtime_module

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("return_dict", [False, True])
def test_predict_launches_package_module_with_unchanged_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, return_dict: bool
) -> None:
    checkpoint, video, camera = (tmp_path / name for name in ("hydra.ckpt", "condition.mp4", "camera.json"))
    for path in (checkpoint, video, camera):
        path.touch()
    base = tmp_path / "wan"
    base.mkdir()
    output = tmp_path / "generated.mp4"
    captured = {}

    def execute(command: list[str], **kwargs: Any) -> dict[str, Any]:
        captured.update(command=command, **kwargs)
        return {"video": str(output), "status": "succeeded"}

    monkeypatch.setattr(runtime_module, "execute_in_tree", execute)
    runtime = runtime_module.HydraRuntime(
        checkpoint_path=checkpoint,
        base_model_path=base,
        python_executable=sys.executable,
        device="cuda:2",
        env={"HF_HUB_OFFLINE": "1"},
    )

    result = runtime.predict(
        prompt="  a moving camera  ", video_path=video, camera_json=camera, output_path=output, return_dict=return_dict
    )

    command = captured["command"]
    assert command[:3] == [
        sys.executable,
        "-m",
        "worldfoundry.synthesis.visual_generation.hydra.hydra_runtime.infer_hydra",
    ]
    args = dict(zip(command[3::2], command[4::2], strict=True))
    assert args == {
        "--cond_video": str(video),
        "--cond_json": str(camera),
        "--caption_txt": str(tmp_path / ".generated_hydra_inputs" / "caption.txt"),
        "--ckpt_path": str(checkpoint),
        "--base_dit_path": str(base / "diffusion_pytorch_model.safetensors"),
        "--base_text_encoder_path": str(base / "models_t5_umt5-xxl-enc-bf16.pth"),
        "--base_vae_path": str(base / "Wan2.1_VAE.pth"),
        "--output_path": str(output),
        "--device": "cuda:2",
        "--height": "480",
        "--width": "832",
        "--cond_frames": "77",
        "--fps": "15",
        "--num_inference_steps": "50",
        "--cfg_scale": "5.0",
        "--seed": "42",
    }
    assert Path(args["--caption_txt"]).read_text() == "a moving camera\n"
    assert captured["cwd"] == runtime.repo_root
    assert captured["env"] == {"HF_HUB_OFFLINE": "1"}
    assert "python_paths" not in captured
    assert result == ({"video": str(output), "status": "succeeded"} if return_dict else str(output))

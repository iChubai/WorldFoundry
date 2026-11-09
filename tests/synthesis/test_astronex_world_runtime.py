# SPDX-License-Identifier: Apache-2.0
"""Weight-free contracts for the vendored Astronex-World inference runtime."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image

from worldfoundry.synthesis.visual_generation.astronex_world import runtime as astronex_runtime


def test_plan_launches_the_vendored_generator_as_a_package_module(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    weights = tmp_path / "weights"
    for relative in ("transformer/config.json", "vae", "text_encoder", "tokenizer", "model-00001-of-00001.safetensors"):
        (weights / relative).parent.mkdir(parents=True, exist_ok=True)
        (weights / relative).write_bytes(b"fixture")
    image = tmp_path / "frame.png"
    Image.new("RGB", (8, 6), "white").save(image)
    runtime = astronex_runtime.AstronexWorldRuntime(checkpoint_dir=weights, python_executable=sys.executable)

    plan = runtime.build_plan(
        request={"prompt": "walk forward", "images": str(image), "latent_frames": 7, "trajectory": "w*7"},
        output_dir=tmp_path / "run",
    )

    assert runtime.preflight()["status"] == "ready"
    assert plan.command[1:3] == ("-m", astronex_runtime.ENTRYPOINT_MODULE)
    assert plan.workdir == str(tmp_path / "run")
    assert plan.env.get("PYTHONPATH", "") == ""
    assert plan.command[plan.command.index("--weights") + 1] == str(weights.resolve())


def test_generator_child_runs_the_sampler_module_without_import_path_edits(tmp_path):
    from worldfoundry.synthesis.visual_generation.astronex_world.astronex_world_runtime import astronex_env

    env = astronex_env.child_env(str(tmp_path))

    assert env["ASTRONEX_WEIGHTS"] == str(tmp_path)
    assert env.get("PYTHONPATH") == os.environ.get("PYTHONPATH")
    assert not hasattr(astronex_env, "setup")
    vendored = Path(astronex_runtime.__file__).with_name("astronex_world_runtime")
    assert (vendored / "utils" / "default_config.yaml").is_file()

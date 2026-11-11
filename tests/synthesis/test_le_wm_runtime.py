# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from worldfoundry.synthesis.visual_generation.le_wm import worldfoundry_runtime as runtime

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("suffix,artifact", [(".mp4", ".result.json"), (".json", ".json")])
@pytest.mark.parametrize("policy", ["random", "learned-policy"])
def test_package_command_preserves_policy_data_options_and_artifact_names(
    tmp_path: Path, suffix: str, artifact: str, policy: str
) -> None:
    options = {
        "config_dir": str(tmp_path / "config"),
        "config_name": "pusht",
        "policy": policy,
        "cache_dir": str(tmp_path / "cache"),
        "dataset_name": str(tmp_path / "data.h5"),
        "seed": 7,
        "num_eval": 2,
        "eval_budget": 32,
        "goal_offset_steps": 5,
        "img_size": 224,
    }
    command = runtime.build_command(
        {
            "python": "/env/python",
            "entrypoint": str(runtime.OFFICIAL_ENTRYPOINT),
            "output_path": str(tmp_path / f"result{suffix}"),
            "output_dir": str(tmp_path),
            "device": "cuda:1",
            "options": options,
        }
    )
    assert command[:3] == ["/env/python", "-m", "worldfoundry.synthesis.visual_generation.le_wm.infer"]
    assert dict(zip(command[3::2], command[4::2], strict=True)) == {
        "--config-dir": options["config_dir"],
        "--config-name": "pusht",
        "--policy": policy,
        "--output-dir": str(tmp_path),
        "--artifact-path": str(tmp_path / f"result{artifact}"),
        "--device": "cuda:1",
        "--cache-dir": options["cache_dir"],
        "--dataset-name": options["dataset_name"],
        "--seed": "7",
        "--num-eval": "2",
        "--eval-budget": "32",
        "--goal-offset-steps": "5",
        "--img-size": "224",
    }

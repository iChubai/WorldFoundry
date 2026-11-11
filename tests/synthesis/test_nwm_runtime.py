# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from worldfoundry.synthesis.visual_generation.nwm import worldfoundry_runtime as runtime

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("config_option", ["exp", "config", "config_path"])
@pytest.mark.parametrize("eval_type", ["rollout", "time"])
def test_nwm_module_command_preserves_evaluation_args(config_option: str, eval_type: str) -> None:
    command = runtime.build_command(
        {
            "python": "/env/python",
            "entrypoint": str(runtime.OFFICIAL_ENTRYPOINT),
            "output_dir": "/out",
            "options": {
                config_option: "nwm.yaml",
                "checkpoint_step": "0200000",
                "dataset": "recon,scand",
                "eval_type": eval_type,
                "num_sec_eval": 2,
                "input_fps": 8,
                "num_workers": 0,
                "batch_size": 2,
                "rollout_fps_values": "1,2,4",
                "gt": 1,
            },
        }
    )
    assert command[:3] == ["/env/python", "-m", "worldfoundry.synthesis.visual_generation.nwm.isolated_nwm_infer"]
    assert dict(zip(command[3::2], command[4::2], strict=True)) == {
        "--output_dir": "/out",
        "--exp": "nwm.yaml",
        "--ckp": "0200000",
        "--datasets": "recon,scand",
        "--eval_type": eval_type,
        "--num_sec_eval": "2",
        "--input_fps": "8",
        "--num_workers": "0",
        "--batch_size": "2",
        "--rollout_fps_values": "1,2,4",
        "--gt": "1",
    }

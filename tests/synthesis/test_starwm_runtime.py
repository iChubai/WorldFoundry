# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from worldfoundry.synthesis.visual_generation.starwm import worldfoundry_runtime as runtime

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("model_option", ["served_model_id", "api_model_id", "vllm_model_id", "upstream_model_id"])
@pytest.mark.parametrize("mode", ["think", "nothink"])
def test_client_module_command_preserves_endpoint_options_and_model_aliases(model_option: str, mode: str) -> None:
    command = runtime.build_command(
        {
            "python": "/env/python",
            "entrypoint": str(runtime.OFFICIAL_ENTRYPOINT),
            "output_path": "/out/fallback.jsonl",
            "options": {
                model_option: "served-starwm",
                "dataset_path": "/data/prompts.json",
                "output_file": "/out/selected.jsonl",
                "mode": mode,
                "api_base": "http://localhost:18000",
                "api_key": "test-key",
                "max_workers": 2,
            },
        }
    )
    assert command[:3] == [
        "/env/python",
        "-m",
        "worldfoundry.synthesis.visual_generation.starwm.offline_infer.run_inference_client",
    ]
    assert dict(zip(command[3::2], command[4::2], strict=True)) == {
        "--input_file": "/data/prompts.json",
        "--output_file": "/out/selected.jsonl",
        "--mode": mode,
        "--api_base": "http://localhost:18000",
        "--api_key": "test-key",
        "--model_id": "served-starwm",
        "--max_workers": "2",
    }

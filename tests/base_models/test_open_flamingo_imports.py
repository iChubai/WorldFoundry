# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
"""The OpenFlamingo package remains import-light without a top-level alias."""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def test_open_flamingo_package_preserves_import_paths() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; paths = list(sys.path); "
            "import worldfoundry.base_models.llm_mllm_core.mllm.open_flamingo; "
            "assert paths == sys.path; "
            "assert 'open_flamingo' not in sys.modules; "
            "assert 'torch' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

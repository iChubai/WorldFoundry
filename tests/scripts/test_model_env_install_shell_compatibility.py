from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _bash_function(source: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}$", source, flags=re.MULTILINE | re.DOTALL)
    assert match is not None, f"missing shell function {name}"
    return match.group(0)


def test_model_env_installer_avoids_bash_namerefs() -> None:
    source = (REPO_ROOT / "scripts/setup/model_env_install.sh").read_text(encoding="utf-8")

    assert "local -n" not in source
    assert "WORLDFOUNDRY_PIP_INDEX_ARGS" in source


def test_solarwm_public_wan_ids_reach_flash_attn_post_install() -> None:
    source = (REPO_ROOT / "scripts/setup/model_env_install.sh").read_text(encoding="utf-8")
    script = "\n".join(
        (
            "set -eu",
            _bash_function(source, "canonical_model_id"),
            _bash_function(source, "post_install_model_env"),
            "install_solarwm_flash_attn() { printf '%s\\n' solarwm-flash-attn; }",
            'post_install_model_env "$(canonical_model_id "$1")" /tmp/solarwm-test-env',
        )
    )

    for model_id in (
        "solarwm",
        "solarwm-wan22",
        "solarwm-wan22-5b",
        "solarwm-wan2.2-5b",
        "solarwm-wan2.2-14b",
    ):
        result = subprocess.run(
            ["bash", "-c", script, "bash", model_id],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, (model_id, result.stderr)
        assert result.stdout.splitlines()[-1] == "solarwm-flash-attn", (model_id, result.stdout)


def test_solarwm_ltx_h3_ids_keep_explicit_interpreter_routes() -> None:
    source = (REPO_ROOT / "scripts/setup/model_env_install.sh").read_text(encoding="utf-8")
    script = "\n".join((_bash_function(source, "canonical_model_id"), 'canonical_model_id "$1"'))

    for model_id in ("solarwm-ltx-22b", "solarwm-h3-33b"):
        result = subprocess.run(
            ["bash", "-c", script, "bash", model_id],
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == model_id

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


def test_public_variant_ids_resolve_to_runtime_conda_env() -> None:
    expected_targets = {
        "dreamzero-droid": "dreamzero",
        "gamma-world-causal-few-step": "gamma-world",
        "hy-embodied-vla-umi": "hy-embodied-vla",
        "kandinsky6-lite": "kandinsky6",
        "kandinsky6-lite-distill": "kandinsky6",
        "kandinsky6-lite-pretrain": "kandinsky6",
        "kandinsky6-pro": "kandinsky6",
        "kandinsky6-pro-distill": "kandinsky6",
        "kandinsky6-pro-pretrain": "kandinsky6",
        "lapa-7b-openx": "lapa",
        "lingbot-va-base": "lingbot-va",
        "lingbot-vla-4b": "lingbot-vla",
        "lingbot-vla-v2-6b": "lingbot-vla-v2",
        "lingbot-world-act": "lingbot-world",
        "local-checkpoint": "mira",
        "matrix-game-3.5-first-person": "matrix-game-3.5",
        "spatial-forcing-libero-spatial": "spatial-forcing",
        "x-wam-robocasa-sft": "x-wam",
    }
    source = (REPO_ROOT / "scripts/setup/model_env_install.sh").read_text(encoding="utf-8")
    script = "\n".join(
        (
            _bash_function(source, "canonical_model_id"),
            'for model_id in "$@"; do printf "%s " "$model_id"; canonical_model_id "$model_id"; done',
        )
    )
    result = subprocess.run(
        ["bash", "-c", script, "bash", *expected_targets],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert dict(line.split(" ", 1) for line in result.stdout.splitlines()) == expected_targets

    from worldfoundry.cli.model_run import load_model_run_schema
    from worldfoundry.runtime.conda import load_runtime_conda_env_spec
    from worldfoundry.zoo.runtime.profiles import load_runtime_profile

    for public_id, canonical_id in expected_targets.items():
        schema = load_model_run_schema(public_id)
        profile = load_runtime_profile(schema.model_id, check_conda_env_exists=False)
        runtime_env = (profile.conda_env or {}).get("env_name") or (profile.conda_env or {}).get("resolved_env_name")
        install_spec = load_runtime_conda_env_spec(canonical_id)
        assert install_spec is not None, public_id
        assert install_spec.model_id == canonical_id
        assert install_spec.env_name == runtime_env, public_id


def test_unrelated_model_ids_keep_identity_canonicalization() -> None:
    model_ids = (
        "dreamzero",
        "gamma-world",
        "hy-embodied-vla",
        "kandinsky6",
        "lapa",
        "lingbot-va",
        "lingbot-vla",
        "lingbot-vla-v2",
        "lingbot-world",
        "matrix-game-3.5",
        "mira",
        "spatial-forcing",
        "wan2.1-t2v-1.3b",
        "x-wam",
    )
    source = (REPO_ROOT / "scripts/setup/model_env_install.sh").read_text(encoding="utf-8")
    script = "\n".join(
        (
            _bash_function(source, "canonical_model_id"),
            'for model_id in "$@"; do canonical_model_id "$model_id"; done',
        )
    )
    result = subprocess.run(
        ["bash", "-c", script, "bash", *model_ids],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == list(model_ids)

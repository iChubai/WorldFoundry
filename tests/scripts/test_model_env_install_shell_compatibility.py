from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_model_env_installer_avoids_bash_namerefs() -> None:
    source = (REPO_ROOT / "scripts/setup/model_env_install.sh").read_text(encoding="utf-8")

    assert "local -n" not in source
    assert "WORLDFOUNDRY_PIP_INDEX_ARGS" in source


def _shell_function(name: str) -> str:
    import re

    source = (REPO_ROOT / "scripts/setup/model_env_install.sh").read_text(encoding="utf-8")
    match = re.search(rf"^{name}\(\) \{{\n.*?^\}}$", source, re.MULTILINE | re.DOTALL)
    assert match is not None
    return match.group()


def test_setup_commands_follow_dependencies_and_precede_validation() -> None:
    source = _shell_function("install_dedicated_env")
    setup = source.index('install_env_setup_commands "$spec_json" "$env_prefix"')
    assert source.index('"${pip_args[@]}"') < setup
    assert source.index('"${req_pip_args[@]}"') < setup
    assert source.index('"${edit_pip_args[@]}"') < setup
    assert setup < source.index('verify_env_imports "$env_prefix"')
    unified = _shell_function("install_model_env")
    assert unified.index('install_unified_env "$env_prefix"') < unified.index("install_env_setup_commands")
    assert 'if [[ "$VERIFY_ONLY" != "1" ]]; then\n      install_env_setup_commands' in unified


def test_setup_commands_execute_in_the_selected_conda_environment(tmp_path: Path) -> None:
    import json
    import shlex
    import subprocess
    import sys

    commands = ["python -m pip install --no-build-isolation 'extension @ git+https://example.invalid/source'"]
    fake_conda = tmp_path / "conda"
    log = tmp_path / "commands.jsonl"
    fake_conda.write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        f"with open({str(log)!r}, 'a') as stream: stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    fake_conda.chmod(0o755)
    script = "\n".join(
        [
            "set -euo pipefail",
            f"PYTHON_BIN={shlex.quote(sys.executable)}",
            f"CONDA_EXE_PATH={shlex.quote(str(fake_conda))}",
            _shell_function("json_array_lines"),
            _shell_function("install_env_setup_commands"),
            "install_env_setup_commands "
            + shlex.quote(json.dumps({"setup_commands": commands}))
            + " '/env with spaces'",
            "install_env_setup_commands '{}' '/unused-env'",
        ]
    )
    subprocess.run(["bash", "-c", script], check=True, timeout=30)
    assert [json.loads(line) for line in log.read_text().splitlines()] == [
        ["run", "-p", "/env with spaces", "bash", "-c", commands[0]]
    ]

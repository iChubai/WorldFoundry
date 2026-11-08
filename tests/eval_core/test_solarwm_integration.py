from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CATALOG_DIR = REPO_ROOT / "worldfoundry" / "data" / "models" / "catalog"
PIPELINE_RUNNER_TARGET = "worldfoundry.evaluation.models.runners.pipeline:WorldFoundryPipelineRunner"


def test_solarwm_runtime_spec_resolves_in_tree_launcher(monkeypatch, tmp_path: Path) -> None:
    from worldfoundry.synthesis.visual_generation.runtime_manifest import (
        resolve_runtime_manifest,
        runtime_spec,
    )

    source = tmp_path / "SolarWM"
    source.mkdir()
    monkeypatch.setenv("WORLDFOUNDRY_SOLARWM_SOURCE", str(source))

    spec = runtime_spec("solarwm")
    runtime_root, entrypoint, blocked_reason = resolve_runtime_manifest(spec)

    assert runtime_root == source.resolve()
    assert entrypoint is not None and entrypoint.name == "infer.py"
    assert "worldfoundry/synthesis/visual_generation/solarwm" in entrypoint.as_posix()
    assert blocked_reason == ""


def _fake_solarwm_source(root: Path) -> Path:
    package = root / "src" / "solarwm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "__main__.py").write_text(
        """
import pathlib
import sys

overrides = [sys.argv[index + 1] for index, value in enumerate(sys.argv) if value == "--set"]
output = next(value.split("=", 1)[1] for value in overrides if value.startswith("runtime.output_dir="))
artifact = pathlib.Path(output) / "generation" / "model_self_forcing_nfe4" / "slot-000000" / "video.mp4"
artifact.parent.mkdir(parents=True)
artifact.write_bytes(b"fake-solarwm-mp4")
""".lstrip(),
        encoding="utf-8",
    )
    config = root / "configs" / "infer.yaml"
    config.parent.mkdir()
    config.write_text("schema: solarwm.run.v1\naction: infer\n", encoding="utf-8")
    return config


def test_solarwm_compatibility_launcher_normalizes_official_artifact(tmp_path: Path) -> None:
    from worldfoundry.synthesis.visual_generation.solarwm.infer import run

    source = tmp_path / "SolarWM"
    config = _fake_solarwm_source(source)
    output = tmp_path / "artifacts" / "solarwm.mp4"
    args = argparse.Namespace(
        source_dir=source,
        config=config,
        base_model_dir=tmp_path / "base",
        checkpoint_dir=tmp_path / "checkpoint",
        data_root=tmp_path / "data",
        output_path=output,
        test_index=None,
        sample_count=1,
        seed=17,
    )

    assert run(args) == output.resolve()
    assert output.read_bytes() == b"fake-solarwm-mp4"


def test_solarwm_command_uses_explicit_inference_only_overrides(tmp_path: Path) -> None:
    from worldfoundry.synthesis.visual_generation.solarwm.infer import build_official_command

    args = argparse.Namespace(
        config=tmp_path / "infer.yaml",
        base_model_dir=tmp_path / "base",
        checkpoint_dir=tmp_path / "checkpoint",
        data_root=tmp_path / "data",
        test_index="recipes/custom/test-index.jsonl.gz",
        sample_count=1,
        seed=9,
    )
    command = build_official_command(args, run_root=(tmp_path / "run").resolve())

    assert command[:4] == [sys.executable, "-m", "solarwm", "infer"]
    assert "inference.output_layout=transaction_v1" in command
    assert "validation.sample_count=1" in command
    assert "validation.selection_seed=9" in command
    assert "data.test_index=recipes/custom/test-index.jsonl.gz" in command
    assert "train" not in command
    assert "preencode" not in command


def test_solarwm_binding_rejects_multi_sample_and_absolute_index(tmp_path: Path) -> None:
    from worldfoundry.synthesis.visual_generation.solarwm.infer import build_official_command

    args = argparse.Namespace(
        config=tmp_path / "infer.yaml",
        base_model_dir=tmp_path / "base",
        checkpoint_dir=tmp_path / "checkpoint",
        data_root=tmp_path / "data",
        test_index="recipes/custom/test-index.jsonl.gz",
        sample_count=2,
        seed=9,
    )
    with pytest.raises(ValueError, match="single-MP4"):
        build_official_command(args, run_root=(tmp_path / "run").resolve())

    args.sample_count = 1
    args.test_index = "/tmp/test-index.jsonl.gz"
    with pytest.raises(ValueError, match="relative POSIX"):
        build_official_command(args, run_root=(tmp_path / "run").resolve())

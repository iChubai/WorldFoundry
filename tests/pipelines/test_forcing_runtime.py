"""Weight-free contracts for the Forcing subprocess bridge."""

from pathlib import Path

import pytest

from worldfoundry.core.io.paths import resolve_data_path
from worldfoundry.synthesis.visual_generation.forcing import runtime as runtime_module
from worldfoundry.synthesis.visual_generation.forcing.runtime import CausalForcingRuntime


class _ValidatedRequest(Exception):
    """Stop predict after validation, before preparing or launching inference."""


@pytest.fixture
def validation_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    def forbid_execution(*args: object, **kwargs: object) -> None:
        pytest.fail("Validation must not load checkpoints, initialize CUDA, or launch inference")

    def stop_after_validation(self: object, root: Path) -> Path:
        raise _ValidatedRequest

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbid_execution)
    monkeypatch.setattr(torch, "load", forbid_execution)
    monkeypatch.setattr(runtime_module.subprocess, "run", forbid_execution)
    monkeypatch.setattr(runtime_module._ForcingRuntime, "_runtime_cwd", stop_after_validation)


@pytest.mark.parametrize(
    ("device", "inherited", "expected"),
    [
        ("cuda:1", "2,5", "5"),
        ("cuda", "2,5", "2,5"),
        ("cuda", "", ""),
        ("cuda:1", None, "1"),
        ("cuda", None, "0"),
    ],
)
def test_subprocess_env_maps_cuda_visibility(
    monkeypatch: pytest.MonkeyPatch,
    validation_boundary: None,
    device: str,
    inherited: str | None,
    expected: str,
) -> None:
    if inherited is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", inherited)
    runtime = CausalForcingRuntime(device=device)

    assert runtime._subprocess_env()["CUDA_VISIBLE_DEVICES"] == expected
    if inherited is None:
        assert "CUDA_VISIBLE_DEVICES" not in runtime_module.os.environ
    else:
        assert runtime_module.os.environ["CUDA_VISIBLE_DEVICES"] == inherited


_CAUSAL_CONFIG_DIR = resolve_data_path("models", "runtime", "configs", "causal_forcing")


@pytest.mark.parametrize("config_path", sorted(_CAUSAL_CONFIG_DIR.glob("*framewise*.yaml")), ids=lambda p: p.name)
@pytest.mark.parametrize("i2v,num_output_frames", [(False, 2), (True, 3)])
def test_predict_accepts_published_causal_framewise_counts(
    tmp_path: Path, validation_boundary: None, config_path: Path, i2v: bool, num_output_frames: int
) -> None:
    runtime = CausalForcingRuntime(config_path=config_path, checkpoint_path=tmp_path / "missing.pt")

    with pytest.raises(_ValidatedRequest):
        runtime.predict(output_path=tmp_path / "output.mp4", i2v=i2v, num_output_frames=num_output_frames)


@pytest.mark.parametrize("i2v,num_output_frames", [(False, 0), (False, -1), (True, 1)])
def test_predict_rejects_empty_causal_framewise_generation(
    tmp_path: Path, validation_boundary: None, i2v: bool, num_output_frames: int
) -> None:
    runtime = CausalForcingRuntime(config_path=_CAUSAL_CONFIG_DIR / "causal_forcing_dmd_framewise.yaml")

    with pytest.raises(ValueError, match="num_output_frames"):
        runtime.predict(output_path=tmp_path / "output.mp4", i2v=i2v, num_output_frames=num_output_frames)


@pytest.mark.parametrize(
    "i2v,num_output_frames,valid",
    [(False, 3, True), (True, 4, True), (False, 2, False), (True, 3, False)],
)
def test_predict_preserves_causal_chunkwise_counts(
    tmp_path: Path, validation_boundary: None, i2v: bool, num_output_frames: int, valid: bool
) -> None:
    runtime = CausalForcingRuntime()

    with pytest.raises(_ValidatedRequest if valid else ValueError):
        runtime.predict(output_path=tmp_path / "output.mp4", i2v=i2v, num_output_frames=num_output_frames)


def test_predict_merges_causal_default_config_with_interpolation(tmp_path: Path, validation_boundary: None) -> None:
    (tmp_path / "default_config.yaml").write_text(
        "independent_first_frame: false\nblock_size: 1\nnum_frame_per_block: 3\n", encoding="utf-8"
    )
    config_path = tmp_path / "framewise.yaml"
    config_path.write_text("num_frame_per_block: ${block_size}\n", encoding="utf-8")
    runtime = CausalForcingRuntime(config_path=config_path)

    with pytest.raises(_ValidatedRequest):
        runtime.predict(output_path=tmp_path / "output.mp4", num_output_frames=2)


@pytest.mark.parametrize("num_output_frames,valid", [(1, True), (4, True), (3, False)])
def test_predict_respects_causal_independent_first_frame(
    tmp_path: Path, validation_boundary: None, num_output_frames: int, valid: bool
) -> None:
    (tmp_path / "default_config.yaml").write_text(
        "independent_first_frame: true\nnum_frame_per_block: 3\n", encoding="utf-8"
    )
    config_path = tmp_path / "selected.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    runtime = CausalForcingRuntime(config_path=config_path)

    with pytest.raises(_ValidatedRequest if valid else ValueError):
        runtime.predict(output_path=tmp_path / "output.mp4", num_output_frames=num_output_frames)


def test_subprocess_env_prepends_shared_torchvision_shim(validation_boundary: None) -> None:
    runtime = CausalForcingRuntime()
    compat_dir = runtime_module.TORCHVISION_VIDEO_COMPAT_DIR

    env = runtime._subprocess_env(compat_dir=compat_dir)

    assert (compat_dir / "sitecustomize.py").is_file()
    assert env["PYTHONPATH"].split(runtime_module.os.pathsep)[0] == str(compat_dir)

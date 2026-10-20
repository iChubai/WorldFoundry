from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "workspace" / "validate_generated_videos.py"
SPEC = importlib.util.spec_from_file_location("worldfoundry_validate_generated_videos", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
validate_generated_videos = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validate_generated_videos
SPEC.loader.exec_module(validate_generated_videos)


def test_duration_uses_ffprobe_value() -> None:
    assert validate_generated_videos._duration_seconds(
        {"duration": "2.75"}, decoded_frames=12, fps=4.0
    ) == pytest.approx(2.75)


@pytest.mark.parametrize("raw_duration", [None, "N/A", "nan", "-1"])
def test_duration_falls_back_to_decoded_frames_and_fps(raw_duration: str | None) -> None:
    assert validate_generated_videos._duration_seconds(
        {"duration": raw_duration}, decoded_frames=12, fps=4.0
    ) == pytest.approx(3.0)


def test_duration_reports_zero_without_usable_timing() -> None:
    assert validate_generated_videos._duration_seconds({"duration": "N/A"}, decoded_frames=12, fps=0.0) == 0.0


def test_video_validation_uses_decoded_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"video")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"metadata": {"result": {"artifact_path": str(video), "artifact_sha256": "stale"}}}))
    monkeypatch.setattr(
        validate_generated_videos,
        "_probe",
        lambda path: {
            "width": 2,
            "height": 2,
            "nb_read_frames": "2",
            "avg_frame_rate": "1/1",
        },
    )
    monkeypatch.setattr(
        validate_generated_videos,
        "_decode_statistics",
        lambda path: {
            "decoded_frames": 2,
            "decoded_width": 2,
            "decoded_height": 2,
            "inconsistent_frame_shape": False,
            "pixel_std": 8.0,
            "max_temporal_mean_absdiff": 4.0,
            "pixel_min": 0,
            "pixel_max": 255,
            "pixel_mean": 32.0,
        },
    )

    row = validate_generated_videos.validate_video(manifest, video)

    assert row["ok"] is True
    assert row["decoded_frames"] == 2
    assert row["duration_seconds"] == 2.0


def test_video_validation_rejects_undecodable_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"broken")

    def fail_probe(path: Path) -> dict:
        raise RuntimeError("ffprobe could not decode video")

    monkeypatch.setattr(validate_generated_videos, "_probe", fail_probe)

    row = validate_generated_videos.validate_video(tmp_path / "manifest.json", video)

    assert row["ok"] is False
    assert "could not decode" in row["error"]

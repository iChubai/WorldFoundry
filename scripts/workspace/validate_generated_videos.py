#!/usr/bin/env python3
"""Strictly validate generated videos referenced by Workspace manifests.

Unlike the matrix runner's lightweight artifact check, this command decodes
every frame and verifies that the result is non-empty, non-constant, and has
temporal variation.  It emits one JSON object per video so that validation can
be appended to a persistent JSONL audit trail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_video_paths(manifest: dict[str, Any], manifest_path: Path) -> list[Path]:
    candidates: list[str] = []
    preview = str(manifest.get("preview_video") or "")
    if preview:
        candidates.append(preview)
    for artifact in manifest.get("artifacts") or ():
        value = str(artifact or "")
        if Path(value).suffix.lower() in VIDEO_SUFFIXES:
            candidates.append(value)

    paths: list[Path] = []
    for value in candidates:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        if path not in paths:
            paths.append(path)
    return paths


def _expected_sha256(manifest: dict[str, Any], path: Path) -> str:
    """Find a manifest digest only when it is unambiguously tied to the video."""

    metadata = manifest.get("metadata") or {}
    raw_result = metadata.get("result") or {}
    results = raw_result if isinstance(raw_result, list) else [raw_result]
    for result in results:
        if not isinstance(result, dict):
            continue
        artifact_path = Path(str(result.get("artifact_path") or "")).expanduser()
        digest = str(
            result.get("artifact_sha256")
            or result.get("video_sha256")
            or result.get("sha256")
            or ""
        )
        if digest and artifact_path.name == path.name:
            return digest
    return ""


def _probe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,nb_read_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"ffprobe failed: {completed.stderr[-2000:]}")
    streams = json.loads(completed.stdout).get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream, found {len(streams)}")
    return streams[0]


def _parse_fps(value: str) -> float:
    numerator, separator, denominator = str(value or "").partition("/")
    if not separator:
        return float(numerator)
    denominator_value = float(denominator)
    return float(numerator) / denominator_value if denominator_value else 0.0


def _duration_seconds(probe: dict[str, Any], *, decoded_frames: int, fps: float) -> float:
    """Return a finite positive duration, falling back to decoded geometry."""

    try:
        duration = float(probe.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if math.isfinite(duration) and duration > 0.0:
        return duration
    if decoded_frames > 0 and math.isfinite(fps) and fps > 0.0:
        return decoded_frames / fps
    return 0.0


def _decode_statistics(path: Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - exercised in runtime env validation
        raise RuntimeError("opencv-python is required for strict video validation") from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the video")

    decoded_frames = 0
    pixel_count = 0
    pixel_sum = 0.0
    pixel_square_sum = 0.0
    pixel_min = 255
    pixel_max = 0
    max_temporal_mean_absdiff = 0.0
    previous = None
    decoded_width = 0
    decoded_height = 0
    inconsistent_shape = False
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            if decoded_frames == 0:
                decoded_width, decoded_height = width, height
            elif (width, height) != (decoded_width, decoded_height):
                inconsistent_shape = True

            decoded_frames += 1
            pixel_count += int(frame.size)
            pixel_sum += float(frame.sum(dtype="float64"))
            pixel_square_sum += float((frame.astype("float64") ** 2).sum())
            pixel_min = min(pixel_min, int(frame.min()))
            pixel_max = max(pixel_max, int(frame.max()))
            if previous is not None:
                difference = cv2.absdiff(frame, previous)
                max_temporal_mean_absdiff = max(
                    max_temporal_mean_absdiff, float(difference.mean())
                )
            previous = frame
    finally:
        capture.release()

    if not decoded_frames or not pixel_count:
        raise RuntimeError("video decoded zero frames")
    pixel_mean = pixel_sum / pixel_count
    variance = max(0.0, pixel_square_sum / pixel_count - pixel_mean * pixel_mean)
    return {
        "decoded_frames": decoded_frames,
        "decoded_height": decoded_height,
        "decoded_width": decoded_width,
        "inconsistent_frame_shape": inconsistent_shape,
        "max_temporal_mean_absdiff": max_temporal_mean_absdiff,
        "pixel_max": pixel_max,
        "pixel_mean": pixel_mean,
        "pixel_min": pixel_min,
        "pixel_std": math.sqrt(variance),
    }


def validate_video(manifest_path: Path, path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row: dict[str, Any] = {
        "manifest": str(manifest_path),
        "path": str(path),
        "ok": False,
    }
    if not path.is_file():
        row["error"] = "video does not exist"
        return row
    try:
        probe = _probe(path)
        statistics = _decode_statistics(path)
        actual_sha256 = _sha256(path)
        expected_sha256 = _expected_sha256(manifest, path)
        expected_frames_text = str(probe.get("nb_read_frames") or probe.get("nb_frames") or "0")
        expected_frames = int(expected_frames_text) if expected_frames_text.isdigit() else 0
        width = int(probe.get("width") or 0)
        height = int(probe.get("height") or 0)
        fps = _parse_fps(str(probe.get("avg_frame_rate") or "0"))
        duration_seconds = _duration_seconds(
            probe,
            decoded_frames=int(statistics["decoded_frames"]),
            fps=fps,
        )
        sha256_verified = bool(expected_sha256) and actual_sha256 == expected_sha256
        errors: list[str] = []
        if expected_frames and statistics["decoded_frames"] != expected_frames:
            errors.append(
                f"decoded {statistics['decoded_frames']} frames but ffprobe reported {expected_frames}"
            )
        if (statistics["decoded_width"], statistics["decoded_height"]) != (width, height):
            errors.append("decoded dimensions do not match ffprobe")
        if statistics["inconsistent_frame_shape"]:
            errors.append("decoded frames do not have a consistent shape")
        if statistics["pixel_std"] <= 1.0:
            errors.append("video is blank or nearly constant")
        if statistics["decoded_frames"] > 1 and statistics["max_temporal_mean_absdiff"] <= 0.1:
            errors.append("video has no meaningful temporal change")
        if expected_sha256 and not sha256_verified:
            errors.append("video sha256 does not match manifest")

        row.update(
            decoded_frames=statistics["decoded_frames"],
            duration_seconds=duration_seconds,
            fps=fps,
            height=height,
            max_temporal_mean_absdiff=statistics["max_temporal_mean_absdiff"],
            pixel_max=statistics["pixel_max"],
            pixel_mean=statistics["pixel_mean"],
            pixel_min=statistics["pixel_min"],
            pixel_std=statistics["pixel_std"],
            sha256=actual_sha256,
            sha256_verified=sha256_verified,
            size_bytes=path.stat().st_size,
            temporal_change=(
                statistics["decoded_frames"] <= 1
                or statistics["max_temporal_mean_absdiff"] > 0.1
            ),
            width=width,
        )
        if errors:
            row["errors"] = errors
        else:
            row["ok"] = True
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def _rows(manifests: Iterable[Path]) -> Iterable[dict[str, Any]]:
    line = 0
    for raw_manifest_path in manifests:
        manifest_path = raw_manifest_path.expanduser().resolve()
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            video_paths = _manifest_video_paths(manifest, manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            line += 1
            yield {
                "error": f"{type(exc).__name__}: {exc}",
                "line": line,
                "manifest": str(manifest_path),
                "ok": False,
            }
            continue
        if not video_paths:
            line += 1
            yield {
                "error": "manifest references no video artifacts",
                "line": line,
                "manifest": str(manifest_path),
                "ok": False,
            }
            continue
        for path in video_paths:
            line += 1
            row = validate_video(manifest_path, path)
            row["line"] = line
            yield row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSONL atomically to this path in addition to stdout.",
    )
    args = parser.parse_args()
    rows = list(_rows(args.manifest))
    output = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    sys.stdout.write(output)
    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(output, encoding="utf-8")
        os.replace(temporary, output_path)
    return 0 if rows and all(row.get("ok") for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

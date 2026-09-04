#!/usr/bin/env python3
"""Strictly validate WorldFoundry video artifact manifests with PyAV."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import av
import numpy as np


VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}
REPO_ROOT = Path(__file__).resolve().parents[2]


def _is_video_row(row: dict[str, Any]) -> bool:
    mime_type = str(row.get("mime_type") or "").lower()
    kind = str(row.get("kind") or "").lower()
    uri = str(row.get("uri") or "")
    return (
        mime_type.startswith("video/")
        or kind in {"generated_video", "video"}
        or Path(uri).suffix.lower() in VIDEO_SUFFIXES
    )


def _workspace_video_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a Studio ``manifest.json`` payload into artifact-ref rows."""

    values: list[str] = []
    preview_video = str(payload.get("preview_video") or "")
    if preview_video:
        values.append(preview_video)
    for artifact in payload.get("artifacts") or ():
        if isinstance(artifact, str):
            values.append(artifact)
        elif isinstance(artifact, dict) and _is_video_row(artifact):
            uri = str(artifact.get("uri") or artifact.get("path") or "")
            if uri:
                values.append(uri)

    unique_videos = list(
        dict.fromkeys(value for value in values if Path(value).suffix.lower() in VIDEO_SUFFIXES)
    )
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    result = metadata.get("result") if isinstance(metadata.get("result"), dict) else {}
    video_sha = str(
        result.get("video_sha256")
        or metadata.get("video_sha256")
        or payload.get("video_sha256")
        or ""
    )
    artifact_path = str(result.get("artifact_path") or "")
    artifact_sha = str(result.get("artifact_sha256") or "")

    rows: list[dict[str, Any]] = []
    for uri in unique_videos:
        row: dict[str, Any] = {
            "kind": "generated_video",
            "mime_type": "video/mp4" if Path(uri).suffix.lower() == ".mp4" else "video/*",
            "uri": uri,
        }
        # Studio manifests describe one primary preview. Only attach its
        # checksum; applying it to secondary videos would create false
        # mismatches.
        if video_sha and (uri == preview_video or len(unique_videos) == 1):
            row["sha256"] = video_sha
        elif artifact_sha and artifact_path and Path(uri).name == Path(artifact_path).name:
            # In-tree Workspace runtimes use ``artifact_path`` /
            # ``artifact_sha256`` for the primary generated video.  Bind the
            # digest by filename so it is never copied onto secondary preview
            # or control videos listed by the same manifest.
            row["sha256"] = artifact_sha
        rows.append(row)
    return rows


def _matrix_video_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert completed rows from a Workspace model matrix into artifacts."""

    rows: list[dict[str, Any]] = []
    for model in payload.get("models") or ():
        if not isinstance(model, dict) or model.get("status") != "completed":
            continue
        validation = model.get("validation")
        if not isinstance(validation, dict):
            continue
        for video in validation.get("videos") or ():
            if not isinstance(video, dict):
                continue
            uri = str(video.get("path") or video.get("uri") or "")
            if not uri:
                continue
            row: dict[str, Any] = {
                "kind": "generated_video",
                "mime_type": "video/mp4" if Path(uri).suffix.lower() == ".mp4" else "video/*",
                "model_id": str(model.get("model_id") or model.get("id") or ""),
                "source_manifest": str(
                    video.get("manifest") or validation.get("manifest_path") or ""
                ),
                "uri": uri,
            }
            sha256 = str(video.get("sha256") or "")
            if sha256:
                row["sha256"] = sha256
            rows.append(row)
    return rows


def _expand_manifest_value(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from _expand_manifest_value(item)
        return
    if not isinstance(value, dict):
        raise ValueError(f"manifest entry must be an object, got {type(value).__name__}")
    if "uri" in value or "mime_type" in value or "kind" in value:
        yield value
        return
    if "models" in value:
        yield from _matrix_video_rows(value)
        return
    if "artifacts" in value or "preview_video" in value:
        yield from _workspace_video_rows(value)


def _load_manifest_rows(manifest: Path) -> list[tuple[int, dict[str, Any]]]:
    """Load JSON, JSON-array, JSONL, or Studio workspace manifests."""

    text = manifest.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows: list[tuple[int, dict[str, Any]]] = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON/JSONL at {manifest}:{line_number}: {exc.msg}"
                ) from exc
            rows.extend((line_number, row) for row in _expand_manifest_value(value))
        return rows
    return [(1, row) for row in _expand_manifest_value(payload)]


def _resolve_uri(manifest: Path, uri: str) -> Path:
    expanded_uri = os.path.expandvars(uri)
    for token in ("${WORLDFOUNDRY_REPO_ROOT}", "$WORLDFOUNDRY_REPO_ROOT"):
        expanded_uri = expanded_uri.replace(token, str(REPO_ROOT))
    path = Path(expanded_uri).expanduser()

    candidates: list[Path]
    if path.is_absolute():
        candidates = [path]
        # Workspace manifests are intended to be portable, but older runs
        # sometimes captured either ``/artifacts/...`` or the absolute path
        # of a different WorldFoundry checkout. Rebase only well-known run
        # roots instead of guessing for arbitrary absolute paths.
        for anchor in ("artifacts", "tmp"):
            if anchor in path.parts:
                anchor_index = path.parts.index(anchor)
                candidates.append(REPO_ROOT.joinpath(*path.parts[anchor_index:]))
    else:
        candidates = [Path.cwd() / path, manifest.parent / path]

    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_video(path: Path) -> dict[str, object]:
    count = 0
    value_sum = 0.0
    square_sum = 0.0
    value_count = 0
    minimum = 255
    maximum = 0
    width = height = 0
    fps = 0.0
    duration_seconds = 0.0
    first_frame_time: float | None = None
    last_frame_time: float | None = None
    last_frame: np.ndarray | None = None
    temporal_change = False

    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"video stream is missing: {path}")
        stream = container.streams.video[0]
        if stream.average_rate is not None:
            fps = float(stream.average_rate)
        if stream.duration is not None and stream.time_base is not None:
            duration_seconds = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_seconds = float(container.duration / av.time_base)
        for frame in container.decode(stream):
            if frame.time is not None:
                frame_time = float(frame.time)
                if first_frame_time is None:
                    first_frame_time = frame_time
                last_frame_time = frame_time
            pixels = frame.to_ndarray(format="rgb24")
            if pixels.ndim != 3 or pixels.shape[2] != 3:
                raise ValueError(f"unexpected decoded frame shape {pixels.shape}: {path}")
            if count == 0:
                height, width = pixels.shape[:2]
            elif pixels.shape[:2] != (height, width):
                raise ValueError(f"decoded frame geometry changed in {path}")
            minimum = min(minimum, int(pixels.min()))
            maximum = max(maximum, int(pixels.max()))
            float_pixels = pixels.astype(np.float64)
            value_sum += float(float_pixels.sum())
            square_sum += float(np.square(float_pixels).sum())
            value_count += int(pixels.size)
            if last_frame is not None and not temporal_change:
                temporal_change = not np.array_equal(pixels, last_frame)
            last_frame = pixels
            count += 1

    if count == 0 or value_count == 0:
        raise ValueError(f"video decoded no frames: {path}")
    mean = value_sum / value_count
    variance = max(square_sum / value_count - mean * mean, 0.0)
    std = math.sqrt(variance)
    if maximum == 0 or maximum == minimum or std == 0.0:
        raise ValueError(f"video is black or pixel-constant: {path}")
    if count > 1 and not temporal_change:
        raise ValueError(f"all decoded video frames are identical: {path}")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        if (
            first_frame_time is not None
            and last_frame_time is not None
            and last_frame_time >= first_frame_time
        ):
            duration_seconds = last_frame_time - first_frame_time
            if fps > 0.0:
                duration_seconds += 1.0 / fps
        elif fps > 0.0:
            duration_seconds = count / fps
    return {
        "decoded_frames": count,
        "duration_seconds": duration_seconds,
        "width": width,
        "height": height,
        "fps": fps,
        "pixel_min": minimum,
        "pixel_max": maximum,
        "pixel_mean": mean,
        "pixel_std": std,
        "temporal_change": temporal_change,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results: list[dict[str, object]] = []
    failures = 0
    matched_artifacts = 0
    for raw_manifest in args.manifests:
        manifest = raw_manifest.expanduser().resolve()
        for line_number, row in _load_manifest_rows(manifest):
            if not _is_video_row(row):
                continue
            matched_artifacts += 1
            path = _resolve_uri(manifest, str(row["uri"]))
            base_result = {
                "manifest": str(manifest),
                "line": line_number,
                "path": str(path),
            }
            for key in ("model_id", "source_manifest"):
                if row.get(key):
                    base_result[key] = str(row[key])
            try:
                if not path.is_file():
                    raise FileNotFoundError(f"video artifact is missing: {path}")
                actual_sha = _sha256(path)
                expected_sha = str(row.get("sha256") or "")
                if expected_sha and actual_sha != expected_sha:
                    raise ValueError(
                        f"SHA256 mismatch for {path}: expected {expected_sha!r}, got {actual_sha}"
                    )
                result = {
                    **base_result,
                    "ok": True,
                    "sha256": actual_sha,
                    "sha256_verified": bool(expected_sha),
                    "size_bytes": path.stat().st_size,
                    **_validate_video(path),
                }
            except Exception as exc:
                failures += 1
                result = {
                    **base_result,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)

    if matched_artifacts == 0:
        raise ValueError("no video artifacts were found in the supplied manifests")
    rendered = "".join(json.dumps(row, sort_keys=True) + "\n" for row in results)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

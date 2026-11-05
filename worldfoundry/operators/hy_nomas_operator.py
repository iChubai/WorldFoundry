# Copyright 2026 The Hy-Nomas Team and The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured action timing and canonical camera trajectory adaptation for Hy-Nomas."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import torch


def _timestamp(value: Any) -> float:
    parts = str(value).split(":")
    if len(parts) > 3:
        raise ValueError(f"invalid action timestamp: {value}")
    return sum(float(part) * 60**i for i, part in enumerate(reversed(parts)))


def action_windows(prompt: str, frames: int, fps: float) -> list[tuple[int, int, int]] | None:
    """Convert the released JSON action timestamps into non-overlapping latent windows."""
    try:
        obj = json.loads(prompt)
    except ValueError:
        return None
    segments = obj.get("segments") if isinstance(obj, dict) else None
    if not isinstance(segments, list) or not segments or not isinstance(segments[0], dict):
        return None
    actions = segments[0].get("contained_actions", [])
    if isinstance(actions, dict):
        actions = list(actions.values())
    if not isinstance(actions, list) or not actions:
        return None
    spans = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict) or "start_time" not in action or "end_time" not in action:
            return None
        start, end = [
            max(0, min(frames, int(max(_timestamp(action[k]), 0) * fps) // 4)) for k in ("start_time", "end_time")
        ]
        if end > start:
            spans.append((start, end, index))
    if not spans:
        return None
    spans.sort()
    cursor, windows = 0, []
    for start, end, index in spans:
        if start < cursor:
            raise ValueError("action time windows must not overlap")
        if start > cursor:
            windows.append((cursor, start, -1))
        windows.append((start, end, index))
        cursor = end
    if cursor < frames:
        windows.append((cursor, frames, -1))
    return None if len(windows) == len(actions) == 1 else windows


def realtime_camera(
    interactions: list[str],
    frames: int,
    *,
    pose: Any = None,
    segments: list[Mapping[str, Any]] | None = None,
    fps: int = 24,
) -> tuple[torch.Tensor, Any]:
    """Sample held keys and reuse the shared OpenCV camera-to-world integrator."""
    import numpy as np

    from worldfoundry.core.geometry.trajectory import rollout_wasd_camera_actions

    tokens = dict(
        zip(
            ("forward", "backward", "left", "right", "camera_up", "camera_down", "camera_l", "camera_r"),
            "wsad ikjl".replace(" ", ""),
        )
    )

    def keys(values: Any) -> str:
        return "".join(tokens.get(value, value if value in "wasdijkl" else "") for value in values)

    held = [keys(interactions)] * frames
    if segments:
        rows, end = [], 0.0
        for segment in segments:
            end += float(segment.get("duration", 0))
            rows.append((end, keys(segment.get("keys", ()))))
        held = [next((value for end, value in rows if (i + 0.5) / fps < end), "") for i in range(frames)]
    local = rollout_wasd_camera_actions(held, num_frames=frames + 1)[1:]
    anchor = np.eye(4, dtype=np.float32) if pose is None else pose
    poses = anchor[None] @ local
    return torch.from_numpy(poses).unsqueeze(0), poses[-1].copy()

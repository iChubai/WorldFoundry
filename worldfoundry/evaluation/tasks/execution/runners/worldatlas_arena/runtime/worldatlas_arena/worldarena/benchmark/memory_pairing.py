"""Locate revisit events in a generated rollout from its estimated camera trajectory.

Memory scoring needs to know which generated frames look at the same place. Deriving
that from the ground-truth action program is unsound: models under- and over-execute
motion, so symmetric action indices land at different positions and a frame-pair score
ends up measuring control error rather than memory decay. Everything here works from
the estimated pose track instead, so a model that fails to walk all the way back is
reported as not having returned rather than as having forgotten.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np


# Below this departure extent a monocular pose track carries no usable translation
# scale, so pure-rotation loops fall back to a rotation-only pose distance.
DEGENERATE_TRANSLATION_EXTENT = 1e-6
DEFAULT_ROTATION_WEIGHT = 1.0
DEFAULT_MAX_PAIR_DISTANCE = 0.15
DEFAULT_MIN_GAP_SECONDS = 1.0
# A rollout must get at least this many pair-thresholds away from its anchor before
# coming back counts as a revisit; otherwise the camera never left and first and last
# frames match trivially.
DEFAULT_MIN_DEPARTURE_FACTOR = 3.0
# Two frames being close is not enough to call the second one a revisit: the camera has
# to have gone somewhere in between. Without this, any slow continuous motion pairs
# every frame with its own recent past, since the pair threshold is relative to the
# trajectory's total extent.
DEFAULT_MIN_EXCURSION_FACTOR = 3.0


@dataclass(frozen=True, slots=True)
class RevisitPair:
    """Two frames that the estimated trajectory places at the same viewpoint."""

    first_index: int
    revisit_index: int
    pose_distance: float
    translation_distance: float
    rotation_degrees: float
    gap_seconds: float
    excursion: float

    def as_dict(self) -> dict[str, float | int]:
        """Serialize the pair for metric detail payloads."""
        return {
            "first_index": int(self.first_index),
            "revisit_index": int(self.revisit_index),
            "pose_distance": round(float(self.pose_distance), 6),
            "translation_distance": round(float(self.translation_distance), 6),
            "rotation_degrees": round(float(self.rotation_degrees), 4),
            "gap_seconds": round(float(self.gap_seconds), 4),
            "excursion": round(float(self.excursion), 6),
        }


@dataclass(frozen=True, slots=True)
class RevisitStructure:
    """The revisit events a rollout actually performed, with its departure evidence."""

    pairs: tuple[RevisitPair, ...]
    turnaround_index: int
    frame_count: int
    fps: float
    translation_extent: float
    rotation_extent_degrees: float
    departure_distance: float
    required_departure_distance: float
    scale_is_degenerate: bool

    @property
    def departed(self) -> bool:
        """Whether the rollout moved far enough from its anchor to test memory."""
        return self.departure_distance >= self.required_departure_distance

    @property
    def returned(self) -> bool:
        """Whether the rollout came back to any previously visited viewpoint."""
        return bool(self.pairs)

    def gaps_seconds(self) -> tuple[float, ...]:
        """Time gaps of every revisit pair, in seconds."""
        return tuple(pair.gap_seconds for pair in self.pairs)

    def as_details(self) -> dict[str, Any]:
        """Summarize the structure for metric detail payloads."""
        return {
            "revisit_pair_count": len(self.pairs),
            "turnaround_index": int(self.turnaround_index),
            "frame_count": int(self.frame_count),
            "fps": round(float(self.fps), 4),
            "translation_extent": round(float(self.translation_extent), 6),
            "rotation_extent_degrees": round(float(self.rotation_extent_degrees), 4),
            "departure_distance": round(float(self.departure_distance), 4),
            "required_departure_distance": round(float(self.required_departure_distance), 4),
            "departed": self.departed,
            "returned": self.returned,
            "scale_is_degenerate": self.scale_is_degenerate,
        }


def load_pose_array(path: Path) -> np.ndarray:
    """Load an ``[N, 4, 4]`` camera-to-world pose track from a directory, NPY, or NPZ."""
    path = Path(path)
    if path.is_dir():
        for name in ("poses_c2w.npy", "poses.npy", "camera_poses.npy"):
            candidate = path / name
            if candidate.exists():
                return load_pose_array(candidate)
        raise FileNotFoundError(f"pose directory has no supported pose file: {path}")
    if path.suffix.lower() == ".npz":
        payload = np.load(path)
        for key in ("poses_c2w", "poses", "data"):
            if key in payload:
                poses = payload[key]
                break
        else:
            raise KeyError(f"pose archive has no poses_c2w/poses/data array: {path}")
    else:
        poses = np.load(path)
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"expected pose array [N,4,4], got {poses.shape} from {path}")
    if not np.isfinite(poses).all():
        raise ValueError(f"pose array contains non-finite values: {path}")
    return poses


@dataclass(frozen=True, slots=True)
class PoseTrack:
    """An estimated camera trajectory aligned back to its source video frames."""

    poses_c2w: np.ndarray
    frame_indices: np.ndarray
    intrinsics: np.ndarray | None
    source: str
    sidecar_dir: Path

    def video_frame(self, pose_index: int) -> int:
        """Map a pose-track index to the frame index in the decoded video."""
        return int(self.frame_indices[int(pose_index)])

    def intrinsics_at(self, pose_index: int) -> np.ndarray:
        """Return the 3x3 intrinsics for one pose, raising when none were exported."""
        if self.intrinsics is None:
            raise ValueError("pose track has no intrinsics; geometric memory scoring needs them")
        return self.intrinsics[min(int(pose_index), len(self.intrinsics) - 1)]


def _pose_sidecar_candidates(prediction_path: Path, runtime: dict[str, Any]) -> list[Path]:
    """List the sidecar locations searched for an estimated trajectory."""
    base = prediction_path.parent
    stem = prediction_path.stem
    candidates: list[Path] = []
    for key in ("pose_path", "prediction_pose_path", "vipe_annotation_path"):
        value = runtime.get(key)
        if not value:
            continue
        candidate = Path(str(value)).expanduser()
        candidates.append(candidate if candidate.is_absolute() else base / candidate)
    candidates.extend(
        [
            base / "annotations_vipe" / f"{stem}_vipe_ann",
            base / f"{stem}_vipe_ann",
            base / f"{stem}_poses_c2w.npy",
            base / f"{stem}_poses.npy",
        ]
    )
    return candidates


def resolve_pose_sidecar(prediction_path: Path, runtime: dict[str, Any] | None = None) -> tuple[np.ndarray, str]:
    """Find and load the estimated camera trajectory beside a prediction video."""
    prediction_path = Path(prediction_path).resolve()
    for candidate in _pose_sidecar_candidates(prediction_path, dict(runtime or {})):
        if candidate.exists():
            return load_pose_array(candidate), str(candidate)
    raise FileNotFoundError(
        f"memory metrics require an estimated camera-pose sidecar beside {prediction_path}"
    )


def resolve_pose_track(prediction_path: Path, runtime: dict[str, Any] | None = None) -> PoseTrack:
    """Load poses together with the video frame indices and intrinsics they belong to.

    A ViPE export only annotates the frames it could solve, so pose index and video
    frame index are not interchangeable. Carrying the mapping keeps a revisit pair
    pointing at the frames it was actually derived from.
    """
    prediction_path = Path(prediction_path).resolve()
    for candidate in _pose_sidecar_candidates(prediction_path, dict(runtime or {})):
        if not candidate.exists():
            continue
        poses = load_pose_array(candidate)
        directory = candidate if candidate.is_dir() else candidate.parent
        indices_path = directory / "frame_indices.npy"
        intrinsics_path = directory / "intrinsics.npy"
        frame_indices = (
            np.asarray(np.load(indices_path), dtype=np.int64)
            if indices_path.exists()
            else np.arange(len(poses), dtype=np.int64)
        )
        if len(frame_indices) != len(poses):
            frame_indices = np.arange(len(poses), dtype=np.int64)
        intrinsics = None
        if intrinsics_path.exists():
            loaded = np.asarray(np.load(intrinsics_path), dtype=np.float64)
            if loaded.ndim == 3 and loaded.shape[1:] == (3, 3):
                intrinsics = loaded
        return PoseTrack(
            poses_c2w=poses,
            frame_indices=frame_indices,
            intrinsics=intrinsics,
            source=str(candidate),
            sidecar_dir=directory,
        )
    raise FileNotFoundError(
        f"memory metrics require an estimated camera-pose sidecar beside {prediction_path}"
    )


def _rotation_angles_degrees(rotations: np.ndarray) -> np.ndarray:
    """Pairwise geodesic SO(3) angles for a stack of rotation matrices, in degrees."""
    flat = np.asarray(rotations, dtype=np.float64).reshape(len(rotations), 9)
    traces = flat @ flat.T
    angles = np.degrees(np.arccos(np.clip((traces - 1.0) * 0.5, -1.0, 1.0)))
    # arccos near 1 amplifies float32 pose round-off into a spurious self-angle, so the
    # exactly-zero diagonal is restored rather than left at that noise floor.
    np.fill_diagonal(angles, 0.0)
    return angles


def trajectory_extent(poses_c2w: np.ndarray) -> tuple[float, float]:
    """Return how far the trajectory departs from its anchor in translation and rotation."""
    poses = np.asarray(poses_c2w, dtype=np.float64)
    translations = poses[:, :3, 3] - poses[0, :3, 3]
    translation_extent = float(np.max(np.linalg.norm(translations, axis=1)))
    anchor = poses[0, :3, :3]
    relative = np.einsum("ba,nbc->nac", anchor, poses[:, :3, :3])
    traces = np.trace(relative, axis1=1, axis2=2)
    angles = np.degrees(np.arccos(np.clip((traces - 1.0) * 0.5, -1.0, 1.0)))
    return translation_extent, float(np.max(angles))


def pose_distance_matrix(
    poses_c2w: np.ndarray,
    *,
    rotation_weight: float = DEFAULT_ROTATION_WEIGHT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Build scale-free pairwise pose distances plus their translation and rotation parts.

    Translation is divided by the trajectory's own departure extent because monocular
    pose estimates have arbitrary metric scale. When a rollout barely translates at all
    — an in-place rotation loop — that normalizer is meaningless noise, so the distance
    degenerates to rotation only.
    """
    poses = np.asarray(poses_c2w, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        raise ValueError(f"expected at least two [4,4] poses, got {poses.shape}")

    translations = poses[:, :3, 3]
    deltas = translations[:, None, :] - translations[None, :, :]
    translation_distances = np.linalg.norm(deltas, axis=2)
    rotation_degrees = _rotation_angles_degrees(poses[:, :3, :3])

    translation_extent, _ = trajectory_extent(poses)
    degenerate = translation_extent <= DEGENERATE_TRANSLATION_EXTENT
    normalized_translation = (
        np.zeros_like(translation_distances)
        if degenerate
        else translation_distances / translation_extent
    )
    distances = normalized_translation + float(rotation_weight) * (rotation_degrees / 180.0)
    return distances, normalized_translation, rotation_degrees, degenerate


def find_revisit_pairs(
    poses_c2w: np.ndarray,
    *,
    fps: float,
    max_pair_distance: float = DEFAULT_MAX_PAIR_DISTANCE,
    min_gap_seconds: float = DEFAULT_MIN_GAP_SECONDS,
    rotation_weight: float = DEFAULT_ROTATION_WEIGHT,
    min_departure_factor: float = DEFAULT_MIN_DEPARTURE_FACTOR,
    min_excursion_factor: float = DEFAULT_MIN_EXCURSION_FACTOR,
) -> RevisitStructure:
    """Pair every frame with the earlier viewpoint it returns to.

    For each frame ``j`` the geometrically closest earlier frame ``i`` is selected,
    subject to a minimum time gap and to the camera having actually travelled away from
    ``i`` before coming back. Both tests are read off the estimated trajectory, so no
    action labels are involved and a model that mis-executes its actions is judged on
    where it actually went.
    """
    poses = np.asarray(poses_c2w, dtype=np.float64)
    frame_count = len(poses)
    frame_rate = max(float(fps), 1e-6)
    min_gap_frames = max(int(round(float(min_gap_seconds) * frame_rate)), 1)

    distances, translation_distances, rotation_degrees, degenerate = pose_distance_matrix(
        poses, rotation_weight=rotation_weight
    )
    translation_extent, rotation_extent = trajectory_extent(poses)

    # Departure is measured in the same normalized pose units as the pair threshold, so
    # the two criteria stay comparable. Normalized translation peaks at exactly 1.0 by
    # construction, which is why a rollout that translates at all clears the bar while
    # an in-place rotation must accumulate real angle to do so.
    departure_distance = float(np.max(distances[0]))
    required_departure = float(min_departure_factor) * float(max_pair_distance)
    required_excursion = float(min_excursion_factor) * float(max_pair_distance)

    pairs: list[RevisitPair] = []
    threshold = float(max_pair_distance)
    for revisit_index in range(min_gap_frames, frame_count):
        window = distances[revisit_index, : revisit_index - min_gap_frames + 1]
        if window.size == 0:
            continue
        first_index = int(np.argmin(window))
        distance = float(window[first_index])
        if distance > threshold:
            continue
        between = distances[first_index, first_index + 1 : revisit_index]
        excursion = float(np.max(between)) if between.size else 0.0
        if excursion < required_excursion:
            continue
        pairs.append(
            RevisitPair(
                first_index=first_index,
                revisit_index=revisit_index,
                pose_distance=distance,
                translation_distance=float(translation_distances[revisit_index, first_index]),
                rotation_degrees=float(rotation_degrees[revisit_index, first_index]),
                gap_seconds=(revisit_index - first_index) / frame_rate,
                excursion=excursion,
            )
        )

    return RevisitStructure(
        pairs=tuple(pairs),
        turnaround_index=int(np.argmax(distances[0])),
        frame_count=frame_count,
        fps=frame_rate,
        translation_extent=translation_extent,
        rotation_extent_degrees=rotation_extent,
        departure_distance=departure_distance,
        required_departure_distance=required_departure,
        scale_is_degenerate=degenerate,
    )


def select_diverse_pairs(
    pairs: tuple[RevisitPair, ...] | list[RevisitPair],
    *,
    max_pairs: int,
) -> tuple[RevisitPair, ...]:
    """Subsample pairs while preserving the spread of time gaps.

    The forgetting curve is fitted over gap length, so thinning must not collapse the
    sample onto one gap. Pairs are bucketed by gap and the tightest geometric match in
    each bucket is kept. The extreme gaps are retained verbatim because they anchor the
    head and tail of the fitted decay.
    """
    ordered = sorted(pairs, key=lambda pair: pair.gap_seconds)
    if max_pairs <= 0 or len(ordered) <= max_pairs:
        return tuple(ordered)

    buckets: dict[int, RevisitPair] = {}
    shortest = ordered[0].gap_seconds
    longest = ordered[-1].gap_seconds
    span = max(longest - shortest, 1e-8)
    for pair in ordered:
        index = min(int((pair.gap_seconds - shortest) / span * max_pairs), max_pairs - 1)
        current = buckets.get(index)
        if current is None or pair.pose_distance < current.pose_distance:
            buckets[index] = pair
    buckets[0] = ordered[0]
    buckets[max_pairs - 1] = ordered[-1]
    return tuple(sorted(buckets.values(), key=lambda pair: pair.gap_seconds))


def observation_revisit_split(
    structure: RevisitStructure,
    *,
    min_segment_frames: int = 2,
) -> tuple[range, range]:
    """Split a rollout into its outbound and return segments at the turnaround.

    The split point is where the camera is farthest from its anchor, which is a purely
    geometric read on the trajectory the model actually produced. Point-cloud memory
    scoring uses these ranges to decide which geometry counts as observed and which as
    regenerated from memory.
    """
    if not structure.returned:
        raise ValueError("cannot split a rollout that never revisits an earlier viewpoint")
    turnaround = int(structure.turnaround_index)
    if turnaround < min_segment_frames:
        raise ValueError(
            f"outbound segment is too short: {turnaround} frames before the turnaround"
        )
    if structure.frame_count - turnaround < min_segment_frames:
        raise ValueError(
            f"return segment is too short: {structure.frame_count - turnaround} frames"
        )
    return range(0, turnaround), range(turnaround, structure.frame_count)


def fit_exponential_half_life(
    gaps_seconds: list[float] | tuple[float, ...],
    values: list[float] | tuple[float, ...],
    *,
    min_points: int = 5,
    min_r_squared: float = 0.3,
    min_value_range: float = 0.02,
    max_tau_gap_multiple: float = 10.0,
) -> dict[str, Any]:
    """Fit ``f(gap) = a * exp(-gap / tau) + b`` and report the resulting half-life.

    The offset ``b`` is swept rather than solved jointly: with it held fixed the model
    is linear in log-space, so every candidate offset has a closed-form least-squares
    solution and the best is chosen by fit quality. That avoids a nonlinear solver and
    its convergence failures on short, noisy pair sets. The sweep is refined around its
    own best coarse guess because the recovered decay constant is sensitive to the
    offset being slightly wrong.
    """
    gaps = np.asarray(gaps_seconds, dtype=np.float64)
    scores = np.asarray(values, dtype=np.float64)
    if gaps.shape != scores.shape:
        raise ValueError("gaps and values must have the same shape")
    finite = np.isfinite(gaps) & np.isfinite(scores)
    gaps = gaps[finite]
    scores = scores[finite]
    if len(gaps) < int(min_points):
        raise ValueError(f"half-life fitting needs at least {min_points} pairs, got {len(gaps)}")
    gap_span = float(np.ptp(gaps))
    if gap_span <= 0.0:
        raise ValueError("half-life fitting needs revisit pairs at more than one time gap")
    if float(np.ptp(scores)) < float(min_value_range):
        raise ValueError(
            "revisit fidelity is effectively flat across time gaps, so no decay is measurable"
        )

    total_variance = float(np.sum((scores - float(np.mean(scores))) ** 2))
    if total_variance <= 1e-12:
        raise ValueError("revisit fidelity has no variance to fit")

    def _fit_at_offset(offset: float) -> dict[str, Any] | None:
        residual = scores - offset
        if float(np.min(residual)) <= 1e-9:
            return None
        slope, intercept = np.polyfit(gaps, np.log(residual), 1)
        if slope >= 0.0:
            return None
        tau = -1.0 / float(slope)
        if tau > float(max_tau_gap_multiple) * gap_span:
            return None
        amplitude = float(np.exp(intercept))
        predicted = amplitude * np.exp(-gaps / tau) + offset
        residual_sum = float(np.sum((scores - predicted) ** 2))
        return {
            "amplitude": amplitude,
            "offset": float(offset),
            "tau_seconds": tau,
            "half_life_seconds": tau * math.log(2.0),
            "r_squared": 1.0 - residual_sum / total_variance,
            "point_count": int(len(gaps)),
        }

    ceiling = max(float(np.min(scores)) - 1e-6, 0.0)
    best: dict[str, Any] | None = None
    for offset in np.linspace(0.0, ceiling, num=41):
        candidate = _fit_at_offset(float(offset))
        if candidate is not None and (best is None or candidate["r_squared"] > best["r_squared"]):
            best = candidate

    if best is not None:
        window = ceiling / 40.0
        low = max(best["offset"] - window, 0.0)
        high = min(best["offset"] + window, ceiling)
        for offset in np.linspace(low, high, num=41):
            candidate = _fit_at_offset(float(offset))
            if candidate is not None and candidate["r_squared"] > best["r_squared"]:
                best = candidate

    if best is None:
        raise ValueError("no decaying exponential fits the revisit pairs")
    if best["r_squared"] < float(min_r_squared):
        raise ValueError(
            f"half-life fit is unreliable: R^2 {best['r_squared']:.3f} below {min_r_squared}"
        )
    return best


__all__ = [
    "DEFAULT_MAX_PAIR_DISTANCE",
    "DEFAULT_MIN_DEPARTURE_FACTOR",
    "DEFAULT_MIN_EXCURSION_FACTOR",
    "DEFAULT_MIN_GAP_SECONDS",
    "DEFAULT_ROTATION_WEIGHT",
    "PoseTrack",
    "RevisitPair",
    "RevisitStructure",
    "find_revisit_pairs",
    "fit_exponential_half_life",
    "load_pose_array",
    "observation_revisit_split",
    "pose_distance_matrix",
    "resolve_pose_sidecar",
    "resolve_pose_track",
    "select_diverse_pairs",
    "trajectory_extent",
]

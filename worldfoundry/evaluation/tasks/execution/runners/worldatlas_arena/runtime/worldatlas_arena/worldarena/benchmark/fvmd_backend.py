"""Frechet Video Motion Distance (FVMD) backend.

Upstream's ``fvmd.fvmd.fvmd()`` entry point cannot be used here. It drives both
sides through a single loop bounded by ``len(gen_dataset)`` and then reshapes the
reference histograms with the prediction's window count, so an independently
sized reference corpus is either silently truncated, silently duplicated, or a
shape error. This module therefore drives the decoupled upstream subfunctions
(``tracking_fullseq`` -> ``calc_hist`` -> Frechet distance) once per side.

``tests/test_fvmd_metric.py`` pins the orchestration against upstream
``fvmd()`` on equal-sized input so the split cannot silently drift.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from worldarena.benchmark.distribution_clips import (
    ClipSamplingSpec,
    atomic_save_npy,
    atomic_write_text,
    build_clip_array,
    clip_set_signature,
    reference_runtime,
)
from worldarena.benchmark.metrics.base import normalize_score
from worldarena.common.checkpoints import checkpoint_env, filter_checkpoint_env_overrides


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FVMD_REPO_CANDIDATES = (
    PROJECT_ROOT / "thirdparty" / "FVMD-frechet-video-motion-distance",
)
DEFAULT_ANALYSIS_SIZE = 256

# Upstream's histogram layout is fixed: 400 tracked points fold into a 20x20
# grid of 4x4 cells, 16 frames fold into 4 temporal cubes, and each cell carries
# 8 angle bins. Velocity and acceleration each contribute 4*4*4*8 = 512 dims.
FVMD_TRACKED_POINTS = 400
FVMD_CLIP_FRAMES = 16
FVMD_FEATURE_DIM = 1024


def _as_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _resolve_fvmd_repo_root(runtime: dict[str, Any]) -> Path | None:
    configured = runtime.get("repo_root")
    if configured:
        repo_root = Path(str(configured)).expanduser()
        if not repo_root.is_absolute():
            repo_root = PROJECT_ROOT / repo_root
        if not repo_root.exists():
            raise FileNotFoundError(f"FVMD repo_root does not exist: {repo_root}")
        if not (repo_root / "fvmd").is_dir():
            raise FileNotFoundError(
                f"FVMD repo_root does not contain an fvmd package: {repo_root}"
            )
        return repo_root.resolve()

    for candidate in DEFAULT_FVMD_REPO_CANDIDATES:
        if (candidate / "fvmd").is_dir():
            return candidate.resolve()
    return None


def _load_fvmd_modules(runtime: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    """Import the upstream subfunctions this backend orchestrates."""
    repo_root = _resolve_fvmd_repo_root(runtime)
    if repo_root is not None:
        repo_path = str(repo_root)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

    try:
        tracking = importlib.import_module("fvmd.keypoint_tracking")
        features = importlib.import_module("fvmd.extract_motion_features")
        frechet = importlib.import_module("fvmd.frechet_distance")
        datasets = importlib.import_module("fvmd.datasets.video_datasets")
    except Exception as exc:  # pragma: no cover - depends on optional FVMD runtime
        raise ModuleNotFoundError(
            "FVMD package is unavailable; install fvmd or set "
            "benchmark.metric_runtime.fvmd.repo_root to the cloned repository. "
            f"Import error: {exc.__class__.__name__}: {exc}"
        ) from exc

    return (
        {
            "tracking": tracking,
            "calc_hist": features.calc_hist,
            "activation_statistics": frechet.calculate_activation_statistics,
            "frechet_distance": frechet.calculate_frechet_distance,
            "VideoDatasetNP": datasets.VideoDatasetNP,
        },
        repo_root,
    )


def _prepare_runtime_environment(runtime: dict[str, Any]) -> None:
    os.environ.update(checkpoint_env())
    os.environ.update(filter_checkpoint_env_overrides(dict(runtime.get("env", {}))))


def _build_tracker(modules: dict[str, Any], runtime: dict[str, Any]) -> Any:
    import torch

    tracking = modules["tracking"]
    device_ids = [int(value) for value in runtime.get("device_ids", [0])]
    if not torch.cuda.is_available():
        raise RuntimeError("FVMD keypoint tracking requires a CUDA device")

    model = tracking.Pips(stride=_as_positive_int(runtime.get("tracking_stride"), 8))
    model = model.to(f"cuda:{device_ids[0]}")
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    state_dict = tracking.load_state_dict_from_url(tracking.PIPS_WEIGHTS, progress=False)
    model.module.load_state_dict(state_dict["model_state_dict"])
    model.eval()
    return model


def _motion_features(
    clips: np.ndarray,
    *,
    model: Any,
    modules: dict[str, Any],
    spec: ClipSamplingSpec,
    runtime: dict[str, Any],
) -> np.ndarray:
    """Track one clip set and fold it into ``(clips, 1024)`` motion histograms."""
    import torch
    from torch.utils.data import DataLoader

    tracking = modules["tracking"]
    calc_hist = modules["calc_hist"]
    points = _as_positive_int(runtime.get("tracking_points"), FVMD_TRACKED_POINTS)
    iters = _as_positive_int(runtime.get("tracking_iters"), 16)

    # seq_len/stride are pinned to the clip length so each clip yields exactly one
    # tracking window; upstream's stride=1 default would emit overlapping windows.
    dataset = modules["VideoDatasetNP"](
        clips,
        img_size=spec.analysis_size,
        seq_len=spec.clip_frame_count,
        stride=spec.clip_frame_count,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    velocities: list[Any] = []
    accelerations: list[Any] = []
    with torch.no_grad():
        for sample in loader:
            _, velocity, acceleration = tracking.tracking_fullseq(
                model,
                sample,
                None,
                N=points,
                iters=iters,
                S_max=spec.clip_frame_count,
            )
            velocities.extend(velocity)
            accelerations.extend(acceleration)

    if not velocities:
        raise ValueError("FVMD keypoint tracking produced no motion windows")

    velocity_array = torch.cat(velocities, dim=0).cpu().numpy()
    acceleration_array = torch.cat(accelerations, dim=0).cpu().numpy()
    window_count = velocity_array.shape[0]
    return np.concatenate(
        (
            calc_hist(velocity_array).reshape(window_count, -1),
            calc_hist(acceleration_array).reshape(window_count, -1),
        ),
        axis=1,
    )


def _validate_spec(spec: ClipSamplingSpec) -> None:
    if spec.clip_frame_count != FVMD_CLIP_FRAMES:
        raise ValueError(
            "FVMD's histogram layout is defined for "
            f"{FVMD_CLIP_FRAMES}-frame windows; clip_frame_count={spec.clip_frame_count} "
            "would change the feature dimension and make scores incomparable"
        )
    if spec.analysis_size % 32 != 0:
        raise ValueError(
            f"FVMD keypoint tracking requires analysis_size divisible by 32, got {spec.analysis_size}"
        )


def compute_frechet_video_motion_distance(
    *,
    prediction_entries: Sequence[tuple[Path, str]],
    reference_entries: Sequence[tuple[Path, str]],
    normalization: dict[str, Any],
    runtime: dict[str, Any],
    output_dir: Path,
    reference_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Frechet distance between prediction and reference motion distributions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_spec = ClipSamplingSpec.from_runtime(
        runtime, default_analysis_size=DEFAULT_ANALYSIS_SIZE
    )
    reference_spec = ClipSamplingSpec.from_runtime(
        reference_runtime(runtime), default_analysis_size=DEFAULT_ANALYSIS_SIZE
    )
    _validate_spec(prediction_spec)
    _validate_spec(reference_spec)

    _prepare_runtime_environment(runtime)
    modules, repo_root = _load_fvmd_modules(runtime)

    reference_signature = clip_set_signature(reference_entries, reference_spec)
    prediction_signature = clip_set_signature(prediction_entries, prediction_spec)
    cache_dir = reference_cache_dir or (output_dir / "reference_cache")
    cached_features = cache_dir / f"{reference_signature}.npy"
    cached_report = cache_dir / f"{reference_signature}.json"
    reference_cache_hit = cached_features.is_file()

    details: dict[str, Any] = {
        "reference_cache_hit": reference_cache_hit,
        "reference_signature": reference_signature,
        "prediction_signature": prediction_signature,
        "prediction_clip_spec": prediction_spec.describe(),
        "reference_clip_spec": reference_spec.describe(),
    }

    prediction_clips, prediction_report = build_clip_array(
        prediction_entries, prediction_spec, label="prediction"
    )
    details["prediction"] = prediction_report.describe()

    reference_clips: np.ndarray | None = None
    if reference_cache_hit:
        reference_features = np.load(cached_features)
        details["reference"] = (
            json.loads(cached_report.read_text(encoding="utf-8"))
            if cached_report.is_file()
            else {"candidate_video_count": len(reference_entries)}
        )
    else:
        reference_clips, reference_report = build_clip_array(
            reference_entries, reference_spec, label="reference"
        )
        details["reference"] = reference_report.describe()

    model = _build_tracker(modules, runtime)
    prediction_features = _motion_features(
        prediction_clips,
        model=model,
        modules=modules,
        spec=prediction_spec,
        runtime=runtime,
    )
    if reference_clips is not None:
        reference_features = _motion_features(
            reference_clips,
            model=model,
            modules=modules,
            spec=reference_spec,
            runtime=runtime,
        )
        atomic_save_npy(cached_features, reference_features)
        atomic_write_text(
            cached_report, json.dumps(details["reference"], indent=2, sort_keys=True) + "\n"
        )

    prediction_mu, prediction_sigma = modules["activation_statistics"](prediction_features)
    reference_mu, reference_sigma = modules["activation_statistics"](reference_features)
    raw = float(
        modules["frechet_distance"](
            prediction_mu, prediction_sigma, reference_mu, reference_sigma
        )
    )
    if not np.isfinite(raw):
        raise ValueError(f"FVMD returned a non-finite score: {raw}")

    prediction_count = int(prediction_features.shape[0])
    reference_count = int(reference_features.shape[0])
    feature_dim = int(prediction_features.shape[1])
    details.update(
        {
            "reference_cache_dir": str(cache_dir),
            "repo_root": str(repo_root) if repo_root is not None else None,
            "feature_dim": feature_dim,
            "prediction_window_count": prediction_count,
            "reference_window_count": reference_count,
            # A Frechet distance estimates a full covariance, so a side with fewer
            # windows than dimensions has a singular covariance and the score is
            # only meaningful when compared at matching window counts.
            "singular_covariance_sides": [
                side
                for side, count in (
                    ("prediction", prediction_count),
                    ("reference", reference_count),
                )
                if count <= feature_dim
            ],
            "torch_home": os.environ.get("TORCH_HOME"),
            "raw_direction": "lower_is_better",
        }
    )
    return {
        "raw": raw,
        "normalized": normalize_score(raw, normalization),
        "backend": "fvmd",
        "details": details,
    }


__all__ = ["FVMD_FEATURE_DIM", "compute_frechet_video_motion_distance"]

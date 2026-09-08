"""JEDi (JEPA Embedding Distance) backend for distribution-level video quality."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from worldarena.benchmark.distribution_clips import (
    ClipSamplingSpec,
    atomic_write_text,
    build_clip_array,
    clip_set_signature,
    reference_runtime,
)
from worldarena.benchmark.metrics.base import normalize_score
from worldarena.common.checkpoints import checkpoint_env, filter_checkpoint_env_overrides


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JEDI_REPO_CANDIDATES = (PROJECT_ROOT / "thirdparty" / "JEDi",)
DEFAULT_ANALYSIS_SIZE = 224


def _as_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _resolve_jedi_repo_root(runtime: dict[str, Any]) -> Path | None:
    configured = runtime.get("repo_root")
    if configured:
        repo_root = Path(str(configured)).expanduser()
        if not repo_root.is_absolute():
            repo_root = PROJECT_ROOT / repo_root
        if not repo_root.exists():
            raise FileNotFoundError(f"JEDi repo_root does not exist: {repo_root}")
        if not (repo_root / "videojedi").is_dir():
            raise FileNotFoundError(
                f"JEDi repo_root does not contain a videojedi package: {repo_root}"
            )
        return repo_root.resolve()

    for candidate in DEFAULT_JEDI_REPO_CANDIDATES:
        if (candidate / "videojedi").is_dir():
            return candidate.resolve()
    return None


def _load_jedi_metric_class(runtime: dict[str, Any]) -> tuple[Any, Path | None]:
    repo_root = _resolve_jedi_repo_root(runtime)
    if repo_root is not None:
        repo_path = str(repo_root)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

    try:
        module = importlib.import_module("videojedi")
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise ModuleNotFoundError(
            "JEDi package is unavailable; install videojedi or set "
            "benchmark.metric_runtime.jedi.repo_root to the cloned repository. "
            f"Import error: {exc.__class__.__name__}: {exc}"
        ) from exc

    metric_class = getattr(module, "JEDiMetric", None)
    if metric_class is None:
        raise AttributeError("videojedi package does not expose JEDiMetric")
    return metric_class, repo_root


def _prepare_runtime_environment(runtime: dict[str, Any]) -> None:
    os.environ.update(checkpoint_env())
    os.environ.update(filter_checkpoint_env_overrides(dict(runtime.get("env", {}))))


def _resolve_runtime_path(value: Any, base_dir: Path) -> str | None:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _require_configured_path(path: str | None, *, label: str, kind: str) -> None:
    if path is None:
        return
    candidate = Path(path)
    exists = candidate.is_dir() if kind == "directory" else candidate.is_file()
    if not exists:
        raise FileNotFoundError(f"JEDi {label} {kind} does not exist: {candidate}")


def _dependency_tokens(*, model_dir: str, config_path: str | None) -> list[str]:
    """Checkpoint identity, so a weight change invalidates cached features."""
    tokens: list[str] = []
    candidates = [Path(model_dir) / "vith16.pth.tar", Path(model_dir) / "ssv2-probe.pth.tar"]
    if config_path is not None:
        candidates.append(Path(config_path))
    for path in candidates:
        try:
            stat = path.stat()
        except OSError:
            tokens.append(str(path))
            continue
        tokens.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
    return tokens


def _make_loader(clips: np.ndarray, batch_size: int) -> Any:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    tensor = torch.from_numpy(clips).permute(0, 1, 4, 2, 3).float() / 255.0
    labels = torch.zeros((tensor.shape[0],), dtype=torch.long)
    return DataLoader(TensorDataset(tensor, labels), batch_size=batch_size, shuffle=False)


def _materialize(source: Path, destination: Path) -> None:
    """Place ``source`` at ``destination`` without ever exposing a partial file.

    ``os.link`` is already atomic. The copy fallback stages into the destination
    directory first, so a concurrent run cannot load a half-written cache entry.
    """
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return
    except OSError:
        pass
    staging = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source, staging)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            staging.unlink(missing_ok=True)


def compute_jedi_distance(
    *,
    prediction_entries: Sequence[tuple[Path, str]],
    reference_entries: Sequence[tuple[Path, str]],
    normalization: dict[str, Any],
    runtime: dict[str, Any],
    output_dir: Path,
    reference_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """MMD between the prediction and reference clip distributions.

    The two sides are independent sets: JEDi's estimator has separate
    ``1/m^2``, ``1/n^2`` and ``1/(mn)`` terms and never pairs samples, so the
    reference corpus can be much larger than the prediction set.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_spec = ClipSamplingSpec.from_runtime(
        runtime, default_analysis_size=DEFAULT_ANALYSIS_SIZE
    )
    reference_spec = ClipSamplingSpec.from_runtime(
        reference_runtime(runtime), default_analysis_size=DEFAULT_ANALYSIS_SIZE
    )

    _prepare_runtime_environment(runtime)
    metric_class, repo_root = _load_jedi_metric_class(runtime)
    model_base = repo_root or PROJECT_ROOT
    model_dir = _resolve_runtime_path(runtime.get("model_dir"), PROJECT_ROOT) or str(model_base)
    config_path = _resolve_runtime_path(runtime.get("config_path"), model_base)
    if runtime.get("model_dir") is not None:
        _require_configured_path(model_dir, label="model_dir", kind="directory")
    if runtime.get("config_path") is not None:
        _require_configured_path(config_path, label="config_path", kind="file")

    dependency_tokens = _dependency_tokens(model_dir=model_dir, config_path=config_path)
    reference_signature = clip_set_signature(
        reference_entries, reference_spec, extra=dependency_tokens
    )
    prediction_signature = clip_set_signature(
        prediction_entries, prediction_spec, extra=dependency_tokens
    )

    cache_dir = reference_cache_dir or (output_dir / "reference_cache")
    cached_features = cache_dir / f"{reference_signature}.npy"
    cached_report = cache_dir / f"{reference_signature}.json"
    reference_cache_hit = cached_features.is_file()

    feature_dir = output_dir / "features" / f"{reference_signature}__{prediction_signature}"
    feature_dir.mkdir(parents=True, exist_ok=True)

    details: dict[str, Any] = {
        "reference_cache_hit": reference_cache_hit,
        "reference_signature": reference_signature,
        "prediction_signature": prediction_signature,
        "prediction_clip_spec": prediction_spec.describe(),
        "reference_clip_spec": reference_spec.describe(),
    }

    prediction_array, prediction_report = build_clip_array(
        prediction_entries, prediction_spec, label="prediction"
    )
    details["prediction"] = prediction_report.describe()

    reference_array: np.ndarray | None = None
    if reference_cache_hit:
        _materialize(cached_features, feature_dir / "train.npy")
        if cached_report.is_file():
            details["reference"] = json.loads(cached_report.read_text(encoding="utf-8"))
        else:
            details["reference"] = {"candidate_video_count": len(reference_entries)}
    else:
        reference_array, reference_report = build_clip_array(
            reference_entries, reference_spec, label="reference"
        )
        details["reference"] = reference_report.describe()

    reference_clip_count = int(
        reference_array.shape[0]
        if reference_array is not None
        else details["reference"].get("clip_count", 0)
    )
    batch_size = _as_positive_int(runtime.get("batch_size"), 4)
    # load_features applies one num_samples cap to both sides, so anything below
    # the larger side would truncate the reference corpus back down to the
    # prediction count and undo the point of a fixed corpus.
    num_samples = _as_positive_int(
        runtime.get("num_samples"),
        max(reference_clip_count, int(prediction_array.shape[0])),
    )

    metric = metric_class(
        feature_path=str(feature_dir),
        model_dir=model_dir,
        config_path=config_path,
    )
    metric.load_features(
        train_loader=(
            None if reference_array is None else _make_loader(reference_array, batch_size)
        ),
        test_loader=_make_loader(prediction_array, batch_size),
        num_samples=num_samples,
    )
    raw = float(metric.compute_metric())
    if not np.isfinite(raw):
        raise ValueError(f"JEDi returned a non-finite score: {raw}")

    produced_reference = feature_dir / "train.npy"
    if not reference_cache_hit and produced_reference.is_file():
        _materialize(produced_reference, cached_features)
        atomic_write_text(
            cached_report, json.dumps(details["reference"], indent=2, sort_keys=True) + "\n"
        )

    train_features = getattr(metric, "train_features", None)
    test_features = getattr(metric, "test_features", None)
    details.update(
        {
            "feature_dir": str(feature_dir),
            "reference_cache_dir": str(cache_dir),
            "repo_root": str(repo_root) if repo_root is not None else None,
            "model_dir": model_dir,
            "config_path": config_path,
            "batch_size": batch_size,
            "num_samples": num_samples,
            "reference_feature_count": (
                None if train_features is None else int(np.asarray(train_features).shape[0])
            ),
            "prediction_feature_count": (
                None if test_features is None else int(np.asarray(test_features).shape[0])
            ),
            "torch_home": os.environ.get("TORCH_HOME"),
            "raw_direction": "lower_is_better",
        }
    )
    return {
        "raw": raw,
        "normalized": normalize_score(raw, normalization),
        "backend": "jedi",
        "details": details,
    }


__all__ = ["compute_jedi_distance"]

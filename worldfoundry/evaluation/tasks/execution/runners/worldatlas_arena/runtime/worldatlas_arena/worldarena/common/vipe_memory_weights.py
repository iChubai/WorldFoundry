"""Offline pose/depth weights for memory-track ViPE sidecars.

Hope nodes have no DNS. The official default pipeline needs UniDepth-L,
Video-Depth-Anything-Small, and Prior-Depth-Anything on local disk. This
module resolves those files, writes the HuggingFace ``refs/main`` pin so
``from_pretrained`` can stay offline, and optionally copies them to
node-local scratch before the eight GPU shards start.
"""

from __future__ import annotations

import os
from pathlib import Path

from worldarena.benchmark.metric_errors import MetricLoadError
from worldarena.common.local_checkpoints import (
    UNIDEPTH_HF_REPO,
    UNIDEPTH_REVISION,
    _usable_file,
    exclusive_file_lock,
    local_cache_candidates,
    materialize_local_file,
    project_root,
    unidepth_weight_path,
)

VIPE_KIND = "vipe_memory"
VDA_FILENAME = "video_depth_anything_vits.pth"
PRIORDA_DAV2_FILENAME = "depth_anything_v2_vitb.pth"
PRIORDA_MODEL_FILENAME = "prior_depth_anything_vitb.pth"
VDA_MIN_BYTES = 50_000_000
PRIORDA_DAV2_MIN_BYTES = 200_000_000
PRIORDA_MODEL_MIN_BYTES = 50_000_000
UNIDEPTH_MIN_BYTES = 1_000_000_000


def checkpoint_root() -> Path:
    override = os.environ.get("WORLDARENA_CHECKPOINT_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    return project_root() / "ckpt"


def _usable_min(path: Path, minimum: int) -> bool:
    if not _usable_file(path):
        return False
    try:
        return path.resolve().stat().st_size >= minimum
    except OSError:
        return False


def _hf_home_candidates() -> list[Path]:
    seen: list[Path] = []
    for raw in (
        os.environ.get("HF_HOME", "").strip(),
        os.environ.get("WORLDARENA_SOURCE_HF_HOME", "").strip(),
        str(checkpoint_root() / "huggingface"),
        str(project_root().parent / "conda" / "hf-home"),
    ):
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path not in seen:
            seen.append(path)
    return seen


def unidepth_snapshot_dir() -> Path:
    for hf_home in _hf_home_candidates():
        weight = unidepth_weight_path(hf_home)
        if _usable_min(weight, UNIDEPTH_MIN_BYTES):
            return weight.parent
    raise MetricLoadError(
        "UniDepth-L snapshot missing offline. Expected "
        f"hub/{UNIDEPTH_HF_REPO}/snapshots/{UNIDEPTH_REVISION}/model.safetensors "
        f"under one of: {', '.join(str(path) for path in _hf_home_candidates())}"
    )


def ensure_unidepth_refs(snapshot: Path | None = None) -> Path:
    """Write ``refs/main`` so HuggingFace can resolve the pinned snapshot offline."""
    snap = snapshot or unidepth_snapshot_dir()
    hub_dir = snap.parent.parent
    if hub_dir.name != UNIDEPTH_HF_REPO:
        return snap
    refs_dir = hub_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    main = refs_dir / "main"
    payload = f"{UNIDEPTH_REVISION}\n"
    if not main.is_file() or main.read_text(encoding="utf-8") != payload:
        main.write_text(payload, encoding="utf-8")
    return snap


def vda_checkpoint_path() -> Path:
    explicit = os.environ.get("WORLDFOUNDRY_VIDEO_DEPTH_ANYTHING_CKPT", "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if _usable_min(path, VDA_MIN_BYTES):
            return path
        raise MetricLoadError(f"WORLDFOUNDRY_VIDEO_DEPTH_ANYTHING_CKPT is unusable: {path}")
    root = checkpoint_root()
    for relative in (
        Path("Video-Depth-Anything-Small") / VDA_FILENAME,
        Path("Video-Depth-Anything") / VDA_FILENAME,
        Path("video_depth_anything") / VDA_FILENAME,
    ):
        candidate = root / relative
        if _usable_min(candidate, VDA_MIN_BYTES):
            return candidate
    raise MetricLoadError(
        f"Video-Depth-Anything-Small weight missing offline: {root / 'Video-Depth-Anything-Small' / VDA_FILENAME}"
    )


def priorda_dir() -> Path:
    explicit = os.environ.get("WORLDFOUNDRY_PRIORDA_DIR", "").strip()
    roots = [Path(explicit).expanduser()] if explicit else []
    roots.append(checkpoint_root() / "Prior-Depth-Anything")
    ckpt = os.environ.get("WORLDFOUNDRY_CKPT_DIR", "").strip()
    if ckpt:
        roots.append(Path(ckpt).expanduser() / "Prior-Depth-Anything")
    for root in roots:
        dav2 = root / PRIORDA_DAV2_FILENAME
        model = root / PRIORDA_MODEL_FILENAME
        if _usable_min(dav2, PRIORDA_DAV2_MIN_BYTES) and _usable_min(model, PRIORDA_MODEL_MIN_BYTES):
            return root
    raise MetricLoadError(
        "Prior-Depth-Anything weights missing offline. Need "
        f"{PRIORDA_DAV2_FILENAME} and {PRIORDA_MODEL_FILENAME} under "
        f"{checkpoint_root() / 'Prior-Depth-Anything'}"
    )


def require_memory_vipe_weights() -> dict[str, Path]:
    snapshot = ensure_unidepth_refs()
    config = snapshot / "config.json"
    if not _usable_file(config):
        raise MetricLoadError(f"UniDepth config missing: {config}")
    return {
        "unidepth_snapshot": snapshot,
        "unidepth_weight": snapshot / "model.safetensors",
        "vda": vda_checkpoint_path(),
        "priorda_dir": priorda_dir(),
        "priorda_dav2": priorda_dir() / PRIORDA_DAV2_FILENAME,
        "priorda_model": priorda_dir() / PRIORDA_MODEL_FILENAME,
    }


def _copy_named(src: Path, dest: Path) -> Path:
    src = src.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    lock = dest.with_name(f"{dest.name}.lock")
    with exclusive_file_lock(lock):
        if dest.is_file():
            try:
                if dest.stat().st_size == src.stat().st_size:
                    return dest
            except OSError:
                pass
        temporary = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
        if temporary.exists():
            temporary.unlink()
        import shutil

        shutil.copyfile(src, temporary)
        os.replace(temporary, dest)
    return dest


def _materialize_unidepth_snapshot(snapshot: Path) -> Path:
    config = snapshot / "config.json"
    weight = snapshot / "model.safetensors"
    for cache in local_cache_candidates(VIPE_KIND):
        try:
            cache.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        dest_dir = cache / "unidepth-l"
        dest_config = dest_dir / "config.json"
        dest_weight = dest_dir / "model.safetensors"
        try:
            if (
                dest_config.is_file()
                and dest_weight.is_file()
                and dest_weight.stat().st_size == weight.resolve().stat().st_size
            ):
                return dest_dir
        except OSError:
            continue
        try:
            _copy_named(config, dest_config)
            _copy_named(weight, dest_weight)
            return dest_dir
        except OSError:
            continue
    return snapshot


def prepare_memory_vipe_checkpoints() -> dict[str, str]:
    """Resolve offline weights and copy them off dolphinfs once per node."""
    weights = require_memory_vipe_weights()
    vda = materialize_local_file(weights["vda"], kind=VIPE_KIND)
    dav2 = materialize_local_file(weights["priorda_dav2"], kind=VIPE_KIND)
    priorda = materialize_local_file(weights["priorda_model"], kind=VIPE_KIND)
    if dav2.parent != priorda.parent:
        shared = dav2.parent
        priorda = _copy_named(priorda, shared / PRIORDA_MODEL_FILENAME)
    snapshot = _materialize_unidepth_snapshot(weights["unidepth_snapshot"])
    hf_home = checkpoint_root() / "huggingface"
    return {
        "WORLDARENA_CHECKPOINT_ROOT": str(checkpoint_root()),
        "WORLDFOUNDRY_CKPT_DIR": str(checkpoint_root()),
        "WORLDFOUNDRY_UNIDEPTH_MODEL": str(snapshot),
        "WORLDFOUNDRY_VIDEO_DEPTH_ANYTHING_CKPT": str(vda),
        "WORLDFOUNDRY_PRIORDA_DIR": str(dav2.parent),
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_home / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def apply_memory_vipe_env(env: dict[str, str] | None = None) -> dict[str, str]:
    resolved = env or prepare_memory_vipe_checkpoints()
    os.environ.update(resolved)
    return resolved


def main() -> int:
    """Print ``KEY=VAL`` lines for the Hope runner to ``export``.

    ``materialize_local_file`` logs copy progress on stdout. That must not
    mix with the env dump: ``run_vipe_job.sh`` does ``export "$line"`` under
    ``set -e``, and a ``[progress]`` line is not a valid identifier.
    """
    import io
    import sys
    from contextlib import redirect_stdout

    captured = io.StringIO()
    with redirect_stdout(captured):
        env = prepare_memory_vipe_checkpoints()
    progress = captured.getvalue()
    if progress:
        sys.stderr.write(progress)
        sys.stderr.flush()
    for key, value in env.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

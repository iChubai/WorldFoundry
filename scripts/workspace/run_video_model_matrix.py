#!/usr/bin/env python3
"""Run the Workspace video/world catalog on a fixed set of GPU workers.

The Workspace owns model loading and artifact persistence.  This client keeps
exactly one job assigned to each GPU, records catalog/checkpoint blockers, and
validates every completed preview video by decoding the whole stream.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
SUCCESS_MANIFEST_STATUSES = {"completed", "succeeded", "success"}
VIDEO_CATEGORIES = {"Video Generation", "Video-to-Video"}
VIDEO_WORKLOADS = {"t2v", "i2v", "v2v", "world"}
WORKSPACE_REQUEST_TIMEOUT_SECONDS = 900
DEFAULT_SUBMISSION_MAX_ATTEMPTS = 5
MODEL_REFERENCE_TOKENS = {
    "adapter",
    "checkpoint",
    "ckpt",
    "codec",
    "da3",
    "dit",
    "encoder",
    "flux",
    "gemma",
    "lora",
    "model",
    "pretrained",
    "tokenizer",
    "transformer",
    "upsampler",
    "vae",
    "vggt",
    "wan",
    "weight",
}
NON_INPUT_PATH_KEYS = {
    "artifact_path",
    "cache_dir",
    "output_dir",
    "output_path",
    "save_dir",
    "save_path",
}
EXPLICIT_MODEL_PATH_KEYS = {
    "conda_dir",
    "python_env_dir",
    "python_executable",
    "repo_root",
    "runtime_root",
    "scene_path",
    "source_root",
}
# These full-quality recipes require a coordinated multi-GPU allocation.  This
# runner intentionally owns independent single-GPU workers, so submitting one
# of them as ``cuda:N`` would either fail topology validation or collide with
# other matrix jobs.  They are run in a separate four-GPU gang phase and then
# folded back into the same report through Workspace job discovery.
MULTI_GPU_GANG_SIZES = {
    "lingbot-world-v2": 4,
    "wan2.1-vace": 4,
}
CHECKPOINT_PAYLOAD_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".distcp",
    ".pt",
    ".pth",
    ".safetensors",
}
CHECKPOINT_INDEX_SUFFIXES = (
    ".bin.index.json",
    ".safetensors.index.json",
)
CHECKPOINT_INDEX_GLOBS = (
    "*.index.json",
    "*/*.index.json",
    "*/*/*.index.json",
    "*/*/*/*.index.json",
)
PRIMARY_CHECKPOINT_GLOBS = (
    "model.pt",
    "model.pth",
    "model.ckpt",
    "model.safetensors",
    "diffusion_pytorch_model.safetensors",
    "pytorch_model.bin",
    "Wan*_VAE.pth",
    "models_t5_*.pth",
    "*/model.pt",
    "*/model.pth",
    "*/model.ckpt",
    "*/model.safetensors",
    "*/diffusion_pytorch_model.safetensors",
    "*/pytorch_model.bin",
    "*/*/mp_rank_*_model_states*.pt",
    "*/Wan*_VAE.pth",
    "*/models_t5_*.pth",
)
SHARDED_CHECKPOINT_GLOBS = (
    "*-*-of-*.*",
    "*/*-*-of-*.*",
    "*/*/*-*-of-*.*",
    "*/*/*/*-*-of-*.*",
)
SHARDED_CHECKPOINT_NAME = re.compile(
    r"^.+-\d{5}-of-\d{5}\.(?:bin|safetensors)$"
)
STRICT_VALIDATION_GLOBS = (
    "strict_video_validation*.jsonl",
    "strict-video-validation*.jsonl",
)


def _multi_gpu_gang_blocker(model_id: str) -> str:
    size = MULTI_GPU_GANG_SIZES.get(str(model_id or ""))
    if size is None:
        return ""
    return (
        f"deferred to the exclusive {size}-GPU gang phase; the single-GPU matrix "
        "must not submit this topology as cuda:N"
    )


def _request_json(base_url: str, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        # A cold /api/models request resolves every catalog/checkpoint contract.
        # On large shared checkpoint roots this can legitimately take several
        # minutes, while inference job polling remains fast after warm-up.
        with urllib.request.urlopen(request, timeout=WORKSPACE_REQUEST_TIMEOUT_SECONDS) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Workspace {method} {path} returned HTTP {exc.code}: {detail}") from exc


def _default_variant(model: dict[str, Any]) -> dict[str, Any]:
    variants = list(model.get("variants") or ())
    default_id = str(model.get("default_variant_id") or "")
    return next((item for item in variants if item.get("variant_id") == default_id), variants[0] if variants else {})


def _input_blocker(model: dict[str, Any], readiness: dict[str, Any]) -> str:
    """Return a model-specific input blocker after checkpoint readiness passes."""

    model_id = str(model.get("id") or model.get("model_id") or "")
    if model_id == "hyworld-worldgen" and not readiness.get("ready"):
        return (
            "required materialized scene input is unavailable or incomplete: "
            "render_results/global_pcd.ply"
        )
    if model_id != "dreamdojo" or not readiness.get("ready"):
        return ""

    variant = _default_variant(model)
    dataset_value = str((variant.get("load_kwargs") or {}).get("dataset_path") or "")
    if not dataset_value:
        return "DreamDojo default variant does not declare its required GR-1 dataset_path"
    dataset_path = Path(os.path.expandvars(dataset_value)).expanduser()
    stats_path = dataset_path / "meta" / "stats.json"
    if stats_path.is_file():
        return ""
    return f"DreamDojo GR-1 input dataset is incomplete: missing {stats_path}"


def _candidate_local_paths(raw_ref: str, checkpoint_root: Path) -> list[Path]:
    raw_ref = str(raw_ref or "").strip()
    if not raw_ref:
        return []
    path = Path(raw_ref).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
        for marker in ("/ckpt/", "/ckpts/"):
            if marker in raw_ref:
                relative = Path(raw_ref.split(marker, maxsplit=1)[1])
                candidates.append(checkpoint_root / relative)
                # Local HF mirrors are deployed both under ckpts/hfd/<repo>
                # and directly under ckpts/<repo>.  Accept either layout
                # without creating aliases inside the user's checkpoint tree.
                # Preserve the complete repository-relative suffix when the
                # legacy reference includes the ``hfd`` namespace.  Using
                # only ``relative.name`` loses component paths such as
                # ``dit/model.pth`` and can silently resolve an unrelated
                # basename at the checkpoint root.
                if relative.parts and relative.parts[0] == "hfd" and len(relative.parts) > 1:
                    candidates.append(checkpoint_root.joinpath(*relative.parts[1:]))
                candidates.append(checkpoint_root / relative.name)
    elif "://" not in raw_ref:
        candidates.extend(
            (
                Path.cwd() / path,
                checkpoint_root / raw_ref.replace("/", "--"),
                checkpoint_root / raw_ref.split("/")[-1],
                checkpoint_root / "hfd" / raw_ref.replace("/", "--"),
                checkpoint_root / "hfd" / raw_ref.split("/")[-1],
            )
        )
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _declared_load_ref_items(value: Any, *, key: str = "") -> list[tuple[str, str]]:
    """Return ``(option name, reference)`` pairs embedded in load kwargs."""

    if isinstance(value, dict):
        refs: list[tuple[str, str]] = []
        for child_key, child_value in value.items():
            refs.extend(_declared_load_ref_items(child_value, key=str(child_key)))
        return list(dict.fromkeys(refs))
    if isinstance(value, (list, tuple)):
        refs: list[tuple[str, str]] = []
        for child_value in value:
            refs.extend(_declared_load_ref_items(child_value, key=key))
        return list(dict.fromkeys(refs))
    if not isinstance(value, str) or not value.strip():
        return []

    normalized_key = key.strip().lower().replace("-", "_")
    if (
        normalized_key in NON_INPUT_PATH_KEYS
        or normalized_key.endswith("_name")
        or normalized_key.endswith("_id")
    ):
        return []
    key_tokens = set(normalized_key.split("_"))
    is_model_reference = normalized_key in EXPLICIT_MODEL_PATH_KEYS or bool(
        key_tokens & MODEL_REFERENCE_TOKENS
    )
    return [(normalized_key, value.strip())] if is_model_reference else []


def _declared_load_refs(value: Any, *, key: str = "") -> list[str]:
    """Return model/checkpoint references embedded in variant load kwargs."""

    return list(dict.fromkeys(ref for _, ref in _declared_load_ref_items(value, key=key)))


def _resolve_ref(raw_ref: str, checkpoint_root: Path) -> str:
    for candidate in _candidate_local_paths(raw_ref, checkpoint_root):
        if candidate.exists():
            return str(candidate.resolve())
    return ""


def _component_structure_error(option_name: str, resolved_ref: str) -> str:
    """Validate directory layouts whose existence alone is insufficient."""

    if option_name == "flux_path":
        root = Path(resolved_ref)
        required = (
            "model_index.json",
            "scheduler/scheduler_config.json",
            "text_encoder/config.json",
            "text_encoder_2/config.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer_2/tokenizer_config.json",
            "transformer/config.json",
            "vae/config.json",
        )
        missing = [relative for relative in required if not (root / relative).is_file()]
        return (
            ""
            if not missing
            else f"{option_name} is incomplete at {root}: missing {', '.join(missing)}"
        )
    if option_name == "scene_path":
        root = Path(resolved_ref)
        point_cloud = root / "render_results" / "global_pcd.ply"
        return "" if point_cloud.is_file() else f"{option_name} is incomplete at {root}: missing render_results/global_pcd.ply"
    if option_name in {"gemma_path", "text_encoder_dir"}:
        root = Path(resolved_ref)
        missing: list[str] = []
        if not (root / "config.json").is_file():
            missing.append("config.json")
        if not tuple(root.glob("*.safetensors")):
            missing.append("*.safetensors")
        return "" if not missing else f"{option_name} is incomplete at {root}: missing {', '.join(missing)}"
    if option_name != "gemma_root":
        return ""
    root = Path(resolved_ref)
    shards = tuple(root.glob("model-*.safetensors")) if root.is_dir() else ()
    tokenizer = root / "tokenizer.model"
    sibling_tokenizer = root.parent / "tokenizer" / "tokenizer.model"
    missing: list[str] = []
    if not shards:
        missing.append("model-*.safetensors")
    if not tokenizer.is_file() and not sibling_tokenizer.is_file():
        missing.append("tokenizer.model")
    if not missing:
        return ""
    return f"{option_name} is incomplete at {root}: missing {', '.join(missing)}"


def _checkpoint_payload_error(path: Path) -> str:
    """Return an error for an interrupted or sparse checkpoint payload.

    HFD/aria2 creates the final-size destination before the download is
    complete.  A plain ``Path.exists()`` therefore accepts files that contain
    only holes and fail much later inside safetensors or torch.load.  Keep the
    check metadata/header-only so auditing never reads multi-gigabyte weights.
    """

    control_path = Path(f"{path}.aria2")
    if control_path.is_file():
        return f"checkpoint download is incomplete: {control_path} exists"
    if not path.is_file():
        return f"checkpoint payload is missing: {path}"
    try:
        stat_result = path.stat()
    except OSError as exc:
        return f"checkpoint payload cannot be inspected at {path}: {exc}"
    # Preserve the existing lightweight path-contract tests, which use empty
    # marker files.  Real interrupted HFD payloads advertise their final size.
    if stat_result.st_size < 1024 * 1024:
        return ""
    # Do not use ``st_blocks`` as a materialization signal.  Distributed
    # filesystems such as BeeGFS can report zero allocated POSIX blocks for a
    # fully materialized, readable file.  Active aria2 downloads are already
    # identified by their control sidecar above, while a genuinely untouched
    # preallocated destination is caught by the all-zero header check below.
    try:
        with path.open("rb") as handle:
            header = handle.read(4096)
    except OSError as exc:
        return f"checkpoint payload cannot be read at {path}: {exc}"
    if header and not any(header):
        return f"checkpoint payload starts with an unmaterialized sparse hole: {path}"
    return ""


def _checkpoint_directory_error(root: Path, checkpoint_root: Path) -> str:
    """Validate interrupted downloads and shards named by weight indexes."""

    try:
        if root.resolve() == checkpoint_root.resolve():
            # Some runtimes deliberately receive the shared root as a cache
            # search path. Recursively auditing every unrelated model would
            # both be expensive and incorrectly couple their readiness.
            return ""
    except OSError:
        return ""

    seen: set[Path] = set()
    for pattern in CHECKPOINT_INDEX_GLOBS:
        for index_path in root.glob(pattern):
            if index_path in seen or not index_path.name.endswith(CHECKPOINT_INDEX_SUFFIXES):
                continue
            seen.add(index_path)
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                return f"checkpoint index is unreadable at {index_path}: {exc}"
            weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
            if not isinstance(weight_map, dict):
                return f"checkpoint index has no weight_map: {index_path}"
            for shard_name in dict.fromkeys(str(value) for value in weight_map.values()):
                error = _checkpoint_payload_error(index_path.parent / shard_name)
                if error:
                    return error

    # Index-free checkpoints commonly use one of these canonical primary
    # names. This catches model.pt and distributed mp_rank placeholders while
    # leaving unindexed optional duplicate exports alone.
    for pattern in PRIMARY_CHECKPOINT_GLOBS:
        for payload_path in root.glob(pattern):
            if payload_path.suffix.lower() not in CHECKPOINT_PAYLOAD_SUFFIXES:
                continue
            error = _checkpoint_payload_error(payload_path)
            if error:
                return error

    # Some official repositories omit an index JSON but still use the
    # transformers/diffusers ``00001-of-000NN`` convention. Every member of
    # such a group is required; a sparse member makes that component unusable.
    for pattern in SHARDED_CHECKPOINT_GLOBS:
        for payload_path in root.glob(pattern):
            if not SHARDED_CHECKPOINT_NAME.fullmatch(payload_path.name):
                continue
            error = _checkpoint_payload_error(payload_path)
            if error:
                return error

    # aria2 control files are an explicit, format-independent signal that a
    # requested payload has not completed. Limit traversal depth to the same
    # bounded component layout used for checkpoint indexes.
    for pattern in ("*.aria2", "*/*.aria2", "*/*/*.aria2", "*/*/*/*.aria2"):
        for control_path in root.glob(pattern):
            target = Path(str(control_path)[: -len(".aria2")])
            if target.suffix.lower() in CHECKPOINT_PAYLOAD_SUFFIXES:
                return f"checkpoint download is incomplete: {control_path} exists"
    return ""


def _checkpoint_artifact_structure_error(resolved_ref: str, checkpoint_root: Path) -> str:
    path = Path(resolved_ref)
    if path.is_file():
        if path.suffix.lower() not in CHECKPOINT_PAYLOAD_SUFFIXES:
            return ""
        return _checkpoint_payload_error(path)
    if path.is_dir():
        return _checkpoint_directory_error(path, checkpoint_root)
    return ""


def _is_unused_flux_single_file_export_error(
    option_names: set[str],
    resolved_ref: str,
    payload_error: str,
    component_errors: list[str],
) -> bool:
    """Ignore an interrupted optional FLUX single-file export.

    MoVerse loads ``FLUX.1-Fill-dev`` through Diffusers
    ``from_pretrained(directory)``.  That path consumes ``model_index.json``
    plus the indexed component shards and never opens the repository's
    redundant root-level ``flux1-fill-dev.safetensors`` export.  HFD may leave
    a sparse destination and ``.aria2`` sidecar for that optional export even
    when every directory-format component is complete.  Keep rejecting the
    marker for all other components and whenever the Diffusers layout itself
    is incomplete.
    """

    if "flux_path" not in option_names or component_errors:
        return False
    marker = Path(resolved_ref) / "flux1-fill-dev.safetensors.aria2"
    return payload_error == f"checkpoint download is incomplete: {marker} exists"


def _model_checkpoint_structure_error(model_id: str, resolved_model_ref: str) -> str:
    """Run cheap, model-aware header checks that generic path audits cannot express."""

    if not model_id.startswith("echo-memory-") or not Path(resolved_model_ref).is_file():
        return ""
    try:
        from worldfoundry.base_models.diffusion_model.models.denoisers.echo_memory_checkpoint import (
            inspect_echo_checkpoint,
        )
        from worldfoundry.base_models.diffusion_model.models.denoisers.echo_memory_spec import (
            get_echo_memory_model_spec,
        )

        recipe = get_echo_memory_model_spec(model_id).recipe
        inspect_echo_checkpoint(resolved_model_ref, recipe)
    except KeyError:
        return ""
    except Exception as exc:  # noqa: BLE001 - one corrupt checkpoint must not abort the full audit.
        return f"{model_id} checkpoint architecture is incompatible: {exc}"
    return ""


def _checkpoint_readiness(
    model: dict[str, Any],
    checkpoint_root: Path,
    *,
    resolution_cache: dict[str, str] | None = None,
    structure_cache: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve one catalog entry, optionally reusing per-audit filesystem work.

    Large model families frequently share the same Wan, text encoder, or VAE
    directory.  Bounded recursive index/header checks are deterministic during
    one startup audit, so scanning the same multi-hundred-GB tree once per
    catalog row only adds BeeGFS metadata traffic without improving safety.
    """

    def resolve(ref: str) -> str:
        if resolution_cache is None:
            return _resolve_ref(ref, checkpoint_root)
        if ref not in resolution_cache:
            resolution_cache[ref] = _resolve_ref(ref, checkpoint_root)
        return resolution_cache[ref]

    def artifact_error(resolved_ref: str) -> str:
        if structure_cache is None:
            return _checkpoint_artifact_structure_error(resolved_ref, checkpoint_root)
        if resolved_ref not in structure_cache:
            structure_cache[resolved_ref] = _checkpoint_artifact_structure_error(
                resolved_ref, checkpoint_root
            )
        return structure_cache[resolved_ref]

    variant = _default_variant(model)
    raw_model_ref = str(model.get("model_ref") or variant.get("model_ref") or "")
    checkpoints = [item for item in variant.get("checkpoints") or () if item.get("required", True)]
    model_id = str(model.get("id") or model.get("model_id") or "")
    refs = [
        raw_model_ref if raw_model_ref and str(item.get("uri") or "") == model_id else str(item.get("uri") or "")
        for item in checkpoints
    ]
    if not refs and raw_model_ref:
        refs = [raw_model_ref]
    load_ref_items = _declared_load_ref_items(variant.get("load_kwargs") or {})
    refs.extend(ref for _, ref in load_ref_items)
    refs = list(dict.fromkeys(ref for ref in refs if ref))
    option_names_by_ref: dict[str, set[str]] = {}
    for option_name, ref in load_ref_items:
        option_names_by_ref.setdefault(ref, set()).add(option_name)

    resolved: list[str] = []
    missing: list[str] = []
    component_errors: list[str] = []
    for ref in refs:
        local = resolve(ref)
        option_names = option_names_by_ref.get(ref, set())
        structure_errors = [
            error
            for option_name in option_names
            if (error := _component_structure_error(option_name, local))
        ] if local else []
        if local:
            payload_error = artifact_error(local)
            if payload_error and not _is_unused_flux_single_file_export_error(
                option_names,
                local,
                payload_error,
                structure_errors,
            ):
                structure_errors.append(payload_error)
            if ref == raw_model_ref:
                model_error = _model_checkpoint_structure_error(model_id, local)
                if model_error:
                    structure_errors.append(model_error)
        if local and not structure_errors:
            resolved.append(local)
        else:
            missing.append(ref or "<empty>")
            component_errors.extend(structure_errors)

    resolved_model_ref = resolve(raw_model_ref) if raw_model_ref else ""
    if not refs:
        reason = "default variant declares no checkpoint or model_ref"
    elif component_errors:
        reason = "required checkpoint component is structurally incomplete"
    elif missing:
        reason = "required checkpoint path is not locally resolvable"
    else:
        reason = ""
    return {
        "ready": bool(refs) and not missing,
        "reason": reason,
        "raw_model_ref": raw_model_ref,
        "resolved_model_ref": resolved_model_ref,
        "required_refs": refs,
        "resolved_refs": resolved,
        "missing_refs": missing,
        "component_errors": component_errors,
    }


def _can_reuse_previous_row(
    previous_row: dict[str, Any],
    readiness: dict[str, Any],
    *,
    retry_this_model: bool,
) -> bool:
    """Return whether a terminal row is still authoritative for this audit.

    A checkpoint audit is intentionally recomputed on every invocation.  In
    particular, an older ``blocked_checkpoint`` row must not mask a checkpoint
    that has since been downloaded or whose catalog declaration was fixed.
    """

    status = str(previous_row.get("status") or "")
    if retry_this_model or status in {"pending", "running", "queued", "submission_retry"}:
        return False
    if status == "blocked_checkpoint" and readiness.get("ready"):
        return False
    return True


def _refresh_checkpoint_blocker(row: dict[str, Any], readiness: dict[str, Any]) -> None:
    """Keep reused checkpoint rows aligned with the current catalog audit."""

    if row.get("status") == "blocked_checkpoint" and not readiness.get("ready"):
        row["blocker"] = readiness.get("reason") or "required checkpoint is unavailable"


def _record_submission_failure(
    *,
    queue: list[dict[str, Any]],
    model: dict[str, Any],
    row: dict[str, Any],
    gpu: int,
    attempt: int,
    max_attempts: int,
    exc: BaseException,
) -> bool:
    """Persist a failed POST and requeue it while retry budget remains.

    Returning ``True`` tells the dispatcher that this was a recoverable queue
    transition.  The caller deliberately leaves the GPU unassigned and waits
    for a fresh stable-idle gate before attempting another Workspace POST.
    """

    bounded_max_attempts = max(1, int(max_attempts))
    error = f"{type(exc).__name__}: {exc}"
    row.update(
        gpu=gpu,
        error=error,
        submission_attempts=attempt,
        submission_max_attempts=bounded_max_attempts,
    )
    if attempt < bounded_max_attempts:
        row["status"] = "submission_retry"
        queue.append(model)
        return True
    row["status"] = "submission_failed"
    return False


def _compact_job(job: dict[str, Any]) -> dict[str, Any]:
    logs = list(job.get("logs") or ())
    return {
        "id": job.get("id"),
        "model_id": job.get("model_id"),
        "status": job.get("status"),
        "title": job.get("title"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "error": job.get("error"),
        "metadata": job.get("metadata"),
        "result": job.get("result"),
        "log_tail": logs[-80:],
    }


def _video_paths(job: dict[str, Any]) -> list[Path]:
    result = job.get("result") or {}
    values: list[str] = []
    preview = str(result.get("preview_video") or "")
    if preview:
        values.append(preview)
    for artifact in result.get("artifacts") or ():
        value = str(artifact or "")
        if Path(value).suffix.lower() in {".mp4", ".webm", ".mov", ".mkv", ".avi"}:
            values.append(value)
    unique: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path not in unique:
            unique.append(path)
    return unique


def _find_executable(name: str) -> str | None:
    executable = shutil.which(name)
    if executable:
        return executable
    sibling = Path(sys.executable).resolve().parent / name
    return str(sibling) if sibling.is_file() else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_video(path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "full_decode_ok": False}
    if not path.is_file():
        row["error"] = "video path does not exist"
        return row
    try:
        row["size_bytes"] = path.stat().st_size
        row["sha256"] = _sha256_file(path)
    except OSError as exc:
        row["error"] = f"video hashing failed: {exc}"
        return row
    ffprobe = _find_executable("ffprobe")
    ffmpeg = _find_executable("ffmpeg")
    if not ffprobe or not ffmpeg:
        row["error"] = "ffprobe/ffmpeg is unavailable"
        return row
    probe = subprocess.run(
        [
            ffprobe,
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
        timeout=180,
        check=False,
    )
    row["probe_returncode"] = probe.returncode
    if probe.returncode == 0:
        try:
            row["probe"] = json.loads(probe.stdout)
        except json.JSONDecodeError:
            row["probe_stdout"] = probe.stdout
    else:
        row["probe_error"] = probe.stderr[-4000:]
    decode = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    row["decode_returncode"] = decode.returncode
    row["full_decode_ok"] = decode.returncode == 0
    if decode.returncode:
        row["decode_error"] = decode.stderr[-4000:]
    return row


def _validate_completed_job(job: dict[str, Any]) -> dict[str, Any]:
    result = job.get("result") or {}
    manifest_path = Path(str(result.get("manifest_path") or "")).expanduser()
    videos = [_validate_video(path) for path in _video_paths(job)]
    manifest: dict[str, Any] | None = None
    manifest_error = ""
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            manifest_error = str(exc)
    else:
        manifest_error = "manifest path does not exist"
    return {
        "manifest_path": str(manifest_path),
        "manifest_exists": manifest_path.is_file(),
        "manifest_status": (manifest or {}).get("status"),
        "manifest_error": manifest_error,
        "videos": videos,
        "valid": bool(videos)
        and all(item.get("full_decode_ok") for item in videos)
        and manifest_path.is_file()
        and str((manifest or {}).get("status") or "").lower() in SUCCESS_MANIFEST_STATUSES,
    }


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _strict_validation_candidate(
    records: list[dict[str, Any]],
    *,
    source: Path,
    expected_model_id: str = "",
    allowed_model_ids: set[str] | None = None,
    hash_cache: dict[tuple[str, int, int], str] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Convert persisted full-decode records into a matrix completion row.

    The strict validator writes one JSONL record per video but intentionally
    does not duplicate the model id. Resolve identity and completion status
    from the referenced Workspace manifest so an output-directory naming
    convention can never credit the wrong model.
    """

    if not records:
        return None
    manifest_values = {str(record.get("manifest") or "") for record in records}
    if len(manifest_values) != 1 or not next(iter(manifest_values), ""):
        return None
    manifest_path = Path(next(iter(manifest_values))).expanduser()
    manifest = _read_json_object(manifest_path)
    if manifest is None:
        return None
    model_id = str(manifest.get("model_id") or "")
    manifest_status = str(manifest.get("status") or "").lower()
    if not model_id or (expected_model_id and model_id != expected_model_id):
        return None
    if allowed_model_ids is not None and model_id not in allowed_model_ids:
        return None
    if manifest_status not in SUCCESS_MANIFEST_STATUSES:
        return None

    videos: list[dict[str, Any]] = []
    for record in records:
        video_path = Path(str(record.get("path") or "")).expanduser()
        sha256 = str(record.get("sha256") or "").lower()
        try:
            stat = video_path.stat()
            cache_key = (str(video_path.resolve()), stat.st_size, stat.st_mtime_ns)
            current_sha256 = (hash_cache or {}).get(cache_key, "")
            if not current_sha256:
                current_sha256 = _sha256_file(video_path)
                if hash_cache is not None:
                    hash_cache[cache_key] = current_sha256
        except OSError:
            current_sha256 = ""
        valid = (
            record.get("ok") is True
            and video_path.is_file()
            and bool(sha256)
            and current_sha256 == sha256
            and int(record.get("decoded_frames") or 0) > 0
            and int(record.get("width") or 0) > 0
            and int(record.get("height") or 0) > 0
            and record.get("temporal_change") is True
        )
        if not valid:
            return None
        video = dict(record)
        video.update(
            path=str(video_path),
            exists=True,
            full_decode_ok=True,
        )
        videos.append(video)

    validation = {
        "manifest_path": str(manifest_path),
        "manifest_exists": True,
        "manifest_status": manifest.get("status"),
        "manifest_error": "",
        "videos": videos,
        "valid": True,
        "evidence_source": str(source),
        "evidence_kind": "strict_video_validation_jsonl",
    }
    return model_id, {
        "status": "completed",
        "validation": validation,
        "persistent_evidence": {
            "kind": "strict_video_validation_jsonl",
            "source": str(source),
            "manifest_path": str(manifest_path),
        },
    }


def _strictly_validate_completed_job(
    job: dict[str, Any],
    *,
    output_dir: Path,
    model_id: str,
) -> dict[str, Any]:
    """Run the same non-blank/temporal validator used by the GPU gang runner."""

    result = job.get("result") or {}
    manifest_path = Path(str(result.get("manifest_path") or "")).expanduser()
    safe_model_id = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("._") or "model"
    validation_path = output_dir / f"strict_video_validation_{safe_model_id}.jsonl"
    command = [
        os.fspath(Path(sys.executable)),
        os.fspath(Path(__file__).with_name("validate_generated_videos.py")),
        "--output",
        os.fspath(validation_path),
        os.fspath(manifest_path),
    ]
    records: list[dict[str, Any]] = []
    validator_error = ""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        if validation_path.is_file():
            records = [
                json.loads(line)
                for line in validation_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        if completed.returncode:
            validator_error = (
                f"strict video validator exited with status {completed.returncode}: "
                f"{completed.stderr[-2000:]}"
            ).strip()
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        validator_error = f"{type(exc).__name__}: {exc}"

    candidate = _strict_validation_candidate(
        records,
        source=validation_path,
        expected_model_id=model_id,
    )
    if not validator_error and candidate is not None:
        return candidate[1]["validation"]

    # Preserve ordinary full-decode diagnostics as well as the strict records
    # so a failed model can be repaired without rerunning merely to learn
    # whether the video container itself was corrupt.
    validation = _validate_completed_job(job)
    validation.update(
        valid=False,
        strict_validation_path=str(validation_path),
        strict_validation_records=records,
        strict_validation_error=validator_error or "strict video validation did not pass",
    )
    return validation


def _load_strict_validation_file(
    path: Path,
    *,
    allowed_model_ids: set[str] | None = None,
    hash_cache: dict[tuple[str, int, int], str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        manifest = str(record.get("manifest") or "")
        if manifest:
            grouped.setdefault(manifest, []).append(record)
    candidates: list[tuple[str, dict[str, Any]]] = []
    for records in grouped.values():
        candidate = _strict_validation_candidate(
            records,
            source=path,
            allowed_model_ids=allowed_model_ids,
            hash_cache=hash_cache,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _load_gang_report(
    path: Path,
    *,
    allowed_model_ids: set[str] | None = None,
    hash_cache: dict[tuple[str, int, int], str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    payload = _read_json_object(path)
    if payload is None:
        return []
    gpus = [int(gpu) for gpu in payload.get("gpus") or ()]
    candidates: list[tuple[str, dict[str, Any]]] = []
    for gang_row in payload.get("models") or ():
        if not isinstance(gang_row, dict) or gang_row.get("status") != "completed":
            continue
        model_id = str(gang_row.get("model_id") or "")
        if allowed_model_ids is not None and model_id not in allowed_model_ids:
            continue
        strict_records = [
            record
            for record in gang_row.get("strict_validation") or ()
            if isinstance(record, dict)
        ]
        candidate = _strict_validation_candidate(
            strict_records,
            source=path,
            expected_model_id=model_id,
            hash_cache=hash_cache,
        )
        if candidate is None:
            continue
        _, completion = candidate
        completion["persistent_evidence"] = {
            "kind": "multi_gpu_gang_report",
            "source": str(path),
            "manifest_path": completion["validation"]["manifest_path"],
        }
        completion.update(
            job_id=gang_row.get("job_id"),
            gpus=gpus,
            gpu=",".join(str(gpu) for gpu in gpus),
            elapsed_seconds=gang_row.get("elapsed_seconds_observed"),
            peak_memory_mib=gang_row.get("peak_memory_mib"),
            peak_memory_mib_by_gpu=gang_row.get("peak_memory_mib_by_gpu"),
        )
        job = gang_row.get("job")
        if isinstance(job, dict):
            completion["job"] = job
        candidates.append((model_id, completion))
    return candidates


def _persistent_evidence_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    """Prefer richer gang evidence, then the lexically newest manifest run."""

    evidence = row.get("persistent_evidence") or {}
    kind = str(evidence.get("kind") or "")
    manifest = str(evidence.get("manifest_path") or "")
    return (
        1 if kind == "multi_gpu_gang_report" else 0,
        1 if row.get("elapsed_seconds") is not None else 0,
        manifest,
    )


def _discover_persistent_evidence(
    roots: list[Path],
    *,
    model_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load durable validation/gang reports below bounded result roots."""

    candidates: dict[str, list[dict[str, Any]]] = {}
    seen: set[Path] = set()
    hash_cache: dict[tuple[str, int, int], str] = {}
    for raw_root in roots:
        root = raw_root.expanduser().resolve()
        if not root.is_dir():
            continue
        strict_paths: list[Path] = []
        gang_paths: list[Path] = []
        # ``Path.rglob`` walks the complete tree once per pattern.  Result
        # roots can contain hundreds of archived videos and benchmark runs,
        # and repeated BeeGFS metadata scans used to make a targeted retry
        # spend minutes here before it could submit a job.  Classify all
        # supported evidence names during one traversal instead.
        for directory, directory_names, file_names in os.walk(root):
            directory_names.sort()
            for file_name in sorted(file_names):
                path = Path(directory) / file_name
                if file_name == "gang.json":
                    gang_paths.append(path)
                elif any(
                    fnmatch.fnmatchcase(file_name, pattern)
                    for pattern in STRICT_VALIDATION_GLOBS
                ):
                    strict_paths.append(path)
        for path in sorted(dict.fromkeys(strict_paths)):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            for model_id, row in _load_strict_validation_file(
                path,
                allowed_model_ids=model_ids,
                hash_cache=hash_cache,
            ):
                candidates.setdefault(model_id, []).append(row)
        for path in sorted(gang_paths):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            for model_id, row in _load_gang_report(
                path,
                allowed_model_ids=model_ids,
                hash_cache=hash_cache,
            ):
                candidates.setdefault(model_id, []).append(row)
    return {
        model_id: max(rows, key=_persistent_evidence_sort_key)
        for model_id, rows in candidates.items()
        if rows
    }


def _gpu_memory_mib() -> dict[int, int]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
        return {int(index.strip()): int(memory.strip()) for index, memory in (line.split(",", 1) for line in completed.stdout.splitlines())}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def _gpu_activity() -> dict[int, dict[str, Any]]:
    """Return memory, utilization, and compute PIDs for every visible GPU."""

    gpu_command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    app_command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
    ]
    try:
        gpu_result = subprocess.run(gpu_command, capture_output=True, text=True, timeout=30, check=True)
        app_result = subprocess.run(app_command, capture_output=True, text=True, timeout=30, check=True)
        activity: dict[int, dict[str, Any]] = {}
        uuid_to_index: dict[str, int] = {}
        for line in gpu_result.stdout.splitlines():
            index_text, uuid, memory_text, utilization_text = (part.strip() for part in line.split(",", 3))
            index = int(index_text)
            activity[index] = {
                "memory_mib": int(memory_text),
                "utilization_percent": int(utilization_text),
                "compute_pids": [],
            }
            uuid_to_index[uuid] = index
        for line in app_result.stdout.splitlines():
            if not line.strip():
                continue
            uuid, pid_text = (part.strip() for part in line.split(",", 1))
            index = uuid_to_index.get(uuid)
            if index is not None:
                activity[index]["compute_pids"].append(int(pid_text))
        return activity
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def _gpu_is_idle(
    activity: dict[int, dict[str, Any]],
    gpu: int,
    *,
    memory_limit_mib: int,
    utilization_limit_percent: int,
    ignored_compute_pids: set[int] | None = None,
) -> bool:
    state = activity.get(gpu)
    ignored_compute_pids = ignored_compute_pids or set()
    blocking_compute_pids = {
        int(pid)
        for pid in (state or {}).get("compute_pids") or ()
        if int(pid) not in ignored_compute_pids
    }
    return bool(state) and (
        int(state.get("memory_mib") or 0) <= memory_limit_mib
        and int(state.get("utilization_percent") or 0) <= utilization_limit_percent
        and not blocking_compute_pids
    )


def _job_elapsed_seconds(job: dict[str, Any]) -> float | None:
    """Return the persisted wall time between a job's start and finish."""

    try:
        started = datetime.fromisoformat(str(job.get("started_at") or "").replace("Z", "+00:00"))
        finished = datetime.fromisoformat(str(job.get("finished_at") or "").replace("Z", "+00:00"))
    except ValueError:
        return None
    return round(max(0.0, (finished - started).total_seconds()), 3)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    rows = list(report["models"])
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    lines = [
        "# Workspace full video/world model matrix",
        "",
        f"Generated: `{report['updated_at']}`",
        "",
        f"Catalog targets: **{len(rows)}**; " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())),
        "",
        "| Model | Workload | Status | GPU | Peak MiB | Video decode | Detail |",
        "|---|---:|---|---:|---:|---|---|",
    ]
    for row in rows:
        validation = row.get("validation") or {}
        videos = validation.get("videos") or []
        decoded = "yes" if videos and all(item.get("full_decode_ok") for item in videos) else "no"
        detail = str(row.get("error") or row.get("blocker") or "").replace("\n", " ").replace("|", "\\|")[:300]
        lines.append(
            f"| `{row['model_id']}` | {row.get('workload_type', '')} | {row.get('status', '')} | "
            f"{row.get('gpu', '')} | {row.get('peak_memory_mib', '')} | {decoded} | {detail} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _persist(output_dir: Path, report: dict[str, Any]) -> None:
    report["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(output_dir / "matrix.json", report)
    _write_markdown(output_dir / "matrix.md", report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-url", default="http://127.0.0.1:7870")
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", action="append", type=int, dest="gpus")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--job-timeout-seconds", type=float, default=3600.0)
    parser.add_argument(
        "--submission-max-attempts",
        type=int,
        default=DEFAULT_SUBMISSION_MAX_ATTEMPTS,
        help="Retry a Workspace job POST this many times before recording submission_failed.",
    )
    parser.add_argument(
        "--gpu-idle-samples",
        type=int,
        default=3,
        help="Require this many consecutive idle samples before assigning a job to a GPU.",
    )
    parser.add_argument(
        "--gpu-idle-memory-mib",
        type=int,
        default=32,
        help="A free GPU must use no more than this much baseline memory.",
    )
    parser.add_argument(
        "--gpu-idle-utilization-percent",
        type=int,
        default=0,
        help="A free GPU must remain at or below this utilization threshold.",
    )
    parser.add_argument(
        "--gpu-idle-ignore-compute-pid",
        action="append",
        type=int,
        dest="gpu_idle_ignored_compute_pids",
        help=(
            "Ignore this explicitly trusted nvidia-smi compute PID when applying the idle gate; "
            "repeat for multiple long-lived Workspace processes. Memory and utilization limits "
            "still apply, and every unlisted PID remains blocking."
        ),
    )
    parser.add_argument(
        "--require-all-gpus-idle-before-start",
        action="store_true",
        help=(
            "Wait for every selected GPU to pass the stable-idle gate before the first submission. "
            "By default each GPU joins independently so shared-cluster work does not block free cards."
        ),
    )
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--max-models", type=int)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Refresh the full catalog/checkpoint report without submitting inference jobs.",
    )
    parser.add_argument(
        "--previous-report",
        type=Path,
        help="Reuse completed/validated rows from another matrix.json while writing a new report.",
    )
    parser.add_argument(
        "--evidence-root",
        action="append",
        type=Path,
        dest="evidence_roots",
        help=(
            "Recursively import strict_video_validation JSONL and gang.json completion evidence. "
            "Defaults to the output directory's parent; repeat for additional bounded roots."
        ),
    )
    parser.add_argument("--retry-existing-failures", action="store_true")
    parser.add_argument(
        "--retry-model",
        action="append",
        dest="retry_models",
        help="Retry only this model while preserving the complete catalog report; repeat as needed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gpus = list(dict.fromkeys(args.gpus or [0, 1, 2, 3]))
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_roots = [
        path.expanduser().resolve()
        for path in (args.evidence_roots or [output_dir.parent])
    ]
    previous_rows: dict[str, dict[str, Any]] = {}
    previous_report_path = (
        args.previous_report.expanduser().resolve()
        if args.previous_report is not None
        else output_dir / "matrix.json"
    )
    if previous_report_path.is_file():
        try:
            previous_report = json.loads(previous_report_path.read_text(encoding="utf-8"))
            previous_rows = {
                str(row.get("model_id") or ""): row
                for row in previous_report.get("models") or ()
                if row.get("model_id")
            }
        except (OSError, json.JSONDecodeError):
            previous_rows = {}

    catalog = _request_json(args.workspace_url, "GET", "/api/models")
    targets = [
        model
        for model in catalog
        if model.get("category") in VIDEO_CATEGORIES and model.get("workload_type") in VIDEO_WORKLOADS
    ]
    if args.models:
        selected = set(args.models)
        targets = [model for model in targets if model.get("id") in selected]
    targets.sort(key=lambda item: str(item.get("id") or ""))
    persistent_evidence = _discover_persistent_evidence(
        evidence_roots,
        model_ids={str(model.get("id") or "") for model in targets},
    )

    existing_jobs = _request_json(args.workspace_url, "GET", "/api/jobs")
    existing_by_model: dict[str, dict[str, Any]] = {}
    for job in existing_jobs:
        model_id = str(job.get("model_id") or "")
        current = existing_by_model.get(model_id)
        job_key = (str(job.get("created_at") or ""), str(job.get("id") or ""))
        current_key = (str((current or {}).get("created_at") or ""), str((current or {}).get("id") or ""))
        if current is None or job_key > current_key:
            existing_by_model[model_id] = job

    rows_by_model: dict[str, dict[str, Any]] = {}
    runnable: list[dict[str, Any]] = []
    retry_models = set(args.retry_models or ())
    resolution_cache: dict[str, str] = {}
    structure_cache: dict[str, str] = {}
    for model in targets:
        model_id = str(model["id"])
        retry_this_model = args.retry_existing_failures or model_id in retry_models
        readiness = _checkpoint_readiness(
            model,
            checkpoint_root,
            resolution_cache=resolution_cache,
            structure_cache=structure_cache,
        )
        row = {
            "model_id": model_id,
            "name": model.get("name"),
            "category": model.get("category"),
            "workload_type": model.get("workload_type"),
            "default_variant_id": model.get("default_variant_id"),
            "default_task_id": model.get("default_task_id"),
            "checkpoint": readiness,
            "status": "pending" if readiness["ready"] else "blocked_checkpoint",
            "blocker": readiness["reason"],
        }
        previous = existing_by_model.get(model_id)
        previous_row = previous_rows.get(model_id)
        completion_row: dict[str, Any] | None = None
        if (
            previous_row
            and previous_row.get("status") == "completed"
            and (previous_row.get("validation") or {}).get("valid")
        ):
            completion_row = dict(previous_row)
        evidence_row = persistent_evidence.get(model_id)
        if evidence_row is not None:
            completion_row = {**(completion_row or {}), **evidence_row}
        if completion_row is not None and not retry_this_model:
            row.update(completion_row)
            row["checkpoint"] = readiness
        elif previous and not (retry_this_model and previous.get("status") in {"failed", "cancelled"}):
            row.update(
                {
                    "job_id": previous.get("id"),
                    "gpu": str((previous.get("metadata") or {}).get("device") or "").removeprefix("cuda:"),
                    "status": previous.get("status"),
                    "error": previous.get("error"),
                    "job": _compact_job(previous),
                }
            )
            if previous.get("status") == "completed":
                row["validation"] = _validate_completed_job(previous)
                if not row["validation"]["valid"]:
                    row["status"] = "validation_failed"
        elif previous_row and _can_reuse_previous_row(
            previous_row,
            readiness,
            retry_this_model=retry_this_model,
        ):
            row.update(previous_row)
            row["checkpoint"] = readiness
        elif readiness["ready"] and (not retry_models or model_id in retry_models):
            gang_blocker = _multi_gpu_gang_blocker(model_id)
            if gang_blocker:
                row["status"] = "pending_multi_gpu_gang"
                row["blocker"] = gang_blocker
            else:
                runnable.append(model)
        _refresh_checkpoint_blocker(row, readiness)
        input_blocker = _input_blocker(model, readiness)
        if input_blocker:
            row["status"] = "blocked_input"
            row["blocker"] = input_blocker
        if row.get("status") == "completed":
            compact_job = row.get("job") or {}
            elapsed_seconds = _job_elapsed_seconds(compact_job)
            if elapsed_seconds is not None:
                row["elapsed_seconds"] = elapsed_seconds
            videos = ((row.get("validation") or {}).get("videos") or [])
            if compact_job and videos and any(not item.get("sha256") for item in videos):
                row["validation"] = _validate_completed_job(compact_job)
        rows_by_model[model_id] = row

    if args.max_models is not None:
        runnable = runnable[: max(0, args.max_models)]
    if args.audit_only:
        runnable = []

    report: dict[str, Any] = {
        "schema_version": "worldfoundry-workspace-video-matrix-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workspace_url": args.workspace_url,
        "checkpoint_root": str(checkpoint_root),
        "evidence_roots": [str(path) for path in evidence_roots],
        "persistent_evidence_models": sorted(persistent_evidence),
        "gpus": gpus,
        "audit_only": bool(args.audit_only),
        "models": [rows_by_model[str(model["id"])] for model in targets],
    }
    _persist(output_dir, report)

    active: dict[int, dict[str, Any]] = {}
    for job in existing_jobs:
        if job.get("status") not in TERMINAL_STATUSES:
            device = str((job.get("metadata") or {}).get("device") or "")
            if device.startswith("cuda:") and device[5:].isdigit():
                gpu = int(device[5:])
                if gpu in gpus:
                    active[gpu] = {"job_id": job["id"], "model_id": job["model_id"], "started": time.monotonic()}

    queue = [model for model in runnable if str(model["id"]) not in {item["model_id"] for item in active.values()}]
    print(f"targets={len(targets)} runnable={len(runnable)} queued={len(queue)} active={len(active)}", flush=True)

    required_idle_samples = max(1, int(args.gpu_idle_samples))
    submission_max_attempts = max(1, int(args.submission_max_attempts))
    submission_attempts: dict[str, int] = {}
    idle_samples = {gpu: 0 for gpu in gpus}
    initial_global_gate_complete = (
        not args.require_all_gpus_idle_before_start or not queue or bool(active)
    )
    initial_global_idle_samples = 0
    last_idle_report = 0.0
    while queue or active:
        activity = _gpu_activity() if queue else {}
        for gpu in gpus:
            if gpu in active:
                idle_samples[gpu] = 0
            elif _gpu_is_idle(
                activity,
                gpu,
                memory_limit_mib=max(0, int(args.gpu_idle_memory_mib)),
                utilization_limit_percent=max(0, int(args.gpu_idle_utilization_percent)),
                ignored_compute_pids=set(args.gpu_idle_ignored_compute_pids or ()),
            ):
                idle_samples[gpu] = min(required_idle_samples, idle_samples[gpu] + 1)
            else:
                idle_samples[gpu] = 0
        if not initial_global_gate_complete:
            if all(idle_samples[gpu] > 0 for gpu in gpus):
                initial_global_idle_samples += 1
            else:
                initial_global_idle_samples = 0
            if initial_global_idle_samples >= required_idle_samples:
                initial_global_gate_complete = True
                print(
                    f"all {len(gpus)} GPUs remained idle for {required_idle_samples} consecutive samples",
                    flush=True,
                )
        waiting_gpus = [
            gpu
            for gpu in gpus
            if gpu not in active and queue and idle_samples[gpu] < required_idle_samples
        ]
        if (waiting_gpus or not initial_global_gate_complete) and time.monotonic() - last_idle_report >= 60.0:
            states = ", ".join(
                f"gpu={gpu} samples={idle_samples[gpu]}/{required_idle_samples} state={activity.get(gpu, {})}"
                for gpu in gpus
            )
            prefix = "waiting for all GPUs to become stably idle" if not initial_global_gate_complete else "waiting for stable idle GPUs"
            print(f"{prefix}: {states}", flush=True)
            last_idle_report = time.monotonic()
        if not initial_global_gate_complete:
            time.sleep(max(1.0, args.poll_seconds))
            continue
        for gpu in gpus:
            if gpu in active or not queue or idle_samples[gpu] < required_idle_samples:
                continue
            model = queue.pop(0)
            model_id = str(model["id"])
            readiness = rows_by_model[model_id]["checkpoint"]
            payload: dict[str, Any] = {
                "job_type": "inference",
                "model_id": model_id,
                "device": f"cuda:{gpu}",
            }
            resolved_ref = str(readiness.get("resolved_model_ref") or "")
            raw_ref = str(readiness.get("raw_model_ref") or "")
            if resolved_ref and resolved_ref != raw_ref:
                payload["model_ref"] = resolved_ref
            attempt = submission_attempts.get(model_id, 0) + 1
            submission_attempts[model_id] = attempt
            try:
                job = _request_json(args.workspace_url, "POST", "/api/jobs", payload)
            except Exception as exc:  # noqa: BLE001 - every model must remain represented in the matrix.
                retrying = _record_submission_failure(
                    queue=queue,
                    model=model,
                    row=rows_by_model[model_id],
                    gpu=gpu,
                    attempt=attempt,
                    max_attempts=submission_max_attempts,
                    exc=exc,
                )
                outcome = "submission_retry" if retrying else "submission_failed"
                print(
                    f"gpu={gpu} model={model_id} {outcome} "
                    f"attempt={attempt}/{submission_max_attempts}: {exc}",
                    flush=True,
                )
                # A POST failure commonly means the whole Workspace endpoint is
                # temporarily unavailable.  Reset every idle sample and leave
                # this dispatch pass so four GPUs cannot burn the retry budget
                # in one loop iteration.
                for idle_gpu in idle_samples:
                    idle_samples[idle_gpu] = 0
                _persist(output_dir, report)
                break
            active[gpu] = {"job_id": job["id"], "model_id": model_id, "started": time.monotonic()}
            idle_samples[gpu] = 0
            rows_by_model[model_id].pop("error", None)
            rows_by_model[model_id].update(
                status=job.get("status"),
                job_id=job["id"],
                gpu=gpu,
                peak_memory_mib=0,
                submission_attempts=attempt,
                submission_max_attempts=submission_max_attempts,
            )
            print(f"gpu={gpu} model={model_id} job={job['id']} submitted", flush=True)
            _persist(output_dir, report)

        time.sleep(max(1.0, args.poll_seconds))
        gpu_memory = _gpu_memory_mib()
        for gpu, active_job in list(active.items()):
            model_id = active_job["model_id"]
            row = rows_by_model[model_id]
            row["peak_memory_mib"] = max(int(row.get("peak_memory_mib") or 0), gpu_memory.get(gpu, 0))
            elapsed = time.monotonic() - active_job["started"]
            if elapsed > args.job_timeout_seconds:
                _request_json(args.workspace_url, "POST", f"/api/jobs/{active_job['job_id']}/stop", {})
                row.update(
                    status="timeout",
                    error=f"job exceeded {args.job_timeout_seconds:.0f}s",
                    elapsed_seconds=round(elapsed, 3),
                )
                print(f"gpu={gpu} model={model_id} timeout", flush=True)
                del active[gpu]
                _persist(output_dir, report)
                continue
            job = _request_json(args.workspace_url, "GET", f"/api/jobs/{active_job['job_id']}")
            row["status"] = job.get("status")
            if job.get("status") not in TERMINAL_STATUSES:
                continue
            row["error"] = job.get("error")
            row["job"] = _compact_job(job)
            row["elapsed_seconds"] = _job_elapsed_seconds(job) or round(elapsed, 3)
            if job.get("status") == "completed":
                row["validation"] = _strictly_validate_completed_job(
                    job,
                    output_dir=output_dir,
                    model_id=model_id,
                )
                if not row["validation"]["valid"]:
                    row["status"] = "validation_failed"
            print(
                f"gpu={gpu} model={model_id} status={row['status']} peak_memory_mib={row.get('peak_memory_mib', 0)}",
                flush=True,
            )
            del active[gpu]
            _persist(output_dir, report)

    _persist(output_dir, report)
    failures = sum(1 for row in report["models"] if row.get("status") != "completed")
    print(f"matrix complete: targets={len(targets)} failures_or_blockers={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

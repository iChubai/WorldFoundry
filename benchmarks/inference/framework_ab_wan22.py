"""Reproducible Wan2.2-TI2V-5B E2E comparison with FastVideo/LightX2V.

The two frameworks intentionally run in separate Python environments.  This
module keeps imports framework-local, freezes all semantic generation inputs,
measures model load and generation separately, samples whole-card memory, and
fully decodes every emitted video before accepting a sample.

Generation timing includes text encoding, denoising, VAE decode, host transfer,
frame conversion, and MP4 encoding.  It excludes Python import, model load, and
post-run validation.  Run the paired launcher in ``run_framework_ab_wan22.sh``
for globally serial, order-balanced sampling on one physical GPU.

Independent framework implementations are not expected to map a seed to the
same diffusion trajectory. Cross-framework PSNR/SSIM/LPIPS is therefore an
opt-in pixel diagnostic, not a default quality-preservation gate. Exact and
approximate WorldFoundry optimizations are certified against their own frozen
baseline by ``wan22_multigpu_e2e.py``; framework A/B always requires complete
video decode and leaves semantic/absolute quality to the benchmark suite and
human review.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
import uuid as uuidlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from benchmarks.inference.approximate_attention_receipts import (
    audit_approximate_attention_receipt,
)

SCHEMA_VERSION = "worldfoundry-framework-ab-v4"
ISOLATION_AUDIT_VERSION = "worldfoundry-gpu-isolation-v4"
MAX_CERTIFYING_POLL_INTERVAL_SECONDS = 0.25
GPU_QUERY_TIMEOUT_SECONDS = 10.0
MIN_CERTIFYING_PAIRS = 6
MIN_CERTIFYING_PAIRS_PER_ORDER = 3
PROTOCOL_ID = "wan22-ti2v-5b-e2e-v2"
DEFAULT_PROMPT = "a cat walking in a garden, cinematic"
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
WORLD_FOUNDRY_FEATURE_CACHE_ALGORITHMS = (
    "teacache",
    "magcache",
    "adacache",
    "taylorseer",
    "blocktaylorseer",
    "custom",
    "firstblock",
    "dualblock",
    "dynamicblock",
)
WORLD_FOUNDRY_BLOCK_FEATURE_CACHE_ALGORITHMS = frozenset(
    {"blocktaylorseer", "firstblock", "dualblock", "dynamicblock"}
)
WORLD_FOUNDRY_REQUEST_RECEIPT_CACHE_ALGORITHMS = frozenset(
    {*WORLD_FOUNDRY_BLOCK_FEATURE_CACHE_ALGORITHMS, "custom"}
)
WAN22_TI2V_TRANSFORMER_BLOCKS = 30
WAN22_TI2V_ATTENTION_HEADS = 24
WAN22_TI2V_ATTENTION_HEAD_SIZE = 128
WORLD_FOUNDRY_VMOBA_PROFILE_FIELDS = frozenset(
    {
        "temporal_chunk_size",
        "temporal_topk",
        "spatial_chunk_size",
        "spatial_topk",
        "st_chunk_size",
        "st_topk",
        "moba_select_mode",
        "moba_threshold",
        "moba_threshold_type",
        "first_full_step",
        "first_full_layer",
        "temporal_layer",
        "spatial_layer",
        "st_layer",
    }
)
WORLD_FOUNDRY_VMOBA_PROVIDER_PATH = "fastvideo_kernel.moba_attn_varlen"
WORLD_FOUNDRY_VMOBA_PROVIDER_SYMBOLS = (
    "moba_attn_varlen",
    "process_moba_input",
    "process_moba_output",
)
WORLD_FOUNDRY_LIGHTX2V_SPARSE_KINDS = (
    "dynamic_sparse",
    "sparge",
    "nbhd",
    "lightx2v_sla_mask",
    "flexblock",
    "lightx2v_spas_sage",
    "draft_attn",
    "radial_attn",
    "rainfusion_attn",
    "svg_attn",
    "svg2_attn",
    "lightx2v_svg_mask",
)
WORLD_FOUNDRY_LIGHTX2V_PROFILE_FIELDS = frozenset(
    {
        "lightx2v_operator",
        "nbhd_coefficient",
        "nbhd_min_width",
        "attnmap_frame_num",
        "lightx2v_per_block_mean",
        "lightx2v_pool_size",
        "lightx2v_skip_timesteps",
        "lightx2v_dense_attn_type",
        "svg_sample_mse_max_row",
        "svg_num_sampled_rows",
        "svg_context_length",
    }
)
WORLD_FOUNDRY_PINNED_LIGHTX2V_COMMIT = (
    "6fb7c1362b89d4908a9ea197bac4fbd7482ee2d5"
)
WORLD_FOUNDRY_FASTVIDEO_SLA_KINDS = (
    "fastvideo_sla",
    "fastvideo_sagesla",
)
WORLD_FOUNDRY_FASTVIDEO_SLA_PROFILE_FIELDS = frozenset(
    {
        "fastvideo_topk_ratio",
        "fastvideo_feature_map",
    }
)
WORLD_FOUNDRY_PINNED_FASTVIDEO_COMMIT = (
    "1b2b2a0161bc6b3b80158d1fa6380a051c6530c7"
)
FASTVIDEO_SLA_CHECKPOINT_LAYOUT = "fastvideo_diffusers"
FASTVIDEO_SLA_REFERENCE_CONTRACTS = {
    "fastvideo_sla": {
        "backend": "SLA_ATTN",
        "provider_path": "fastvideo.attention.backends.sla.SLAAttentionImpl",
        "provider_family": "fastvideo/sparse-linear-attention",
        "topk_ratio": 0.1,
    },
    "fastvideo_sagesla": {
        "backend": "SAGE_SLA_ATTN",
        "provider_path": (
            "fastvideo.attention.backends.sla.SageSLAAttentionImpl"
        ),
        "provider_family": "fastvideo/sage-sparse-linear-attention",
        "topk_ratio": 0.5,
    },
}
WORLD_FOUNDRY_STA_SUPPORTED_GRIDS = frozenset(
    {(36, 48, 48), (18, 48, 80)}
)
WORLD_FOUNDRY_APPROXIMATE_PROVIDER_PATHS = {
    "sta": frozenset({"fastvideo_kernel.sliding_tile_attention"}),
    "vsa": frozenset(
        {
            "fastvideo_kernel.video_sparse_attn",
            "fastvideo_kernel.video_sparse_attn_bshd",
        }
    ),
    "vmoba": frozenset({WORLD_FOUNDRY_VMOBA_PROVIDER_PATH}),
}


def _wan22_ti2v_patch_grid(*, frames: int, height: int, width: int) -> tuple[int, int, int]:
    """Return the post-patch Wan2.2 TI2V-5B grid used by sparse providers."""

    if (frames - 1) % 4:
        raise ValueError("Wan2.2 TI2V frames must satisfy (frames - 1) % 4 == 0")
    # TI2V-5B uses VAE spatial compression 16 followed by a 2x2 DiT patch.
    if height % 32 or width % 32:
        raise ValueError(
            "Wan2.2 TI2V sparse attention requires height and width divisible by 32; "
            f"got {height}x{width}"
        )
    return ((frames - 1) // 4 + 1, height // 32, width // 32)


def _validate_vmoba_benchmark_profile(
    config: Any,
    *,
    grid: tuple[int, int, int],
    steps: int,
    wrapped_blocks: int,
) -> None:
    """Reject a VMoBA profile that cannot exercise the complete benchmark."""

    failures: list[str] = []
    if grid[0] % int(config.temporal_chunk_size):
        failures.append(
            f"temporal grid {grid[0]} is not divisible by "
            f"temporal_chunk_size={config.temporal_chunk_size}"
        )
    spatial_chunk = tuple(int(value) for value in config.spatial_chunk_size)
    if grid[1] % spatial_chunk[0] or grid[2] % spatial_chunk[1]:
        failures.append(
            f"spatial grid {grid[1:]} is not divisible by "
            f"spatial_chunk_size={spatial_chunk}"
        )
    st_chunk = tuple(int(value) for value in config.st_chunk_size)
    if any(size % chunk for size, chunk in zip(grid, st_chunk)):
        failures.append(
            f"patch grid {grid} is not divisible by st_chunk_size={st_chunk}"
        )
    if int(config.first_full_step) >= steps:
        failures.append(
            f"first_full_step={config.first_full_step} leaves no sparse step "
            f"in a {steps}-step workload"
        )
    if int(config.first_full_layer) >= wrapped_blocks:
        failures.append(
            f"first_full_layer={config.first_full_layer} leaves no sparse layer "
            f"in a {wrapped_blocks}-block workload"
        )
    cycle = (
        ("temporal", int(config.temporal_layer)),
        ("spatial", int(config.spatial_layer)),
        ("spatiotemporal", int(config.st_layer)),
    )
    cycle_length = sum(count for _, count in cycle)
    exercised_routes: set[str] = set()
    for layer_idx in range(int(config.first_full_layer), wrapped_blocks):
        position = (layer_idx - int(config.first_full_layer)) % cycle_length
        for route, count in cycle:
            if position < count:
                exercised_routes.add(route)
                break
            position -= count
    missing_routes = sorted(
        {"temporal", "spatial", "spatiotemporal"} - exercised_routes
    )
    if missing_routes:
        failures.append(
            "active VMoBA layers do not exercise every route; missing "
            f"{missing_routes}"
        )
    if failures:
        raise ValueError(
            f"VMoBA profile is incompatible with patch grid {grid}: "
            + "; ".join(failures)
        )


def _lightx2v_sparse_expectation(config: Any) -> dict[str, str]:
    """Resolve the exact pinned upstream operator selected by one config."""

    from worldfoundry.base_models.diffusion_model.optimizations.lightx2v_sparse import (
        PINNED_LIGHTX2V_COMMIT,
        LightX2VSparseConfig,
        lightx2v_kernel_symbol,
        lightx2v_provider_family,
    )

    lightx2v = LightX2VSparseConfig(
        kind=config.kind,
        sparsity_ratio=config.sparsity,
        operator=config.lightx2v_operator,
        nbhd_coefficient=config.nbhd_coefficient,
        nbhd_min_width=config.nbhd_min_width,
        attnmap_frame_num=config.attnmap_frame_num,
        per_block_mean=config.lightx2v_per_block_mean,
        pool_size=config.lightx2v_pool_size,
        skip_timesteps=config.lightx2v_skip_timesteps,
        dense_attn_type=config.lightx2v_dense_attn_type,
        svg_sample_mse_max_row=config.svg_sample_mse_max_row,
        svg_num_sampled_rows=config.svg_num_sampled_rows,
        svg_context_length=config.svg_context_length,
    )
    if PINNED_LIGHTX2V_COMMIT != WORLD_FOUNDRY_PINNED_LIGHTX2V_COMMIT:
        raise RuntimeError(
            "benchmark/core LightX2V commit contract mismatch: "
            f"{WORLD_FOUNDRY_PINNED_LIGHTX2V_COMMIT} != {PINNED_LIGHTX2V_COMMIT}"
        )
    return {
        "operator": str(lightx2v.operator),
        "provider_symbol": lightx2v_kernel_symbol(lightx2v),
        "provider_family": lightx2v_provider_family(lightx2v),
        "commit": PINNED_LIGHTX2V_COMMIT,
        "source_fingerprint": hashlib.sha256(
            f"{PINNED_LIGHTX2V_COMMIT}\0True\0".encode()
        ).hexdigest(),
    }


def _fastvideo_sla_source_identity(repository: Path) -> dict[str, Any]:
    """Reproduce the core adapter's pinned FastVideo source fingerprint."""

    repository = repository.expanduser().resolve(strict=True)
    revision = _run(
        ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD"],
        check=False,
    )
    status = _run(
        [
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            "fastvideo",
            "fastvideo-kernel",
        ],
        check=False,
    )
    commit = revision.stdout.strip().casefold()
    clean = status.returncode == 0 and not status.stdout.strip()
    if revision.returncode != 0 or commit != WORLD_FOUNDRY_PINNED_FASTVIDEO_COMMIT:
        raise RuntimeError(
            "FastVideo learned-SLA benchmarking requires pinned commit "
            f"{WORLD_FOUNDRY_PINNED_FASTVIDEO_COMMIT}; got {commit!r}"
        )
    if not clean:
        raise RuntimeError(
            "FastVideo learned-SLA benchmarking requires a clean fastvideo/"
            "fastvideo-kernel source tree"
        )

    source_file = repository / "fastvideo" / "attention" / "backends" / "sla.py"
    kernel_file = (
        repository
        / "fastvideo-kernel"
        / "python"
        / "fastvideo_kernel"
        / "triton_kernels"
        / "sla_triton.py"
    )
    if not source_file.is_file():
        raise FileNotFoundError(source_file)
    source_files = [source_file]
    if kernel_file.is_file():
        source_files.append(kernel_file)
    digest = hashlib.sha256()
    digest.update(f"{commit}\0{clean}\0{status.stdout}".encode())
    for path in sorted(source_files, key=str):
        digest.update(str(path.relative_to(repository)).encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return {
        "commit": commit,
        "clean": clean,
        "source_fingerprint": digest.hexdigest(),
        "source_root": str(repository),
        "source_file": str(source_file),
    }


def _fastvideo_sla_checkpoint_expectation(checkpoint: Path) -> dict[str, Any]:
    """Hash the exact trained ``proj_l`` set before either framework runs.

    The preflight opens only safetensors shards on CPU.  It deliberately uses
    the same resolver as the WorldFoundry adapter, while keeping the resulting
    expectation independent from both frameworks' later runtime receipts.
    """

    from worldfoundry.base_models.diffusion_model.optimizations.fastvideo_sla import (
        FastVideoSLAContractError,
        FastVideoSLALayerSpec,
        resolve_fastvideo_sla_projection_weights,
    )

    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError(
            "FastVideo learned-SLA A/B preflight requires safetensors"
        ) from error

    resolved_checkpoint = checkpoint.expanduser().resolve()
    if resolved_checkpoint.is_file():
        tensor_files = (
            [resolved_checkpoint]
            if resolved_checkpoint.suffix == ".safetensors"
            else []
        )
    elif resolved_checkpoint.is_dir():
        tensor_files = sorted(resolved_checkpoint.rglob("*.safetensors"))
    else:
        raise FileNotFoundError(resolved_checkpoint)
    if not tensor_files:
        raise RuntimeError(
            "FastVideo learned-SLA A/B requires a safetensors checkpoint "
            "containing trained proj_l tensors"
        )

    projection_state: dict[str, Any] = {}
    projection_sources: dict[str, Path] = {}
    for tensor_file in tensor_files:
        try:
            with safe_open(tensor_file, framework="pt", device="cpu") as handle:
                for key in (name for name in handle.keys() if "proj_l" in name):
                    if key in projection_state:
                        raise RuntimeError(
                            "FastVideo learned-SLA checkpoint contains duplicate "
                            f"projection key {key!r} in "
                            f"{projection_sources[key]} and {tensor_file}"
                        )
                    projection_state[key] = handle.get_tensor(key)
                    projection_sources[key] = tensor_file
        except RuntimeError:
            raise
        except Exception as error:
            raise RuntimeError(
                "cannot inspect FastVideo learned-SLA checkpoint shard "
                f"{tensor_file}: {type(error).__name__}: {error}"
            ) from error
    if not projection_state:
        raise RuntimeError(
            "FastVideo learned-SLA requires trained proj_l tensors; none were "
            f"found under {resolved_checkpoint}"
        )

    specs = tuple(
        FastVideoSLALayerSpec(
            layer_index=layer_index,
            num_heads=WAN22_TI2V_ATTENTION_HEADS,
            head_size=WAN22_TI2V_ATTENTION_HEAD_SIZE,
        )
        for layer_index in range(WAN22_TI2V_TRANSFORMER_BLOCKS)
    )
    try:
        resolved = resolve_fastvideo_sla_projection_weights(
            projection_state,
            specs,
        )
    except FastVideoSLAContractError as error:
        raise RuntimeError(
            "FastVideo learned-SLA checkpoint projection preflight failed: "
            f"{error}"
        ) from error
    if resolved.layout != FASTVIDEO_SLA_CHECKPOINT_LAYOUT:
        raise RuntimeError(
            "FastVideo framework A/B requires the upstream Diffusers proj_l "
            f"layout {FASTVIDEO_SLA_CHECKPOINT_LAYOUT!r}; got {resolved.layout!r}"
        )
    layer_fingerprints = {
        str(layer_index): projection.fingerprint
        for layer_index, projection in resolved.projections.items()
    }
    canonical_fingerprint = _fastvideo_projection_set_fingerprint(
        layer_fingerprints,
        checkpoint_layout=None,
    )
    recomputed_layout_fingerprint = _fastvideo_projection_set_fingerprint(
        layer_fingerprints,
        checkpoint_layout=resolved.layout,
    )
    if recomputed_layout_fingerprint != resolved.fingerprint:
        raise RuntimeError(
            "FastVideo learned-SLA checkpoint fingerprint contract drift: "
            f"{recomputed_layout_fingerprint} != {resolved.fingerprint}"
        )
    return {
        "checkpoint_path": str(resolved_checkpoint),
        "checkpoint_layout": resolved.layout,
        "projection_fingerprint": resolved.fingerprint,
        "canonical_projection_fingerprint": canonical_fingerprint,
        "layer_projection_fingerprints": layer_fingerprints,
    }


def _fastvideo_sla_checkpoint_expectation_for_args(
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Cache one immutable CPU preflight result on the parsed request."""

    checkpoint = (args.fastvideo_checkpoint or args.checkpoint).expanduser().resolve()
    cached = getattr(args, "_fastvideo_sla_checkpoint_preflight", None)
    if (
        isinstance(cached, Mapping)
        and cached.get("checkpoint_path") == str(checkpoint)
    ):
        return dict(cached)
    expectation = _fastvideo_sla_checkpoint_expectation(checkpoint)
    setattr(args, "_fastvideo_sla_checkpoint_preflight", dict(expectation))
    return expectation


def _fastvideo_sla_expectation(
    config: Any,
    *,
    fastvideo_repository: Path,
    checkpoint_expectation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze the learned provider/source contract independently of receipts."""

    contract = FASTVIDEO_SLA_REFERENCE_CONTRACTS.get(str(config.kind))
    if contract is None:
        raise ValueError(f"unsupported FastVideo learned-SLA kind: {config.kind!r}")
    source = _fastvideo_sla_source_identity(fastvideo_repository)
    expectation = {
        "provider_path": contract["provider_path"],
        "provider_family": contract["provider_family"],
        "commit": source["commit"],
        "source_fingerprint": source["source_fingerprint"],
        "checkpoint_layout": FASTVIDEO_SLA_CHECKPOINT_LAYOUT,
        "projection_fingerprint": None,
    }
    if checkpoint_expectation is not None:
        expectation.update(
            checkpoint_layout=checkpoint_expectation.get("checkpoint_layout"),
            projection_fingerprint=checkpoint_expectation.get(
                "projection_fingerprint"
            ),
            canonical_projection_fingerprint=checkpoint_expectation.get(
                "canonical_projection_fingerprint"
            ),
            layer_projection_fingerprints=dict(
                checkpoint_expectation.get("layer_projection_fingerprints")
                or {}
            ),
        )
    return expectation


def _normalized_approximate_attention_profile(config: Any) -> dict[str, Any]:
    """Freeze parser defaults as the trusted side of a runtime receipt audit."""

    return {
        "kind": config.kind,
        "sparsity": config.sparsity,
        "dense_steps": config.dense_steps,
        "window": list(config.window),
        "block_tile": list(config.block_tile),
        "lightx2v_operator": config.lightx2v_operator,
        "nbhd_coefficient": list(config.nbhd_coefficient),
        "nbhd_min_width": config.nbhd_min_width,
        "attnmap_frame_num": config.attnmap_frame_num,
        "lightx2v_per_block_mean": config.lightx2v_per_block_mean,
        "lightx2v_pool_size": config.lightx2v_pool_size,
        "lightx2v_skip_timesteps": config.lightx2v_skip_timesteps,
        "lightx2v_dense_attn_type": config.lightx2v_dense_attn_type,
        "svg_sample_mse_max_row": config.svg_sample_mse_max_row,
        "svg_num_sampled_rows": config.svg_num_sampled_rows,
        "svg_context_length": config.svg_context_length,
        "fastvideo_topk_ratio": config.fastvideo_topk_ratio,
        "fastvideo_feature_map": config.fastvideo_feature_map,
        "temporal_chunk_size": config.temporal_chunk_size,
        "temporal_topk": config.temporal_topk,
        "spatial_chunk_size": list(config.spatial_chunk_size),
        "spatial_topk": config.spatial_topk,
        "st_chunk_size": list(config.st_chunk_size),
        "st_topk": config.st_topk,
        "moba_select_mode": config.moba_select_mode,
        "moba_threshold": config.moba_threshold,
        "moba_threshold_type": config.moba_threshold_type,
        "first_full_step": config.first_full_step,
        "first_full_layer": config.first_full_layer,
        "temporal_layer": config.temporal_layer,
        "spatial_layer": config.spatial_layer,
        "st_layer": config.st_layer,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(
    command: list[str],
    *,
    check: bool = True,
    timeout_s: float | None = 120.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )


def _validation_ffmpeg_executable() -> str:
    """Resolve one deterministic decoder for post-run video validation.

    Generation and validation are separate timing domains, but both must avoid
    an arbitrary PATH binary when ImageIO already provides the pinned FFmpeg
    used by WorldFoundry's default save path.  An explicit benchmark override
    remains available for hermetic release workers.
    """

    explicit = os.getenv("WAN22_AB_VALIDATION_FFMPEG")
    if explicit is not None:
        normalized = explicit.strip()
        if not normalized:
            raise ValueError("WAN22_AB_VALIDATION_FFMPEG must not be empty")
        return normalized
    try:
        import imageio_ffmpeg

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, OSError, RuntimeError):
        bundled = None
    if isinstance(bundled, str) and bundled.strip():
        return bundled
    return shutil.which("ffmpeg") or "ffmpeg"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in (
        "fastvideo",
        "fastvideo-kernel",
        "lightx2v",
        "torch",
        "flash-attn",
        "sageattention",
        "triton",
        "vllm",
        "q8-kernels",
        "sgl-kernel",
        "diffusers",
        "transformers",
        "safetensors",
        "numpy",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _git_state(path: Path, *, include_dirty: bool = True) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        commit = _run(["git", "-C", str(resolved), "rev-parse", "HEAD"]).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {
            "path": str(resolved),
            "commit": None,
            "dirty": None,
            "dirty_fingerprint_sha256": None,
        }
    if not include_dirty:
        return {
            "path": str(resolved),
            "commit": commit,
            "dirty": None,
            "dirty_fingerprint_sha256": None,
            "scope": "commit plus separate content-level runtime source fingerprint",
        }
    status = _run(
        [
            "git",
            "-C",
            str(resolved),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=False,
    )
    diff = _run(
        ["git", "-C", str(resolved), "diff", "--binary", "HEAD", "--"],
        check=False,
    )
    status_lines = sorted(line.rstrip() for line in status.stdout.splitlines() if line.strip())
    tracked_diff_sha256 = hashlib.sha256(diff.stdout.encode("utf-8")).hexdigest()
    dirty_identity = {
        "commit": commit,
        "status": status_lines,
        "tracked_diff_sha256": tracked_diff_sha256,
    }
    return {
        "path": str(resolved),
        "commit": commit,
        "dirty": bool(status_lines or diff.stdout),
        "dirty_entry_count": len(status_lines),
        "dirty_status_sha256": _canonical_digest(status_lines),
        "tracked_diff_sha256": tracked_diff_sha256,
        "dirty_fingerprint_sha256": _canonical_digest(dirty_identity),
    }


def _source_file_fingerprint(path: Path, *, repository: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    repository = repository.resolve()
    try:
        relative_path = str(resolved.relative_to(repository))
    except ValueError:
        relative_path = str(resolved)
    status = _run(
        [
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            relative_path,
        ],
        check=False,
    )
    git_status = status.stdout.strip() if status.returncode == 0 else None
    return {
        "path": str(resolved),
        "relative_path": relative_path,
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
        "git_status": git_status,
        "dirty": bool(git_status),
    }


def _benchmark_source_fingerprint() -> dict[str, Any]:
    runner = Path(__file__).resolve()
    repository = runner.parents[2]
    files = {
        "runner": _source_file_fingerprint(runner, repository=repository),
        "launcher": _source_file_fingerprint(
            runner.with_name("run_framework_ab_wan22.sh"),
            repository=repository,
        ),
    }
    identity = {
        name: {
            "relative_path": item["relative_path"],
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
            "git_status": item["git_status"],
        }
        for name, item in files.items()
    }
    return {
        "files": files,
        "dirty": any(bool(item["dirty"]) for item in files.values()),
        "identity_sha256": _canonical_digest(identity),
    }


def _source_tree_fingerprint(
    *,
    repository: Path,
    roots: Iterable[Path],
    explicit_files: Iterable[Path] = (),
    suffixes: tuple[str, ...] = (".py",),
) -> dict[str, Any]:
    """Hash runtime source content, including untracked files."""

    repository = repository.resolve()
    files = {path.resolve() for path in explicit_files if path.is_file()}
    for root in roots:
        if not root.is_dir():
            continue
        for suffix in suffixes:
            files.update(path.resolve() for path in root.rglob(f"*{suffix}") if path.is_file())
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.as_posix()):
        try:
            relative = path.relative_to(repository).as_posix()
        except ValueError:
            relative = path.as_posix()
        file_sha256 = _sha256(path)
        records.append({"path": relative, "sha256": file_sha256})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256.encode("ascii"))
        digest.update(b"\n")
    if not records:
        raise FileNotFoundError(f"runtime source fingerprint found no files under {repository}")
    return {
        "repository": str(repository),
        "sha256": digest.hexdigest(),
        "file_count": len(records),
        "files": records,
    }


def _untracked_source_fingerprint(
    *,
    repository: Path,
    roots: Iterable[Path],
) -> dict[str, Any]:
    """Hash untracked runtime sources; tracked content is covered by git state."""

    repository = repository.resolve()
    pathspecs = []
    for root in roots:
        try:
            pathspecs.append(root.resolve().relative_to(repository).as_posix())
        except ValueError:
            continue
    command = [
        "git",
        "-C",
        str(repository),
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        *pathspecs,
    ]
    completed = _run(command, check=False)
    files = [
        repository / relative
        for relative in completed.stdout.splitlines()
        if relative.strip()
    ]
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(repository).as_posix()
        file_sha256 = _sha256(path)
        records.append({"path": relative, "sha256": file_sha256})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256.encode("ascii"))
        digest.update(b"\n")
    return {
        "repository": str(repository),
        "sha256": digest.hexdigest(),
        "file_count": len(records),
        "scope": "untracked runtime source; tracked content is covered by commit and diff",
        "files": records,
    }


def _runtime_source_fingerprints(
    *,
    worldfoundry_repository: Path,
    fastvideo_repository: Path,
    lightx2v_repository: Path,
) -> dict[str, dict[str, Any]]:
    worldfoundry_repository = worldfoundry_repository.resolve()
    worldfoundry = worldfoundry_repository / "worldfoundry"
    diffusion = worldfoundry / "base_models" / "diffusion_model"
    worldfoundry_roots = (
        worldfoundry / "pipelines" / "wan",
        diffusion / "loaders",
        diffusion / "models" / "autoencoders" / "wan",
        diffusion / "models" / "encoders" / "wan",
        diffusion / "models" / "networks" / "wan",
        diffusion / "optimizations",
        diffusion / "recipes" / "wan_configs" / "wan22",
        diffusion / "runners",
        worldfoundry / "core" / "acceleration",
        worldfoundry / "core" / "attention",
        worldfoundry / "core" / "configuration",
        worldfoundry / "core" / "distributed",
        worldfoundry / "core" / "io",
        worldfoundry / "core" / "kernels",
        worldfoundry / "core" / "memory",
        worldfoundry / "core" / "model_loading",
    )
    worldfoundry_explicit = (
        *(diffusion.glob("*.py")),
        diffusion / "models" / "denoisers" / "wan.py",
        diffusion / "recipes" / "wan.py",
        diffusion / "registry.py",
        diffusion / "schedulers" / "wan.py",
        worldfoundry / "core" / "contracts.py",
        worldfoundry / "core" / "device.py",
        worldfoundry / "core" / "inference.py",
        worldfoundry / "core" / "video_postprocess.py",
    )
    fastvideo_repository = fastvideo_repository.resolve()
    lightx2v_repository = lightx2v_repository.resolve()
    return {
        "worldfoundry": _source_tree_fingerprint(
            repository=worldfoundry_repository,
            roots=worldfoundry_roots,
            explicit_files=worldfoundry_explicit,
        ),
        "fastvideo": _untracked_source_fingerprint(
            repository=fastvideo_repository,
            roots=(fastvideo_repository / "fastvideo",),
        ),
        "lightx2v": _untracked_source_fingerprint(
            repository=lightx2v_repository,
            roots=(lightx2v_repository / "lightx2v",),
        ),
    }


def _checkpoint_fingerprint(checkpoint: Path) -> dict[str, Any]:
    resolved = checkpoint.expanduser().resolve(strict=True)
    candidates = (
        "config.json",
        "configuration.json",
        "diffusion_pytorch_model.safetensors.index.json",
        "model_index.json",
        ".hfd/repo_info",
        ".hfd/repo_metadata.json",
        "scheduler/scheduler_config.json",
        "text_encoder/model.safetensors.index.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "vae/config.json",
    )
    files: list[dict[str, Any]] = []
    for relative in candidates:
        path = resolved / relative
        if path.is_file():
            files.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    if not files:
        raise FileNotFoundError(f"checkpoint metadata is incomplete: {resolved}")
    tensor_files = sorted(
        {
            str(path.relative_to(resolved)): path.stat().st_size
            for pattern in ("*.safetensors", "*.pth")
            for path in resolved.rglob(pattern)
            if path.is_file()
        }.items()
    )
    identity = {"metadata": files, "tensor_files": tensor_files}
    return {
        "requested_path": str(checkpoint),
        "resolved_path": str(resolved),
        "identity_sha256": _canonical_digest(identity),
        "metadata": files,
        "tensor_files": [{"path": name, "size_bytes": size} for name, size in tensor_files],
        "total_tensor_bytes": sum(size for _, size in tensor_files),
    }


def _normalize_gpu_uuid(value: object, *, source: str) -> str:
    """Return one canonical physical GPU UUID or fail closed."""

    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if len(raw) == 16:
            normalized = f"GPU-{uuidlib.UUID(bytes=raw)}"
        else:
            try:
                normalized = raw.decode("ascii").strip()
            except UnicodeDecodeError as error:
                raise RuntimeError(f"{source} returned a non-text GPU UUID") from error
    else:
        normalized = str(value or "").strip()
    # NVIDIA management APIs use ``GPU-<uuid>`` while recent PyTorch builds
    # expose the same RFC-4122 value without NVIDIA's prefix.  Canonicalize
    # both before comparing physical-device identity.  Keep accepting the
    # prefixed opaque identifiers used by older drivers and contract tests.
    uuid_body = normalized.removeprefix("GPU-")
    try:
        parsed_uuid = uuidlib.UUID(uuid_body)
    except (ValueError, AttributeError):
        parsed_uuid = None
    if parsed_uuid is not None:
        return f"GPU-{parsed_uuid}"
    if not normalized.startswith("GPU-") or not uuid_body:
        raise RuntimeError(
            f"{source} returned an invalid physical GPU UUID: {normalized!r}"
        )
    return normalized


def _torch_cuda_device_uuid(device: int = 0) -> str:
    """Read the UUID of the CUDA device the framework process can execute on."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; runtime GPU affinity cannot be certified")
    properties = torch.cuda.get_device_properties(int(device))
    if not hasattr(properties, "uuid"):
        raise RuntimeError(
            "this PyTorch build does not expose CUDA device UUIDs; runtime GPU "
            "affinity cannot be certified"
        )
    return _normalize_gpu_uuid(
        properties.uuid,
        source="torch.cuda device properties",
    )


def _gpu_query(physical_gpu: int) -> dict[str, Any]:
    fields = (
        "index,uuid,name,memory.total,memory.used,utilization.gpu,"
        "temperature.gpu,power.draw,clocks.current.sm,compute_mode"
    )
    output = _run(
        [
            "nvidia-smi",
            "-i",
            str(physical_gpu),
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=GPU_QUERY_TIMEOUT_SECONDS,
    ).stdout.strip()
    values = [item.strip() for item in output.split(",")]
    if len(values) != 10:
        raise RuntimeError(f"unexpected nvidia-smi row: {output!r}")
    return {
        "index": int(values[0]),
        "uuid": _normalize_gpu_uuid(values[1], source="nvidia-smi"),
        "name": values[2],
        "memory_total_mib": float(values[3]),
        "memory_used_mib": float(values[4]),
        "utilization_percent": float(values[5]),
        "temperature_c": float(values[6]),
        "power_w": float(values[7]),
        "sm_clock_mhz": float(values[8]),
        "compute_mode": values[9],
    }


def _gpu_compute_apps(physical_gpu: int) -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "-i",
        str(physical_gpu),
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = _run(
            command,
            check=False,
            timeout_s=GPU_QUERY_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"nvidia-smi compute-app query timed out after {GPU_QUERY_TIMEOUT_SECONDS:.1f}s"
        ) from error
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(
            "nvidia-smi compute-app query failed with "
            f"exit {completed.returncode}: {stderr or '<no stderr>'}"
        )
    apps_by_pid: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(completed.stdout.splitlines(), start=1):
        if not line.strip():
            continue
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 2 or not parts[0].isdigit():
            raise RuntimeError(
                "malformed nvidia-smi compute-app row "
                f"{line_number}: {line!r}"
            )
        try:
            pid = int(parts[0])
            used_memory_mib = float(parts[1])
        except ValueError as error:
            raise RuntimeError(
                "malformed nvidia-smi compute-app row "
                f"{line_number}: {line!r}"
            ) from error
        if pid <= 0 or not math.isfinite(used_memory_mib) or used_memory_mib < 0:
            raise RuntimeError(
                "invalid nvidia-smi compute-app values in row "
                f"{line_number}: {line!r}"
            )
        apps_by_pid[pid] = {
            "pid": pid,
            "used_memory_mib": used_memory_mib,
        }
    return [apps_by_pid[pid] for pid in sorted(apps_by_pid)]


def _gpu_device_client_pids(
    physical_gpu: int,
    candidate_pids: set[int] | None = None,
) -> set[int]:
    """Return container-visible processes with the target NVIDIA device open."""

    completed = _run(
        [
            "nvidia-smi",
            "-i",
            str(physical_gpu),
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=GPU_QUERY_TIMEOUT_SECONDS,
    )
    try:
        gpu_uuid = completed.stdout.strip()
        if not gpu_uuid.startswith("GPU-"):
            raise RuntimeError(f"unexpected NVIDIA GPU UUID: {gpu_uuid!r}")
        minor = None
        for information in Path("/proc/driver/nvidia/gpus").glob("*/information"):
            fields = {}
            for line in information.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition(":")
                if separator:
                    fields[key.strip()] = value.strip()
            if fields.get("GPU UUID") == gpu_uuid:
                minor = int(fields["Device Minor"])
                break
        if minor is None:
            raise RuntimeError(
                f"cannot resolve device minor for physical GPU {physical_gpu} ({gpu_uuid})"
            )
        device_rdev = os.stat(f"/dev/nvidia{minor}").st_rdev
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"cannot inspect target NVIDIA device for GPU {physical_gpu}: {error}"
        ) from error
    clients: set[int] = set()
    process_dirs = (
        (Path("/proc") / str(pid) for pid in candidate_pids)
        if candidate_pids is not None
        else (path for path in Path("/proc").iterdir() if path.name.isdigit())
    )
    for process_dir in process_dirs:
        try:
            descriptors = (process_dir / "fd").iterdir()
            for descriptor in descriptors:
                try:
                    stat = descriptor.stat()
                except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                    continue
                if stat.st_rdev == device_rdev:
                    clients.add(int(process_dir.name))
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
    return clients


def _generation_isolation_report(
    compute_apps_before: Iterable[Mapping[str, Any]],
    compute_apps_after: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    def normalize(apps: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        by_pid: dict[int, dict[str, Any]] = {}
        for app in apps:
            pid = int(app["pid"])
            by_pid[pid] = {
                "pid": pid,
                "used_memory_mib": float(app.get("used_memory_mib", 0.0)),
            }
        return [by_pid[pid] for pid in sorted(by_pid)]

    before = normalize(compute_apps_before)
    after = normalize(compute_apps_after)
    before_pids = {item["pid"] for item in before}
    after_pids = {item["pid"] for item in after}
    added = [item for item in after if item["pid"] not in before_pids]
    removed = [item for item in before if item["pid"] not in after_pids]
    return {
        "compute_apps_before": before,
        "compute_apps_after": after,
        "added_compute_apps": added,
        "removed_compute_apps": removed,
        "contaminated": bool(added),
        "policy": (
            "framework CUDA contexts must be initialized before generation; "
            "any PID first observed after generation starts is external contamination"
        ),
    }


def _process_parent_pid(pid: int) -> int | None:
    """Read one Linux process parent without adding a psutil dependency."""

    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError):
        return None
    closing_paren = stat.rfind(")")
    if closing_paren < 0:
        return None
    fields = stat[closing_paren + 2 :].split()
    if len(fields) < 2:
        return None
    try:
        return int(fields[1])
    except ValueError:
        return None


def _is_descendant_process(pid: int, root_pids: set[int]) -> bool:
    """Return whether a live process descends from one of ``root_pids``."""

    current = int(pid)
    visited: set[int] = set()
    while current > 1 and current not in visited:
        if current in root_pids:
            return True
        visited.add(current)
        parent = _process_parent_pid(current)
        if parent is None or parent == current:
            return False
        current = parent
    return current in root_pids


def _descendant_process_pids(root_pids: set[int]) -> set[int]:
    """Return the container-visible process tree rooted at ``root_pids``."""

    parents: dict[int, int] = {}
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit():
            continue
        pid = int(process_dir.name)
        parent = _process_parent_pid(pid)
        if parent is not None:
            parents[pid] = parent
    descendants = set(root_pids)
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    return descendants


def _process_identity(pid: int) -> dict[str, Any] | None:
    """Return a PID-reuse-safe local identity and a proven host PID, if exposed.

    Linux exposes the PID hierarchy in ``NSpid`` only when the procfs view
    permits it. A single numeric PID is deliberately *not* treated as proof:
    an unrelated host PID can have the same number as a container PID.
    """

    pid = int(pid)
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None
    closing_paren = stat.rfind(")")
    if closing_paren < 0:
        return None
    fields = stat[closing_paren + 2 :].split()
    if len(fields) <= 19:
        return None
    try:
        starttime_ticks = int(fields[19])
    except ValueError:
        return None
    nspids: list[int] = []
    for line in status.splitlines():
        key, separator, value = line.partition(":")
        if separator and key == "NSpid":
            try:
                nspids = [int(item) for item in value.split()]
            except ValueError:
                return None
            break
    if not nspids or nspids[-1] != pid:
        return None
    host_pid = nspids[0] if len(nspids) > 1 else None
    return {
        "pid": pid,
        "starttime_ticks": starttime_ticks,
        "nspid": nspids,
        "host_pid": host_pid,
        "host_pid_mapping_proven": host_pid is not None,
    }


def _identity_key(identity: Mapping[str, Any]) -> tuple[int, int]:
    return int(identity["pid"]), int(identity["starttime_ticks"])


class ComputeAppMonitor:
    """Audit compute identities from the idle gate until output completion.

    Host NVML PIDs and container ``/proc`` PIDs are never compared as bare
    integers. A host PID is accepted only when ``NSpid`` proves the mapping or
    when the card itself is in exclusive-process compute mode. The latter uses
    a non-reusable temporal binding after an empty-card gate.
    """

    def __init__(
        self,
        physical_gpu: int,
        baseline_apps: Iterable[Mapping[str, Any]],
        *,
        interval_s: float = 0.1,
        allowed_root_pids: Iterable[int] | None = None,
        allow_descendants: bool = False,
        exclusive_compute_mode: bool = False,
        allow_unproven_pid_binding: bool = False,
        monitored_gpu_uuid: str | None = None,
    ) -> None:
        self.physical_gpu = physical_gpu
        self.interval_s = float(interval_s)
        if not 0 < self.interval_s <= 60:
            raise ValueError("GPU isolation polling interval must be in (0, 60]")
        self.baseline_apps = [dict(item) for item in baseline_apps]
        baseline_pids = {int(item["pid"]) for item in self.baseline_apps}
        self._allowed_root_pids = (
            {int(pid) for pid in allowed_root_pids}
            if allowed_root_pids is not None
            else baseline_pids
        )
        self.allow_descendants = bool(allow_descendants)
        self.exclusive_compute_mode = bool(exclusive_compute_mode)
        self.allow_unproven_pid_binding = bool(allow_unproven_pid_binding)
        self.monitored_gpu_uuid = (
            _normalize_gpu_uuid(monitored_gpu_uuid, source="idle gate")
            if monitored_gpu_uuid is not None
            else None
        )
        self._allowed_pids = set(self._allowed_root_pids)
        self._allowed_identities: dict[tuple[int, int], dict[str, Any]] = {}
        self._latest_apps = list(self.baseline_apps)
        self._added: dict[int, dict[str, Any]] = {}
        self._host_pid_bindings: dict[int, tuple[int, int]] = {}
        self._binding_proofs: dict[int, str] = {}
        self._retired_host_pids: set[int] = set()
        self._last_app_pids: set[int] = baseline_pids
        self._local_gpu_client_pids: set[int] = set()
        self._errors: list[str] = []
        self._samples = 0
        self._sample_timestamps: list[float] = []
        self._query_latencies: list[float] = []
        self._window_started_monotonic: float | None = None
        self._window_stopped_monotonic: float | None = None
        self._window_started_at: str | None = None
        self._window_stopped_at: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _append_error_locked(self, message: str) -> None:
        if message not in self._errors:
            self._errors.append(message)

    def _sample(self) -> None:
        if self._window_started_monotonic is None:
            self._window_started_monotonic = time.monotonic()
            self._window_started_at = _utc_now()
        sample_started = time.monotonic()
        try:
            apps = _gpu_compute_apps(self.physical_gpu)
            app_pids = {int(app["pid"]) for app in apps}
            allowed_process_tree = (
                _descendant_process_pids(self._allowed_root_pids)
                if self.allow_descendants
                else set(self._allowed_root_pids)
            )
            local_gpu_client_pids = _gpu_device_client_pids(
                self.physical_gpu,
                allowed_process_tree,
            )
            identities: dict[tuple[int, int], dict[str, Any]] = {}
            local_client_identities: dict[tuple[int, int], dict[str, Any]] = {}
            for pid in sorted(allowed_process_tree):
                identity = _process_identity(pid)
                if identity is None:
                    continue
                key = _identity_key(identity)
                identities[key] = identity
                if pid in local_gpu_client_pids:
                    local_client_identities[key] = identity
            exact_host_mapping = {
                int(identity["host_pid"]): (key, identity)
                for key, identity in local_client_identities.items()
                if identity.get("host_pid_mapping_proven")
            }
            with self._lock:
                self._samples += 1
                self._latest_apps = [dict(item) for item in apps]
                self._allowed_pids.update(allowed_process_tree)
                self._allowed_identities.update(identities)
                self._local_gpu_client_pids = set(local_gpu_client_pids)

                disappeared = (
                    self._last_app_pids & set(self._host_pid_bindings)
                ) - app_pids
                for pid in sorted(disappeared):
                    self._retired_host_pids.add(pid)
                    self._append_error_locked(
                        f"bound host compute PID {pid} disappeared before the audit window ended"
                    )
                for pid in sorted(app_pids & self._retired_host_pids):
                    self._append_error_locked(
                        f"retired host compute PID {pid} reappeared during the audit window"
                    )

                baseline_pids = {int(item["pid"]) for item in self.baseline_apps}
                for app in apps:
                    pid = int(app["pid"])
                    if pid in baseline_pids:
                        self._added.setdefault(pid, dict(app))
                        continue
                    exact = exact_host_mapping.get(pid)
                    if exact is not None:
                        key, _identity = exact
                        previous = self._host_pid_bindings.get(pid)
                        if previous is not None and previous != key:
                            self._append_error_locked(
                                f"host compute PID {pid} changed local process identity"
                            )
                            self._added.setdefault(pid, dict(app))
                            continue
                        self._host_pid_bindings[pid] = key
                        self._binding_proofs[pid] = "linux-nspid"

                unbound_apps = [
                    app
                    for app in apps
                    if int(app["pid"]) not in self._host_pid_bindings
                    and int(app["pid"]) not in baseline_pids
                ]
                if unbound_apps and (
                    self.exclusive_compute_mode or self.allow_unproven_pid_binding
                ):
                    assigned_identities = set(self._host_pid_bindings.values())
                    available_identities = [
                        key
                        for key in sorted(local_client_identities)
                        if key not in assigned_identities
                    ]
                    bind_count = min(len(unbound_apps), len(available_identities))
                    for app, key in zip(
                        sorted(
                            unbound_apps,
                            key=lambda item: int(item["pid"]),
                        )[:bind_count],
                        available_identities[:bind_count],
                        strict=True,
                    ):
                        pid = int(app["pid"])
                        self._host_pid_bindings[pid] = key
                        self._binding_proofs[pid] = (
                            "exclusive-process-temporal-binding"
                            if self.exclusive_compute_mode
                            else "unproven-temporal-binding"
                        )
                    unbound_apps = unbound_apps[bind_count:]
                for app in unbound_apps:
                    self._added.setdefault(int(app["pid"]), dict(app))
                self._last_app_pids = app_pids
        except Exception as error:
            with self._lock:
                self._append_error_locked(f"{type(error).__name__}: {error}")
        finally:
            sample_finished = time.monotonic()
            with self._lock:
                self._sample_timestamps.append(sample_started)
                self._query_latencies.append(sample_finished - sample_started)

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU isolation monitor has already been started")
        self._window_started_monotonic = time.monotonic()
        self._window_started_at = _utc_now()
        self._sample()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name=f"gpu-isolation-{self.physical_gpu}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(
                timeout=max(
                    2.0,
                    GPU_QUERY_TIMEOUT_SECONDS + self.interval_s + 1.0,
                )
            )
        thread_stopped = self._thread is None or not self._thread.is_alive()
        if thread_stopped:
            self._sample()
        self._window_stopped_monotonic = time.monotonic()
        self._window_stopped_at = _utc_now()
        with self._lock:
            latest_apps = [dict(item) for item in self._latest_apps]
            added = [dict(self._added[pid]) for pid in sorted(self._added)]
            errors = list(self._errors)
            samples = int(self._samples)
            timestamps = list(self._sample_timestamps)
            query_latencies = list(self._query_latencies)
            allowed_pids = sorted(self._allowed_pids)
            allowed_identities = [
                dict(self._allowed_identities[key])
                for key in sorted(self._allowed_identities)
            ]
            local_gpu_client_pids = sorted(self._local_gpu_client_pids)
            bindings = [
                {
                    "host_pid": host_pid,
                    "local_pid": key[0],
                    "local_starttime_ticks": key[1],
                    "proof": self._binding_proofs[host_pid],
                }
                for host_pid, key in sorted(self._host_pid_bindings.items())
            ]
        if not thread_stopped:
            errors.append("monitor thread did not stop before the audit report was frozen")
        points = [
            float(self._window_started_monotonic),
            *timestamps,
            float(self._window_stopped_monotonic),
        ]
        max_gap = max(
            (later - earlier for earlier, later in zip(points, points[1:])),
            default=0.0,
        )
        final_report = _generation_isolation_report(
            self.baseline_apps,
            latest_apps,
        )
        exact_bindings = all(item["proof"] == "linux-nspid" for item in bindings)
        identity_proof = (
            "exclusive-process-compute-mode"
            if self.exclusive_compute_mode
            else "linux-nspid"
            if exact_bindings
            else "unproven"
        )
        bounded_polling_complete = (
            samples >= 2
            and thread_stopped
            and max_gap
            <= self.interval_s
            + (max(query_latencies) if query_latencies else 0.0)
            + 0.05
        )
        certifying = (
            self.exclusive_compute_mode
            and self.interval_s <= MAX_CERTIFYING_POLL_INTERVAL_SECONDS
            and bounded_polling_complete
            and not self.baseline_apps
            and not added
            and not errors
        )
        return {
            **final_report,
            "audit_version": ISOLATION_AUDIT_VERSION,
            "window_policy": "post-idle-gate-through-completed-output",
            "window_started_at": self._window_started_at,
            "window_stopped_at": self._window_stopped_at,
            "window_started_monotonic_seconds": self._window_started_monotonic,
            "window_stopped_monotonic_seconds": self._window_stopped_monotonic,
            "allowed_root_pids": sorted(self._allowed_root_pids),
            "monitored_gpu_uuid": self.monitored_gpu_uuid,
            "allowed_compute_pids": allowed_pids,
            "allowed_process_identities": allowed_identities,
            "bound_host_compute_pids": [item["host_pid"] for item in bindings],
            "host_pid_bindings": bindings,
            "local_gpu_client_pids": local_gpu_client_pids,
            "allow_descendants": self.allow_descendants,
            "exclusive_compute_mode": self.exclusive_compute_mode,
            "allow_unproven_pid_binding": self.allow_unproven_pid_binding,
            "identity_proof": identity_proof,
            "namespace_mapping_proven": exact_bindings,
            "added_compute_apps": added,
            "monitor_errors": errors,
            "samples": samples,
            "poll_interval_seconds": self.interval_s,
            "sample_timestamps_monotonic_seconds": timestamps,
            "max_sample_gap_seconds": max_gap,
            "query_latency_seconds": {
                "raw": query_latencies,
                "max": max(query_latencies, default=0.0),
                "median": statistics.median(query_latencies) if query_latencies else 0.0,
            },
            "thread_stopped": thread_stopped,
            "bounded_polling_complete": bounded_polling_complete,
            "certifying": certifying,
            "contaminated": bool(added or errors),
            "policy": (
                "after the pre-load idle gate, only the benchmark process and "
                "its live descendants are allowed; compute PIDs are sampled "
                "through model loading and completed output, and any other PID "
                "or monitor error invalidates the sample; host/container PID "
                "identity requires Linux NSpid evidence or an empty-card gate "
                "plus NVIDIA exclusive-process compute mode; temporal bindings "
                "are never reused"
            ),
        }


def _raise_for_generation_contamination(
    measurements: Mapping[str, Any],
    *,
    enabled: bool,
) -> None:
    report = measurements.get("generation_isolation")
    if not isinstance(report, Mapping):
        if enabled:
            raise RuntimeError("generation isolation audit is missing")
        return
    if enabled and (
        bool(report.get("contaminated")) or not bool(report.get("certifying"))
    ):
        raise RuntimeError(
            "GPU isolation certification failed: "
            f"added_compute_apps={json.dumps(report.get('added_compute_apps') or [], sort_keys=True)}, "
            f"monitor_errors={json.dumps(report.get('monitor_errors') or [], sort_keys=True)}, "
            f"identity_proof={report.get('identity_proof')!r}, "
            f"certifying={report.get('certifying')!r}"
        )


def _require_idle_gpu(physical_gpu: int, *, samples: int, interval_s: float) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for index in range(samples):
        snapshot = _gpu_query(physical_gpu)
        snapshot["compute_apps"] = _gpu_compute_apps(physical_gpu)
        observations.append(snapshot)
        if index + 1 < samples:
            time.sleep(interval_s)
    busy = [
        item
        for item in observations
        if item["compute_apps"] or item["utilization_percent"] > 5 or item["memory_used_mib"] > 512
    ]
    if busy:
        raise RuntimeError(
            f"GPU {physical_gpu} is not isolated; refusing benchmark: "
            f"{json.dumps(busy[-1], ensure_ascii=False, sort_keys=True)}"
        )
    return observations


def _cuda_visibility_evidence(
    physical_gpu: int,
    gpu_snapshot: Mapping[str, Any],
    *,
    current_cuda_uuid: str | None = None,
) -> dict[str, Any]:
    """Prove that framework-local CUDA device 0 is the monitored card.

    Environment visibility is only the requested mapping.  Certification also
    reads the UUID from the live CUDA runtime after the isolation monitor has
    started, so a stale or remapped ``CUDA_VISIBLE_DEVICES`` token cannot be
    accepted on its own.
    """

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    tokens = [item.strip() for item in visible.split(",") if item.strip()] if visible else []
    monitored_uuid = _normalize_gpu_uuid(
        gpu_snapshot.get("uuid"),
        source="idle-gate nvidia-smi",
    )
    if not tokens:
        visible_token_matches = int(physical_gpu) == 0
        selected = "0 (implicit)"
    elif len(tokens) == 1:
        selected = tokens[0]
        if selected.isdigit():
            visible_token_matches = int(selected) == int(physical_gpu)
        else:
            visible_token_matches = (
                selected.casefold() == monitored_uuid.casefold()
                or monitored_uuid.casefold().startswith(selected.casefold())
            )
    else:
        selected = tokens[0]
        visible_token_matches = False
    runtime_uuid = _normalize_gpu_uuid(
        current_cuda_uuid
        if current_cuda_uuid is not None
        else _torch_cuda_device_uuid(0),
        source="current CUDA device",
    )
    runtime_uuid_matches = runtime_uuid.casefold() == monitored_uuid.casefold()
    checks = {
        "visible_token_selects_monitored_gpu": bool(visible_token_matches),
        "current_cuda_uuid_matches_monitored_gpu": bool(runtime_uuid_matches),
    }
    evidence = {
        "cuda_visible_devices": visible,
        "framework_local_device": 0,
        "current_cuda_device": 0,
        "selected_visible_device": selected,
        "monitored_physical_gpu": int(physical_gpu),
        "monitored_gpu_uuid": monitored_uuid,
        "current_cuda_uuid": runtime_uuid,
        "checks": checks,
        "single_device_mapping_proven": all(checks.values()),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES and the live CUDA UUID do not prove that "
            f"framework-local device 0 is physical GPU {physical_gpu}: "
            f"{json.dumps(evidence, sort_keys=True)}"
        )
    return evidence


class GpuSampler:
    def __init__(self, physical_gpu: int, interval_s: float = 0.1) -> None:
        self.physical_gpu = physical_gpu
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(_gpu_query(self.physical_gpu))
            except (OSError, RuntimeError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        if not self.samples:
            self.samples.append(_gpu_query(self.physical_gpu))

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {}
        return {
            "sample_count": len(self.samples),
            "peak_memory_used_mib": max(item["memory_used_mib"] for item in self.samples),
            "peak_power_w": max(item["power_w"] for item in self.samples),
            "peak_temperature_c": max(item["temperature_c"] for item in self.samples),
            "median_sm_clock_mhz": statistics.median(item["sm_clock_mhz"] for item in self.samples),
            "median_utilization_percent": statistics.median(item["utilization_percent"] for item in self.samples),
        }


_FASTVIDEO_REFERENCE_PROBE_ATTRIBUTE = "_worldfoundry_fastvideo_reference_probe_v1"
_LIGHTX2V_REFERENCE_PROBE_ATTRIBUTE = "_worldfoundry_lightx2v_reference_probe_v1"
_FASTVIDEO_NON_ACCELERATED_BACKENDS = {"NO_ATTENTION", "TORCH_SDPA"}
_FASTVIDEO_SLA_BACKEND_CONTRACTS = {
    str(contract["backend"]): {"kind": kind, **contract}
    for kind, contract in FASTVIDEO_SLA_REFERENCE_CONTRACTS.items()
}
_LIGHTX2V_ATTENTION_KERNEL_CONTRACTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "flash_attn2": (
        "lightx2v.common.ops.attn.flash_attn.FlashAttn2Weight",
        ("flash_attn_func_v2", "flash_attn_varlen_func_v2"),
    ),
    "flash_attn3": (
        "lightx2v.common.ops.attn.flash_attn.FlashAttn3Weight",
        ("flash_attn_func_v3", "flash_attn_varlen_func_v3"),
    ),
    "flash_attn4": (
        "lightx2v.common.ops.attn.flash_attn.FlashAttn4Weight",
        ("flash_attn_func_v4",),
    ),
    "sage_attn2": (
        "lightx2v.common.ops.attn.sage_attn.SageAttn2Weight",
        ("sageattn",),
    ),
    "sage_attn3": (
        "lightx2v.common.ops.attn.sage_attn.SageAttn3Weight",
        ("sageattn3_blackwell",),
    ),
}
_LIGHTX2V_FAIL_CLOSED_ATTENTION_MODES = {
    "sage_attn2_k_int8_v_fp8": (
        "expects pre-quantized (k_int8, k_scale) and (v_fp8, v_scale) tuples, "
        "but the pinned Wan TI2V generator passes floating-point K/V tensors "
        "and has no audited per-layer quantize/cache/invalidation lifecycle"
    ),
}
_LIGHTX2V_DYNAMIC_SPARSE_PROVIDER_PATHS = {
    "triton": (
        "lightx2v.common.ops.attn.dynamic_sparse_attn."
        "DynamicSparseAttnWeight.apply_triton"
    ),
    "sage2": (
        "lightx2v.common.ops.attn.dynamic_sparse_attn."
        "DynamicSparseAttnWeight.apply_sage2"
    ),
    "sage3": (
        "lightx2v.common.ops.attn.dynamic_sparse_attn."
        "DynamicSparseAttnWeight.apply_sage3"
    ),
    "fa4": (
        "lightx2v.common.ops.attn.dynamic_sparse_attn."
        "DynamicSparseAttnWeight.apply_fa4"
    ),
    "magi": (
        "lightx2v.common.ops.attn.dynamic_sparse_attn."
        "DynamicSparseAttnWeight.apply_magi"
    ),
}
_LIGHTX2V_DYNAMIC_SPARSE_KERNEL_ATTRIBUTES = {
    "triton": "_attention.apply",
    "sage2": "sage2_block_sparse_attn",
    "sage3": "sage3_block_sparse_attn",
    "fa4": "flash_attn_func_v4",
    "magi": "magi_ffa_func",
}
_LIGHTX2V_SAGE2_SPARSE_KERNEL_ATTRIBUTES_BY_ARCH = {
    "sm80": (
        "qattn."
        "qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold"
    ),
    "sm86": (
        "qattn."
        "qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold"
    ),
    "sm87": (
        "qattn."
        "qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold"
    ),
    "sm90": (
        "qattn."
        "qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_sm90"
    ),
}
_LIGHTX2V_SAGE2_SPARSE_DEFAULT_FP16_KERNEL_ATTRIBUTE = (
    "qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_"
    "with_pv_threshold"
)
_LIGHTX2V_SAGE2_SPARSE_DEFAULT_FP32_KERNEL_ATTRIBUTE = (
    "qattn."
    "qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_"
    "with_pv_threshold"
)
_LIGHTX2V_GENERAL_SPARSE_OPERATOR_PATHS = {
    "sla_triton_operator": (
        "lightx2v.common.ops.attn.sparse_operator.SlaTritonOperator"
    ),
    "spas_sage2_operator": (
        "lightx2v.common.ops.attn.sparse_operator."
        "SparseSageAttentionV2Operator"
    ),
    "spas_sage3_operator": (
        "lightx2v.common.ops.attn.sparse_operator."
        "SparseSageAttentionV3Operator"
    ),
    "spas_fa4_operator": (
        "lightx2v.common.ops.attn.sparse_operator."
        "SparseFlashAttentionV4Operator"
    ),
    "magi_operator": "lightx2v.common.ops.attn.sparse_operator.MagiOperator",
    "flex_block_operator": (
        "lightx2v.common.ops.attn.sparse_operator.FlexBlockOperator"
    ),
    "flashinfer_operator": (
        "lightx2v.common.ops.attn.sparse_operator.FlashinferOperator"
    ),
}
_LIGHTX2V_GENERAL_SPARSE_MASK_PATHS = {
    "sla_mask_generator": (
        "lightx2v.common.ops.attn.sparse_mask_generator.SlaMaskGenerator"
    ),
    "sparge_mask_generator": (
        "lightx2v.common.ops.attn.sparse_mask_generator.SpargeMaskGenerator"
    ),
    "nbhd_mask_generator": (
        "lightx2v.common.ops.attn.sparse_mask_generator.NbhdMaskGenerator"
    ),
    "svg_mask_generator": (
        "lightx2v.common.ops.attn.sparse_mask_generator.SvgMaskGenerator"
    ),
}
_LIGHTX2V_SPARSE_ATTENTION_CONTRACTS: dict[str, dict[str, Any]] = {
    "dynamic_sparse_attn": {
        "implementation_class": (
            "lightx2v.common.ops.attn.dynamic_sparse_attn."
            "DynamicSparseAttnWeight"
        ),
        "setting_key": "dynamic_sparse_attn_setting",
        "required_setting_fields": frozenset({"sparsity_ratio", "operator"}),
        "allowed_setting_fields": frozenset(
            {"sparsity_ratio", "operator", "per_block_mean"}
        ),
        "provider_kind": "dynamic-apply-func",
    },
    "general_sparse_attn": {
        "implementation_class": (
            "lightx2v.common.ops.attn.general_sparse_attn."
            "GeneralSparseAttnWeight"
        ),
        "setting_key": "general_sparse_attn_setting",
        "required_setting_fields": frozenset(
            {"sparse_mask_generator", "sparse_operator", "sparse_setting"}
        ),
        "allowed_setting_fields": frozenset(
            {
                "sparse_mask_generator",
                "sparse_operator",
                "sparse_setting",
                "operator_setting",
            }
        ),
        "provider_kind": "general-operator",
    },
}
_BACKEND_CAPABILITY_CONTRACTS: dict[
    str,
    dict[str, dict[str, Any]],
] = {
    "worldfoundry": {
        "torch": {"requirements": ()},
        "auto": {"requirements": (), "runtime_resolved": True},
        "flash": {"requirements": (), "runtime_resolved": True},
        "flash2": {
            "requirements": (
                ("flash_attn", ("flash_attn_func",)),
                (
                    "flash_attn.flash_attn_interface",
                    ("flash_attn_varlen_func",),
                ),
            ),
            "minimum_cuda_major": 8,
        },
        "flash3": {
            "requirements": (
                (
                    "flash_attn_interface",
                    ("flash_attn_func", "flash_attn_varlen_func"),
                ),
            ),
            "accepted_cuda_capabilities": ((9, 0),),
        },
        "flash4": {
            "requirements": (("flash_attn.cute", ("flash_attn_func",)),),
            "minimum_cuda_major": 10,
        },
        "sage": {
            "requirements": (("sageattention", ("sageattn",)),),
        },
        "sage3": {
            "requirements": (("sageattn3", ("sageattn3_blackwell",)),),
            "minimum_cuda_major": 10,
        },
        "xformers": {
            "requirements": (
                ("xformers.ops", ("memory_efficient_attention",)),
            ),
        },
    },
    "fastvideo": {
        "flash_attn": {
            "requirements": (
                ("flash_attn", ("flash_attn_func",)),
                (
                    "flash_attn.flash_attn_interface",
                    ("flash_attn_varlen_func",),
                ),
            ),
            "minimum_cuda_major": 8,
        },
        "sage_attn": {
            "requirements": (("sageattention", ("sageattn",)),),
        },
        "sage_attn_three": {
            "requirements": (("sageattn3", ("sageattn3_blackwell",)),),
            "minimum_cuda_major": 10,
            "minimum_python": (3, 13),
        },
        "sla_attn": {
            "requirements": (
                (
                    "fastvideo_kernel.triton_kernels.sla_triton",
                    ("_attention.apply",),
                ),
            ),
        },
        "sage_sla_attn": {
            "requirements": (
                (
                    "fastvideo_kernel.triton_kernels.sla_triton",
                    ("_attention.apply",),
                ),
                (
                    "spas_sage_attn._fused",
                    ("transpose_pad_permute_cuda", "scale_fuse_quant_cuda"),
                ),
                (
                    "spas_sage_attn.utils",
                    ("get_vanilla_qk_quant", "block_map_lut_triton"),
                ),
            ),
            "requirements_by_cuda_capability": {
                (9, 0): (
                    (
                        "spas_sage_attn._qattn",
                        (
                            "qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_sm90",
                        ),
                    ),
                ),
            },
            # FastVideo checks the f16 entry first and otherwise executes the
            # f32 entry on every non-sm90 architecture.
            "default_cuda_one_of_requirements": (
                (
                    (
                        "spas_sage_attn._qattn",
                        (
                            "qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold",
                        ),
                    ),
                    (
                        "spas_sage_attn._qattn",
                        (
                            "qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold",
                        ),
                    ),
                ),
            ),
        },
    },
    "lightx2v": {
        "flash_attn2": {
            "requirements": (
                ("flash_attn", ("flash_attn_func",)),
                (
                    "flash_attn.flash_attn_interface",
                    ("flash_attn_varlen_func",),
                ),
            ),
            "minimum_cuda_major": 8,
        },
        "flash_attn3": {
            "requirements": (
                (
                    "flash_attn_interface",
                    ("flash_attn_func", "flash_attn_varlen_func"),
                ),
            ),
            # The external flash_attn_interface package is the upstream
            # Hopper-only FA3 implementation.  A broad major>=9 check would
            # incorrectly certify Blackwell sm100/sm120.
            "accepted_cuda_capabilities": ((9, 0),),
        },
        "flash_attn4": {
            "requirements": (("flash_attn.cute", ("flash_attn_func",)),),
            "minimum_cuda_major": 10,
        },
        "sage_attn2": {
            "requirements": (("sageattention", ()),),
            "minimum_cuda_major": 8,
            # LightX2V binds a different SageAttention2 entry point on Ada
            # sm89 and consumer Blackwell sm120 at module-import time.
            "requirements_by_cuda_capability": {
                (8, 9): (
                    (
                        "sageattention",
                        ("sageattn_qk_int8_pv_fp16_triton",),
                    ),
                ),
                (12, 0): (
                    (
                        "sageattention",
                        ("sageattn_qk_int8_pv_fp16_triton",),
                    ),
                ),
            },
            "default_cuda_requirements": (
                ("sageattention", ("sageattn",)),
            ),
        },
        "sage_attn3": {
            "requirements": (("sageattn3", ("sageattn3_blackwell",)),),
            "minimum_cuda_major": 10,
        },
        "dynamic_sparse_attn": {
            "requirements": (),
            "minimum_cuda_major": 8,
            "runtime_resolved": True,
        },
        "general_sparse_attn": {
            "requirements": (),
            "minimum_cuda_major": 8,
            "runtime_resolved": True,
        },
    },
}


class BackendCapabilityError(RuntimeError):
    """A requested backend is unavailable before framework model loading."""

    def __init__(self, receipt: Mapping[str, Any]) -> None:
        self.receipt = _jsonable(receipt)
        super().__init__(
            "backend capability gate failed: "
            + json.dumps(self.receipt, ensure_ascii=False, sort_keys=True)
        )


def _runtime_type_identity(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _runtime_callable_identity(value: Any) -> str | None:
    if value is None:
        return None
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return _runtime_type_identity(value)


def _resolve_dotted_attribute(value: Any, dotted_name: str) -> Any:
    resolved = value
    for component in str(dotted_name).split("."):
        resolved = getattr(resolved, component)
    return resolved


def _probe_backend_requirement(
    module_name: str,
    required_symbols: Iterable[str],
) -> tuple[dict[str, Any], list[str]]:
    """Import one dependency and resolve its exact callable entry points."""

    symbols = tuple(str(symbol) for symbol in required_symbols)
    issues: list[str] = []
    module_report: dict[str, Any] = {
        "module": str(module_name),
        "required_symbols": list(symbols),
        "imported": False,
        "module_file": None,
        "module_version": None,
        "symbols": [],
        "error": None,
    }
    try:
        module = importlib.import_module(str(module_name))
    except BaseException as error:
        module_report["error"] = f"{type(error).__name__}: {error}"
        issues.append(
            f"cannot import {module_name}: {type(error).__name__}: {error}"
        )
        return module_report, issues

    module_report.update(
        imported=True,
        module_file=(
            str(Path(module.__file__).resolve())
            if isinstance(getattr(module, "__file__", None), str)
            else None
        ),
        module_version=getattr(module, "__version__", None),
    )
    for symbol_name in symbols:
        try:
            symbol = _resolve_dotted_attribute(module, symbol_name)
        except AttributeError:
            symbol = None
        available = callable(symbol)
        module_report["symbols"].append(
            {
                "name": symbol_name,
                "available": available,
                "callable_identity": _runtime_callable_identity(symbol),
            }
        )
        if not available:
            issues.append(
                f"{module_name}.{symbol_name} is missing or not callable"
            )
    return module_report, issues


def _backend_capability_receipt(
    framework: str,
    requested_backend: str,
    *,
    cuda_capability: tuple[int, int] | None,
) -> dict[str, Any]:
    """Probe exact external modules/symbols without treating them as runtime proof."""

    normalized_framework = str(framework).strip().casefold()
    normalized_backend = str(requested_backend).strip().casefold()
    contracts = _BACKEND_CAPABILITY_CONTRACTS.get(normalized_framework, {})
    contract = contracts.get(normalized_backend)
    framework_modules_before_probe = sorted(
        name
        for name in sys.modules
        if name == normalized_framework
        or name.startswith(f"{normalized_framework}.")
    )
    issues: list[str] = []
    requirement_reports: list[dict[str, Any]] = []
    if contract is None:
        issues.append(
            f"no capability contract for {normalized_framework} backend "
            f"{normalized_backend!r}"
        )
        contract = {"requirements": ()}

    minimum_python_value = contract.get("minimum_python")
    minimum_python = (
        tuple(int(value) for value in minimum_python_value)
        if isinstance(minimum_python_value, (list, tuple))
        else None
    )
    python_version = (sys.version_info.major, sys.version_info.minor)
    python_supported = minimum_python is None or python_version >= minimum_python
    if not python_supported:
        issues.append(
            f"Python {python_version} is below required {minimum_python}"
        )

    normalized_cuda_capability = (
        tuple(int(value) for value in cuda_capability)
        if isinstance(cuda_capability, tuple) and len(cuda_capability) == 2
        else None
    )
    minimum_cuda_major_value = contract.get("minimum_cuda_major")
    minimum_cuda_major = (
        int(minimum_cuda_major_value)
        if isinstance(minimum_cuda_major_value, int)
        and not isinstance(minimum_cuda_major_value, bool)
        else None
    )
    minimum_cuda_supported = (
        minimum_cuda_major is None
        or (
            normalized_cuda_capability is not None
            and normalized_cuda_capability[0] >= minimum_cuda_major
        )
    )
    if not minimum_cuda_supported:
        issues.append(
            f"CUDA capability {cuda_capability!r} is below required "
            f"major {minimum_cuda_major}"
        )

    accepted_cuda_capabilities_value = contract.get(
        "accepted_cuda_capabilities"
    )
    accepted_cuda_capabilities = (
        tuple(
            tuple(int(component) for component in capability)
            for capability in accepted_cuda_capabilities_value
        )
        if isinstance(accepted_cuda_capabilities_value, (list, tuple))
        else ()
    )
    accepted_cuda_supported = (
        not accepted_cuda_capabilities
        or normalized_cuda_capability in accepted_cuda_capabilities
    )
    if not accepted_cuda_supported:
        issues.append(
            f"CUDA capability {cuda_capability!r} is not one of the accepted "
            f"capabilities {accepted_cuda_capabilities!r}"
        )
    cuda_supported = minimum_cuda_supported and accepted_cuda_supported

    requirements = list(contract.get("requirements", ()))
    requirements_by_capability = contract.get(
        "requirements_by_cuda_capability"
    )
    selected_cuda_requirement_branch = "base-only"
    if isinstance(requirements_by_capability, Mapping):
        branch = requirements_by_capability.get(normalized_cuda_capability)
        if branch is not None:
            requirements.extend(branch)
            selected_cuda_requirement_branch = (
                f"cuda-{normalized_cuda_capability[0]}.{normalized_cuda_capability[1]}"
            )
        else:
            requirements.extend(contract.get("default_cuda_requirements", ()))
            selected_cuda_requirement_branch = "cuda-default"

    for module_name, required_symbols in requirements:
        module_report, requirement_issues = _probe_backend_requirement(
            str(module_name),
            required_symbols,
        )
        requirement_reports.append(module_report)
        issues.extend(requirement_issues)

    one_of_groups = list(contract.get("one_of_requirements", ()))
    one_of_by_capability = contract.get(
        "one_of_requirements_by_cuda_capability"
    )
    if isinstance(one_of_by_capability, Mapping):
        branch = one_of_by_capability.get(normalized_cuda_capability)
        if branch is not None:
            one_of_groups.extend(branch)
            selected_cuda_requirement_branch += "+one-of-exact"
        else:
            one_of_groups.extend(
                contract.get("default_cuda_one_of_requirements", ())
            )
            selected_cuda_requirement_branch += "+one-of-default"
    elif isinstance(requirements_by_capability, Mapping):
        if normalized_cuda_capability not in requirements_by_capability:
            one_of_groups.extend(
                contract.get("default_cuda_one_of_requirements", ())
            )

    one_of_reports: list[dict[str, Any]] = []
    for group_index, candidates in enumerate(one_of_groups):
        candidate_reports = []
        selected_candidate = None
        candidate_issue_sets: list[list[str]] = []
        for candidate_index, (module_name, required_symbols) in enumerate(candidates):
            candidate_report, candidate_issues = _probe_backend_requirement(
                str(module_name),
                required_symbols,
            )
            candidate_report["passed"] = not candidate_issues
            candidate_reports.append(candidate_report)
            candidate_issue_sets.append(candidate_issues)
            if selected_candidate is None and not candidate_issues:
                selected_candidate = candidate_index
        group_report = {
            "group_index": group_index,
            "policy": "first-callable-candidate",
            "candidates": candidate_reports,
            "selected_candidate": selected_candidate,
            "passed": selected_candidate is not None,
        }
        one_of_reports.append(group_report)
        if selected_candidate is None:
            issues.append(
                f"one-of capability group {group_index} has no usable candidate: "
                + " | ".join(
                    "; ".join(candidate_issues)
                    for candidate_issues in candidate_issue_sets
                )
            )

    return {
        "schema_version": "worldfoundry-backend-capability-v1",
        "certification_role": "diagnostic-only",
        "framework": normalized_framework,
        "requested_backend": normalized_backend,
        "checked_before_model_load": True,
        "framework_modules_before_probe": framework_modules_before_probe,
        "runtime_resolved_request": bool(contract.get("runtime_resolved")),
        "python_version": list(python_version),
        "minimum_python": list(minimum_python) if minimum_python else None,
        "python_supported": python_supported,
        "cuda_capability": (
            list(normalized_cuda_capability)
            if normalized_cuda_capability is not None
            else None
        ),
        "minimum_cuda_major": minimum_cuda_major,
        "accepted_cuda_capabilities": [
            list(capability) for capability in accepted_cuda_capabilities
        ],
        "cuda_supported": cuda_supported,
        "selected_cuda_requirement_branch": selected_cuda_requirement_branch,
        "requirements": requirement_reports,
        "one_of_requirements": one_of_reports,
        "issues": issues,
        "passed": not issues,
    }


def _require_backend_capability(
    framework: str,
    requested_backend: str,
    *,
    cuda_capability: tuple[int, int] | None,
) -> dict[str, Any]:
    receipt = _backend_capability_receipt(
        framework,
        requested_backend,
        cuda_capability=cuda_capability,
    )
    if receipt["passed"] is not True:
        raise BackendCapabilityError(receipt)
    return receipt


def _runtime_backend_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value).strip().upper()


def _fastvideo_projection_fingerprint(
    layer_index: int,
    weight: Any,
    bias: Any,
) -> str:
    """Match the core adapter's canonical per-layer FP32 tensor digest."""

    import torch

    digest = hashlib.sha256()
    for name, tensor in (("weight", weight), ("bias", bias)):
        normalized = tensor.detach().to(device="cpu").contiguous()
        digest.update(f"{layer_index}:{name}".encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(normalized.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(normalized.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(normalized.view(torch.uint8).numpy().tobytes())
        digest.update(b"\n")
    return digest.hexdigest()


def _fastvideo_projection_set_fingerprint(
    layer_fingerprints: Mapping[str, str],
    *,
    checkpoint_layout: str | None,
) -> str:
    """Digest one complete ordered projection set, optionally binding layout."""

    digest = hashlib.sha256()
    if checkpoint_layout is not None:
        digest.update(checkpoint_layout.encode("ascii"))
        digest.update(b"\n")
    for raw_layer, fingerprint in sorted(
        layer_fingerprints.items(),
        key=lambda item: int(item[0]),
    ):
        digest.update(str(int(raw_layer)).encode("ascii"))
        digest.update(b":")
        digest.update(str(fingerprint).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _fastvideo_sla_projection_layer(
    implementation: Any,
    *,
    layer_index: int,
) -> tuple[dict[str, Any], Any | None, list[str]]:
    """Return tensor-free learned-projection evidence for one self-attention."""

    import torch

    issues: list[str] = []
    projection = getattr(implementation, "proj_l", None)
    weight = getattr(projection, "weight", None)
    bias = getattr(projection, "bias", None)
    source_prefix = f"blocks.{layer_index}.attn1.attn_impl.proj_l"
    report: dict[str, Any] = {
        "projection_class": (
            _runtime_type_identity(projection) if projection is not None else None
        ),
        "projection_source_keys": [
            f"{source_prefix}.weight",
            f"{source_prefix}.bias",
        ],
        "projection_weight_dtype": (
            str(weight.dtype) if isinstance(weight, torch.Tensor) else None
        ),
        "projection_bias_dtype": (
            str(bias.dtype) if isinstance(bias, torch.Tensor) else None
        ),
        "projection_weight_shape": (
            list(weight.shape) if isinstance(weight, torch.Tensor) else None
        ),
        "projection_bias_shape": (
            list(bias.shape) if isinstance(bias, torch.Tensor) else None
        ),
        "projection_weight_fingerprint": None,
        "projection_fp32": False,
        "projection_finite": False,
        "projection_nonzero": False,
        "projection_forward_calls": 0,
        "projection_hook_removed_after_success": False,
        "projection_cleanup_removed_uncalled_hook": False,
    }
    if not isinstance(projection, torch.nn.Linear):
        issues.append("proj_l is missing or is not torch.nn.Linear")
        return report, None, issues
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        issues.append("proj_l weight/bias tensors are missing")
        return report, None, issues
    if weight.device.type == "meta" or bias.device.type == "meta":
        issues.append("proj_l weight/bias remain on the meta device")
        return report, None, issues
    if weight.ndim != 2 or bias.ndim != 1:
        issues.append(
            "proj_l geometry is malformed: "
            f"weight={tuple(weight.shape)}, bias={tuple(bias.shape)}"
        )
        return report, projection, issues
    expected_weight_shape = (
        WAN22_TI2V_ATTENTION_HEAD_SIZE,
        WAN22_TI2V_ATTENTION_HEAD_SIZE,
    )
    if (
        tuple(weight.shape) != expected_weight_shape
        or tuple(bias.shape) != (expected_weight_shape[0],)
    ):
        issues.append(
            "proj_l geometry is malformed: "
            f"weight={tuple(weight.shape)}, bias={tuple(bias.shape)}"
        )
        return report, projection, issues
    report["projection_fp32"] = (
        weight.dtype is torch.float32 and bias.dtype is torch.float32
    )
    if not report["projection_fp32"]:
        issues.append(
            "proj_l weight/bias must remain torch.float32: "
            f"weight={weight.dtype}, bias={bias.dtype}"
        )
    report["projection_finite"] = bool(
        torch.isfinite(weight).all().item() and torch.isfinite(bias).all().item()
    )
    if not report["projection_finite"]:
        issues.append("proj_l contains non-finite learned weights")
    report["projection_nonzero"] = bool(torch.count_nonzero(weight).item())
    if not report["projection_nonzero"]:
        issues.append("proj_l.weight is all zero (untrained FastVideo default)")
    if report["projection_fp32"] and report["projection_finite"]:
        report["projection_weight_fingerprint"] = (
            _fastvideo_projection_fingerprint(layer_index, weight, bias)
        )
    return report, projection, issues


def _fastvideo_attention_roles(
    rpc_worker: Any,
    *,
    self_attention_only: bool = False,
) -> dict[str, list[Any]]:
    pipeline = getattr(rpc_worker, "pipeline", None)
    modules = getattr(pipeline, "modules", None)
    if not isinstance(modules, Mapping):
        raise RuntimeError("FastVideo worker pipeline does not expose a modules mapping")
    transformer = modules.get("transformer")
    while transformer is not None and getattr(transformer, "_orig_mod", None) is not None:
        transformer = transformer._orig_mod
    blocks = list(getattr(transformer, "blocks", ()) or ())
    if not blocks:
        raise RuntimeError("FastVideo Wan transformer exposes no attention blocks")

    roles: dict[str, list[Any]] = {"self": []}
    if not self_attention_only:
        roles["cross1"] = []
    for block_index, block in enumerate(blocks):
        self_attention = getattr(block, "attn1", None)
        if self_attention is None:
            raise RuntimeError(f"FastVideo Wan block {block_index} has no attn1 module")
        roles["self"].append(self_attention)
        if not self_attention_only:
            cross_container = getattr(block, "attn2", None)
            cross_attention = getattr(cross_container, "attn", None)
            if cross_attention is None:
                raise RuntimeError(
                    f"FastVideo Wan T2V block {block_index} has no attn2.attn module"
                )
            roles["cross1"].append(cross_attention)
    return roles


def _public_fastvideo_reference_probe(probe: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "probe_version": probe.get("probe_version"),
        "phase": probe.get("phase"),
        "requested_backend": probe.get("requested_backend"),
        "sla_kind": probe.get("sla_kind"),
        "self_attention_only": probe.get("self_attention_only"),
        "roles": _jsonable(probe.get("roles") or {}),
        "flash_attention_versions": _jsonable(
            probe.get("flash_attention_versions") or {}
        ),
        "flash_kernel_candidates": _jsonable(
            probe.get("flash_kernel_candidates") or []
        ),
        "kernel_wrappers_restored": probe.get("kernel_wrappers_restored"),
        "fallbacks": list(probe.get("fallbacks") or []),
        "issues": list(probe.get("issues") or []),
    }


def _fastvideo_flash_kernel_targets() -> tuple[
    list[dict[str, Any]], dict[str, str]
]:
    """Resolve the concrete FA4 -> FA3 -> FA2 callables selected upstream."""

    default_module = importlib.import_module(
        "fastvideo.attention.utils.flash_attn_default"
    )
    no_pad_module = importlib.import_module(
        "fastvideo.attention.utils.flash_attn_no_pad"
    )
    default_version = str(getattr(default_module, "fa_version", ""))
    varlen_version = str(getattr(no_pad_module, "_FA_VARLEN_VERSION", ""))
    if default_version not in {"2", "3", "4"}:
        raise RuntimeError(
            "FastVideo default FlashAttention resolver returned invalid version "
            f"{default_version!r}"
        )
    if varlen_version not in {"2", "3", "4"}:
        raise RuntimeError(
            "FastVideo varlen FlashAttention resolver returned invalid version "
            f"{varlen_version!r}"
        )
    default_attribute = "flash_attn_func" if default_version == "4" else "_fa_default"
    specs = (
        (default_module, default_attribute, "default", default_version),
        (
            no_pad_module,
            "flash_attn_varlen_qkvpacked_func",
            "masked_self_qkvpacked",
            "2",
        ),
        (
            no_pad_module,
            "flash_attn_varlen_func_impl",
            "varlen_qkv",
            varlen_version,
        ),
    )
    targets: list[dict[str, Any]] = []
    for module, attribute, entrypoint, version in specs:
        target = getattr(module, attribute, None)
        if not callable(target):
            raise RuntimeError(
                "FastVideo FlashAttention runtime target is unavailable: "
                f"{module.__name__}.{attribute}"
            )
        identity = _runtime_callable_identity(target)
        if not isinstance(identity, str) or not identity:
            raise RuntimeError(
                "FastVideo FlashAttention target has no callable identity: "
                f"{module.__name__}.{attribute}"
            )
        targets.append(
            {
                "module": module,
                "attribute": attribute,
                "entrypoint": entrypoint,
                "version": version,
                "identity": identity,
                "callable": target,
            }
        )
    return targets, {"default": default_version, "varlen": varlen_version}


def _restore_fastvideo_flash_kernel_wrappers(probe: dict[str, Any]) -> None:
    for target in probe.get("_flash_kernel_targets", ()):
        setattr(target["module"], target["attribute"], target["callable"])
    if probe.get("_flash_kernel_targets"):
        probe["kernel_wrappers_restored"] = True
    probe["_flash_kernel_targets"] = []


def _cleanup_fastvideo_flash_kernel_probe(probe: dict[str, Any]) -> None:
    for role, handles in list(
        probe.get("_flash_kernel_context_handles", {}).items()
    ):
        for handle in handles:
            handle.remove()
        role_report = probe.get("roles", {}).get(role, {})
        if isinstance(role_report, dict) and not role_report.get(
            "kernel_hook_removed_after_success"
        ):
            role_report["kernel_cleanup_removed_uncalled_hook"] = True
    probe.get("_flash_kernel_context_handles", {}).clear()
    context = probe.get("_flash_kernel_context")
    if context is not None and hasattr(context, "role"):
        del context.role
    _restore_fastvideo_flash_kernel_wrappers(probe)


def _install_fastvideo_flash_kernel_probe(
    probe: dict[str, Any],
    role_instances: Mapping[str, list[Any]],
) -> None:
    """Wrap real FA callables and attribute their execution to self/cross roles."""

    try:
        targets, versions = _fastvideo_flash_kernel_targets()
    except Exception as exc:
        probe["issues"].append(
            "FastVideo FLASH_ATTN concrete-kernel probe failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return
    probe["flash_attention_versions"] = versions
    probe["flash_kernel_candidates"] = [
        {
            key: target[key]
            for key in ("attribute", "entrypoint", "version", "identity")
        }
        for target in targets
    ]
    probe["kernel_wrappers_restored"] = False
    probe["_flash_kernel_targets"] = targets
    context = threading.local()
    probe["_flash_kernel_context"] = context

    for target in targets:
        original = target["callable"]

        @functools.wraps(original)
        def observed_kernel(
            *args: Any,
            _original: Callable[..., Any] = original,
            _target: Mapping[str, Any] = target,
            **kwargs: Any,
        ) -> Any:
            role = getattr(context, "role", None)
            if isinstance(role, str) and role in probe["roles"]:
                role_report = probe["roles"][role]
                role_report["kernel_calls"] += 1
                for field, value in (
                    ("kernel_paths", _target["identity"]),
                    ("kernel_versions", _target["version"]),
                    ("kernel_entrypoints", _target["entrypoint"]),
                ):
                    if value not in role_report[field]:
                        role_report[field].append(value)
            return _original(*args, **kwargs)

        setattr(target["module"], target["attribute"], observed_kernel)

    probe["_flash_kernel_context_handles"] = {}
    for role, instances in role_instances.items():
        if not instances:
            continue

        def enter_role(
            _module: Any,
            _inputs: Any,
            *,
            _role: str = role,
        ) -> None:
            context.role = _role

        def leave_role(
            _module: Any,
            _inputs: Any,
            _output: Any,
            *,
            _role: str = role,
        ) -> None:
            role_report = probe["roles"][_role]
            if getattr(context, "role", None) == _role:
                del context.role
            handles = probe["_flash_kernel_context_handles"].pop(_role, ())
            for handle in handles:
                handle.remove()
            if role_report["kernel_calls"] > 0:
                role_report["kernel_hook_removed_after_success"] = True
            else:
                role_report["kernel_cleanup_removed_uncalled_hook"] = True
            if all(
                int(probe["roles"][name].get("kernel_calls") or 0) > 0
                for name in role_instances
            ):
                _restore_fastvideo_flash_kernel_wrappers(probe)

        try:
            pre_handle = instances[0].register_forward_pre_hook(enter_role)
            try:
                post_handle = instances[0].register_forward_hook(
                    leave_role,
                    always_call=True,
                )
            except TypeError:
                post_handle = instances[0].register_forward_hook(leave_role)
            probe["_flash_kernel_context_handles"][role] = (
                pre_handle,
                post_handle,
            )
        except Exception as exc:
            probe["issues"].append(
                f"FastVideo {role} concrete-kernel context hook failed: "
                f"{type(exc).__name__}: {exc}"
            )


def _install_fastvideo_reference_probe(rpc_worker: Any) -> dict[str, Any]:
    """Install worker-local one-shot hooks on the executed Wan T2V attention roles."""

    requested_backend = str(
        os.environ.get("FASTVIDEO_ATTENTION_BACKEND") or "auto"
    ).strip().upper()
    sla_contract = _FASTVIDEO_SLA_BACKEND_CONTRACTS.get(requested_backend)
    self_attention_only = sla_contract is not None
    probe: dict[str, Any] = {
        "probe_version": "fastvideo-worker-attention-v3",
        "phase": "installing",
        "requested_backend": requested_backend,
        "sla_kind": sla_contract.get("kind") if sla_contract is not None else None,
        "self_attention_only": self_attention_only,
        "roles": {},
        "fallbacks": [],
        "issues": [],
        "_handles": {},
        "_handle_metadata": {},
    }
    setattr(rpc_worker, _FASTVIDEO_REFERENCE_PROBE_ATTRIBUTE, probe)
    try:
        role_instances = _fastvideo_attention_roles(
            rpc_worker,
            self_attention_only=self_attention_only,
        )
    except Exception as exc:
        probe["issues"].append(str(exc))
        probe["phase"] = "installation-failed"
        return _public_fastvideo_reference_probe(probe)

    projection_targets: dict[tuple[str, int], Any] = {}
    for role, instances in role_instances.items():
        layer_identities: list[dict[str, Any]] = []
        for block_index, instance in enumerate(instances):
            implementation = getattr(instance, "attn_impl", None)
            layer_report = {
                "block_index": block_index,
                "backend": _runtime_backend_name(
                    getattr(instance, "backend", None)
                ),
                "attention_class": _runtime_type_identity(instance),
                "implementation_class": (
                    _runtime_type_identity(implementation)
                    if implementation is not None
                    else None
                ),
            }
            if sla_contract is not None and role == "self":
                projection_report, projection, projection_issues = (
                    _fastvideo_sla_projection_layer(
                        implementation,
                        layer_index=block_index,
                    )
                )
                layer_report.update(
                    topk_ratio=getattr(implementation, "topk_ratio", None),
                    **projection_report,
                )
                if projection is not None:
                    projection_targets[(role, block_index)] = projection
                probe["issues"].extend(
                    f"FastVideo {role} block {block_index}: {issue}"
                    for issue in projection_issues
                )
            layer_identities.append(layer_report)
        backends = sorted({str(item["backend"]) for item in layer_identities})
        implementation_classes = sorted(
            {
                str(item["implementation_class"])
                for item in layer_identities
                if item["implementation_class"] is not None
            }
        )
        role_report = {
            "layer_count": len(instances),
            "layers": layer_identities,
            "backends": backends,
            "implementation_classes": implementation_classes,
            "identity_consistent": len(backends) == 1 and len(implementation_classes) == 1,
            "successful_calls": 0,
            "hook_removed_after_success": False,
            "cleanup_removed_uncalled_hook": False,
        }
        if requested_backend == "FLASH_ATTN":
            role_report.update(
                kernel_calls=0,
                kernel_paths=[],
                kernel_versions=[],
                kernel_entrypoints=[],
                kernel_hook_removed_after_success=False,
                kernel_cleanup_removed_uncalled_hook=False,
            )
        if sla_contract is not None and role == "self":
            layer_fingerprints = {
                str(item["block_index"]): str(
                    item["projection_weight_fingerprint"]
                )
                for item in layer_identities
                if isinstance(item.get("projection_weight_fingerprint"), str)
            }
            projection_source_keys = {
                str(item["block_index"]): list(item["projection_source_keys"])
                for item in layer_identities
                if isinstance(item.get("projection_source_keys"), (list, tuple))
            }
            complete_fingerprints = len(layer_fingerprints) == len(instances)
            role_report.update(
                checkpoint_layout=FASTVIDEO_SLA_CHECKPOINT_LAYOUT,
                expected_provider_path=sla_contract["provider_path"],
                expected_topk_ratio=sla_contract["topk_ratio"],
                projection_fp32=bool(layer_identities)
                and all(bool(item.get("projection_fp32")) for item in layer_identities),
                projection_finite=bool(layer_identities)
                and all(
                    bool(item.get("projection_finite")) for item in layer_identities
                ),
                projection_nonzero=bool(layer_identities)
                and all(
                    bool(item.get("projection_nonzero")) for item in layer_identities
                ),
                layer_projection_fingerprints=layer_fingerprints,
                projection_source_keys=projection_source_keys,
                all_projection_weights_fingerprint=(
                    _fastvideo_projection_set_fingerprint(
                        layer_fingerprints,
                        checkpoint_layout=FASTVIDEO_SLA_CHECKPOINT_LAYOUT,
                    )
                    if complete_fingerprints
                    else None
                ),
                canonical_projection_fingerprint=(
                    _fastvideo_projection_set_fingerprint(
                        layer_fingerprints,
                        checkpoint_layout=None,
                    )
                    if complete_fingerprints
                    else None
                ),
                projection_forward_complete=False,
            )
        probe["roles"][role] = role_report
        if len(backends) != 1:
            probe["issues"].append(
                f"FastVideo {role} layers resolved mixed attention backends: {backends!r}"
            )
        if len(implementation_classes) != 1:
            probe["issues"].append(
                f"FastVideo {role} layers resolved mixed or missing implementation classes: "
                f"{implementation_classes!r}"
            )
        if sla_contract is not None and role == "self":
            if len(instances) != WAN22_TI2V_TRANSFORMER_BLOCKS:
                probe["issues"].append(
                    "FastVideo learned SLA requires every Wan2.2 TI2V block; "
                    f"got {len(instances)}, expected {WAN22_TI2V_TRANSFORMER_BLOCKS}"
                )
            if implementation_classes != [sla_contract["provider_path"]]:
                probe["issues"].append(
                    "FastVideo self attention resolved the wrong learned-SLA "
                    f"implementation: {implementation_classes!r}"
                )
            invalid_topk = [
                item["block_index"]
                for item in layer_identities
                if item.get("topk_ratio") != sla_contract["topk_ratio"]
            ]
            if invalid_topk:
                probe["issues"].append(
                    "FastVideo learned-SLA topk_ratio mismatch in blocks "
                    f"{invalid_topk!r}"
                )
        for backend in backends:
            if backend in _FASTVIDEO_NON_ACCELERATED_BACKENDS or backend in {"", "NONE"}:
                probe["fallbacks"].append(
                    f"{role}: resolved non-accelerated attention backend {backend!r}"
                )
            elif requested_backend not in {"AUTO", ""} and backend != requested_backend:
                probe["fallbacks"].append(
                    f"{role}: requested {requested_backend!r} but resolved {backend!r}"
                )

    if (
        requested_backend == "FLASH_ATTN"
        and not probe["issues"]
        and not probe["fallbacks"]
    ):
        _install_fastvideo_flash_kernel_probe(probe, role_instances)

    if probe["issues"] or probe["fallbacks"]:
        _cleanup_fastvideo_flash_kernel_probe(probe)
        probe["phase"] = "installation-failed"
        return _public_fastvideo_reference_probe(probe)

    for role, instances in role_instances.items():
        role_report = probe["roles"][role]

        if sla_contract is not None and role == "self":
            for block_index in range(len(instances)):
                projection = projection_targets[(role, block_index)]
                handle_key = f"{role}:{block_index}"

                def successful_projection_hook(
                    _module: Any,
                    _inputs: Any,
                    _output: Any,
                    *,
                    _role: str = role,
                    _block_index: int = block_index,
                    _handle_key: str = handle_key,
                ) -> None:
                    current_role = probe["roles"][_role]
                    current_layer = current_role["layers"][_block_index]
                    current_layer["projection_forward_calls"] += 1
                    current_role["successful_calls"] += 1
                    handle = probe["_handles"].pop(_handle_key, None)
                    probe["_handle_metadata"].pop(_handle_key, None)
                    if handle is not None:
                        handle.remove()
                        current_layer[
                            "projection_hook_removed_after_success"
                        ] = True
                    if current_role["successful_calls"] == current_role["layer_count"]:
                        current_role["hook_removed_after_success"] = True
                        current_role["projection_forward_complete"] = True

                try:
                    probe["_handles"][handle_key] = projection.register_forward_hook(
                        successful_projection_hook
                    )
                    probe["_handle_metadata"][handle_key] = (role, block_index)
                except Exception as exc:
                    role_report["hook_install_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    probe["issues"].append(
                        "FastVideo self proj_l one-shot forward hook installation "
                        f"failed for block {block_index}: {exc}"
                    )
            continue

        def successful_forward_hook(
            _module: Any,
            _inputs: Any,
            _output: Any,
            *,
            _role: str = role,
        ) -> None:
            current_role = probe["roles"][_role]
            current_role["successful_calls"] += 1
            handle = probe["_handles"].pop(_role, None)
            probe["_handle_metadata"].pop(_role, None)
            if handle is not None:
                handle.remove()
                current_role["hook_removed_after_success"] = True

        try:
            probe["_handles"][role] = instances[0].register_forward_hook(
                successful_forward_hook
            )
            probe["_handle_metadata"][role] = (role, None)
        except Exception as exc:
            role_report["hook_install_error"] = f"{type(exc).__name__}: {exc}"
            probe["issues"].append(
                f"FastVideo {role} one-shot forward hook installation failed: {exc}"
            )

    if probe["issues"]:
        for handle_key, handle in list(probe["_handles"].items()):
            handle.remove()
            role, block_index = probe["_handle_metadata"].get(
                handle_key,
                (handle_key, None),
            )
            if block_index is None:
                probe["roles"][role]["cleanup_removed_uncalled_hook"] = True
            else:
                probe["roles"][role]["layers"][block_index][
                    "projection_cleanup_removed_uncalled_hook"
                ] = True
        probe["_handles"].clear()
        probe["_handle_metadata"].clear()
        probe["phase"] = "installation-failed"
    else:
        probe["phase"] = "installed"
    return _public_fastvideo_reference_probe(probe)


def _collect_fastvideo_reference_probe(rpc_worker: Any) -> dict[str, Any]:
    probe = getattr(rpc_worker, _FASTVIDEO_REFERENCE_PROBE_ATTRIBUTE, None)
    if not isinstance(probe, dict):
        return {
            "probe_version": "fastvideo-worker-attention-v3",
            "phase": "missing",
            "requested_backend": None,
            "sla_kind": None,
            "self_attention_only": None,
            "roles": {},
            "fallbacks": [],
            "issues": ["FastVideo worker reference probe was not installed"],
        }
    _cleanup_fastvideo_flash_kernel_probe(probe)
    for handle_key, handle in list(probe.get("_handles", {}).items()):
        handle.remove()
        role, block_index = probe.get("_handle_metadata", {}).get(
            handle_key,
            (handle_key, None),
        )
        if block_index is None:
            probe["roles"][role]["cleanup_removed_uncalled_hook"] = True
        else:
            probe["roles"][role]["layers"][block_index][
                "projection_cleanup_removed_uncalled_hook"
            ] = True
    probe.get("_handles", {}).clear()
    probe.get("_handle_metadata", {}).clear()
    probe["phase"] = "collected"
    return _public_fastvideo_reference_probe(probe)


def _verify_fastvideo_reference_probe(
    worker_records: Iterable[Mapping[str, Any]],
    *,
    require_successful_calls: bool,
    expected_projection: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    records = list(worker_records)
    checks: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    fallbacks: list[str] = []

    def require(name: str, passed: bool, actual: Any, expected: Any) -> None:
        checks[name] = {
            "actual": _jsonable(actual),
            "expected": _jsonable(expected),
            "passed": bool(passed),
        }
        if not passed:
            issues.append(f"{name}: actual={actual!r}, expected={expected!r}")

    sla_worker_fingerprints: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    require("workers", bool(records), len(records), "> 0")
    for worker_index, record in enumerate(records):
        expected_phase = "collected" if require_successful_calls else "installed"
        require(
            f"worker[{worker_index}].phase",
            record.get("phase") == expected_phase,
            record.get("phase"),
            expected_phase,
        )
        worker_issues = list(record.get("issues") or [])
        worker_fallbacks = list(record.get("fallbacks") or [])
        fallbacks.extend(f"worker[{worker_index}]: {item}" for item in worker_fallbacks)
        require(
            f"worker[{worker_index}].issues",
            not worker_issues,
            worker_issues,
            [],
        )
        require(
            f"worker[{worker_index}].fallbacks",
            not worker_fallbacks,
            worker_fallbacks,
            [],
        )
        roles = record.get("roles")
        if not isinstance(roles, Mapping):
            roles = {}
        requested_backend = str(record.get("requested_backend") or "").upper()
        sla_contract = _FASTVIDEO_SLA_BACKEND_CONTRACTS.get(requested_backend)
        expected_roles = ("self",) if sla_contract is not None else ("self", "cross1")
        flash_candidates_value = record.get("flash_kernel_candidates")
        flash_candidates = (
            list(flash_candidates_value)
            if isinstance(flash_candidates_value, (list, tuple))
            else []
        )
        flash_candidate_contracts = {
            (
                candidate.get("identity"),
                candidate.get("version"),
                candidate.get("entrypoint"),
            )
            for candidate in flash_candidates
            if isinstance(candidate, Mapping)
        }
        if requested_backend == "FLASH_ATTN":
            versions = record.get("flash_attention_versions")
            if not isinstance(versions, Mapping):
                versions = {}
            require(
                f"worker[{worker_index}].flash_attention.resolution",
                versions.get("default") in {"2", "3", "4"}
                and versions.get("varlen") == versions.get("default")
                and len(flash_candidates) == 3
                and len(flash_candidate_contracts) == 3
                and all(
                    isinstance(identity, str)
                    and bool(identity)
                    and version in {"2", "3", "4"}
                    and entrypoint
                    in {"default", "masked_self_qkvpacked", "varlen_qkv"}
                    for identity, version, entrypoint in flash_candidate_contracts
                ),
                {
                    "versions": versions,
                    "candidates": flash_candidates,
                },
                "one concrete FA4/FA3/FA2 resolution and three callable targets",
            )
        require(
            f"worker[{worker_index}].roles",
            set(roles) == set(expected_roles),
            sorted(roles),
            sorted(expected_roles),
        )
        if sla_contract is not None:
            require(
                f"worker[{worker_index}].sla_kind",
                record.get("sla_kind") == sla_contract["kind"]
                and record.get("self_attention_only") is True,
                {
                    "sla_kind": record.get("sla_kind"),
                    "self_attention_only": record.get("self_attention_only"),
                },
                {
                    "sla_kind": sla_contract["kind"],
                    "self_attention_only": True,
                },
            )
        for role in expected_roles:
            role_report = roles.get(role)
            if not isinstance(role_report, Mapping):
                role_report = {}
            backends = list(role_report.get("backends") or [])
            implementations = list(role_report.get("implementation_classes") or [])
            require(
                f"worker[{worker_index}].{role}.identity_consistent",
                bool(role_report.get("identity_consistent")),
                {
                    "backends": backends,
                    "implementation_classes": implementations,
                },
                "one backend and one implementation class across all layers",
            )
            require(
                f"worker[{worker_index}].{role}.accelerated_backend",
                len(backends) == 1
                and backends[0] not in _FASTVIDEO_NON_ACCELERATED_BACKENDS
                and backends[0] not in {"", "NONE"},
                backends,
                "one accelerated FastVideo backend",
            )
            if requested_backend == "FLASH_ATTN" and require_successful_calls:
                kernel_paths = list(role_report.get("kernel_paths") or [])
                kernel_versions = list(role_report.get("kernel_versions") or [])
                kernel_entrypoints = list(
                    role_report.get("kernel_entrypoints") or []
                )
                observed_contracts = {
                    (path, version, entrypoint)
                    for path in kernel_paths
                    for version in kernel_versions
                    for entrypoint in kernel_entrypoints
                }
                require(
                    f"worker[{worker_index}].{role}.concrete_kernel_calls",
                    int(role_report.get("kernel_calls") or 0) > 0
                    and len(kernel_paths) == 1
                    and len(kernel_versions) == 1
                    and len(kernel_entrypoints) == 1
                    and observed_contracts.issubset(flash_candidate_contracts)
                    and role_report.get("kernel_hook_removed_after_success")
                    is True
                    and role_report.get("kernel_cleanup_removed_uncalled_hook")
                    is False,
                    {
                        "kernel_calls": role_report.get("kernel_calls"),
                        "kernel_paths": kernel_paths,
                        "kernel_versions": kernel_versions,
                        "kernel_entrypoints": kernel_entrypoints,
                        "hook_removed_after_success": role_report.get(
                            "kernel_hook_removed_after_success"
                        ),
                        "cleanup_removed_uncalled_hook": role_report.get(
                            "kernel_cleanup_removed_uncalled_hook"
                        ),
                    },
                    "a directly observed concrete FlashAttention callable",
                )
            if sla_contract is not None:
                layers_value = role_report.get("layers")
                layers = (
                    list(layers_value)
                    if isinstance(layers_value, (list, tuple))
                    else []
                )
                layer_fingerprints_value = role_report.get(
                    "layer_projection_fingerprints"
                )
                layer_fingerprints = (
                    {
                        str(key): str(value)
                        for key, value in layer_fingerprints_value.items()
                    }
                    if isinstance(layer_fingerprints_value, Mapping)
                    else {}
                )
                source_keys_value = role_report.get("projection_source_keys")
                source_keys = (
                    {str(key): value for key, value in source_keys_value.items()}
                    if isinstance(source_keys_value, Mapping)
                    else {}
                )
                expected_layer_keys = {
                    str(index) for index in range(WAN22_TI2V_TRANSFORMER_BLOCKS)
                }
                malformed_layers: list[int] = []
                for layer_index, layer_value in enumerate(layers):
                    layer = layer_value if isinstance(layer_value, Mapping) else {}
                    expected_prefix = (
                        f"blocks.{layer_index}.attn1.attn_impl.proj_l"
                    )
                    if not (
                        layer.get("block_index") == layer_index
                        and layer.get("backend") == requested_backend
                        and layer.get("implementation_class")
                        == sla_contract["provider_path"]
                        and layer.get("topk_ratio") == sla_contract["topk_ratio"]
                        and layer.get("projection_weight_dtype")
                        == "torch.float32"
                        and layer.get("projection_bias_dtype") == "torch.float32"
                        and layer.get("projection_fp32") is True
                        and layer.get("projection_finite") is True
                        and layer.get("projection_nonzero") is True
                        and layer.get("projection_source_keys")
                        == [f"{expected_prefix}.weight", f"{expected_prefix}.bias"]
                        and layer.get("projection_weight_fingerprint")
                        == layer_fingerprints.get(str(layer_index))
                    ):
                        malformed_layers.append(layer_index)
                    if require_successful_calls and not (
                        layer.get("projection_forward_calls") == 1
                        and layer.get("projection_hook_removed_after_success") is True
                        and layer.get("projection_cleanup_removed_uncalled_hook")
                        is False
                    ):
                        malformed_layers.append(layer_index)
                fingerprints_valid = (
                    set(layer_fingerprints) == expected_layer_keys
                    and all(
                        len(value) == 64
                        and all(character in "0123456789abcdef" for character in value)
                        for value in layer_fingerprints.values()
                    )
                )
                expected_global = (
                    _fastvideo_projection_set_fingerprint(
                        layer_fingerprints,
                        checkpoint_layout=FASTVIDEO_SLA_CHECKPOINT_LAYOUT,
                    )
                    if fingerprints_valid
                    else None
                )
                expected_canonical = (
                    _fastvideo_projection_set_fingerprint(
                        layer_fingerprints,
                        checkpoint_layout=None,
                    )
                    if fingerprints_valid
                    else None
                )
                expected_source_keys = {
                    str(index): [
                        f"blocks.{index}.attn1.attn_impl.proj_l.weight",
                        f"blocks.{index}.attn1.attn_impl.proj_l.bias",
                    ]
                    for index in range(WAN22_TI2V_TRANSFORMER_BLOCKS)
                }
                require(
                    f"worker[{worker_index}].{role}.learned_projection",
                    len(layers) == WAN22_TI2V_TRANSFORMER_BLOCKS
                    and not malformed_layers
                    and backends == [requested_backend]
                    and implementations == [sla_contract["provider_path"]]
                    and role_report.get("checkpoint_layout")
                    == FASTVIDEO_SLA_CHECKPOINT_LAYOUT
                    and role_report.get("expected_provider_path")
                    == sla_contract["provider_path"]
                    and role_report.get("expected_topk_ratio")
                    == sla_contract["topk_ratio"]
                    and role_report.get("projection_fp32") is True
                    and role_report.get("projection_finite") is True
                    and role_report.get("projection_nonzero") is True
                    and fingerprints_valid
                    and source_keys == expected_source_keys
                    and role_report.get("all_projection_weights_fingerprint")
                    == expected_global
                    and role_report.get("canonical_projection_fingerprint")
                    == expected_canonical,
                    {
                        "layer_count": len(layers),
                        "malformed_layers": sorted(set(malformed_layers)),
                        "backends": backends,
                        "implementations": implementations,
                        "checkpoint_layout": role_report.get("checkpoint_layout"),
                        "projection_fp32": role_report.get("projection_fp32"),
                        "projection_finite": role_report.get("projection_finite"),
                        "projection_nonzero": role_report.get("projection_nonzero"),
                        "fingerprints_valid": fingerprints_valid,
                        "all_projection_weights_fingerprint": role_report.get(
                            "all_projection_weights_fingerprint"
                        ),
                        "canonical_projection_fingerprint": role_report.get(
                            "canonical_projection_fingerprint"
                        ),
                    },
                    "30 FP32, finite, nonzero learned projections with canonical "
                    "per-layer/global fingerprints",
                )
                if expected_projection is not None:
                    expected_checkpoint_layers_value = expected_projection.get(
                        "layer_projection_fingerprints"
                    )
                    expected_checkpoint_layers = (
                        {
                            str(key): str(value)
                            for key, value in expected_checkpoint_layers_value.items()
                        }
                        if isinstance(expected_checkpoint_layers_value, Mapping)
                        else {}
                    )
                    checkpoint_layers_valid = (
                        set(expected_checkpoint_layers) == expected_layer_keys
                        and all(
                            len(value) == 64
                            and all(
                                character in "0123456789abcdef"
                                for character in value
                            )
                            for value in expected_checkpoint_layers.values()
                        )
                    )
                    checkpoint_canonical = (
                        _fastvideo_projection_set_fingerprint(
                            expected_checkpoint_layers,
                            checkpoint_layout=None,
                        )
                        if checkpoint_layers_valid
                        else None
                    )
                    checkpoint_layout_bound = (
                        _fastvideo_projection_set_fingerprint(
                            expected_checkpoint_layers,
                            checkpoint_layout=FASTVIDEO_SLA_CHECKPOINT_LAYOUT,
                        )
                        if checkpoint_layers_valid
                        else None
                    )
                    require(
                        f"worker[{worker_index}].{role}.checkpoint_preflight",
                        checkpoint_layers_valid
                        and expected_projection.get("checkpoint_layout")
                        == FASTVIDEO_SLA_CHECKPOINT_LAYOUT
                        and expected_projection.get("projection_fingerprint")
                        == checkpoint_layout_bound
                        and expected_projection.get(
                            "canonical_projection_fingerprint"
                        )
                        == checkpoint_canonical
                        and layer_fingerprints == expected_checkpoint_layers
                        and expected_global == checkpoint_layout_bound
                        and expected_canonical == checkpoint_canonical,
                        {
                            "checkpoint_layout": expected_projection.get(
                                "checkpoint_layout"
                            ),
                            "checkpoint_projection_fingerprint": (
                                expected_projection.get("projection_fingerprint")
                            ),
                            "checkpoint_canonical_projection_fingerprint": (
                                expected_projection.get(
                                    "canonical_projection_fingerprint"
                                )
                            ),
                            "runtime_projection_fingerprint": expected_global,
                            "runtime_canonical_projection_fingerprint": (
                                expected_canonical
                            ),
                            "layer_fingerprints_match": (
                                layer_fingerprints == expected_checkpoint_layers
                            ),
                        },
                        "runtime projections exactly match the independent "
                        "safetensors checkpoint preflight",
                    )
                if isinstance(expected_canonical, str):
                    sla_worker_fingerprints.append(
                        (
                            expected_canonical,
                            tuple(sorted(layer_fingerprints.items())),
                        )
                    )
            if require_successful_calls:
                expected_calls = (
                    WAN22_TI2V_TRANSFORMER_BLOCKS
                    if sla_contract is not None
                    else 1
                )
                require(
                    f"worker[{worker_index}].{role}.successful_calls",
                    int(role_report.get("successful_calls") or 0)
                    == expected_calls,
                    role_report.get("successful_calls"),
                    expected_calls,
                )
                require(
                    f"worker[{worker_index}].{role}.hook_removed_after_success",
                    bool(role_report.get("hook_removed_after_success")),
                    role_report.get("hook_removed_after_success"),
                    True,
                )
                if sla_contract is not None:
                    require(
                        f"worker[{worker_index}].{role}.projection_forward_complete",
                        role_report.get("projection_forward_complete") is True,
                        role_report.get("projection_forward_complete"),
                        True,
                    )

        if requested_backend == "FLASH_ATTN" and require_successful_calls:
            require(
                f"worker[{worker_index}].flash_attention.wrappers_restored",
                record.get("kernel_wrappers_restored") is True,
                record.get("kernel_wrappers_restored"),
                True,
            )

    if sla_worker_fingerprints:
        require(
            "workers.learned_projection_consistent",
            len(set(sla_worker_fingerprints)) == 1
            and len(sla_worker_fingerprints) == len(records),
            sla_worker_fingerprints,
            "one identical learned projection set on every FastVideo worker",
        )

    verification = {
        "passed": not issues,
        "checks": checks,
        "issues": issues,
        "fallbacks": fallbacks,
        "workers": _jsonable(records),
    }
    if issues:
        raise RuntimeError(
            "FastVideo runtime effectiveness gate failed: " + "; ".join(issues)
        )
    return verification


def _lightx2v_attention_request(args: argparse.Namespace) -> dict[str, Any]:
    """Resolve the legacy dense shorthand into explicit self/cross roles."""

    legacy_mode = str(args.lightx2v_attention_mode).strip().casefold()
    explicit_self_mode = getattr(args, "lightx2v_self_attention_mode", None)
    explicit_cross_mode = getattr(args, "lightx2v_cross_attention_mode", None)
    self_mode = str(explicit_self_mode or legacy_mode).strip().casefold()
    unsupported_self = _LIGHTX2V_FAIL_CLOSED_ATTENTION_MODES.get(self_mode)
    if unsupported_self is not None:
        raise ValueError(
            f"LightX2V self attention {self_mode!r} is intentionally "
            f"fail-closed: {unsupported_self}"
        )
    if explicit_cross_mode is not None:
        cross_mode = str(explicit_cross_mode).strip().casefold()
    elif legacy_mode in _LIGHTX2V_ATTENTION_KERNEL_CONTRACTS:
        cross_mode = legacy_mode
    else:
        raise ValueError(
            "a sparse LightX2V self-attention request requires an explicit "
            "dense --lightx2v-cross-attention-mode"
        )

    unsupported_cross = _LIGHTX2V_FAIL_CLOSED_ATTENTION_MODES.get(cross_mode)
    if unsupported_cross is not None:
        raise ValueError(
            f"LightX2V cross attention {cross_mode!r} is intentionally "
            f"fail-closed: {unsupported_cross}"
        )
    if cross_mode not in _LIGHTX2V_ATTENTION_KERNEL_CONTRACTS:
        raise ValueError(
            "LightX2V cross attention must use a certifying dense provider; "
            f"got {cross_mode!r}"
        )

    raw_settings = getattr(
        args,
        "lightx2v_self_attention_settings_json",
        "{}",
    )
    settings = (
        dict(raw_settings)
        if isinstance(raw_settings, Mapping)
        else json.loads(str(raw_settings))
    )
    if not isinstance(settings, dict):
        raise TypeError(
            "--lightx2v-self-attention-settings-json must decode to an object"
        )

    sparse_contract = _LIGHTX2V_SPARSE_ATTENTION_CONTRACTS.get(self_mode)
    if sparse_contract is None:
        if self_mode not in _LIGHTX2V_ATTENTION_KERNEL_CONTRACTS:
            raise ValueError(
                f"LightX2V self attention {self_mode!r} has no fail-closed "
                "dense or sparse runtime contract"
            )
        if settings:
            raise ValueError(
                "LightX2V self-attention settings are only valid for a "
                "certifying sparse self provider"
            )
        return {
            "legacy_attention_mode": legacy_mode,
            "self_mode": self_mode,
            "cross_mode": cross_mode,
            "self_family": "dense",
            "self_settings": {},
            "self_setting_key": None,
        }

    required_fields = set(sparse_contract["required_setting_fields"])
    allowed_fields = set(sparse_contract["allowed_setting_fields"])
    missing_fields = sorted(required_fields - set(settings))
    unknown_fields = sorted(set(settings) - allowed_fields)
    if missing_fields or unknown_fields:
        raise ValueError(
            f"LightX2V {self_mode} settings mismatch: "
            f"missing={missing_fields!r}, unknown={unknown_fields!r}"
        )

    if self_mode == "dynamic_sparse_attn":
        sparsity_ratio = settings.get("sparsity_ratio")
        if (
            isinstance(sparsity_ratio, bool)
            or not isinstance(sparsity_ratio, (int, float))
            or not 0.0 < float(sparsity_ratio) < 1.0
        ):
            raise ValueError(
                "dynamic_sparse_attn sparsity_ratio must be strictly between 0 and 1"
            )
        operator = str(settings.get("operator")).strip().casefold()
        if operator not in _LIGHTX2V_DYNAMIC_SPARSE_PROVIDER_PATHS:
            raise ValueError(
                f"dynamic_sparse_attn operator {operator!r} has no certifying "
                "sparse provider contract"
            )
        settings["sparsity_ratio"] = float(sparsity_ratio)
        settings["operator"] = operator
        if "per_block_mean" in settings and not isinstance(
            settings["per_block_mean"], bool
        ):
            raise TypeError("dynamic_sparse_attn per_block_mean must be boolean")
    else:
        mask_name = str(settings["sparse_mask_generator"]).strip().casefold()
        operator_name = str(settings["sparse_operator"]).strip().casefold()
        sparse_setting = settings["sparse_setting"]
        operator_setting = settings.get("operator_setting", {})
        if mask_name not in _LIGHTX2V_GENERAL_SPARSE_MASK_PATHS:
            raise ValueError(
                f"general_sparse_attn mask generator {mask_name!r} has no "
                "certifying sparse contract"
            )
        if operator_name not in _LIGHTX2V_GENERAL_SPARSE_OPERATOR_PATHS:
            raise ValueError(
                f"general_sparse_attn operator {operator_name!r} has no "
                "certifying sparse provider contract"
            )
        if not isinstance(sparse_setting, dict) or not sparse_setting:
            raise TypeError(
                "general_sparse_attn sparse_setting must be a non-empty object"
            )
        if not isinstance(operator_setting, dict):
            raise TypeError(
                "general_sparse_attn operator_setting must be an object"
            )
        settings.update(
            sparse_mask_generator=mask_name,
            sparse_operator=operator_name,
            sparse_setting=dict(sparse_setting),
            operator_setting=dict(operator_setting),
        )

    return {
        "legacy_attention_mode": legacy_mode,
        "self_mode": self_mode,
        "cross_mode": cross_mode,
        "self_family": "sparse",
        "self_settings": settings,
        "self_setting_key": sparse_contract["setting_key"],
    }


def _lightx2v_attention_roles(pipeline: Any) -> dict[str, list[Any]]:
    runner = getattr(pipeline, "runner", None)
    model = getattr(runner, "model", None)
    transformer_weights = getattr(model, "transformer_weights", None)
    blocks = list(getattr(transformer_weights, "blocks", ()) or ())
    if not blocks:
        raise RuntimeError("LightX2V Wan runner exposes no transformer weight blocks")
    roles: dict[str, list[Any]] = {"self": [], "cross1": []}
    for block_index, block in enumerate(blocks):
        phases = list(getattr(block, "compute_phases", ()) or ())
        if len(phases) < 2:
            raise RuntimeError(
                f"LightX2V Wan block {block_index} exposes fewer than two compute phases"
            )
        self_attention = getattr(phases[0], "self_attn_1", None)
        cross_attention = getattr(phases[1], "cross_attn_1", None)
        if self_attention is None:
            raise RuntimeError(f"LightX2V Wan block {block_index} has no self_attn_1 instance")
        if cross_attention is None:
            raise RuntimeError(f"LightX2V Wan T2V block {block_index} has no cross_attn_1 instance")
        roles["self"].append(self_attention)
        roles["cross1"].append(cross_attention)
    return roles


def _public_lightx2v_reference_probe(probe: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "probe_version": probe.get("probe_version"),
        "phase": probe.get("phase"),
        "requested_mode": probe.get("requested_mode"),
        "requested_roles": _jsonable(probe.get("requested_roles") or {}),
        "self_family": probe.get("self_family"),
        "self_settings": _jsonable(probe.get("self_settings") or {}),
        "roles": _jsonable(probe.get("roles") or {}),
        "fallbacks": list(probe.get("fallbacks") or []),
        "issues": list(probe.get("issues") or []),
    }


def _lightx2v_sparse_mask_receipt(mask: Any) -> dict[str, Any]:
    receipt = {
        "shape": list(getattr(mask, "shape", ()) or ()),
        "dtype": str(getattr(mask, "dtype", None)),
        "device": str(getattr(mask, "device", None)),
        "numel": None,
        "nonzero": None,
        "strictly_sparse": False,
        "error": None,
    }
    try:
        numel = int(mask.numel())
        nonzero_value = mask.count_nonzero()
        nonzero = int(
            nonzero_value.item()
            if callable(getattr(nonzero_value, "item", None))
            else nonzero_value
        )
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        return receipt
    receipt.update(
        numel=numel,
        nonzero=nonzero,
        strictly_sparse=numel > 0 and 0 < nonzero < numel,
    )
    return receipt


def _install_lightx2v_one_shot_callable(
    probe: dict[str, Any],
    *,
    owner: Any,
    attribute: str,
    role: str,
    restorer_key: str,
    successful_calls_field: str,
    restored_field: str,
    cleanup_field: str,
    result_receipt: Callable[[Any], Any] | None = None,
) -> None:
    role_report = probe["roles"][role]
    original = getattr(owner, attribute)
    owner_attributes = vars(owner)
    had_instance_attribute = attribute in owner_attributes
    previous_instance_attribute = owner_attributes.get(attribute)
    restore_state = {"restored": False}

    def restore() -> None:
        if restore_state["restored"]:
            return
        if had_instance_attribute:
            setattr(owner, attribute, previous_instance_attribute)
        elif attribute in vars(owner):
            delattr(owner, attribute)
        restore_state["restored"] = True
        role_report[restored_field] = True

    def one_shot(*call_args: Any, **call_kwargs: Any) -> Any:
        result = original(*call_args, **call_kwargs)
        if result_receipt is not None:
            role_report["sparse_mask"] = result_receipt(result)
        role_report[successful_calls_field] += 1
        restore()
        return result

    setattr(owner, attribute, one_shot)
    probe["_restorers"][restorer_key] = restore
    probe["_restorer_metadata"][restorer_key] = {
        "role": role,
        "successful_calls_field": successful_calls_field,
        "restored_field": restored_field,
        "cleanup_field": cleanup_field,
    }


def _lightx2v_dynamic_sparse_kernel_target(
    instance: Any,
    operator: str,
) -> tuple[Any, str, str, Any]:
    module_name = type(instance).__module__
    module = sys.modules.get(module_name)
    if module is None:
        raise RuntimeError(
            f"dynamic sparse implementation module {module_name!r} is not loaded"
        )
    dotted_attribute = _LIGHTX2V_DYNAMIC_SPARSE_KERNEL_ATTRIBUTES[operator]
    target_module = module
    target_module_name = module_name
    if operator == "sage2":
        dispatcher = getattr(module, dotted_attribute)
        if not callable(dispatcher):
            raise TypeError(
                f"{module_name}.{dotted_attribute} is not a callable dispatcher"
            )
        dispatcher_module_name = getattr(dispatcher, "__module__", None)
        if not isinstance(dispatcher_module_name, str):
            raise RuntimeError(
                "sage2 sparse dispatcher does not expose its implementation module"
            )
        dispatcher_module = sys.modules.get(dispatcher_module_name)
        if dispatcher_module is None:
            raise RuntimeError(
                f"sage2 sparse dispatcher module {dispatcher_module_name!r} "
                "is not loaded"
            )
        arch = str(getattr(instance, "arch", ""))
        dotted_attribute = _LIGHTX2V_SAGE2_SPARSE_KERNEL_ATTRIBUTES_BY_ARCH.get(
            arch,
            (
                _LIGHTX2V_SAGE2_SPARSE_DEFAULT_FP16_KERNEL_ATTRIBUTE
                if getattr(dispatcher_module, "SAGE2PP_ENABLED", False) is True
                else _LIGHTX2V_SAGE2_SPARSE_DEFAULT_FP32_KERNEL_ATTRIBUTE
            ),
        )
        target_module = dispatcher_module
        target_module_name = dispatcher_module_name
    components = dotted_attribute.split(".")
    owner = target_module
    for component in components[:-1]:
        owner = getattr(owner, component)
    attribute = components[-1]
    return (
        owner,
        attribute,
        f"{target_module_name}.{dotted_attribute}",
        getattr(owner, attribute),
    )


def _lightx2v_sparse_provider_report(
    instances: list[Any],
    *,
    requested_mode: str,
    settings: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    if requested_mode == "dynamic_sparse_attn":
        operator = str(settings["operator"])
        expected_provider_path = _LIGHTX2V_DYNAMIC_SPARSE_PROVIDER_PATHS[operator]
        provider_paths = sorted(
            {
                str(_runtime_callable_identity(getattr(instance, "apply_func", None)))
                for instance in instances
            }
        )
        dispatcher_attribute = None
        dispatcher_callable = None
        if operator == "sage2":
            dispatcher_attribute = (
                f"{type(instances[0]).__module__}."
                f"{_LIGHTX2V_DYNAMIC_SPARSE_KERNEL_ATTRIBUTES[operator]}"
            )
            implementation_module = sys.modules.get(type(instances[0]).__module__)
            dispatcher_callable = (
                getattr(
                    implementation_module,
                    _LIGHTX2V_DYNAMIC_SPARSE_KERNEL_ATTRIBUTES[operator],
                    None,
                )
                if implementation_module is not None
                else None
            )
            if not callable(dispatcher_callable):
                issues.append(
                    f"dynamic_sparse_attn sage2 dispatcher "
                    f"{dispatcher_attribute} is not callable"
                )
        try:
            _, _, kernel_attribute, kernel_callable = (
                _lightx2v_dynamic_sparse_kernel_target(instances[0], operator)
            )
        except Exception as exc:
            kernel_attribute = (
                f"{type(instances[0]).__module__}."
                f"{_LIGHTX2V_DYNAMIC_SPARSE_KERNEL_ATTRIBUTES[operator]}"
            )
            kernel_callable = None
            issues.append(
                "dynamic_sparse_attn could not resolve its selected sparse "
                f"kernel {kernel_attribute}: {type(exc).__name__}: {exc}"
            )
        kernel_available = callable(kernel_callable)
        if not kernel_available and not any(
            "could not resolve its selected sparse kernel" in issue
            for issue in issues
        ):
            issues.append(
                f"dynamic_sparse_attn sparse kernel {kernel_attribute} is not callable"
            )
        effective_settings = [
            {
                "sparsity_ratio": getattr(instance, "sparsity_ratio", None),
                "operator": getattr(instance, "operator", None),
                "per_block_mean": getattr(instance, "per_block_mean", None),
                "topk": getattr(instance, "topk", None),
                "block_q": getattr(instance, "BLKQ", None),
                "block_k": getattr(instance, "BLKK", None),
                "arch": getattr(instance, "arch", None),
            }
            for instance in instances
        ]
        expected_ratio = float(settings["sparsity_ratio"])
        settings_match = bool(effective_settings) and all(
            item["operator"] == operator
            and item["sparsity_ratio"] == expected_ratio
            and item["topk"] == 1.0 - expected_ratio
            and (
                "per_block_mean" not in settings
                or item["per_block_mean"] is settings["per_block_mean"]
            )
            for item in effective_settings
        )
        if provider_paths != [expected_provider_path]:
            issues.append(
                "dynamic_sparse_attn resolved the wrong provider method: "
                f"{provider_paths!r}, expected {[expected_provider_path]!r}"
            )
        if not settings_match:
            issues.append(
                "dynamic_sparse_attn instances did not consume the requested "
                f"settings: {effective_settings!r}"
            )
        return {
            "kind": "dynamic-apply-func",
            "operator": operator,
            "expected_provider_path": expected_provider_path,
            "provider_paths": provider_paths,
            "dispatcher_attribute": dispatcher_attribute,
            "dispatcher_callable_identity": _runtime_callable_identity(
                dispatcher_callable
            ),
            "dispatcher_available": (
                callable(dispatcher_callable) if operator == "sage2" else None
            ),
            "kernel_attribute": kernel_attribute,
            "kernel_callable_identity": _runtime_callable_identity(
                kernel_callable
            ),
            "kernel_available": kernel_available,
            "effective_settings": effective_settings,
            "settings_match": settings_match,
        }, issues

    expected_operator_path = _LIGHTX2V_GENERAL_SPARSE_OPERATOR_PATHS[
        str(settings["sparse_operator"])
    ]
    expected_mask_path = _LIGHTX2V_GENERAL_SPARSE_MASK_PATHS[
        str(settings["sparse_mask_generator"])
    ]
    operator_paths = sorted(
        {_runtime_type_identity(getattr(instance, "operator", None)) for instance in instances}
    )
    mask_paths = sorted(
        {
            _runtime_type_identity(getattr(instance, "mask_generator", None))
            for instance in instances
        }
    )
    effective_settings = [
        {
            "sparse_operator": getattr(instance, "sparse_operator", None),
            "sparse_mask_generator": getattr(
                instance,
                "sparse_mask_generator",
                None,
            ),
            "sparse_setting": _jsonable(
                getattr(instance, "sparse_setting", None)
            ),
            "operator_setting": _jsonable(
                getattr(instance, "operator_setting", None)
            ),
        }
        for instance in instances
    ]
    settings_match = bool(effective_settings) and all(
        item["sparse_operator"] == settings["sparse_operator"]
        and item["sparse_mask_generator"] == settings["sparse_mask_generator"]
        and item["sparse_setting"] == _jsonable(settings["sparse_setting"])
        and item["operator_setting"] == _jsonable(settings.get("operator_setting", {}))
        for item in effective_settings
    )
    if operator_paths != [expected_operator_path]:
        issues.append(
            "general_sparse_attn resolved the wrong sparse operator: "
            f"{operator_paths!r}, expected {[expected_operator_path]!r}"
        )
    if mask_paths != [expected_mask_path]:
        issues.append(
            "general_sparse_attn resolved the wrong mask generator: "
            f"{mask_paths!r}, expected {[expected_mask_path]!r}"
        )
    if not settings_match:
        issues.append(
            "general_sparse_attn instances did not consume the requested "
            f"settings: {effective_settings!r}"
        )
    return {
        "kind": "general-operator",
        "expected_provider_path": expected_operator_path,
        "provider_paths": operator_paths,
        "expected_mask_generator_path": expected_mask_path,
        "mask_generator_paths": mask_paths,
        "effective_settings": effective_settings,
        "settings_match": settings_match,
    }, issues


def _install_lightx2v_reference_probe(
    pipeline: Any,
    requested_mode: str | Mapping[str, Any],
) -> dict[str, Any]:
    """Prove role-specific T2V providers with one-shot successful calls."""

    if isinstance(requested_mode, Mapping):
        self_mode = str(requested_mode["self_mode"]).strip().casefold()
        cross_mode = str(requested_mode["cross_mode"]).strip().casefold()
        self_family = str(requested_mode["self_family"])
        self_settings = dict(requested_mode.get("self_settings") or {})
    else:
        self_mode = str(requested_mode).strip().casefold()
        cross_mode = self_mode
        self_family = "dense"
        self_settings = {}
    requested_roles = {"self": self_mode, "cross1": cross_mode}
    probe: dict[str, Any] = {
        "probe_version": "lightx2v-role-attention-v2",
        "phase": "installing",
        "requested_mode": self_mode,
        "requested_roles": requested_roles,
        "self_family": self_family,
        "self_settings": self_settings,
        "roles": {},
        "fallbacks": [],
        "issues": [],
        "_restorers": {},
        "_restorer_metadata": {},
    }
    setattr(pipeline, _LIGHTX2V_REFERENCE_PROBE_ATTRIBUTE, probe)
    if self_family == "sparse":
        self_contract = _LIGHTX2V_SPARSE_ATTENTION_CONTRACTS.get(self_mode)
    else:
        self_contract = _LIGHTX2V_ATTENTION_KERNEL_CONTRACTS.get(self_mode)
    cross_contract = _LIGHTX2V_ATTENTION_KERNEL_CONTRACTS.get(cross_mode)
    if self_contract is None or cross_contract is None:
        probe["issues"].append(
            "LightX2V role request has no certifying provider contract: "
            f"self={self_mode!r}, cross1={cross_mode!r}"
        )
        probe["phase"] = "installation-failed"
        return _public_lightx2v_reference_probe(probe)
    try:
        role_instances = _lightx2v_attention_roles(pipeline)
    except Exception as exc:
        probe["issues"].append(str(exc))
        probe["phase"] = "installation-failed"
        return _public_lightx2v_reference_probe(probe)

    for role, instances in role_instances.items():
        role_mode = requested_roles[role]
        role_family = "sparse" if role == "self" and self_family == "sparse" else "dense"
        if role_family == "sparse":
            sparse_role_contract = _LIGHTX2V_SPARSE_ATTENTION_CONTRACTS[role_mode]
            expected_class = str(sparse_role_contract["implementation_class"])
            expected_symbols: tuple[str, ...] = ()
        else:
            expected_class, expected_symbols = _LIGHTX2V_ATTENTION_KERNEL_CONTRACTS[
                role_mode
            ]
        instance_classes = [_runtime_type_identity(instance) for instance in instances]
        implementation_classes = sorted(set(instance_classes))
        implementation_module_name = type(instances[0]).__module__
        implementation_module = sys.modules.get(implementation_module_name)
        kernel_symbols = []
        for symbol_name in expected_symbols:
            symbol = (
                getattr(implementation_module, symbol_name, None)
                if implementation_module is not None
                else None
            )
            kernel_symbols.append(
                {
                    "module_attribute": f"{implementation_module_name}.{symbol_name}",
                    "callable_identity": _runtime_callable_identity(symbol),
                    "available": callable(symbol),
                }
            )
        role_report = {
            "requested_mode": role_mode,
            "attention_family": role_family,
            "layer_count": len(instances),
            "implementation_classes": implementation_classes,
            "expected_class": expected_class,
            "kernel_symbols": kernel_symbols,
            "identity_consistent": len(implementation_classes) == 1,
            "successful_calls": 0,
            "method_restored_after_success": False,
            "cleanup_removed_uncalled_wrapper": False,
        }
        if role_family == "sparse":
            sparse_provider, sparse_issues = _lightx2v_sparse_provider_report(
                instances,
                requested_mode=role_mode,
                settings=self_settings,
            )
            role_report.update(
                sparse_provider=sparse_provider,
                sparse_provider_successful_calls=0,
                sparse_provider_restored_after_success=False,
                sparse_provider_cleanup_removed_uncalled_wrapper=False,
                sparse_dispatch_successful_calls=0,
                sparse_final_kernel_successful_calls=0,
                sparse_final_kernel_restored_after_success=False,
                sparse_final_kernel_cleanup_removed_uncalled_wrapper=False,
                sparse_mask_successful_calls=0,
                sparse_mask_restored_after_success=False,
                sparse_mask_cleanup_removed_uncalled_wrapper=False,
                sparse_mask=None,
                dense_provider_successful_calls=0,
            )
            probe["issues"].extend(
                f"LightX2V self: {issue}" for issue in sparse_issues
            )
        probe["roles"][role] = role_report
        if implementation_classes != [expected_class]:
            probe["fallbacks"].append(
                f"{role}: requested {role_mode!r} ({expected_class}) but resolved "
                f"classes {implementation_classes!r}"
            )
        if role_family == "dense" and not all(
            symbol["available"] for symbol in kernel_symbols
        ):
            missing = [
                symbol["module_attribute"]
                for symbol in kernel_symbols
                if not symbol["available"]
            ]
            probe["fallbacks"].append(
                f"{role}: selected {role_mode!r} kernel symbols are unavailable: {missing!r}"
            )
        if len(implementation_classes) != 1:
            probe["issues"].append(
                f"LightX2V {role} layers resolved mixed implementation classes: "
                f"{implementation_classes!r}"
            )

    if probe["issues"] or probe["fallbacks"]:
        probe["phase"] = "installation-failed"
        return _public_lightx2v_reference_probe(probe)

    if self_family == "sparse":
        self_instance = role_instances["self"][0]
        try:
            if self_mode == "dynamic_sparse_attn":
                kernel_owner, kernel_attribute, _, _ = (
                    _lightx2v_dynamic_sparse_kernel_target(
                        self_instance,
                        str(self_settings["operator"]),
                    )
                )
                _install_lightx2v_one_shot_callable(
                    probe,
                    owner=kernel_owner,
                    attribute=kernel_attribute,
                    role="self",
                    restorer_key="self:sparse-final-kernel",
                    successful_calls_field="sparse_final_kernel_successful_calls",
                    restored_field="sparse_final_kernel_restored_after_success",
                    cleanup_field=(
                        "sparse_final_kernel_cleanup_removed_uncalled_wrapper"
                    ),
                )
                implementation_module = sys.modules[type(self_instance).__module__]

                def dynamic_mask_receipt(result: Any) -> dict[str, Any]:
                    mask = result[0] if isinstance(result, (list, tuple)) else None
                    return _lightx2v_sparse_mask_receipt(mask)

                _install_lightx2v_one_shot_callable(
                    probe,
                    owner=implementation_module,
                    attribute="get_block_map",
                    role="self",
                    restorer_key="self:sparse-mask",
                    successful_calls_field="sparse_mask_successful_calls",
                    restored_field="sparse_mask_restored_after_success",
                    cleanup_field="sparse_mask_cleanup_removed_uncalled_wrapper",
                    result_receipt=dynamic_mask_receipt,
                )
            else:
                _install_lightx2v_one_shot_callable(
                    probe,
                    owner=self_instance,
                    attribute="operator",
                    role="self",
                    restorer_key="self:sparse-provider",
                    successful_calls_field="sparse_provider_successful_calls",
                    restored_field="sparse_provider_restored_after_success",
                    cleanup_field=(
                        "sparse_provider_cleanup_removed_uncalled_wrapper"
                    ),
                )
                _install_lightx2v_one_shot_callable(
                    probe,
                    owner=self_instance,
                    attribute="mask_generator",
                    role="self",
                    restorer_key="self:sparse-mask",
                    successful_calls_field="sparse_mask_successful_calls",
                    restored_field="sparse_mask_restored_after_success",
                    cleanup_field="sparse_mask_cleanup_removed_uncalled_wrapper",
                    result_receipt=_lightx2v_sparse_mask_receipt,
                )
        except Exception as exc:
            probe["issues"].append(
                f"LightX2V self sparse-provider wrapper installation failed: {exc}"
            )

    for role, instances in role_instances.items():
        instance = instances[0]
        original_apply = instance.apply
        instance_attributes = vars(instance)
        had_instance_apply = "apply" in instance_attributes
        previous_instance_apply = instance_attributes.get("apply")
        restore_state = {"restored": False}

        def restore_method(
            *,
            _instance: Any = instance,
            _had_instance_apply: bool = had_instance_apply,
            _previous_instance_apply: Any = previous_instance_apply,
            _role: str = role,
            _restore_state: dict[str, bool] = restore_state,
        ) -> None:
            if _restore_state["restored"]:
                return
            if _had_instance_apply:
                setattr(_instance, "apply", _previous_instance_apply)
            elif "apply" in vars(_instance):
                delattr(_instance, "apply")
            _restore_state["restored"] = True
            probe["roles"][_role]["method_restored_after_success"] = True

        def one_shot_apply(
            *call_args: Any,
            _original_apply: Callable[..., Any] = original_apply,
            _restore_method: Callable[[], None] = restore_method,
            _role: str = role,
            _counts_sparse_dispatch: bool = (
                role == "self"
                and self_family == "sparse"
                and self_mode == "dynamic_sparse_attn"
            ),
            **call_kwargs: Any,
        ) -> Any:
            result = _original_apply(*call_args, **call_kwargs)
            probe["roles"][_role]["successful_calls"] += 1
            if _counts_sparse_dispatch:
                probe["roles"][_role]["sparse_dispatch_successful_calls"] += 1
            _restore_method()
            return result

        try:
            setattr(instance, "apply", one_shot_apply)
            restorer_key = f"{role}:apply"
            probe["_restorers"][restorer_key] = restore_method
            probe["_restorer_metadata"][restorer_key] = {
                "role": role,
                "successful_calls_field": "successful_calls",
                "restored_field": "method_restored_after_success",
                "cleanup_field": "cleanup_removed_uncalled_wrapper",
            }
        except Exception as exc:
            probe["issues"].append(
                f"LightX2V {role} one-shot apply wrapper installation failed: {exc}"
            )

    if probe["issues"]:
        for restorer_key, restorer in list(probe["_restorers"].items()):
            metadata = probe["_restorer_metadata"][restorer_key]
            role = metadata["role"]
            restorer()
            probe["roles"][role][metadata["restored_field"]] = False
            probe["roles"][role][metadata["cleanup_field"]] = True
        probe["_restorers"].clear()
        probe["_restorer_metadata"].clear()
        probe["phase"] = "installation-failed"
    else:
        probe["phase"] = "installed"
    return _public_lightx2v_reference_probe(probe)


def _collect_lightx2v_reference_probe(pipeline: Any) -> dict[str, Any]:
    probe = getattr(pipeline, _LIGHTX2V_REFERENCE_PROBE_ATTRIBUTE, None)
    if not isinstance(probe, dict):
        return {
            "probe_version": "lightx2v-role-attention-v2",
            "phase": "missing",
            "requested_mode": None,
            "requested_roles": {},
            "self_family": None,
            "self_settings": {},
            "roles": {},
            "fallbacks": [],
            "issues": ["LightX2V reference probe was not installed"],
        }
    metadata_by_key = probe.get("_restorer_metadata", {})
    for restorer_key, restorer in list(probe.get("_restorers", {}).items()):
        metadata = metadata_by_key[restorer_key]
        role = metadata["role"]
        role_report = probe["roles"][role]
        if int(role_report.get(metadata["successful_calls_field"]) or 0) == 0:
            restorer()
            role_report[metadata["restored_field"]] = False
            role_report[metadata["cleanup_field"]] = True
    probe.get("_restorers", {}).clear()
    probe.get("_restorer_metadata", {}).clear()
    probe["phase"] = "collected"
    return _public_lightx2v_reference_probe(probe)


def _verify_lightx2v_reference_probe(
    report: Mapping[str, Any],
    *,
    require_successful_calls: bool,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    fallbacks = list(report.get("fallbacks") or [])

    def require(name: str, passed: bool, actual: Any, expected: Any) -> None:
        checks[name] = {
            "actual": _jsonable(actual),
            "expected": _jsonable(expected),
            "passed": bool(passed),
        }
        if not passed:
            issues.append(f"{name}: actual={actual!r}, expected={expected!r}")

    reported_issues = list(report.get("issues") or [])
    expected_phase = "collected" if require_successful_calls else "installed"
    require("phase", report.get("phase") == expected_phase, report.get("phase"), expected_phase)
    require("issues", not reported_issues, reported_issues, [])
    require("fallbacks", not fallbacks, fallbacks, [])
    roles = report.get("roles")
    if not isinstance(roles, Mapping):
        roles = {}
    require("roles", set(roles) == {"self", "cross1"}, sorted(roles), ["cross1", "self"])
    for role in ("self", "cross1"):
        role_report = roles.get(role)
        if not isinstance(role_report, Mapping):
            role_report = {}
        attention_family = role_report.get("attention_family")
        kernel_symbols = list(role_report.get("kernel_symbols") or [])
        require(
            f"{role}.identity_consistent",
            bool(role_report.get("identity_consistent")),
            role_report.get("implementation_classes"),
            "one implementation class across all layers",
        )
        if attention_family == "sparse":
            sparse_provider = role_report.get("sparse_provider")
            if not isinstance(sparse_provider, Mapping):
                sparse_provider = {}
            require(
                f"{role}.sparse_provider_identity",
                sparse_provider.get("provider_paths")
                == [sparse_provider.get("expected_provider_path")],
                sparse_provider.get("provider_paths"),
                [sparse_provider.get("expected_provider_path")],
            )
            require(
                f"{role}.sparse_settings",
                sparse_provider.get("settings_match") is True,
                sparse_provider.get("effective_settings"),
                "requested sparse settings on every layer",
            )
            if sparse_provider.get("kind") == "dynamic-apply-func":
                require(
                    f"{role}.sparse_final_kernel_available",
                    sparse_provider.get("kernel_available") is True,
                    {
                        "kernel_attribute": sparse_provider.get(
                            "kernel_attribute"
                        ),
                        "kernel_callable_identity": sparse_provider.get(
                            "kernel_callable_identity"
                        ),
                        "kernel_available": sparse_provider.get(
                            "kernel_available"
                        ),
                    },
                    "selected dynamic sparse final kernel callable",
                )
                if sparse_provider.get("operator") == "sage2":
                    require(
                        f"{role}.sparse_sage2_dispatcher_available",
                        sparse_provider.get("dispatcher_available") is True,
                        {
                            "dispatcher_attribute": sparse_provider.get(
                                "dispatcher_attribute"
                            ),
                            "dispatcher_callable_identity": sparse_provider.get(
                                "dispatcher_callable_identity"
                            ),
                            "dispatcher_available": sparse_provider.get(
                                "dispatcher_available"
                            ),
                        },
                        "selected Sage2 sparse dispatcher callable",
                    )
        else:
            require(
                f"{role}.attention_family",
                attention_family == "dense",
                attention_family,
                "dense",
            )
            require(
                f"{role}.kernel_symbols",
                bool(kernel_symbols)
                and all(bool(item.get("available")) for item in kernel_symbols),
                kernel_symbols,
                "all selected dense kernel symbols callable",
            )
        if require_successful_calls:
            require(
                f"{role}.successful_calls",
                int(role_report.get("successful_calls") or 0) == 1,
                role_report.get("successful_calls"),
                1,
            )
            require(
                f"{role}.method_restored_after_success",
                bool(role_report.get("method_restored_after_success")),
                role_report.get("method_restored_after_success"),
                True,
            )
            if attention_family == "sparse":
                sparse_provider = role_report.get("sparse_provider")
                if not isinstance(sparse_provider, Mapping):
                    sparse_provider = {}
                if sparse_provider.get("kind") == "dynamic-apply-func":
                    require(
                        f"{role}.sparse_dispatch_successful_calls",
                        int(
                            role_report.get("sparse_dispatch_successful_calls")
                            or 0
                        )
                        == 1,
                        role_report.get("sparse_dispatch_successful_calls"),
                        1,
                    )
                    require(
                        f"{role}.sparse_final_kernel_successful_calls",
                        int(
                            role_report.get(
                                "sparse_final_kernel_successful_calls"
                            )
                            or 0
                        )
                        == 1,
                        role_report.get("sparse_final_kernel_successful_calls"),
                        1,
                    )
                    require(
                        f"{role}.sparse_final_kernel_restored_after_success",
                        bool(
                            role_report.get(
                                "sparse_final_kernel_restored_after_success"
                            )
                        ),
                        role_report.get(
                            "sparse_final_kernel_restored_after_success"
                        ),
                        True,
                    )
                else:
                    require(
                        f"{role}.sparse_provider_successful_calls",
                        int(
                            role_report.get("sparse_provider_successful_calls")
                            or 0
                        )
                        == 1,
                        role_report.get("sparse_provider_successful_calls"),
                        1,
                    )
                    require(
                        f"{role}.sparse_provider_restored_after_success",
                        bool(
                            role_report.get(
                                "sparse_provider_restored_after_success"
                            )
                        ),
                        role_report.get(
                            "sparse_provider_restored_after_success"
                        ),
                        True,
                    )
                require(
                    f"{role}.sparse_mask_successful_calls",
                    int(role_report.get("sparse_mask_successful_calls") or 0)
                    == 1,
                    role_report.get("sparse_mask_successful_calls"),
                    1,
                )
                require(
                    f"{role}.sparse_mask_restored_after_success",
                    bool(
                        role_report.get("sparse_mask_restored_after_success")
                    ),
                    role_report.get("sparse_mask_restored_after_success"),
                    True,
                )
                sparse_mask = role_report.get("sparse_mask")
                if not isinstance(sparse_mask, Mapping):
                    sparse_mask = {}
                require(
                    f"{role}.strictly_sparse_mask",
                    sparse_mask.get("strictly_sparse") is True,
                    sparse_mask,
                    "observed mask with 0 < nonzero < numel",
                )
                require(
                    f"{role}.dense_provider_successful_calls",
                    int(role_report.get("dense_provider_successful_calls") or 0)
                    == 0,
                    role_report.get("dense_provider_successful_calls"),
                    0,
                )

    verification = {
        "passed": not issues,
        "checks": checks,
        "issues": issues,
        "fallbacks": fallbacks,
        "probe": _jsonable(report),
    }
    if issues:
        raise RuntimeError(
            "LightX2V runtime effectiveness gate failed: " + "; ".join(issues)
        )
    return verification


def _worldfoundry_optimization_report(pipeline: Any) -> dict[str, Any]:
    native = getattr(pipeline, "native_pipeline", None)
    runner = getattr(native, "runner", None)
    components = getattr(runner, "components", None)
    denoiser = getattr(components, "denoiser", None)
    decoder = getattr(components, "decoder", None)
    vae = getattr(decoder, "vae", None)
    vae_reporter = getattr(decoder, "runtime_optimization_report", None)
    vae_report = vae_reporter() if callable(vae_reporter) else None
    decode_autocast_dtype = getattr(vae, "decode_autocast_dtype", None)
    vae_weight_dtype = getattr(decoder, "dtype", None)
    if callable(vae_weight_dtype):
        vae_weight_dtype = vae_weight_dtype()
    model = getattr(denoiser, "model", denoiser)
    parameter_dtypes: dict[str, int] = {}
    if model is not None and hasattr(model, "parameters"):
        for parameter in model.parameters():
            name = str(parameter.dtype).removeprefix("torch.")
            parameter_dtypes[name] = parameter_dtypes.get(name, 0) + parameter.numel()
    dit_weight_dtype = getattr(model, "_worldfoundry_dit_weight_dtype", None)
    applied = getattr(model, "_worldfoundry_applied_optimizations", None)
    cross_kv_cache = getattr(model, "_worldfoundry_static_cross_kv", None)
    cache_report = cross_kv_cache.report() if cross_kv_cache is not None else None
    runtime_reporter = getattr(denoiser, "runtime_optimization_report", None)
    runtime_report = runtime_reporter() if callable(runtime_reporter) else None
    if applied is None:
        return {
            "available": False,
            "vae_decode_autocast": str(decode_autocast_dtype).removeprefix("torch.")
            if decode_autocast_dtype is not None
            else None,
            "vae_weight_dtype": (
                str(vae_weight_dtype).removeprefix("torch.")
                if vae_weight_dtype is not None
                else None
            ),
            "vae_optimization_report": vae_report,
            "static_cross_kv_cache": cache_report,
            "dit_weight_dtype": (
                str(dit_weight_dtype).removeprefix("torch.") if dit_weight_dtype is not None else None
            ),
            "dit_parameter_dtypes": parameter_dtypes,
        }
    report = {
        "available": True,
        "requested": dict(applied.requested),
        "effective": dict(applied.effective),
        "fallbacks": list(applied.fallbacks),
        "quality_tier": applied.quality_tier,
        "vae_decode_autocast": str(decode_autocast_dtype).removeprefix("torch.")
        if decode_autocast_dtype is not None
        else None,
        "vae_weight_dtype": (
            str(vae_weight_dtype).removeprefix("torch.")
            if vae_weight_dtype is not None
            else None
        ),
        "vae_optimization_report": vae_report,
        "static_cross_kv_cache": cache_report,
        "dit_weight_dtype": (str(dit_weight_dtype).removeprefix("torch.") if dit_weight_dtype is not None else None),
        "dit_parameter_dtypes": parameter_dtypes,
    }
    if runtime_report is not None:
        report.update(
            requested=dict(runtime_report["requested"]),
            effective=dict(runtime_report["effective"]),
            fallbacks=list(runtime_report["fallbacks"]),
            quality_tier=runtime_report["quality_tier"],
            runtime=runtime_report["runtime"],
        )
    if isinstance(vae_report, Mapping):
        vae_effective = vae_report.get("effective")
        if isinstance(vae_effective, Mapping):
            report["vae_channels_last_3d"] = vae_effective.get(
                "vae_channels_last_3d"
            )
            report["vae_channels_last_3d_conv3d"] = vae_effective.get(
                "vae_channels_last_3d_conv3d"
            )
            report["vae_conv3d_count"] = vae_effective.get(
                "vae_conv3d_count"
            )
    return report


def _verify_worldfoundry_optimization_report(
    report: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Reject samples that requested a speedup without runtime proof."""

    checks: dict[str, dict[str, Any]] = {}
    issues: list[str] = []

    def require(name: str, passed: bool, actual: Any, expected: Any) -> None:
        checks[name] = {
            "actual": _jsonable(actual),
            "expected": _jsonable(expected),
            "passed": bool(passed),
        }
        if not passed:
            issues.append(f"{name}: actual={actual!r}, expected={expected!r}")

    def runtime_counter(counters: Mapping[str, Any], name: str) -> int | None:
        value = counters.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    require("report.available", bool(report.get("available")), report.get("available"), True)
    fallbacks = list(report.get("fallbacks") or [])
    require("fallbacks", not fallbacks, fallbacks, [])
    if not bool(args.worldfoundry_optimized):
        return {"passed": not issues, "checks": checks, "issues": issues}

    effective = report.get("effective")
    requested = report.get("requested")
    runtime = report.get("runtime")
    if not isinstance(requested, Mapping):
        requested = {}
    if not isinstance(effective, Mapping):
        effective = {}
    if not isinstance(runtime, Mapping):
        runtime = {}

    expected_rope = (
        f"fused-{args.worldfoundry_rope_precision}"
        if args.worldfoundry_fused_rope
        else f"complex-{args.worldfoundry_rope_precision}"
    )
    require(
        "rope_precision.requested",
        requested.get("rope_precision") == args.worldfoundry_rope_precision,
        requested.get("rope_precision"),
        args.worldfoundry_rope_precision,
    )
    require(
        "rope_precision.effective",
        effective.get("rope_precision") == expected_rope,
        effective.get("rope_precision"),
        expected_rope,
    )
    rms_effective = effective.get("rms_norm_precision")
    if not isinstance(rms_effective, Mapping):
        rms_effective = {}
    require(
        "rms_norm_precision.requested",
        requested.get("rms_norm_precision")
        == args.worldfoundry_rms_norm_precision,
        requested.get("rms_norm_precision"),
        args.worldfoundry_rms_norm_precision,
    )
    require(
        "rms_norm_precision.mode",
        rms_effective.get("mode") == args.worldfoundry_rms_norm_precision,
        rms_effective.get("mode"),
        args.worldfoundry_rms_norm_precision,
    )
    rms_modules = runtime_counter(rms_effective, "configured_modules")
    require(
        "rms_norm_precision.configured_modules",
        rms_modules is not None and rms_modules > 0,
        rms_effective.get("configured_modules"),
        "> 0",
    )

    timestep = runtime.get("wan_timestep") or {}
    if not isinstance(timestep, Mapping):
        timestep = {}
    timestep_calls = runtime_counter(timestep, "calls")
    global_timestep_calls = runtime_counter(timestep, "global_calls")
    explicit_all_ones_calls = runtime_counter(
        timestep,
        "explicit_all_ones_calls",
    )
    require(
        "wan_timestep.mode",
        effective.get("wan_timestep_mode") == "global-explicit-all-ones",
        effective.get("wan_timestep_mode"),
        "global-explicit-all-ones",
    )
    require(
        "wan_timestep.global_explicit_calls",
        timestep_calls is not None
        and timestep_calls > 0
        and global_timestep_calls == timestep_calls
        and explicit_all_ones_calls == timestep_calls,
        {
            "calls": timestep_calls,
            "global_calls": global_timestep_calls,
            "explicit_all_ones_calls": explicit_all_ones_calls,
        },
        "calls == global_calls == explicit_all_ones_calls > 0",
    )
    require(
        "wan_timestep.per_token_calls",
        runtime_counter(timestep, "per_token_calls") == 0,
        timestep.get("per_token_calls"),
        0,
    )

    expected_dtype = {
        "fp32": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
    }[args.worldfoundry_dit_weight_dtype]
    require(
        "dit_weight_dtype",
        report.get("dit_weight_dtype") == expected_dtype,
        report.get("dit_weight_dtype"),
        expected_dtype,
    )
    expected_vae_dtype = (
        None
        if args.worldfoundry_vae_decode_autocast == "none"
        else {"fp16": "float16", "bf16": "bfloat16"}[
            args.worldfoundry_vae_decode_autocast
        ]
    )
    require(
        "vae_decode_autocast",
        report.get("vae_decode_autocast") == expected_vae_dtype,
        report.get("vae_decode_autocast"),
        expected_vae_dtype,
    )
    requested_vae_weight_dtype = getattr(
        args,
        "worldfoundry_vae_weight_dtype",
        None,
    )
    if requested_vae_weight_dtype is not None:
        expected_vae_weight_dtype = {
            "fp32": "float32",
            "fp16": "float16",
            "bf16": "bfloat16",
        }[requested_vae_weight_dtype]
        require(
            "vae_weight_dtype",
            report.get("vae_weight_dtype") == expected_vae_weight_dtype,
            report.get("vae_weight_dtype"),
            expected_vae_weight_dtype,
        )
    if bool(getattr(args, "worldfoundry_vae_channels_last_3d", False)):
        converted = report.get("vae_channels_last_3d_conv3d")
        total = report.get("vae_conv3d_count")
        require(
            "vae_channels_last_3d",
            report.get("vae_channels_last_3d") == "enabled",
            report.get("vae_channels_last_3d"),
            "enabled",
        )
        require(
            "vae_channels_last_3d_conv3d",
            isinstance(converted, int)
            and not isinstance(converted, bool)
            and isinstance(total, int)
            and not isinstance(total, bool)
            and total > 0
            and converted == total,
            {"effective": converted, "total": total},
            "effective == total > 0",
        )

    attention = str(effective.get("attention") or "")
    requested_attention = str(args.worldfoundry_attention_backend)
    attention_prefixes = {
        "torch": ("torch", "pytorch"),
        "flash": ("flash_attention",),
        "flash2": ("flash_attention_2",),
        "flash3": ("flash_attention_3",),
        "flash4": ("flash_attention_4",),
        "sage": ("sage",),
        "sage3": ("sage",),
        "xformers": ("xformers",),
        "auto": ("torch", "pytorch", "flash", "sage", "xformers"),
    }[requested_attention]
    require(
        "attention",
        bool(attention) and attention.startswith(attention_prefixes),
        attention,
        requested_attention,
    )
    canonical_attention_providers = {
        "torch": "torch",
        "torch_sdpa": "torch",
        "pytorch": "torch",
        "pytorch_sdpa": "torch",
        "flash_attention_2": "flash_attention_2",
        "flash_attention_3": "flash_attention_3",
        "flash_attention_4": "flash_attention_4",
        "sage_attention": "sage_attention",
        "sage_attention_3": "sage_attention_3",
        "xformers": "xformers",
    }
    target_provider = canonical_attention_providers.get(attention.casefold(), attention.casefold())
    attention_dispatch = runtime.get("attention_dispatch") or {}
    provider_calls = (
        attention_dispatch.get("provider_calls")
        if isinstance(attention_dispatch, Mapping)
        else None
    )
    if not isinstance(provider_calls, Mapping):
        provider_calls = {}
    provider_counters = provider_calls.get(target_provider)
    if not isinstance(provider_counters, Mapping):
        provider_counters = {}
    attempts = runtime_counter(provider_counters, "attempts")
    successes = runtime_counter(provider_counters, "successes")
    compiled_graph_traces = runtime_counter(
        provider_counters,
        "compiled_graph_traces",
    )
    require(
        "attention_dispatch.target_provider",
        bool(target_provider) and target_provider in provider_calls,
        {
            "target": target_provider,
            "available": sorted(str(provider) for provider in provider_calls),
        },
        f"provider_calls contains {target_provider!r}",
    )
    if args.worldfoundry_compile:
        compile_runtime = runtime.get("compile") or {}
        if not isinstance(compile_runtime, Mapping):
            compile_runtime = {}
        compile_calls = runtime_counter(compile_runtime, "calls")
        provider_graph_traces = compile_runtime.get("attention_provider_graph_traces") or {}
        if not isinstance(provider_graph_traces, Mapping):
            provider_graph_traces = {}
        scoped_graph_traces = runtime_counter(provider_graph_traces, target_provider)
        require(
            "compile_runtime.attention_provider_graph_traces",
            scoped_graph_traces is not None and scoped_graph_traces > 0,
            {
                "target": target_provider,
                "scoped_graph_traces": scoped_graph_traces,
                "global_graph_traces_after_request_reset": compiled_graph_traces,
            },
            f"scoped graph traces for {target_provider!r} > 0",
        )
        require(
            "compile_runtime.calls",
            compile_calls is not None and compile_calls > 0,
            compile_calls,
            "> 0",
        )
        require(
            "compile_runtime.wrapper_installed",
            compile_runtime.get("wrapper_installed") is True,
            compile_runtime.get("wrapper_installed"),
            True,
        )
        require(
            "compile_runtime.failures",
            runtime_counter(compile_runtime, "failures") == 0,
            compile_runtime.get("failures"),
            0,
        )
        require(
            "compile_runtime.last_error",
            compile_runtime.get("last_error") is None,
            compile_runtime.get("last_error"),
            None,
        )
    else:
        require(
            "attention_dispatch.provider.successes",
            successes is not None and successes > 0,
            successes,
            "> 0",
        )
    require(
        "attention_dispatch.provider.attempts_equal_successes",
        attempts is not None and successes is not None and attempts == successes,
        {"attempts": attempts, "successes": successes},
        "attempts == successes",
    )
    for counter_name in ("fallbacks", "errors", "quarantined_skips"):
        counter_value = runtime_counter(provider_counters, counter_name)
        require(
            f"attention_dispatch.provider.{counter_name}",
            counter_value == 0,
            counter_value,
            0,
        )
    if args.worldfoundry_fuse_qkv:
        qkv_fusion = runtime.get("qkv_fusion") or {}
        if not isinstance(qkv_fusion, Mapping):
            qkv_fusion = {}
        require(
            "fuse_qkv_blocks",
            int(effective.get("fuse_qkv_blocks") or 0) > 0,
            effective.get("fuse_qkv_blocks"),
            "> 0",
        )
        require(
            "qkv_fusion.fused_blocks",
            runtime_counter(qkv_fusion, "fused_blocks") is not None
            and int(qkv_fusion["fused_blocks"]) > 0,
            qkv_fusion.get("fused_blocks"),
            "> 0",
        )
        eager_projection_calls = runtime_counter(
            qkv_fusion,
            "eager_projection_calls",
        )
        qkv_graph_traces = runtime_counter(
            qkv_fusion,
            "compiled_graph_traces",
        )
        require(
            "qkv_fusion.eager_projection_calls",
            eager_projection_calls is not None,
            qkv_fusion.get("eager_projection_calls"),
            "a non-negative integer",
        )
        require(
            "qkv_fusion.compiled_graph_traces",
            qkv_graph_traces is not None,
            qkv_fusion.get("compiled_graph_traces"),
            "a non-negative integer",
        )
        require(
            "qkv_fusion.strategy",
            qkv_fusion.get("strategy") == args.worldfoundry_qkv_strategy,
            qkv_fusion.get("strategy"),
            args.worldfoundry_qkv_strategy,
        )
        require(
            "qkv_fusion.split_threshold",
            runtime_counter(qkv_fusion, "split_threshold")
            == args.worldfoundry_qkv_split_threshold,
            qkv_fusion.get("split_threshold"),
            args.worldfoundry_qkv_split_threshold,
        )
        eager_paths = {
            path: runtime_counter(
                qkv_fusion,
                f"eager_{path}_projection_calls",
            )
            for path in ("packed", "split")
        }
        compiled_paths = {
            path: runtime_counter(
                qkv_fusion,
                f"compiled_{path}_graph_traces",
            )
            for path in ("packed", "split")
        }
        require(
            "qkv_fusion.eager_path_totals",
            all(value is not None for value in eager_paths.values())
            and sum(int(value) for value in eager_paths.values())
            == eager_projection_calls,
            eager_paths,
            "packed + split == eager_projection_calls",
        )
        require(
            "qkv_fusion.compiled_path_totals",
            all(value is not None for value in compiled_paths.values())
            and sum(int(value) for value in compiled_paths.values())
            == qkv_graph_traces,
            compiled_paths,
            "packed + split == compiled_graph_traces",
        )
        expected_qkv_path = str(args.worldfoundry_qkv_strategy)
        if expected_qkv_path == "auto":
            flattened_rows = math.prod(
                _wan22_ti2v_patch_grid(
                    frames=int(args.frames),
                    height=int(args.height),
                    width=int(args.width),
                )
            )
            expected_qkv_path = (
                "split"
                if flattened_rows >= args.worldfoundry_qkv_split_threshold
                else "packed"
            )
        if args.worldfoundry_compile:
            require(
                "qkv_fusion.compiled_execution",
                qkv_graph_traces is not None and qkv_graph_traces > 0,
                qkv_graph_traces,
                "compiled_graph_traces > 0 with an executed compile wrapper",
            )
            require(
                "qkv_fusion.compiled_path",
                (compiled_paths.get(expected_qkv_path) or 0) > 0,
                compiled_paths,
                f"{expected_qkv_path} graph traces > 0",
            )
        else:
            require(
                "qkv_fusion.eager_execution",
                eager_projection_calls is not None and eager_projection_calls > 0,
                eager_projection_calls,
                "> 0",
            )
            require(
                "qkv_fusion.eager_path",
                (eager_paths.get(expected_qkv_path) or 0) > 0,
                eager_paths,
                f"{expected_qkv_path} projection calls > 0",
            )
    if args.worldfoundry_inplace_residual:
        inplace_residual = runtime.get("inplace_residual") or {}
        if not isinstance(inplace_residual, Mapping):
            inplace_residual = {}
        calls = runtime_counter(inplace_residual, "calls")
        inplace_calls = runtime_counter(inplace_residual, "inplace_calls")
        require(
            "inplace_residual.effective",
            effective.get("inplace_residual") == "in-place-executed",
            effective.get("inplace_residual"),
            "in-place-executed",
        )
        require(
            "inplace_residual.calls",
            calls is not None and calls > 0,
            inplace_residual.get("calls"),
            "> 0",
        )
        require(
            "inplace_residual.inplace_calls_equal_calls",
            calls is not None
            and inplace_calls is not None
            and inplace_calls == calls,
            {"calls": calls, "inplace_calls": inplace_calls},
            "inplace_calls == calls",
        )
        for counter_name in (
            "functional_calls",
            "autograd_fallback_calls",
            "feature_cache_fallback_calls",
        ):
            counter_value = runtime_counter(inplace_residual, counter_name)
            require(
                f"inplace_residual.{counter_name}",
                counter_value == 0,
                inplace_residual.get(counter_name),
                0,
            )
    if args.worldfoundry_static_cross_kv:
        static_cross_kv = runtime.get("static_cross_kv") or {}
        if not isinstance(static_cross_kv, Mapping):
            static_cross_kv = {}
        require(
            "static_cross_kv_blocks",
            int(effective.get("static_cross_kv_blocks") or 0) > 0,
            effective.get("static_cross_kv_blocks"),
            "> 0",
        )
        require(
            "static_cross_kv.wrapped_blocks",
            runtime_counter(static_cross_kv, "wrapped_blocks") is not None
            and int(static_cross_kv["wrapped_blocks"]) > 0,
            static_cross_kv.get("wrapped_blocks"),
            "> 0",
        )
        require(
            "static_cross_kv.effective",
            static_cross_kv.get("effective") == "kv-reuse",
            static_cross_kv.get("effective"),
            "kv-reuse",
        )
        for counter_name in ("hits", "misses", "condition_hits"):
            counter_value = runtime_counter(static_cross_kv, counter_name)
            require(
                f"static_cross_kv.{counter_name}",
                counter_value is not None and counter_value > 0,
                static_cross_kv.get(counter_name),
                "> 0",
            )
        require(
            "static_cross_kv.bypasses",
            runtime_counter(static_cross_kv, "bypasses") == 0,
            static_cross_kv.get("bypasses"),
            0,
        )
        bypass_kwargs = static_cross_kv.get("bypass_kwargs")
        require(
            "static_cross_kv.bypass_kwargs",
            isinstance(bypass_kwargs, (list, tuple)) and not bypass_kwargs,
            bypass_kwargs,
            [],
        )
    if args.worldfoundry_fused_rope:
        require("fused_rope", bool(effective.get("fused_rope")), effective.get("fused_rope"), True)
        fused_rope = runtime.get("fused_rope") or {}
        if not isinstance(fused_rope, Mapping):
            fused_rope = {}
        installed_blocks = runtime_counter(fused_rope, "installed_blocks")
        eager_calls = runtime_counter(fused_rope, "eager_calls")
        fused_graph_traces = runtime_counter(
            fused_rope,
            "compiled_graph_traces",
        )
        require(
            "fused_rope.installed_blocks",
            installed_blocks is not None and installed_blocks > 0,
            fused_rope.get("installed_blocks"),
            "> 0",
        )
        require(
            "fused_rope.eager_calls",
            eager_calls is not None,
            fused_rope.get("eager_calls"),
            "a non-negative integer",
        )
        require(
            "fused_rope.compiled_graph_traces",
            fused_graph_traces is not None,
            fused_rope.get("compiled_graph_traces"),
            "a non-negative integer",
        )
        for counter_name in (
            "torch_fallback_calls",
            "provider_failures",
            "quarantined_skips",
            "malformed_receipts",
        ):
            counter_value = runtime_counter(fused_rope, counter_name)
            require(
                f"fused_rope.{counter_name}",
                counter_value == 0,
                fused_rope.get(counter_name),
                0,
            )
        if args.worldfoundry_compile:
            require(
                "fused_rope.compiled_execution",
                fused_graph_traces is not None and fused_graph_traces > 0,
                fused_graph_traces,
                "compiled_graph_traces > 0 with an executed compile wrapper",
            )
        else:
            provider_calls = runtime_counter(fused_rope, "provider_calls")
            require(
                "fused_rope.provider_calls",
                provider_calls is not None and provider_calls > 0,
                fused_rope.get("provider_calls"),
                "> 0",
            )
            require(
                "fused_rope.provider_calls_equal_eager_calls",
                provider_calls is not None
                and eager_calls is not None
                and provider_calls == eager_calls,
                {"provider_calls": provider_calls, "eager_calls": eager_calls},
                "provider_calls == eager_calls",
            )
            provider_paths = fused_rope.get("provider_paths")
            require(
                "fused_rope.provider_paths",
                isinstance(provider_paths, (list, tuple))
                and bool(provider_paths)
                and set(provider_paths)
                == {"triton_hidden_qk_rmsnorm_rope_3d"},
                provider_paths,
                ["triton_hidden_qk_rmsnorm_rope_3d"],
            )
    if args.worldfoundry_compile:
        require(
            "compile",
            effective.get("compile") == "compile-wrapper-executed",
            effective.get("compile"),
            "compile-wrapper-executed",
        )
    if args.worldfoundry_cuda_graph:
        graph = runtime.get("cuda_graph") or {}
        if not isinstance(graph, Mapping):
            graph = {}
        graph_effective = effective.get("cuda_graph")
        require(
            "cuda_graph.effective",
            graph_effective
            in {"captured-and-replayed", "replayed-existing-graph"},
            graph_effective,
            "captured-and-replayed or replayed-existing-graph",
        )
        capture_calls = runtime_counter(graph, "capture")
        cached_graphs = runtime_counter(graph, "graphs")
        require(
            "cuda_graph.capture_or_existing_graph",
            capture_calls is not None
            and cached_graphs is not None
            and (capture_calls > 0 or cached_graphs > 0),
            {"capture": graph.get("capture"), "graphs": graph.get("graphs")},
            "capture > 0 or graphs > 0",
        )
        require(
            "cuda_graph.replay",
            (runtime_counter(graph, "replay") or 0) > 0,
            graph.get("replay"),
            "> 0",
        )
        require(
            "cuda_graph.eager",
            runtime_counter(graph, "eager") == 0,
            graph.get("eager"),
            0,
        )
        require(
            "cuda_graph.capture_failed",
            runtime_counter(graph, "capture_failed") == 0,
            graph.get("capture_failed"),
            0,
        )
    requested_feature_cache = str(args.worldfoundry_feature_cache)
    if (
        requested_feature_cache == "none"
        and args.worldfoundry_teacache_threshold is not None
    ):
        requested_feature_cache = "teacache"
    if requested_feature_cache != "none":
        cache = runtime.get("feature_cache") or runtime.get("teacache") or {}
        if not isinstance(cache, Mapping):
            cache = {}
        expected_cache_algorithm = requested_feature_cache
        require(
            "feature_cache.enabled",
            bool(cache.get("enabled")),
            cache,
            f"enabled {args.worldfoundry_feature_cache}",
        )
        require(
            "feature_cache.algorithm",
            cache.get("algorithm") == expected_cache_algorithm,
            cache.get("algorithm"),
            expected_cache_algorithm,
        )
        require(
            "feature_cache.hits",
            (runtime_counter(cache, "hits") or 0) > 0,
            cache.get("hits"),
            "> 0",
        )
        receipt_algorithm = (
            expected_cache_algorithm
            in WORLD_FOUNDRY_REQUEST_RECEIPT_CACHE_ALGORITHMS
        )
        if receipt_algorithm:
            request_id = cache.get("request_id")
            request_epoch = runtime_counter(cache, "request_epoch")
            require(
                "feature_cache.request_id",
                isinstance(request_id, str) and bool(request_id.strip()),
                request_id,
                "non-empty current request identity",
            )
            require(
                "feature_cache.request_epoch",
                request_epoch is not None and request_epoch > 0,
                cache.get("request_epoch"),
                "> 0",
            )
            require(
                "feature_cache.request_local",
                cache.get("request_local") is True,
                cache.get("request_local"),
                True,
            )
            require(
                "feature_cache.finalized",
                cache.get("finalized") is True
                and cache.get("release_reason") == "completed",
                {
                    "finalized": cache.get("finalized"),
                    "release_reason": cache.get("release_reason"),
                },
                "immutable completed-request receipt",
            )
            branches = cache.get("branches")
            if not isinstance(branches, Mapping):
                branches = {}
            cfg_workload = bool(str(args.negative_prompt).strip()) and not math.isclose(
                float(args.guidance),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            expected_branches = (
                {"positive", "negative"} if cfg_workload else {"positive"}
            )
            require(
                "feature_cache.branches",
                set(str(branch) for branch in branches) == expected_branches,
                sorted(str(branch) for branch in branches),
                sorted(expected_branches),
            )
        if expected_cache_algorithm in WORLD_FOUNDRY_BLOCK_FEATURE_CACHE_ALGORITHMS:
            request_epoch = runtime_counter(cache, "request_epoch")
            request_id = cache.get("request_id")
            skipped_block_calls = runtime_counter(cache, "skipped_block_calls")
            dense_block_calls = runtime_counter(cache, "dense_block_calls")
            require(
                "feature_cache.skipped_block_calls",
                skipped_block_calls is not None and skipped_block_calls > 0,
                cache.get("skipped_block_calls"),
                "> 0",
            )
            require(
                "feature_cache.dense_block_calls",
                dense_block_calls is not None and dense_block_calls > 0,
                cache.get("dense_block_calls"),
                "> 0",
            )
            branches = cache.get("branches")
            if not isinstance(branches, Mapping):
                branches = {}
            branch_skipped_total = 0
            branch_dense_total = 0
            branch_hits_total = 0
            branch_events_total = 0
            for branch, raw_branch_report in branches.items():
                branch_name = str(branch)
                branch_report = (
                    raw_branch_report
                    if isinstance(raw_branch_report, Mapping)
                    else {}
                )
                prefix = f"feature_cache.branches[{branch_name!r}]"
                require(
                    f"{prefix}.algorithm",
                    branch_report.get("algorithm") == expected_cache_algorithm,
                    branch_report.get("algorithm"),
                    expected_cache_algorithm,
                )
                require(
                    f"{prefix}.request_id",
                    branch_report.get("request_id") == request_id,
                    branch_report.get("request_id"),
                    request_id,
                )
                branch_epoch = runtime_counter(branch_report, "request_epoch")
                require(
                    f"{prefix}.request_epoch",
                    branch_epoch is not None and branch_epoch == request_epoch,
                    branch_report.get("request_epoch"),
                    request_epoch,
                )
                require(
                    f"{prefix}.request_local",
                    branch_report.get("request_local") is True,
                    branch_report.get("request_local"),
                    True,
                )
                branch_skipped = runtime_counter(
                    branch_report,
                    "skipped_block_calls",
                )
                branch_dense = runtime_counter(branch_report, "dense_block_calls")
                require(
                    f"{prefix}.skipped_block_calls",
                    branch_skipped is not None and branch_skipped > 0,
                    branch_report.get("skipped_block_calls"),
                    "> 0",
                )
                require(
                    f"{prefix}.dense_block_calls",
                    branch_dense is not None and branch_dense > 0,
                    branch_report.get("dense_block_calls"),
                    "> 0",
                )
                branch_skipped_total += branch_skipped or 0
                branch_dense_total += branch_dense or 0
                branch_hits = runtime_counter(branch_report, "hits")
                branch_events = runtime_counter(branch_report, "events")
                require(
                    f"{prefix}.hits",
                    branch_hits is not None and branch_hits > 0,
                    branch_report.get("hits"),
                    "> 0 in each CFG branch",
                )
                branch_hits_total += branch_hits or 0
                branch_events_total += branch_events or 0
                receipt = branch_report.get("receipt")
                if not isinstance(receipt, Mapping):
                    receipt = {}
                require(
                    f"{prefix}.receipt.algorithm",
                    receipt.get("algorithm") == expected_cache_algorithm,
                    receipt.get("algorithm"),
                    expected_cache_algorithm,
                )
                require(
                    f"{prefix}.receipt.branch",
                    receipt.get("branch") == branch_name,
                    receipt.get("branch"),
                    branch_name,
                )
                require(
                    f"{prefix}.receipt.request_id",
                    receipt.get("request_id") == request_id,
                    receipt.get("request_id"),
                    request_id,
                )
                receipt_epoch = runtime_counter(receipt, "request_epoch")
                require(
                    f"{prefix}.receipt.request_epoch",
                    receipt_epoch is not None and receipt_epoch == request_epoch,
                    receipt.get("request_epoch"),
                    request_epoch,
                )
                receipt_events = runtime_counter(receipt, "events")
                event_receipts = receipt.get("event_receipts")
                if not isinstance(event_receipts, (list, tuple)):
                    event_receipts = ()
                require(
                    f"{prefix}.receipt.events",
                    receipt_events is not None
                    and receipt_events == branch_events
                    and receipt_events == len(event_receipts),
                    {
                        "receipt": receipt.get("events"),
                        "branch": branch_report.get("events"),
                        "event_receipts": len(event_receipts),
                    },
                    "all event counts equal",
                )
                event_dense_total = 0
                event_skipped_total = 0
                event_hits = 0
                valid_events = True
                event_steps: list[int] = []
                expected_block_ids = set(
                    range(WAN22_TI2V_TRANSFORMER_BLOCKS)
                )
                cache_options = json.loads(
                    args.worldfoundry_feature_cache_options_json
                )
                front_blocks = int(cache_options.get("front_blocks", 5))
                back_blocks = int(cache_options.get("back_blocks", 5))
                for event in event_receipts:
                    if not isinstance(event, Mapping):
                        valid_events = False
                        continue
                    dense_blocks = event.get("dense_blocks")
                    skipped_blocks = event.get("skipped_blocks")
                    if not isinstance(dense_blocks, (list, tuple)) or not isinstance(
                        skipped_blocks,
                        (list, tuple),
                    ):
                        valid_events = False
                        continue
                    dense_count = runtime_counter(event, "dense_block_calls")
                    skipped_count = runtime_counter(event, "skipped_block_calls")
                    event_hit = event.get("hit")
                    block_ids_valid = all(
                        isinstance(block_id, int)
                        and not isinstance(block_id, bool)
                        and block_id >= 0
                        for block_id in (*dense_blocks, *skipped_blocks)
                    )
                    step_value = event.get("step")
                    dense_ids = list(dense_blocks)
                    skipped_ids = list(skipped_blocks)
                    unique_partition = (
                        len(dense_ids) == len(set(dense_ids))
                        and len(skipped_ids) == len(set(skipped_ids))
                        and not (set(dense_ids) & set(skipped_ids))
                        and set(dense_ids) | set(skipped_ids)
                        == expected_block_ids
                    )
                    layout_valid = True
                    if event_hit is True:
                        if expected_cache_algorithm == "firstblock":
                            layout_valid = (
                                dense_ids == [0]
                                and skipped_ids
                                == list(range(1, WAN22_TI2V_TRANSFORMER_BLOCKS))
                            )
                        elif expected_cache_algorithm == "dualblock":
                            expected_dense = list(range(front_blocks)) + list(
                                range(
                                    WAN22_TI2V_TRANSFORMER_BLOCKS - back_blocks,
                                    WAN22_TI2V_TRANSFORMER_BLOCKS,
                                )
                            )
                            expected_skipped = list(
                                range(
                                    front_blocks,
                                    WAN22_TI2V_TRANSFORMER_BLOCKS - back_blocks,
                                )
                            )
                            layout_valid = (
                                dense_ids == expected_dense
                                and skipped_ids == expected_skipped
                            )
                        elif expected_cache_algorithm == "blocktaylorseer":
                            layout_valid = (
                                dense_ids == []
                                and skipped_ids
                                == list(range(WAN22_TI2V_TRANSFORMER_BLOCKS))
                            )
                    else:
                        layout_valid = (
                            dense_ids
                            == list(range(WAN22_TI2V_TRANSFORMER_BLOCKS))
                            and skipped_ids == []
                        )
                    valid_events = valid_events and (
                        event.get("algorithm") == expected_cache_algorithm
                        and isinstance(step_value, int)
                        and not isinstance(step_value, bool)
                        and step_value >= 0
                        and step_value < int(args.steps)
                        and block_ids_valid
                        and unique_partition
                        and layout_valid
                        and dense_count == len(dense_blocks)
                        and skipped_count == len(skipped_blocks)
                        and isinstance(event_hit, bool)
                        and event_hit is bool(skipped_blocks)
                    )
                    if isinstance(step_value, int) and not isinstance(
                        step_value,
                        bool,
                    ):
                        event_steps.append(step_value)
                    event_dense_total += dense_count or 0
                    event_skipped_total += skipped_count or 0
                    event_hits += int(event_hit is True)
                expected_event_steps = list(range(int(args.steps)))
                valid_events = valid_events and event_steps == expected_event_steps
                if event_receipts:
                    first_event = event_receipts[0]
                    valid_events = valid_events and (
                        isinstance(first_event, Mapping)
                        and first_event.get("step") == 0
                        and first_event.get("hit") is False
                        and first_event.get("reason") == "seed"
                    )
                if expected_cache_algorithm == "blocktaylorseer":
                    dense_pattern = receipt.get("dense_pattern")
                    pattern_valid = (
                        isinstance(dense_pattern, (list, tuple))
                        and bool(dense_pattern)
                        and all(isinstance(value, bool) for value in dense_pattern)
                        and any(dense_pattern)
                    )
                    valid_events = valid_events and pattern_valid
                    if pattern_valid:
                        dense_last = int(receipt.get("dense_last", 0))
                        for event in event_receipts:
                            assert isinstance(event, Mapping)
                            event_step = int(event["step"])
                            expected_hit = not bool(
                                dense_pattern[event_step % len(dense_pattern)]
                            ) and event_step < int(args.steps) - dense_last
                            valid_events = valid_events and (
                                event.get("hit") is expected_hit
                            )
                    valid_events = valid_events and (
                        receipt.get("prediction_scope")
                        == "per-block-self-cross-ffn"
                        and runtime_counter(receipt, "taylor_order") == 1
                    )
                receipt_dense = runtime_counter(receipt, "dense_block_calls")
                receipt_skipped = runtime_counter(receipt, "skipped_block_calls")
                receipt_hits = runtime_counter(receipt, "hits")
                require(
                    f"{prefix}.receipt.event_consistency",
                    valid_events
                    and event_dense_total == receipt_dense == branch_dense
                    and event_skipped_total == receipt_skipped == branch_skipped
                    and event_hits == receipt_hits == branch_hits,
                    {
                        "event_dense": event_dense_total,
                        "receipt_dense": receipt.get("dense_block_calls"),
                        "branch_dense": branch_report.get("dense_block_calls"),
                        "event_skipped": event_skipped_total,
                        "receipt_skipped": receipt.get("skipped_block_calls"),
                        "branch_skipped": branch_report.get("skipped_block_calls"),
                        "event_hits": event_hits,
                        "receipt_hits": receipt.get("hits"),
                        "branch_hits": branch_report.get("hits"),
                    },
                    "event, receipt, and branch counters agree",
                )
            require(
                "feature_cache.branch_totals",
                branch_skipped_total == skipped_block_calls
                and branch_dense_total == dense_block_calls
                and branch_hits_total == runtime_counter(cache, "hits")
                and branch_events_total == runtime_counter(cache, "events"),
                {
                    "branch_skipped": branch_skipped_total,
                    "top_skipped": skipped_block_calls,
                    "branch_dense": branch_dense_total,
                    "top_dense": dense_block_calls,
                    "branch_hits": branch_hits_total,
                    "top_hits": cache.get("hits"),
                    "branch_events": branch_events_total,
                    "top_events": cache.get("events"),
                },
                "branch totals equal request totals for the current epoch",
            )
        elif expected_cache_algorithm == "custom":
            request_epoch = runtime_counter(cache, "request_epoch")
            request_id = cache.get("request_id")
            branches = cache.get("branches")
            if not isinstance(branches, Mapping):
                branches = {}
            branch_hits_total = 0
            branch_events_total = 0
            for branch, raw_branch_report in branches.items():
                branch_name = str(branch)
                branch_report = (
                    raw_branch_report
                    if isinstance(raw_branch_report, Mapping)
                    else {}
                )
                prefix = f"feature_cache.branches[{branch_name!r}]"
                branch_hits = runtime_counter(branch_report, "hits")
                branch_events = runtime_counter(branch_report, "events")
                branch_hits_total += branch_hits or 0
                branch_events_total += branch_events or 0
                require(
                    f"{prefix}.identity",
                    branch_report.get("algorithm") == "custom"
                    and branch_report.get("request_id") == request_id
                    and runtime_counter(branch_report, "request_epoch")
                    == request_epoch
                    and branch_report.get("request_local") is True,
                    {
                        "algorithm": branch_report.get("algorithm"),
                        "request_id": branch_report.get("request_id"),
                        "request_epoch": branch_report.get("request_epoch"),
                        "request_local": branch_report.get("request_local"),
                    },
                    "current request/branch custom cache",
                )
                require(
                    f"{prefix}.hits",
                    branch_hits is not None and branch_hits > 0,
                    branch_report.get("hits"),
                    "> 0 in each CFG branch",
                )
                receipt = branch_report.get("receipt")
                if not isinstance(receipt, Mapping):
                    receipt = {}
                event_receipts = receipt.get("event_receipts")
                if not isinstance(event_receipts, (list, tuple)):
                    event_receipts = ()
                event_steps: list[int] = []
                event_hits = 0
                valid_events = True
                for event in event_receipts:
                    if not isinstance(event, Mapping):
                        valid_events = False
                        continue
                    step_value = event.get("step")
                    event_hit = event.get("hit")
                    valid_events = valid_events and (
                        isinstance(step_value, int)
                        and not isinstance(step_value, bool)
                        and 0 <= step_value < int(args.steps)
                        and isinstance(event_hit, bool)
                        and isinstance(event.get("reason"), str)
                    )
                    if isinstance(step_value, int) and not isinstance(
                        step_value,
                        bool,
                    ):
                        event_steps.append(step_value)
                    event_hits += int(event_hit is True)
                valid_events = valid_events and event_steps == list(
                    range(int(args.steps))
                )
                if event_receipts:
                    first_event = event_receipts[0]
                    valid_events = valid_events and (
                        isinstance(first_event, Mapping)
                        and first_event.get("step") == 0
                        and first_event.get("hit") is False
                        and first_event.get("reason") == "seed"
                    )
                require(
                    f"{prefix}.receipt",
                    receipt.get("algorithm") == "custom"
                    and receipt.get("request_id") == request_id
                    and runtime_counter(receipt, "request_epoch") == request_epoch
                    and receipt.get("branch") == branch_name
                    and receipt.get("decision") == "teacache-polynomial"
                    and receipt.get("prediction")
                    == "first-order-stack-residual"
                    and runtime_counter(receipt, "events") == branch_events
                    and len(event_receipts) == branch_events
                    and runtime_counter(receipt, "hits") == branch_hits
                    and event_hits == branch_hits
                    and valid_events,
                    receipt,
                    "complete request-local Tea-decision/Taylor-prediction receipt",
                )
            require(
                "feature_cache.branch_totals",
                branch_hits_total == runtime_counter(cache, "hits")
                and branch_events_total == runtime_counter(cache, "events"),
                {
                    "branch_hits": branch_hits_total,
                    "top_hits": cache.get("hits"),
                    "branch_events": branch_events_total,
                    "top_events": cache.get("events"),
                },
                "branch totals equal current custom request totals",
            )
    if args.worldfoundry_approximate_attention != "none":
        approximate = runtime.get("approximate_attention") or {}
        if not isinstance(approximate, Mapping):
            approximate = {}
        require(
            "approximate_attention_kernel",
            effective.get("approximate_attention_kernel")
            == args.worldfoundry_approximate_attention,
            effective.get("approximate_attention_kernel"),
            args.worldfoundry_approximate_attention,
        )
        require(
            "approximate_attention.kind",
            approximate.get("kind") == args.worldfoundry_approximate_attention,
            approximate.get("kind"),
            args.worldfoundry_approximate_attention,
        )
        require(
            "approximate_attention.runtime_effective",
            approximate.get("runtime_effective") is True,
            approximate.get("runtime_effective"),
            True,
        )
        kernel_attempts = runtime_counter(approximate, "kernel_attempts")
        sparse_calls = runtime_counter(approximate, "sparse_calls")
        require(
            "approximate_attention.kernel_attempts",
            kernel_attempts is not None and kernel_attempts > 0,
            approximate.get("kernel_attempts"),
            "> 0",
        )
        require(
            "approximate_attention.sparse_calls",
            sparse_calls is not None and sparse_calls > 0,
            approximate.get("sparse_calls"),
            "> 0",
        )
        require(
            "approximate_attention.attempts_equal_sparse_calls",
            kernel_attempts is not None
            and sparse_calls is not None
            and kernel_attempts == sparse_calls,
            {"kernel_attempts": kernel_attempts, "sparse_calls": sparse_calls},
            "kernel_attempts == sparse_calls",
        )
        for counter_name in ("kernel_fallbacks", "dense_fallback_calls"):
            counter_value = runtime_counter(approximate, counter_name)
            require(
                f"approximate_attention.{counter_name}",
                counter_value == 0,
                approximate.get(counter_name),
                0,
            )
        approximate_request = _worldfoundry_approximate_attention_request(args)
        if not isinstance(approximate_request, Mapping):
            approximate_request = {"kind": args.worldfoundry_approximate_attention}
        from worldfoundry.base_models.diffusion_model.optimizations.approximate_attention import (
            parse_approximate_attention,
        )

        approximate_config = parse_approximate_attention(approximate_request)
        expected_profile = _normalized_approximate_attention_profile(
            approximate_config
        )
        lightx2v_expectation = None
        fastvideo_sla_expectation = None
        if (
            args.worldfoundry_approximate_attention
            in WORLD_FOUNDRY_LIGHTX2V_SPARSE_KINDS
        ):
            lightx2v_expectation = _lightx2v_sparse_expectation(
                approximate_config
            )
            allowed_provider_paths = frozenset(
                {lightx2v_expectation["provider_symbol"]}
            )
        elif (
            args.worldfoundry_approximate_attention
            in WORLD_FOUNDRY_FASTVIDEO_SLA_KINDS
        ):
            fastvideo_sla_expectation = _fastvideo_sla_expectation(
                approximate_config,
                fastvideo_repository=args.fastvideo_source,
                checkpoint_expectation=(
                    _fastvideo_sla_checkpoint_expectation_for_args(args)
                ),
            )
            allowed_provider_paths = frozenset(
                {fastvideo_sla_expectation["provider_path"]}
            )
        else:
            allowed_provider_paths = WORLD_FOUNDRY_APPROXIMATE_PROVIDER_PATHS[
                args.worldfoundry_approximate_attention
            ]
        provider_path = approximate.get("provider_path")
        require(
            "approximate_attention.provider_path",
            isinstance(provider_path, str) and provider_path in allowed_provider_paths,
            provider_path,
            sorted(allowed_provider_paths),
        )
        branch_names = (
            ("positive",)
            if float(args.guidance) == 1.0
            else ("positive", "negative")
        )
        receipt_audit = audit_approximate_attention_receipt(
            approximate,
            expected_kind=args.worldfoundry_approximate_attention,
            allowed_provider_paths=allowed_provider_paths,
            expected_grid=_wan22_ti2v_patch_grid(
                frames=int(args.frames),
                height=int(args.height),
                width=int(args.width),
            ),
            expected_wrapped_blocks=WAN22_TI2V_TRANSFORMER_BLOCKS,
            expected_total_steps=int(args.steps),
            expected_routed_steps=False,
            expected_branch_steps={
                branch: tuple(range(int(args.steps))) for branch in branch_names
            },
            expected_profile=expected_profile,
            expected_provider_symbols=(
                WORLD_FOUNDRY_VMOBA_PROVIDER_SYMBOLS
                if args.worldfoundry_approximate_attention == "vmoba"
                else ()
            ),
            expected_lightx2v=lightx2v_expectation,
            expected_fastvideo_sla=fastvideo_sla_expectation,
        )
        for receipt_name, receipt_check in receipt_audit["checks"].items():
            checks[f"approximate_attention.{receipt_name}"] = receipt_check
        issues.extend(
            f"approximate_attention.{issue}"
            for issue in receipt_audit["issues"]
        )
        if args.worldfoundry_approximate_attention == "vmoba":
            vmoba = approximate.get("vmoba")
            if not isinstance(vmoba, Mapping):
                vmoba = {}
            provider_symbols = vmoba.get("provider_symbols")
            require(
                "approximate_attention.vmoba.provider_symbols",
                isinstance(provider_symbols, (list, tuple))
                and tuple(provider_symbols)
                == WORLD_FOUNDRY_VMOBA_PROVIDER_SYMBOLS,
                provider_symbols,
                list(WORLD_FOUNDRY_VMOBA_PROVIDER_SYMBOLS),
            )
            chunk_calls = vmoba.get("chunk_calls")
            if not isinstance(chunk_calls, Mapping):
                chunk_calls = {}
            chunk_total = 0
            for chunk_kind in ("temporal", "spatial", "spatiotemporal"):
                count = runtime_counter(chunk_calls, chunk_kind)
                require(
                    f"approximate_attention.vmoba.chunk_calls.{chunk_kind}",
                    count is not None and count > 0,
                    chunk_calls.get(chunk_kind),
                    "> 0",
                )
                chunk_total += count or 0
            events = vmoba.get("events")
            if not isinstance(events, (list, tuple)):
                events = ()
            require(
                "approximate_attention.vmoba.events",
                bool(events),
                len(events),
                "> 0",
            )
            malformed_events = [
                index
                for index, event in enumerate(events)
                if not isinstance(event, Mapping)
                or event.get("algorithm") != "vmoba"
                or event.get("provider_path")
                != WORLD_FOUNDRY_VMOBA_PROVIDER_PATH
                or event.get("chunk_kind")
                not in {"temporal", "spatial", "spatiotemporal"}
            ]
            require(
                "approximate_attention.vmoba.event_receipts",
                not malformed_events,
                malformed_events,
                [],
            )
            require(
                "approximate_attention.vmoba.receipt_totals",
                sparse_calls is not None
                and len(events) == sparse_calls
                and chunk_total == sparse_calls,
                {
                    "events": len(events),
                    "chunk_total": chunk_total,
                    "sparse_calls": sparse_calls,
                },
                "events == chunk_total == sparse_calls",
            )
    if args.worldfoundry_quantization != "none":
        quantization = runtime.get("quantization") or {}
        if not isinstance(quantization, Mapping):
            quantization = {}
        kernel_calls = runtime_counter(quantization, "low_precision_kernel_calls")
        require(
            "quantization.low_precision_kernel_calls",
            kernel_calls is not None and kernel_calls > 0,
            kernel_calls,
            "> 0",
        )
        for counter_name in (
            "packed_weight_calls",
            "dense_policy_calls",
            "dense_fallback_calls",
        ):
            counter_value = runtime_counter(quantization, counter_name)
            require(
                f"quantization.{counter_name}",
                counter_value == 0,
                counter_value,
                0,
            )
        fallback_reasons = quantization.get("fallback_reasons")
        require(
            "quantization.fallback_reasons",
            isinstance(fallback_reasons, (list, tuple)) and not fallback_reasons,
            fallback_reasons,
            [],
        )

    verification = {"passed": not issues, "checks": checks, "issues": issues}
    if issues and bool(getattr(args, "worldfoundry_require_effective", True)):
        raise RuntimeError(
            "WorldFoundry optimization effectiveness gate failed: " + "; ".join(issues)
        )
    return verification


def _worldfoundry_approximate_attention_request(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Resolve a sparse-attention request without relying on hidden defaults."""

    selected = str(args.worldfoundry_approximate_attention).strip().casefold()
    inline_profile = args.worldfoundry_approximate_attention_profile_json
    profile_path = args.worldfoundry_approximate_attention_profile
    if inline_profile is not None and profile_path is not None:
        raise ValueError(
            "use only one of --worldfoundry-approximate-attention-profile-json "
            "and --worldfoundry-approximate-attention-profile"
        )

    profile: dict[str, Any] | None = None
    profile_source = ""
    if profile_path is not None:
        resolved = profile_path.expanduser()
        if not resolved.is_file():
            raise FileNotFoundError(profile_path)
        profile_source = f"approximate-attention profile {resolved}"
        raw_profile = resolved.read_text(encoding="utf-8")
    elif inline_profile is not None:
        profile_source = "--worldfoundry-approximate-attention-profile-json"
        raw_profile = inline_profile
    else:
        raw_profile = None
    if raw_profile is not None:
        try:
            decoded = json.loads(raw_profile)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{profile_source} must contain valid JSON: {error}"
            ) from error
        if not isinstance(decoded, dict):
            raise TypeError(f"{profile_source} must decode to an object")
        profile = {str(key): value for key, value in decoded.items()}

    if selected == "none":
        if profile is not None:
            raise ValueError(
                "an approximate-attention profile requires "
                "--worldfoundry-approximate-attention"
            )
        return None

    profile = {} if profile is None else profile
    configured_kind = profile.pop("kind", selected)
    if str(configured_kind).strip().casefold() != selected:
        raise ValueError(
            "approximate-attention profile kind conflicts with "
            f"--worldfoundry-approximate-attention={selected!r}"
        )
    common_fields = {"sparsity", "dense_steps"}
    if selected == "sta":
        allowed_fields = common_fields | {"window"}
    elif selected == "vsa":
        allowed_fields = common_fields | {"block_tile"}
    elif selected == "vmoba":
        allowed_fields = common_fields | set(
            WORLD_FOUNDRY_VMOBA_PROFILE_FIELDS
        )
    elif selected in WORLD_FOUNDRY_FASTVIDEO_SLA_KINDS:
        allowed_fields = {"dense_steps"} | set(
            WORLD_FOUNDRY_FASTVIDEO_SLA_PROFILE_FIELDS
        )
    else:
        allowed_fields = common_fields | set(
            WORLD_FOUNDRY_LIGHTX2V_PROFILE_FIELDS
        )
    unknown = sorted(set(profile) - allowed_fields)
    if unknown:
        raise ValueError(f"unknown approximate-attention profile fields: {unknown}")
    if selected == "vmoba":
        missing = sorted(WORLD_FOUNDRY_VMOBA_PROFILE_FIELDS - set(profile))
        if missing:
            raise ValueError(
                "VMoBA benchmarking requires a complete grid-specific profile; "
                f"missing fields: {missing}"
            )
        request = {"kind": selected, **profile}
    else:
        request: dict[str, Any] = {
            "kind": selected,
            "sparsity": args.worldfoundry_sparse_sparsity,
            "dense_steps": args.worldfoundry_sparse_dense_steps,
        }
        if selected == "sta":
            request["window"] = tuple(args.worldfoundry_sta_window)
        elif selected == "vsa":
            request["block_tile"] = tuple(args.worldfoundry_vsa_block_tile)
        elif selected in WORLD_FOUNDRY_FASTVIDEO_SLA_KINDS:
            contract = FASTVIDEO_SLA_REFERENCE_CONTRACTS[selected]
            request.update(
                fastvideo_topk_ratio=contract["topk_ratio"],
                fastvideo_feature_map="softmax",
            )
        request.update(profile)

    from worldfoundry.base_models.diffusion_model.optimizations.approximate_attention import (
        parse_approximate_attention,
    )

    config = parse_approximate_attention(request)
    grid = _wan22_ti2v_patch_grid(
        frames=int(args.frames),
        height=int(args.height),
        width=int(args.width),
    )
    if selected == "vmoba":
        _validate_vmoba_benchmark_profile(
            config,
            grid=grid,
            steps=int(args.steps),
            wrapped_blocks=WAN22_TI2V_TRANSFORMER_BLOCKS,
        )
    if selected == "sta" and grid not in WORLD_FOUNDRY_STA_SUPPORTED_GRIDS:
        raise ValueError(
            f"STA has no tuned provider plan for post-patch grid {grid}; "
            f"supported grids are {sorted(WORLD_FOUNDRY_STA_SUPPORTED_GRIDS)!r}"
        )
    if (
        selected in WORLD_FOUNDRY_LIGHTX2V_SPARSE_KINDS
        and config.attnmap_frame_num is not None
        and int(config.attnmap_frame_num) != grid[0]
    ):
        raise ValueError(
            "LightX2V attnmap_frame_num must equal the post-patch temporal grid; "
            f"got {config.attnmap_frame_num}, expected {grid[0]}"
        )
    if selected in WORLD_FOUNDRY_FASTVIDEO_SLA_KINDS:
        contract = FASTVIDEO_SLA_REFERENCE_CONTRACTS[selected]
        if (
            config.fastvideo_topk_ratio != contract["topk_ratio"]
            or config.fastvideo_feature_map != "softmax"
            or int(config.dense_steps) != 0
        ):
            raise ValueError(
                f"{selected} framework A/B requires the pinned FastVideo runtime "
                f"profile topk_ratio={contract['topk_ratio']}, "
                "feature_map='softmax', dense_steps=0"
            )
    return request if bool(args.worldfoundry_optimized) else None


def _validate_fastvideo_sla_pairing_request(args: argparse.Namespace) -> None:
    """Keep the two learned-SLA lanes on the same explicit upstream backend."""

    selected = str(args.worldfoundry_approximate_attention).strip().casefold()
    backend = str(args.fastvideo_attention_backend).strip().upper()
    if selected in WORLD_FOUNDRY_FASTVIDEO_SLA_KINDS:
        expected = str(FASTVIDEO_SLA_REFERENCE_CONTRACTS[selected]["backend"])
        if backend != expected:
            raise ValueError(
                f"{selected} requires --fastvideo-attention-backend={expected}; "
                f"got {backend}"
            )
        _fastvideo_sla_source_identity(args.fastvideo_source)
        _fastvideo_sla_checkpoint_expectation_for_args(args)
    elif backend in _FASTVIDEO_SLA_BACKEND_CONTRACTS:
        raise ValueError(
            f"--fastvideo-attention-backend={backend} requires matching "
            "--worldfoundry-approximate-attention=fastvideo_sla or "
            "fastvideo_sagesla"
        )


def _validate_worldfoundry_kernel_request(args: argparse.Namespace) -> None:
    """Reject internally inconsistent Wan kernel controls before loading."""

    if int(args.worldfoundry_qkv_split_threshold) <= 0:
        raise ValueError("--worldfoundry-qkv-split-threshold must be positive")
    if not bool(args.worldfoundry_fused_rope):
        return
    if str(args.worldfoundry_rope_precision) != "fp32":
        raise ValueError(
            "--worldfoundry-fused-rope requires "
            "--worldfoundry-rope-precision=fp32"
        )
    if str(args.worldfoundry_rms_norm_precision) != "fp32":
        raise ValueError(
            "--worldfoundry-fused-rope requires "
            "--worldfoundry-rms-norm-precision=fp32"
        )


def _worldfoundry_feature_cache_request(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Build the generic, auditable feature-cache option for one run."""

    options = json.loads(args.worldfoundry_feature_cache_options_json)
    if not isinstance(options, dict):
        raise TypeError("--worldfoundry-feature-cache-options-json must decode to an object")
    if "algorithm" in options or "kind" in options:
        raise ValueError(
            "feature-cache JSON must not override --worldfoundry-feature-cache"
        )
    selected = str(args.worldfoundry_feature_cache)
    legacy_threshold = args.worldfoundry_teacache_threshold
    if not bool(args.worldfoundry_optimized):
        return None
    if selected == "none":
        if legacy_threshold is None:
            if options:
                raise ValueError(
                    "--worldfoundry-feature-cache-options-json requires a selected cache"
                )
            return None
        selected = "teacache"
    elif legacy_threshold is not None and selected != "teacache":
        raise ValueError(
            "--worldfoundry-teacache-threshold requires "
            "--worldfoundry-feature-cache=teacache"
        )
    if legacy_threshold is not None:
        options["threshold"] = legacy_threshold
    options["algorithm"] = selected
    return options


def _run_worldfoundry(args: argparse.Namespace, output_path: Path) -> dict[str, Any]:
    _validate_worldfoundry_kernel_request(args)
    import_started = time.perf_counter()
    import torch

    torch.cuda.set_device(0)
    attention_backend = (
        args.worldfoundry_attention_backend
        if args.worldfoundry_optimized
        else "torch"
    )
    backend_capability = _require_backend_capability(
        "worldfoundry",
        attention_backend,
        cuda_capability=tuple(torch.cuda.get_device_capability(0)),
    )
    from worldfoundry.core.attention import reset_attention_provider_runtime
    from worldfoundry.pipelines.wan.pipeline_wan_2p2 import Wan2p2Pipeline

    import_s = time.perf_counter() - import_started

    vae_decode_autocast = (
        None
        if not args.worldfoundry_optimized or args.worldfoundry_vae_decode_autocast == "none"
        else args.worldfoundry_vae_decode_autocast
    )
    dit_weight_dtype = args.worldfoundry_dit_weight_dtype if args.worldfoundry_optimized else "fp32"
    vae_weight_dtype = (
        args.worldfoundry_vae_weight_dtype or "fp32"
        if args.worldfoundry_optimized
        else "fp32"
    )
    vae_channels_last_3d = bool(
        args.worldfoundry_optimized
        and args.worldfoundry_vae_channels_last_3d
    )
    approximate_attention = _worldfoundry_approximate_attention_request(args)
    compile_options = json.loads(args.worldfoundry_compile_options_json)
    if not isinstance(compile_options, dict):
        raise TypeError("--worldfoundry-compile-options-json must decode to an object")
    feature_cache_request = _worldfoundry_feature_cache_request(args)
    quantization_config = json.loads(args.worldfoundry_quantization_config_json)
    if not isinstance(quantization_config, dict):
        raise TypeError("--worldfoundry-quantization-config-json must decode to an object")
    quantization_request: str | dict[str, Any] = "none"
    if args.worldfoundry_optimized and args.worldfoundry_quantization != "none":
        quantization_request = {
            "mode": args.worldfoundry_quantization,
            **quantization_config,
        }
    if args.worldfoundry_cuda_graph and args.worldfoundry_static_cross_kv and args.worldfoundry_optimized:
        raise ValueError(
            "--worldfoundry-cuda-graph conflicts with the default static cross-KV cache; "
            "pass --no-worldfoundry-static-cross-kv"
        )
    optimization_options = {
        "fuse_qkv": args.worldfoundry_optimized and args.worldfoundry_fuse_qkv,
        "qkv_strategy": args.worldfoundry_qkv_strategy,
        "qkv_split_threshold": args.worldfoundry_qkv_split_threshold,
        "inplace_residual": (
            args.worldfoundry_optimized
            and args.worldfoundry_inplace_residual
        ),
        "static_cross_kv": args.worldfoundry_optimized and args.worldfoundry_static_cross_kv,
        "fused_rope": args.worldfoundry_optimized and args.worldfoundry_fused_rope,
        "rope_precision": args.worldfoundry_rope_precision,
        "rms_norm_precision": args.worldfoundry_rms_norm_precision,
        "cuda_graph": args.worldfoundry_optimized and args.worldfoundry_cuda_graph,
        "approximate_attention": approximate_attention,
    }
    if feature_cache_request is not None:
        optimization_options["feature_cache"] = feature_cache_request
    if args.worldfoundry_optimized and args.worldfoundry_compile:
        optimization_options.update(
            compile_backend=args.worldfoundry_compile_backend,
            compile_fullgraph=args.worldfoundry_compile_fullgraph,
            compile_dynamic=args.worldfoundry_compile_dynamic,
        )
        if args.worldfoundry_compile_mode != "none":
            optimization_options["compile_mode"] = args.worldfoundry_compile_mode
        if compile_options:
            optimization_options["compile_options"] = compile_options
    checkpoint = args.worldfoundry_checkpoint or args.checkpoint
    load_started = time.perf_counter()
    pipeline = Wan2p2Pipeline.from_pretrained(
        model_path=str(checkpoint),
        checkpoint_overrides={role: str(checkpoint) for role in ("dit", "text-encoder", "tokenizer", "vae")},
        device="cuda",
        torch_dtype="bfloat16",
        offload_mode="none",
        attention_backend=attention_backend,
        quantization=quantization_request,
        vae_decode_autocast=vae_decode_autocast,
        vae_weight_dtype=vae_weight_dtype,
        vae_channels_last_3d=vae_channels_last_3d,
        dit_weight_dtype=dit_weight_dtype,
        torch_compile=args.worldfoundry_optimized and args.worldfoundry_compile,
        **optimization_options,
    )
    torch.cuda.synchronize()
    load_s = time.perf_counter() - load_started
    baseline = _gpu_query(args.physical_gpu)
    isolation_monitor = getattr(args, "_compute_app_monitor", None)
    owns_isolation_monitor = isolation_monitor is None
    generation_isolation = None
    if owns_isolation_monitor:
        generation_compute_apps_before = _gpu_compute_apps(args.physical_gpu)
        isolation_monitor = ComputeAppMonitor(
            args.physical_gpu,
            generation_compute_apps_before,
        )
        isolation_monitor.start()
    reset_attention_provider_runtime()
    torch.cuda.reset_peak_memory_stats()
    sampler = GpuSampler(args.physical_gpu)
    sampler.start()
    try:
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            fps=args.fps,
            output_path=output_path,
            return_dict=True,
        )
        torch.cuda.synchronize()
        generation_s = time.perf_counter() - started
    finally:
        sampler.stop()
        if owns_isolation_monitor:
            generation_isolation = isolation_monitor.stop()
    actual_output = Path(result["artifact_path"] or output_path).resolve()
    optimization_report = _worldfoundry_optimization_report(pipeline)
    optimization_report["backend_capability"] = backend_capability
    optimization_verification = _verify_worldfoundry_optimization_report(
        optimization_report,
        args,
    )
    return {
        "framework_import_s": import_s,
        "model_load_s": load_s,
        "generation_s": generation_s,
        "output_path": str(actual_output),
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "gpu_at_generation_start": baseline,
        "gpu_generation": sampler.summary(),
        "generation_isolation": generation_isolation,
        "optimization_report": optimization_report,
        "optimization_verification": optimization_verification,
        "framework_module": str(Path(sys.modules["worldfoundry"].__file__).resolve()),
    }


def _configure_fastvideo_attention_environment(args: argparse.Namespace) -> dict[str, Any]:
    """Set the process-wide backend before any FastVideo module can be imported."""

    requested = str(args.fastvideo_attention_backend).strip().upper()
    imported_modules = sorted(
        name
        for name in sys.modules
        if name == "fastvideo" or name.startswith("fastvideo.")
    )
    if imported_modules:
        raise RuntimeError(
            "FASTVIDEO_ATTENTION_BACKEND must be configured before importing "
            f"FastVideo; already imported modules: {imported_modules[:8]!r}"
        )
    ambient = os.environ.get("FASTVIDEO_ATTENTION_BACKEND")
    if ambient is not None and ambient.strip().upper() != requested:
        raise RuntimeError(
            "ambient FASTVIDEO_ATTENTION_BACKEND conflicts with the benchmark "
            f"request: ambient={ambient!r}, requested={requested!r}"
        )
    os.environ["FASTVIDEO_ATTENTION_BACKEND"] = requested
    return {
        "name": "FASTVIDEO_ATTENTION_BACKEND",
        "value": requested,
        "configured_before_import": True,
        "ambient_value": ambient,
        "preexisting_fastvideo_modules": imported_modules,
    }


def _run_fastvideo(args: argparse.Namespace, output_path: Path) -> dict[str, Any]:
    expected_projection = (
        _fastvideo_sla_checkpoint_expectation_for_args(args)
        if str(args.fastvideo_attention_backend).strip().upper()
        in _FASTVIDEO_SLA_BACKEND_CONTRACTS
        else None
    )
    backend_environment = _configure_fastvideo_attention_environment(args)
    import_started = time.perf_counter()
    import torch

    torch.cuda.set_device(0)
    backend_capability = _require_backend_capability(
        "fastvideo",
        args.fastvideo_attention_backend,
        cuda_capability=tuple(torch.cuda.get_device_capability(0)),
    )
    from fastvideo import VideoGenerator

    import_s = time.perf_counter() - import_started

    load_started = time.perf_counter()
    checkpoint = args.fastvideo_checkpoint or args.checkpoint
    generator = VideoGenerator.from_pretrained(
        str(checkpoint),
        num_gpus=1,
        use_fsdp_inference=False,
        dit_cpu_offload=False,
        vae_cpu_offload=False,
        text_encoder_cpu_offload=False,
        pin_cpu_memory=True,
    )
    # Match the WorldFoundry and LightX2V boundaries exactly: asynchronous
    # model-init work must finish before load timing stops, and no queued work
    # may leak into or out of the public generation interval.
    torch.cuda.synchronize()
    load_s = time.perf_counter() - load_started
    try:
        installed_probe_records = generator.executor.collective_rpc(
            _install_fastvideo_reference_probe
        )
        _verify_fastvideo_reference_probe(
            installed_probe_records,
            require_successful_calls=False,
            expected_projection=expected_projection,
        )
        baseline = _gpu_query(args.physical_gpu)
        isolation_monitor = getattr(args, "_compute_app_monitor", None)
        owns_isolation_monitor = isolation_monitor is None
        generation_isolation = None
        if owns_isolation_monitor:
            generation_compute_apps_before = _gpu_compute_apps(args.physical_gpu)
            isolation_monitor = ComputeAppMonitor(
                args.physical_gpu,
                generation_compute_apps_before,
            )
            isolation_monitor.start()
        sampler = GpuSampler(args.physical_gpu)
        sampler.start()
        try:
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = generator.generate_video(
                args.prompt,
                negative_prompt=args.negative_prompt,
                output_path=str(output_path),
                save_video=True,
                return_frames=False,
                height=args.height,
                width=args.width,
                num_frames=args.frames,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                seed=args.seed,
                fps=args.fps,
            )
            torch.cuda.synchronize()
            generation_s = time.perf_counter() - started
        finally:
            sampler.stop()
            if owns_isolation_monitor:
                generation_isolation = isolation_monitor.stop()
        reference_probe_records = generator.executor.collective_rpc(
            _collect_fastvideo_reference_probe
        )
        optimization_verification = _verify_fastvideo_reference_probe(
            reference_probe_records,
            require_successful_calls=True,
            expected_projection=expected_projection,
        )
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            generator.shutdown()
        except Exception:
            # Preserve the generation failure when cleanup also fails.  A
            # cleanup-only failure remains fatal because it can strand a GPU
            # worker and invalidate the next sample.
            if not active_error:
                raise
    if isinstance(result, list):
        if len(result) != 1:
            raise RuntimeError(f"FastVideo returned {len(result)} results for one prompt")
        result = result[0]
    actual_output = Path(result.get("video_path") or output_path).resolve()
    resolved_backends = sorted(
        {
            backend
            for worker in reference_probe_records
            for role in (worker.get("roles") or {}).values()
            for backend in role.get("backends") or []
        }
    )
    attention_is_approximate = any(
        "SAGE" in backend
        or backend in {"SLA_ATTN", "SAGE_SLA_ATTN", "VIDEO_SPARSE_ATTN", "VMOBA_ATTN"}
        for backend in resolved_backends
    )
    return {
        "framework_import_s": import_s,
        "model_load_s": load_s,
        "generation_s": generation_s,
        "framework_reported_generation_s": result.get("generation_time"),
        "framework_reported_e2e_s": result.get("e2e_latency"),
        "framework_reported_peak_memory_mib": result.get("peak_memory_mb"),
        "output_path": str(actual_output),
        "gpu_at_generation_start": baseline,
        "gpu_generation": sampler.summary(),
        "generation_isolation": generation_isolation,
        "optimization_verification": optimization_verification,
        "optimization_report": {
            "engine": "FastVideo public Wan2.2 TI2V preset",
            "offload": "none",
            "fsdp": False,
            "requested": {
                "attention_backend_cli": args.fastvideo_attention_backend,
                "attention_backend_environment": backend_environment,
                "attention_backend": sorted(
                    {
                        str(worker.get("requested_backend"))
                        for worker in reference_probe_records
                    }
                ),
            },
            "backend_capability": backend_capability,
            "effective": {
                "attention_backends": resolved_backends,
                "attention_roles": [
                    {
                        "worker": worker_index,
                        "role": role,
                        "backends": role_report.get("backends"),
                        "implementation_classes": role_report.get(
                            "implementation_classes"
                        ),
                    }
                    for worker_index, worker in enumerate(reference_probe_records)
                    for role, role_report in (worker.get("roles") or {}).items()
                ],
            },
            "effectiveness_proof": (
                "worker collective_rpc identities plus one-shot successful forward hooks"
            ),
            "runtime_probe": reference_probe_records,
            "effectiveness_verification": optimization_verification,
            "fallbacks": list(optimization_verification["fallbacks"]),
            "quality_tier": (
                "numerically-approximate" if attention_is_approximate else "exact"
            ),
        },
        "framework_module": str(Path(sys.modules["fastvideo"].__file__).resolve()),
    }


def _lightx2v_vae_runtime_report(pipeline: Any) -> dict[str, Any]:
    """Inspect the resident upstream VAE rather than trusting its env flag."""

    import torch

    runner = getattr(pipeline, "runner", None)
    decoder = getattr(runner, "vae_decoder", None)
    model = getattr(decoder, "model", decoder)
    if not isinstance(model, torch.nn.Module):
        return {
            "available": False,
            "weight_dtypes": {},
            "conv3d_count": 0,
            "channels_last_3d_conv3d": 0,
        }
    weight_dtypes: dict[str, int] = {}
    for parameter in model.parameters():
        dtype_name = str(parameter.dtype).removeprefix("torch.")
        weight_dtypes[dtype_name] = (
            weight_dtypes.get(dtype_name, 0) + parameter.numel()
        )
    conv3d_weights = [
        child.weight
        for child in model.modules()
        if isinstance(child, torch.nn.Conv3d)
    ]
    return {
        "available": True,
        "weight_dtypes": weight_dtypes,
        "conv3d_count": len(conv3d_weights),
        "channels_last_3d_conv3d": sum(
            weight.is_contiguous(memory_format=torch.channels_last_3d)
            for weight in conv3d_weights
        ),
    }


def _lightx2v_wan_math_runtime_report(pipeline: Any) -> dict[str, Any]:
    """Inspect Wan RoPE/RMS implementations instead of trusting config flags."""

    runner = getattr(pipeline, "runner", None)
    model = getattr(runner, "model", None)
    weights = getattr(model, "transformer_weights", None)
    blocks = getattr(weights, "blocks", None)
    try:
        phase = blocks[0].compute_phases[0]
    except (AttributeError, IndexError, KeyError, TypeError):
        return {
            "available": False,
            "reason": "first Wan self-attention phase is unavailable",
        }

    rope = getattr(phase, "rope", None)
    seen: set[int] = set()
    while rope is not None and not hasattr(rope, "compute_dtype"):
        if id(rope) in seen:
            break
        seen.add(id(rope))
        rope = getattr(rope, "inner", None)
    rope_dtype = getattr(rope, "compute_dtype", None)

    norm_reports: dict[str, dict[str, Any]] = {}
    for role, attribute in (
        ("q", "self_attn_norm_q"),
        ("k", "self_attn_norm_k"),
    ):
        norm = getattr(phase, attribute, None)
        infer_dtype = getattr(norm, "infer_dtype", None)
        sensitive_dtype = getattr(norm, "sensitive_layer_dtype", None)
        implementation_module = sys.modules.get(
            type(norm).__module__ if norm is not None else ""
        )
        norm_reports[role] = {
            "available": norm is not None,
            "implementation_class": (
                f"{type(norm).__module__}.{type(norm).__qualname__}"
                if norm is not None
                else None
            ),
            "infer_dtype": (
                str(infer_dtype).removeprefix("torch.")
                if infer_dtype is not None
                else None
            ),
            "sensitive_layer_dtype": (
                str(sensitive_dtype).removeprefix("torch.")
                if sensitive_dtype is not None
                else None
            ),
            "compute_mode": (
                "input"
                if infer_dtype is not None and sensitive_dtype == infer_dtype
                else "fp32"
            ),
            "sgl_kernel_available": bool(
                getattr(implementation_module, "sgl_kernel", None)
            ),
        }
    available = rope_dtype is not None and all(
        report["available"] for report in norm_reports.values()
    )
    return {
        "available": available,
        "rope_implementation_class": (
            f"{type(rope).__module__}.{type(rope).__qualname__}"
            if rope is not None
            else None
        ),
        "rope_compute_dtype": (
            str(rope_dtype).removeprefix("torch.")
            if rope_dtype is not None
            else None
        ),
        "attn_rms_norm_type": getattr(phase, "attn_rms_norm_type", None),
        "rms_norm": norm_reports,
    }


def _run_lightx2v(args: argparse.Namespace, output_path: Path) -> dict[str, Any]:
    """Run LightX2V through its public pipeline with the shared semantic inputs."""

    attention_request = _lightx2v_attention_request(args)
    shared_vae_dtype = getattr(args, "worldfoundry_vae_weight_dtype", None)
    shared_channels_last_3d = bool(
        getattr(args, "worldfoundry_vae_channels_last_3d", False)
    )
    if shared_vae_dtype is not None:
        expected_env_dtype = {
            "bf16": "BF16",
            "fp16": "FP16",
            "fp32": "FP32",
        }[shared_vae_dtype]
        if os.getenv("DTYPE", "BF16").upper() != expected_env_dtype:
            raise RuntimeError(
                "LightX2V DTYPE does not match the shared VAE policy: "
                f"actual={os.getenv('DTYPE', 'BF16')!r}, "
                f"expected={expected_env_dtype!r}"
            )
    if shared_channels_last_3d and os.getenv("USE_CHANNELS_LAST_3D", "0") != "1":
        raise RuntimeError(
            "LightX2V shared VAE policy requested channels_last_3d but "
            "USE_CHANNELS_LAST_3D is not 1"
        )
    import_started = time.perf_counter()
    import torch

    torch.cuda.set_device(0)
    cuda_capability = tuple(torch.cuda.get_device_capability(0))
    self_backend_capability = _require_backend_capability(
        "lightx2v",
        attention_request["self_mode"],
        cuda_capability=cuda_capability,
    )
    cross_backend_capability = _require_backend_capability(
        "lightx2v",
        attention_request["cross_mode"],
        cuda_capability=cuda_capability,
    )
    backend_capability = {
        "schema_version": "worldfoundry-lightx2v-role-capability-v1",
        "certification_role": "diagnostic-only",
        "passed": (
            self_backend_capability["passed"] is True
            and cross_backend_capability["passed"] is True
        ),
        "roles": {
            "self": self_backend_capability,
            "cross1": cross_backend_capability,
            "cross2": cross_backend_capability,
        },
        "runtime_proof_required": True,
        "runtime_proof": (
            "role-specific provider identities, observed sparse mask, and "
            "successful provider/apply calls"
        ),
    }
    from lightx2v import LightX2VPipeline

    import_s = time.perf_counter() - import_started
    checkpoint = args.lightx2v_checkpoint or args.checkpoint
    load_started = time.perf_counter()
    pipeline = LightX2VPipeline(
        model_path=str(checkpoint),
        model_cls="wan2.2",
        task="t2v",
    )
    pipeline.enable_offload(
        cpu_offload=False,
        text_encoder_offload=False,
        image_encoder_offload=False,
        vae_offload=False,
    )
    # Current LightX2V classifies target_video_length as request input and
    # filters it out of create_generator(**kwargs), while WanScheduler still
    # requires it in the runner config. Its own Wan2.2 example therefore uses
    # the JSON path. Emit the same public config contract so this benchmark is
    # compatible with both the released API and the current checkout.
    generator_config = {
        "infer_steps": int(args.steps),
        "target_video_length": int(args.frames),
        "text_len": 512,
        "target_height": int(args.height),
        "target_width": int(args.width),
        "num_channels_latents": 48,
        "vae_stride": [4, 16, 16],
        "self_attn_1_type": attention_request["self_mode"],
        "cross_attn_1_type": attention_request["cross_mode"],
        "cross_attn_2_type": attention_request["cross_mode"],
        "sample_guide_scale": float(args.guidance),
        "sample_shift": float(args.lightx2v_sample_shift),
        "enable_cfg": float(args.guidance) != 1.0,
        "cpu_offload": False,
        "offload_granularity": "model",
        "t5_cpu_offload": False,
        "vae_cpu_offload": False,
        "fps": int(args.fps),
        "rope_type": args.lightx2v_rope_type,
        "double_precision_rope": bool(args.lightx2v_double_precision_rope),
    }
    if attention_request["self_setting_key"] is not None:
        generator_config[attention_request["self_setting_key"]] = dict(
            attention_request["self_settings"]
        )
    generator_config_path = output_path.with_suffix(".lightx2v-config.json")
    _write_json(generator_config_path, generator_config)
    # LightX2V's create_generator(config_json=...) reads the file into pipeline
    # attributes but does not retain the path. Its following set_config() only
    # replays request-filtered attributes unless config_json is also bound.
    pipeline.config_json = str(generator_config_path)
    pipeline.create_generator(config_json=str(generator_config_path))
    configured_self_attention = getattr(pipeline, "self_attn_1_type", None)
    configured_cross1_attention = getattr(pipeline, "cross_attn_1_type", None)
    configured_cross2_attention = getattr(pipeline, "cross_attn_2_type", None)
    configured_rope = getattr(pipeline, "rope_type", None)
    configured_double_rope = getattr(pipeline, "double_precision_rope", None)
    lightx_configuration_issues = []
    if configured_self_attention != attention_request["self_mode"]:
        lightx_configuration_issues.append(
            "self_attn_1_type="
            f"{configured_self_attention!r}, expected {attention_request['self_mode']!r}"
        )
    if configured_cross1_attention != attention_request["cross_mode"]:
        lightx_configuration_issues.append(
            "cross_attn_1_type="
            f"{configured_cross1_attention!r}, expected {attention_request['cross_mode']!r}"
        )
    if configured_cross2_attention != attention_request["cross_mode"]:
        lightx_configuration_issues.append(
            "cross_attn_2_type="
            f"{configured_cross2_attention!r}, expected {attention_request['cross_mode']!r}"
        )
    if configured_rope != args.lightx2v_rope_type:
        lightx_configuration_issues.append(
            f"rope={configured_rope!r}, expected {args.lightx2v_rope_type!r}"
        )
    if bool(configured_double_rope) is not bool(args.lightx2v_double_precision_rope):
        lightx_configuration_issues.append(
            "double_precision_rope="
            f"{configured_double_rope!r}, expected {args.lightx2v_double_precision_rope!r}"
        )
    if lightx_configuration_issues:
        raise RuntimeError(
            "LightX2V public pipeline did not consume the requested optimization config: "
            + "; ".join(lightx_configuration_issues)
        )
    math_runtime_report = _lightx2v_wan_math_runtime_report(pipeline)
    if not math_runtime_report.get("available"):
        raise RuntimeError(
            "LightX2V Wan math implementation could not be inspected: "
            f"{math_runtime_report}"
        )
    if math_runtime_report.get("rope_compute_dtype") != "float32":
        raise RuntimeError(
            "LightX2V Wan effective RoPE precision is not fp32: "
            f"{math_runtime_report.get('rope_compute_dtype')!r}"
        )
    lightx_rms_modes = {
        report.get("compute_mode")
        for report in (math_runtime_report.get("rms_norm") or {}).values()
        if isinstance(report, Mapping)
    }
    if lightx_rms_modes != {"input"}:
        raise RuntimeError(
            "LightX2V Wan effective RMSNorm mode is not input precision: "
            f"{sorted(str(mode) for mode in lightx_rms_modes)!r}"
        )
    torch.cuda.synchronize()
    load_s = time.perf_counter() - load_started
    installed_reference_probe = _install_lightx2v_reference_probe(
        pipeline,
        attention_request,
    )
    _verify_lightx2v_reference_probe(
        installed_reference_probe,
        require_successful_calls=False,
    )
    baseline = _gpu_query(args.physical_gpu)
    isolation_monitor = getattr(args, "_compute_app_monitor", None)
    owns_isolation_monitor = isolation_monitor is None
    generation_isolation = None
    if owns_isolation_monitor:
        generation_compute_apps_before = _gpu_compute_apps(args.physical_gpu)
        isolation_monitor = ComputeAppMonitor(
            args.physical_gpu,
            generation_compute_apps_before,
        )
        isolation_monitor.start()
    torch.cuda.reset_peak_memory_stats()
    sampler = GpuSampler(args.physical_gpu)
    sampler.start()
    try:
        torch.cuda.synchronize()
        started = time.perf_counter()
        pipeline.generate(
            seed=args.seed,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            save_result_path=str(output_path),
            return_result_tensor=False,
        )
        torch.cuda.synchronize()
        generation_s = time.perf_counter() - started
    finally:
        try:
            sampler.stop()
            if owns_isolation_monitor:
                generation_isolation = isolation_monitor.stop()
        finally:
            reference_probe = _collect_lightx2v_reference_probe(pipeline)
    optimization_verification = _verify_lightx2v_reference_probe(
        reference_probe,
        require_successful_calls=True,
    )
    vae_runtime_report = _lightx2v_vae_runtime_report(pipeline)
    if shared_vae_dtype is not None:
        expected_dtype = {
            "bf16": "bfloat16",
            "fp16": "float16",
            "fp32": "float32",
        }[shared_vae_dtype]
        dtype_keys = set(vae_runtime_report["weight_dtypes"])
        if not vae_runtime_report["available"] or dtype_keys != {expected_dtype}:
            raise RuntimeError(
                "LightX2V resident VAE dtype proof failed: "
                f"actual={sorted(dtype_keys)!r}, expected={[expected_dtype]!r}"
            )
    if shared_channels_last_3d:
        converted = int(vae_runtime_report["channels_last_3d_conv3d"])
        total = int(vae_runtime_report["conv3d_count"])
        if total <= 0 or converted != total:
            raise RuntimeError(
                "LightX2V channels_last_3d proof failed: "
                f"effective={converted}, total={total}"
            )
    actual_output = output_path.resolve()
    attention_is_approximate = (
        attention_request["self_family"] == "sparse"
        or "sage" in attention_request["self_mode"]
        or "sage" in attention_request["cross_mode"]
    )
    return {
        "framework_import_s": import_s,
        "model_load_s": load_s,
        "generation_s": generation_s,
        "output_path": str(actual_output),
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "gpu_at_generation_start": baseline,
        "gpu_generation": sampler.summary(),
        "generation_isolation": generation_isolation,
        "optimization_verification": optimization_verification,
        "optimization_report": {
            "engine": "LightX2V public LightX2VPipeline",
            "requested": {
                "attention_mode": attention_request["legacy_attention_mode"],
                "self_attention_mode": attention_request["self_mode"],
                "cross_attention_mode": attention_request["cross_mode"],
                "self_attention_family": attention_request["self_family"],
                "self_attention_settings": attention_request["self_settings"],
                "rope_type": args.lightx2v_rope_type,
                "double_precision_rope": args.lightx2v_double_precision_rope,
                "double_precision_rope_effect_for_wan": "inert",
                "offload": False,
            },
            "backend_capability": backend_capability,
            "effective": {
                "attention_mode": configured_self_attention,
                "self_attention_mode": configured_self_attention,
                "cross1_attention_mode": configured_cross1_attention,
                "cross2_attention_mode": configured_cross2_attention,
                "rope_type": configured_rope,
                "double_precision_rope_configured": bool(configured_double_rope),
                "double_precision_rope": False,
                "rope_compute_dtype": math_runtime_report[
                    "rope_compute_dtype"
                ],
                "rms_norm_compute_mode": "input",
                "attention_roles": {
                    role: {
                        "requested_mode": role_report.get("requested_mode"),
                        "attention_family": role_report.get(
                            "attention_family"
                        ),
                        "implementation_classes": role_report.get(
                            "implementation_classes"
                        ),
                        "kernel_symbols": role_report.get("kernel_symbols"),
                        "sparse_provider": role_report.get("sparse_provider"),
                        "sparse_mask": role_report.get("sparse_mask"),
                        "successful_calls": role_report.get("successful_calls"),
                        "sparse_provider_successful_calls": role_report.get(
                            "sparse_provider_successful_calls"
                        ),
                        "sparse_dispatch_successful_calls": role_report.get(
                            "sparse_dispatch_successful_calls"
                        ),
                        "sparse_final_kernel_successful_calls": role_report.get(
                            "sparse_final_kernel_successful_calls"
                        ),
                    }
                    for role, role_report in (reference_probe.get("roles") or {}).items()
                },
            },
            "effectiveness_proof": (
                "role-specific concrete classes; dense kernel symbols and apply "
                "calls; sparse dispatch plus final-kernel call and an observed "
                "strictly sparse mask"
            ),
            "runtime_probe": reference_probe,
            "wan_math_runtime": math_runtime_report,
            "vae_runtime": vae_runtime_report,
            "effectiveness_verification": optimization_verification,
            "generator_config": str(generator_config_path.resolve()),
            "fallbacks": list(optimization_verification["fallbacks"]),
            "quality_tier": ("numerically-approximate" if attention_is_approximate else "exact"),
        },
        "framework_module": str(Path(sys.modules["lightx2v"].__file__).resolve()),
    }


def _validate_video(path: Path, expected: argparse.Namespace) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"generated video is missing or empty: {path}")
    probe = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_frames,nb_read_frames,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    payload = json.loads(probe.stdout)
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream, found {len(streams)}")
    stream = streams[0]
    decoded_frames = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
    if int(stream.get("width") or 0) != expected.width:
        raise RuntimeError(f"video width mismatch: {stream.get('width')} != {expected.width}")
    if int(stream.get("height") or 0) != expected.height:
        raise RuntimeError(f"video height mismatch: {stream.get('height')} != {expected.height}")
    if decoded_frames != expected.frames:
        raise RuntimeError(f"decoded frame mismatch: {decoded_frames} != {expected.frames}")
    frame_rate_text = str(stream.get("r_frame_rate") or "")
    try:
        decoded_fps = float(Fraction(frame_rate_text))
    except (ValueError, ZeroDivisionError) as error:
        raise RuntimeError(
            f"video frame rate is missing or malformed: {frame_rate_text!r}"
        ) from error
    if not math.isclose(decoded_fps, float(expected.fps), rel_tol=0.0, abs_tol=1e-3):
        raise RuntimeError(
            f"video frame rate mismatch: {decoded_fps:g} != {int(expected.fps)}"
        )
    ffmpeg_executable = _validation_ffmpeg_executable()
    decode = _run(
        [
            ffmpeg_executable,
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        check=False,
    )
    if decode.returncode != 0:
        raise RuntimeError(f"full FFmpeg decode failed: {decode.stderr.strip()}")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "decoded_frames": decoded_frames,
        "decoded_fps": decoded_fps,
        "full_decode_ok": True,
        "ffmpeg_executable": ffmpeg_executable,
        "stream": stream,
    }


def _semantic_config(
    args: argparse.Namespace,
    checkpoints: Mapping[str, Mapping[str, Any]],
    *,
    execution_fingerprint_sha256: str | None = None,
) -> dict[str, Any]:
    _validate_worldfoundry_kernel_request(args)
    _validate_fastvideo_sla_pairing_request(args)
    lightx2v_attention_request = _lightx2v_attention_request(args)
    selected_approximate_attention = str(
        args.worldfoundry_approximate_attention
    ).strip().casefold()
    learned_sla_contract = None
    if selected_approximate_attention in WORLD_FOUNDRY_FASTVIDEO_SLA_KINDS:
        learned_sla_contract = _fastvideo_sla_expectation(
            argparse.Namespace(kind=selected_approximate_attention),
            fastvideo_repository=args.fastvideo_source,
            checkpoint_expectation=(
                _fastvideo_sla_checkpoint_expectation_for_args(args)
            ),
        )
    config = {
        "protocol_id": PROTOCOL_ID,
        "model": "Wan2.2-TI2V-5B",
        "checkpoint_layouts": {name: checkpoint["identity_sha256"] for name, checkpoint in checkpoints.items()},
        "checkpoint_equivalence": "official native and Diffusers layouts of Wan2.2-TI2V-5B",
        "mode": "text-to-video",
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "frames": args.frames,
        "steps": args.steps,
        "guidance_scale": args.guidance,
        "seed": args.seed,
        "fps": args.fps,
        "dtype": "bfloat16",
        "offload": "none",
        "save_video": True,
        "timing_boundary": "public generation call including MP4 save; model load and validation excluded",
        "isolation_policy": {
            "require_idle_at_start": bool(args.require_idle),
            "fail_on_generation_contamination": bool(getattr(args, "fail_on_contamination", True)),
            "proof_mode": getattr(args, "gpu_isolation_proof", "compute-exclusive"),
            "poll_interval_seconds": float(
                getattr(args, "isolation_poll_interval", 0.1)
            ),
            "allow_unproven_pid_binding": bool(
                getattr(args, "allow_unproven_pid_binding", False)
            ),
        },
        "worldfoundry_optimization_profile": {
            "enabled": args.worldfoundry_optimized,
            "require_effective": args.worldfoundry_require_effective,
            "attention_backend": args.worldfoundry_attention_backend,
            "vae_decode_autocast": args.worldfoundry_vae_decode_autocast,
            "vae_weight_dtype": args.worldfoundry_vae_weight_dtype,
            "vae_channels_last_3d": args.worldfoundry_vae_channels_last_3d,
            "dit_weight_dtype": args.worldfoundry_dit_weight_dtype,
            "quantization": args.worldfoundry_quantization,
            "quantization_config": json.loads(
                args.worldfoundry_quantization_config_json
            ),
            "fuse_qkv": args.worldfoundry_fuse_qkv,
            "qkv_strategy": args.worldfoundry_qkv_strategy,
            "qkv_split_threshold": args.worldfoundry_qkv_split_threshold,
            "inplace_residual": args.worldfoundry_inplace_residual,
            "static_cross_kv": args.worldfoundry_static_cross_kv,
            "fused_rope": args.worldfoundry_fused_rope,
            "rope_precision": args.worldfoundry_rope_precision,
            "rms_norm_precision": args.worldfoundry_rms_norm_precision,
            "compile": args.worldfoundry_compile,
            "compile_backend": args.worldfoundry_compile_backend,
            "compile_mode": args.worldfoundry_compile_mode,
            "compile_fullgraph": args.worldfoundry_compile_fullgraph,
            "compile_dynamic": args.worldfoundry_compile_dynamic,
            "cuda_graph": args.worldfoundry_cuda_graph,
            "teacache_threshold": args.worldfoundry_teacache_threshold,
            "feature_cache": args.worldfoundry_feature_cache,
            "feature_cache_options": json.loads(args.worldfoundry_feature_cache_options_json),
            "approximate_attention": args.worldfoundry_approximate_attention,
            "approximate_attention_request": (
                _worldfoundry_approximate_attention_request(args)
            ),
            "sparse_sparsity": args.worldfoundry_sparse_sparsity,
            "sparse_dense_steps": args.worldfoundry_sparse_dense_steps,
            "sta_window": args.worldfoundry_sta_window,
            "vsa_block_tile": args.worldfoundry_vsa_block_tile,
        },
        "lightx2v_optimization_profile": {
            "attention_mode": args.lightx2v_attention_mode,
            "self_attention_mode": lightx2v_attention_request["self_mode"],
            "cross_attention_mode": lightx2v_attention_request["cross_mode"],
            "self_attention_family": lightx2v_attention_request["self_family"],
            "self_attention_settings": lightx2v_attention_request[
                "self_settings"
            ],
            "sample_shift": args.lightx2v_sample_shift,
            "rope_type": args.lightx2v_rope_type,
            "double_precision_rope": args.lightx2v_double_precision_rope,
            "vae_weight_dtype": os.getenv("DTYPE", "BF16"),
            "vae_channels_last_3d": os.getenv(
                "USE_CHANNELS_LAST_3D",
                "0",
            )
            == "1",
            "offload": False,
        },
        "fastvideo_optimization_profile": {
            "attention_backend": args.fastvideo_attention_backend,
            "offload": False,
            "fsdp": False,
            "learned_sla_contract": learned_sla_contract,
        },
    }
    if execution_fingerprint_sha256 is not None:
        config["execution_fingerprint_sha256"] = execution_fingerprint_sha256
    return config


def run_one(args: argparse.Namespace) -> int:
    # Reject incomplete or contradictory profiles before GPU inspection or
    # framework imports so a benchmark cannot silently use sparse defaults.
    _validate_fastvideo_sla_pairing_request(args)
    _worldfoundry_approximate_attention_request(args)
    output_dir = args.output_dir.expanduser().resolve()
    record_path = output_dir / "records" / f"{args.run_id}.json"
    video_path = output_dir / "videos" / f"{args.run_id}.mp4"
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at": _utc_now(),
        "run": {
            "run_id": args.run_id,
            "pair_id": args.pair_id,
            "order": args.order,
            "framework": args.framework,
            "physical_gpu": args.physical_gpu,
            "pid": os.getpid(),
            "tag": args.tag,
        },
    }
    _write_json(record_path, record)
    isolation_monitor: ComputeAppMonitor | None = None
    isolation_report: dict[str, Any] | None = None
    measurements: dict[str, Any] | None = None
    try:
        # Fingerprint the already-imported runner as close to process start as
        # possible. Checkpoint traversal and the idle gate can both take long
        # enough for an unrelated worktree edit to otherwise be misattributed.
        benchmark_sources = _benchmark_source_fingerprint()
        repository_states = {
            "worldfoundry": _git_state(
                Path(__file__).resolve().parents[2],
                include_dirty=False,
            ),
            "fastvideo": _git_state(args.fastvideo_source),
            "lightx2v": _git_state(args.lightx2v_source),
        }
        runtime_sources = _runtime_source_fingerprints(
            worldfoundry_repository=Path(__file__).resolve().parents[2],
            fastvideo_repository=args.fastvideo_source,
            lightx2v_repository=args.lightx2v_source,
        )
        record.update(
            benchmark_sources=benchmark_sources,
            source_repository_states=repository_states,
            runtime_sources=runtime_sources,
        )
        if args.require_idle:
            record["idle_gate"] = _require_idle_gpu(
                args.physical_gpu,
                samples=args.idle_samples,
                interval_s=args.idle_interval,
            )
        else:
            record["idle_gate"] = [
                {**_gpu_query(args.physical_gpu), "compute_apps": _gpu_compute_apps(args.physical_gpu)}
            ]
        idle_snapshot = record["idle_gate"][-1]
        idle_compute_apps = list(idle_snapshot["compute_apps"])
        proof_mode = getattr(args, "gpu_isolation_proof", "compute-exclusive")
        compute_mode = str(idle_snapshot.get("compute_mode") or "").replace(" ", "_").casefold()
        exclusive_compute_mode = (
            bool(args.require_idle)
            and proof_mode == "compute-exclusive"
            and compute_mode == "exclusive_process"
        )
        record["gpu_isolation_preconditions"] = {
            "proof_mode": proof_mode,
            "reported_compute_mode": idle_snapshot.get("compute_mode"),
            "exclusive_process_proven": exclusive_compute_mode,
            "empty_card_gate_passed": bool(args.require_idle and not idle_compute_apps),
        }
        if proof_mode == "compute-exclusive" and not exclusive_compute_mode:
            raise RuntimeError(
                "certifying framework A/B requires NVIDIA compute mode "
                f"Exclusive_Process; GPU {args.physical_gpu} reports "
                f"{idle_snapshot.get('compute_mode')!r}"
            )
        if (
            bool(getattr(args, "allow_unproven_pid_binding", False))
            and bool(getattr(args, "fail_on_contamination", True))
        ):
            raise ValueError(
                "--allow-unproven-pid-binding is smoke-only and requires "
                "--no-fail-on-contamination"
            )
        isolation_monitor = ComputeAppMonitor(
            args.physical_gpu,
            idle_compute_apps,
            interval_s=float(getattr(args, "isolation_poll_interval", 0.1)),
            allowed_root_pids={os.getpid()},
            allow_descendants=True,
            exclusive_compute_mode=exclusive_compute_mode,
            allow_unproven_pid_binding=bool(
                getattr(args, "allow_unproven_pid_binding", False)
            ),
            monitored_gpu_uuid=str(idle_snapshot["uuid"]),
        )
        isolation_monitor.start()
        # Reading torch's current CUDA UUID may initialize the driver, so it
        # must happen only after the full-window isolation monitor is live.
        record["cuda_device_mapping"] = _cuda_visibility_evidence(
            args.physical_gpu,
            idle_snapshot,
        )
        # Framework runners reuse this monitor so no model-load PID can be
        # grandfathered as a generation baseline.
        args._compute_app_monitor = isolation_monitor
        checkpoint_paths = {
            "worldfoundry": args.worldfoundry_checkpoint or args.checkpoint,
            "fastvideo": args.fastvideo_checkpoint or args.checkpoint,
            "lightx2v": args.lightx2v_checkpoint or args.checkpoint,
        }
        checkpoints = {name: _checkpoint_fingerprint(path) for name, path in checkpoint_paths.items()}
        execution_identity = {
            "benchmark_sources_sha256": benchmark_sources["identity_sha256"],
            "runtime_sources": {
                name: source["sha256"] for name, source in runtime_sources.items()
            },
            "repositories": {
                name: {
                    "commit": state.get("commit"),
                    "dirty_fingerprint_sha256": state.get("dirty_fingerprint_sha256"),
                }
                for name, state in repository_states.items()
            },
        }
        execution_fingerprint_sha256 = _canonical_digest(execution_identity)
        semantic_config = _semantic_config(
            args,
            checkpoints,
            execution_fingerprint_sha256=execution_fingerprint_sha256,
        )
        record.update(
            checkpoints=checkpoints,
            benchmark_sources=benchmark_sources,
            execution_fingerprint={
                **execution_identity,
                "identity_sha256": execution_fingerprint_sha256,
            },
            semantic_config=semantic_config,
            semantic_config_sha256=_canonical_digest(semantic_config),
            environment={
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "packages": _package_versions(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "threading": {
                    name: os.environ.get(name)
                    for name in ("OMP_NUM_THREADS", "NUMEXPR_MAX_THREADS", "NUMEXPR_NUM_THREADS")
                },
                **repository_states,
            },
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        runners: dict[str, Callable[[argparse.Namespace, Path], dict[str, Any]]] = {
            "worldfoundry": _run_worldfoundry,
            "fastvideo": _run_fastvideo,
            "lightx2v": _run_lightx2v,
        }
        measurements = runners[args.framework](args, video_path)
        isolation_report = isolation_monitor.stop()
        measurements["generation_isolation"] = isolation_report
        record.update(
            measurements=measurements,
            gpu_isolation=isolation_report,
        )
        validation = _validate_video(Path(measurements["output_path"]), args)
        record.update(
            validation=validation,
            compute_apps_after=_gpu_compute_apps(args.physical_gpu),
        )
        _raise_for_generation_contamination(
            measurements,
            enabled=bool(getattr(args, "fail_on_contamination", True)),
        )
        record.update(status="completed", completed_at=_utc_now())
        _write_json(record_path, record)
        print(json.dumps({"record": str(record_path), "status": "completed"}, ensure_ascii=False))
        return 0
    except BaseException as error:
        if isolation_monitor is not None and isolation_report is None:
            isolation_report = isolation_monitor.stop()
        if isolation_report is not None:
            record["gpu_isolation"] = isolation_report
            if measurements is not None:
                measurements["generation_isolation"] = isolation_report
                record["measurements"] = measurements
        if isinstance(error, BackendCapabilityError):
            record["backend_capability"] = error.receipt
        record.update(
            status="failed",
            completed_at=_utc_now(),
            error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
        )
        _write_json(record_path, record)
        print(json.dumps({"record": str(record_path), "status": "failed", "error": str(error)}, ensure_ascii=False))
        return 1


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary_stats(values: list[float]) -> dict[str, Any]:
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    return {
        "n": len(values),
        "raw": values,
        "median": median,
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "mad": statistics.median(deviations),
        "p10": _percentile(values, 0.10),
        "p90": _percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def _quality_metric(value: object) -> float:
    if isinstance(value, str) and value.casefold() in {
        "inf",
        "+inf",
        "infinity",
        "+infinity",
    }:
        return math.inf
    number = float(value)
    if math.isnan(number):
        raise ValueError("video quality metric is NaN")
    return number


def _video_quality_gate(
    metrics: Mapping[str, Any],
    *,
    min_psnr: float,
    min_ssim: float,
    max_lpips: float,
) -> dict[str, Any]:
    psnr = _quality_metric(metrics["psnr_db_mean"])
    ssim = _quality_metric(metrics["ssim_mean"])
    lpips = _quality_metric(metrics["lpips_alex_mean"])
    checks = {
        "psnr_db_mean": {
            "actual": psnr if math.isfinite(psnr) else "infinity",
            "operator": ">=",
            "threshold": float(min_psnr),
            "passed": psnr >= float(min_psnr),
        },
        "ssim_mean": {
            "actual": ssim,
            "operator": ">=",
            "threshold": float(min_ssim),
            "passed": ssim >= float(min_ssim),
        },
        "lpips_alex_mean": {
            "actual": lpips,
            "operator": "<=",
            "threshold": float(max_lpips),
            "passed": lpips <= float(max_lpips),
        },
    }
    return {
        "checks": checks,
        "passed": all(bool(check["passed"]) for check in checks.values()),
    }


def _performance_gate(
    *,
    sample_sufficient: bool,
    paired_speedup_median: float,
    bootstrap_ci_lower: float,
    min_paired_speedup: float,
    min_ci_lower: float,
) -> dict[str, Any]:
    checks = {
        "sample_sufficient": {
            "actual": bool(sample_sufficient),
            "operator": "is",
            "threshold": True,
            "passed": bool(sample_sufficient),
        },
        "paired_speedup_median": {
            "actual": float(paired_speedup_median),
            "operator": ">=",
            "threshold": float(min_paired_speedup),
            "passed": float(paired_speedup_median) >= float(min_paired_speedup),
        },
        "bootstrap_95ci_lower": {
            "actual": float(bootstrap_ci_lower),
            "operator": ">=",
            "threshold": float(min_ci_lower),
            "passed": float(bootstrap_ci_lower) >= float(min_ci_lower),
        },
    }
    return {
        "checks": checks,
        "passed": all(bool(check["passed"]) for check in checks.values()),
    }


def _bootstrap_median_ci(values: list[float], *, samples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    medians: list[float] = []
    for _ in range(samples):
        resampled = [rng.choice(values) for _ in values]
        medians.append(statistics.median(resampled))
    return _percentile(medians, 0.025), _percentile(medians, 0.975)


def _load_completed_records(records_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(records_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schema_version") == SCHEMA_VERSION and payload.get("status") == "completed":
            payload["_record_path"] = str(path.resolve())
            records.append(payload)
    return records


def _generation_isolation_audit_issues(report: Mapping[str, Any]) -> list[str]:
    """Validate a certifying v4 isolation record without trusting its booleans."""

    issues: list[str] = []
    required_fields = {
        "audit_version",
        "window_policy",
        "window_started_monotonic_seconds",
        "window_stopped_monotonic_seconds",
        "compute_apps_before",
        "compute_apps_after",
        "added_compute_apps",
        "monitor_errors",
        "samples",
        "poll_interval_seconds",
        "sample_timestamps_monotonic_seconds",
        "max_sample_gap_seconds",
        "query_latency_seconds",
        "allowed_root_pids",
        "monitored_gpu_uuid",
        "allowed_process_identities",
        "bound_host_compute_pids",
        "host_pid_bindings",
        "identity_proof",
        "exclusive_compute_mode",
        "thread_stopped",
        "bounded_polling_complete",
        "certifying",
        "contaminated",
    }
    missing = sorted(required_fields - set(report))
    if missing:
        return [f"missing fields: {', '.join(missing)}"]
    if report.get("audit_version") != ISOLATION_AUDIT_VERSION:
        issues.append(f"unexpected audit_version={report.get('audit_version')!r}")
    if report.get("window_policy") != "post-idle-gate-through-completed-output":
        issues.append(f"unexpected window_policy={report.get('window_policy')!r}")
    try:
        _normalize_gpu_uuid(
            report.get("monitored_gpu_uuid"),
            source="generation isolation audit",
        )
    except RuntimeError as error:
        issues.append(str(error))
    try:
        started = float(report["window_started_monotonic_seconds"])
        stopped = float(report["window_stopped_monotonic_seconds"])
        interval = float(report["poll_interval_seconds"])
        reported_max_gap = float(report["max_sample_gap_seconds"])
        samples = int(report["samples"])
    except (TypeError, ValueError, OverflowError) as error:
        issues.append(f"invalid numeric audit field: {error}")
        return issues
    if not all(math.isfinite(value) for value in (started, stopped, interval, reported_max_gap)):
        issues.append("audit window contains a non-finite value")
    if stopped <= started:
        issues.append("audit window is empty or reversed")
    if samples < 2:
        issues.append(f"samples must be >= 2, got {samples}")
    if not 0 < interval <= MAX_CERTIFYING_POLL_INTERVAL_SECONDS:
        issues.append(
            f"poll interval {interval} exceeds certifying maximum "
            f"{MAX_CERTIFYING_POLL_INTERVAL_SECONDS}"
        )
    timestamps_raw = report.get("sample_timestamps_monotonic_seconds")
    if not isinstance(timestamps_raw, list):
        issues.append("sample timestamps are not a list")
        timestamps: list[float] = []
    else:
        try:
            timestamps = [float(value) for value in timestamps_raw]
        except (TypeError, ValueError, OverflowError):
            timestamps = []
            issues.append("sample timestamps contain a non-numeric value")
    if len(timestamps) != samples:
        issues.append(
            f"sample timestamp count {len(timestamps)} does not match samples {samples}"
        )
    if any(not math.isfinite(value) for value in timestamps):
        issues.append("sample timestamps contain a non-finite value")
    if timestamps != sorted(timestamps):
        issues.append("sample timestamps are not monotonic")
    if timestamps and (timestamps[0] < started or timestamps[-1] > stopped):
        issues.append("sample timestamps fall outside the declared audit window")
    points = [started, *timestamps, stopped]
    actual_max_gap = max(
        (later - earlier for earlier, later in zip(points, points[1:])),
        default=0.0,
    )
    if not math.isclose(reported_max_gap, actual_max_gap, rel_tol=1e-6, abs_tol=1e-6):
        issues.append(
            f"reported max sample gap {reported_max_gap} != recomputed {actual_max_gap}"
        )
    latency = report.get("query_latency_seconds")
    if not isinstance(latency, Mapping) or not isinstance(latency.get("raw"), list):
        issues.append("query latency evidence is missing raw samples")
    else:
        try:
            raw_latency = [float(value) for value in latency["raw"]]
        except (TypeError, ValueError, OverflowError):
            raw_latency = []
            issues.append("query latency evidence contains a non-numeric value")
        if len(raw_latency) != len(timestamps):
            issues.append("query latency count does not match timestamp count")
        if any(value < 0 or not math.isfinite(value) for value in raw_latency):
            issues.append("query latency contains an invalid value")
        reported_latency_max = latency.get("max")
        if raw_latency and (
            not isinstance(reported_latency_max, (int, float))
            or not math.isclose(
                float(reported_latency_max),
                max(raw_latency),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        ):
            issues.append("reported maximum query latency is inconsistent")
    before = report.get("compute_apps_before")
    added = report.get("added_compute_apps")
    errors = report.get("monitor_errors")
    if not isinstance(before, list) or before:
        issues.append("certifying audit did not start from an empty compute-app set")
    if not isinstance(added, list) or added:
        issues.append("audit observed unapproved compute apps")
    if not isinstance(errors, list) or errors:
        issues.append("audit contains monitor errors")
    roots = report.get("allowed_root_pids")
    identities = report.get("allowed_process_identities")
    bindings = report.get("host_pid_bindings")
    bound_pids = report.get("bound_host_compute_pids")
    if not isinstance(roots, list) or not roots:
        issues.append("allowed root PID evidence is empty")
    if not isinstance(identities, list) or not identities:
        issues.append("allowed process identity evidence is empty")
    if not isinstance(bindings, list) or not bindings:
        issues.append("host PID binding evidence is empty")
    elif not all(
        isinstance(item, Mapping)
        and isinstance(item.get("host_pid"), int)
        and isinstance(item.get("local_pid"), int)
        and isinstance(item.get("local_starttime_ticks"), int)
        and item.get("proof")
        in {"linux-nspid", "exclusive-process-temporal-binding"}
        for item in bindings
    ):
        issues.append("host PID binding evidence is malformed or unproven")
    if isinstance(bindings, list) and isinstance(bound_pids, list):
        derived_bound = sorted(
            int(item["host_pid"])
            for item in bindings
            if isinstance(item, Mapping) and isinstance(item.get("host_pid"), int)
        )
        if derived_bound != sorted(int(value) for value in bound_pids):
            issues.append("bound host PID list disagrees with binding evidence")
    else:
        issues.append("bound host PID evidence is malformed")
    if report.get("identity_proof") != "exclusive-process-compute-mode":
        issues.append("full-window exclusion is not backed by exclusive-process compute mode")
    if report.get("exclusive_compute_mode") is not True:
        issues.append("exclusive-process compute mode was not proven")
    if report.get("thread_stopped") is not True:
        issues.append("monitor thread was not stopped before report freeze")
    if report.get("bounded_polling_complete") is not True:
        issues.append("bounded polling coverage check failed")
    recomputed_contaminated = bool(
        (added if isinstance(added, list) else ["malformed"])
        or (errors if isinstance(errors, list) else ["malformed"])
    )
    if bool(report.get("contaminated")) != recomputed_contaminated:
        issues.append("contaminated flag disagrees with raw audit evidence")
    if report.get("certifying") is not True:
        issues.append("audit explicitly marks itself non-certifying")
    return issues


def _record_cuda_affinity_issues(record: Mapping[str, Any]) -> list[str]:
    """Cross-check idle, window, generation, visibility, and live CUDA UUIDs."""

    issues: list[str] = []
    run = record.get("run")
    idle_gate = record.get("idle_gate")
    measurements = record.get("measurements")
    mapping = record.get("cuda_device_mapping")
    if not isinstance(run, Mapping):
        return ["run metadata is missing for CUDA affinity validation"]
    if not isinstance(idle_gate, list) or not idle_gate or not isinstance(idle_gate[-1], Mapping):
        return ["idle-gate GPU UUID evidence is missing"]
    if not isinstance(measurements, Mapping):
        return ["generation measurements are missing for CUDA affinity validation"]
    generation = measurements.get("gpu_at_generation_start")
    isolation = measurements.get("generation_isolation")
    if not isinstance(generation, Mapping):
        issues.append("generation-start GPU UUID evidence is missing")
    if not isinstance(isolation, Mapping):
        issues.append("generation-window GPU UUID evidence is missing")
    if not isinstance(mapping, Mapping):
        issues.append("top-level CUDA device mapping evidence is missing")
        return issues

    try:
        run_gpu = int(run["physical_gpu"])
        mapped_gpu = int(mapping["monitored_physical_gpu"])
    except (KeyError, TypeError, ValueError, OverflowError):
        issues.append("physical GPU index evidence is malformed")
    else:
        if run_gpu != mapped_gpu:
            issues.append(
                f"run physical GPU {run_gpu} disagrees with CUDA mapping GPU {mapped_gpu}"
            )
    if mapping.get("framework_local_device") != 0 or mapping.get("current_cuda_device") != 0:
        issues.append("framework did not prove execution on its sole local CUDA device 0")
    if mapping.get("single_device_mapping_proven") is not True:
        issues.append("top-level CUDA device mapping is explicitly unproven")

    raw_uuids = {
        "idle_gate": idle_gate[-1].get("uuid"),
        "visibility_mapping": mapping.get("monitored_gpu_uuid"),
        "torch_current_cuda": mapping.get("current_cuda_uuid"),
        "generation_start": generation.get("uuid") if isinstance(generation, Mapping) else None,
        "generation_window": (
            isolation.get("monitored_gpu_uuid")
            if isinstance(isolation, Mapping)
            else None
        ),
    }
    normalized: dict[str, str] = {}
    for source, value in raw_uuids.items():
        try:
            normalized[source] = _normalize_gpu_uuid(value, source=source)
        except RuntimeError as error:
            issues.append(str(error))
    if len(normalized) == len(raw_uuids):
        distinct = {value.casefold() for value in normalized.values()}
        if len(distinct) != 1:
            issues.append(
                "GPU UUID evidence disagrees across idle gate, CUDA runtime, "
                f"generation window, and generation start: {normalized!r}"
            )
    return issues


def _record_smoke_mode_reasons(record: Mapping[str, Any]) -> list[str]:
    """Return run-time escape hatches that make a record diagnostic-only."""

    reasons: list[str] = []
    semantic = record.get("semantic_config")
    if not isinstance(semantic, Mapping):
        return ["record is missing semantic isolation policy"]
    isolation = semantic.get("isolation_policy")
    if not isinstance(isolation, Mapping):
        reasons.append("record is missing semantic isolation policy")
    else:
        if isolation.get("require_idle_at_start") is not True:
            reasons.append("run used --no-require-idle")
        if isolation.get("fail_on_generation_contamination") is not True:
            reasons.append("run used --no-fail-on-contamination")
        if isolation.get("allow_unproven_pid_binding") is not False:
            reasons.append("run enabled --allow-unproven-pid-binding")
        if isolation.get("proof_mode") != "compute-exclusive":
            reasons.append("run did not use compute-exclusive isolation proof")
    worldfoundry_profile = semantic.get("worldfoundry_optimization_profile")
    if (
        isinstance(worldfoundry_profile, Mapping)
        and worldfoundry_profile.get("require_effective") is not True
    ):
        reasons.append("run used --no-worldfoundry-require-effective")
    return reasons


def _filter_generation_isolation_records(
    records: Iterable[dict[str, Any]],
    *,
    allow_contaminated: bool = False,
    allow_unaudited: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for record in records:
        report = record.get("measurements", {}).get("generation_isolation")
        run = record.get("run", {})
        identity = {
            "record": record.get("_record_path"),
            "run_id": run.get("run_id"),
            "framework": run.get("framework"),
        }
        if not isinstance(report, Mapping):
            if allow_unaudited:
                record.setdefault("_noncertifying_reasons", []).append(
                    "missing_generation_isolation_audit"
                )
                accepted.append(record)
            else:
                exclusions.append({"reason": "missing_generation_isolation_audit", **identity})
            continue
        audit_issues = [
            *_generation_isolation_audit_issues(report),
            *_record_cuda_affinity_issues(record),
        ]
        raw_contamination = bool(
            report.get("contaminated")
            or report.get("added_compute_apps")
            or report.get("monitor_errors")
        )
        non_contamination_issues = [
            issue
            for issue in audit_issues
            if issue
            not in {
                "audit observed unapproved compute apps",
                "audit contains monitor errors",
                "contaminated flag disagrees with raw audit evidence",
                "audit explicitly marks itself non-certifying",
            }
        ]
        if non_contamination_issues and not allow_unaudited:
            exclusions.append(
                {
                    "reason": "invalid_generation_isolation_audit",
                    **identity,
                    "issues": audit_issues,
                }
            )
            continue
        if raw_contamination and not allow_contaminated:
            exclusions.append(
                {
                    "reason": "generation_compute_app_contamination",
                    **identity,
                    "added_compute_apps": list(report.get("added_compute_apps") or []),
                }
            )
            continue
        if audit_issues:
            record.setdefault("_noncertifying_reasons", []).extend(audit_issues)
        smoke_reasons = _record_smoke_mode_reasons(record)
        if smoke_reasons:
            record.setdefault("_noncertifying_reasons", []).extend(smoke_reasons)
        accepted.append(record)
    return accepted, exclusions


def _projection_fingerprint_evidence(
    record: Mapping[str, Any],
    *,
    framework: str,
) -> dict[str, Any] | None:
    """Extract one independently reported learned-projection set."""

    measurements = record.get("measurements")
    if not isinstance(measurements, Mapping):
        return None
    optimization = measurements.get("optimization_report")
    if not isinstance(optimization, Mapping):
        return None
    if framework == "worldfoundry":
        runtime = optimization.get("runtime")
        if not isinstance(runtime, Mapping):
            return None
        approximate = runtime.get("approximate_attention")
        if not isinstance(approximate, Mapping):
            return None
        learned = approximate.get("fastvideo_sla")
        candidates = [learned] if isinstance(learned, Mapping) else []
    elif framework == "fastvideo":
        probes = optimization.get("runtime_probe")
        if not isinstance(probes, (list, tuple)):
            return None
        candidates = []
        for probe in probes:
            if not isinstance(probe, Mapping):
                return None
            roles = probe.get("roles")
            self_role = roles.get("self") if isinstance(roles, Mapping) else None
            if not isinstance(self_role, Mapping):
                return None
            candidates.append(self_role)
    else:
        return None
    if not candidates:
        return None

    normalized: list[dict[str, Any]] = []
    expected_layers = {
        str(index) for index in range(WAN22_TI2V_TRANSFORMER_BLOCKS)
    }
    for candidate in candidates:
        layer_value = candidate.get("layer_projection_fingerprints")
        if not isinstance(layer_value, Mapping):
            return None
        layers = {str(key): str(value) for key, value in layer_value.items()}
        if set(layers) != expected_layers or any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in layers.values()
        ):
            return None
        canonical = _fastvideo_projection_set_fingerprint(
            layers,
            checkpoint_layout=None,
        )
        layout_bound = _fastvideo_projection_set_fingerprint(
            layers,
            checkpoint_layout=FASTVIDEO_SLA_CHECKPOINT_LAYOUT,
        )
        if framework == "fastvideo":
            reported_canonical = candidate.get("canonical_projection_fingerprint")
            reported_layout_bound = candidate.get(
                "all_projection_weights_fingerprint"
            )
        else:
            reported_canonical = canonical
            reported_layout_bound = candidate.get(
                "all_projection_weights_fingerprint"
            )
        if (
            reported_canonical != canonical
            or reported_layout_bound != layout_bound
        ):
            return None
        normalized.append(
            {
                "canonical_projection_fingerprint": canonical,
                "all_projection_weights_fingerprint": layout_bound,
                "layer_projection_fingerprints": layers,
            }
        )
    if any(item != normalized[0] for item in normalized[1:]):
        return None
    return normalized[0]


def _projection_fingerprint_expectation(
    record: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Extract and independently recompute the semantic checkpoint preflight."""

    semantic = record.get("semantic_config")
    if not isinstance(semantic, Mapping):
        return None
    fastvideo_profile = semantic.get("fastvideo_optimization_profile")
    if not isinstance(fastvideo_profile, Mapping):
        return None
    contract = fastvideo_profile.get("learned_sla_contract")
    if not isinstance(contract, Mapping):
        return None
    layer_value = contract.get("layer_projection_fingerprints")
    if not isinstance(layer_value, Mapping):
        return None
    layers = {str(key): str(value) for key, value in layer_value.items()}
    expected_layers = {
        str(index) for index in range(WAN22_TI2V_TRANSFORMER_BLOCKS)
    }
    if set(layers) != expected_layers or any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in layers.values()
    ):
        return None
    canonical = _fastvideo_projection_set_fingerprint(
        layers,
        checkpoint_layout=None,
    )
    layout_bound = _fastvideo_projection_set_fingerprint(
        layers,
        checkpoint_layout=FASTVIDEO_SLA_CHECKPOINT_LAYOUT,
    )
    if (
        contract.get("checkpoint_layout") != FASTVIDEO_SLA_CHECKPOINT_LAYOUT
        or contract.get("projection_fingerprint") != layout_bound
        or contract.get("canonical_projection_fingerprint") != canonical
    ):
        return None
    return {
        "canonical_projection_fingerprint": canonical,
        "all_projection_weights_fingerprint": layout_bound,
        "layer_projection_fingerprints": layers,
    }


def _paired_records(
    records: Iterable[dict[str, Any]],
    *,
    reference_framework: str = "fastvideo",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return complete same-GPU WorldFoundry/reference pairs and exclusions."""

    if reference_framework not in {"fastvideo", "lightx2v"}:
        raise ValueError(f"unsupported reference framework: {reference_framework}")
    framework_names = ("worldfoundry", reference_framework)

    def parse_interval(record: Mapping[str, Any]) -> tuple[datetime, datetime] | None:
        try:
            started = datetime.fromisoformat(str(record["started_at"]))
            completed = datetime.fromisoformat(str(record["completed_at"]))
        except (KeyError, TypeError, ValueError):
            return None
        if started.tzinfo is None or completed.tzinfo is None or completed <= started:
            return None
        return started, completed

    def gpu_uuid(record: Mapping[str, Any]) -> str | None:
        idle_gate = record.get("idle_gate")
        measurements = record.get("measurements")
        idle_uuid = None
        generation_uuid = None
        if isinstance(idle_gate, list) and idle_gate and isinstance(idle_gate[-1], Mapping):
            idle_uuid = idle_gate[-1].get("uuid")
        if isinstance(measurements, Mapping):
            generation = measurements.get("gpu_at_generation_start")
            if isinstance(generation, Mapping):
                generation_uuid = generation.get("uuid")
        if not isinstance(idle_uuid, str) or not idle_uuid.startswith("GPU-"):
            return None
        if generation_uuid != idle_uuid:
            return None
        return idle_uuid

    grouped: dict[str, list[dict[str, Any]]] = {}
    exclusions: list[dict[str, Any]] = []
    for record in records:
        pair_id = str(record.get("run", {}).get("pair_id") or "")
        if not pair_id:
            exclusions.append(
                {
                    "reason": "missing_pair_id",
                    "record": record.get("_record_path"),
                }
            )
            continue
        grouped.setdefault(pair_id, []).append(record)

    pairs: list[dict[str, Any]] = []
    for pair_id, items in sorted(grouped.items()):
        by_framework = {
            framework: [item for item in items if item.get("run", {}).get("framework") == framework]
            for framework in framework_names
        }
        if any(len(framework_items) != 1 for framework_items in by_framework.values()):
            exclusions.append(
                {
                    "reason": "pair_requires_exactly_one_record_per_framework",
                    "pair_id": pair_id,
                    "counts": {framework: len(framework_items) for framework, framework_items in by_framework.items()},
                }
            )
            continue
        worldfoundry = by_framework["worldfoundry"][0]
        reference = by_framework[reference_framework][0]
        wf_run = worldfoundry["run"]
        reference_run = reference["run"]
        if wf_run.get("physical_gpu") != reference_run.get("physical_gpu"):
            exclusions.append(
                {
                    "reason": "physical_gpu_mismatch",
                    "pair_id": pair_id,
                    "worldfoundry_gpu": wf_run.get("physical_gpu"),
                    f"{reference_framework}_gpu": reference_run.get("physical_gpu"),
                }
            )
            continue
        if wf_run.get("order") != reference_run.get("order"):
            exclusions.append(
                {
                    "reason": "order_mismatch",
                    "pair_id": pair_id,
                    "worldfoundry_order": wf_run.get("order"),
                    f"{reference_framework}_order": reference_run.get("order"),
                }
            )
            continue
        order = wf_run.get("order")
        if order not in {"AB", "BA"}:
            exclusions.append(
                {
                    "reason": "non_serial_pair_order",
                    "pair_id": pair_id,
                    "order": order,
                }
            )
            continue
        worldfoundry_interval = parse_interval(worldfoundry)
        reference_interval = parse_interval(reference)
        if worldfoundry_interval is None or reference_interval is None:
            exclusions.append(
                {
                    "reason": "missing_or_invalid_run_interval",
                    "pair_id": pair_id,
                }
            )
            continue
        first_interval, second_interval = (
            (worldfoundry_interval, reference_interval)
            if order == "AB"
            else (reference_interval, worldfoundry_interval)
        )
        if first_interval[1] > second_interval[0]:
            exclusions.append(
                {
                    "reason": "declared_order_not_observed_or_runs_overlap",
                    "pair_id": pair_id,
                    "order": order,
                }
            )
            continue
        worldfoundry_uuid = gpu_uuid(worldfoundry)
        reference_uuid = gpu_uuid(reference)
        if worldfoundry_uuid is None or reference_uuid is None:
            exclusions.append(
                {
                    "reason": "missing_or_inconsistent_gpu_uuid_evidence",
                    "pair_id": pair_id,
                }
            )
            continue
        if worldfoundry_uuid != reference_uuid:
            exclusions.append(
                {
                    "reason": "gpu_uuid_mismatch",
                    "pair_id": pair_id,
                    "worldfoundry_gpu_uuid": worldfoundry_uuid,
                    f"{reference_framework}_gpu_uuid": reference_uuid,
                }
            )
            continue
        semantic = worldfoundry.get("semantic_config")
        profile = (
            semantic.get("worldfoundry_optimization_profile")
            if isinstance(semantic, Mapping)
            else None
        )
        approximate_kind = (
            str(profile.get("approximate_attention") or "").casefold()
            if isinstance(profile, Mapping)
            else ""
        )
        if approximate_kind in WORLD_FOUNDRY_FASTVIDEO_SLA_KINDS:
            if reference_framework != "fastvideo":
                exclusions.append(
                    {
                        "reason": "learned_sla_requires_fastvideo_reference",
                        "pair_id": pair_id,
                        "reference_framework": reference_framework,
                    }
                )
                continue
            worldfoundry_projection = _projection_fingerprint_evidence(
                worldfoundry,
                framework="worldfoundry",
            )
            fastvideo_projection = _projection_fingerprint_evidence(
                reference,
                framework="fastvideo",
            )
            worldfoundry_preflight = _projection_fingerprint_expectation(
                worldfoundry
            )
            fastvideo_preflight = _projection_fingerprint_expectation(reference)
            if (
                worldfoundry_projection is None
                or fastvideo_projection is None
                or worldfoundry_preflight is None
                or fastvideo_preflight is None
                or worldfoundry_preflight != fastvideo_preflight
                or worldfoundry_projection != worldfoundry_preflight
                or fastvideo_projection != fastvideo_preflight
            ):
                exclusions.append(
                    {
                        "reason": "learned_sla_projection_fingerprint_mismatch",
                        "pair_id": pair_id,
                        "worldfoundry_projection": worldfoundry_projection,
                        "fastvideo_projection": fastvideo_projection,
                        "worldfoundry_preflight": worldfoundry_preflight,
                        "fastvideo_preflight": fastvideo_preflight,
                    }
                )
                continue
        pairs.append(
            {
                "pair_id": pair_id,
                "physical_gpu": wf_run.get("physical_gpu"),
                "gpu_uuid": worldfoundry_uuid,
                "order": wf_run.get("order"),
                "worldfoundry": worldfoundry,
                reference_framework: reference,
            }
        )
    return pairs, exclusions


def summarize(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    reference_framework = getattr(args, "reference_framework", "fastvideo")
    records = _load_completed_records(output_dir / "records")
    if args.tag:
        records = [record for record in records if record["run"].get("tag") == args.tag]
    records, isolation_exclusions = _filter_generation_isolation_records(
        records,
        allow_contaminated=bool(getattr(args, "allow_contaminated", False)),
        allow_unaudited=bool(getattr(args, "allow_unaudited_generation_isolation", False)),
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["semantic_config_sha256"], []).append(record)
    if not groups:
        raise RuntimeError("no completed, compatible records found")
    config_digest, selected = max(groups.items(), key=lambda item: len(item[1]))
    pairs, excluded_pairs = _paired_records(
        selected,
        reference_framework=reference_framework,
    )
    if not pairs:
        raise RuntimeError(f"no complete same-GPU WorldFoundry/{reference_framework} pairs found")
    pair_gpu_indices = {int(pair["physical_gpu"]) for pair in pairs}
    pair_gpu_uuids = {str(pair["gpu_uuid"]) for pair in pairs}
    if len(pair_gpu_indices) != 1 or len(pair_gpu_uuids) != 1:
        raise RuntimeError(
            "certifying framework A/B repeats must all use one physical GPU/UUID; "
            f"indices={sorted(pair_gpu_indices)}, uuids={sorted(pair_gpu_uuids)}"
        )
    order_counts = {
        order: sum(pair["order"] == order for pair in pairs)
        for order in ("AB", "BA")
    }
    if len(pairs) >= 2 and (
        min(order_counts.values()) == 0
        or abs(order_counts["AB"] - order_counts["BA"]) > 1
    ):
        raise RuntimeError(
            "certifying framework A/B repeats must be order-balanced; "
            f"counts={order_counts}"
        )
    formal_protocol_reasons: list[str] = []
    if len(pairs) < MIN_CERTIFYING_PAIRS:
        formal_protocol_reasons.append(
            "formal certification requires at least "
            f"{MIN_CERTIFYING_PAIRS} complete pairs; got {len(pairs)}"
        )
    if any(
        order_counts[order] < MIN_CERTIFYING_PAIRS_PER_ORDER
        for order in ("AB", "BA")
    ):
        formal_protocol_reasons.append(
            "formal certification requires at least "
            f"{MIN_CERTIFYING_PAIRS_PER_ORDER} pairs in each order; "
            f"counts={order_counts}"
        )
    if abs(order_counts["AB"] - order_counts["BA"]) > 1:
        formal_protocol_reasons.append(
            f"formal certification requires balanced AB/BA order; counts={order_counts}"
        )
    all_intervals: list[tuple[datetime, datetime, str]] = []
    for pair in pairs:
        for framework in ("worldfoundry", reference_framework):
            record = pair[framework]
            all_intervals.append(
                (
                    datetime.fromisoformat(str(record["started_at"])),
                    datetime.fromisoformat(str(record["completed_at"])),
                    str(record.get("run", {}).get("run_id")),
                )
            )
    all_intervals.sort(key=lambda item: item[0])
    for previous, current in zip(all_intervals, all_intervals[1:]):
        if previous[1] > current[0]:
            raise RuntimeError(
                "certifying framework A/B runs must be globally sequential; "
                f"{previous[2]!r} overlaps {current[2]!r}"
            )
    frameworks = {name: [pair[name] for pair in pairs] for name in ("worldfoundry", reference_framework)}
    sample_sufficient = len(pairs) >= args.min_samples
    if not args.allow_insufficient and not sample_sufficient:
        raise RuntimeError(f"need at least {args.min_samples} complete same-GPU pairs; got {len(pairs)}")
    timing: dict[str, dict[str, Any]] = {}
    importing: dict[str, dict[str, Any]] = {}
    loading: dict[str, dict[str, Any]] = {}
    memory: dict[str, dict[str, Any]] = {}
    for name, items in frameworks.items():
        timing[name] = _summary_stats([float(item["measurements"]["generation_s"]) for item in items])
        importing[name] = _summary_stats([float(item["measurements"]["framework_import_s"]) for item in items])
        loading[name] = _summary_stats([float(item["measurements"]["model_load_s"]) for item in items])
        memory[name] = _summary_stats(
            [float(item["measurements"]["gpu_generation"]["peak_memory_used_mib"]) for item in items]
        )
    paired_samples = []
    for pair in pairs:
        worldfoundry_s = float(pair["worldfoundry"]["measurements"]["generation_s"])
        reference_s = float(pair[reference_framework]["measurements"]["generation_s"])
        paired_samples.append(
            {
                "pair_id": pair["pair_id"],
                "physical_gpu": pair["physical_gpu"],
                "order": pair["order"],
                "worldfoundry_generation_s": worldfoundry_s,
                f"{reference_framework}_generation_s": reference_s,
                f"worldfoundry_speedup_vs_{reference_framework}": (reference_s / worldfoundry_s),
                "worldfoundry_record": pair["worldfoundry"]["_record_path"],
                f"{reference_framework}_record": pair[reference_framework]["_record_path"],
            }
        )
    quality_enabled = bool(getattr(args, "quality", False))
    quality_gate: dict[str, Any] = {
        "enabled": quality_enabled,
        "passed": None,
        "pairs": [],
        "mode": "cross-framework-pixel-diagnostic",
        "certification_role": "diagnostic-only",
        "disabled_reason": (
            None
            if quality_enabled
            else "independent implementations do not guarantee identical seed-to-latent trajectories"
        ),
    }
    if quality_enabled:
        import torch

        from benchmarks.inference.compare_video_quality import _lpips_metric, compare

        quality_device = torch.device(getattr(args, "quality_device", "cuda"))
        lpips_metric = _lpips_metric(quality_device)
        pair_quality = []
        for pair, paired_sample in zip(pairs, paired_samples, strict=True):
            reference_video = Path(pair[reference_framework]["measurements"]["output_path"])
            worldfoundry_video = Path(pair["worldfoundry"]["measurements"]["output_path"])
            metrics = compare(
                reference_video,
                worldfoundry_video,
                device=quality_device,
                lpips_metric=lpips_metric,
            )
            gate = _video_quality_gate(
                metrics,
                min_psnr=float(getattr(args, "min_video_psnr", 30.0)),
                min_ssim=float(getattr(args, "min_video_ssim", 0.95)),
                max_lpips=float(getattr(args, "max_video_lpips", 0.10)),
            )
            record = {
                "pair_id": pair["pair_id"],
                "reference_framework": reference_framework,
                "reference_video": str(reference_video.resolve()),
                "worldfoundry_video": str(worldfoundry_video.resolve()),
                "metrics": metrics,
                "gate": gate,
            }
            pair_quality.append(record)
            paired_sample["video_quality"] = record
        quality_gate = {
            "enabled": True,
            "mode": "cross-framework-pixel-diagnostic",
            "certification_role": "diagnostic-only",
            "disabled_reason": None,
            "metric_contract": {
                "psnr": "per-frame RGB uint8 mean",
                "ssim": "per-frame RGB scikit-image mean",
                "lpips": "per-frame AlexNet LPIPS mean",
                "device": str(quality_device),
            },
            "thresholds": {
                "min_video_psnr": float(getattr(args, "min_video_psnr", 30.0)),
                "min_video_ssim": float(getattr(args, "min_video_ssim", 0.95)),
                "max_video_lpips": float(getattr(args, "max_video_lpips", 0.10)),
            },
            "pairs": pair_quality,
            "passed": all(bool(item["gate"]["passed"]) for item in pair_quality),
        }
    paired_speedups = [float(item[f"worldfoundry_speedup_vs_{reference_framework}"]) for item in paired_samples]
    speedup = statistics.median(paired_speedups)
    ratio_of_medians = timing[reference_framework]["median"] / timing["worldfoundry"]["median"]
    ci_low, ci_high = _bootstrap_median_ci(paired_speedups, samples=args.bootstrap_samples, seed=20260830)
    performance_enabled = bool(getattr(args, "performance_gate", False))
    performance_gate = {
        "enabled": performance_enabled,
        **_performance_gate(
            sample_sufficient=sample_sufficient,
            paired_speedup_median=speedup,
            bootstrap_ci_lower=ci_low,
            min_paired_speedup=float(getattr(args, "min_paired_speedup", 1.0)),
            min_ci_lower=float(getattr(args, "min_speedup_ci_lower", 0.97)),
        ),
    }
    noncertifying_reasons = {
        str(reason)
        for record in selected
        for reason in record.get("_noncertifying_reasons", [])
    }
    noncertifying_reasons.update(formal_protocol_reasons)
    if bool(getattr(args, "allow_contaminated", False)):
        noncertifying_reasons.add("--allow-contaminated was enabled")
    if bool(getattr(args, "allow_unaudited_generation_isolation", False)):
        noncertifying_reasons.add(
            "--allow-unaudited-generation-isolation was enabled"
        )
    if bool(getattr(args, "allow_insufficient", False)):
        noncertifying_reasons.add("--allow-insufficient was enabled")
    noncertifying_reasons = sorted(noncertifying_reasons)
    summary_certifying = not noncertifying_reasons and sample_sufficient
    claim = (
        "non_certifying_diagnostic"
        if not summary_certifying
        else "quality_gate_failed"
        if quality_enabled and not quality_gate["passed"]
        else "performance_gate_failed"
        if performance_enabled and not performance_gate["passed"]
        else "worldfoundry_faster"
        if ci_low > 1.0
        else f"{reference_framework}_faster"
        if ci_high < 1.0
        else "statistically_inconclusive"
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "semantic_config_sha256": config_digest,
        "semantic_config": selected[0]["semantic_config"],
        "sample_counts": {name: len(items) for name, items in frameworks.items()},
        "paired_sample_count": len(pairs),
        "paired_samples": paired_samples,
        "record_exclusions": isolation_exclusions,
        "excluded_pairs": excluded_pairs,
        "generation_seconds": timing,
        "framework_import_seconds": importing,
        "model_load_seconds": loading,
        "whole_gpu_peak_memory_mib": memory,
        "video_quality_gate": quality_gate,
        "performance_gate": performance_gate,
        "certification": {
            "certifying": summary_certifying,
            "noncertifying_reasons": noncertifying_reasons,
            "single_physical_gpu": next(iter(pair_gpu_indices)),
            "single_gpu_uuid": next(iter(pair_gpu_uuids)),
            "order_counts": order_counts,
            "globally_sequential": True,
            "minimum_formal_pairs": MIN_CERTIFYING_PAIRS,
            "minimum_formal_pairs_per_order": MIN_CERTIFYING_PAIRS_PER_ORDER,
            "formal_pair_count_sufficient": len(pairs) >= MIN_CERTIFYING_PAIRS,
            "formal_order_count_sufficient": all(
                order_counts[order] >= MIN_CERTIFYING_PAIRS_PER_ORDER
                for order in ("AB", "BA")
            ),
        },
        "reference_framework": reference_framework,
        f"worldfoundry_speedup_vs_{reference_framework}": speedup,
        "worldfoundry_ratio_of_framework_medians": ratio_of_medians,
        "worldfoundry_speedup_bootstrap_95ci": [ci_low, ci_high],
        "claim": claim,
        "minimum_samples_required": args.min_samples,
        "sample_sufficient": sample_sufficient,
        "records": [
            pair[framework]["_record_path"] for pair in pairs for framework in ("worldfoundry", reference_framework)
        ],
    }
    _write_json(output_dir / "summary.json", summary)
    wf = timing["worldfoundry"]
    reference = timing[reference_framework]
    reference_label = "FastVideo" if reference_framework == "fastvideo" else "LightX2V"
    lines = [
        f"# Wan2.2-TI2V-5B: WorldFoundry vs {reference_label}",
        "",
        f"配置摘要：`{config_digest}`；计时包含完整生成与 MP4 保存，不含模型加载和输出验证。",
        "",
        "| Framework | n | generation median (s) | p10–p90 (s) | MAD (s) | import median (s) | load median (s) | peak whole-GPU memory median (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| WorldFoundry | {wf['n']} | {wf['median']:.3f} | {wf['p10']:.3f}–{wf['p90']:.3f} | {wf['mad']:.3f} | {importing['worldfoundry']['median']:.3f} | {loading['worldfoundry']['median']:.3f} | {memory['worldfoundry']['median']:.1f} |",
        f"| {reference_label} | {reference['n']} | {reference['median']:.3f} | {reference['p10']:.3f}–{reference['p90']:.3f} | {reference['mad']:.3f} | {importing[reference_framework]['median']:.3f} | {loading[reference_framework]['median']:.3f} | {memory[reference_framework]['median']:.1f} |",
        "",
        f"WorldFoundry paired speedup（{reference_label} / WorldFoundry）：`{speedup:.4f}×`，paired bootstrap 95% CI `[{ci_low:.4f}, {ci_high:.4f}]`。",
        f"框架中位数之比（仅作辅助统计）：`{ratio_of_medians:.4f}×`。",
        f"统计判定：`{summary['claim']}`。",
        "",
        "所有纳入样本均通过 FFprobe 帧数/分辨率检查和完整 FFmpeg 解码。",
    ]
    if not summary_certifying:
        lines.extend(
            (
                "",
                "该汇总仅供非认证诊断，不能支持性能领先声明："
                + "；".join(noncertifying_reasons),
            )
        )
    if quality_enabled:
        lines.extend(
            (
                "",
                "跨框架视频质量门："
                f"`{'passed' if quality_gate['passed'] else 'failed'}`；"
                "逐配对 PSNR/SSIM/LPIPS 见 summary.json。该门只诊断像素轨迹差异，"
                "不能替代各框架相对自身 baseline 的质量保持验证。",
            )
        )
    else:
        lines.extend(
            (
                "",
                "跨框架逐像素质量门默认关闭：独立实现不保证同一 seed 映射到同一扩散轨迹。"
                "输出仍必须通过完整解码；优化质量由各自 frozen baseline、绝对质量基准与人工复核认证。",
            )
        )
    if performance_enabled:
        lines.extend(
            (
                "",
                "性能门："
                f"`{'passed' if performance_gate['passed'] else 'failed'}`；"
                "成对中位 speedup 与 bootstrap CI 下界见 summary.json。",
            )
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    gates_passed = (not quality_enabled or bool(quality_gate["passed"])) and (
        not performance_enabled or bool(performance_gate["passed"])
    )
    return 0 if gates_passed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run one isolated framework sample")
    run.add_argument(
        "--framework",
        choices=("worldfoundry", "fastvideo", "lightx2v"),
        required=True,
    )
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument(
        "--worldfoundry-checkpoint",
        type=Path,
        help="optional official/native checkpoint layout used only by WorldFoundry",
    )
    run.add_argument(
        "--fastvideo-checkpoint",
        type=Path,
        help="optional Diffusers checkpoint layout used only by FastVideo",
    )
    run.add_argument(
        "--lightx2v-checkpoint",
        type=Path,
        help="optional official checkpoint layout used only by LightX2V",
    )
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--pair-id", default="")
    run.add_argument("--order", choices=("AB", "BA", "parallel"), default="parallel")
    run.add_argument("--tag", default="release")
    run.add_argument("--physical-gpu", type=int, required=True)
    run.add_argument("--prompt", default=DEFAULT_PROMPT)
    run.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    run.add_argument("--height", type=int, default=704)
    run.add_argument("--width", type=int, default=1280)
    run.add_argument("--frames", type=int, default=121)
    run.add_argument("--steps", type=int, default=50)
    run.add_argument("--guidance", type=float, default=5.0)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--fps", type=int, default=24)
    run.add_argument("--worldfoundry-optimized", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument(
        "--worldfoundry-require-effective",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "fail when a requested WorldFoundry speedup falls back or has no "
            "runtime proof; disable only for diagnostic smoke runs"
        ),
    )
    run.add_argument(
        "--worldfoundry-attention-backend",
        choices=(
            "auto",
            "torch",
            "flash",
            "flash2",
            "flash3",
            "flash4",
            "sage",
            "sage3",
            "xformers",
        ),
        default="flash",
        help="explicit exact attention provider for the optimized WorldFoundry lane",
    )
    run.add_argument(
        "--worldfoundry-vae-decode-autocast",
        choices=("none", "fp16", "bf16"),
        default="bf16",
        help="output-only VAE decode precision for the optimized WorldFoundry lane",
    )
    run.add_argument(
        "--worldfoundry-vae-weight-dtype",
        choices=("fp32", "fp16", "bf16"),
        default=None,
        help=(
            "resident Wan VAE parameter dtype; omitted keeps the historical "
            "FP32 default"
        ),
    )
    run.add_argument(
        "--worldfoundry-vae-channels-last-3d",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="store every Wan VAE Conv3d weight in channels_last_3d format",
    )
    run.add_argument(
        "--worldfoundry-dit-weight-dtype",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
        help="resident Wan DiT parameter dtype for the optimized WorldFoundry lane",
    )
    run.add_argument(
        "--worldfoundry-quantization",
        choices=("none", "fp8", "nvfp4", "int8", "int4", "gguf"),
        default="none",
    )
    run.add_argument(
        "--worldfoundry-quantization-config-json",
        default="{}",
        help=(
            "JSON object merged with the selected quantization mode; supports "
            "exclude and options such as min_features/keep_dense_fallback"
        ),
    )
    run.add_argument(
        "--worldfoundry-fuse-qkv",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run.add_argument(
        "--worldfoundry-qkv-strategy",
        choices=("auto", "packed", "split"),
        default="auto",
        help="merged-weight QKV dispatch; auto splits large Wan token workloads",
    )
    run.add_argument(
        "--worldfoundry-qkv-split-threshold",
        type=int,
        default=8192,
        help="flattened token rows where QKV auto switches to three GEMMs",
    )
    run.add_argument(
        "--worldfoundry-inplace-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "reuse the Wan residual stream during no-grad, cache-free inference"
        ),
    )
    run.add_argument(
        "--worldfoundry-static-cross-kv",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run.add_argument(
        "--worldfoundry-fused-rope",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run.add_argument(
        "--worldfoundry-rope-precision",
        choices=("fp32", "fp64"),
        default="fp32",
    )
    run.add_argument(
        "--worldfoundry-rms-norm-precision",
        choices=("fp32", "input"),
        default="input",
    )
    run.add_argument(
        "--worldfoundry-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run.add_argument("--worldfoundry-compile-backend", default="inductor")
    run.add_argument(
        "--worldfoundry-compile-mode",
        choices=(
            "none",
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="none",
    )
    run.add_argument(
        "--worldfoundry-compile-fullgraph",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run.add_argument(
        "--worldfoundry-compile-dynamic",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    run.add_argument("--worldfoundry-compile-options-json", default="{}")
    run.add_argument(
        "--worldfoundry-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=("capture the stateless Wan DiT path; requires --no-worldfoundry-static-cross-kv and no feature cache"),
    )
    run.add_argument(
        "--worldfoundry-feature-cache",
        choices=("none", *WORLD_FOUNDRY_FEATURE_CACHE_ALGORITHMS),
        default="none",
    )
    run.add_argument(
        "--worldfoundry-feature-cache-options-json",
        default="{}",
        help=(
            "algorithm options; MagCache requires calibrated ratios; First/Dual/"
            "DynamicBlock require residual_diff_threshold; BlockTaylorSeer defaults "
            "to LightX2V's [dense, skip, skip, skip] schedule"
        ),
    )
    run.add_argument(
        "--worldfoundry-teacache-threshold",
        type=float,
        help="legacy TeaCache threshold alias",
    )
    run.add_argument(
        "--worldfoundry-approximate-attention",
        choices=(
            "none",
            "sta",
            "vsa",
            "vmoba",
            *WORLD_FOUNDRY_FASTVIDEO_SLA_KINDS,
            *WORLD_FOUNDRY_LIGHTX2V_SPARSE_KINDS,
        ),
        default="none",
    )
    run.add_argument(
        "--worldfoundry-approximate-attention-profile-json",
        help=(
            "inline sparse-attention JSON object; VMoBA requires the complete "
            "grid-specific profile; LightX2V lanes accept explicit operator/mask "
            "fields; FastVideo learned-SLA lanes accept fastvideo_topk_ratio/"
            "fastvideo_feature_map and require learned proj_l checkpoint tensors"
        ),
    )
    run.add_argument(
        "--worldfoundry-approximate-attention-profile",
        type=Path,
        help="path to the same sparse-attention JSON object",
    )
    run.add_argument("--worldfoundry-sparse-sparsity", type=float, default=0.9)
    run.add_argument("--worldfoundry-sparse-dense-steps", type=int, default=0)
    run.add_argument("--worldfoundry-sta-window", type=int, nargs=3, default=(3, 3, 3))
    run.add_argument("--worldfoundry-vsa-block-tile", type=int, nargs=3, default=(4, 4, 4))
    run.add_argument("--require-idle", action=argparse.BooleanOptionalAction, default=True)
    run.add_argument("--idle-samples", type=int, default=3)
    run.add_argument("--idle-interval", type=float, default=1.0)
    run.add_argument(
        "--gpu-isolation-proof",
        choices=("compute-exclusive", "bounded-polling"),
        default="compute-exclusive",
        help=(
            "compute-exclusive is required for certifying runs; bounded-polling "
            "has a finite blind window and is diagnostic-only"
        ),
    )
    run.add_argument(
        "--isolation-poll-interval",
        type=float,
        default=0.1,
        help="compute-app polling interval; certifying maximum is 0.25 seconds",
    )
    run.add_argument(
        "--allow-unproven-pid-binding",
        action="store_true",
        help=(
            "allow non-reusable temporal host/container PID binding when NSpid "
            "is hidden; smoke-only and requires --no-fail-on-contamination"
        ),
    )
    run.add_argument(
        "--fail-on-contamination",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "fail a completed generation if a new compute-app PID appears; "
            "use --no-fail-on-contamination only for non-release smoke runs"
        ),
    )
    run.add_argument(
        "--fastvideo-source",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "tmp" / "refs" / "FastVideo",
    )
    run.add_argument(
        "--fastvideo-attention-backend",
        choices=(
            "FLASH_ATTN",
            "SAGE_ATTN",
            "SAGE_ATTN_THREE",
            "SLA_ATTN",
            "SAGE_SLA_ATTN",
        ),
        default="FLASH_ATTN",
        help=(
            "explicit process-wide FastVideo backend, installed before importing "
            "FastVideo; learned-SLA comparisons require SLA_ATTN/SAGE_SLA_ATTN"
        ),
    )
    run.add_argument(
        "--lightx2v-source",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "tmp" / "refs" / "LightX2V",
    )
    run.add_argument("--lightx2v-attention-mode", default="flash_attn2")
    run.add_argument(
        "--lightx2v-self-attention-mode",
        default=None,
        help=(
            "optional role-specific self provider; dense legacy runs inherit "
            "--lightx2v-attention-mode"
        ),
    )
    run.add_argument(
        "--lightx2v-cross-attention-mode",
        default=None,
        help=(
            "role-specific dense cross provider; required when the legacy "
            "attention mode is itself sparse"
        ),
    )
    run.add_argument(
        "--lightx2v-self-attention-settings-json",
        default="{}",
        help=(
            "mode-specific upstream self-attention settings object, for "
            "example dynamic_sparse_attn sparsity_ratio/operator"
        ),
    )
    run.add_argument("--lightx2v-sample-shift", type=float, default=5.0)
    run.add_argument("--lightx2v-rope-type", default="torch_complex_rope")
    run.add_argument(
        "--lightx2v-double-precision-rope",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run.set_defaults(func=run_one)

    summary = subparsers.add_parser("summarize", help="summarize compatible completed records")
    summary.add_argument("--output-dir", type=Path, required=True)
    summary.add_argument("--tag", default="release")
    summary.add_argument("--min-samples", type=int, default=MIN_CERTIFYING_PAIRS)
    summary.add_argument("--bootstrap-samples", type=int, default=20_000)
    summary.add_argument("--allow-insufficient", action="store_true")
    summary.add_argument(
        "--allow-contaminated",
        action="store_true",
        help="include records marked with generation compute-app contamination",
    )
    summary.add_argument(
        "--allow-unaudited-generation-isolation",
        action="store_true",
        help="include legacy records that do not contain a generation isolation audit",
    )
    summary.add_argument(
        "--reference-framework",
        choices=("fastvideo", "lightx2v"),
        default="fastvideo",
    )
    summary.add_argument(
        "--quality",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "opt-in cross-framework pixel diagnostic; not a substitute for "
            "per-framework baseline or absolute quality evaluation"
        ),
    )
    summary.add_argument("--quality-device", default="cuda")
    summary.add_argument("--min-video-psnr", type=float, default=30.0)
    summary.add_argument("--min-video-ssim", type=float, default=0.95)
    summary.add_argument("--max-video-lpips", type=float, default=0.10)
    summary.add_argument(
        "--performance-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="require paired speed and bootstrap lower-bound thresholds",
    )
    summary.add_argument("--min-paired-speedup", type=float, default=1.0)
    summary.add_argument("--min-speedup-ci-lower", type=float, default=0.97)
    summary.set_defaults(func=summarize)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

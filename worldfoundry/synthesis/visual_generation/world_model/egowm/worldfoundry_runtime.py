"""Bind WorldFoundry to the official EgoWM SVD inference checkout."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Mapping

from worldfoundry.core.io.paths import checkpoint_root_path, official_runtime_repo_path
from worldfoundry.runtime.assets import expand_worldfoundry_path
from worldfoundry.synthesis.visual_generation.world_model.runtime_manifest import command_settings


RUNTIME_DIR = Path(__file__).resolve().parent
OFFICIAL_ENTRYPOINT = RUNTIME_DIR / "infer.py"
OFFICIAL_REPO_URL = "https://github.com/miccooper9/egowm"
SOURCE_ENV_VAR = "WORLDFOUNDRY_EGOWM_SOURCE"
SOURCE_DIR_NAME = "egowm"

DEFAULT_CHECKPOINT_PATH = checkpoint_root_path("anuragba--egowm", "svd_25dof_nav.pth")
DEFAULT_BASE_MODEL_DIR = checkpoint_root_path("stabilityai--stable-video-diffusion-img2vid")

# Dynamic checks are authoritative; a static reason would keep a staged runtime blocked.
BLOCKED_REASON = ""


def runtime_root() -> Path:
    """Resolve the user-staged official EgoWM checkout."""
    return official_runtime_repo_path(SOURCE_DIR_NAME, specific_env=SOURCE_ENV_VAR)


def _option(options: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = options.get(name)
        if value not in (None, ""):
            return value
    return default


def _path_option(options: Mapping[str, Any], *names: str, default: Path) -> Path:
    return expand_worldfoundry_path(str(_option(options, *names, default=default)))


def _checkpoint_path(options: Mapping[str, Any]) -> Path:
    return _path_option(
        options,
        "egowm_checkpoint_path",
        "checkpoint_path",
        "model_path",
        default=DEFAULT_CHECKPOINT_PATH,
    )


def _base_model_dir(options: Mapping[str, Any]) -> Path:
    return _path_option(
        options,
        "svd_base_model_dir",
        "base_model_dir",
        "pretrained_path",
        default=DEFAULT_BASE_MODEL_DIR,
    )


def missing_requirements(*, options, runtime_root, entrypoint, profile) -> list[dict[str, str]]:
    """Report missing source, checkpoint, base-model, input, or dependency assets."""
    del profile
    options = dict(options or {})
    source = Path(str(runtime_root)).expanduser()
    missing: list[dict[str, str]] = []

    required_source_files = ("models/svd_wrapper.py", "data/cmu_clicks/realw_0.png")
    if not source.is_dir():
        missing.append(
            {
                "kind": "source_repo",
                "path": str(source),
                "reason": (
                    f"EgoWM checkout is not staged; clone {OFFICIAL_REPO_URL} here or set "
                    f"{SOURCE_ENV_VAR}"
                ),
            }
        )
    else:
        for relative in required_source_files:
            path = source / relative
            if not path.is_file():
                missing.append(
                    {"kind": "source_repo", "path": str(path), "reason": "official EgoWM source asset is missing"}
                )

    if entrypoint is None or not Path(str(entrypoint)).is_file():
        missing.append(
            {
                "kind": "entrypoint",
                "path": str(entrypoint or ""),
                "reason": "WorldFoundry EgoWM compatibility launcher is missing",
            }
        )

    checkpoint = _checkpoint_path(options)
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        missing.append(
            {"kind": "checkpoint", "path": str(checkpoint), "reason": "EgoWM 25-DoF checkpoint is missing or empty"}
        )

    base_model = _base_model_dir(options)
    for relative in (
        "model_index.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/diffusion_pytorch_model.fp16.safetensors",
        "image_encoder/model.fp16.safetensors",
    ):
        path = base_model / relative
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(
                {"kind": "checkpoint", "path": str(path), "reason": "Stable Video Diffusion base-model asset is missing or empty"}
            )

    image_path = _option(options, "image_path", "input_image")
    if image_path not in (None, ""):
        image = expand_worldfoundry_path(str(image_path))
        if not image.is_file():
            missing.append({"kind": "asset", "path": str(image), "reason": "EgoWM input image is missing"})

    for module_name in ("torch", "diffusers", "transformers", "PIL", "einops", "imageio"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(
                {
                    "kind": "python_module",
                    "path": module_name,
                    "reason": f"required EgoWM dependency {module_name!r} is not importable",
                }
            )
    return missing


def build_command(context: Mapping[str, Any]) -> list[str]:
    """Build the compatibility-launcher command for official EgoWM SVD inference."""
    settings = command_settings(context)
    source = Path(str(context["runtime_root"]))
    image_path = _option(settings, "image_path", "input_image", default=source / "data/cmu_clicks/realw_0.png")
    command = [
        str(context["python"]),
        str(context["entrypoint"]),
        "--source-dir",
        str(source),
        "--checkpoint-path",
        str(_checkpoint_path(settings)),
        "--base-model-dir",
        str(_base_model_dir(settings)),
        "--input-image",
        str(expand_worldfoundry_path(str(image_path))),
        "--output-path",
        str(context["output_path"]),
        "--device",
        str(context.get("device") or "cuda"),
        "--num-frames",
        str(settings.get("num_frames", 8)),
        "--num-inference-steps",
        str(settings.get("num_inference_steps", 8)),
        "--fps",
        str(settings.get("fps", 7)),
        "--seed",
        str(settings.get("seed", 42)),
        "--height",
        str(settings.get("height", 512)),
        "--width",
        str(settings.get("width", 512)),
        "--motion-bucket-id",
        str(settings.get("motion_bucket_id", 180)),
        "--action-scale",
        str(settings.get("action_scale", 0.0)),
    ]
    return command


__all__ = [
    "BLOCKED_REASON",
    "DEFAULT_BASE_MODEL_DIR",
    "DEFAULT_CHECKPOINT_PATH",
    "OFFICIAL_ENTRYPOINT",
    "RUNTIME_DIR",
    "SOURCE_ENV_VAR",
    "build_command",
    "missing_requirements",
    "runtime_root",
]

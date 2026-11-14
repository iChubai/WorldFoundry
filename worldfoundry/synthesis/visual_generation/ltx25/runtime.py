"""In-process input and artifact adapter for the native LTX-2.5 recipe.

All checkpoint loading, conditioning, sampling and decoding belong to the
canonical base-model components and shared joint-multistage runner. Prediction
uses local checkpoints and the active ltx-2.5 environment; it never downloads.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from worldfoundry.core.execution.deadline import check_deadline
from worldfoundry.runtime.assets import expand_worldfoundry_path, local_file_is_ready
from worldfoundry.runtime.env import resolve_hfd_root

if TYPE_CHECKING:
    from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline

UPSTREAM_REVISION = "a95ab856bf29407b6b066ede0abe1846050db56c"
CHECKPOINT_REVISION = "bf86adedf518142442575d1ce2e767b7d01c8c76"
DEFAULT_CHECKPOINT_ROOT = resolve_hfd_root() / "Lightricks--LTX-2.5"

DEFAULT_COMPONENTS = {
    "transformer_path": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoder_path": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video_vae_path": "vae/ltx-2.5-video-vae-bf16.safetensors",
    "audio_vae_path": "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "spatial_upsampler_path": ("latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
    "duration_head_path": "model_patches/ltx-2.5-duration-head-bf16.safetensors",
}


def _expand_path(value: str | Path | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    expanded = os.path.expandvars(str(value))
    return Path(expand_worldfoundry_path(expanded)).expanduser().resolve()


def _conditioning_images(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike, Mapping)):
        values: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        raise TypeError("LTX-2.5 images must be a local path, mapping, or sequence of those values.")

    resolved: list[dict[str, Any]] = []
    for index, item in enumerate(values):
        if isinstance(item, Mapping):
            path_value = item.get("path") or item.get("image") or item.get("image_path")
            frame_index = int(item.get("frame_index", item.get("frame_idx", 0)))
            strength = float(item.get("strength", item.get("image_strength", 1.0)))
            crf = item.get("crf")
        else:
            path_value = item
            frame_index = 0
            strength = 1.0
            crf = None
        path = _expand_path(path_value)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"LTX-2.5 conditioning image {index} not found: {path}")
        if frame_index < 0:
            raise ValueError("LTX-2.5 conditioning frame_index must be non-negative.")
        if strength < 0:
            raise ValueError("LTX-2.5 conditioning strength must be non-negative.")
        resolved.append({"path": str(path), "frame_index": frame_index, "strength": strength, "crf": crf})
    return resolved


@dataclass(frozen=True)
class LTX25RuntimePlan:
    """Normalized native request and placement for one audio-video artifact."""

    request: Mapping[str, Any]
    output_path: str
    task: str
    offload_mode: str
    quantization: str | None
    diffvae_optimization: str

    def to_dict(self) -> dict[str, Any]:
        """Describe the native recipe without importing or loading model weights."""
        return {
            "recipe": "ltx-2.5",
            "request": dict(self.request),
            "output_path": self.output_path,
            "task": self.task,
            "offload_mode": self.offload_mode,
            "quantization": self.quantization,
            "diffvae_optimization": self.diffvae_optimization,
            "upstream_revision": UPSTREAM_REVISION,
            "checkpoint_revision": CHECKPOINT_REVISION,
        }


class LTX25DistilledRuntime:
    """Expose the native LTX-2.5 recipe through the existing T2V/I2V surface."""

    transformer_path: Path
    text_encoder_path: Path
    video_vae_path: Path
    audio_vae_path: Path
    spatial_upsampler_path: Path
    duration_head_path: Path

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        *,
        transformer_path: str | Path | None = None,
        text_encoder_path: str | Path | None = None,
        video_vae_path: str | Path | None = None,
        audio_vae_path: str | Path | None = None,
        spatial_upsampler_path: str | Path | None = None,
        duration_head_path: str | Path | None = None,
        prompt_enhancer_gemma_root: str | Path | None = None,
        offload_mode: str = "cpu",
        quantization: str | None = None,
        diffvae_optimization: str = "chunked_eager",
        device: str = "cuda",
    ) -> None:
        self.checkpoint_dir = _expand_path(checkpoint_dir) or DEFAULT_CHECKPOINT_ROOT
        self.device = str(device)
        self.offload_mode = str(offload_mode).strip().lower()
        self.quantization = str(quantization).strip().lower() if quantization else None
        self.diffvae_optimization = str(diffvae_optimization).strip().lower()
        self._validate_execution_options(self.offload_mode, self.quantization, self.diffvae_optimization)
        overrides = {
            "transformer_path": transformer_path,
            "text_encoder_path": text_encoder_path,
            "video_vae_path": video_vae_path,
            "audio_vae_path": audio_vae_path,
            "spatial_upsampler_path": spatial_upsampler_path,
            "duration_head_path": duration_head_path,
        }
        for name, relative in DEFAULT_COMPONENTS.items():
            setattr(self, name, _expand_path(overrides[name]) or self.checkpoint_dir / relative)
        self.prompt_enhancer_gemma_root = _expand_path(prompt_enhancer_gemma_root)
        self._pipeline: NativeDiffusionPipeline | None = None
        self._pipeline_options: tuple[object, ...] | None = None

    @staticmethod
    def _validate_execution_options(offload: str, quantization: str | None, diffvae: str) -> None:
        if offload not in {"none", "cpu", "disk"}:
            raise ValueError("LTX-2.5 offload_mode must be one of: none, cpu, disk.")
        if quantization not in {None, "fp8-cast", "fp8-scaled-mm", "nvfp4-cast", "nvfp4-prequant"}:
            raise ValueError("Unsupported LTX-2.5 quantization policy.")
        if diffvae not in {"chunked_eager", "chunked_compile", "combined_compile", "blackwell_dsl"}:
            raise ValueError(f"Unsupported LTX-2.5 diffvae_optimization: {diffvae!r}")

    def preflight(self) -> dict[str, Any]:
        """Inspect local assets and tools without importing the model environment."""
        required = [getattr(self, name) for name in DEFAULT_COMPONENTS if name != "duration_head_path"]
        missing = [str(path) for path in required if not path.is_file()]
        incomplete = [str(path) for path in required if path.is_file() and not local_file_is_ready(path)]
        ffmpeg = shutil.which("ffmpeg")
        device_ready = self.device.lower().startswith(("cuda", "mps"))
        blocked = [] if device_ready else ["LTX-2.5 inference requires a CUDA or MPS accelerator."]
        blocked.extend(f"Checkpoint download is incomplete or empty: {path}" for path in incomplete)
        return {
            "status": "ready" if not missing and not incomplete and ffmpeg and device_ready else "blocked",
            "model_id": "ltx-2.5",
            "pipeline": "ltx-2.5",
            "checkpoint_dir": str(self.checkpoint_dir),
            "component_paths": {name: str(getattr(self, name)) for name in DEFAULT_COMPONENTS},
            "prompt_enhancer_gemma_root": str(self.prompt_enhancer_gemma_root)
            if self.prompt_enhancer_gemma_root
            else None,
            "missing_checkpoint_files": missing,
            "incomplete_checkpoint_files": incomplete,
            "missing_system_tools": [] if ffmpeg else ["ffmpeg"],
            "blocked_reasons": blocked,
            "device": self.device,
            "device_ready": device_ready,
            "ffmpeg": ffmpeg,
            "license": "LTX-2.x Community License Agreement (August 11, 2026)",
            "upstream_revision": UPSTREAM_REVISION,
            "checkpoint_revision": CHECKPOINT_REVISION,
        }

    def build_plan(self, *, request: Mapping[str, Any], output_path: str | Path) -> LTX25RuntimePlan:
        """Validate public request options before constructing native components."""
        prompt = str(request.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("LTX-2.5 inference requires a non-empty prompt.")
        output = Path(output_path).expanduser().resolve()
        if output.suffix.lower() != ".mp4":
            output = output.with_suffix(".mp4")
        images = _conditioning_images(request.get("images") or request.get("image"))
        height, width = int(request.get("height", 512)), int(request.get("width", 768))
        frame_rate = float(request.get("fps", request.get("frame_rate", 24.0)))
        offload = str(request.get("offload_mode", self.offload_mode)).strip().lower()
        quantization = request.get("quantization", self.quantization)
        quantization = str(quantization).strip().lower() if quantization else None
        diffvae = str(request.get("diffvae_optimization", self.diffvae_optimization)).strip().lower()
        self._validate_execution_options(offload, quantization, diffvae)
        if height <= 0 or width <= 0 or height % 64 or width % 64:
            raise ValueError("Two-stage LTX-2.5 height and width must be positive multiples of 64.")
        if frame_rate <= 0:
            raise ValueError("LTX-2.5 frame_rate must be positive.")
        auto_duration = request.get("auto_duration")
        if request.get("num_frames") is not None or auto_duration is None:
            num_frames = int(request.get("num_frames", 121))
            if num_frames < 1 or (num_frames - 1) % 8:
                raise ValueError("LTX-2.5 num_frames must satisfy num_frames = 8 * k + 1.")
            auto_duration = None
        else:
            if not self.duration_head_path.is_file():
                raise FileNotFoundError(f"LTX-2.5 auto_duration requires duration_head_path: {self.duration_head_path}")
            if not isinstance(auto_duration, Sequence) or len(auto_duration) != 2:
                raise ValueError("LTX-2.5 auto_duration must contain [min_seconds, max_seconds].")
            auto_duration = tuple(float(value) for value in auto_duration)
            num_frames = None
        enhance_prompt = bool(request.get("enhance_prompt", False))
        if enhance_prompt and (self.prompt_enhancer_gemma_root is None or not self.prompt_enhancer_gemma_root.is_dir()):
            raise FileNotFoundError("LTX-2.5 enhance_prompt requires a local prompt_enhancer_gemma_root.")
        generated_keyframes = int(request.get("num_generated_keyframes", 0))
        if generated_keyframes < 0:
            raise ValueError("LTX-2.5 num_generated_keyframes must be non-negative.")
        return LTX25RuntimePlan(
            request={
                "prompt": prompt,
                "height": height,
                "width": width,
                "frame_rate": frame_rate,
                "seed": int(request.get("seed", 42)),
                "num_frames": num_frames,
                "auto_duration": auto_duration,
                "images": images,
                "enhance_prompt": enhance_prompt,
                "num_generated_keyframes": generated_keyframes,
            },
            output_path=str(output),
            task="image-to-video" if images else "text-to-video",
            offload_mode=offload,
            quantization=quantization,
            diffvae_optimization=diffvae,
        )

    def _native_pipeline(self, plan: LTX25RuntimePlan) -> NativeDiffusionPipeline:
        import torch

        from worldfoundry.base_models.diffusion_model import NativeDiffusionPipeline
        from worldfoundry.base_models.diffusion_model.loaders import CheckpointSpec
        from worldfoundry.base_models.diffusion_model.optimizations import (
            OffloadMode,
            OffloadPolicy,
            RuntimePolicy,
            parse_offload_policy,
            parse_quantization_policy,
        )

        load_duration = self.duration_head_path.is_file()
        enhance = bool(plan.request["enhance_prompt"])
        options = (plan.offload_mode, plan.quantization, plan.diffvae_optimization, load_duration, enhance)
        if self._pipeline is None or self._pipeline_options != options:
            self._pipeline = None
            self._pipeline_options = None
            offload = parse_offload_policy(plan.offload_mode, owner="LTX")
            if offload.mode is OffloadMode.BLOCK and torch.device(self.device).type == "mps":
                offload = OffloadPolicy(mode=OffloadMode.COMPONENT, target="cpu")
            checkpoints = {
                "transformer": self.transformer_path,
                "text_encoder": self.text_encoder_path,
                "video_vae": self.video_vae_path,
                "audio_vae": self.audio_vae_path,
                "upsampler": self.spatial_upsampler_path,
                "duration_head": self.duration_head_path,
            }
            conditioner_options: dict[str, object] = {"load_duration_head": load_duration}
            if enhance:
                conditioner_options["prompt_enhancer_checkpoint"] = CheckpointSpec(
                    source=str(self.prompt_enhancer_gemma_root)
                )
            self._pipeline = NativeDiffusionPipeline.from_pretrained(
                "ltx-2.5",
                policy=RuntimePolicy(
                    device=self.device,
                    dtype=torch.bfloat16,
                    offload=offload,
                    quantization=parse_quantization_policy(plan.quantization, owner="LTX"),
                ),
                checkpoint_overrides={name: CheckpointSpec(source=str(path)) for name, path in checkpoints.items()},
                component_options={
                    "conditioner:main": conditioner_options,
                    "decoder:main": {"diffvae_optimization": plan.diffvae_optimization},
                },
            )
            self._pipeline_options = options
        return self._pipeline

    def run_plan(self, plan: LTX25RuntimePlan, *, timeout_seconds: float = 7200) -> dict[str, Any]:
        """Run the native recipe and write its audio-video artifact in-process.

        The cooperative timeout is checked between stages/steps and before
        artifact writes. It cannot preempt a running CUDA kernel.
        """
        from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest, SamplingConfig
        from worldfoundry.core.media.codecs.pyav import save_audio_video_h264

        started = time.monotonic()
        deadline = started + timeout_seconds
        generated: list[str] = []
        error = None
        try:
            check_deadline(deadline, stage="LTX-2.5 component loading")
            pipeline = self._native_pipeline(plan)
            values = plan.request
            output = pipeline(
                DiffusionRequest(
                    prompt=values["prompt"],
                    height=values["height"],
                    width=values["width"],
                    num_frames=values["num_frames"] or 1,
                    sampling=SamplingConfig(num_inference_steps=11, seed=values["seed"], guidance_scale=1.0),
                    inputs={
                        "frame_rate": values["frame_rate"],
                        "images": values["images"],
                        "auto_duration": values["auto_duration"],
                        "enhance_prompt": values["enhance_prompt"],
                        "num_generated_keyframes": values["num_generated_keyframes"],
                        "deadline_monotonic": deadline,
                    },
                )
            )
            check_deadline(deadline, stage="LTX-2.5 artifact encoding")
            save_audio_video_h264(
                output.sample,
                plan.output_path,
                fps=values["frame_rate"],
                waveform=output.artifacts["audio"],
                sample_rate=int(output.artifacts["audio_sampling_rate"]),
                deadline_monotonic=deadline,
            )
            generated = [plan.output_path]
        except TimeoutError:
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        return {
            "ok": error is None,
            "status": "success" if error is None else "failed",
            "duration_seconds": round(time.monotonic() - started, 3),
            "generated_files": generated,
            "error": error,
            "runtime_plan": plan.to_dict(),
        }


__all__ = [
    "CHECKPOINT_REVISION",
    "DEFAULT_CHECKPOINT_ROOT",
    "DEFAULT_COMPONENTS",
    "UPSTREAM_REVISION",
    "LTX25DistilledRuntime",
    "LTX25RuntimePlan",
]

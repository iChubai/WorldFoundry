"""Training adapter for the native StepVideo text-to-video graph.

StepVideo latents are laid out ``[B, F, C, H, W]`` (frame before channel), the
codec applies spatial compression with no temporal downsampling, and the
denoiser reads separate ``prompt_embeds``/``clip_embeds``/``attention_mask``
conditioning.  The adapter freezes the codec and conditioner, translates flow
sigmas onto the model timestep scale, and calls the ordinary denoiser graph
with autograd enabled.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.components import ComponentKey, ComponentKind
from worldfoundry.base_models.diffusion_model.contracts import (
    Conditioning,
    DenoiserInput,
    DiffusionRequest,
    SamplingConfig,
)
from worldfoundry.training.api.contracts import ObjectiveBatch, PreparedBatch, TrainingBatch
from worldfoundry.training.models._common import (
    component_module as _component_module,
    freeze as _freeze,
    merge_without_overwrite as _common_merge,
    module_device_dtype as _module_device_dtype,
    normalize_effective_sigmas as _normalize_effective_sigmas,
    positive_int as _positive_int,
)

STEP_VIDEO_DEFAULT_TRAIN_TIMESTEPS = 1000
# StepVideo's AdaLayerNormSingle rescales the incoming timestep by 1000
# internally ("timestep usually in [0,1], we rescale it to [0,1000]"), so the
# adapter must feed the raw flow sigma in [0, 1] rather than sigma * 1000.
STEP_VIDEO_DEFAULT_MODEL_TIMESTEP_SCALE = 1.0
STEP_VIDEO_DEFAULT_FPS = 25.0


def _merge_without_overwrite(
    destination: dict[str, object],
    source: Mapping[str, object],
    *,
    source_name: str,
) -> None:
    _common_merge(destination, source, source_name=source_name, family="StepVideo")


class StepVideoTrainAdapter:
    """Own StepVideo conditioning, latent geometry, and training forward."""

    prediction_type = "flow_velocity"
    lora_target_preset = "step-video-attention"
    conditioning_dropout_owner = "none"
    model_family = "step-video"

    def __init__(
        self,
        denoiser: object,
        codec: object | None,
        conditioner: object | None,
        *,
        model_timestep_scale: float = STEP_VIDEO_DEFAULT_MODEL_TIMESTEP_SCALE,
        num_train_timesteps: int = STEP_VIDEO_DEFAULT_TRAIN_TIMESTEPS,
        expected_latent_channels: int = 64,
        spatial_compression: int = 16,
        default_fps: float = STEP_VIDEO_DEFAULT_FPS,
    ) -> None:
        trainable_module = _component_module(denoiser, "model")
        if trainable_module is None:
            raise TypeError("StepVideo denoiser must expose an nn.Module as 'model'")
        if not callable(denoiser):
            raise TypeError("StepVideo denoiser must be callable")
        if codec is not None and not callable(getattr(codec, "encode", None)):
            raise TypeError("StepVideo codec must expose encode(videos)")
        if conditioner is not None and not callable(getattr(conditioner, "encode", None)):
            raise TypeError("StepVideo conditioner must expose encode(request, device=..., dtype=...)")

        scale = float(model_timestep_scale)
        if not isfinite(scale) or scale <= 0:
            raise ValueError("model_timestep_scale must be finite and positive")
        fps = float(default_fps)
        if not isfinite(fps) or fps <= 0:
            raise ValueError("default_fps must be finite and positive")

        self.denoiser = denoiser
        self.codec = codec
        self.conditioner = conditioner
        self.trainable_module = trainable_module
        self.model_timestep_scale = scale
        self.num_train_timesteps = _positive_int(num_train_timesteps, field_name="num_train_timesteps")
        if self.num_train_timesteps < 2:
            raise ValueError("num_train_timesteps must be at least two")
        self.expected_latent_channels = _positive_int(
            expected_latent_channels,
            field_name="expected_latent_channels",
        )
        self.spatial_compression = _positive_int(spatial_compression, field_name="spatial_compression")
        self.default_fps = fps

        codec_module = _component_module(codec, "vae", "model")
        conditioner_module = _component_module(conditioner, "text_encoder", "encoder", "model")
        self.frozen_modules = tuple(
            dict.fromkeys(module for module in (codec_module, conditioner_module) if module is not None)
        )
        for module in self.frozen_modules:
            _freeze(module)
        self.trainable_module.train()

        blocks = getattr(self.trainable_module, "transformer_blocks", ())
        self.fsdp_block_classes = tuple(dict.fromkeys(type(block) for block in blocks if isinstance(block, nn.Module)))
        self.conditioning_dropout_probability = 0.0

    def _keep_frozen_modules_in_eval(self) -> None:
        for module in self.frozen_modules:
            module.eval()

    def _encoded_conditioning(
        self,
        batch: TrainingBatch,
        *,
        frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, object]:
        raw = dict(batch.conditions)
        for reserved in ("clean_latents", "latent_loss_mask", "valid_latent_mask"):
            raw.pop(reserved, None)

        has_precomputed = "prompt_embeds" in raw
        if has_precomputed:
            values = raw
        else:
            if self.conditioner is None:
                raise RuntimeError(
                    "StepVideo batch has no precomputed prompt_embeds and the adapter has no conditioner"
                )
            raw_inputs = batch.metadata.get("conditioner_inputs", {})
            if not isinstance(raw_inputs, Mapping):
                raise TypeError("metadata.conditioner_inputs must be a mapping")
            request = DiffusionRequest(
                prompt=batch.prompts,
                height=height,
                width=width,
                num_frames=frames,
                sampling=SamplingConfig(guidance_scale=1.0),
                inputs=dict(raw_inputs),
                metadata=dict(batch.metadata),
            )
            with torch.no_grad():
                encoded = self.conditioner.encode(request, device=device, dtype=dtype)
            if not isinstance(encoded, Conditioning):
                raise TypeError(f"StepVideo conditioner returned {type(encoded).__name__}, expected Conditioning")
            values = dict(encoded.positive)
            _merge_without_overwrite(values, encoded.shared, source_name="conditioner.shared")
            _merge_without_overwrite(values, raw, source_name="TrainingBatch.conditions")

        normalized: dict[str, object] = {}
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                target_dtype = dtype if value.is_floating_point() else value.dtype
                normalized[key] = value.detach().to(device=device, dtype=target_dtype)
            else:
                normalized[key] = value
        normalized.setdefault("fps", self.default_fps)

        for required in ("prompt_embeds", "clip_embeds", "attention_mask"):
            value = normalized.get(required)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"StepVideo conditioning requires a tensor {required!r}")
            if int(value.shape[0]) != batch.batch_size:
                raise ValueError(
                    f"StepVideo {required} must start with batch size {batch.batch_size}; got {tuple(value.shape)}"
                )
        if not bool(torch.isfinite(normalized["prompt_embeds"]).all()):
            raise ValueError("StepVideo prompt_embeds contains NaN or infinity")
        return normalized

    def _validate_latent_geometry(self, clean_latents: torch.Tensor, *, batch_size: int) -> None:
        if clean_latents.ndim != 5:
            raise ValueError(f"StepVideo clean_latents must be [B,F,C,H,W]; got {tuple(clean_latents.shape)}")
        if int(clean_latents.shape[0]) != batch_size:
            raise ValueError("StepVideo clean latents do not match the batch dimension")
        if int(clean_latents.shape[2]) != self.expected_latent_channels:
            raise ValueError(
                f"StepVideo clean latents contain {int(clean_latents.shape[2])} channels; "
                f"expected {self.expected_latent_channels}"
            )

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        if not isinstance(batch, TrainingBatch):
            raise TypeError("batch must be a TrainingBatch")
        pixels = batch.pixel_values
        cached_latents = batch.conditions.get("clean_latents")
        if cached_latents is not None and pixels is not None:
            raise ValueError("StepVideo batch cannot provide both pixel_values and precomputed clean_latents")

        self._keep_frozen_modules_in_eval()
        device, dtype = _module_device_dtype(self.trainable_module)
        if cached_latents is not None:
            if not isinstance(cached_latents, torch.Tensor):
                raise TypeError("precomputed StepVideo clean_latents must be a torch.Tensor")
            clean_latents = cached_latents
            pixel_shape: tuple[int, ...] | None = None
        else:
            if not isinstance(pixels, torch.Tensor):
                raise TypeError("StepVideo training requires pixel_values or precomputed clean_latents")
            if pixels.ndim != 5 or int(pixels.shape[1]) != 3:
                raise ValueError(f"StepVideo pixels must be [B,3,T,H,W]; got {tuple(pixels.shape)}")
            if self.codec is None:
                raise RuntimeError("StepVideo batch has no precomputed clean_latents and the adapter has no codec")
            with torch.no_grad():
                clean_latents = self.codec.encode(pixels)
            pixel_shape = tuple(int(value) for value in pixels.shape)

        if not isinstance(clean_latents, torch.Tensor):
            raise TypeError(f"StepVideo clean latents must be a tensor; got {type(clean_latents).__name__}")
        self._validate_latent_geometry(clean_latents, batch_size=batch.batch_size)
        clean_latents = clean_latents.detach().to(device=device, dtype=dtype)

        latent_frames = int(clean_latents.shape[1])
        latent_height = int(clean_latents.shape[3])
        latent_width = int(clean_latents.shape[4])
        conditioning = self._encoded_conditioning(
            batch,
            frames=latent_frames,
            height=latent_height * self.spatial_compression,
            width=latent_width * self.spatial_compression,
            device=device,
            dtype=dtype,
        )

        loss_mask = batch.conditions.get("latent_loss_mask")
        if loss_mask is not None:
            if not isinstance(loss_mask, torch.Tensor):
                raise TypeError("precomputed StepVideo latent_loss_mask must be a torch.Tensor")
            loss_mask = loss_mask.detach().to(device=device, dtype=torch.float32)
        sample_weights = batch.sample_weights
        if isinstance(sample_weights, torch.Tensor):
            sample_weights = sample_weights.to(device=device, dtype=torch.float32)

        metadata: dict[str, Any] = dict(batch.metadata)
        metadata.update(
            {
                "model_family": self.model_family,
                "prediction_type": self.prediction_type,
                "model_timestep_scale": self.model_timestep_scale,
                "num_train_timesteps": self.num_train_timesteps,
                "pixel_shape": pixel_shape,
                "latent_shape": tuple(int(value) for value in clean_latents.shape),
                "precomputed_latents": cached_latents is not None,
                "conditioning_dropout_owner": self.conditioning_dropout_owner,
                "conditioning_dropout_probability": self.conditioning_dropout_probability,
            }
        )
        return PreparedBatch(
            sample_ids=batch.sample_ids,
            clean_latents=clean_latents,
            conditioning=conditioning,
            loss_mask=loss_mask,
            sample_weights=sample_weights,
            metadata=metadata,
        )

    def forward_train(self, batch: ObjectiveBatch) -> torch.Tensor:
        return self.forward_model(batch, training=True)

    def forward_model(
        self,
        batch: ObjectiveBatch,
        *,
        training: bool,
        branch: str = "positive",
    ) -> torch.Tensor:
        """Forward a trainable policy or frozen teacher/reference role."""

        if not isinstance(training, bool):
            raise TypeError("training must be a bool")
        if branch not in {"positive", "negative"}:
            raise ValueError("branch must be 'positive' or 'negative'")
        if not isinstance(batch, ObjectiveBatch):
            raise TypeError("batch must be an ObjectiveBatch")
        if isinstance(batch.model_input, Mapping) or not isinstance(batch.model_input, torch.Tensor):
            raise TypeError("StepVideo training requires one tensor model_input")
        latents = batch.model_input
        if latents.ndim != 5 or int(latents.shape[2]) != self.expected_latent_channels:
            raise ValueError(
                f"StepVideo model_input must be [B,F,{self.expected_latent_channels},H,W]; got {tuple(latents.shape)}"
            )
        prediction_type = batch.metadata.get("prediction_type")
        if prediction_type is not None and prediction_type != self.prediction_type:
            raise ValueError(f"StepVideo adapter predicts {self.prediction_type!r}, got objective {prediction_type!r}")
        sigmas = _normalize_effective_sigmas(
            batch.sigmas,
            batch_size=int(latents.shape[0]),
            device=latents.device,
            family="StepVideo",
        )

        self.trainable_module.train(training)
        self._keep_frozen_modules_in_eval()
        model_timesteps = sigmas * self.model_timestep_scale
        output = self.denoiser(
            DenoiserInput(
                latents=latents,
                timestep=model_timesteps,
                next_timestep=torch.zeros_like(model_timesteps),
                conditioning=dict(batch.conditioning),
                step_index=0,
                total_steps=self.num_train_timesteps,
                branch=branch,
            )
        )
        sample = getattr(output, "sample", None)
        if not isinstance(sample, torch.Tensor):
            raise TypeError(f"StepVideo denoiser returned {type(output).__name__} without a tensor sample")
        if sample.shape != latents.shape:
            raise ValueError(
                f"StepVideo prediction shape {tuple(sample.shape)} does not match input {tuple(latents.shape)}"
            )
        return sample


def build_step_video_train_adapter(
    components: Mapping[ComponentKey, object],
    **options: object,
) -> StepVideoTrainAdapter:
    """Build a raw-video StepVideo adapter from native training components."""

    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping")

    def require(kind: ComponentKind, name: str = "main") -> object:
        key = ComponentKey(kind, name)
        try:
            return components[key]
        except KeyError as error:
            raise KeyError(f"StepVideo training components are missing {key}") from error

    return StepVideoTrainAdapter(
        denoiser=require(ComponentKind.DENOISER),
        codec=require(ComponentKind.DECODER),
        conditioner=require(ComponentKind.CONDITIONER),
        **options,
    )


def build_cached_step_video_train_adapter(
    components: Mapping[ComponentKey, object],
    **options: object,
) -> StepVideoTrainAdapter:
    """Build a StepVideo adapter that consumes only immutable cached features."""

    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping")
    denoiser_key = ComponentKey(ComponentKind.DENOISER)
    unexpected = sorted(str(key) for key in components if key != denoiser_key)
    if unexpected:
        raise ValueError(f"cached StepVideo training accepts only the denoiser component; got {unexpected}")
    try:
        denoiser = components[denoiser_key]
    except KeyError as error:
        raise KeyError(f"cached StepVideo training components are missing {denoiser_key}") from error
    return StepVideoTrainAdapter(denoiser=denoiser, codec=None, conditioner=None, **options)


__all__ = [
    "STEP_VIDEO_DEFAULT_FPS",
    "STEP_VIDEO_DEFAULT_MODEL_TIMESTEP_SCALE",
    "STEP_VIDEO_DEFAULT_TRAIN_TIMESTEPS",
    "StepVideoTrainAdapter",
    "build_cached_step_video_train_adapter",
    "build_step_video_train_adapter",
]

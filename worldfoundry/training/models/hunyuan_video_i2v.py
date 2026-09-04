"""Training adapters for the conditioned HunyuanVideo image-to-video graphs.

Two distinct conditioning mechanisms share the HunyuanVideo family:

* **Original I2V (token replacement)** — the first latent frame is a clean image
  condition.  The denoiser zeroes the model's first-frame *output*, so a correct
  training forward must (a) keep the first frame clean in the model input after
  corruption and (b) mask the first frame out of the flow-matching loss.  This
  mirrors the Echo context-suffix freeze, but on the temporal *prefix*.

* **HunyuanVideo 1.5 (concat condition)** — the condition latents are
  concatenated on the *channel* axis inside the denoiser, not stored as clean
  frames of the trajectory.  Every frame is generated and scored; the adapter
  only threads ``condition_latents`` plus the glyph (ByT5) and optional vision
  conditioning through to the denoiser.

Both use the ``sigma * num_train_timesteps`` timestep domain (the bound
HunyuanVideo flow-match scheduler subclasses Wan's) so ``model_timestep_scale``
stays 1000.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.components import ComponentKey, ComponentKind
from worldfoundry.base_models.diffusion_model.contracts import DenoiserInput
from worldfoundry.training.api.contracts import ObjectiveBatch, PreparedBatch, TrainingBatch
from worldfoundry.training.models._common import (
    component_module as _component_module,
    freeze as _freeze,
    module_device_dtype as _module_device_dtype,
    normalize_effective_sigmas as _normalize_effective_sigmas,
    positive_int as _positive_int,
)

HUNYUAN_VIDEO_I2V_DEFAULT_TRAIN_TIMESTEPS = 1000

_CLEAN_PREFIX_KEY = "_hunyuan_clean_prefix"


class HunyuanVideoI2VTrainAdapter:
    """Original token-replacement I2V: clean first frame, first-frame loss mask."""

    prediction_type = "flow_velocity"
    lora_target_preset = "hunyuan-video-attention"
    conditioning_dropout_owner = "none"
    model_family = "hunyuan-video-i2v"

    def __init__(
        self,
        denoiser: object,
        codec: object | None,
        conditioner: object | None,
        *,
        model_timestep_scale: float = HUNYUAN_VIDEO_I2V_DEFAULT_TRAIN_TIMESTEPS,
        num_train_timesteps: int = HUNYUAN_VIDEO_I2V_DEFAULT_TRAIN_TIMESTEPS,
        expected_latent_channels: int = 16,
        temporal_compression: int = 4,
        spatial_compression: int = 8,
    ) -> None:
        trainable_module = _component_module(denoiser, "model")
        if trainable_module is None:
            raise TypeError("HunyuanVideo I2V denoiser must expose an nn.Module as 'model'")
        if not callable(denoiser):
            raise TypeError("HunyuanVideo I2V denoiser must be callable")

        scale = float(model_timestep_scale)
        if not isfinite(scale) or scale <= 0:
            raise ValueError("model_timestep_scale must be finite and positive")

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
        self.temporal_compression = _positive_int(temporal_compression, field_name="temporal_compression")
        self.spatial_compression = _positive_int(spatial_compression, field_name="spatial_compression")

        codec_module = _component_module(codec, "vae", "model")
        conditioner_module = _component_module(conditioner, "text_encoder", "encoder", "model")
        self.frozen_modules = tuple(
            dict.fromkeys(module for module in (codec_module, conditioner_module) if module is not None)
        )
        for module in self.frozen_modules:
            _freeze(module)
        self.trainable_module.train()

        blocks = getattr(
            self.trainable_module,
            "blocks",
            getattr(self.trainable_module, "double_blocks", ()),
        )
        self.fsdp_block_classes = tuple(dict.fromkeys(type(block) for block in blocks if isinstance(block, nn.Module)))
        self.conditioning_dropout_probability = 0.0

    def _keep_frozen_modules_in_eval(self) -> None:
        for module in self.frozen_modules:
            module.eval()

    def _text_conditioning(
        self,
        batch: TrainingBatch,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, object]:
        values: dict[str, object] = {}
        for key in ("text_states", "text_mask", "text_states_2"):
            value = batch.conditions.get(key)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"HunyuanVideo I2V conditioning requires a tensor {key!r}")
            if int(value.shape[0]) != batch.batch_size:
                raise ValueError(f"HunyuanVideo I2V {key} batch dimension does not match the batch")
            values[key] = value.to(device=device, dtype=dtype)
        if "embedded_guidance_scale" in batch.conditions:
            values["embedded_guidance_scale"] = float(batch.conditions["embedded_guidance_scale"])
        return values

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        if not isinstance(batch, TrainingBatch):
            raise TypeError("batch must be a TrainingBatch")
        cached_latents = batch.conditions.get("clean_latents")
        if not isinstance(cached_latents, torch.Tensor):
            raise TypeError("HunyuanVideo I2V training requires precomputed clean_latents")
        if cached_latents.ndim != 5:
            raise ValueError(f"HunyuanVideo I2V clean_latents must be [B,C,T,H,W]; got {tuple(cached_latents.shape)}")

        self._keep_frozen_modules_in_eval()
        device, dtype = _module_device_dtype(self.trainable_module)
        clean_latents = cached_latents.detach().to(device=device, dtype=dtype)
        if int(clean_latents.shape[1]) != self.expected_latent_channels:
            raise ValueError(
                f"HunyuanVideo I2V clean latents contain {int(clean_latents.shape[1])} channels; "
                f"expected {self.expected_latent_channels}"
            )
        total_frames = int(clean_latents.shape[2])
        if total_frames < 2:
            raise ValueError("HunyuanVideo I2V requires at least two latent frames (conditioning + generated)")

        conditioning = self._text_conditioning(batch, device=device, dtype=dtype)
        # The clean first frame is the image condition; keep a copy to re-freeze
        # after corruption and mask it out of the loss.
        conditioning[_CLEAN_PREFIX_KEY] = clean_latents[:, :, :1].clone()

        loss_mask = torch.ones((batch.batch_size, 1, total_frames, 1, 1), device=device, dtype=torch.float32)
        loss_mask[:, :, :1] = 0.0

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
                "latent_shape": tuple(int(value) for value in clean_latents.shape),
                "precomputed_latents": True,
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
        if not isinstance(training, bool):
            raise TypeError("training must be a bool")
        if branch not in {"positive", "negative"}:
            raise ValueError("branch must be 'positive' or 'negative'")
        if not isinstance(batch, ObjectiveBatch):
            raise TypeError("batch must be an ObjectiveBatch")
        if isinstance(batch.model_input, Mapping) or not isinstance(batch.model_input, torch.Tensor):
            raise TypeError("HunyuanVideo I2V training requires one tensor model_input")
        latents = batch.model_input
        if latents.ndim != 5 or int(latents.shape[1]) != self.expected_latent_channels:
            channels = self.expected_latent_channels
            raise ValueError(f"HunyuanVideo I2V model_input must be [B,{channels},T,H,W]; got {tuple(latents.shape)}")
        prediction_type = batch.metadata.get("prediction_type")
        if prediction_type is not None and prediction_type != self.prediction_type:
            raise ValueError(
                f"HunyuanVideo I2V adapter predicts {self.prediction_type!r}, got objective {prediction_type!r}"
            )

        conditioning = dict(batch.conditioning)
        clean_prefix = conditioning.pop(_CLEAN_PREFIX_KEY, None)
        if not isinstance(clean_prefix, torch.Tensor):
            raise ValueError("HunyuanVideo I2V forward requires the clean first frame threaded from prepare_batch")

        sigmas = _normalize_effective_sigmas(
            batch.sigmas,
            batch_size=int(latents.shape[0]),
            device=latents.device,
            family="HunyuanVideo I2V",
        )

        # Re-freeze the image-condition frame to its clean value.
        model_latents = latents.clone()
        model_latents[:, :, :1] = clean_prefix.to(device=latents.device, dtype=latents.dtype)

        self.trainable_module.train(training)
        self._keep_frozen_modules_in_eval()
        model_timesteps = sigmas * self.model_timestep_scale
        output = self.denoiser(
            DenoiserInput(
                latents=model_latents,
                timestep=model_timesteps,
                next_timestep=torch.zeros_like(model_timesteps),
                conditioning=conditioning,
                step_index=0,
                total_steps=self.num_train_timesteps,
                branch=branch,
            )
        )
        sample = getattr(output, "sample", None)
        if not isinstance(sample, torch.Tensor):
            raise TypeError(f"HunyuanVideo I2V denoiser returned {type(output).__name__} without a tensor sample")
        if sample.shape != latents.shape:
            raise ValueError(
                f"HunyuanVideo I2V prediction shape {tuple(sample.shape)} does not match input {tuple(latents.shape)}"
            )
        return sample


class HunyuanVideo15TrainAdapter:
    """HunyuanVideo 1.5 concat-condition adapter: 32-channel, full-frame loss."""

    prediction_type = "flow_velocity"
    lora_target_preset = "hunyuan-video-attention"
    conditioning_dropout_owner = "none"
    model_family = "hunyuan-video-1.5"

    def __init__(
        self,
        denoiser: object,
        codec: object | None,
        conditioner: object | None,
        *,
        model_timestep_scale: float = HUNYUAN_VIDEO_I2V_DEFAULT_TRAIN_TIMESTEPS,
        num_train_timesteps: int = HUNYUAN_VIDEO_I2V_DEFAULT_TRAIN_TIMESTEPS,
        expected_latent_channels: int = 32,
        temporal_compression: int = 4,
        spatial_compression: int = 16,
    ) -> None:
        trainable_module = _component_module(denoiser, "model")
        if trainable_module is None:
            raise TypeError("HunyuanVideo 1.5 denoiser must expose an nn.Module as 'model'")
        if not callable(denoiser):
            raise TypeError("HunyuanVideo 1.5 denoiser must be callable")

        scale = float(model_timestep_scale)
        if not isfinite(scale) or scale <= 0:
            raise ValueError("model_timestep_scale must be finite and positive")

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
        self.temporal_compression = _positive_int(temporal_compression, field_name="temporal_compression")
        self.spatial_compression = _positive_int(spatial_compression, field_name="spatial_compression")

        codec_module = _component_module(codec, "vae", "model")
        conditioner_module = _component_module(conditioner, "text_encoder", "encoder", "model")
        self.frozen_modules = tuple(
            dict.fromkeys(module for module in (codec_module, conditioner_module) if module is not None)
        )
        for module in self.frozen_modules:
            _freeze(module)
        self.trainable_module.train()

        blocks = getattr(
            self.trainable_module,
            "blocks",
            getattr(self.trainable_module, "transformer_blocks", getattr(self.trainable_module, "double_blocks", ())),
        )
        self.fsdp_block_classes = tuple(dict.fromkeys(type(block) for block in blocks if isinstance(block, nn.Module)))
        self.conditioning_dropout_probability = 0.0

    def _keep_frozen_modules_in_eval(self) -> None:
        for module in self.frozen_modules:
            module.eval()

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        if not isinstance(batch, TrainingBatch):
            raise TypeError("batch must be a TrainingBatch")
        cached_latents = batch.conditions.get("clean_latents")
        if not isinstance(cached_latents, torch.Tensor):
            raise TypeError("HunyuanVideo 1.5 training requires precomputed clean_latents")
        if cached_latents.ndim != 5:
            raise ValueError(f"HunyuanVideo 1.5 clean_latents must be [B,C,T,H,W]; got {tuple(cached_latents.shape)}")

        self._keep_frozen_modules_in_eval()
        device, dtype = _module_device_dtype(self.trainable_module)
        clean_latents = cached_latents.detach().to(device=device, dtype=dtype)
        if int(clean_latents.shape[1]) != self.expected_latent_channels:
            raise ValueError(
                f"HunyuanVideo 1.5 clean latents contain {int(clean_latents.shape[1])} channels; "
                f"expected {self.expected_latent_channels}"
            )

        conditioning: dict[str, object] = {}
        for key in ("text_states", "text_mask", "condition_latents", "byt5_text_states", "byt5_text_mask"):
            value = batch.conditions.get(key)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"HunyuanVideo 1.5 conditioning requires a tensor {key!r}")
            conditioning[key] = value.to(device=device, dtype=dtype)
        condition = conditioning["condition_latents"]
        if condition.shape[0] != clean_latents.shape[0] or condition.shape[2:] != clean_latents.shape[2:]:
            raise ValueError("HunyuanVideo 1.5 condition_latents geometry must match the noise trajectory")
        for optional in ("text_states_2", "vision_states"):
            value = batch.conditions.get(optional)
            if isinstance(value, torch.Tensor):
                conditioning[optional] = value.to(device=device, dtype=dtype)

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
                "latent_shape": tuple(int(value) for value in clean_latents.shape),
                "precomputed_latents": True,
                "conditioning_dropout_owner": self.conditioning_dropout_owner,
                "conditioning_dropout_probability": self.conditioning_dropout_probability,
            }
        )
        return PreparedBatch(
            sample_ids=batch.sample_ids,
            clean_latents=clean_latents,
            conditioning=conditioning,
            loss_mask=None,
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
        if not isinstance(training, bool):
            raise TypeError("training must be a bool")
        if branch not in {"positive", "negative"}:
            raise ValueError("branch must be 'positive' or 'negative'")
        if not isinstance(batch, ObjectiveBatch):
            raise TypeError("batch must be an ObjectiveBatch")
        if isinstance(batch.model_input, Mapping) or not isinstance(batch.model_input, torch.Tensor):
            raise TypeError("HunyuanVideo 1.5 training requires one tensor model_input")
        latents = batch.model_input
        if latents.ndim != 5 or int(latents.shape[1]) != self.expected_latent_channels:
            channels = self.expected_latent_channels
            raise ValueError(f"HunyuanVideo 1.5 model_input must be [B,{channels},T,H,W]; got {tuple(latents.shape)}")
        prediction_type = batch.metadata.get("prediction_type")
        if prediction_type is not None and prediction_type != self.prediction_type:
            raise ValueError(
                f"HunyuanVideo 1.5 adapter predicts {self.prediction_type!r}, got objective {prediction_type!r}"
            )
        sigmas = _normalize_effective_sigmas(
            batch.sigmas,
            batch_size=int(latents.shape[0]),
            device=latents.device,
            family="HunyuanVideo 1.5",
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
            raise TypeError(f"HunyuanVideo 1.5 denoiser returned {type(output).__name__} without a tensor sample")
        if sample.shape != latents.shape:
            raise ValueError(
                f"HunyuanVideo 1.5 prediction shape {tuple(sample.shape)} does not match input {tuple(latents.shape)}"
            )
        return sample


def build_cached_hunyuan_video_i2v_train_adapter(
    components: Mapping[ComponentKey, object],
    **options: object,
) -> HunyuanVideoI2VTrainAdapter:
    """Build an original-I2V adapter that consumes only immutable cached features."""

    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping")
    denoiser_key = ComponentKey(ComponentKind.DENOISER)
    unexpected = sorted(str(key) for key in components if key != denoiser_key)
    if unexpected:
        raise ValueError(f"cached HunyuanVideo I2V training accepts only the denoiser component; got {unexpected}")
    try:
        denoiser = components[denoiser_key]
    except KeyError as error:
        raise KeyError(f"cached HunyuanVideo I2V training components are missing {denoiser_key}") from error
    return HunyuanVideoI2VTrainAdapter(denoiser=denoiser, codec=None, conditioner=None, **options)


def build_cached_hunyuan_video15_train_adapter(
    components: Mapping[ComponentKey, object],
    **options: object,
) -> HunyuanVideo15TrainAdapter:
    """Build a HunyuanVideo 1.5 concat-condition adapter from cached features."""

    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping")
    denoiser_key = ComponentKey(ComponentKind.DENOISER)
    unexpected = sorted(str(key) for key in components if key != denoiser_key)
    if unexpected:
        raise ValueError(f"cached HunyuanVideo 1.5 training accepts only the denoiser component; got {unexpected}")
    try:
        denoiser = components[denoiser_key]
    except KeyError as error:
        raise KeyError(f"cached HunyuanVideo 1.5 training components are missing {denoiser_key}") from error
    return HunyuanVideo15TrainAdapter(denoiser=denoiser, codec=None, conditioner=None, **options)


__all__ = [
    "HUNYUAN_VIDEO_I2V_DEFAULT_TRAIN_TIMESTEPS",
    "HunyuanVideo15TrainAdapter",
    "HunyuanVideoI2VTrainAdapter",
    "build_cached_hunyuan_video15_train_adapter",
    "build_cached_hunyuan_video_i2v_train_adapter",
]

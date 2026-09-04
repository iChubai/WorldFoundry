"""Training adapter for the native Cosmos1 GEN3C (EDM x0) video graph.

Cosmos1's GEN3C denoiser applies EDM preconditioning and returns a clean ``x0``
estimate, so it trains against the x0 objective (EDM weighting) rather than the
flow-velocity seam.  It is a pose-conditioned video2world model: the denoiser
freezes the condition frames (``active_indicator``) to the true latents and
concatenates an input mask + camera pose on the channel axis.  The adapter
reuses the denoiser directly, threads the six required condition tensors, passes
the raw Karras ``sigma`` as timestep (``model_timestep_scale = 1``; the denoiser
derives ``c_noise = 0.25*ln(sigma)`` internally), and masks the condition frames
out of the loss.
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
    positive_int as _positive_int,
)
from worldfoundry.training.objectives.x0_diffusion import X0_PREDICTION_TYPE

COSMOS1_DEFAULT_FPS = 24.0
_REQUIRED_CONDITIONS = (
    "context",
    "condition_latents",
    "condition_indicator",
    "condition_video_input_mask",
    "condition_video_pose",
    "condition_noise",
)


def _positive_sigmas(sigmas: object, *, batch_size: int, device: torch.device) -> torch.Tensor:
    if not isinstance(sigmas, torch.Tensor):
        raise TypeError("Cosmos1 objective sigmas must be a torch.Tensor")
    resolved = sigmas.to(device=device, dtype=torch.float32)
    if resolved.ndim == 0:
        resolved = resolved.expand(batch_size)
    elif resolved.numel() == batch_size:
        resolved = resolved.reshape(batch_size)
    else:
        raise ValueError("Cosmos1 requires one sigma per sample")
    if not bool(torch.isfinite(resolved).all()) or not bool((resolved > 0).all()):
        raise ValueError("Cosmos1 Karras sigmas must be finite and positive")
    return resolved


class Cosmos1TrainAdapter:
    """Own Cosmos1 GEN3C pose-conditioned x0 training forward."""

    prediction_type = X0_PREDICTION_TYPE
    lora_target_preset = "cosmos1-attention"
    conditioning_dropout_owner = "none"
    model_family = "cosmos1-gen3c-video"

    def __init__(
        self,
        denoiser: object,
        codec: object | None,
        conditioner: object | None,
        *,
        model_timestep_scale: float = 1.0,
        expected_latent_channels: int = 16,
        temporal_compression: int = 4,
        spatial_compression: int = 8,
        default_fps: float = COSMOS1_DEFAULT_FPS,
    ) -> None:
        trainable_module = _component_module(denoiser, "model")
        if trainable_module is None:
            raise TypeError("Cosmos1 denoiser must expose an nn.Module as 'model'")
        if not callable(denoiser):
            raise TypeError("Cosmos1 denoiser must be callable")

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
        self.expected_latent_channels = _positive_int(
            expected_latent_channels,
            field_name="expected_latent_channels",
        )
        self.temporal_compression = _positive_int(temporal_compression, field_name="temporal_compression")
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

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        if not isinstance(batch, TrainingBatch):
            raise TypeError("batch must be a TrainingBatch")
        if batch.batch_size != 1:
            raise ValueError(
                "Cosmos1 training uses per-media conditioning; materialize one media per microbatch (batch size one)"
            )
        cached_latents = batch.conditions.get("clean_latents")
        if not isinstance(cached_latents, torch.Tensor):
            raise TypeError("Cosmos1 training requires precomputed clean_latents")
        if cached_latents.ndim != 5:
            raise ValueError(f"Cosmos1 clean_latents must be [B,C,T,H,W]; got {tuple(cached_latents.shape)}")

        self._keep_frozen_modules_in_eval()
        device, dtype = _module_device_dtype(self.trainable_module)
        clean_latents = cached_latents.detach().to(device=device, dtype=dtype)
        if int(clean_latents.shape[1]) != self.expected_latent_channels:
            raise ValueError(
                f"Cosmos1 clean latents contain {int(clean_latents.shape[1])} channels; "
                f"expected {self.expected_latent_channels}"
            )

        conditioning: dict[str, object] = {}
        for key in _REQUIRED_CONDITIONS:
            value = batch.conditions.get(key)
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Cosmos1 GEN3C training requires a tensor {key!r}")
            conditioning[key] = value.to(device=device, dtype=dtype)
        conditioning["fps"] = float(batch.conditions.get("fps", batch.metadata.get("fps", self.default_fps)))
        if "condition_augment_sigma" in batch.conditions:
            conditioning["condition_augment_sigma"] = float(batch.conditions["condition_augment_sigma"])
        for optional in ("padding_mask", "context_mask"):
            value = batch.conditions.get(optional)
            if isinstance(value, torch.Tensor):
                conditioning[optional] = value.to(device=device, dtype=dtype)

        indicator = conditioning["condition_indicator"]
        # active_indicator freezes the condition frames; score only generated frames.
        collapsed = indicator
        while collapsed.ndim > 5:
            collapsed = collapsed.squeeze(-1)
        loss_mask = 1.0 - collapsed.to(dtype=torch.float32)

        sample_weights = batch.sample_weights
        if isinstance(sample_weights, torch.Tensor):
            sample_weights = sample_weights.to(device=device, dtype=torch.float32)

        metadata: dict[str, Any] = dict(batch.metadata)
        metadata.update(
            {
                "model_family": self.model_family,
                "prediction_type": self.prediction_type,
                "model_timestep_scale": self.model_timestep_scale,
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
            raise TypeError("Cosmos1 training requires one tensor model_input")
        latents = batch.model_input
        if latents.ndim != 5 or int(latents.shape[1]) != self.expected_latent_channels:
            channels = self.expected_latent_channels
            raise ValueError(f"Cosmos1 model_input must be [B,{channels},T,H,W]; got {tuple(latents.shape)}")
        if int(latents.shape[0]) != 1:
            raise ValueError("Cosmos1 training forward requires batch size one")
        prediction_type = batch.metadata.get("prediction_type")
        if prediction_type is not None and prediction_type != self.prediction_type:
            raise ValueError(f"Cosmos1 adapter predicts {self.prediction_type!r}, got objective {prediction_type!r}")
        sigmas = _positive_sigmas(batch.sigmas, batch_size=int(latents.shape[0]), device=latents.device)

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
                total_steps=1000,
                branch=branch,
            )
        )
        sample = getattr(output, "sample", None)
        if not isinstance(sample, torch.Tensor):
            raise TypeError(f"Cosmos1 denoiser returned {type(output).__name__} without a tensor sample")
        if sample.shape != latents.shape:
            raise ValueError(
                f"Cosmos1 prediction shape {tuple(sample.shape)} does not match input {tuple(latents.shape)}"
            )
        return sample


def build_cached_cosmos1_train_adapter(
    components: Mapping[ComponentKey, object],
    **options: object,
) -> Cosmos1TrainAdapter:
    """Build a Cosmos1 GEN3C adapter that consumes only immutable cached features."""

    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping")
    denoiser_key = ComponentKey(ComponentKind.DENOISER)
    unexpected = sorted(str(key) for key in components if key != denoiser_key)
    if unexpected:
        raise ValueError(f"cached Cosmos1 training accepts only the denoiser component; got {unexpected}")
    try:
        denoiser = components[denoiser_key]
    except KeyError as error:
        raise KeyError(f"cached Cosmos1 training components are missing {denoiser_key}") from error
    return Cosmos1TrainAdapter(denoiser=denoiser, codec=None, conditioner=None, **options)


__all__ = [
    "COSMOS1_DEFAULT_FPS",
    "Cosmos1TrainAdapter",
    "build_cached_cosmos1_train_adapter",
]

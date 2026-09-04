"""Training adapter for the native Gamma-World bidirectional video graph.

Gamma-World bidirectional is a Cosmos-style conditioned flow-velocity model: the
denoiser blends the clean condition frames (``gt_frames`` where
``condition_video_input_mask`` is 1) over the corrupted latent internally, then
returns pure velocity.  So a correct training forward keeps ``model_input`` the
fully corrupted latent, threads the condition tensors plus the required
per-agent ``action_inputs`` through as conditioning, and masks the condition
frames out of the loss (``loss_mask = 1 - indicator``).  It must NOT pass
``block_noise``/``initial_noise`` — that would trigger the denoiser's inference
velocity-override on the condition frames, which the loss mask already handles.

The trainable module is exposed as ``.net`` (not ``.model``); the network
rescales the incoming timestep by ``timestep_scale`` internally, and the bound
Wan flow-unipc scheduler emits ``sigma * num_train_timesteps``, so the adapter
keeps ``model_timestep_scale = 1000``.  The causal and few-step variants (KV
cache / x0 output) are out of scope for this flow-velocity adapter.
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

GAMMA_WORLD_DEFAULT_TRAIN_TIMESTEPS = 1000


class GammaWorldBidirectionalTrainAdapter:
    """Own Gamma-World bidirectional conditioning and training forward."""

    prediction_type = "flow_velocity"
    lora_target_preset = "gamma-world-attention"
    conditioning_dropout_owner = "none"
    model_family = "gamma-world-bidirectional-video"

    def __init__(
        self,
        denoiser: object,
        codec: object | None,
        conditioner: object | None,
        *,
        model_timestep_scale: float = GAMMA_WORLD_DEFAULT_TRAIN_TIMESTEPS,
        num_train_timesteps: int = GAMMA_WORLD_DEFAULT_TRAIN_TIMESTEPS,
        expected_latent_channels: int = 16,
        temporal_compression: int = 4,
        spatial_compression: int = 8,
    ) -> None:
        trainable_module = _component_module(denoiser, "net")
        if trainable_module is None:
            raise TypeError("Gamma-World denoiser must expose an nn.Module as 'net'")
        if not callable(denoiser):
            raise TypeError("Gamma-World denoiser must be callable")
        if getattr(denoiser, "is_bidirectional", True) is False:
            raise ValueError("this adapter supports only the bidirectional Gamma-World denoiser")

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

        blocks = getattr(self.trainable_module, "blocks", ())
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
            raise TypeError("Gamma-World training requires precomputed clean_latents")
        if cached_latents.ndim != 5:
            raise ValueError(f"Gamma-World clean_latents must be [B,C,T,H,W]; got {tuple(cached_latents.shape)}")

        self._keep_frozen_modules_in_eval()
        device, dtype = _module_device_dtype(self.trainable_module)
        clean_latents = cached_latents.detach().to(device=device, dtype=dtype)
        if int(clean_latents.shape[1]) != self.expected_latent_channels:
            raise ValueError(
                f"Gamma-World clean latents contain {int(clean_latents.shape[1])} channels; "
                f"expected {self.expected_latent_channels}"
            )

        context = batch.conditions.get("crossattn_emb", batch.conditions.get("context"))
        if not isinstance(context, torch.Tensor):
            raise ValueError("Gamma-World training requires a precomputed 'crossattn_emb' text tensor")
        actions = batch.conditions.get("action_inputs")
        if actions is None:
            raise ValueError("Gamma-World bidirectional training requires 'action_inputs' (per-agent controls)")

        mask = batch.conditions.get("condition_video_input_mask_B_C_T_H_W")
        gt_frames = batch.conditions.get("gt_frames")
        if isinstance(mask, torch.Tensor) != isinstance(gt_frames, torch.Tensor):
            raise ValueError("Gamma-World gt_frames and condition_video_input_mask must be provided together")

        conditioning: dict[str, object] = {
            "crossattn_emb": context.to(device=device, dtype=dtype),
            "use_video_condition": True,
            "action_inputs": actions,
        }
        loss_mask: torch.Tensor | None = None
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().to(device=device, dtype=dtype)
            gt_frames = gt_frames.detach().to(device=device, dtype=dtype)
            conditioning["condition_video_input_mask_B_C_T_H_W"] = mask
            conditioning["gt_frames"] = gt_frames
            # The mask marks clean given frames; score only the generated frames.
            indicator = mask.amax(dim=1, keepdim=True)
            loss_mask = 1.0 - indicator.to(dtype=torch.float32)
        for optional in ("view_indices_B_T", "padding_mask", "fps"):
            if optional in batch.conditions:
                value = batch.conditions[optional]
                if isinstance(value, torch.Tensor):
                    value = value.to(device=device)
                conditioning[optional] = value

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
            raise TypeError("Gamma-World training requires one tensor model_input")
        latents = batch.model_input
        if latents.ndim != 5 or int(latents.shape[1]) != self.expected_latent_channels:
            channels = self.expected_latent_channels
            raise ValueError(f"Gamma-World model_input must be [B,{channels},T,H,W]; got {tuple(latents.shape)}")
        prediction_type = batch.metadata.get("prediction_type")
        if prediction_type is not None and prediction_type != self.prediction_type:
            raise ValueError(
                f"Gamma-World adapter predicts {self.prediction_type!r}, got objective {prediction_type!r}"
            )
        sigmas = _normalize_effective_sigmas(
            batch.sigmas,
            batch_size=int(latents.shape[0]),
            device=latents.device,
            family="Gamma-World",
        )

        # Never thread block_noise/initial_noise: without them the denoiser
        # returns the raw velocity for the condition frames, which the loss mask
        # then excludes; passing them would trigger the inference override path.
        conditioning = {
            key: value
            for key, value in dict(batch.conditioning).items()
            if key not in {"block_noise", "initial_noise"}
        }

        self.trainable_module.train(training)
        self._keep_frozen_modules_in_eval()
        model_timesteps = sigmas * self.model_timestep_scale
        output = self.denoiser(
            DenoiserInput(
                latents=latents,
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
            raise TypeError(f"Gamma-World denoiser returned {type(output).__name__} without a tensor sample")
        if sample.shape != latents.shape:
            raise ValueError(
                f"Gamma-World prediction shape {tuple(sample.shape)} does not match input {tuple(latents.shape)}"
            )
        return sample


def build_cached_gamma_world_bidirectional_train_adapter(
    components: Mapping[ComponentKey, object],
    **options: object,
) -> GammaWorldBidirectionalTrainAdapter:
    """Build a bidirectional Gamma-World adapter from immutable cached features."""

    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping")
    denoiser_key = ComponentKey(ComponentKind.DENOISER)
    unexpected = sorted(str(key) for key in components if key != denoiser_key)
    if unexpected:
        raise ValueError(f"cached Gamma-World training accepts only the denoiser component; got {unexpected}")
    try:
        denoiser = components[denoiser_key]
    except KeyError as error:
        raise KeyError(f"cached Gamma-World training components are missing {denoiser_key}") from error
    return GammaWorldBidirectionalTrainAdapter(denoiser=denoiser, codec=None, conditioner=None, **options)


__all__ = [
    "GAMMA_WORLD_DEFAULT_TRAIN_TIMESTEPS",
    "GammaWorldBidirectionalTrainAdapter",
    "build_cached_gamma_world_bidirectional_train_adapter",
]

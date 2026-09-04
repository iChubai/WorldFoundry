"""Training adapter for the native Echo-Memory video graph.

Echo-Memory is a Wan-shaped flow-velocity model (``[B,C,T,H,W]`` 16-channel
latents, ``sigma * num_train_timesteps`` timestep domain) with an action-driven
memory adapter.  Its context frames are stored as a *clean temporal suffix*
inside the denoised latent: at inference a frozen-latent mask restores that
suffix to the clean value on every step, and the memory adapter reads those
clean frames.  A correct training forward must mirror this:

* the last ``num_context_frames`` latent frames stay clean in the model input
  even after the objective corrupts the whole tensor, and
* the flow-matching loss is computed only on the leading target frames.

The adapter therefore re-freezes the context suffix in :meth:`forward_model`
and emits a per-frame ``loss_mask`` that zeros the context suffix.  Plain
full-frame corruption + unmasked MSE would feed noised memory to the adapter
and train against a target the model never predicts.
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

ECHO_MEMORY_DEFAULT_TRAIN_TIMESTEPS = 1000

# Conditioning keys the adapter threads through the objective for its own use;
# they are removed before the denoiser call so the model only sees its native
# conditioning contract.
_CLEAN_CONTEXT_KEY = "_echo_clean_context_suffix"
_CONTEXT_FRAMES_KEY = "_echo_num_context_frames"


class EchoMemoryTrainAdapter:
    """Own Echo-Memory conditioning, context-suffix freezing, and forward."""

    prediction_type = "flow_velocity"
    lora_target_preset = "wan-attention"
    conditioning_dropout_owner = "none"
    model_family = "echo-memory-video"

    def __init__(
        self,
        denoiser: object,
        codec: object | None,
        conditioner: object | None,
        *,
        model_timestep_scale: float = ECHO_MEMORY_DEFAULT_TRAIN_TIMESTEPS,
        num_train_timesteps: int = ECHO_MEMORY_DEFAULT_TRAIN_TIMESTEPS,
        expected_latent_channels: int = 16,
        temporal_compression: int = 4,
        spatial_compression: int = 8,
        action_dim: int = 12,
    ) -> None:
        trainable_module = _component_module(denoiser, "model")
        if trainable_module is None:
            raise TypeError("Echo-Memory denoiser must expose an nn.Module as 'model'")
        if not callable(denoiser):
            raise TypeError("Echo-Memory denoiser must be callable")

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
        self.action_dim = _positive_int(action_dim, field_name="action_dim")

        codec_module = _component_module(codec, "vae", "model")
        conditioner_module = _component_module(conditioner, "text_encoder", "encoder", "model")
        self.frozen_modules = tuple(
            dict.fromkeys(module for module in (codec_module, conditioner_module) if module is not None)
        )
        for module in self.frozen_modules:
            _freeze(module)
        # Echo models are consumed as frozen memory backbones in inference; the
        # trainable_module keeps its recipe-declared grad state and is only put
        # into train mode explicitly by the engine.
        self.trainable_module.train()

        blocks = getattr(self.trainable_module, "blocks", ())
        self.fsdp_block_classes = tuple(dict.fromkeys(type(block) for block in blocks if isinstance(block, nn.Module)))
        self.conditioning_dropout_probability = 0.0

    def _keep_frozen_modules_in_eval(self) -> None:
        for module in self.frozen_modules:
            module.eval()

    def _context_frames(self, batch: TrainingBatch, *, total_frames: int) -> int:
        raw = batch.conditions.get("num_context_frames", batch.metadata.get("num_context_frames"))
        if raw is None:
            raise ValueError("Echo-Memory training requires 'num_context_frames'")
        frames = _positive_int(raw, field_name="num_context_frames")
        if frames >= total_frames:
            raise ValueError(
                f"Echo-Memory num_context_frames ({frames}) must be smaller than latent frames ({total_frames})"
            )
        return frames

    def _actions(
        self,
        batch: TrainingBatch,
        *,
        total_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raw = batch.conditions.get("actions")
        if raw is None:
            raise ValueError("Echo-Memory training requires an 'actions' conditioning tensor")
        if not isinstance(raw, torch.Tensor):
            raise TypeError("Echo-Memory actions must be a torch.Tensor")
        actions = raw
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        if actions.ndim != 3:
            raise ValueError(f"Echo-Memory actions must be [B,T,{self.action_dim}]; got {tuple(raw.shape)}")
        if int(actions.shape[0]) != batch.batch_size:
            raise ValueError("Echo-Memory actions batch dimension does not match the batch")
        if int(actions.shape[1]) != total_frames:
            raise ValueError(
                f"Echo-Memory actions must cover all {total_frames} latent frames; got {int(actions.shape[1])}"
            )
        if int(actions.shape[2]) != self.action_dim:
            raise ValueError(f"Echo-Memory actions last dim must be {self.action_dim}; got {int(actions.shape[2])}")
        return actions.to(device=device, dtype=dtype)

    def _text_context(self, batch: TrainingBatch, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        context = batch.conditions.get("context")
        if not isinstance(context, torch.Tensor):
            raise ValueError("Echo-Memory training requires a precomputed 'context' text tensor")
        if int(context.shape[0]) != batch.batch_size:
            raise ValueError("Echo-Memory context batch dimension does not match the batch")
        if not bool(torch.isfinite(context).all()):
            raise ValueError("Echo-Memory context contains NaN or infinity")
        return context.to(device=device, dtype=dtype)

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        if not isinstance(batch, TrainingBatch):
            raise TypeError("batch must be a TrainingBatch")
        cached_latents = batch.conditions.get("clean_latents")
        if not isinstance(cached_latents, torch.Tensor):
            raise TypeError("Echo-Memory training requires precomputed clean_latents")
        if cached_latents.ndim != 5:
            raise ValueError(f"Echo-Memory clean_latents must be [B,C,T,H,W]; got {tuple(cached_latents.shape)}")

        self._keep_frozen_modules_in_eval()
        device, dtype = _module_device_dtype(self.trainable_module)
        clean_latents = cached_latents.detach().to(device=device, dtype=dtype)
        if int(clean_latents.shape[0]) != batch.batch_size:
            raise ValueError("Echo-Memory clean latents do not match the batch dimension")
        if int(clean_latents.shape[1]) != self.expected_latent_channels:
            raise ValueError(
                f"Echo-Memory clean latents contain {int(clean_latents.shape[1])} channels; "
                f"expected {self.expected_latent_channels}"
            )
        total_frames = int(clean_latents.shape[2])
        context_frames = self._context_frames(batch, total_frames=total_frames)

        text_context = self._text_context(batch, device=device, dtype=dtype)
        actions = self._actions(batch, total_frames=total_frames, device=device, dtype=dtype)

        # The context suffix (last ``context_frames`` latent frames) is the clean
        # memory the adapter must preserve; keep a detached copy for re-freezing
        # after corruption, and mask it out of the loss.
        clean_context_suffix = clean_latents[:, :, total_frames - context_frames :].clone()
        loss_mask = torch.ones(
            (batch.batch_size, 1, total_frames, 1, 1),
            device=device,
            dtype=torch.float32,
        )
        loss_mask[:, :, total_frames - context_frames :] = 0.0

        conditioning: dict[str, object] = {
            "context": text_context,
            "actions": actions,
            "num_context_frames": context_frames,
            _CLEAN_CONTEXT_KEY: clean_context_suffix,
            _CONTEXT_FRAMES_KEY: context_frames,
        }

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
                "num_context_frames": context_frames,
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
        """Forward a trainable policy or frozen role, re-freezing context frames."""

        if not isinstance(training, bool):
            raise TypeError("training must be a bool")
        if branch not in {"positive", "negative"}:
            raise ValueError("branch must be 'positive' or 'negative'")
        if not isinstance(batch, ObjectiveBatch):
            raise TypeError("batch must be an ObjectiveBatch")
        if isinstance(batch.model_input, Mapping) or not isinstance(batch.model_input, torch.Tensor):
            raise TypeError("Echo-Memory training requires one tensor model_input")
        latents = batch.model_input
        if latents.ndim != 5 or int(latents.shape[1]) != self.expected_latent_channels:
            channels = self.expected_latent_channels
            raise ValueError(
                f"Echo-Memory model_input must be [B,{channels},T,H,W]; got {tuple(latents.shape)}"
            )
        prediction_type = batch.metadata.get("prediction_type")
        if prediction_type is not None and prediction_type != self.prediction_type:
            raise ValueError(
                f"Echo-Memory adapter predicts {self.prediction_type!r}, got objective {prediction_type!r}"
            )

        conditioning = dict(batch.conditioning)
        clean_context = conditioning.pop(_CLEAN_CONTEXT_KEY, None)
        context_frames = conditioning.pop(_CONTEXT_FRAMES_KEY, None)
        if not isinstance(clean_context, torch.Tensor) or not isinstance(context_frames, int):
            raise ValueError("Echo-Memory forward requires the clean context suffix threaded from prepare_batch")
        total_frames = int(latents.shape[2])
        if context_frames <= 0 or context_frames >= total_frames:
            raise ValueError("Echo-Memory context frame count is inconsistent with the model input")
        if tuple(clean_context.shape) != (
            int(latents.shape[0]),
            int(latents.shape[1]),
            context_frames,
            int(latents.shape[3]),
            int(latents.shape[4]),
        ):
            raise ValueError("Echo-Memory clean context suffix shape does not match the model input")

        sigmas = _normalize_effective_sigmas(
            batch.sigmas,
            batch_size=int(latents.shape[0]),
            device=latents.device,
            family="Echo-Memory",
        )

        # Re-freeze the context suffix to its clean value so the memory adapter
        # reads unnoised frames, exactly as the inference frozen-latent mask does.
        model_latents = latents.clone()
        model_latents[:, :, total_frames - context_frames :] = clean_context.to(
            device=latents.device,
            dtype=latents.dtype,
        )

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
            raise TypeError(f"Echo-Memory denoiser returned {type(output).__name__} without a tensor sample")
        if sample.shape != latents.shape:
            raise ValueError(
                f"Echo-Memory prediction shape {tuple(sample.shape)} does not match input {tuple(latents.shape)}"
            )
        return sample


def build_cached_echo_memory_train_adapter(
    components: Mapping[ComponentKey, object],
    **options: object,
) -> EchoMemoryTrainAdapter:
    """Build an Echo-Memory adapter that consumes only immutable cached features."""

    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping")
    denoiser_key = ComponentKey(ComponentKind.DENOISER)
    unexpected = sorted(str(key) for key in components if key != denoiser_key)
    if unexpected:
        raise ValueError(f"cached Echo-Memory training accepts only the denoiser component; got {unexpected}")
    try:
        denoiser = components[denoiser_key]
    except KeyError as error:
        raise KeyError(f"cached Echo-Memory training components are missing {denoiser_key}") from error
    return EchoMemoryTrainAdapter(denoiser=denoiser, codec=None, conditioner=None, **options)


__all__ = [
    "ECHO_MEMORY_DEFAULT_TRAIN_TIMESTEPS",
    "EchoMemoryTrainAdapter",
    "build_cached_echo_memory_train_adapter",
]

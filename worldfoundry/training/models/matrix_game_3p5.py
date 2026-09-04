"""Training adapter for the native Matrix-Game-3.5 interactive video graph.

Matrix-Game-3.5 conditions on a clean first-frame anchor plus camera (PRoPE),
optional mosaic memory, and subject references.  Unlike Cosmos/Echo, none of
those conditions live inside the denoised latent: the denoiser concatenates the
clean anchor/mosaic internally and slices the output back to only the noisy
generating window, returning pure model velocity for those frames.  So a
correct training forward feeds ``model_input`` = the generating window only,
routes the anchor and camera/mosaic tensors through as conditioning, and needs
no first-frame freeze and no loss mask — plain full-frame corruption + unmasked
MSE on the window is exactly right.

The denoiser reshapes the timestep to one shared scalar and the subject-ref
path rejects batches larger than one, so this path is single-media (batch size
one).  Timesteps ride the ``sigma * num_train_timesteps`` (0..1000) domain;
condition/hole frames are re-timed internally by the denoiser.
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

MATRIX_GAME_35_DEFAULT_TRAIN_TIMESTEPS = 1000

# Conditioning tensors the denoiser forwards to ``matrix_game_35_forward``.  Any
# subset the data pipeline provides is passed through unchanged; only ``context``
# and ``first_frame_latents`` are required for a video2world training step.
_PASSTHROUGH_CONDITION_KEYS = (
    "first_frame_latents",
    "mosaic_latent",
    "mosaic_frame_indices",
    "mosaic_view_change",
    "subject_ref_latents",
    "latent_rope_time_indices",
    "clean_latent_indices_prope_intrinsic",
    "clean_latent_indices_prope_extrinsic",
    "noisy_latent_indices_prope_intrinsic",
    "noisy_latent_indices_prope_extrinsic",
    "view_indices_B_T",
    "action_inputs",
)


class MatrixGame35TrainAdapter:
    """Own Matrix-Game-3.5 conditioning marshaling and training forward."""

    prediction_type = "flow_velocity"
    lora_target_preset = "matrix-game-attention"
    conditioning_dropout_owner = "none"
    model_family = "matrix-game-3.5-video"

    def __init__(
        self,
        denoiser: object,
        codec: object | None,
        conditioner: object | None,
        *,
        model_timestep_scale: float = MATRIX_GAME_35_DEFAULT_TRAIN_TIMESTEPS,
        num_train_timesteps: int = MATRIX_GAME_35_DEFAULT_TRAIN_TIMESTEPS,
        expected_latent_channels: int = 48,
        temporal_compression: int = 4,
        spatial_compression: int = 16,
    ) -> None:
        trainable_module = _component_module(denoiser, "model")
        if trainable_module is None:
            raise TypeError("Matrix-Game-3.5 denoiser must expose an nn.Module as 'model'")
        if not callable(denoiser):
            raise TypeError("Matrix-Game-3.5 denoiser must be callable")

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
        if batch.batch_size != 1:
            raise ValueError(
                "Matrix-Game-3.5 training uses one shared timestep and per-media conditioning; "
                "materialize one conditioning media per microbatch (batch size one)"
            )
        cached_latents = batch.conditions.get("clean_latents")
        if not isinstance(cached_latents, torch.Tensor):
            raise TypeError("Matrix-Game-3.5 training requires precomputed generating-window clean_latents")
        if cached_latents.ndim != 5:
            raise ValueError(f"Matrix-Game-3.5 clean_latents must be [B,C,T,H,W]; got {tuple(cached_latents.shape)}")

        self._keep_frozen_modules_in_eval()
        device, dtype = _module_device_dtype(self.trainable_module)
        clean_latents = cached_latents.detach().to(device=device, dtype=dtype)
        if int(clean_latents.shape[1]) != self.expected_latent_channels:
            raise ValueError(
                f"Matrix-Game-3.5 clean latents contain {int(clean_latents.shape[1])} channels; "
                f"expected {self.expected_latent_channels}"
            )

        context = batch.conditions.get("context")
        if not isinstance(context, torch.Tensor):
            raise ValueError("Matrix-Game-3.5 training requires a precomputed 'context' text tensor")
        first_frame = batch.conditions.get("first_frame_latents")
        if not isinstance(first_frame, torch.Tensor):
            raise ValueError("Matrix-Game-3.5 video2world training requires 'first_frame_latents' (clean anchor)")

        def to_device(value: object) -> object:
            if isinstance(value, torch.Tensor):
                target = dtype if value.is_floating_point() else value.dtype
                return value.detach().to(device=device, dtype=target)
            return value

        conditioning: dict[str, object] = {"context": context.to(device=device, dtype=dtype)}
        for key in _PASSTHROUGH_CONDITION_KEYS:
            if key in batch.conditions:
                conditioning[key] = to_device(batch.conditions[key])
        if "fuse_vae_embedding_in_latents" in batch.conditions:
            conditioning["fuse_vae_embedding_in_latents"] = bool(batch.conditions["fuse_vae_embedding_in_latents"])
        for scalar in ("height", "width", "fps"):
            if scalar in batch.conditions:
                conditioning[scalar] = batch.conditions[scalar]

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
            raise TypeError("Matrix-Game-3.5 training requires one tensor model_input")
        latents = batch.model_input
        if latents.ndim != 5 or int(latents.shape[1]) != self.expected_latent_channels:
            channels = self.expected_latent_channels
            raise ValueError(f"Matrix-Game-3.5 model_input must be [B,{channels},T,H,W]; got {tuple(latents.shape)}")
        if int(latents.shape[0]) != 1:
            raise ValueError("Matrix-Game-3.5 training forward requires batch size one")
        prediction_type = batch.metadata.get("prediction_type")
        if prediction_type is not None and prediction_type != self.prediction_type:
            raise ValueError(
                f"Matrix-Game-3.5 adapter predicts {self.prediction_type!r}, got objective {prediction_type!r}"
            )
        sigmas = _normalize_effective_sigmas(
            batch.sigmas,
            batch_size=int(latents.shape[0]),
            device=latents.device,
            family="Matrix-Game-3.5",
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
            raise TypeError(f"Matrix-Game-3.5 denoiser returned {type(output).__name__} without a tensor sample")
        if sample.shape != latents.shape:
            raise ValueError(
                f"Matrix-Game-3.5 prediction shape {tuple(sample.shape)} does not match input {tuple(latents.shape)}"
            )
        return sample


def build_cached_matrix_game_35_train_adapter(
    components: Mapping[ComponentKey, object],
    **options: object,
) -> MatrixGame35TrainAdapter:
    """Build a Matrix-Game-3.5 adapter that consumes only immutable cached features."""

    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping")
    denoiser_key = ComponentKey(ComponentKind.DENOISER)
    unexpected = sorted(str(key) for key in components if key != denoiser_key)
    if unexpected:
        raise ValueError(f"cached Matrix-Game-3.5 training accepts only the denoiser component; got {unexpected}")
    try:
        denoiser = components[denoiser_key]
    except KeyError as error:
        raise KeyError(f"cached Matrix-Game-3.5 training components are missing {denoiser_key}") from error
    return MatrixGame35TrainAdapter(denoiser=denoiser, codec=None, conditioner=None, **options)


__all__ = [
    "MATRIX_GAME_35_DEFAULT_TRAIN_TIMESTEPS",
    "MatrixGame35TrainAdapter",
    "build_cached_matrix_game_35_train_adapter",
]

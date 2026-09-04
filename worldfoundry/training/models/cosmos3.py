"""Training adapter for the native Cosmos3 omni (multi-modal) video graph.

Cosmos3 is an omni sequence-packing model: its denoiser reads ``input_ids`` (the
tokenized prompt) plus per-modality ``ModalityState`` values and derives the full
joint layout (text/vision indexes, positions, MSE-loss indexes) deterministically
via ``build_cosmos3_sequence_layout``.  The transformer predicts a **flow
velocity** on the noisy vision frames (scheduler ``prediction_type=flow_prediction``),
so training reuses the audited ``FlowMatchingObjective`` unchanged.

This adapter integrates the video-only text-to-video path — the tractable, fully
verified slice of the omni model.  ``prepare_batch`` builds a video
``ModalityState`` from a clean ``[B,C,T,H,W]`` latent with an all-ones denoise
mask (every frame generated, matching the T2V initializer) and ``arange`` frame
positions, and threads ``input_ids`` from the cached conditioner.  ``forward_model``
reassembles the ``MultiModalDenoiserInput`` from the corrupted latent and returns
the per-modality velocity as a ``{"video": ...}`` tree.  The denoiser reshapes the
timestep to one shared scalar, so this path is single media (batch size one).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.components import ComponentKey, ComponentKind
from worldfoundry.base_models.diffusion_model.contracts import (
    ModalityState,
    MultiModalDenoiserInput,
)
from worldfoundry.training.api.contracts import ObjectiveBatch, PreparedBatch, TrainingBatch
from worldfoundry.training.models._common import (
    component_module as _component_module,
    freeze as _freeze,
    module_device_dtype as _module_device_dtype,
    normalize_effective_sigmas as _normalize_effective_sigmas,
    positive_int as _positive_int,
)

COSMOS3_DEFAULT_FPS = 24.0
_INPUT_IDS_KEY = "_cosmos3_input_ids"
_DENOISE_MASK_KEY = "_cosmos3_video_denoise_mask"
_POSITIONS_KEY = "_cosmos3_video_positions"


class Cosmos3VideoTrainAdapter:
    """Own Cosmos3 omni video-only conditioning and multi-modal training forward."""

    prediction_type = "flow_velocity"
    lora_target_preset = "cosmos3-attention"
    conditioning_dropout_owner = "none"
    model_family = "cosmos3-omni-video"

    def __init__(
        self,
        denoiser: object,
        codec: object | None,
        conditioner: object | None,
        *,
        model_timestep_scale: float = 1.0,
        expected_latent_channels: int = 48,
        temporal_compression: int = 4,
        spatial_compression: int = 16,
        default_fps: float = COSMOS3_DEFAULT_FPS,
    ) -> None:
        trainable_module = _component_module(denoiser, "model")
        if trainable_module is None:
            raise TypeError("Cosmos3 denoiser must expose an nn.Module as 'model'")
        if not callable(denoiser):
            raise TypeError("Cosmos3 denoiser must be callable")

        from math import isfinite

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

        blocks = getattr(self.trainable_module, "layers", getattr(self.trainable_module, "blocks", ()))
        self.fsdp_block_classes = tuple(dict.fromkeys(type(block) for block in blocks if isinstance(block, nn.Module)))
        self.conditioning_dropout_probability = 0.0

    def _keep_frozen_modules_in_eval(self) -> None:
        for module in self.frozen_modules:
            module.eval()

    def prepare_batch(self, batch: TrainingBatch) -> PreparedBatch:
        if not isinstance(batch, TrainingBatch):
            raise TypeError("batch must be a TrainingBatch")
        if batch.batch_size != 1:
            raise ValueError("Cosmos3 packs one joint sequence per step; use batch size one")
        cached_latents = batch.conditions.get("clean_latents")
        if not isinstance(cached_latents, torch.Tensor):
            raise TypeError("Cosmos3 training requires precomputed clean_latents")
        if cached_latents.ndim != 5:
            raise ValueError(f"Cosmos3 clean_latents must be [B,C,T,H,W]; got {tuple(cached_latents.shape)}")

        self._keep_frozen_modules_in_eval()
        device, dtype = _module_device_dtype(self.trainable_module)
        clean_latents = cached_latents.detach().to(device=device, dtype=dtype)
        if int(clean_latents.shape[1]) != self.expected_latent_channels:
            raise ValueError(
                f"Cosmos3 clean latents contain {int(clean_latents.shape[1])} channels; "
                f"expected {self.expected_latent_channels}"
            )
        latent_frames = int(clean_latents.shape[2])

        input_ids = batch.conditions.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            raise ValueError("Cosmos3 training requires precomputed 'input_ids' from the prompt conditioner")

        # Text-to-video training: every frame is generated (denoise mask all ones);
        # positions index the latent frames (matching the T2V initializer).
        denoise_mask = torch.ones_like(clean_latents)
        positions = torch.arange(latent_frames, device=device)

        conditioning: dict[str, object] = {
            _INPUT_IDS_KEY: input_ids.to(device=device),
            _DENOISE_MASK_KEY: denoise_mask,
            _POSITIONS_KEY: positions,
            "fps": float(batch.conditions.get("fps", batch.metadata.get("fps", self.default_fps))),
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
                "latent_shape": tuple(int(value) for value in clean_latents.shape),
                "precomputed_latents": True,
                "conditioning_dropout_owner": self.conditioning_dropout_owner,
                "conditioning_dropout_probability": self.conditioning_dropout_probability,
            }
        )
        return PreparedBatch(
            sample_ids=batch.sample_ids,
            clean_latents={"video": clean_latents},
            conditioning=conditioning,
            loss_mask=None,
            sample_weights=sample_weights,
            metadata=metadata,
        )

    def forward_train(self, batch: ObjectiveBatch):
        return self.forward_model(batch, training=True)

    def forward_model(
        self,
        batch: ObjectiveBatch,
        *,
        training: bool,
        branch: str = "positive",
    ):
        if not isinstance(training, bool):
            raise TypeError("training must be a bool")
        if branch not in {"positive", "negative"}:
            raise ValueError("branch must be 'positive' or 'negative'")
        if not isinstance(batch, ObjectiveBatch):
            raise TypeError("batch must be an ObjectiveBatch")
        model_input = batch.model_input
        if not isinstance(model_input, Mapping) or set(model_input) != {"video"}:
            raise TypeError("Cosmos3 training requires a {'video': tensor} model_input tree")
        noisy = model_input["video"]
        if not torch.is_tensor(noisy) or noisy.ndim != 5:
            raise ValueError(f"Cosmos3 video latent must be [B,C,T,H,W]; got {getattr(noisy, 'shape', None)}")
        if int(noisy.shape[0]) != 1:
            raise ValueError("Cosmos3 training forward requires batch size one")
        prediction_type = batch.metadata.get("prediction_type")
        if prediction_type is not None and prediction_type != self.prediction_type:
            raise ValueError(f"Cosmos3 adapter predicts {self.prediction_type!r}, got objective {prediction_type!r}")

        conditioning = dict(batch.conditioning)
        input_ids = conditioning.pop(_INPUT_IDS_KEY, None)
        denoise_mask = conditioning.pop(_DENOISE_MASK_KEY, None)
        positions = conditioning.pop(_POSITIONS_KEY, None)
        if not isinstance(input_ids, torch.Tensor) or not isinstance(denoise_mask, torch.Tensor):
            raise ValueError("Cosmos3 forward requires input_ids/denoise_mask threaded from prepare_batch")
        if not isinstance(positions, torch.Tensor):
            raise ValueError("Cosmos3 forward requires frame positions threaded from prepare_batch")

        sigmas = _normalize_effective_sigmas(
            batch.sigmas,
            batch_size=int(noisy.shape[0]),
            device=noisy.device,
            family="Cosmos3",
        )
        conditioning["input_ids"] = input_ids

        state = ModalityState(
            latent=noisy,
            denoise_mask=denoise_mask.to(device=noisy.device, dtype=noisy.dtype),
            positions=positions.to(device=noisy.device),
            clean_latent=noisy,
        )
        self.trainable_module.train(training)
        self._keep_frozen_modules_in_eval()
        model_timesteps = sigmas * self.model_timestep_scale
        output = self.denoiser(
            MultiModalDenoiserInput(
                modalities={"video": state},
                timestep=model_timesteps,
                conditioning=conditioning,
                step_index=0,
                total_steps=1000,
                branch=branch,
            )
        )
        samples = getattr(output, "samples", None)
        if not isinstance(samples, Mapping) or "video" not in samples:
            raise TypeError("Cosmos3 denoiser must return a MultiModalDenoiserOutput with a 'video' sample")
        velocity = samples["video"]
        if not torch.is_tensor(velocity) or velocity.shape != noisy.shape:
            raise ValueError("Cosmos3 video sample must match the noised latent shape")
        return {"video": velocity}


def build_cached_cosmos3_video_train_adapter(
    components: Mapping[ComponentKey, object],
    **options: object,
) -> Cosmos3VideoTrainAdapter:
    """Build a Cosmos3 video adapter that consumes only immutable cached features."""

    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping")
    denoiser_key = ComponentKey(ComponentKind.DENOISER)
    unexpected = sorted(str(key) for key in components if key != denoiser_key)
    if unexpected:
        raise ValueError(f"cached Cosmos3 training accepts only the denoiser component; got {unexpected}")
    try:
        denoiser = components[denoiser_key]
    except KeyError as error:
        raise KeyError(f"cached Cosmos3 training components are missing {denoiser_key}") from error
    return Cosmos3VideoTrainAdapter(denoiser=denoiser, codec=None, conditioner=None, **options)


__all__ = [
    "COSMOS3_DEFAULT_FPS",
    "Cosmos3VideoTrainAdapter",
    "build_cached_cosmos3_video_train_adapter",
]

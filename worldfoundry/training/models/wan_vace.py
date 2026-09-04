"""Training adapter for the native Wan2.1 VACE control-video graph.

VACE reuses the Wan video geometry, codec, and text conditioner.  Its denoiser
additionally consumes an encoded control signal (``vace_context``) and a scalar
mixing weight (``vace_context_scale``).  Those keys already flow through Wan's
conditioning merge and forward; this adapter only validates their presence and
shape so a control run cannot silently degrade to plain text conditioning.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from worldfoundry.base_models.diffusion_model.components import ComponentKey, ComponentKind
from worldfoundry.training.api.contracts import TrainingBatch
from worldfoundry.training.models.wan import WanTrainAdapter


class WanVaceTrainAdapter(WanTrainAdapter):
    """Own Wan VACE conditioning on top of the shared Wan video forward."""

    lora_target_preset = "wan-attention"
    model_family = "wan-vace-video"

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
        conditioning = super()._encoded_conditioning(
            batch,
            frames=frames,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )
        vace_context = conditioning.get("vace_context")
        if not isinstance(vace_context, torch.Tensor):
            raise ValueError("Wan VACE training requires a tensor 'vace_context' control signal")
        if int(vace_context.shape[0]) != batch.batch_size:
            raise ValueError(
                f"Wan VACE vace_context must start with batch size {batch.batch_size}; "
                f"got {tuple(vace_context.shape)}"
            )
        if not bool(torch.isfinite(vace_context).all()):
            raise ValueError("Wan VACE vace_context contains NaN or infinity")
        scale = conditioning.get("vace_context_scale")
        if scale is not None and not isinstance(scale, (int, float, torch.Tensor)):
            raise TypeError("Wan VACE vace_context_scale must be a number or tensor")
        return conditioning


def build_wan_vace_train_adapter(
    components: Mapping[ComponentKey, object],
    **options: object,
) -> WanVaceTrainAdapter:
    """Build a raw-video Wan VACE adapter from native training components."""

    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping")

    def require(kind: ComponentKind, name: str = "main") -> object:
        key = ComponentKey(kind, name)
        try:
            return components[key]
        except KeyError as error:
            raise KeyError(f"Wan VACE training components are missing {key}") from error

    return WanVaceTrainAdapter(
        denoiser=require(ComponentKind.DENOISER),
        codec=require(ComponentKind.LATENT_ENCODER, "codec"),
        conditioner=require(ComponentKind.CONDITIONER),
        **options,
    )


def build_cached_wan_vace_train_adapter(
    components: Mapping[ComponentKey, object],
    **options: object,
) -> WanVaceTrainAdapter:
    """Build a Wan VACE adapter that consumes only immutable cached features."""

    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping")
    denoiser_key = ComponentKey(ComponentKind.DENOISER)
    unexpected = sorted(str(key) for key in components if key != denoiser_key)
    if unexpected:
        raise ValueError(f"cached Wan VACE training accepts only the denoiser component; got {unexpected}")
    try:
        denoiser = components[denoiser_key]
    except KeyError as error:
        raise KeyError(f"cached Wan VACE training components are missing {denoiser_key}") from error
    return WanVaceTrainAdapter(denoiser=denoiser, codec=None, conditioner=None, **options)


__all__ = [
    "WanVaceTrainAdapter",
    "build_cached_wan_vace_train_adapter",
    "build_wan_vace_train_adapter",
]

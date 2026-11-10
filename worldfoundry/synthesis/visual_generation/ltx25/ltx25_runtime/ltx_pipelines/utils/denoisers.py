"""Flat denoiser classes — transformer received at call time, not stored.
Three implementations of the :class:`~ltx_pipelines.utils.types.Denoiser` protocol:
* :class:`SimpleDenoiser` — single transformer call, no guidance.
* :class:`GuidedDenoiser` — static guiders, handles CFG + STG + isolated modality.
* :class:`FactoryGuidedDenoiser` — resolves guiders per-step from sigma.
``GuidedDenoiser`` and ``FactoryGuidedDenoiser`` share the core multi-pass
logic via the module-level :func:`_guided_denoise` function, which batches
all guidance passes into a single transformer call.
"""

import torch

from worldfoundry.base_models.diffusion_model.models.networks.ltx import X0Model

from ...ltx_core.types import LatentState
from .helpers import modality_from_latent_state
from .types import DenoisedLatentResult

"""Guider that only runs the conditioned pass and returns cond unchanged."""


class SimpleDenoiser:
    """Single transformer call, no guidance.
    Passes ``None`` Modality for absent modalities.
    """

    def __init__(
        self,
        v_context: torch.Tensor | None,
        a_context: torch.Tensor | None,
    ) -> None:
        self.v_context = v_context
        self.a_context = a_context

    def __call__(
        self,
        transformer: X0Model,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[DenoisedLatentResult | None, DenoisedLatentResult | None]:
        # Modality.sigma is (B,); expand the scalar schedule value per modality so a rank-stable
        # sigma reaches the (compiled) block instead of a 0-d scalar that recompiles on sigma.ndim.
        sigma = sigmas[step_index]
        pos_video = (
            modality_from_latent_state(video_state, self.v_context, sigma.expand(video_state.latent.shape[0]))
            if video_state is not None
            else None
        )
        pos_audio = (
            modality_from_latent_state(audio_state, self.a_context, sigma.expand(audio_state.latent.shape[0]))
            if audio_state is not None
            else None
        )
        denoised_video, denoised_audio = transformer(video=pos_video, audio=pos_audio, perturbations=None)
        return (
            DenoisedLatentResult.result_or_none(denoised=denoised_video),
            DenoisedLatentResult.result_or_none(denoised=denoised_audio),
        )

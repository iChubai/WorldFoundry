from collections.abc import Callable
from dataclasses import dataclass, replace

import torch
from tqdm import tqdm

from worldfoundry.base_models.diffusion_model.models.networks.ltx import X0Model
from worldfoundry.base_models.diffusion_model.models.representations.ltx.protocols import DiffusionStepProtocol

from ...ltx_core.components.diffusion_steps import EulerAncestralDiffusionStep
from .helpers import post_process_latent
from .types import DenoisedLatentResult, Denoiser, LatentState


def _step_state(
    state: LatentState | None,
    denoised: torch.Tensor | None,
    stepper: DiffusionStepProtocol,
    sigmas: torch.Tensor,
    step_idx: int,
) -> LatentState | None:
    """Advance one diffusion step for a single modality, or return ``None`` if absent."""
    if state is None or denoised is None:
        return state
    denoised = post_process_latent(denoised, state.denoise_mask, state.clean_latent)
    return replace(state, latent=stepper.step(state.latent, denoised, sigmas, step_idx))


def euler_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    stepper: DiffusionStepProtocol,
    transformer: X0Model,
    denoiser: Denoiser,
) -> tuple[LatentState | None, LatentState | None]:
    """
    Perform the joint audio-video denoising loop over a diffusion schedule.
    Either ``video_state`` or ``audio_state`` may be ``None`` for absent
    modalities; the absent modality is passed through unchanged.
    ### Parameters
    sigmas:
        A 1D tensor of noise levels (diffusion sigmas) defining the sampling
        schedule. All steps except the last element are iterated over.
    video_state:
        The current video :class:`LatentState`, or ``None`` if video is absent.
    audio_state:
        The current audio :class:`LatentState`, or ``None`` if audio is absent.
    stepper:
        An implementation of :class:`DiffusionStepProtocol` that updates a
        latent given the current latent, its denoised estimate, the full
        ``sigmas`` schedule, and the current step index.
    transformer:
        The diffusion model passed to the denoiser at each step.
    denoiser:
        A callable implementing :class:`Denoiser`. It is invoked as
        ``denoiser(transformer, video_state, audio_state, sigmas, step_index)``
        and must return a :class:`~ltx_pipelines.utils.types.DenoisedLatentResult`.
    ### Returns
    tuple[LatentState | None, LatentState | None]
        Final ``(video_state, audio_state)`` after the denoising loop.
    """
    for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
        video_result, audio_result = denoiser(transformer, video_state, audio_state, sigmas, step_idx)
        denoised_video = video_result.denoised if video_result is not None else None
        denoised_audio = audio_result.denoised if audio_result is not None else None

        video_state = _step_state(video_state, denoised_video, stepper, sigmas, step_idx)
        audio_state = _step_state(audio_state, denoised_audio, stepper, sigmas, step_idx)

    return (video_state, audio_state)


def _get_plain_noise(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Draw standard Gaussian noise matching the shape, dtype, and device of ``x``."""
    return torch.randn(x.shape, generator=generator, dtype=x.dtype, device=x.device)


@dataclass(frozen=True)
class _ModalityStep:
    """One modality's resolved inputs for a single ancestral-Euler step.
    Attributes:
        modality: ``"video"`` or ``"audio"``.
        state: The modality's current latent state.
        step_fn: The step callable for this modality, already bound to any
            stepper-specific inputs.
        denoised: Mask-corrected x_0 prediction in float32, or ``None`` when the
            denoiser produced no prediction for this modality (in which case the
            modality is passed through unchanged this step).
    """

    modality: str
    state: LatentState
    step_fn: Callable[..., torch.Tensor]
    denoised: torch.Tensor | None

    @classmethod
    def from_modality_result(
        cls,
        modality: str,
        state: LatentState,
        result: DenoisedLatentResult | None,
        make_step_fn: Callable[[DenoisedLatentResult | None, str], Callable[..., torch.Tensor]],
    ) -> "_ModalityStep":
        """Resolve one modality's step inputs from a denoiser result.
        ``make_step_fn`` is invoked for every present modality, including one
        whose result carries no denoised tensor, so that any validation it
        performs fires in a stable order regardless of the denoiser's output.
        """
        step_fn = make_step_fn(result, modality)
        denoised = None
        if result is not None and result.denoised is not None:
            denoised = post_process_latent(result.denoised.float(), state.denoise_mask, state.clean_latent)
        return cls(modality=modality, state=state, step_fn=step_fn, denoised=denoised)


def _ancestral_euler_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    transformer: X0Model,
    denoiser: Denoiser,
    noise_seed: int,
    new_noise_fn: Callable[[torch.Tensor, torch.Generator], torch.Tensor],
    model_dtype: torch.dtype,
    draw_noise: bool,
    make_step_fn: Callable[[DenoisedLatentResult | None, str], Callable[..., torch.Tensor]],
) -> tuple[LatentState | None, LatentState | None]:
    """
    Shared driver for the ancestral (noise-injecting) Euler denoising loops.
    Steps each present modality in float32, re-applies the conditioning mask
    after noise injection, and short-circuits to the denoised prediction on a
    terminal zero sigma. When both modalities are present, noise is drawn from
    the same seeded generator (video first, audio second) so the random
    sequence is consistent.
    ### Parameters
    draw_noise:
        Whether to draw a noise tensor per modality per step. When ``False``,
        ``noise=None`` is passed to the stepper.
    make_step_fn:
        ``(result, modality) -> step_fn`` factory returning the step callable
        for that modality, with any stepper-specific inputs already bound.
        ``step_fn`` is invoked as
        ``step_fn(sample=, denoised_sample=, sigmas=, step_index=, noise=)``.
        Called once per present modality per step right after the denoiser
        returns (before the terminal-sigma check), so it can also validate the
        denoiser output. ``modality`` is ``"video"`` or ``"audio"``.
    ### Returns
    tuple[LatentState | None, LatentState | None]
        Final ``(video_state, audio_state)`` after the denoising loop.
    """
    if video_state is None and audio_state is None:
        raise ValueError("At least one of video_state or audio_state must be provided")

    present_state = video_state if video_state is not None else audio_state
    generator = torch.Generator(device=present_state.latent.device).manual_seed(noise_seed)
    states: dict[str, LatentState | None] = {"video": video_state, "audio": audio_state}

    for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
        video_result, audio_result = denoiser(transformer, states["video"], states["audio"], sigmas, step_idx)
        pending = [
            _ModalityStep.from_modality_result(modality, state, result, make_step_fn)
            for modality, state, result in (
                ("video", states["video"], video_result),
                ("audio", states["audio"], audio_result),
            )
            if state is not None
        ]

        terminal_step = sigmas[step_idx + 1] == 0
        for step in pending:
            if step.denoised is None:
                continue
            if terminal_step:
                states[step.modality] = replace(step.state, latent=step.denoised.to(model_dtype))
                continue
            noise = new_noise_fn(step.state.latent, generator) if draw_noise else None
            x_next = step.step_fn(
                sample=step.state.latent.float(),
                denoised_sample=step.denoised,
                sigmas=sigmas,
                step_index=step_idx,
                noise=noise,
            )
            if draw_noise:
                x_next = post_process_latent(x_next, step.state.denoise_mask, step.state.clean_latent)
            states[step.modality] = replace(step.state, latent=x_next.to(model_dtype))

        if terminal_step:
            break

    return states["video"], states["audio"]


def euler_ancestral_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState | None,
    audio_state: LatentState | None,
    stepper: EulerAncestralDiffusionStep,
    transformer: X0Model,
    denoiser: Denoiser,
    noise_seed: int = -1,
    new_noise_fn: Callable[[torch.Tensor, torch.Generator], torch.Tensor] = _get_plain_noise,
    model_dtype: torch.dtype = torch.bfloat16,
) -> tuple[LatentState | None, LatentState | None]:
    """
    Joint audio-video denoising loop using the ancestral (SDE) Euler sampler.
    Each step takes a deterministic Euler step to an intermediate ``sigma_down``
    and renoises back up to the next sigma; see
    :class:`~ltx_core.components.diffusion_steps.EulerAncestralDiffusionStep`.
    Works with any denoiser (no ``uncond`` prediction required). With
    ``stepper=EulerAncestralDiffusionStep(eta=0.0)`` this reduces to
    :func:`euler_denoising_loop`.
    Either ``video_state`` or ``audio_state`` may be ``None`` for absent
    modalities. When both are present, noise is drawn from the same seeded
    generator (video first, audio second).
    ### Parameters
    sigmas:
        1-D tensor of noise levels defining the sampling schedule.
    video_state:
        Current video :class:`~ltx_core.types.LatentState`, or ``None``.
    audio_state:
        Current audio :class:`~ltx_core.types.LatentState`, or ``None``.
    stepper:
        :class:`~ltx_core.components.diffusion_steps.EulerAncestralDiffusionStep`
        instance carrying ``eta`` and ``s_noise`` parameters.
    transformer:
        The diffusion model passed to the denoiser at each step.
    denoiser:
        Callable implementing :class:`~ltx_pipelines.utils.types.Denoiser`.
    noise_seed:
        Integer seed for the noise generator. Default ``-1``.
    new_noise_fn:
        ``(latent, generator) -> noise`` callable. Defaults to plain
        ``torch.randn`` (no channel-wise normalization). Pass
        :func:`_get_new_noise` for the normalized variant used in res2s.
    model_dtype:
        Dtype for latent state updates. Default ``bfloat16``. Pass
        ``torch.float32`` to keep the sampling trajectory in full precision.
    ### Returns
    tuple[LatentState | None, LatentState | None]
        Final ``(video_state, audio_state)`` after the denoising loop.
    """
    if not isinstance(stepper, EulerAncestralDiffusionStep):
        raise ValueError(f"stepper must be an instance of EulerAncestralDiffusionStep, got {type(stepper).__name__}")

    return _ancestral_euler_denoising_loop(
        sigmas=sigmas,
        video_state=video_state,
        audio_state=audio_state,
        transformer=transformer,
        denoiser=denoiser,
        noise_seed=noise_seed,
        new_noise_fn=new_noise_fn,
        model_dtype=model_dtype,
        # Gated on eta alone: at s_noise=0 the step still applies its variance-preserving
        # rescale, so it must not fall back to the noise-free branch.
        draw_noise=stepper.eta > 0,
        # The ancestral step needs nothing beyond the standard arguments.
        make_step_fn=lambda _result, _modality: stepper.step,
    )

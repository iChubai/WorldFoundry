import logging
from collections.abc import Sequence

import torch

from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.diffusion_tiling import (
    recommended_decode_tiling_config,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.diffusion_video_decoder import (
    diffvae_tiling_geometry_from_vae_config,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.transformer.config import DiffVAEMode
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.transformer.fallback_na import (
    joint_sdpa_materializes_scores,
)
from worldfoundry.base_models.diffusion_model.models.autoencoders.ltx.video.video_vae import VideoEncoder
from worldfoundry.base_models.diffusion_model.models.encoders.ltx.gemma import LTXGemmaTextEncoder
from worldfoundry.base_models.diffusion_model.models.networks.ltx import Modality
from worldfoundry.base_models.diffusion_model.models.representations.ltx.tiling import AutoTiling, TilingConfig
from worldfoundry.base_models.diffusion_model.models.representations.ltx.types import (
    VIDEO_SCALE_FACTORS,
    SpatioTemporalScaleFactors,
    VideoPixelShape,
)

from ...ltx_core.components.noisers import Noiser
from ...ltx_core.conditioning import (
    ConditioningItem,
    VideoConditionByKeyframeIndex,
    VideoConditionByLatentIndex,
    VideoGeneratedKeyframeSlots,
)
from ...ltx_core.devices import activation_budget_bytes, cleanup_accelerator_memory, get_preferred_device
from ...ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ...ltx_core.model.video_vae import estimate_diffusion_decoder_weight_bytes, require_diffusion_video_vae
from ...ltx_core.tools import LatentTools
from ...ltx_core.types import LatentState
from .args import ImageConditioningInput
from .media_io import decode_image, load_image_and_preprocess, resize_aspect_ratio_preserving
from .media_io.color_config import HDRColorSpace


def get_device() -> torch.device:
    return get_preferred_device()


def cleanup_memory() -> None:
    cleanup_accelerator_memory()


def tiling_scale_factors_for_vae(vae_checkpoint_path: str) -> SpatioTemporalScaleFactors:
    """Scale factors decode will pass to ``to_splitters`` for this checkpoint."""
    require_diffusion_video_vae(vae_checkpoint_path)
    metadata = SafetensorsModelStateDictLoader().metadata(vae_checkpoint_path)
    vae_config = metadata.get("config", {}).get("vae", {})
    return diffvae_tiling_geometry_from_vae_config(vae_config)["pixel_scale"]


def tiling_config_for_vae(
    vae_checkpoint_path: str,
    *,
    height: int,
    width: int,
    num_frames: int,
    diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
    device: torch.device | None = None,
    free_bytes: int | None = None,
    keyframes: bool = False,
) -> TilingConfig:
    """Halo- and memory-aware diffusion-VAE decode tiling (:func:`recommended_decode_tiling_config`).
    Pass ``keyframes=True`` for a keyframe-aware ``decode_video(keyframes=)`` decode: joint attention runs
    eager blocks, which changes both the stage-5 coefficient and the withheld reserve.
    """
    require_diffusion_video_vae(vae_checkpoint_path)
    metadata = SafetensorsModelStateDictLoader().metadata(vae_checkpoint_path)
    vae_config = metadata.get("config", {}).get("vae", {})
    geometry = diffvae_tiling_geometry_from_vae_config(vae_config)
    model_bytes = estimate_diffusion_decoder_weight_bytes(vae_checkpoint_path)

    resolved_device = device if device is not None else get_device()
    if free_bytes is None:
        free_bytes = activation_budget_bytes(resolved_device)

    return recommended_decode_tiling_config(
        **geometry,
        height=height,
        width=width,
        num_frames=num_frames,
        mode=diffvae_optimization,
        free_bytes=free_bytes,
        model_bytes=model_bytes,
        keyframes=keyframes,
        joint_sdpa_materializes=keyframes and joint_sdpa_materializes_scores(resolved_device),
    )


def ensure_tiling_config(
    tiling_config: TilingConfig | AutoTiling | None,
    *,
    scale_factors: SpatioTemporalScaleFactors,
    video_shape: VideoPixelShape,
    vae_checkpoint_path: str,
    diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
    device: torch.device | None = None,
    free_bytes: int | None = None,
    keyframes: bool = False,
) -> TilingConfig | None:
    """Resolve pipeline tiling: ``AUTO_TILING`` → recommend; ``None`` → untiled; else validate.
    ``scale_factors`` must be the same factors decode will pass to ``to_splitters``
    (from :func:`tiling_scale_factors_for_vae`). Call only after ``video_shape.frames`` is known.
    Pass ``keyframes=True`` when the decode will use the keyframe-aware VAE path.
    """
    if tiling_config is None:
        return None
    if isinstance(tiling_config, AutoTiling):
        return tiling_config_for_vae(
            vae_checkpoint_path,
            height=video_shape.height,
            width=video_shape.width,
            num_frames=video_shape.frames,
            diffvae_optimization=diffvae_optimization,
            device=device,
            free_bytes=free_bytes,
            keyframes=keyframes,
        )
    tiling_config.validate(scale_factors, video_shape)
    return tiling_config


def combined_image_conditionings(
    images: list[ImageConditioningInput],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
    color_space: HDRColorSpace | None = None,
) -> list[ConditioningItem]:
    """Create a list of conditionings by replacing the latent at the first frame with the encoded image if present
    and using other encoded images as the keyframe conditionings."""
    conditionings = []
    for img in images:
        image = load_image_and_preprocess(
            image_path=img.path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            crf=img.crf,
            color_space=color_space,
        )
        encoded_image = video_encoder(image)
        if img.frame_idx == 0:
            conditioning = VideoConditionByLatentIndex(
                latent=encoded_image,
                strength=img.strength,
                latent_idx=0,
            )
        else:
            conditioning = VideoConditionByKeyframeIndex(
                keyframes=encoded_image,
                strength=img.strength,
                frame_idx=img.frame_idx,
            )
        conditionings.append(conditioning)
    return conditionings


def evenly_spaced_keyframe_positions(num_keyframes: int, num_frames: int) -> list[int]:
    """Interior pixel-frame positions for *num_keyframes* generated keyframes. Endpoints excluded."""
    if num_keyframes < 0:
        raise ValueError(f"num_keyframes must be non-negative, got {num_keyframes}")
    if num_keyframes == 0:
        return []
    if num_frames < num_keyframes + 2:
        raise ValueError(
            f"Generated keyframes need at least num_keyframes + 2 target frames, got "
            f"num_keyframes={num_keyframes}, num_frames={num_frames}"
        )
    return torch.linspace(0, num_frames - 1, num_keyframes + 2).round().to(torch.int64).tolist()[1:-1]


def has_generated_keyframes(generated_keyframes: int | Sequence[int]) -> bool:
    """Whether a ``generated_keyframes`` request asks for any slots.
    Callers must not test the argument's truthiness directly: a ``Sequence`` may be a tensor or an
    array, whose truth value is ambiguous and raises.
    """
    if isinstance(generated_keyframes, int):
        return generated_keyframes > 0
    return len(generated_keyframes) > 0


def resolve_generated_keyframes(
    generated_keyframes: int | Sequence[int],
    num_frames: int,
) -> list[int]:
    """Normalize the pipeline-level ``generated_keyframes`` argument to pixel-frame indices.
    An ``int`` requests that many evenly spaced interior keyframes; a sequence gives the target
    pixel-frame indices explicitly. ``0`` / empty means the feature is off.
    """
    if isinstance(generated_keyframes, int):
        return evenly_spaced_keyframe_positions(generated_keyframes, num_frames)

    positions = sorted({int(position) for position in generated_keyframes})
    if positions and (positions[0] < 0 or positions[-1] >= num_frames):
        raise ValueError(
            f"Generated keyframe positions must lie in [0, {num_frames}), got "
            f"{sorted(int(p) for p in generated_keyframes)}"
        )
    return positions


def generated_keyframe_conditionings(
    generated_keyframes: int | Sequence[int],
    num_frames: int,
) -> list[ConditioningItem]:
    """Build the generated-keyframe conditioning, or an empty list when the feature is off.
    All slots go into one :class:`VideoGeneratedKeyframeSlots` so their tokens form a single
    contiguous, exactly-locatable range.
    """
    positions = resolve_generated_keyframes(generated_keyframes, num_frames)
    if not positions:
        return []
    return [VideoGeneratedKeyframeSlots(pixel_frame_indices=positions)]


def create_noised_state(
    tools: LatentTools,
    conditionings: list[ConditioningItem],
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
    noise_scale: float = 1.0,
    initial_latent: torch.Tensor | None = None,
) -> LatentState:
    """Create a noised latent state from empty state, conditionings, and noiser.
    Creates an empty latent state, applies conditionings, and then adds noise
    using the provided noiser. Returns the final noised state ready for diffusion.
    """
    state = tools.create_initial_state(device, dtype, initial_latent)
    state = state_with_conditionings(state, conditionings, tools)
    state = noiser(state, noise_scale)

    return state


def state_with_conditionings(
    latent_state: LatentState, conditioning_items: list[ConditioningItem], latent_tools: LatentTools
) -> LatentState:
    """Apply a list of conditionings to a latent state.
    Iterates through the conditioning items and applies each one to the latent
    state in sequence. Returns the modified state with all conditionings applied.
    """
    for conditioning in conditioning_items:
        latent_state = conditioning.apply_to(latent_state=latent_state, latent_tools=latent_tools)

    return latent_state


def post_process_latent(denoised: torch.Tensor, denoise_mask: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Blend denoised output with clean state based on mask."""
    return (denoised * denoise_mask + clean.float() * (1 - denoise_mask)).to(denoised.dtype)


def modality_from_latent_state(
    state: LatentState,
    context: torch.Tensor,
    sigma: torch.Tensor,
    enabled: bool = True,
) -> Modality:
    """Create a Modality from a latent state.
    Constructs a Modality object with the latent state's data, timesteps derived
    from the denoise mask and sigma, positions, and the provided context.
    When ``state.frozen`` is True, ``Modality.sigma`` is forced to 0 so prompt AdaLN
    and cross-modality gates match frozen conditioning (per-token timesteps are
    already zeroed via ``denoise_mask``).
    """
    if state.frozen:
        sigma = torch.zeros_like(sigma)
    return Modality(
        enabled=enabled,
        latent=state.latent,
        sigma=sigma,
        timesteps=timesteps_from_mask(state.denoise_mask, sigma),
        positions=state.positions,
        context=context,
        context_mask=None,
        attention_mask=state.attention_mask,
        keyframes_mask=state.keyframes_mask,
    )


def timesteps_from_mask(denoise_mask: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
    """Compute timesteps from a denoise mask and sigma value.
    Multiplies the denoise mask by sigma to produce timesteps for each position
    in the latent state. Areas where the mask is 0 will have zero timesteps.
    When sigma is ``(B,)`` it is reshaped to ``(B, 1, ...)`` so the batch
    dimension aligns correctly with ``denoise_mask``.
    """
    if isinstance(sigma, torch.Tensor) and sigma.dim() == 1:
        sigma = sigma.view(-1, *([1] * (denoise_mask.dim() - 1)))
    return denoise_mask * sigma


_UNICODE_REPLACEMENTS = str.maketrans("\u2018\u2019\u201c\u201d\u2014\u2013\u00a0\u2032\u2212", "''\"\"-- '-")


def clean_response(text: str) -> str:
    """Clean a response from curly quotes and leading non-letter characters which Gemma tends to insert."""
    text = text.translate(_UNICODE_REPLACEMENTS)

    # Remove leading non-letter characters
    for i, char in enumerate(text):
        if char.isalpha():
            return text[i:]
    return text


def generate_enhanced_prompt(
    text_encoder: LTXGemmaTextEncoder,
    prompt: str,
    image_path: str | None = None,
    image_long_side: int = 896,
    seed: int = 42,
    static_cache: bool = False,
) -> str:
    """Generate an enhanced prompt from a text encoder and a prompt."""
    if image_path:
        image = decode_image(image_path=image_path)
        image = torch.tensor(image)
        image = resize_aspect_ratio_preserving(image, image_long_side).to(torch.uint8)
        prompt = text_encoder.enhance_i2v(prompt, image, seed=seed, static_cache=static_cache)
    else:
        prompt = text_encoder.enhance_t2v(prompt, seed=seed, static_cache=static_cache)
    logging.info("Enhanced prompt: %s", prompt)
    return clean_response(prompt)


def assert_resolution(height: int, width: int, is_two_stage: bool, *, divisor: int | None = None) -> None:
    """Assert that the resolution is divisible by the required divisor.
    For two-stage pipelines, the resolution must be divisible by 64.
    For one-stage pipelines, the resolution must be divisible by 32.
    Pass ``divisor`` to override (e.g. 128 for a two-spatial-upsample DFR run).
    """
    if divisor is None:
        divisor = 64 if is_two_stage else 32
    if height % divisor != 0 or width % divisor != 0:
        raise ValueError(
            f"Resolution ({height}x{width}) is not divisible by {divisor}. "
            f"For {'two-stage' if is_two_stage else 'one-stage'} pipelines, "
            f"height and width must be multiples of {divisor}."
        )


def snap_frames_to_grid(frames: int, scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS) -> int:
    """Round ``frames`` down to the nearest ``k * scale_factors.time + 1``.
    The model's frame count must satisfy ``(frames - 1) % scale_factors.time == 0``
    (causal VAE temporal grid).
    """
    if frames < 1:
        raise ValueError(f"frames must be >= 1, got {frames}")
    time_scale = scale_factors.time
    return ((frames - 1) // time_scale) * time_scale + 1


def seconds_to_clamped_num_frames(
    seconds: float,
    *,
    frame_rate: float,
    min_frames: int = 1,
    max_frames: int = 1024,
    scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
) -> int:
    """Convert a duration in seconds to a frame count snapped to the VAE's temporal grid.
    Outlier durations are clamped to ``[min_frames, max_frames]`` (before snapping) so a
    misbehaving prediction can't request an OOM-sized generation. Snapping floors to the
    grid, which can undershoot ``min_frames``; when that happens the result is snapped up
    to the next grid point instead, so the ``[min_frames, max_frames]`` contract always holds.
    """
    raw_frames = round(seconds * frame_rate)
    raw_frames = max(min_frames, min(raw_frames, max_frames))
    frames = snap_frames_to_grid(raw_frames, scale_factors)
    if frames < min_frames:
        time_scale = scale_factors.time
        frames = min(-(-(min_frames - 1) // time_scale) * time_scale + 1, max_frames)
    return frames

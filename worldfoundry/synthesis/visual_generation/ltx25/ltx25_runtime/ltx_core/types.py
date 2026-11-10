"""Denoising-loop latent state for the LTX-2.5 pipelines."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GeneratedKeyframeLayout:
    """Where a state's generated-keyframe slot tokens live, and what they represent.
    Recorded by :class:`~ltx_core.conditioning.types.keyframe_slots.VideoGeneratedKeyframeSlots`
    when it appends the slots, so they can later be located and extracted *exactly* rather
    than by assuming they are the trailing tokens. Conditioning items are applied in list
    order and each appends to the end, so a state built with slots plus any other appending
    conditioning item has no fixed trailing layout.
    Attributes:
        pixel_frame_indices: Target pixel-frame index of each slot, in token order.
        tokens_per_keyframe: Number of tokens one slot occupies (one latent frame's worth).
        first_token: Index of the first slot token in the token sequence.
    """

    pixel_frame_indices: tuple[int, ...]
    tokens_per_keyframe: int
    first_token: int

    @property
    def num_keyframes(self) -> int:
        return len(self.pixel_frame_indices)

    @property
    def num_tokens(self) -> int:
        return self.num_keyframes * self.tokens_per_keyframe

    @property
    def token_slice(self) -> slice:
        return slice(self.first_token, self.first_token + self.num_tokens)


@dataclass(frozen=True)
class LatentState:
    """
    State of latents during the diffusion denoising process.
    Attributes:
        latent: The current noisy latent tensor being denoised.
        denoise_mask: Mask encoding the denoising strength for each token (1 = full denoising, 0 = no denoising).
        positions: Positional indices for each latent element, used for positional embeddings.
        clean_latent: Initial state of the latent before denoising, may include conditioning latents.
        attention_mask: Optional 2D self-attention mask of shape (B, T, T). Values in [0, 1] where 1 = full attention,
            0 = no attention. None means full attention everywhere. Built incrementally by conditioning items.
        keyframes_mask: Optional per-token marker of shape (B, T, 1) -- same layout as
            ``denoise_mask`` -- non-zero on tokens whose latent encodes a *single standalone pixel
            frame* rather than the usual multi-frame span. That set is the target's first latent
            frame (the video encoder is causal, so its first temporal latent frame covers 1 pixel
            frame while the rest cover 8) plus any generated keyframe slots. Selects the tokens
            that receive the model's learned keyframe absolute-position embedding; ignored
            entirely by models built without ``use_keyframes_abs_pos_embedding``.
        generated_keyframe_layout: Set when generated keyframe slots were appended; locates them.
        generated_keyframes: Populated by ``clear_conditioning`` when a layout is present: the
            denoised slot content as an unpatchified ``(B, C, K, H, W)`` latent, one latent frame
            per keyframe. Each frame must be decoded as a standalone one-frame clip, never as a
            K-frame video -- a causal decode would blend slots that were never adjacent.
        frozen: When True, this stream is held fixed: token denoising is disabled (``denoise_mask``
            should be all zeros; pipeline builders enforce that) and the scalar noise level used for
            prompt / cross-modality AdaLN gates is forced to 0 when the state is converted for the
            transformer.
    """

    latent: torch.Tensor
    denoise_mask: torch.Tensor
    positions: torch.Tensor
    clean_latent: torch.Tensor
    attention_mask: torch.Tensor | None = None
    keyframes_mask: torch.Tensor | None = None
    generated_keyframe_layout: GeneratedKeyframeLayout | None = None
    generated_keyframes: torch.Tensor | None = None
    frozen: bool = False

    def clone(self) -> "LatentState":
        return LatentState(
            latent=self.latent.clone(),
            denoise_mask=self.denoise_mask.clone(),
            positions=self.positions.clone(),
            clean_latent=self.clean_latent.clone(),
            attention_mask=self.attention_mask.clone() if self.attention_mask is not None else None,
            keyframes_mask=self.keyframes_mask.clone() if self.keyframes_mask is not None else None,
            generated_keyframe_layout=self.generated_keyframe_layout,
            generated_keyframes=(self.generated_keyframes.clone() if self.generated_keyframes is not None else None),
            frozen=self.frozen,
        )

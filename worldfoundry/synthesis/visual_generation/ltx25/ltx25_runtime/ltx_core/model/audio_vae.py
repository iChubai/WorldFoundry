"""Checkpoint key rules for the LTX audio VAE decoder and vocoder.

The modules are the canonical ``autoencoders.ltx.audio`` implementations; these
operations map released safetensors keys onto them for the staged builders.
"""

import torch

from ..loader.sd_ops import KeyValueOperationResult, SDOps


def _strip_vocoder_prefix(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
    """Strip the leading 'vocoder.' prefix exactly once.
    Uses removeprefix instead of str.replace so that BWE keys like
    'vocoder.vocoder.conv_pre' become 'vocoder.conv_pre' (not 'conv_pre').
    Works identically for legacy keys like 'vocoder.conv_pre' → 'conv_pre'.
    """
    return [KeyValueOperationResult(key.removeprefix("vocoder."), value)]


VOCODER_COMFY_KEYS_FILTER = (
    SDOps("VOCODER_COMFY_KEYS_FILTER")
    .with_matching(prefix="vocoder.")
    .with_kv_operation(operation=_strip_vocoder_prefix, key_prefix="vocoder.")
)

AUDIO_VAE_DECODER_COMFY_KEYS_FILTER = (
    SDOps("AUDIO_VAE_DECODER_COMFY_KEYS_FILTER")
    .with_matching(prefix="audio_vae.decoder.")
    .with_matching(prefix="audio_vae.per_channel_statistics.")
    .with_replacement("audio_vae.decoder.", "")
    .with_replacement("audio_vae.per_channel_statistics.", "per_channel_statistics.")
)

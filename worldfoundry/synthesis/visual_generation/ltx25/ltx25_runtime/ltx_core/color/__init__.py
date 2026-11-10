"""Colour transforms, primaries, YUV packing, and HLG encode.
VAE working-space helpers (``HDRTransfer``, ``TransferEncoding``) live in
:mod:`ltx_core.hdr` — import them from there. This package does not re-export
hdr symbols (that would cycle: ``hdr`` imports ``color.primaries``).
"""

from .hlg import HlgGpuConverter, HlgPyAVEncoder, encode_linear_hdr_frames_to_hlg_mp4, hlg_inverse_oetf
from .primaries import Primaries
from .yuv import (
    ColorRange,
    ColorSpace,
    FrameConverter,
    PixelFormat,
    rgb_to_yuv,
    rgb_to_yuv420,
    rgb_to_yuv420p10,
    yuv420p_bt709_converter_,
)

__all__ = [
    "ColorRange",
    "ColorSpace",
    "FrameConverter",
    "HlgGpuConverter",
    "HlgPyAVEncoder",
    "PixelFormat",
    "Primaries",
    "encode_linear_hdr_frames_to_hlg_mp4",
    "hlg_inverse_oetf",
    "rgb_to_yuv",
    "rgb_to_yuv420",
    "rgb_to_yuv420p10",
    "yuv420p_bt709_converter_",
]

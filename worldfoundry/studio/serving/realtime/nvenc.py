# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in NVENC hardware H.264 encoding for the Studio realtime WebRTC track.

Isolated from the default software path so importing the realtime serving
surface never drags ``PyNvVideoCodec`` in with it: that library's import has
global side effects (CUDA driver init, shared-library loading) that the
software path should not pay. :func:`nvenc_h264_supported` probes availability
with :func:`importlib.util.find_spec` and this module is only imported when the
hardware track is actually requested.

Design note: WorldFoundry's realtime frame buffer (``LatestFrameBuffer``)
delivers frames as CPU ``numpy`` arrays that were already resized and copied
off the GPU by the generation worker. NVENC's largest win — skipping the
GPU->CPU copy — is therefore already spent by the time a frame reaches the
track. What NVENC still buys here is moving the H.264 encode itself off the CPU
(aiortc's default ``libx264``/``openh264``) onto the GPU's dedicated encoder
ASIC, freeing CPU under high resolution/fps. The encoder re-uploads the CPU RGB
frame to CUDA for encoding; enabling it is only worthwhile when profiling shows
CPU software encode is the realtime bottleneck.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from fractions import Fraction
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from av.packet import Packet

# H.264 Annex-B NAL type identifier for an IDR (keyframe) slice.
_H264_NAL_TYPE_IDR = 5

# RTP video clock, per RFC 6184. aiortc's H264Encoder.pack() rescales from the
# packet's declared time_base into this base, so stamping packets on 1/90000 is
# the lowest-conversion choice.
_RTP_VIDEO_CLOCK = 90_000


def nvenc_h264_supported() -> tuple[bool, str]:
    """Return whether the NVENC hardware path can be imported and used.

    Cheap, side-effect-free probe: checks that ``PyNvVideoCodec`` is installed
    without importing it and that a CUDA device is visible. It does not allocate
    an encoder session. Returns ``(ok, reason)`` where ``reason`` is a
    human-readable diagnostic when ``ok`` is ``False``.
    """
    if find_spec("PyNvVideoCodec") is None:
        return False, "PyNvVideoCodec is not installed"
    try:
        import torch
    except ImportError:
        return False, "torch is not available"
    if not torch.cuda.is_available():
        return False, "no CUDA device is available"
    return True, ""


def _payload_contains_nal_type(payload: bytes, nal_type: int) -> bool:
    """Scan an Annex-B H.264 payload for the presence of a specific NAL type."""
    i = 0
    while True:
        idx = payload.find(b"\x00\x00\x01", i)
        if idx < 0:
            return False
        nal_start = idx + 3
        if nal_start >= len(payload):
            return False
        if (payload[nal_start] & 0x1F) == nal_type:
            return True
        i = nal_start + 1


def _rgb_frame_to_abgr_cuda(frame_rgb_uint8: np.ndarray, torch_module: Any) -> Any:
    """Upload one HWC uint8 RGB frame to CUDA as an NVENC ``ABGR`` tensor.

    NVENC ``NV_ENC_BUFFER_FORMAT_ABGR`` is a *word-ordered* token, not
    memory-ordered: a pixel is the 32-bit word ``0xAABBGGRR``, which in
    little-endian memory is the byte sequence ``[R, G, B, A]``. So the
    channel-last tensor handed to the encoder must have channel 0=R, 1=G, 2=B,
    3=A. ABGR (rather than NV12) lets NVENC's driver-side RGB->YUV conversion
    handle the colour transform instead of a bespoke NV12 kernel.
    """
    if frame_rgb_uint8.ndim != 3 or frame_rgb_uint8.shape[2] != 3:
        raise ValueError(
            "expected an HWC RGB frame with 3 channels; "
            f"got shape {tuple(frame_rgb_uint8.shape)}"
        )
    rgb = torch_module.from_numpy(np.ascontiguousarray(frame_rgb_uint8)).cuda()
    if rgb.dtype != torch_module.uint8:
        rgb = rgb.clamp(0, 255).to(torch_module.uint8)
    h, w, _ = rgb.shape
    alpha = torch_module.full((h, w, 1), 255, dtype=torch_module.uint8, device=rgb.device)
    rgba = torch_module.cat([rgb, alpha], dim=-1)
    return rgba.contiguous()


class NVENCFrameEncoder:
    """NVENC H.264 encoder backed by ``PyNvVideoCodec``.

    Accepts CPU RGB ``numpy`` frames (the format the realtime buffer delivers),
    uploads each to CUDA, and emits Annex-B H.264 packets stamped on the RTP
    90 kHz clock so aiortc's ``H264Encoder.pack()`` can rescale losslessly.
    """

    backend = "pynvvideocodec"

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        bitrate: int,
        gpu_id: int = 0,
        gop: int = 30,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be > 0, got {width}x{height}")
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        if bitrate <= 0:
            raise ValueError(f"bitrate must be > 0, got {bitrate}")
        if gop <= 0:
            raise ValueError(f"gop must be > 0, got {gop}")

        import PyNvVideoCodec as nvc
        import torch

        self._torch = torch
        self.fps = fps
        self._time_base = Fraction(1, _RTP_VIDEO_CLOCK)
        self._pts_counter = 0
        self._force_idr_flag = int(nvc.FORCEIDR)

        # repeatspspps=1 prepends SPS+PPS to every IDR: aiortc's
        # H264Encoder.pack() does not synthesize parameter sets, so the RTP
        # stream must carry them in-band or the receiver cannot lock on. bf=0
        # and lookahead=0 keep output strictly 1:1 with input frames, which is
        # what interactive streaming needs.
        self._encoder = nvc.CreateEncoder(
            width=width,
            height=height,
            fmt="ABGR",
            usecpuinputbuffer=False,
            codec="h264",
            preset="P4",
            tuning_info="ultra_low_latency",
            rc="cbr",
            fps=fps,
            bitrate=bitrate,
            bf=0,
            lookahead=0,
            repeatspspps=1,
            idrperiod=gop,
        )

    def encode_frame(
        self,
        frame_rgb_uint8: np.ndarray,
        *,
        force_keyframe: bool = False,
        on_packet: Callable[[Packet], None] | None = None,
    ) -> int:
        """Encode one RGB frame; return the number of packets emitted."""
        from av.packet import Packet

        cuda_frame = _rgb_frame_to_abgr_cuda(frame_rgb_uint8, self._torch)
        if force_keyframe:
            bitstream = self._encoder.Encode(cuda_frame, self._force_idr_flag)
        else:
            bitstream = self._encoder.Encode(cuda_frame)
        if not bitstream:
            return 0
        payload = bytes(bitstream)
        packet = Packet(payload)
        packet.pts = (self._pts_counter * _RTP_VIDEO_CLOCK) // self.fps
        packet.time_base = self._time_base
        self._pts_counter += 1
        if on_packet is not None:
            on_packet(packet)
        return 1

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._encoder.EndEncode()


def bitrate_for_resolution(width: int, height: int, fps: int) -> int:
    """Return a CBR H.264 target bitrate (bps) from resolution and fps.

    Uses ~0.10 bits per pixel per frame, clamped to a sane interactive range.
    This is a heuristic default; override via the caller when a tuned value is
    known for the target stream.
    """
    bits_per_pixel_per_frame = 0.10
    estimated = int(width * height * fps * bits_per_pixel_per_frame)
    return max(1_000_000, min(estimated, 20_000_000))


def build_nvenc_track(
    *,
    frames: Any,
    fps: int,
    maxsize: int = 8,
    gpu_id: int = 0,
    gop: int = 30,
    bitrate: int | None = None,
) -> Any:
    """Build an aiortc track that streams NVENC-encoded packets.

    ``frames`` is the realtime ``LatestFrameBuffer``. A background task pulls
    numpy RGB frames, encodes them on the GPU, and feeds pre-encoded packets to
    the track. ``recv()`` returns ``av.Packet`` (not ``VideoFrame``), which
    aiortc routes through ``H264Encoder.pack()`` for RTP fragmentation only —
    the software encoder is bypassed.

    The encoder is constructed lazily from the first frame's resolution, since
    NVENC requires fixed dimensions and the buffer exposes no non-destructive
    peek. ``bitrate`` defaults to :func:`bitrate_for_resolution`.
    """
    from aiortc import MediaStreamTrack
    from aiortc.mediastreams import MediaStreamError
    from av.packet import Packet

    class NVENCVideoTrack(MediaStreamTrack):
        kind = "video"

        def __init__(self) -> None:
            super().__init__()
            self._packets: asyncio.Queue[Packet | None] = asyncio.Queue(maxsize=maxsize)
            self._interval = 1.0 / fps
            self._closed = False
            self._pump: asyncio.Task[None] | None = None
            self._last_frame: np.ndarray | None = None
            self._first_frame = True
            self._encoder: NVENCFrameEncoder | None = None

        def start(self) -> None:
            if self._pump is None:
                self._pump = asyncio.ensure_future(self._pump_frames())

        def _ensure_encoder(self, frame: np.ndarray) -> NVENCFrameEncoder:
            if self._encoder is None:
                height, width = int(frame.shape[0]), int(frame.shape[1])
                self._encoder = NVENCFrameEncoder(
                    width=width,
                    height=height,
                    fps=fps,
                    bitrate=bitrate or bitrate_for_resolution(width, height, fps),
                    gpu_id=gpu_id,
                    gop=gop,
                )
            return self._encoder

        async def _pump_frames(self) -> None:
            loop = asyncio.get_running_loop()
            next_pull = loop.time()
            try:
                while not self._closed:
                    if self._last_frame is None:
                        try:
                            self._last_frame = await frames.get()
                        except EOFError:
                            break
                    else:
                        next_pull += self._interval
                        now = loop.time()
                        if next_pull > now:
                            await asyncio.sleep(next_pull - now)
                        elif now - next_pull > self._interval:
                            next_pull = now
                        try:
                            self._last_frame = frames.get_nowait()
                        except asyncio.QueueEmpty:
                            # Repeat the held frame to keep a steady RTP clock
                            # and flush the encoder's final source frame.
                            pass
                        except EOFError:
                            break
                    encoder = self._ensure_encoder(self._last_frame)
                    await asyncio.to_thread(
                        encoder.encode_frame,
                        self._last_frame,
                        force_keyframe=self._first_frame,
                        on_packet=self._enqueue_threadsafe(loop),
                    )
                    self._first_frame = False
            finally:
                with contextlib.suppress(Exception):
                    self._packets.put_nowait(None)

        def _enqueue_threadsafe(self, loop: asyncio.AbstractEventLoop) -> Callable[[Packet], None]:
            def _enqueue(packet: Packet) -> None:
                asyncio.run_coroutine_threadsafe(self._packets.put(packet), loop)

            return _enqueue

        async def recv(self) -> Packet:
            if self._closed:
                raise MediaStreamError
            self.start()
            packet = await self._packets.get()
            if packet is None:
                raise MediaStreamError
            return packet

        async def close(self) -> None:
            if self._closed:
                return
            self._closed = True
            if self._pump is not None:
                self._pump.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._pump
            if self._encoder is not None:
                self._encoder.close()

    track = NVENCVideoTrack()
    track.start()
    return track


__all__ = [
    "NVENCFrameEncoder",
    "bitrate_for_resolution",
    "build_nvenc_track",
    "nvenc_h264_supported",
]

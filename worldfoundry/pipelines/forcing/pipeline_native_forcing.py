"""Native Self-Forcing text-to-video pipeline and its resident realtime session."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import torch

from worldfoundry.core.execution.realtime.contracts import RealtimeSpec
from worldfoundry.core.media.codecs.video import video_tensor_to_uint8_frames
from worldfoundry.core.media.processing.postprocess import VideoChunk

from ..native_diffusion import NativeVisualDiffusionPipeline

if TYPE_CHECKING:
    from worldfoundry.base_models.diffusion_model.contracts import DiffusionRequest
    from worldfoundry.base_models.diffusion_model.models.autoencoders.wan.component import WanVideoDecoder
    from worldfoundry.base_models.diffusion_model.pipeline import NativeDiffusionPipeline
    from worldfoundry.base_models.diffusion_model.runners.autoregressive import (
        AutoregressiveStream,
        AutoregressiveWindowRunner,
    )


class NativeSelfForcingPipeline(NativeVisualDiffusionPipeline):
    """Four-step causal Self-Forcing (Wan2.1 T2V 1.3B DMD) on the native window runner.

    The realtime methods keep one rollout resident. ``configure_realtime`` opens an
    :class:`AutoregressiveStream` and a causal Wan codec state, each
    ``stream_realtime`` call commits and decodes one latent block, and
    ``reset_realtime`` releases both while the weights stay loaded. Streaming a
    rollout to completion yields the frames of one ``__call__`` with the same
    prompt, seed and geometry.
    """

    MODEL_ID = "self-forcing"
    OWNER = "Self-Forcing"
    CHECKPOINT_ROLES = ("dit", "text-encoder", "tokenizer", "vae")
    PRIMARY_CHECKPOINT_ROLE = "dit"
    GENERATION_TYPE = "t2v"
    DEFAULT_HEIGHT = 480
    DEFAULT_WIDTH = 832
    DEFAULT_NUM_FRAMES = 81
    DEFAULT_NUM_INFERENCE_STEPS = 4
    DEFAULT_GUIDANCE_SCALE = 1.0
    DEFAULT_FPS = 16
    DEFAULT_NEGATIVE_PROMPT = ""
    DEFAULT_SCHEDULER_OPTIONS: Mapping[str, object] = {"shift": 5.0}

    def __init__(
        self,
        *,
        native_pipeline: NativeDiffusionPipeline,
        device: str,
        model_id: str | None = None,
    ) -> None:
        super().__init__(native_pipeline=native_pipeline, device=device, model_id=model_id)
        self._realtime_stream: AutoregressiveStream | None = None
        self._realtime_decode_state: dict[str, object] | None = None
        self._realtime_prompt = ""
        self._realtime_seed = 0
        self._realtime_options: dict[str, Any] = {}

    @property
    def _window_runner(self) -> AutoregressiveWindowRunner:
        return cast("AutoregressiveWindowRunner", self.native_pipeline.runner)

    @property
    def _codec(self) -> WanVideoDecoder:
        return cast("WanVideoDecoder", self._window_runner.components.decoder)

    def _realtime_spec(self, max_chunks: int) -> RealtimeSpec:
        """One chunk per latent block: the causal codec emits 9 frames first, then 12."""
        block_size = int(self._window_runner.windowed_denoiser.block_size)
        codec = self._codec
        return RealtimeSpec(
            fps=self.DEFAULT_FPS,
            first_chunk_frames=codec.get_pixel_num_frames(block_size),
            steady_chunk_frames=block_size * codec.temporal_compression_factor,
            controls=("prompt_update",),
            keys=(),
            advance="continuous",
            max_chunks=max_chunks,
        )

    def prepare_realtime(self) -> dict[str, Any]:
        """Report the cadence of a default-length rollout (7 chunks for 81 frames)."""
        block_size = int(self._window_runner.windowed_denoiser.block_size)
        latent_frames = self._codec.get_latent_num_frames(self.DEFAULT_NUM_FRAMES)
        return {"realtime_spec": self._realtime_spec(latent_frames // block_size).to_payload()}

    def _open_realtime_rollout(self, prompt: str, request: DiffusionRequest) -> AutoregressiveStream:
        """Open the rollout of ``request`` with fresh codec history."""
        stream = self._window_runner.open_stream(request)
        self._realtime_stream = stream
        self._realtime_decode_state = self._codec.new_stream_state()
        self._realtime_prompt = prompt.strip()
        return stream

    def configure_realtime(
        self,
        images: Any = None,
        prompt: str = "",
        seed: int = 42,
        fps: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Encode ``prompt`` and open a resident rollout without generating frames.

        ``height``, ``width``, ``num_frames`` and the other ``__call__`` options shape
        the request. Playback runs at the model's 16 fps, so ``fps`` is ignored.
        """
        del fps
        self.reset_realtime()
        started = time.perf_counter()
        request, _ = self._prepare_call(prompt=prompt, images=images, seed=seed, **kwargs)
        self._realtime_seed = request.sampling.seed
        self._realtime_options = dict(kwargs)
        try:
            stream = self._open_realtime_rollout(prompt, request)
        except BaseException:
            self.reset_realtime()
            raise
        return {
            "status": "configured",
            "realtime_spec": self._realtime_spec(stream.total_blocks).to_payload(),
            "realtime_metrics": {"configure_ms": (time.perf_counter() - started) * 1000.0},
        }

    def stream_realtime(
        self,
        prompt: str = "",
        interactions: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Commit and decode the next latent block as uint8 THWC frames.

        A non-empty ``prompt`` that differs from the rollout's prompt first reopens
        the rollout with that prompt and the configured seed. ``realtime_metrics``
        reports ``model_ms``, ``decode_ms``, the 1-based ``chunk_index`` and
        ``chunks_total``. A failure releases the rollout; configure again to resume.
        """
        del interactions, kwargs
        stream = self._realtime_stream
        if stream is None:
            raise RuntimeError("Self-Forcing realtime session is not configured; call configure_realtime() first.")
        prompt = str(prompt or "").strip()
        if prompt and prompt != self._realtime_prompt:
            stream.close()
            try:
                request, _ = self._prepare_call(prompt=prompt, seed=self._realtime_seed, **self._realtime_options)
                stream = self._open_realtime_rollout(prompt, request)
            except BaseException:
                self.reset_realtime()
                raise
        if stream.done:
            raise RuntimeError(
                f"Self-Forcing realtime rollout already produced all {stream.total_blocks} chunks; "
                "update the prompt or reset to start a new rollout."
            )
        runner = self._window_runner
        synchronize = runner.device.type == "cuda"
        try:
            first_chunk = stream.completed_blocks == 0
            started = time.perf_counter()
            latents = stream.step()
            if synchronize:
                torch.cuda.synchronize(runner.device)
            decode_started = time.perf_counter()
            pixels = self._codec.decode_chunk(
                latents,
                state=cast(dict, self._realtime_decode_state),
                is_first_chunk=first_chunk,
            )
            if synchronize:
                torch.cuda.synchronize(runner.device)
            decode_finished = time.perf_counter()
            frames = video_tensor_to_uint8_frames(pixels, value_range="-1,1", layout="bcthw")
        except BaseException:
            self.reset_realtime()
            raise
        if stream.done:
            # Release the KV caches now; the finished stream still reports its block counts.
            stream.close()
        return {
            "video": frames,
            "video_chunk": VideoChunk(frames=frames, layout="thwc", value_range="0,255"),
            "realtime_spec": self._realtime_spec(stream.total_blocks).to_payload(),
            "realtime_metrics": {
                "model_ms": (decode_started - started) * 1000.0,
                "decode_ms": (decode_finished - decode_started) * 1000.0,
                "chunk_index": stream.completed_blocks,
                "chunks_total": stream.total_blocks,
            },
        }

    def reset_realtime(self) -> None:
        """Close the open rollout and drop its codec history; weights stay resident."""
        stream = self._realtime_stream
        self._realtime_stream = None
        self._realtime_decode_state = None
        self._realtime_prompt = ""
        self._realtime_options = {}
        if stream is not None:
            stream.close()

    def cleanup(self) -> None:
        """Close any open realtime rollout before the pipeline is unloaded."""
        self.reset_realtime()


__all__ = ["NativeSelfForcingPipeline"]

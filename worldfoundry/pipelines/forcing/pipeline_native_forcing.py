"""Native Self-Forcing and Causal-Forcing pipelines and their resident realtime session."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, cast

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

    Loading with ``kv_window="sink5-window7-rerope"`` selects the approximate
    long-rollout cache: ``__call__`` accepts more than 81 frames, and the
    realtime rollout is unbounded (``max_chunks`` is ``None``) and runs until it
    is reset, its first ``num_frames`` matching ``__call__``.
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
    RUNTIME_POLICY_OPTION_KEYS = (*NativeVisualDiffusionPipeline.RUNTIME_POLICY_OPTION_KEYS, "kv_window")

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

    @property
    def _unbounded_rollout(self) -> bool:
        """Whether the denoiser's rolling KV cache lets a realtime rollout run until reset."""
        return getattr(self._window_runner.windowed_denoiser, "kv_window", None) is not None

    def _realtime_spec(self, max_chunks: int | None) -> RealtimeSpec:
        """One chunk per latent block, sized by the causal codec: 9 then 12 frames for three-frame blocks, 1 then 4 for one-frame blocks."""
        block_size = int(self._window_runner.windowed_denoiser.block_size)
        codec = self._codec
        return RealtimeSpec(
            fps=self.DEFAULT_FPS,
            first_chunk_frames=codec.get_pixel_num_frames(block_size + self._seeded_first_frames),
            steady_chunk_frames=block_size * codec.temporal_compression_factor,
            controls=("prompt_update",),
            keys=(),
            advance="continuous",
            max_chunks=max_chunks,
        )

    @property
    def _seeded_first_frames(self) -> int:
        """Latent frames the open rollout committed from an input image (0 or 1); its first chunk decodes them."""
        stream = self._realtime_stream
        return 0 if stream is None or stream.first_frame_latent is None else 1

    def prepare_realtime(self) -> dict[str, Any]:
        """Report the cadence of a default-length rollout (7 three-frame or 21 one-frame chunks for 81 frames), or an unbounded one."""
        if self._unbounded_rollout:
            return {"realtime_spec": self._realtime_spec(None).to_payload()}
        block_size = int(self._window_runner.windowed_denoiser.block_size)
        latent_frames = self._codec.get_latent_num_frames(self.DEFAULT_NUM_FRAMES)
        return {"realtime_spec": self._realtime_spec(latent_frames // block_size).to_payload()}

    def _open_realtime_rollout(self, prompt: str, request: DiffusionRequest) -> AutoregressiveStream:
        """Open the rollout of ``request`` with fresh codec history."""
        stream = self._window_runner.open_stream(request, unbounded=self._unbounded_rollout)
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
        self._realtime_options = dict(kwargs, images=images)
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

        The first chunk of an image-seeded rollout also decodes the image's latent frame.
        A non-empty ``prompt`` that differs from the rollout's prompt first reopens
        the rollout with that prompt and the configured seed. ``realtime_metrics``
        reports ``model_ms``, ``decode_ms``, the 1-based ``chunk_index`` and
        ``chunks_total``. A failure releases the rollout; configure again to resume.
        """
        del interactions, kwargs
        stream = self._realtime_stream
        if stream is None:
            raise RuntimeError(f"{self.OWNER} realtime session is not configured; call configure_realtime() first.")
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
                f"{self.OWNER} realtime rollout already produced all {stream.total_blocks} chunks; "
                "update the prompt or reset to start a new rollout."
            )
        runner = self._window_runner
        synchronize = runner.device.type == "cuda"
        try:
            first_chunk = stream.completed_blocks == 0
            started = time.perf_counter()
            latents = stream.step()
            if first_chunk and stream.first_frame_latent is not None:
                latents = torch.cat((stream.first_frame_latent, latents), dim=2)
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


class NativeCausalForcingPipeline(NativeSelfForcingPipeline):
    """Released Causal-Forcing generators on the native forcing rollout, offline and in realtime.

    ``model_id`` (or ``variant``) selects ``causal-forcing-chunkwise`` (three-latent-frame
    blocks; the public ``causal-forcing`` ID resolves to it), ``causal-forcing-framewise``
    (one-frame blocks) or the Causal Forcing++ frame-wise ``causal-forcing-framewise-1step``
    and ``-2step`` students, which spend four steps on the first frame. The
    frame-wise student also accepts one image, as the official ``--i2v`` route
    does: the image becomes the clean first latent frame and the rollout produces
    the remaining frames.
    """

    MODEL_ID = "causal-forcing"
    OWNER = "Causal-Forcing"
    ALLOW_MODEL_ID_OVERRIDE = True
    DEFAULT_STEPS_BY_MODEL_ID: ClassVar[Mapping[str, int]] = {
        "causal-forcing-chunkwise": 4,
        "causal-forcing-framewise": 4,
        "causal-forcing-framewise-1step": 1,
        "causal-forcing-framewise-2step": 2,
    }
    IMAGE_MODEL_IDS: ClassVar[frozenset[str]] = frozenset({"causal-forcing-framewise"})

    def __init__(
        self,
        *,
        native_pipeline: NativeDiffusionPipeline,
        device: str,
        model_id: str | None = None,
    ) -> None:
        resolved = native_pipeline.model_id
        if resolved not in self.DEFAULT_STEPS_BY_MODEL_ID:
            raise ValueError(
                f"{resolved!r} is not a native Causal-Forcing recipe; "
                f"expected one of {sorted(self.DEFAULT_STEPS_BY_MODEL_ID)}"
            )
        self.DEFAULT_NUM_INFERENCE_STEPS = self.DEFAULT_STEPS_BY_MODEL_ID[resolved]
        self.ACCEPTS_IMAGES = resolved in self.IMAGE_MODEL_IDS
        self.GENERATION_TYPE = "ti2v" if self.ACCEPTS_IMAGES else "t2v"
        super().__init__(native_pipeline=native_pipeline, device=device, model_id=model_id)


__all__ = ["NativeCausalForcingPipeline", "NativeSelfForcingPipeline"]

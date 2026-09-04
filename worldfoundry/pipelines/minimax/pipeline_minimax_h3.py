"""Bespoke MiniMax H3 audio-video pipeline (native, single-GPU).

MiniMax H3 couples video and audio in one packed-token denoise loop with
independent per-modality sigma schedules and pinned condition rows. That does
not fit WorldFoundry's declarative native-diffusion recipe path (whose
execution-strategy registry is frozen and whose runners assume a single shared
schedule), so this pipeline is a bespoke :class:`PipelineABC` subclass — the
same "Option B" approach HunyuanVideo uses — that owns the orchestration and
loads its components directly via the shared loaders.

Flow per request:

1. tokenize + encode the prompt/presentation with the Qwen3-VL encoder
   → ``[text_len, 5120]`` hidden states;
2. resolve target geometry (short_edge / aspect_ratio / duration) into aligned
   frame / latent-T dims;
3. build the packed sequence for the task (t2va / fl2va / ref2va);
4. seed initial video + audio noise, patchify keyframe/reference conditions;
5. run the coupled rectified-flow Euler eta=0 denoise loop over the DiT;
6. unpatchify + VAE-decode video and audio;
7. mux the 32 kHz stereo track into the output mp4.

The heavy component construction (DiT / VAEs / Qwen3-VL) is done lazily in
:meth:`from_pretrained`; the orchestration in :meth:`generate` accepts injected
components so it stays unit-testable without downloading weights.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from worldfoundry.base_models.diffusion_model.schedulers.minimax_h3 import (
    minimax_h3_align_frame_count,
    minimax_h3_audio_latent_t,
    minimax_h3_time_shift_sigmas,
    minimax_h3_video_latent_t,
)

from ..pipeline_utils import PipelineABC
from ._minimax_h3 import (
    MiniMaxH3DenoiseBranch,
    minimax_h3_denoise_loop,
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
    minimax_h3_task_profile,
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from ._minimax_h3.constants import (
    MINIMAX_H3_MAX_DURATION_SECONDS,
    MINIMAX_H3_MIN_DURATION_SECONDS,
    MINIMAX_H3_SUPPORTED_FPS,
)

_ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "21:9": (21, 9),
    "16:9": (16, 9),
    "4:3": (4, 3),
    "1:1": (1, 1),
    "3:4": (3, 4),
    "9:16": (9, 16),
}

# Video VAE compression: 16x spatial, 4x temporal (see the video VAE config).
_SPATIAL_COMPRESSION = 16
_PATCH_H = 2
_PATCH_W = 2
_VIDEO_LATENT_CHANNELS = 24
_AUDIO_LATENT_CHANNELS = 32
_AUDIO_CHANNELS = 2
_AUDIO_SAMPLE_RATE = 32000


def _round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def _is_module(value: Any) -> bool:
    """True only for an instantiated nn.Module (not a bare class placeholder)."""

    return isinstance(value, torch.nn.Module)


def _reverse_normalize_latents(
    latents: torch.Tensor,
    *,
    mean_values,
    std_values,
) -> torch.Tensor:
    """Undo VAE latent normalization: ``latents * std + mean`` (channel dim=1)."""

    mean = torch.as_tensor(tuple(mean_values), device=latents.device, dtype=latents.dtype)
    std = torch.as_tensor(tuple(std_values), device=latents.device, dtype=latents.dtype)
    if int(latents.shape[1]) != int(mean.shape[0]):
        raise ValueError(
            f"latent normalization channel mismatch: latents.shape[1]={int(latents.shape[1])} "
            f"mean_len={int(mean.shape[0])}"
        )
    view_shape = [1] * latents.ndim
    view_shape[1] = int(mean.shape[0])
    return latents * std.view(*view_shape) + mean.view(*view_shape)


def _resolve_canvas(short_edge: int, aspect_ratio: str) -> tuple[int, int]:
    """Return (height, width) pixels for a short-edge + aspect-ratio target."""

    if aspect_ratio not in _ASPECT_RATIOS:
        raise ValueError(f"unsupported aspect_ratio {aspect_ratio!r}; supported: {sorted(_ASPECT_RATIOS)}")
    aw, ah = _ASPECT_RATIOS[aspect_ratio]
    if aw >= ah:  # landscape or square: height is the short edge
        height = int(short_edge)
        width = int(round(short_edge * aw / ah))
    else:  # portrait: width is the short edge
        width = int(short_edge)
        height = int(round(short_edge * ah / aw))
    # Latent grid must be divisible by (spatial_compression * patch) on each axis.
    unit_h = _SPATIAL_COMPRESSION * _PATCH_H
    unit_w = _SPATIAL_COMPRESSION * _PATCH_W
    return _round_to_multiple(height, unit_h), _round_to_multiple(width, unit_w)


class NativeMiniMaxH3Pipeline(PipelineABC):
    """Public MiniMax H3 pipeline backed by the ported single-GPU components."""

    MODEL_ID = "minimax-h3"
    GENERATION_TYPE = "t2v"
    DEFAULT_SHORT_EDGE = 768
    DEFAULT_ASPECT_RATIO = "16:9"
    DEFAULT_DURATION_SECONDS = 5.0
    DEFAULT_NUM_INFERENCE_STEPS = 50
    DEFAULT_FPS = MINIMAX_H3_SUPPORTED_FPS
    DEFAULT_FLOW_SHIFT = 12.0
    DEFAULT_AUDIO_FLOW_SHIFT = 3.0

    def __init__(
        self,
        *,
        transformer: Any = None,
        video_vae: Any = None,
        audio_vae: Any = None,
        text_encoder: Any = None,
        tokenizer: Any = None,
        device: str = "cuda",
        model_id: str | None = None,
    ) -> None:
        super().__init__(model_id=model_id or self.MODEL_ID, device=device)
        self.transformer = transformer
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------ #
    # Loading.
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Mapping[str, Any] | None = None,
        required_components: Mapping[str, Any] | None = None,
        device: str = "cuda",
        model_id: str | None = None,
        **kwargs: Any,
    ) -> "NativeMiniMaxH3Pipeline":
        """Materialize the DiT / VAEs / Qwen3-VL encoder from a checkpoint root.

        ``model_path`` (or ``kwargs['checkpoint_path']``) should point at a
        MiniMax H3 checkpoint directory containing the ``transformer``,
        ``video_vae``, ``audio_vae`` and text-encoder subfolders. Component
        construction is deferred to the ported modules; this method only wires
        them together.
        """

        options: dict[str, Any] = dict(model_path) if isinstance(model_path, Mapping) else {}
        options.update(required_components or {})
        options.update(kwargs)
        source = options.get("checkpoint_path", options.get("model_path", model_path))
        resolved_model_id = str(options.get("model_id") or model_id or cls.MODEL_ID)

        transformer = options.get("transformer")
        video_vae = options.get("video_vae")
        audio_vae = options.get("audio_vae")
        text_encoder = options.get("text_encoder")
        tokenizer = options.get("tokenizer")

        if source is not None and any(
            component is None for component in (transformer, video_vae, audio_vae, text_encoder)
        ):
            transformer, video_vae, audio_vae, text_encoder, tokenizer = cls._load_components(
                Path(str(source)).expanduser(),
                device=device,
                transformer=transformer,
                video_vae=video_vae,
                audio_vae=audio_vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
            )

        return cls(
            transformer=transformer,
            video_vae=video_vae,
            audio_vae=audio_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            device=device,
            model_id=resolved_model_id,
        )

    @classmethod
    def _load_components(
        cls,
        root: Path,
        *,
        device: str,
        transformer: Any,
        video_vae: Any,
        audio_vae: Any,
        text_encoder: Any,
        tokenizer: Any,
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Load real weights for missing components from a checkpoint directory.

        Resolves the partition subfolder layout of ``MiniMaxAI/MiniMax-H3`` —
        ``<root>/{transformer, video_vae/source, audio_vae, text_encoder,
        processor}`` — falling back to a bare ``<root>`` when a component's
        files sit at the top level. Heavy deps (transformers Qwen3-VL,
        diffusers) load only when a component is actually built.
        """

        if transformer is None:
            transformer = cls._load_transformer(root, device=device)
        if video_vae is None:
            video_vae = cls._load_video_vae(root, device=device)
        if audio_vae is None:
            audio_vae = cls._load_audio_vae(root, device=device)
        if text_encoder is None:
            text_encoder = cls._load_text_encoder(root, device=device)
        if tokenizer is None:
            tokenizer = cls._load_tokenizer(root)
        return transformer, video_vae, audio_vae, text_encoder, tokenizer

    @staticmethod
    def _subdir(root: Path, *candidates: str) -> Path:
        """Return the first existing subfolder among candidates, else root."""

        for name in candidates:
            path = root / name
            if path.is_dir():
                return path
        return root

    @classmethod
    def _load_transformer(cls, root: Path, *, device: str) -> Any:
        import torch

        from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3 import (
            MiniMaxH3DiTArchConfig,
            MiniMaxH3DiTModel,
        )
        from worldfoundry.base_models.diffusion_model.models.networks.minimax_h3.loading import (
            load_minimax_h3_dit_weights,
        )

        tdir = cls._subdir(root, "transformer")
        arch = MiniMaxH3DiTArchConfig()
        model = MiniMaxH3DiTModel(arch)
        if any(tdir.glob("*.safetensors")):
            load_minimax_h3_dit_weights(model, tdir, arch, strict=True)
        return model.to(device).eval()

    @classmethod
    def _load_video_vae(cls, root: Path, *, device: str) -> Any:
        import json

        from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_loading import (
            load_minimax_h3_vae_weights,
        )
        from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_video import (
            MiniMaxH3VideoVAE,
        )
        from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_video.config import (
            MiniMaxH3VideoVAEArchConfig,
            MiniMaxH3VideoVAEConfig,
        )

        vdir = cls._subdir(root, "video_vae", "vae")
        cfg_json = json.loads((vdir / "config.json").read_text())
        arch = MiniMaxH3VideoVAEArchConfig(
            latent_channels=24,
            latents_mean=cfg_json.get("latents_mean"),
            latents_std=cfg_json.get("latents_std"),
        )
        model = MiniMaxH3VideoVAE(MiniMaxH3VideoVAEConfig(arch_config=arch))
        weight_dir = vdir / "source" if (vdir / "source").is_dir() else vdir
        if any(weight_dir.glob("*.safetensors")):
            load_minimax_h3_vae_weights(model, weight_dir, strict=False)
        return model.to(device).eval()

    @classmethod
    def _load_audio_vae(cls, root: Path, *, device: str) -> Any:
        import json

        from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_audio import (
            MiniMaxH3AudioVAE,
        )
        from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_audio.config import (
            MiniMaxH3AudioVAEArchConfig,
            MiniMaxH3AudioVAEConfig,
        )
        from worldfoundry.base_models.diffusion_model.models.autoencoders.minimax_h3_loading import (
            load_minimax_h3_vae_weights,
        )

        adir = cls._subdir(root, "audio_vae")
        cfg_json = json.loads((adir / "config.json").read_text())
        arch = MiniMaxH3AudioVAEArchConfig(
            latent_channels=32,
            latents_mean=cfg_json.get("latents_mean"),
            latents_std=cfg_json.get("latents_std"),
        )
        model = MiniMaxH3AudioVAE(MiniMaxH3AudioVAEConfig(arch_config=arch))
        if any(adir.glob("*.safetensors")):
            load_minimax_h3_vae_weights(model, adir, strict=True)
        return model.to(device).eval()

    @classmethod
    def _load_text_encoder(cls, root: Path, *, device: str) -> Any:
        import torch

        from worldfoundry.base_models.diffusion_model.models.encoders.minimax_h3_qwen3vl import (
            MiniMaxH3Qwen3VLEncoder,
        )

        tdir = cls._subdir(root, "text_encoder")
        return MiniMaxH3Qwen3VLEncoder.from_pretrained(str(tdir), torch_dtype=torch.bfloat16).to(device).eval()

    @classmethod
    def _load_tokenizer(cls, root: Path) -> Any:
        from transformers import AutoTokenizer

        pdir = cls._subdir(root, "processor", "tokenizer")
        return AutoTokenizer.from_pretrained(str(pdir))

    # ------------------------------------------------------------------ #
    # Geometry / plan resolution.
    # ------------------------------------------------------------------ #
    def _resolve_plan(
        self,
        *,
        task: str,
        short_edge: int,
        aspect_ratio: str,
        duration_seconds: float,
    ) -> dict[str, int]:
        if not (MINIMAX_H3_MIN_DURATION_SECONDS <= duration_seconds <= MINIMAX_H3_MAX_DURATION_SECONDS):
            raise ValueError(
                f"duration_seconds must be in [{MINIMAX_H3_MIN_DURATION_SECONDS}, "
                f"{MINIMAX_H3_MAX_DURATION_SECONDS}], got {duration_seconds}"
            )
        height, width = _resolve_canvas(short_edge, aspect_ratio)
        frame_count = minimax_h3_align_frame_count(int(round(duration_seconds * self.DEFAULT_FPS)))
        latent_t = minimax_h3_video_latent_t(frame_count)
        latent_h = height // _SPATIAL_COMPRESSION
        latent_w = width // _SPATIAL_COMPRESSION
        audio_t = minimax_h3_audio_latent_t(duration_seconds)
        return {
            "height": height,
            "width": width,
            "frame_count": frame_count,
            "latent_t": latent_t,
            "latent_h": latent_h,
            "latent_w": latent_w,
            "audio_t": audio_t,
        }

    # ------------------------------------------------------------------ #
    # Core generation (component-injectable, unit-testable).
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def generate(
        self,
        *,
        prompt_embeds: torch.Tensor,
        task: str = "t2va",
        short_edge: int = DEFAULT_SHORT_EDGE,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        duration_seconds: float = DEFAULT_DURATION_SECONDS,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        seed: int = 0,
        flow_shift: float | None = None,
        audio_flow_shift: float | None = None,
        keyframe_frame_indices: tuple[int, ...] | None = None,
        keyframe_cond_rows: torch.Tensor | None = None,
        ref_blocks: list[dict[str, Any]] | None = None,
        audio_ref_rows: torch.Tensor | None = None,
        return_latent_rows: bool = False,
    ) -> dict[str, Any]:
        """Run the coupled denoise loop and return decoded (or raw) modalities."""

        if self.transformer is None:
            raise RuntimeError("transformer is not initialized")
        device = torch.device(self.device)
        profile = minimax_h3_task_profile(task)
        flow_shift = float(flow_shift if flow_shift is not None else profile.default_flow_shift)
        audio_flow_shift = float(
            audio_flow_shift if audio_flow_shift is not None else profile.default_audio_flow_shift
        )
        plan = self._resolve_plan(
            task=task,
            short_edge=short_edge,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
        )
        text_len = int(prompt_embeds.shape[0])

        if task == "ref2va":
            if ref_blocks is None:
                raise ValueError("ref2va requires ref_blocks")
            packed = minimax_h3_packed_sequence_ref2va_blocks(
                text_len=text_len,
                latent_t=plan["latent_t"],
                latent_h=plan["latent_h"],
                latent_w=plan["latent_w"],
                audio_t=plan["audio_t"],
                ref_blocks=ref_blocks,
            )
        else:
            include_keyframe = task == "fl2va"
            packed = minimax_h3_packed_sequence(
                text_len=text_len,
                latent_t=plan["latent_t"],
                latent_h=plan["latent_h"],
                latent_w=plan["latent_w"],
                audio_t=plan["audio_t"],
                include_keyframe_cond=include_keyframe,
                keyframe_frame_indices=keyframe_frame_indices if include_keyframe else None,
                frame_count=plan["frame_count"] if include_keyframe else None,
            )

        branch = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=prompt_embeds,
            token_tags=packed["token_tags"],
            device=device,
        )

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        n_video_rows = int(packed["img_pos"].shape[0])
        n_audio_rows = int(packed["audio_pos"].shape[0])
        initial_video = torch.randn(n_video_rows, 96, generator=generator, dtype=torch.float32)
        # Audio uses an independently reseeded generator (same seed) — matches the
        # reference, which draws audio noise from a fresh generator.
        audio_generator = torch.Generator(device="cpu").manual_seed(int(seed))
        initial_audio = torch.randn(
            n_audio_rows, _AUDIO_LATENT_CHANNELS, generator=audio_generator, dtype=torch.float32
        )

        sigmas_video = minimax_h3_time_shift_sigmas(num_steps=num_inference_steps, shift_scale=flow_shift)
        sigmas_audio = minimax_h3_time_shift_sigmas(num_steps=num_inference_steps, shift_scale=audio_flow_shift)
        steps = min(len(sigmas_video), len(sigmas_audio))
        sigmas_video, sigmas_audio = sigmas_video[:steps], sigmas_audio[:steps]

        video_rows, audio_rows = minimax_h3_denoise_loop(
            model=self.transformer,
            positive=branch,
            initial_video_rows=initial_video,
            initial_audio_rows=initial_audio,
            keyframe_cond_rows=keyframe_cond_rows,
            audio_ref_rows=audio_ref_rows,
            sigmas_video=sigmas_video,
            sigmas_audio=sigmas_audio,
            device=device,
        )

        # Target rows are the suffix after any condition/reference rows.
        video_target = video_rows[branch.video_target_slice]
        audio_target = audio_rows[branch.audio_target_slice]
        result: dict[str, Any] = {"plan": plan, "task": task}
        if return_latent_rows:
            result["video_rows"] = video_target
            result["audio_rows"] = audio_target
            return result

        video_latent = minimax_h3_unpatchify_video_tokens(
            video_target,
            latent_shape=(plan["latent_t"], plan["latent_h"] // _PATCH_H, plan["latent_w"] // _PATCH_W, _VIDEO_LATENT_CHANNELS),
            patch_size=(1, _PATCH_H, _PATCH_W),
        )
        # unpack expects the TOTAL audio row count (audio_t * channels); it
        # divides by channels internally to recover the per-channel length.
        audio_latent = minimax_h3_unpack_audio_tokens(
            audio_target, audio_t=int(audio_target.shape[0]), audio_channel=_AUDIO_CHANNELS
        )
        result["video_latent"] = video_latent
        result["audio_latent"] = audio_latent
        if _is_module(self.video_vae):
            result["video"] = self._decode_video(video_latent.to(device))
        if _is_module(self.audio_vae):
            result["audio"] = self._decode_audio(audio_latent.to(device))
        return result

    def _decode_video(self, latent: torch.Tensor) -> torch.Tensor:
        """Reverse-normalize latents and decode to pixel frames [B,3,T,H,W]."""

        arch = self.video_vae.sglang_config.arch_config
        latent = _reverse_normalize_latents(
            latent, mean_values=arch.latents_mean, std_values=arch.latents_std
        )
        decode_base = getattr(self.video_vae, "decode_base", None)
        if callable(decode_base):
            return decode_base(latent)
        return self.video_vae.decode(latent)

    def _decode_audio(self, latent: torch.Tensor) -> torch.Tensor:
        """Reverse-normalize latents and decode to a stereo waveform."""

        arch = self.audio_vae.sglang_config.arch_config if hasattr(self.audio_vae, "sglang_config") else None
        if arch is not None and arch.latents_mean is not None:
            latent = _reverse_normalize_latents(
                latent, mean_values=arch.latents_mean, std_values=arch.latents_std
            )
        return self.audio_vae.decode(latent)

    # ------------------------------------------------------------------ #
    # Public call: text + optional conditions -> mp4 with audio track.
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        prompt: str,
        *,
        task: str = "t2va",
        short_edge: int = DEFAULT_SHORT_EDGE,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        duration_seconds: float = DEFAULT_DURATION_SECONDS,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        seed: int = 0,
        flow_shift: float | None = None,
        audio_flow_shift: float | None = None,
        output_path: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self.text_encoder is None or self.tokenizer is None:
            raise RuntimeError(
                "text_encoder/tokenizer are not initialized; call from_pretrained with a checkpoint"
            )
        prompt_embeds = self._encode_prompt(prompt)
        result = self.generate(
            prompt_embeds=prompt_embeds,
            task=task,
            short_edge=short_edge,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            num_inference_steps=num_inference_steps,
            seed=seed,
            flow_shift=flow_shift,
            audio_flow_shift=audio_flow_shift,
            **kwargs,
        )
        if output_path is not None and "video" in result:
            result["output_path"] = self._write_mp4_with_audio(
                video=result["video"],
                audio=result.get("audio"),
                output_path=Path(output_path),
                fps=self.DEFAULT_FPS,
            )
        return result

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens["input_ids"].to(self.device)
        return self.text_encoder.encode_ids(input_ids)

    @staticmethod
    def _write_mp4_with_audio(
        *,
        video: torch.Tensor,
        audio: torch.Tensor | None,
        output_path: Path,
        fps: int,
    ) -> str:
        """Save decoded video frames and mux the 32 kHz stereo track via ffmpeg."""

        from worldfoundry.core.io.video import save_image_or_video_tensor

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if audio is None:
            save_image_or_video_tensor(video, str(output_path), fps=fps)
            return str(output_path)

        ffmpeg = _resolve_ffmpeg()
        with tempfile.TemporaryDirectory() as tmp:
            silent = Path(tmp) / "video.mp4"
            save_image_or_video_tensor(video, str(silent), fps=fps)
            wav = Path(tmp) / "audio.wav"
            _write_wav(audio, wav, sample_rate=_AUDIO_SAMPLE_RATE)
            if ffmpeg is None:
                # No ffmpeg: fall back to the silent video and keep the wav beside it.
                save_image_or_video_tensor(video, str(output_path), fps=fps)
                shutil.copy(str(wav), str(output_path.with_suffix(".wav")))
                return str(output_path)
            subprocess.run(
                [
                    ffmpeg, "-y", "-i", str(silent), "-i", str(wav),
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(output_path),
                ],
                check=True,
                capture_output=True,
            )
        return str(output_path)


def _resolve_ffmpeg() -> str | None:
    """Locate an ffmpeg binary robustly.

    ``shutil.which`` misses ffmpeg when the caller's ``PATH`` omits the conda
    ``bin`` (e.g. a subprocess launched without the env activated), so also
    probe the running interpreter's own ``bin`` dir and the ``imageio-ffmpeg``
    bundled binary before giving up.
    """

    found = shutil.which("ffmpeg")
    if found:
        return found
    candidate = Path(sys.executable).parent / "ffmpeg"
    if candidate.is_file():
        return str(candidate)
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _write_wav(audio: torch.Tensor, path: Path, *, sample_rate: int) -> None:
    """Write a ``[channels, samples]`` (or ``[1, C, L]``) waveform to a WAV file."""

    import wave

    waveform = audio.detach().to("cpu", dtype=torch.float32)
    # The audio VAE emits stereo as ``[channels, 1, samples]`` (channels-as-batch
    # with a singleton feature axis). Drop only singleton dims, never a real
    # channel axis, then normalize to ``[channels, samples]``.
    while waveform.ndim > 2:
        squeezed = False
        for dim in range(waveform.ndim):
            if waveform.shape[dim] == 1:
                waveform = waveform.squeeze(dim)
                squeezed = True
                break
        if not squeezed:
            # No singleton left but still >2D: flatten trailing dims into samples.
            waveform = waveform.reshape(waveform.shape[0], -1)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    channels, _ = waveform.shape
    clipped = torch.clamp(waveform, -1.0, 1.0)
    pcm = (clipped * 32767.0).to(torch.int16).transpose(0, 1).contiguous().numpy()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(int(channels))
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


__all__ = ["NativeMiniMaxH3Pipeline"]

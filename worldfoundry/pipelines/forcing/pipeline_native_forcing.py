"""Native Self-Forcing text-to-video pipeline."""

from __future__ import annotations

from collections.abc import Mapping

from ..native_diffusion import NativeVisualDiffusionPipeline


class NativeSelfForcingPipeline(NativeVisualDiffusionPipeline):
    """Four-step causal Self-Forcing (Wan2.1 T2V 1.3B DMD) on the native window runner."""

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


__all__ = ["NativeSelfForcingPipeline"]

"""Product adapter for FastVideo's CausalWan2.2 rollout."""

from __future__ import annotations

from ..native_diffusion import NativeVisualDiffusionPipeline


class FastVideoCausalWanPipeline(NativeVisualDiffusionPipeline):
    """Eight-step, first-frame-conditioned CausalWan2.2 image-to-video pipeline."""

    MODEL_ID = "fastvideo-causal-wan2.2-i2v-14b"
    OWNER = "FastVideo CausalWan2.2"
    CHECKPOINT_ROLES = ("high-dit", "low-dit", "text-encoder", "tokenizer", "vae")
    GENERATION_TYPE = "i2v"
    ACCEPTS_IMAGES = True
    REQUIRES_IMAGES = True
    DEFAULT_HEIGHT = 480
    DEFAULT_WIDTH = 832
    DEFAULT_NUM_FRAMES = 81
    DEFAULT_NUM_INFERENCE_STEPS = 8
    DEFAULT_GUIDANCE_SCALE = 1.0
    DEFAULT_FPS = 16
    DEFAULT_NEGATIVE_PROMPT = ""
    DEFAULT_SCHEDULER_OPTIONS = {"shift": 5.0}


__all__ = ["FastVideoCausalWanPipeline"]

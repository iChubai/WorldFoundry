"""Instance-local memory, control, and research extensions."""

from .approximate_attention_step import ApproximateAttentionStepExtension
from .base import DiffusionExtension, DiffusionRunContext
from .frozen_context import FrozenContextSuffixExtension
from .frozen_mask import FrozenLatentMaskExtension

__all__ = [
    "ApproximateAttentionStepExtension",
    "DiffusionExtension",
    "DiffusionRunContext",
    "FrozenContextSuffixExtension",
    "FrozenLatentMaskExtension",
]

"""Text-embedding configuration types.

Video and world-model runtimes stack encoder hidden states in more
than one way. :class:`EmbeddingConcatStrategy` names the supported
layouts: full concatenation, mean pooling, or pooling every N layers
then concatenating.

This module holds the enum only; encoding and pooling live with the
text encoder that consumes the strategy.
"""

from enum import Enum

# ──────────────────────────────────────────────────────────────────────────
# Layout names only — encoding / pooling live on the consuming text encoder
# ──────────────────────────────────────────────────────────────────────────


class EmbeddingConcatStrategy(str, Enum):
    """How stacked text-encoder hidden states are reduced to one embedding."""

    FULL_CONCAT = "full_concat"
    MEAN_POOLING = "mean_pooling"
    POOL_EVERY_N_LAYERS_AND_CONCAT = "pool_every_n_layers_and_concat"

    def __str__(self) -> str:
        """Return the on-the-wire value so YAML/logs use ``full_concat``, not the member name."""
        return self.value


__all__ = ["EmbeddingConcatStrategy"]

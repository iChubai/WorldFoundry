import torch
from einops import rearrange
from torch import nn

# ---------------------------------------------------------------------------
# Normalization functions
# ---------------------------------------------------------------------------


def _norm_and_concat_padded_batch(
    encoded_text: torch.Tensor,
    sequence_lengths: torch.Tensor,
    padding_side: str = "right",
) -> torch.Tensor:
    """Normalize and flatten multi-layer hidden states, respecting padding.
    Performs per-batch, per-layer normalization using masked mean and range,
    then concatenates across the layer dimension.
    Args:
        encoded_text: Hidden states of shape [batch, seq_len, hidden_dim, num_layers].
        sequence_lengths: Number of valid (non-padded) tokens per batch item.
        padding_side: Whether padding is on "left" or "right".
    Returns:
        Normalized tensor of shape [batch, seq_len, hidden_dim * num_layers],
        with padded positions zeroed out.
    """
    b, t, d, l = encoded_text.shape  # noqa: E741
    device = encoded_text.device

    token_indices = torch.arange(t, device=device)[None, :]

    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]
    elif padding_side == "left":
        start_indices = t - sequence_lengths[:, None]
        mask = token_indices >= start_indices
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")

    mask = rearrange(mask, "b t -> b t 1 1")

    eps = 1e-6

    masked = encoded_text.masked_fill(~mask, 0.0)
    denom = (sequence_lengths * d).view(b, 1, 1, 1)
    mean = masked.sum(dim=(1, 2), keepdim=True) / (denom + eps)

    x_min = encoded_text.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
    x_max = encoded_text.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)
    range_ = x_max - x_min

    normed = 8 * (encoded_text - mean) / (range_ + eps)
    normed = normed.reshape(b, t, -1)

    mask_flattened = rearrange(mask, "b t 1 1 -> b t 1").expand(-1, -1, d * l)
    normed = normed.masked_fill(~mask_flattened, 0.0)

    return normed


# ---------------------------------------------------------------------------
# Feature extractor variants
# ---------------------------------------------------------------------------


class FeatureExtractorV1(nn.Module):
    """19B: per-segment norm -> aggregate_embed -> 3840"""

    def __init__(self, aggregate_embed: nn.Module, is_av: bool = False):
        super().__init__()
        self.aggregate_embed = aggregate_embed
        self.is_av = is_av

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, padding_side: str = "left"
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        encoded = torch.stack(hidden_states, dim=-1) if isinstance(hidden_states, (list, tuple)) else hidden_states
        dtype = encoded.dtype
        sequence_lengths = attention_mask.sum(dim=-1)
        normed = _norm_and_concat_padded_batch(encoded, sequence_lengths, padding_side)
        features = self.aggregate_embed(normed.to(dtype))
        if self.is_av:
            return features, features
        return features, None

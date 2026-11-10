"""Linear primaries: gamut basis, conversion, and EXR chromaticity tags.
Shared by the VAE HDR working-space path (AP1 ↔ Rec.709) and the HLG encode
path (Rec.709 / ACEScg → Rec.2020).
"""

from __future__ import annotations

import enum

import torch
from torch import Tensor

# ACEScg (AP1, D60-adapted) linear → linear sRGB / Rec.709 (D65) primaries.
_ACESCG_TO_SRGB = torch.tensor(
    [
        [1.70505000, -0.62179000, -0.08326000],
        [-0.13026000, 1.14080000, -0.01055000],
        [-0.02400000, -0.12897000, 1.15297000],
    ],
    dtype=torch.float32,
)

# Exact inverse of ``_ACESCG_TO_SRGB`` (float64) for a clean round-trip.
_SRGB_TO_ACESCG = torch.linalg.inv(_ACESCG_TO_SRGB.double()).to(torch.float32)

# Rec.709 (D65) → Rec.2020 (D65) (ITU-R BT.2087).
_REC709_TO_2020 = torch.tensor(
    [
        [0.62740389, 0.32928304, 0.04331307],
        [0.06909729, 0.91954040, 0.01136232],
        [0.01639144, 0.08801331, 0.89559525],
    ],
    dtype=torch.float32,
)

# ACEScg (AP1, D60) → Rec.2020 (D65), Bradford CAT.
_ACESCG_TO_2020 = torch.tensor(
    [
        [1.02582475, -0.02005319, -0.00577156],
        [-0.00223437, 1.00458650, -0.00235213],
        [-0.00501335, -0.02529007, 1.03030342],
    ],
    dtype=torch.float32,
)


def _apply_primaries_matrix(video: Tensor, mat: Tensor) -> Tensor:
    """Apply a 3x3 primaries matrix. Channel axis: dim 1 for 5D, else ``..., C, H, W``."""
    mat = mat.to(device=video.device, dtype=video.dtype)
    if video.ndim == 5:
        return torch.einsum("bcfhw,dc->bdfhw", video, mat)
    return torch.einsum("...chw,dc->...dhw", video, mat)


class Primaries(enum.Enum):
    """Authoring / working-space colour primaries (linear light).
    Rec.2020 is a conversion *target* (HLG master), not an EXR authoring case —
    use :meth:`to_rec2020`.
    """

    REC709 = "rec709"
    ACESCG = "acescg"

    @property
    def exr_chromaticities(self) -> tuple[float, ...]:
        """EXR ``chromaticities`` header (R/G/B/W x,y)."""
        match self:
            case Primaries.REC709:
                return (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290)
            case Primaries.ACESCG:
                return (0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767)

    @property
    def matrix_to_rec2020(self) -> Tensor:
        """3x3 linear map from this basis to Rec.2020 (for HLG)."""
        match self:
            case Primaries.REC709:
                return _REC709_TO_2020
            case Primaries.ACESCG:
                return _ACESCG_TO_2020

    def to(self, target: Primaries, video: Tensor) -> Tensor:
        """Convert linear light from this basis into ``target``. Identity when same."""
        if self is target:
            return video
        match (self, target):
            case (Primaries.REC709, Primaries.ACESCG):
                return _apply_primaries_matrix(video, _SRGB_TO_ACESCG)
            case (Primaries.ACESCG, Primaries.REC709):
                return _apply_primaries_matrix(video, _ACESCG_TO_SRGB)
        raise ValueError(f"No primaries conversion from {self!r} to {target!r}.")

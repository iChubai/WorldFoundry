"""Cross-metric helpers used by metric packages and benchmark runtimes.

- ``aesthetic`` — LAION aesthetic predictor shared by benchmark quality evaluators
- ``bbox`` — IoU helpers for scene-level ``fid`` crops
- ``clip_embed`` — CLIP encode/cosine helpers for benchmark evaluators
- ``images`` — RGB input normalization for image-based helpers
- ``imports`` — import-path setup for bundled metric implementations
- ``lazy`` — lightweight public exports for metric packages
- ``pyiqa`` — pyiqa-backed image quality helpers for benchmark evaluators
- ``torch_fidelity`` — lazy loader for vendored ``vendor/torch_fidelity`` (``fid``, ``kid``, ``inception_score``)

Metric-specific helpers live in the metric package; only code shared by several callers belongs here.
"""

from __future__ import annotations

__all__: list[str] = []

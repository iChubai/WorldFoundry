"""DurationHead: predicts shot duration from Connector token outputs."""

from .duration_head import AttentionPooler, DurationHead
from .model_configurator import DURATION_HEAD_KEY_OPS, DurationHeadConfigurator

__all__ = [
    "DURATION_HEAD_KEY_OPS",
    "AttentionPooler",
    "DurationHead",
    "DurationHeadConfigurator",
]

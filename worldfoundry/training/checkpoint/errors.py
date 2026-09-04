"""Functional errors raised by exact-resume checkpointing."""

from __future__ import annotations

import copy
from collections.abc import Mapping


class TrainingCheckpointError(RuntimeError):
    """Base error for checkpoint validation or I/O failures."""


class IncompleteTrainingCheckpointError(TrainingCheckpointError):
    """Raised when a checkpoint has no valid atomic commit."""


class TrainingCheckpointCompatibilityError(TrainingCheckpointError):
    """Raised when exact-resume identities or topology differ."""

    def __init__(
        self,
        message: str,
        *,
        section_differences: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.section_differences = copy.deepcopy(dict(section_differences or {}))


__all__ = [
    "IncompleteTrainingCheckpointError",
    "TrainingCheckpointCompatibilityError",
    "TrainingCheckpointError",
]

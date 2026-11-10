# SPDX-FileCopyrightText: Copyright (c) 2026 WorldFoundry contributors
# SPDX-License-Identifier: Apache-2.0
"""The checked-in model catalog uses only registered status values or aliases.

An unregistered value silently degrades to a fallback (an open-source model
reads as ``unknown``, a validated run as ``pending``) and makes every catalog
load print a warning.
"""

from __future__ import annotations

import pytest

from worldfoundry.evals.models.catalog import schema
from worldfoundry.evals.models.catalog.zoo_registry import ModelZooRegistry, default_model_zoo_dir

pytestmark = pytest.mark.unit


def test_checked_in_catalog_statuses_are_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    unknown: set[str] = set()

    def record(kind: str, raw_value: str, fallback: str) -> None:
        unknown.add(f"{kind}={raw_value!r} (falls back to {fallback!r})")

    monkeypatch.setattr(schema, "_warn_unknown_status_once", record)
    registry = ModelZooRegistry.from_directory(default_model_zoo_dir())

    assert registry.list(), "the model catalog loaded no entries"
    assert not unknown, f"register these catalog status values in schema.py alias sets: {sorted(unknown)}"

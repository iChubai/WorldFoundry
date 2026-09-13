"""Workspace-facing dispatch for in-tree benchmark runners."""

from worldfoundry.evaluation.tasks.catalog.dispatch import CLI_RUNNERS, OfficialRunnerSpec as WorkspaceRunnerSpec

from .dispatch import *

__all__ = [name for name in globals() if not name.startswith("_")]

"""Cosmos3 Nano/Super adapter for WorldArena.

Cosmos3 is ported in the sibling **WorldFoundry** project as
``worldfoundry.pipelines.cosmos.pipeline_cosmos3.Cosmos3Pipeline``.
WorldArena drives it out of process so the 64B Super (or 16B Nano) transformer
loads once per benchmark shard. Super keeps layers resident across the worker's
visible GPUs (``device_map=balanced``) instead of reloading or CPU-offloading.
The subprocess interpreter must import both ``worldarena`` and ``worldfoundry``;
point ``python_bin`` at the dedicated Cosmos3 env and ``repo_root`` at the
WorldFoundry checkout.
"""

from __future__ import annotations

from worldarena.models.adapters.external_batch import ExternalBatchAdapter


class Cosmos3Adapter(ExternalBatchAdapter):
    """Generate videos with WorldFoundry's native Cosmos3 pipeline."""

    runner_module = "worldarena.models.adapters.cosmos3_batch_runner"
    label = "Cosmos3"

    def batch_checkpoint_load_policy(self) -> str:
        return "load_once"

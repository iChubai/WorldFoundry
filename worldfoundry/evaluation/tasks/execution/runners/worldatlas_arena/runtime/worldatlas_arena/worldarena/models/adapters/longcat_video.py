from __future__ import annotations

from worldarena.models.adapters.external_batch import ExternalBatchAdapter


class LongCatVideoAdapter(ExternalBatchAdapter):
    runner_module = "worldarena.models.adapters.longcat_video_batch_runner"
    label = "LongCat-Video"

    def batch_checkpoint_load_policy(self) -> str:
        """The LongCat process reuses its pipeline for every row in the shard."""
        return "load_once"

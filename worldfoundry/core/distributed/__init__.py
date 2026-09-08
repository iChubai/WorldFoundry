"""Distributed primitives: process groups, splits, and collectives for TP / CP / SP / FSDP / PP.

Each parallel axis has a distinct job so one DeviceMesh does not mix semantics:

- **TP (Tensor Parallel)**: shards Linear/Attention hidden dims across ranks and
  restores them with in-group all-reduce or all-gather. See ``get_tp_*`` in
  ``model_parallel_groups``.
- **CP (Context Parallel)**: shards Q/K/V along the sequence dim (Ulysses / ring /
  xDiT). ``context_parallel`` owns split/cat/broadcast. The Ulysses scheduler in
  the attention package must be paired with these splits or RoPE positions
  misalign.
- **SP (Sequence Parallel)**: the ``sequence_parallel/`` subpackage (process
  state, NCCL wrappers, all-to-all). Often paired with CP: all-to-all head dim
  into sequence shards, then run attention locally.
- **FSDP / FSDP2**: ``fsdp_runtime`` / ``fsdp2_sharding`` / ``block_fsdp``.
  Parameter sharding plus forward all-gather. FSDP2 uses DTensor/DeviceMesh.
  Inference can disable gradient buckets.
- **PP (Pipeline Parallel)**: ``pipeline_parallel.PPScheduler`` schedules
  micro-batches by stage; ``get_pp_*`` finds upstream/downstream ranks.
- **DP**: data-parallel groups (including a gloo backup) for metrics, EMA, and
  broadcasts. They do not shard a single sample's sequence.

``initialize_model_parallel`` builds the TP/CP/PP/DP groups together. Collectives
are split by payload type across ``generic_collectives``, ``tensor_collectives``,
``object_collectives``, and ``device_mesh_collectives`` so CPU pickle and GPU
tensors do not share one path.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────
# Public re-exports — keep this facade import-light; no eager NCCL init
# ──────────────────────────────────────────────────────────────────────────

from .context_parallel import (
    broadcast,
    broadcast_split_tensor,
    cat_outputs_cp,
    cat_outputs_cp_object_list,
    cat_outputs_cp_with_grad,
    find_split,
    robust_broadcast,
    split_inputs_cp,
    split_inputs_cp_object_list,
)
from .device_mesh_collectives import (
    DTensorFastEmaModelUpdater,
    FastEmaModelUpdater,
    broadcast_dtensor_model_states,
    get_local_tensor_if_DTensor,
    get_local_tensor_if_dtensor,
)
from .generic_collectives import (
    get_rank as get_global_rank,
)
from .generic_collectives import (
    get_world_size,
)
from .generic_collectives import (
    is_dist_initialized as is_distributed_initialized,
)
from .inference_runtime import dist_init, is_last_rank, is_last_tp_cp_rank
from .inference_runtime import get_device as get_distributed_device
from .logging import print_per_rank, print_rank_0
from .model_parallel_groups import (
    destroy_model_parallel,
    get_cp_group,
    get_cp_rank,
    get_cp_world_size,
    get_dp_group,
    get_dp_group_gloo,
    get_dp_rank,
    get_dp_world_size,
    get_model_parallel_group,
    get_pipeline_model_parallel_first_rank,
    get_pipeline_model_parallel_last_rank,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
    get_pp_group,
    get_pp_rank,
    get_pp_world_size,
    get_tensor_model_parallel_last_rank,
    get_tensor_model_parallel_ranks,
    get_tensor_model_parallel_src_rank,
    get_tp_group,
    get_tp_rank,
    get_tp_world_size,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from .pipeline_parallel import PPScheduler, init_pp_scheduler, pp_scheduler
from .rank_orchestration import (
    DistributedOpSpec,
    PayloadBus,
    RankCoordinator,
    SignalBus,
    distributed_op,
)

# ──────────────────────────────────────────────────────────────────────────
# __all__ — names callers may import from worldfoundry.core.distributed
# ──────────────────────────────────────────────────────────────────────────

__all__ = [
    "DistributedOpSpec",
    "DTensorFastEmaModelUpdater",
    "FastEmaModelUpdater",
    "PPScheduler",
    "PayloadBus",
    "RankCoordinator",
    "SignalBus",
    "broadcast",
    "broadcast_dtensor_model_states",
    "broadcast_split_tensor",
    "cat_outputs_cp",
    "cat_outputs_cp_object_list",
    "cat_outputs_cp_with_grad",
    "destroy_model_parallel",
    "dist_init",
    "distributed_op",
    "find_split",
    "get_cp_group",
    "get_cp_rank",
    "get_cp_world_size",
    "get_distributed_device",
    "get_dp_group",
    "get_dp_group_gloo",
    "get_dp_rank",
    "get_dp_world_size",
    "get_global_rank",
    "get_local_tensor_if_dtensor",
    "get_local_tensor_if_DTensor",
    "get_model_parallel_group",
    "get_pipeline_model_parallel_first_rank",
    "get_pipeline_model_parallel_last_rank",
    "get_pipeline_model_parallel_next_rank",
    "get_pipeline_model_parallel_prev_rank",
    "get_pp_group",
    "get_pp_rank",
    "get_pp_world_size",
    "get_tensor_model_parallel_last_rank",
    "get_tensor_model_parallel_ranks",
    "get_tensor_model_parallel_src_rank",
    "get_tp_group",
    "get_tp_rank",
    "get_tp_world_size",
    "get_world_size",
    "init_pp_scheduler",
    "initialize_model_parallel",
    "is_distributed_initialized",
    "is_last_rank",
    "is_last_tp_cp_rank",
    "model_parallel_is_initialized",
    "pp_scheduler",
    "print_per_rank",
    "print_rank_0",
    "robust_broadcast",
    "split_inputs_cp",
    "split_inputs_cp_object_list",
]

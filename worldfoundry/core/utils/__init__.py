"""Lazy facade for WorldFoundry core utilities.

Responsibility
    Re-export tensors, images, inference runtime, and the CUDA Graph runner
    without importing torch / cv2 / decord at ``import worldfoundry.core.utils``.

Boundaries
    This package is a name table, not an algorithm. Graph *semantics* live in
    :mod:`inference_graph`; this module only maps names. It is not a place to
    add new ops — put those in the owning submodule so lazy resolve stays thin.

Public surface
    ``_SUBMODULES`` (import the submodule itself) and ``_EXPORT_MODULES``
    (import one symbol). :func:`__getattr__` materializes and caches; unknown
    names raise :exc:`AttributeError`.

Split by use:

- ``inference_graph``: the **CUDA Graph runner**. Read that module for the
  capture/replay contract. It records a CPU-decision-free forward into a Graph;
  any change in shape, address, or control flow requires recapture. Pair with
  ``acceleration.cuda_graph_dispatch``, which selects a Graph by AR index while
  that module performs the actual capture/replay.
- ``cuda_graph``: thinner Graph helpers (events / streams) for the runner or
  external callers.
- ``inference_runtime``: adaptive batching, OOM detection, and
  ``max_new_tokens`` resolution.
- ``torch_utils`` / ``array_tensor_utils`` / ``batch_ops``: device, seed, DDP
  unwrap, and numpy/tensor-polymorphic ops.
- ``image_utils`` / ``video_utils`` / ``feature_extraction`` / ``text_parsing``:
  media and evaluation helpers.
- ``functional_utils`` / ``misc_utils`` / ``lazy_module``: decorators, registries,
  and on-demand imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# ──────────────────────────────────────────────────────────────────────────
# Lazy name tables — resolved on first attribute access, then cached
# ──────────────────────────────────────────────────────────────────────────

_SUBMODULES = {
    "array_tensor_utils": "worldfoundry.core.utils.array_tensor_utils",
    "batch_ops": "worldfoundry.core.utils.batch_ops",
    "functional_utils": "worldfoundry.core.utils.functional_utils",
    "image_utils": "worldfoundry.core.utils.image_utils",
    "inference_runtime": "worldfoundry.core.utils.inference_runtime",
    "lazy_module": "worldfoundry.core.utils.lazy_module",
    "misc_utils": "worldfoundry.core.utils.misc_utils",
    "shape_utils": "worldfoundry.core.utils.shape_utils",
    "torch_utils": "worldfoundry.core.utils.torch_utils",
    "tree_utils": "worldfoundry.core.utils.tree_utils",
    "validator": "worldfoundry.core.utils.validator",
    "video_utils": "worldfoundry.core.utils.video_utils",
}

_EXPORT_MODULES = {
    "batched_image_features": "worldfoundry.core.utils.feature_extraction",
    "AverageMeter": "worldfoundry.core.utils.torch_utils",
    "Bool": "worldfoundry.core.utils.validator",
    "ClassRegistry": "worldfoundry.core.utils.functional_utils",
    "Cv2Display": "worldfoundry.core.utils.image_utils",
    "DDPMethodWrapper": "worldfoundry.core.utils.torch_utils",
    "Every": "worldfoundry.core.utils.misc_utils",
    "Float": "worldfoundry.core.utils.validator",
    "Int": "worldfoundry.core.utils.validator",
    "JsonDict": "worldfoundry.core.utils.validator",
    "LazyModule": "worldfoundry.core.utils.lazy_module",
    "NoopContext": "worldfoundry.core.utils.functional_utils",
    "NoopObject": "worldfoundry.core.utils.functional_utils",
    "Once": "worldfoundry.core.utils.misc_utils",
    "OneOf": "worldfoundry.core.utils.validator",
    "PeriodicEvent": "worldfoundry.core.utils.misc_utils",
    "RunningMeanStd": "worldfoundry.core.utils.torch_utils",
    "String": "worldfoundry.core.utils.validator",
    "Validator": "worldfoundry.core.utils.validator",
    "accepts_kwargs": "worldfoundry.core.utils.functional_utils",
    "accepts_varargs": "worldfoundry.core.utils.functional_utils",
    "adaptive_batched_inference": "worldfoundry.core.utils.inference_runtime",
    "add_batch_dim": "worldfoundry.core.utils.array_tensor_utils",
    "any_assign": "worldfoundry.core.utils.array_tensor_utils",
    "any_chunk": "worldfoundry.core.utils.array_tensor_utils",
    "any_concat": "worldfoundry.core.utils.array_tensor_utils",
    "any_describe": "worldfoundry.core.utils.array_tensor_utils",
    "any_describe_str": "worldfoundry.core.utils.array_tensor_utils",
    "any_fill_": "worldfoundry.core.utils.array_tensor_utils",
    "any_get_shape": "worldfoundry.core.utils.array_tensor_utils",
    "any_mean": "worldfoundry.core.utils.array_tensor_utils",
    "any_ones_like": "worldfoundry.core.utils.array_tensor_utils",
    "any_slice": "worldfoundry.core.utils.array_tensor_utils",
    "any_stack": "worldfoundry.core.utils.array_tensor_utils",
    "any_to_primitive": "worldfoundry.core.utils.array_tensor_utils",
    "any_transpose_first_two_axes": "worldfoundry.core.utils.array_tensor_utils",
    "any_variance": "worldfoundry.core.utils.array_tensor_utils",
    "any_zero_": "worldfoundry.core.utils.array_tensor_utils",
    "any_zeros_like": "worldfoundry.core.utils.array_tensor_utils",
    "argmax": "worldfoundry.core.utils.misc_utils",
    "assert_has_keys": "worldfoundry.core.utils.functional_utils",
    "assert_implements_method": "worldfoundry.core.utils.functional_utils",
    "as_list": "worldfoundry.core.utils.functional_utils",
    "basic_image_tensor_preprocess": "worldfoundry.core.utils.image_utils",
    "broadcast_structures": "worldfoundry.core.utils.tree_utils",
    "batch_add": "worldfoundry.core.utils.batch_ops",
    "batch_div": "worldfoundry.core.utils.batch_ops",
    "batch_mul": "worldfoundry.core.utils.batch_ops",
    "batch_sub": "worldfoundry.core.utils.batch_ops",
    "call_once": "worldfoundry.core.utils.functional_utils",
    "check_shape": "worldfoundry.core.utils.shape_utils",
    "chunk_seq": "worldfoundry.core.utils.array_tensor_utils",
    "classify_accuracy": "worldfoundry.core.utils.torch_utils",
    "clip_grad_norm": "worldfoundry.core.utils.torch_utils",
    "clip_grad_value": "worldfoundry.core.utils.torch_utils",
    "clone_model": "worldfoundry.core.utils.torch_utils",
    "compose_horizontal_views": "worldfoundry.core.utils.image_utils",
    "contains_rnn": "worldfoundry.core.utils.torch_utils",
    "copy_non_leaf": "worldfoundry.core.utils.tree_utils",
    "count_parameters": "worldfoundry.core.utils.torch_utils",
    "deprecated": "worldfoundry.core.utils.functional_utils",
    "divide": "worldfoundry.core.utils.misc_utils",
    "dump_torch": "worldfoundry.core.utils.torch_utils",
    "enable_dict_arg": "worldfoundry.core.utils.functional_utils",
    "enable_kwargs": "worldfoundry.core.utils.functional_utils",
    "enable_list_arg": "worldfoundry.core.utils.functional_utils",
    "enable_varargs": "worldfoundry.core.utils.functional_utils",
    "env_is_true": "worldfoundry.core.utils.misc_utils",
    "eval_mode": "worldfoundry.core.utils.torch_utils",
    "extract_yes_no_answer": "worldfoundry.core.utils.text_parsing",
    "fast_map_structure": "worldfoundry.core.utils.tree_utils",
    "filter_patterns": "worldfoundry.core.utils.misc_utils",
    "fix_random_seeds": "worldfoundry.core.utils.torch_utils",
    "freeze_params": "worldfoundry.core.utils.torch_utils",
    "has_batchnorms": "worldfoundry.core.utils.torch_utils",
    "func_has_arg": "worldfoundry.core.utils.functional_utils",
    "func_parameters": "worldfoundry.core.utils.functional_utils",
    "get_all_frames": "worldfoundry.core.utils.video_utils",
    "get_batch_size": "worldfoundry.core.utils.array_tensor_utils",
    "get_device": "worldfoundry.core.utils.torch_utils",
    "get_frames_by_indices": "worldfoundry.core.utils.video_utils",
    "get_frames_by_timestamps": "worldfoundry.core.utils.video_utils",
    "get_module_device": "worldfoundry.core.utils.torch_utils",
    "get_seed": "worldfoundry.core.utils.torch_utils",
    "getattr_nested": "worldfoundry.core.utils.misc_utils",
    "getitem_nested": "worldfoundry.core.utils.misc_utils",
    "global_n_times": "worldfoundry.core.utils.misc_utils",
    "global_once": "worldfoundry.core.utils.misc_utils",
    "has_keys": "worldfoundry.core.utils.functional_utils",
    "implements_method": "worldfoundry.core.utils.functional_utils",
    "implements_state_dict": "worldfoundry.core.utils.torch_utils",
    "imread": "worldfoundry.core.utils.image_utils",
    "imsave": "worldfoundry.core.utils.image_utils",
    "imshow": "worldfoundry.core.utils.image_utils",
    "is_array_tensor": "worldfoundry.core.utils.array_tensor_utils",
    "is_accelerator_out_of_memory": "worldfoundry.core.utils.inference_runtime",
    "mean_pairwise_cosine_distance": "worldfoundry.core.utils.feature_extraction",
    "is_mapping": "worldfoundry.core.utils.functional_utils",
    "is_numpy": "worldfoundry.core.utils.array_tensor_utils",
    "is_sequence": "worldfoundry.core.utils.functional_utils",
    "is_signature_compatible": "worldfoundry.core.utils.functional_utils",
    "is_tensor": "worldfoundry.core.utils.array_tensor_utils",
    "load_state_dict": "worldfoundry.core.utils.torch_utils",
    "load_pil_image": "worldfoundry.core.utils.image_utils",
    "materialize_image_input": "worldfoundry.core.utils.image_utils",
    "load_torch": "worldfoundry.core.utils.torch_utils",
    "make_list": "worldfoundry.core.utils.functional_utils",
    "make_recursive_func": "worldfoundry.core.utils.functional_utils",
    "make_tuple": "worldfoundry.core.utils.functional_utils",
    "match_patterns": "worldfoundry.core.utils.misc_utils",
    "maybe_transfer_module": "worldfoundry.core.utils.torch_utils",
    "mean_flat": "worldfoundry.core.utils.torch_utils",
    "merge_kwargs": "worldfoundry.core.utils.functional_utils",
    "meta_decorator": "worldfoundry.core.utils.functional_utils",
    "method_decorator": "worldfoundry.core.utils.functional_utils",
    "multi_one_hot": "worldfoundry.core.utils.torch_utils",
    "pack_kwargs": "worldfoundry.core.utils.functional_utils",
    "pack_varargs": "worldfoundry.core.utils.functional_utils",
    "random_derangement": "worldfoundry.core.utils.torch_utils",
    "readable_count_parameters": "worldfoundry.core.utils.torch_utils",
    "remove_batch_dim": "worldfoundry.core.utils.array_tensor_utils",
    "resolve_generation_max_new_tokens": "worldfoundry.core.utils.inference_runtime",
    "resolve_inference_batch_size": "worldfoundry.core.utils.inference_runtime",
    "resize_and_letterbox": "worldfoundry.core.utils.image_utils",
    "safe_hash": "worldfoundry.core.utils.misc_utils",
    "sanity_check_image_tensor": "worldfoundry.core.utils.image_utils",
    "save_torch": "worldfoundry.core.utils.torch_utils",
    "sequential_split_dataset": "worldfoundry.core.utils.torch_utils",
    "set_deterministic": "worldfoundry.core.utils.torch_utils",
    "set_random_seed": "worldfoundry.core.utils.torch_utils",
    "set_os_envs": "worldfoundry.core.utils.misc_utils",
    "set_requires_grad": "worldfoundry.core.utils.torch_utils",
    "set_seed_everywhere": "worldfoundry.core.utils.torch_utils",
    "split_horizontal_views": "worldfoundry.core.utils.image_utils",
    "setattr_nested": "worldfoundry.core.utils.misc_utils",
    "setitem_nested": "worldfoundry.core.utils.misc_utils",
    "shape_avgpool1d": "worldfoundry.core.utils.shape_utils",
    "shape_avgpool2d": "worldfoundry.core.utils.shape_utils",
    "shape_avgpool3d": "worldfoundry.core.utils.shape_utils",
    "shape_conv1d": "worldfoundry.core.utils.shape_utils",
    "shape_conv2d": "worldfoundry.core.utils.shape_utils",
    "shape_conv3d": "worldfoundry.core.utils.shape_utils",
    "shape_convnd": "worldfoundry.core.utils.shape_utils",
    "shape_maxpool1d": "worldfoundry.core.utils.shape_utils",
    "shape_maxpool2d": "worldfoundry.core.utils.shape_utils",
    "shape_maxpool3d": "worldfoundry.core.utils.shape_utils",
    "shape_poolnd": "worldfoundry.core.utils.shape_utils",
    "shape_slice": "worldfoundry.core.utils.shape_utils",
    "shape_transpose_conv1d": "worldfoundry.core.utils.shape_utils",
    "shape_transpose_conv2d": "worldfoundry.core.utils.shape_utils",
    "shape_transpose_conv3d": "worldfoundry.core.utils.shape_utils",
    "shape_transpose_convnd": "worldfoundry.core.utils.shape_utils",
    "state_dict_class": "worldfoundry.core.utils.functional_utils",
    "stack_or_pad_tensors": "worldfoundry.core.utils.batch_ops",
    "stack_sequence_fields": "worldfoundry.core.utils.tree_utils",
    "tensor_hash": "worldfoundry.core.utils.torch_utils",
    "temporal_feature_consistency": "worldfoundry.core.utils.torch_utils",
    "tie_weights": "worldfoundry.core.utils.torch_utils",
    "to_image": "worldfoundry.core.utils.image_utils",
    "to_state_dict": "worldfoundry.core.utils.torch_utils",
    "torch_compute_stats": "worldfoundry.core.utils.torch_utils",
    "torch_flatten_indices": "worldfoundry.core.utils.torch_utils",
    "torch_load": "worldfoundry.core.utils.torch_utils",
    "torch_multi_index_select": "worldfoundry.core.utils.torch_utils",
    "torch_normalize": "worldfoundry.core.utils.torch_utils",
    "torch_save": "worldfoundry.core.utils.torch_utils",
    "tree_assign_at_path": "worldfoundry.core.utils.tree_utils",
    "tree_value_at_path": "worldfoundry.core.utils.tree_utils",
    "unfreeze_params": "worldfoundry.core.utils.torch_utils",
    "unstack_sequence_fields": "worldfoundry.core.utils.tree_utils",
    "unwrap_ddp_model": "worldfoundry.core.utils.torch_utils",
    "update_soft_params": "worldfoundry.core.utils.torch_utils",
    "weight_init": "worldfoundry.core.utils.torch_utils",
}


def __getattr__(name: str) -> Any:
    """Import the owning submodule once and cache the symbol on this module.

    Failure: :exc:`AttributeError` when ``name`` is in neither table. Successful
    lookups write into ``globals()`` so later access skips ``import_module``.
    """
    module_name = _SUBMODULES.get(name) or _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = module if name in _SUBMODULES else getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose already-materialized globals plus every lazy ``__all__`` name."""
    return sorted({*globals(), *__all__})


__all__ = sorted({*_SUBMODULES, *_EXPORT_MODULES})

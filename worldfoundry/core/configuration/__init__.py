"""Reusable structured configuration primitives for model inference.

Several configuration systems coexist here (CF-11); pick by use case:

- ``lazy_config`` (``LazyCall``/``LazyConfig``/``instantiate``): detectron2
  lineage, ``_target_``-based deferred object graphs. Default choice for
  new inference configs.
- ``cosmos_config`` (``Config`` + ``make_freezable`` attrs classes): the
  structured top-level config consumed by Cosmos-family runtimes; composed
  and overridden through ``hydra.override``.
- ``flags``: process-wide boolean switches snapshotted from environment
  variables at import.
- ``model_config`` (``ModelConfig``/``DiTConfig``...): DiT architecture
  hyperparameter dataclasses. NOTE: this ``ModelConfig`` (architecture) is
  unrelated to ``worldfoundry.core.model_loading.config.ModelConfig``
  (checkpoint download/placement), which is what ``worldfoundry.core``
  re-exports as ``ModelConfig``.

Legacy instantiate/registry vocabularies live in
``worldfoundry.core.io.config_utils`` (``cls``/``class`` keys) and
``worldfoundry.core.model_loading.factory``; do not adopt them in new code.

Public surface: the cosmos attrs tree, ``FLAGS``, LazyCall/LazyConfig,
DiT architecture dataclasses, and :func:`instantiate`.
"""

from .flags import FLAGS, INTERNAL, VALIDATION, VERBOSE
from .model_config import (
    ArchConfig,
    DiffusionModelConfig,
    DiTArchConfig,
    DiTConfig,
    ModelConfig,
    build_kwargs_from_config,
    require_config_value,
)


def __getattr__(name: str):
    # Shared dataclass helpers must work without Cosmos / Hydra dependencies.
    if name in {"CheckpointConfig", "Config", "EMAConfig", "ObjectStoreConfig", "make_freezable"}:
        from . import cosmos_config as module
    elif name in {"LazyCall", "LazyConfig", "LazyDict", "instantiate"}:
        from . import lazy_config as module
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "Config",
    "CheckpointConfig",
    "ArchConfig",
    "DiTArchConfig",
    "DiTConfig",
    "DiffusionModelConfig",
    "EMAConfig",
    "FLAGS",
    "INTERNAL",
    "LazyCall",
    "LazyConfig",
    "LazyDict",
    "ObjectStoreConfig",
    "ModelConfig",
    "VALIDATION",
    "VERBOSE",
    "build_kwargs_from_config",
    "instantiate",
    "make_freezable",
    "require_config_value",
]

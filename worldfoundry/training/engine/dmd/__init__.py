"""Family-neutral native DMD materialization.

The DMD math, optimizer engine, session, and checkpoint layer are already
model-neutral (they consume the ``FlowPredictionAdapter`` seam).  This package
holds the family-neutral *materializer* body and the strategy protocol that a
model family supplies to plug its role loader, cache contract, tuning policy,
and run/bundle types into that shared body.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DMDFamilyStrategy": ".strategy",
    "DMDTrainableRoles": ".strategy",
    "materialize_dmd_training_run": ".materialize",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

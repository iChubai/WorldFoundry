from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any


def runtime_root() -> Path:
    return Path(__file__).resolve().parent / "dvlt_runtime"


def runtime_src() -> Path:
    return runtime_root() / "src"


__all__ = [
    "DEFAULT_DVLT_CHECKPOINT",
    "DEFAULT_DVLT_IMAGE_SIZE",
    "DEFAULT_DVLT_INFERENCE_STEPS",
    "DEFAULT_DVLT_PATCH_SIZE",
    "DVLTRuntime",
    "DVLTSynthesis",
    "load_runtime",
    "runtime_root",
    "runtime_src",
]


def __getattr__(name: str) -> Any:
    if name in {
        "DEFAULT_DVLT_CHECKPOINT",
        "DEFAULT_DVLT_IMAGE_SIZE",
        "DEFAULT_DVLT_INFERENCE_STEPS",
        "DEFAULT_DVLT_PATCH_SIZE",
        "DVLTRuntime",
        "DVLTSynthesis",
        "load_runtime",
    }:
        module = import_module(".dvlt_synthesis" if name == "DVLTSynthesis" else ".runtime", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)

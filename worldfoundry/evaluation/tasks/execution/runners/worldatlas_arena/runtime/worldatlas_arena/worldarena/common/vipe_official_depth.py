"""Install Arena official depth clones before WorldFoundry ViPE imports them.

WorldArena and WorldFoundry are independent. Memory ViPE may still use
WorldFoundry only as the SLAM/pipeline orchestrator. UniDepth, Video-Depth-Anything,
and Prior-Depth-Anything must load from ``WorldArena/thirdparty/``. This module
puts those clones on ``sys.path`` and replaces WorldFoundry's depth factory /
package names so ``processors.py`` never executes the vendored copies.
"""

from __future__ import annotations

import sys
import types

from worldarena.common.vipe_official_priorda import (
    ArenaPriorDAModel,
    ensure_priorda_on_path,
    ensure_torch_cluster_shim,
)
from worldarena.common.vipe_official_unidepth import ArenaUniDepth2Model, ensure_unidepth_on_path
from worldarena.common.vipe_official_vda import ArenaVideoDepthAnythingModel, ensure_vda_on_path

WF_DEPTH = "worldfoundry.base_models.three_dimensions.depth"
WF_UNIDEPTH = f"{WF_DEPTH}.unidepth"
WF_VDA = f"{WF_DEPTH}.videodepthanything"
WF_PRIORDA = f"{WF_DEPTH}.priorda"

_ORIGINAL_MAKE = None
_INSTALLED = False


def ensure_official_depth_on_path() -> dict[str, object]:
    ensure_torch_cluster_shim()
    return {
        "unidepth": ensure_unidepth_on_path(),
        "vda": ensure_vda_on_path(),
        "priorda": ensure_priorda_on_path(),
    }


def _seed_module(name: str, **exports: object) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    for key, value in exports.items():
        setattr(module, key, value)


def _patch_imported(module_name: str, **exports: object) -> None:
    module = sys.modules.get(module_name)
    if module is None:
        return
    for key, value in exports.items():
        if hasattr(module, key):
            setattr(module, key, value)


def install_official_depth_override() -> None:
    """Bind Arena official depth wrappers before ViPE imports ``make_depth_model``."""
    global _ORIGINAL_MAKE, _INSTALLED
    ensure_official_depth_on_path()

    _seed_module(WF_UNIDEPTH, UniDepth2Model=ArenaUniDepth2Model)
    _seed_module(WF_VDA, VideoDepthAnythingDepthModel=ArenaVideoDepthAnythingModel)
    _seed_module(WF_PRIORDA, PriorDAModel=ArenaPriorDAModel)

    import worldfoundry.base_models.three_dimensions.depth as depth_pkg

    if _ORIGINAL_MAKE is None:
        _ORIGINAL_MAKE = depth_pkg.make_depth_model

    def _make(model: str):
        if str(model).startswith("unidepth"):
            subtype = model.split("-", 1)[1] if "-" in model else "l"
            return ArenaUniDepth2Model(type=subtype)  # type: ignore[arg-type]
        return _ORIGINAL_MAKE(model)

    depth_pkg.make_depth_model = _make
    for module_name in (
        "worldfoundry.base_models.three_dimensions.general_3d.vipe.slam.system",
        "worldfoundry.base_models.three_dimensions.general_3d.vipe.pipeline.processors",
    ):
        _patch_imported(
            module_name,
            make_depth_model=_make,
            VideoDepthAnythingDepthModel=ArenaVideoDepthAnythingModel,
            PriorDAModel=ArenaPriorDAModel,
        )
    _INSTALLED = True

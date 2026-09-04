"""WorldFoundry ViPE depth-model surface that Arena official wrappers must expose.

``SLAMSystem`` reads ``supported_camera_types`` / ``depth_type`` before
``estimate``. ``AdaptiveDepthProcessor`` reads ``estimate(...).metric_depth``
or ``relative_inv_depth``. Arena wrappers are not WF subclasses, so they must
declare this surface explicitly.
"""

from __future__ import annotations

# Names WorldFoundry ViPE actually reads. Keep in sync with
# DepthEstimationModel + slam/system.py + pipeline/processors.py.
VIPE_DEPTH_MODEL_ATTRS = ("supported_camera_types", "depth_type", "estimate")


def pinhole_supported_camera_types():
    from worldfoundry.base_models.three_dimensions.general_3d.vipe.utils.cameras import CameraType

    return [CameraType.PINHOLE]


def worldfoundry_depth_model_public_attrs() -> tuple[str, ...]:
    """Public DepthEstimationModel names minus ABC helpers."""
    from worldfoundry.base_models.three_dimensions.depth.base import DepthEstimationModel

    skip = {"register"}
    return tuple(
        name
        for name in dir(DepthEstimationModel)
        if not name.startswith("_") and name not in skip
    )

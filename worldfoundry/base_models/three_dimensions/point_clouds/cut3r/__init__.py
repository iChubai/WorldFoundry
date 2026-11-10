"""
CUT3R - Unified inference package
Integrates dust3r and croco inference.
"""

# Export the main classes and functions through relative imports.
from .inference import inference, inference_recurrent, inference_step
from .model import ARCroco3DStereo, ARCroco3DStereoConfig
from .post_process import estimate_focal_knowing_depth
from .utils.camera import pose_encoding_to_camera
from .utils.geometry import depthmap_to_pts3d, geotrf
from .utils.image import load_images

__all__ = [
    "ARCroco3DStereo",
    "ARCroco3DStereoConfig",
    "depthmap_to_pts3d",
    "estimate_focal_knowing_depth",
    "geotrf",
    "inference",
    "inference_recurrent",
    "inference_step",
    "load_images",
    "pose_encoding_to_camera",
]

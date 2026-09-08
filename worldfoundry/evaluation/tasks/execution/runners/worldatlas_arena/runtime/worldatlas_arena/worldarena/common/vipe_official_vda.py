"""Load memory-track Video-Depth-Anything from WorldArena ``thirdparty/``.

The official clone exposes ``infer_video_depth(frames, target_fps, ...)`` and
returns ``(depths, fps)``. WorldFoundry's fork uses a list-only signature.
This wrapper keeps the ViPE ``estimate()`` contract without importing
WorldFoundry's vendored ``videodepthanything`` package.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from worldarena.benchmark.metric_errors import MetricLoadError
from worldarena.common.local_checkpoints import project_root
from worldarena.common.vipe_depth_contract import pinhole_supported_camera_types
from worldarena.common.vipe_memory_weights import vda_checkpoint_path

VDA_REPO = project_root() / "thirdparty" / "Video-Depth-Anything"
MODEL_CONFIG = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def official_vda_root() -> Path:
    explicit = os.environ.get("WORLDARENA_VDA_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return VDA_REPO


def ensure_vda_on_path() -> Path:
    root = official_vda_root()
    package = root / "video_depth_anything"
    if not package.is_dir():
        raise MetricLoadError(
            f"Official Video-Depth-Anything is missing at {root}. Clone "
            "https://github.com/DepthAnything/Video-Depth-Anything into "
            "WorldArena/thirdparty/Video-Depth-Anything."
        )
    inserted = str(root)
    if inserted in sys.path:
        sys.path.remove(inserted)
    sys.path.insert(0, inserted)
    return root


class ArenaVideoDepthAnythingModel:
    """WorldFoundry-compatible VDA wrapper backed by the official clone."""

    def __init__(
        self,
        model: Literal["vits", "vitl"] = "vits",
        input_size: int = 518,
        weights_path: str | None = None,
    ) -> None:
        if model not in MODEL_CONFIG:
            raise ValueError(f"Video-Depth-Anything model {model} not supported")
        ensure_vda_on_path()
        from video_depth_anything.video_depth import VideoDepthAnything

        self.input_size = input_size
        self.use_fp32 = model == "vits"
        self.model = VideoDepthAnything(**MODEL_CONFIG[model])
        weight = Path(weights_path).expanduser() if weights_path else vda_checkpoint_path()
        if not weight.is_file() or weight.stat().st_size <= 0:
            raise MetricLoadError(f"Video-Depth-Anything weight missing: {weight}")
        state = torch.load(str(weight), map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.cuda().eval()

    @property
    def depth_type(self):
        from worldfoundry.base_models.three_dimensions.depth.base import DepthType

        return DepthType.AFFINE_DISP

    @property
    def supported_camera_types(self):
        return pinhole_supported_camera_types()

    def estimate(self, src):
        from worldfoundry.base_models.three_dimensions.depth.base import DepthEstimationResult
        from worldfoundry.base_models.three_dimensions.general_3d.vipe.utils.misc import unpack_optional

        frame_list = unpack_optional(src.video_frame_list)
        frames = np.stack(frame_list, axis=0) if isinstance(frame_list, list) else np.asarray(frame_list)
        depths, _fps = self.model.infer_video_depth(
            frames,
            target_fps=1,
            input_size=self.input_size,
            fp32=self.use_fp32,
        )
        return DepthEstimationResult(relative_inv_depth=torch.from_numpy(depths).float().cuda())

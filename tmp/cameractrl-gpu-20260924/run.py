"""CameraCtrl checkpoint-backed synthetic-trajectory inference probe."""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path


root = Path("tmp/cameractrl-gpu-20260924")
root.mkdir(parents=True, exist_ok=True)
checkpoint_root = Path("../ckpts").resolve()
trajectory = root / "small-right-translation-16f.txt"
started = time.time()


def record(**fields: object) -> None:
    (root / "status.json").write_text(json.dumps(fields, indent=2, default=str) + "\n")


# CameraCtrl's text format: frame id, normalized intrinsics, two unused
# fields, then a 3x4 world-to-camera matrix. The origin is the first frame.
lines = ["frame fx fy cx cy unused unused r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz"]
for frame in range(16):
    tx = -0.12 * frame / 15
    values = [frame, 0.8, 0.8, 0.5, 0.5, 0, 0, 1, 0, 0, tx, 0, 1, 0, 0, 0, 0, 1, 0]
    lines.append(" ".join(str(value) for value in values))
trajectory.write_text("\n".join(lines) + "\n")

components = {
    "sd15_path": checkpoint_root / "stable-diffusion-v1-5--stable-diffusion-v1-5",
    "pose_adaptor_ckpt": checkpoint_root / "hehao13--CameraCtrl/CameraCtrl.ckpt",
    "image_lora_ckpt": checkpoint_root / "hehao13--CameraCtrl/RealEstate10K_LoRA.ckpt",
    "motion_module_ckpt": checkpoint_root / "guoyww--animatediff/v3_sd15_mm.ckpt",
    "motion_adapter_ckpt": checkpoint_root / "guoyww--animatediff/v3_sd15_adapter.ckpt",
}

record(status="starting", gpu=os.getenv("CUDA_VISIBLE_DEVICES"), trajectory=str(trajectory))
try:
    import cv2
    import numpy as np
    import torch

    from worldfoundry.synthesis.visual_generation.cameractrl.runtime import CameraCtrlRuntime

    assert torch.cuda.is_available()
    for path in components.values():
        assert path.exists(), path
    runtime = CameraCtrlRuntime.from_pretrained(
        device="cuda", **{key: str(value) for key, value in components.items()}
    )
    record(status="loading", gpu=os.getenv("CUDA_VISIBLE_DEVICES"))
    output = runtime.predict(
        prompt="A realistic living room with a red sofa and a lamp, the camera slowly moves sideways.",
        trajectory_file=str(trajectory),
        output_path=str(root / "camera_video.mp4"),
        num_frames=16,
        infer_steps=25,
        height=256,
        width=384,
        seed=42,
        fps=8,
    )
    assert output["status"] == "success"
    cap = cv2.VideoCapture(output["artifact_path"])
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame is not None and frame.size and np.isfinite(frame).all()
        frames.append(frame)
    cap.release()
    assert len(frames) == 16, len(frames)
    pixels = np.stack(frames)
    assert pixels.std() > 0
    result = {
        "components": {key: str(value) for key, value in components.items()},
        "trajectory": str(trajectory),
        "synthetic_translation_x_m": [0.0, 0.12],
        "prompt": "A realistic living room with a red sofa and a lamp, the camera slowly moves sideways.",
        "output": output,
        "decoded_frames": len(frames),
        "frame_shape": list(frames[0].shape),
        "pixel_min": int(pixels.min()),
        "pixel_max": int(pixels.max()),
        "pixel_std": float(pixels.std()),
        "elapsed_seconds": time.time() - started,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    record(status="inference_verified", result=str(root / "result.json"), gpu=os.getenv("CUDA_VISIBLE_DEVICES"))
except BaseException as error:
    record(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(), elapsed_seconds=time.time() - started, gpu=os.getenv("CUDA_VISIBLE_DEVICES"))
    traceback.print_exc()
    raise

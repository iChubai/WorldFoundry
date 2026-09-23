"""WoW 1.3B public-pipeline checkpoint-backed GPU probe."""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from pathlib import Path


root = Path("tmp/wow-13b-quality-recheck-20260924")
root.mkdir(parents=True, exist_ok=True)
checkpoint = Path("../ckpts/X-Humanoid--WoW-1-Wan-1.3B-2M").resolve()
input_image = Path("tmp/priorda-preflight-20260922/fixtures/assets/sample-2/rgb.jpg").resolve()
output_path = root / "future_world.mp4"
started = time.time()


def record(**fields: object) -> None:
    (root / "status.json").write_text(json.dumps(fields, indent=2, default=str) + "\n")


record(status="starting", checkpoint=str(checkpoint), gpu=os.getenv("CUDA_VISIBLE_DEVICES"))
try:
    import cv2
    import numpy as np
    import torch

    from worldfoundry.pipelines.wow.pipeline_wow import WoWArgs, WoWPipeline

    assert torch.cuda.is_available()
    assert checkpoint.is_dir() and (checkpoint / "WoW_video_dit.pt").is_file()
    assert input_image.is_file()
    record(status="loading", checkpoint=str(checkpoint), gpu=os.getenv("CUDA_VISIBLE_DEVICES"))
    pipeline = WoWPipeline.from_pretrained(
        model_path=str(checkpoint),
        device="cuda",
        synthesis_args=WoWArgs(
            gpu=0,
            steps=50,
            seed=42,
            num_frames=9,
            enable_vram_management=False,
        ),
    )
    load_seconds = time.time() - started
    record(status="inferencing", load_seconds=load_seconds, gpu=os.getenv("CUDA_VISIBLE_DEVICES"))
    output = pipeline(
        input_path=str(input_image),
        text_prompt="A small turquoise toy stays on a white plate on the red table. The room and the toy remain in view while the camera moves slightly closer.",
        steps=50,
        seed=42,
        num_frames=9,
        output_path=str(output_path),
        return_dict=True,
    )
    assert output["status"] == "ok" and output["runtime"] == "wow-wan-native"
    cap = cv2.VideoCapture(str(output_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame is not None and frame.size and np.isfinite(frame).all()
        frames.append(frame)
    cap.release()
    assert len(frames) == 9, len(frames)
    pixels = np.stack(frames)
    assert pixels.max() > pixels.min() and math.isfinite(float(pixels.std()))
    summary = {
        "checkpoint": str(checkpoint),
        "input_image": str(input_image),
        "prompt": "A small turquoise toy stays on a white plate on the red table. The room and the toy remain in view while the camera moves slightly closer.",
        "runtime": output["runtime"],
        "metadata": output["metadata"],
        "video": str(output_path),
        "decoded_frames": len(frames),
        "frame_shape": list(frames[0].shape),
        "pixel_min": int(pixels.min()),
        "pixel_max": int(pixels.max()),
        "pixel_std": float(pixels.std()),
        "load_seconds": load_seconds,
        "elapsed_seconds": time.time() - started,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    (root / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    record(status="inference_verified", result=str(root / "result.json"), gpu=os.getenv("CUDA_VISIBLE_DEVICES"))
except BaseException as error:
    record(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(), elapsed_seconds=time.time() - started, gpu=os.getenv("CUDA_VISIBLE_DEVICES"))
    traceback.print_exc()
    raise

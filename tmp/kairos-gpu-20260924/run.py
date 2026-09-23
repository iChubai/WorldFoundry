"""Kairos Sensenova 4B 480P checkpoint-backed GPU inference probe."""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path


root = Path("tmp/kairos-gpu-20260924")
root.mkdir(parents=True, exist_ok=True)
ckpts = Path("../ckpts").resolve()
model_root = ckpts / "kairos-agi--kairos-sensenova-4B-480P-pretrained"
source = Path("tmp/priorda-preflight-20260922/fixtures/assets/sample-2/rgb.jpg").resolve()
started = time.time()


def record(**fields: object) -> None:
    (root / "status.json").write_text(json.dumps(fields, indent=2, default=str) + "\n")


record(status="starting", gpu=os.getenv("CUDA_VISIBLE_DEVICES"))
try:
    import cv2
    import numpy as np
    import torch

    from worldfoundry.synthesis.visual_generation.kairos.runtime import KairosRuntime

    assert torch.cuda.is_available()
    dit = model_root / "kairos-common-4B-480P.safetensors"
    text_encoder = ckpts / "Qwen--Qwen2.5-VL-7B-Instruct"
    vae = ckpts / "Wan-AI--Wan2.1-T2V-14B/Wan2.1_VAE.pth"
    for path in (dit, text_encoder, vae, source):
        assert path.exists(), path
    runtime = KairosRuntime.from_pretrained(
        models_root=str(model_root),
        variant="pretrained",
        device="cuda",
        pretrained_dit=str(dit),
        text_encoder_path=str(text_encoder),
        vae_path=str(vae),
        nproc_per_node=1,
        master_port=29661,
        run_manage_libs=False,
        use_prompt_rewriter=False,
    )
    record(status="inferencing", gpu=os.getenv("CUDA_VISIBLE_DEVICES"), checkpoint=str(dit))
    output = runtime.predict(
        prompt="A small turquoise toy sits on a plate on a red table. The camera slowly moves closer.",
        images=str(source),
        output_path=str(root / "future_world.mp4"),
        num_frames=9,
        height=480,
        width=640,
        seed=42,
    )
    path = Path(output["artifact_path"])
    cap = cv2.VideoCapture(str(path))
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
    assert pixels.std() > 0
    summary = {
        "dit": str(dit),
        "text_encoder": str(text_encoder),
        "vae": str(vae),
        "input_image": str(source),
        "prompt": "A small turquoise toy sits on a plate on a red table. The camera slowly moves closer.",
        "video": str(path),
        "decoded_frames": len(frames),
        "frame_shape": list(frames[0].shape),
        "pixel_min": int(pixels.min()),
        "pixel_max": int(pixels.max()),
        "pixel_std": float(pixels.std()),
        "elapsed_seconds": time.time() - started,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    (root / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    record(status="inference_verified", result=str(root / "result.json"), gpu=os.getenv("CUDA_VISIBLE_DEVICES"))
except BaseException as error:
    record(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(), elapsed_seconds=time.time() - started, gpu=os.getenv("CUDA_VISIBLE_DEVICES"))
    traceback.print_exc()
    raise

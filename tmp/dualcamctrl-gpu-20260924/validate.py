"""One local DualCamCtrl GPU smoke with explicit synthetic camera/depth controls."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from worldfoundry.synthesis.visual_generation.dualcamctrl.runtime import DualCamCtrlRuntime


root = Path(__file__).resolve().parents[2]
out = root / "tmp/dualcamctrl-gpu-20260924"
source = root / "tmp/uni3c-validation-20260923/reference/data/demo_uni3c/reference.png"
image = Image.open(source).convert("RGB").resize((480, 320))
image.save(out / "input.png")
# A synthetic depth signal tests the depth branch, not depth-estimation quality.
gray = np.asarray(image.convert("L"), dtype=np.uint8)
Image.fromarray(gray).save(out / "depth-synthetic.png")
cameras = []
for frame in range(9):
    w2c = np.eye(4, dtype=np.float32)
    w2c[0, 3] = -0.025 * frame
    cameras.append([0, 1.0, 1.0, 0.5, 0.5, 0, 0, *w2c[:3].reshape(-1).tolist()])
torch.save({"cameras": torch.tensor(np.asarray(cameras, dtype=np.float32))}, out / "camera-synthetic.torch")
# Validate input serialization before spending minutes loading large weights.
assert DualCamCtrlRuntime.plucker_embedding(out / "camera-synthetic.torch", frame_len=9).shape == (1, 6, 9, 320, 480)

start = time.monotonic()
runtime = DualCamCtrlRuntime(device="cuda", allow_download=False, load_model=True)
result = runtime.predict(
    prompt="A woman in dark clothing stands in a sunlit courtyard. The camera moves slowly to the right.",
    image_path=out / "input.png",
    depth_path=out / "depth-synthetic.png",
    camera_path=out / "camera-synthetic.torch",
    output_path=out / "dualcamctrl-9f-8step.mp4",
    num_frames=9,
    num_inference_steps=8,
    height=320,
    width=480,
    seed=42,
)
result["elapsed_seconds"] = round(time.monotonic() - start, 2)
result["gpu_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
(out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))

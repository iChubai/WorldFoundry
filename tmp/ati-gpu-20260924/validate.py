"""Real ATI checkpoint smoke test through the public WorldFoundry pipeline."""

import io
import json
import os
import traceback
from pathlib import Path

import numpy as np
import torch

from worldfoundry.pipelines.ati.pipeline_ati import ATIPipeline


ROOT = Path(__file__).resolve().parent
CKPTS = Path("../ckpts").resolve()
IMAGE = CKPTS / "Wan2.1-I2V-14B-480P/examples/i2v_input.JPG"


def make_tracks():
    from PIL import Image

    width, height = Image.open(IMAGE).size
    coords = [(x, y) for y in np.linspace(height * 0.2, height * 0.8, 5)
              for x in np.linspace(width * 0.2, width * 0.8, 5)]
    tracks = np.zeros((len(coords), 121, 1, 3), dtype=np.float32)
    for i, (x, y) in enumerate(coords):
        tracks[i, :, 0, 0] = (x + np.linspace(0, 8, 121)) * 8
        tracks[i, :, 0, 1] = y * 8
        tracks[i, :, 0, 2] = 1
    buffer = io.BytesIO()
    np.savez_compressed(buffer, array=tracks)
    path = ROOT / "synthetic-tracks.pt"
    torch.save(buffer.getvalue(), path)
    return path


def main():
    steps = int(os.environ.get("ATI_STEPS", "8"))
    track_path = make_tracks()
    pipe = ATIPipeline.from_pretrained(
        model_path=str(CKPTS / "bytedance-research--ATI"),
        base_model_path=str(CKPTS / "Wan2.1-I2V-14B-480P"),
        device="cuda",
        t5_cpu=True,
        offload_model=True,
    )
    result = pipe(
        prompt="A white cat wearing sunglasses remains on a surfboard in the ocean.",
        image=str(IMAGE),
        track_path=str(track_path),
        output_path=str(ROOT / f"ati-cat-{steps}step.mp4"),
        num_frames=81,
        width=832,
        height=480,
        num_inference_steps=steps,
        return_dict=True,
    )
    (ROOT / f"status-{steps}step.json").write_text(json.dumps({
        "status": result["status"],
        "artifact_path": result["artifact_path"],
        "video_sha256": result["video_sha256"],
        "runtime_plan": result["runtime_plan"],
        "steps": steps,
        "frames": 81,
        "track_fixture": "25 synthetic visible tracks, 8-pixel horizontal drift over 121 track samples",
    }, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        steps = os.environ.get("ATI_STEPS", "8")
        (ROOT / f"status-{steps}step.json").write_text(json.dumps({
            "status": "failed", "error": repr(error), "traceback": traceback.format_exc()
        }, indent=2) + "\n")
        raise

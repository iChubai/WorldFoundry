import json
import os
from pathlib import Path

from worldfoundry.pipelines.spatia.pipeline_spatia import SpatiaPipeline


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT.parent / "ckpts"
OUT = Path(__file__).resolve().parent
FRAMES = int(os.environ.get("SPATIA_FRAMES", "9"))
STEPS = int(os.environ.get("SPATIA_STEPS", "4"))
NAME = os.environ.get("SPATIA_NAME", "demo")
WIDTH = int(os.environ.get("SPATIA_WIDTH", "832"))
HEIGHT = int(os.environ.get("SPATIA_HEIGHT", "480"))

trajectory = OUT / f"w2c-{FRAMES}.jsonl"
with trajectory.open("w", encoding="utf-8") as handle:
    for frame in range(FRAMES):
        w2c = [
            [1.0, 0.0, 0.0, frame * 0.005],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        handle.write(json.dumps(w2c) + "\n")

pipeline = SpatiaPipeline.from_pretrained(
    model_path=str(ASSETS / "Jinjing713--Spatia"),
    base_model_path=str(ASSETS / "Wan-AI--Wan2.2-TI2V-5B"),
    map_model_path=str(ASSETS / "facebook--map-anything"),
    device="cuda:0",
)
result = pipeline(
    prompt="A white cat wearing sunglasses on a surfboard, bright water and distant hills, gentle camera pan right.",
    images=str(ASSETS / "Wan-AI--Wan2.2-TI2V-5B/examples/i2v_input.JPG"),
    w2c_trajectory_file=str(trajectory),
    intrinsics=[[[500.0, 0.0, WIDTH / 2], [0.0, 500.0, HEIGHT / 2], [0.0, 0.0, 1.0]]],
    output_path=str(OUT / f"{NAME}.mp4"),
    num_frames=FRAMES,
    fps=16,
    width=WIDTH,
    height=HEIGHT,
    num_inference_steps=STEPS,
    seed=0,
    return_dict=True,
)
(OUT / f"{NAME}-result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
print(json.dumps(result, indent=2, default=str), flush=True)

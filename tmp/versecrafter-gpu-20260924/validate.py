"""Real-weight VerseCrafter smoke test through the public pipeline."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

from worldfoundry.pipelines.versecrafter import VerseCrafterPipeline


ROOT = Path(__file__).resolve().parent
FRAMES = int(os.environ.get("VERSE_FRAMES", "9"))
STEPS = int(os.environ.get("VERSE_STEPS", "8"))
NAME = os.environ.get("VERSE_NAME", "versecrafter-9f-8step")
OUTPUT = ROOT / f"{NAME}.mp4"
STATUS = ROOT / f"{NAME}-status.json"


def main() -> None:
    model = VerseCrafterPipeline.from_pretrained(
        model_path=Path("../ckpts/TencentARC--VerseCrafter").resolve(),
        base_model_path=Path("../ckpts/Wan-AI--Wan2.1-T2V-14B").resolve(),
        moge_model_path=Path("../ckpts/Ruicheng--moge-2-vitl-normal").resolve(),
        python_executable=str(Path("../envs/worldfoundry-unified-cu121/bin/python").resolve()),
        device="cuda",
        gpu_memory_mode="model_full_load",
    )
    result = model(
        prompt="A woman in dark clothing stands in a courtyard, and the camera slowly moves sideways to reveal the building and trees.",
        images=Path("tmp/uni3c-validation-20260923/reference/data/demo_uni3c/reference.png").resolve(),
        output_path=OUTPUT,
        width=832,
        height=480,
        num_frames=FRAMES,
        fps=16,
        num_inference_steps=STEPS,
        guidance_scale=5.0,
        seed=2025,
        return_dict=True,
    )
    STATUS.write_text(json.dumps({"status": "generated", "result": str(result)}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        STATUS.write_text(json.dumps({"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()}, indent=2))
        raise

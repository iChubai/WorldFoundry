"""Checkpoint-backed HY-World 2.0 WorldMirror through its public pipeline."""

import json
import traceback
from pathlib import Path

from worldfoundry.pipelines.hunyuan_world.pipeline_hy_world_2p0 import HYWorld2Pipeline


ROOT = Path(__file__).resolve().parent
STATUS = ROOT / "status.json"


def main() -> None:
    pipe = HYWorld2Pipeline.from_pretrained(
        model_path=str(Path("../ckpts/tencent--HY-World-2.0").resolve()),
        task="worldrecon",
        device="cuda",
    )
    result = pipe(
        input_path=str(Path("tmp/uni3c-validation-20260923/reference/data/demo_uni3c/video.mp4").resolve()),
        output_path=str(ROOT / "output"),
        strict_output_path=str(ROOT / "output"),
        target_size=518,
        video_strategy="new",
        video_min_frames=4,
        video_max_frames=8,
        fps=4,
        save_depth=True,
        save_normal=True,
        save_gs=True,
        save_camera=True,
        save_points=True,
        save_colmap=False,
        save_conf=False,
        save_rendered=False,
        apply_sky_mask=False,
    )
    STATUS.write_text(json.dumps({"status": "generated", "result": str(result)}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        STATUS.write_text(json.dumps({"status": "failed", "error": repr(error), "traceback": traceback.format_exc()}, indent=2))
        raise

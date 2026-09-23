"""Run the public HyDRA pipeline with staged local checkpoints."""

import argparse
import json
from pathlib import Path

from worldfoundry.pipelines.hydra.pipeline_hydra import HydraPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    ckpts = root.parent / "ckpts"
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source = root / "tmp/longsana-fixed-gpu-20260922/longsana-longlive-81f/demo.mp4"
    camera = root / "tmp/hydra-gpu-20260924/camera.json"
    try:
        pipe = HydraPipeline.from_pretrained(
            model_path={
                "checkpoint_path": ckpts / "H-EmbodVis--HyDRA/hydra.ckpt",
                "base_model_path": ckpts / "Wan-AI--Wan2.1-T2V-1.3B",
            },
            device="cuda:0",
        )
        result = pipe(
            prompt="A quiet room viewed from a gently moving camera.",
            video=source,
            camera_json=camera,
            output_path=output,
            num_frames=77,
            width=832,
            height=480,
            fps=15,
            num_inference_steps=args.steps,
            seed=42,
            return_dict=True,
        )
    except Exception as exc:
        (output.parent / f"{output.stem}-result.json").write_text(
            json.dumps({"status": "failed", "error": repr(exc)}, indent=2),
            encoding="utf-8",
        )
        raise
    (output.parent / f"{output.stem}-result.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

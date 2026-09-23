"""Compatibility launcher for official EgoWM 3-DoF and 25-DoF SVD inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--fps", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--motion-bucket-id", type=int, default=180)
    parser.add_argument("--action-scale", type=float, default=0.0)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_frames < 1:
        raise ValueError("num_frames must be at least 1")
    if args.num_inference_steps < 1:
        raise ValueError("num_inference_steps must be at least 1")
    if args.fps < 1:
        raise ValueError("fps must be at least 1")
    if args.height % 8 or args.width % 8:
        raise ValueError("height and width must be divisible by 8")
    for label, path in (
        ("official source", args.source_dir),
        ("checkpoint", args.checkpoint_path),
        ("base model", args.base_model_dir),
        ("input image", args.input_image),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} is missing: {path}")


def run(args: argparse.Namespace) -> Path:
    """Load the official action/state-conditioned SVD classes and generate a video."""
    _validate_args(args)
    source_dir = args.source_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint_path.expanduser().resolve()
    base_model_dir = args.base_model_dir.expanduser().resolve()
    input_image = args.input_image.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()

    source_text = str(source_dir)
    inserted_source = source_text not in sys.path
    if inserted_source:
        sys.path.insert(0, source_text)
    try:
        import torch
        from PIL import Image

        from models.svd_wrapper import (
            DebugActionUnetFwise2,
            DebugActionUnetFwise2state,
            DebugSVDActionPipeline,
            DebugSVDActionStateConstcfgPipeline,
        )

        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
        state_dict = checkpoint.get("unet", checkpoint)
        action_embedding = state_dict.get("add_action_embedding.linear_1.weight")
        if action_embedding is None:
            raise ValueError("EgoWM checkpoint has no action embedding")
        action_width = action_embedding.shape[1]
        has_state = any(key.startswith("add_state_embedding.") for key in state_dict)
        if action_width == 96 and not has_state:
            action_dims = 3
            unet_class = DebugActionUnetFwise2
            pipeline_class = DebugSVDActionPipeline
        elif action_width == 800 and has_state:
            action_dims = 25
            unet_class = DebugActionUnetFwise2state
            pipeline_class = DebugSVDActionStateConstcfgPipeline
        else:
            raise ValueError(
                f"Unsupported EgoWM checkpoint: action width {action_width}, "
                f"state embedding present={has_state}"
            )
        unet = unet_class.from_pretrained(
            str(base_model_dir),
            subfolder="unet",
            low_cpu_mem_usage=False,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        unet.load_state_dict(state_dict, strict=True)
        del state_dict, checkpoint

        pipeline = pipeline_class.from_pretrained(
            str(base_model_dir),
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipeline.to(args.device)
        pipeline.set_progress_bar_config(disable=False)

        image = Image.open(input_image).convert("RGB").resize((args.width, args.height))
        actions = torch.zeros((args.num_frames, action_dims), dtype=torch.float32)
        actions[:, 0] = float(args.action_scale)
        generator = torch.manual_seed(args.seed)

        inference_kwargs = dict(
            decode_chunk_size=min(8, args.num_frames),
            generator=generator,
            motion_bucket_id=args.motion_bucket_id,
            noise_aug_strength=0.1,
            actions=actions,
            fps=args.fps,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
        )
        if action_dims == 25:
            inference_kwargs["init_state"] = torch.zeros(25, dtype=torch.float32)
        frames = pipeline(image, **inference_kwargs).frames[0]

        from worldfoundry.core.io.video import save_video_h264

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_video_h264([image, *frames], output_path, fps=args.fps)
    finally:
        if inserted_source:
            sys.path.remove(source_text)

    print(f"EgoWM output: {output_path.name}")
    return output_path


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

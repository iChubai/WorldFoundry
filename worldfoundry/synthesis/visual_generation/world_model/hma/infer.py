"""Compatibility launcher for the official HMA continuous interactive model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SVD_SCALE = 0.18215
ACTION_VECTORS = {
    "right": (0.0, 1.0),
    "left": (0.0, -1.0),
    "down": (1.0, 0.0),
    "up": (-1.0, 0.0),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--input-image", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generated-frames", type=int, default=6)
    parser.add_argument("--prompt-horizon", type=int, default=3)
    parser.add_argument("--maskgit-steps", type=int, default=2)
    parser.add_argument("--direction", choices=tuple(ACTION_VECTORS), default="right")
    parser.add_argument("--action-scale", type=float, default=0.05)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not str(args.device).startswith("cuda"):
        raise ValueError("the official HMA sampler requires a CUDA device")
    if args.generated_frames < 1:
        raise ValueError("generated_frames must be at least 1")
    if args.prompt_horizon < 2:
        raise ValueError("prompt_horizon must be at least 2")
    if args.maskgit_steps < 1:
        raise ValueError("maskgit_steps must be at least 1")
    if args.fps < 1:
        raise ValueError("fps must be at least 1")
    for label, path in (
        ("official source", args.source_dir),
        ("checkpoint", args.checkpoint_dir),
        ("base model", args.base_model_dir),
        ("input image", args.input_image),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} is missing: {path}")


def _prepare_image(image, *, size: int = 256):
    """Apply the resize-and-center-crop contract used by HMA's simulator."""
    import cv2
    import numpy as np

    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = array.shape[:2]
    if height < width:
        new_height, new_width = size, int(size * width / height)
    else:
        new_height, new_width = int(size * height / width), size
    array = cv2.resize(array, (new_width, new_height))
    top = (new_height - size) // 2
    left = (new_width - size) // 2
    return array[top : top + size, left : left + size]


def run(args: argparse.Namespace) -> Path:
    """Generate an action-conditioned rollout with official HMA model classes."""
    _validate_args(args)
    source_dir = args.source_dir.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    base_model_dir = args.base_model_dir.expanduser().resolve()
    input_image = args.input_image.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()

    source_text = str(source_dir)
    inserted_source = source_text not in sys.path
    if inserted_source:
        sys.path.insert(0, source_text)
    try:
        import einops
        import numpy as np
        import torch
        from diffusers import AutoencoderKLTemporalDecoder
        from PIL import Image

        from hma.model.st_mar import STMAR

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        device = torch.device(args.device)

        image_encoder = AutoencoderKLTemporalDecoder.from_pretrained(
            str(base_model_dir),
            subfolder="vae",
            variant="fp16",
            torch_dtype=torch.bfloat16,
        ).to(device=device).eval()
        backbone = STMAR.from_pretrained(str(checkpoint_dir)).to(device=device).eval()

        initial_frame = _prepare_image(Image.open(input_image))
        normalized = initial_frame.astype(np.float32) / 127.5 - 1.0
        image_tensor = (
            torch.from_numpy(normalized.transpose(2, 0, 1))
            .to(device=device, dtype=torch.bfloat16)
            .unsqueeze(0)
        )
        latent = image_encoder.encode(image_tensor).latent_dist.mean * SVD_SCALE
        latent = einops.rearrange(latent, "b c h w -> b h w c").squeeze(0).to(torch.float32)
        cached_latents = torch.stack([latent.clone() for _ in range(args.prompt_horizon)])
        cached_actions = torch.zeros((args.prompt_horizon - 1, 1, 2), device=device, dtype=torch.float32)

        base_action = torch.tensor(ACTION_VECTORS[args.direction], device=device, dtype=torch.float32)
        action = base_action * float(args.action_scale)
        frames = [initial_frame]

        for _ in range(args.generated_frames):
            input_latents = torch.cat([cached_latents, torch.zeros_like(cached_latents[[0]])]).unsqueeze(0)
            input_latents = input_latents[:, : args.prompt_horizon + 1]
            input_latents[:, -1] = backbone.mask_token

            input_actions = torch.cat(
                [cached_actions, action.view(1, 1, 2), action.view(1, 1, 2)],
                dim=0,
            ).view(1, -1, 2)
            input_actions = input_actions[:, : args.prompt_horizon + 1]

            next_latent = backbone.maskgit_generate(
                input_latents,
                out_t=input_latents.shape[1] - 1,
                maskgit_steps=args.maskgit_steps,
                temperature=0.0,
                action_ids=input_actions,
                domain=["language_table"],
            )[0].squeeze(0)

            decoded_latent = einops.rearrange(next_latent.unsqueeze(0), "b h w c -> b c h w") / SVD_SCALE
            decoded_latent = torch.clamp(decoded_latent.to(dtype=torch.bfloat16), -25, 25)
            decoded = (
                image_encoder.decode(decoded_latent, num_frames=1)
                .sample.squeeze(0)
                .float()
                .detach()
                .cpu()
                .numpy()
            )
            frame = np.clip((decoded.transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)
            frames.append(frame)

            cached_latents = torch.cat([cached_latents[1:], next_latent.unsqueeze(0)])
            cached_actions = torch.cat([cached_actions[1:], action.view(1, 1, 2)])

        from worldfoundry.core.io.video import save_video_h264

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_video_h264(frames, output_path, fps=args.fps)
    finally:
        if inserted_source:
            sys.path.remove(source_text)

    print(f"HMA output: {output_path.name}")
    return output_path


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

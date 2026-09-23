"""Read-only asset preflight for remaining WorldGen and AC3D routes."""

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CKPTS = ROOT.parent / "ckpts"
DEFAULT_CKPT = ROOT.parent / "ckpt"
HF_HUB = Path(os.environ.get("HF_HUB_CACHE") or
              Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")


def entry(path):
    path = Path(path)
    return {"path": str(path), "exists": path.exists(), "is_file": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None}


def main():
    flux_dev = [CKPTS / "FLUX.1-dev", CKPTS / "hfd/black-forest-labs--FLUX.1-dev",
                DEFAULT_CKPT / "FLUX.1-dev",
                Path.home() / ".cache/worldfoundry/checkpoints/FLUX.1-dev",
                HF_HUB / "models--black-forest-labs--FLUX.1-dev"]
    ac3d = [DEFAULT_CKPT / "ac3d/2B/checkpoint-10000.pt",
            DEFAULT_CKPT / "ac3d/5B/checkpoint-10000.pt",
            CKPTS / "ac3d/2B/checkpoint-10000.pt",
            CKPTS / "ac3d/5B/checkpoint-10000.pt"]
    summary = {
        "worldgen": {
            "required_flux_dev_candidates": [entry(p) for p in flux_dev],
            "staged_loras": [entry(CKPTS / "LeoXie--WorldGen/models--WorldGen-Flux-Lora" / name)
                              for name in ("worldgen_text2scene.safetensors", "worldgen_img2scene.safetensors")],
            "source_reference": "worldfoundry/synthesis/visual_generation/worldgen/worldgen_runtime/src/worldgen/pano_gen.py:66",
            "blocker": "FLUX.1-dev base checkpoint absent from the checked local cache candidates; WorldGen LoRA alone cannot run.",
        },
        "ac3d": {
            "controlnet_candidates": [entry(p) for p in ac3d],
            "runtime_source": entry(ROOT / "worldfoundry/synthesis/visual_generation/ac3d/ac3d_runtime"),
            "source_reference": "worldfoundry/synthesis/visual_generation/ac3d/runtime.py:73",
            "blocker": "Neither 2B nor 5B AC3D ControlNet checkpoint is staged in the default or adjacent checkpoint roots.",
        },
    }
    out = Path(__file__).with_name("asset-preflight.json")
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

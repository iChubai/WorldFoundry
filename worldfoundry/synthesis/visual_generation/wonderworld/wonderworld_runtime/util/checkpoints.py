from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from worldfoundry.core.io.paths import resolve_local_hf_model_path


ONEFORMER_MODEL_ID = "shi-labs/oneformer_ade20k_swin_large"
ONEFORMER_REQUIRED_FILES = (
    "config.json",
    "preprocessor_config.json",
    "model.safetensors",
    "ade20k_panoptic.json",
    "merges.txt",
    "vocab.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
)


def checkpoint_roots() -> tuple[Path, ...]:
    primary = Path(
        os.environ.get(
            "WORLDFOUNDRY_CKPT_DIR",
            Path(__file__).resolve().parents[7] / "ckpt",
        )
    ).expanduser()
    roots = [primary]
    hfd_root = os.environ.get("WORLDFOUNDRY_HFD_ROOT")
    if hfd_root:
        roots.append(Path(hfd_root).expanduser())
    alternate_name = {"ckpt": "ckpts", "ckpts": "ckpt"}.get(primary.name)
    if alternate_name:
        roots.append(primary.with_name(alternate_name))
    return tuple(dict.fromkeys(roots))


def local_model_ref(default: str, *relative_paths: str) -> str:
    for root in checkpoint_roots():
        for relative_path in relative_paths:
            candidate = root / relative_path
            if candidate.exists():
                return str(candidate)
    return default


def resolve_oneformer_checkpoint() -> Path:
    return resolve_local_hf_model_path(
        ONEFORMER_MODEL_ID,
        required_files=ONEFORMER_REQUIRED_FILES,
    )


def local_processor_config(checkpoint: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    config_path = checkpoint_path / "preprocessor_config.json"
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)

    class_info_file = config.get("class_info_file")
    if not class_info_file or not (checkpoint_path / class_info_file).is_file():
        raise FileNotFoundError(
            f"WonderWorld OneFormer class metadata is missing from {checkpoint_path}"
        )
    config["repo_path"] = str(checkpoint_path)
    return config


def diffusers_local_load_kwargs(model_ref: str | Path) -> dict[str, Any]:
    model_path = Path(model_ref).expanduser()
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "use_safetensors": True,
    }
    fp16_components = (
        model_path / "text_encoder" / "model.fp16.safetensors",
        model_path / "unet" / "diffusion_pytorch_model.fp16.safetensors",
        model_path / "vae" / "diffusion_pytorch_model.fp16.safetensors",
    )
    if all(path.is_file() for path in fp16_components):
        kwargs["variant"] = "fp16"
    return kwargs


def load_oneformer():
    from transformers import (
        CLIPTokenizerFast,
        OneFormerForUniversalSegmentation,
        OneFormerImageProcessor,
        OneFormerProcessor,
    )

    checkpoint = resolve_oneformer_checkpoint()
    image_processor = OneFormerImageProcessor.from_dict(
        local_processor_config(checkpoint)
    )
    tokenizer = CLIPTokenizerFast.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    processor = OneFormerProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
    )
    model = OneFormerForUniversalSegmentation.from_pretrained(
        checkpoint,
        local_files_only=True,
        use_safetensors=True,
    )
    return processor, model

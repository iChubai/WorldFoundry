"""Dependency-free CLI compatibility for the Stable Virtual Camera demo."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any


def _coerce_value(value: str) -> object:
    lowered = value.strip().casefold()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_main_args(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Parse the subset of Python Fire syntax emitted by WorldFoundry."""

    parser = argparse.ArgumentParser(description="Stable Virtual Camera inference")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--data_items", nargs="*")
    parser.add_argument("--version", type=float, default=1.1)
    parser.add_argument("--task", default="img2img")
    parser.add_argument("--save_subdir", default="")
    parser.add_argument("--H", type=int)
    parser.add_argument("--W", type=int)
    parser.add_argument("--T", type=int)
    parser.add_argument("--use_traj_prior", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="stabilityai/stable-virtual-camera",
    )
    parser.add_argument("--weight_name", default="model.safetensors")
    parser.add_argument("--seed", type=int, default=23)
    namespace, unknown = parser.parse_known_args(argv)

    overwrite_options: dict[str, object] = {}
    index = 0
    while index < len(unknown):
        token = unknown[index]
        if not token.startswith("--"):
            parser.error(f"unexpected argument: {token}")
        key_value = token[2:].split("=", 1)
        key = key_value[0]
        if not key:
            parser.error("empty option name")
        if len(key_value) == 2:
            overwrite_options[key] = _coerce_value(key_value[1])
            index += 1
            continue
        if index + 1 < len(unknown) and not unknown[index + 1].startswith("--"):
            overwrite_options[key] = _coerce_value(unknown[index + 1])
            index += 2
        else:
            overwrite_options[key] = True
            index += 1

    result = vars(namespace)
    result.update(overwrite_options)
    return result


__all__ = ["parse_main_args"]

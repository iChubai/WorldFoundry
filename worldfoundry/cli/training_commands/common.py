"""Small, accelerator-free helpers shared by training CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from worldfoundry.training.recipes import PostTrainingRecipe, TrainingRecipe


# Canonical family tokens keyed by normalized native-recipe id or alias.  Only
# the recipe ids that a native training path can materialize are listed; the
# per-algorithm CLI handler still validates that the family matches the recipe.
_FAMILY_BY_RECIPE_ID = {
    "wan2.1-t2v-1.3b": "wan",
    "wan2.1-vace": "wan_vace",
    "wan-vace": "wan_vace",
    "wan2.1-vace-14b": "wan_vace",
    "step-video-t2v": "step_video",
    "stepvideo": "step_video",
    "step-video": "step_video",
    "stepvideo-t2v": "step_video",
    "vchitect-2-t2v": "vchitect",
    "vchitect": "vchitect",
    "vchitect-2": "vchitect",
}


def training_family(model_recipe: str) -> str:
    model_id = str(model_recipe).strip().lower().replace("_", "-")
    if model_id.startswith("sana-"):
        return "sana"
    family = _FAMILY_BY_RECIPE_ID.get(model_id)
    if family is not None:
        return family
    raise ValueError(f"native training does not support model recipe {model_recipe!r}")


def training_base_dir(value: Path | None) -> Path:
    """Resolve recipe paths against an explicit root or the launch directory."""

    if value is None:
        return Path.cwd().resolve()
    return value.expanduser().resolve()


def checkpoint_overrides(
    declarations: list[str] | None,
    *,
    base_dir: Path,
) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for declaration in declarations or ():
        key, separator, raw_path = str(declaration).partition("=")
        key = key.strip()
        raw_path = raw_path.strip()
        if not separator or not key or not raw_path:
            raise ValueError("checkpoint overrides must use NAME=PATH")
        if key in overrides:
            raise ValueError(f"duplicate checkpoint override: {key!r}")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        overrides[key] = str(path.resolve())
    return overrides


def load_cache_recipe(path: Path) -> TrainingRecipe | PostTrainingRecipe:
    """Load either native pre-training/SFT or post-training cache contracts."""

    import yaml

    from worldfoundry.training.recipes import (
        POST_TRAINING_RECIPE_SCHEMA,
        TRAINING_RECIPE_SCHEMA,
        PostTrainingRecipe,
        TrainingRecipe,
    )

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"training recipe must be .json, .yaml, or .yml: {path}")
    if not isinstance(payload, dict):
        raise TypeError(f"training recipe must contain a mapping: {path}")
    schema = payload.get("schema")
    if schema == TRAINING_RECIPE_SCHEMA:
        return TrainingRecipe.from_mapping(payload)
    if schema == POST_TRAINING_RECIPE_SCHEMA:
        return PostTrainingRecipe.from_mapping(payload)
    raise ValueError(f"unsupported native training recipe schema: {schema!r}")


__all__ = [
    "checkpoint_overrides",
    "load_cache_recipe",
    "training_base_dir",
    "training_family",
]

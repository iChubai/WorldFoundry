"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat -> src -> misc -> collation.py functionality."""

from collections.abc import Callable
from typing import Dict, Union

from torch import Tensor

Tree = Union[dict[str, "Tree"], Tensor]


def collate(trees: list[Tree], merge_fn: Callable[[list[Tensor]], Tensor]) -> Tree:
    """Merge nested dictionaries of tensors."""
    if isinstance(trees[0], Tensor):
        return merge_fn(trees)
    else:
        return {key: collate([tree[key] for tree in trees], merge_fn) for key in trees[0]}

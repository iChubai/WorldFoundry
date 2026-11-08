from collections.abc import Callable, Iterable, Sized
from typing import (
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch

# Config type
from omegaconf import DictConfig, OmegaConf

# PyTorch Tensor type
from torch import Tensor


def broadcast(tensor, src=0):
    if not _distributed_available():
        return tensor
    else:
        torch.distributed.broadcast(tensor, src=src)
        return tensor


def _distributed_available():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def parse_structured(fields: Any, cfg: dict | DictConfig | None = None) -> Any:
    # added by Xavier -- delete '--local-rank' in multi-nodes training, don't know why there is such a keyword
    if "--local-rank" in cfg:
        del cfg["--local-rank"]
    # added by Xavier -- delete '--local-rank' in multi-nodes training, don't know why there is such a keyword
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg

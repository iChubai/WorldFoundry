# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0
"""DINO construction leaves shared by dense vision backbones."""

from torch import Tensor, nn
from torch.nn.init import trunc_normal_


def init_weights_vit_timm(module: nn.Module, name: str = "") -> None:
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class BlockChunk(nn.ModuleList):
    def forward(self, x: Tensor | list[Tensor]) -> Tensor | list[Tensor]:
        for b in self:
            x = b(x)
        return x

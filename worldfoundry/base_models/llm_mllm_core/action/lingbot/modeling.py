# SPDX-FileCopyrightText: 2026 OpenEnvision
# SPDX-License-Identifier: Apache-2.0
# Adapted from Robbyant LingBot-VLA (Apache-2.0) and OpenFlamingo (MIT).
"""Task-token alignment heads with the released LingBot-VLA parameter hierarchy."""

import torch
from torch import nn

from worldfoundry.base_models.diffusion_model.models.encoders.lvdm.resampler import (
    FeedForward,
    PerceiverAttention,
)


class TaskTokenResampler(nn.Module):
    def __init__(
        self,
        dim_in: int = 768,
        dim_mid: int = 1024,
        dim_head: int = 64,
        dim_out: int = 1024,
        num_layers: int = 8,
        num_queries: int = 8,
        num_heads: int = 16,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()

        self.num_queries = num_queries
        self.proj_in1 = nn.Linear(dim_in, dim_mid)
        self.proj_in2 = nn.Linear(dim_in, dim_mid)
        self.proj_out = nn.Linear(dim_mid, dim_out)
        self.norm_out = nn.LayerNorm(dim_out)

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim_mid, dim_head=dim_head, heads=num_heads),
                        FeedForward(dim=dim_mid, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        queries = self.proj_in1(queries)
        x = self.proj_in2(x)

        for attn, ff in self.layers:
            queries = attn(x, queries) + queries
            queries = ff(queries) + queries

        queries = self.proj_out(queries)
        queries = self.norm_out(queries)
        return queries


class TaskTokenDepthHead(nn.Module):
    def __init__(
        self,
        proj_config: dict[str, int],
        llm_hidden_size: int = 4096,
        use_intermediate_depth: bool = False,
    ) -> None:
        super().__init__()

        self.projector = TaskTokenResampler(
            dim_in=llm_hidden_size,
            dim_mid=llm_hidden_size,
            dim_head=proj_config["dim_head"],
            dim_out=proj_config["dim_out"],
            num_layers=proj_config["num_layers"],
            num_heads=proj_config["num_heads"],
            num_queries=proj_config["num_backbone_tokens"],
            ff_mult=proj_config["ff_mult"],
        )

    def forward(self, llm_feats: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        queries = self.projector(llm_feats, queries)
        return queries

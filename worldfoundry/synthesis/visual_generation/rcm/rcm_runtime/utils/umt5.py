# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc

import torch

from worldfoundry.base_models.diffusion_model.models.encoders.wan.model import HuggingfaceTokenizer
from worldfoundry.base_models.diffusion_model.models.encoders.wan.reference import umt5_xxl
from worldfoundry.core.distributed.runtime import torch_process_group as distributed
from worldfoundry.core.io.assets.easy_io import easy_io
from worldfoundry.core.observability.rank_logging import log


def load_model_torch(model, ckpt_path):
    if distributed.is_rank0():
        ckpt = easy_io.load(
            ckpt_path,
            map_location="cuda",
        )
        model.load_state_dict(ckpt)

    distributed.sync_model_states(model, src=0)
    return model


class UMT5EncoderModel:
    def __init__(
        self,
        text_len=512,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),  # noqa: B008 - Default evaluated once at definition, as upstream.
        checkpoint_path="models_t5_umt5-xxl-enc-bf16.pth",
        tokenizer_path="google/umt5-xxl",
    ):
        self.text_len = text_len
        self.dtype = dtype
        self.device = device

        # init model
        model = umt5_xxl(encoder_only=True, dtype=dtype, device=device).eval().requires_grad_(False)
        log.info(f"loading {checkpoint_path}")
        assert checkpoint_path.endswith(".pth")
        model = load_model_torch(model, checkpoint_path)
        self.model = model
        self.model.to(self.device)
        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=text_len, clean="whitespace")

    def __call__(self, texts, device: torch.device | None = None):
        if device is None:
            device = self.device
        ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.model(ids, mask)
        # return [u[:v] for u, v in zip(context, seq_lens)]
        stack_emb = []
        for u, length in zip(context, seq_lens, strict=False):
            if length > self.text_len:
                stack_emb.append(u[: self.text_len])
            else:
                # pad with zeros to max_length
                zeros = torch.zeros(self.text_len - length, u.shape[1]).to(u)
                stack_emb.append(torch.cat([u[:length], zeros], dim=0))
        return torch.stack(stack_emb)


t5_encoder: UMT5EncoderModel | None = None


def get_umt5_embedding(
    checkpoint_path: str,
    prompts: str | list[str],
    device: str = "cuda",
    max_length: int = 512,
) -> torch.Tensor:
    global t5_encoder
    if t5_encoder is None:
        t5_encoder = UMT5EncoderModel(text_len=max_length, device=device, checkpoint_path=checkpoint_path)
    return t5_encoder(prompts, device=device)


def clear_umt5_memory():
    """
    Clears the GPU memory by deleting the global t5_encoder model and emptying the CUDA cache.
    """
    global t5_encoder
    if t5_encoder is not None:
        del t5_encoder
        t5_encoder = None
    gc.collect()
    torch.cuda.empty_cache()

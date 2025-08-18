# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Decoders for input reconstruction from learned representations.

This module provides decoder architectures for reconstructing original inputs
(images or time series) from latent representations. These decoders help evaluate
how well the learned representations preserve information from the input space.

The module includes:
- SignalDecoder: Time series reconstruction decoder
"""

import einops
import torch
import torch.nn as nn

from tactile_ssl.model.signal_transformer import SignalTransformer


class SignalDecoder(SignalTransformer):
    def __init__(self, input_embed_dim: int = 768, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder_embed = nn.Linear(input_embed_dim, self.embed_dim, bias=True)
        output_dim = self.time_chunk_size * self.in_chans
        self.decoder_pred = nn.Linear(self.embed_dim, output_dim, bias=True)

        self.init_weights()

    def forward(self, x):
        x = self.decoder_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        x = self.decoder_pred(x_norm)
        return x


class SignalMaskDecoder(SignalTransformer):
    def __init__(self, input_embed_dim: int = 768, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder_embed = nn.Linear(input_embed_dim, self.embed_dim, bias=True)
        self.patch_embed = nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        output_dim = self.time_chunk_size * self.in_chans
        self.decoder_pred = nn.Linear(self.embed_dim, output_dim, bias=True)
        self.init_weights()

    def forward(self, x, ids_restore):
        x = self.decoder_embed(x)

        mask_tokens = einops.repeat(
            self.mask_token,
            "1 1 c -> b n c",
            b=x.shape[0],
            n=ids_restore.shape[-1] - x.shape[-2],
        )
        x_ = torch.cat([x, mask_tokens], dim=-2)
        x = torch.gather(x_, dim=-2, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[-1]))  # unshuffle

        if self.pos_embed_fn == "sinusoidal":
            pos_embed = self.pos_embed(x_.device).float().unsqueeze(0)
        elif self.pos_embed_fn == "learned":
            pos_embed = self.pos_embed.float()
        else:
            raise NotImplementedError("Unknown position embeding function")

        x = x + pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.decoder_pred(x)
        return x


def SignalDecoderBase(**kwargs):
    model = SignalDecoder(num_heads=12, mlp_ratio=4, qkv_bias=True, pos_embed_fn="sinusoidal", **kwargs)
    return model

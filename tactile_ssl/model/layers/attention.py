# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

from torch import Tensor
from torch import nn

import torch
import torch.nn.functional as F
from tactile_ssl.model.layers.mlp import Mlp as MLP
import einops
import math

from .utils import RMSNorm


logger = logging.getLogger("dinov2")


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, attn_bias=None, return_attn=False) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)
        if attn_bias is not None:
            attn += attn_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return attn
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, return_attn=False) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, attn_bias, return_attn=return_attn)
        if return_attn:
            return super().forward(x, attn_bias, return_attn=True)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        if attn_bias is not None:
            attn_bias = attn_bias.expand(B, -1, -1, -1)
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DiffAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        depth: int = 0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        assert dim % (num_heads * 2) == 0, "dim must be divided by num_heads*2"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads // 2
        self.scale = self.head_dim**-0.5

        self.depth = depth
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.lambda_init = self.lambda_init_fn(depth)
        self.lambda_q = nn.Parameter(torch.zeros(2, self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k = nn.Parameter(torch.zeros(2, self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.rms_norm = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=False)

    @staticmethod
    def lambda_init_fn(depth):
        return 0.8 - 0.6 * math.exp(-0.3 * depth)

    def forward(self, x: Tensor, attn_bias=None, return_attn=False) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        q = einops.rearrange(q, "b h n (l d)->l b h n d", l=2)
        k = einops.rearrange(k, "b h n (l d)->l b h n d", l=2)
        attn = q @ k.transpose(-2, -1)
        if attn_bias is not None:
            attn += attn_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        lambda_qk = torch.exp(torch.sum(self.lambda_q * self.lambda_k, dim=-1).float()).type_as(q)
        scale = lambda_qk[0] - lambda_qk[1] + self.lambda_init
        attn = attn[0] - scale * attn[1]
        x = (attn @ v).transpose(1, 2)

        x = self.rms_norm(x)
        x = x * (1 - self.lambda_init)
        x = x.reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return attn
        return x


class MemEffDiffAttention(DiffAttention):
    def forward(self, x: Tensor, attn_bias=None, return_attn=False) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, attn_bias, return_attn=return_attn)
        if return_attn:
            return super().forward(x, attn_bias, return_attn=True)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)
        q = einops.rearrange(q, "b n h (l d)->(l b) n h d", l=2)
        k = einops.rearrange(k, "b n h (l d)->(l b) n h d", l=2)
        v = einops.repeat(v, "b n h d->(l b) n h d", l=2)

        if attn_bias is not None:
            attn_bias = attn_bias.expand(B, -1, -1, -1)
        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)

        lambda_qk = torch.exp(torch.sum(self.lambda_q * self.lambda_k, dim=-1).float()).type_as(q)
        scale = lambda_qk[0] - lambda_qk[1] + self.lambda_init
        x = x[:B] - scale * x[B:]

        x = self.rms_norm(x)
        x = x * (1 - self.lambda_init)
        x = x.reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Cross-Attention
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,
        qkv_bias=False,
        proj_bias=True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_sdpa=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim * 2), bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa

    def forward(self, q: Tensor, x: Tensor, attn_bias=None):
        B, n, C = q.shape
        q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                q = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        else:
            xattn = (q @ k.transpose(-2, -1)) * self.scale
            if attn_bias is not None:
                xattn += attn_bias
            xattn = xattn.softmax(dim=-1)  # (batch_size, num_heads, query_len, seq_len)
            xattn = self.attn_drop(xattn)
            q = xattn @ v

        q = q.transpose(1, 2).reshape(B, n, C)
        q = self.proj(q)
        q = self.proj_drop(q)
        return q


class MemEffCrossAttention(CrossAttention):
    def forward(self, q: Tensor, x: Tensor, attn_bias=None) -> Tensor:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(q, x)

        B, n, C = q.shape
        q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads)

        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads)
        k, v = unbind(kv, 2)

        q = memory_efficient_attention(q, k, v, attn_bias=attn_bias, p=self.attn_drop.p)
        q = q.reshape([B, n, C])
        q = self.proj(q)
        q = self.proj_drop(q)

        return q


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, q, x, attn_bias=None):
        y = self.xattn(q, self.norm1(x), attn_bias=attn_bias)
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


from functools import partial
from typing import Callable, Optional, List, Literal
from omegaconf import DictConfig

import einops
import torch
import torch.nn as nn

from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.data.xela.utils import XELA_FLATTEN_ORDER
from tactile_ssl.model import SignalTransformer

from .layers import PatchEmbed1d, PatchEmbed

log = get_pylogger(__name__)


class BraincoTransformer(SignalTransformer):
    def __init__(
        self,
        in_dim: int,
        in_chans: int,
        time_chunk_size: int,
        sequence_length: int,
        embed_dim: int,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        ffn_layer: str = "mlp",
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        head: Optional[nn.Module] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        pos_embed_fn: Literal["sinusoidal", "learned"] = "learned",
        init_values: Optional[float] = None,
        num_register_tokens: int = 0,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        with_masktoken: bool = False,
        causal: bool = False,
        normalization: Optional[DictConfig] = None,
        input_type: str = "signal",
        patch_size: int = 16,
    ):
        self.in_dim: int = in_dim
        self.in_chans: int = in_chans
        self.sequence_length: int = sequence_length
        self.time_chunk_size: int = time_chunk_size
        self.num_chunks: int = int(sequence_length // time_chunk_size)
        self.input_type = input_type
        
        if self.input_type == "signal":
            assert sequence_length % time_chunk_size == 0, "sequence length must be divisible by patch size"

        super().__init__(
            in_dim=in_dim,
            in_chans=in_chans,
            time_chunk_size=self.time_chunk_size,
            sequence_length=self.sequence_length,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            ffn_layer=ffn_layer,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            act_layer=act_layer,
            norm_layer=norm_layer,
            pos_embed_fn=pos_embed_fn,
            init_values=init_values,
            num_register_tokens=num_register_tokens,
            drop_path_rate=drop_path_rate,
            drop_path_uniform=drop_path_uniform,
            with_masktoken=with_masktoken,
            causal=causal,
        )

        if normalization is not None:
            self.register_buffer("signal_mean", torch.tensor(normalization.mean))
            self.register_buffer("signal_std", torch.tensor(normalization.std))
        else:
            self.register_buffer("signal_mean", torch.tensor([0.0]*in_chans))
            self.register_buffer("signal_std", torch.tensor([1.0]*in_chans))
        print(f"signal mean: {self.signal_mean}, signal std: {self.signal_std}")
        
        if self.input_type == "signal":
            self.patch_embed = nn.Linear(in_chans, self.embed_dim)
        else:
            raise ValueError(f"Unknown input_type: {self.input_type}")
        
        self.head = nn.Identity() if head is None else head

        self.init_weights()

    def update_stats(self, signal_mean, signal_std):
        assert isinstance(signal_mean, torch.Tensor) and isinstance(signal_std, torch.Tensor)
        assert signal_mean.shape[-1] == signal_std.shape[-1] == self.in_chans
        self.signal_mean = signal_mean
        self.signal_std = signal_std

    def normalize(self, x: torch.Tensor):
        if hasattr(self, "signal_mean") and hasattr(self, "signal_std"):
            if self.in_chans == 4:
                x = (x - self.signal_mean) / self.signal_std
            elif self.in_chans == 10:
                x = (x - self.signal_mean) / self.signal_std
            else:
                raise ValueError("Bad number of channels, must be 3 or 6")

        return x

    def pre_embed(self, x: torch.Tensor):
        b = x.shape[0]
        x = self.normalize(x)
        sensor_embed = self.patch_embed(x)
        return sensor_embed

    def create_causal_mask(self, x):
        """
        Create lower triangular block mask for Xela signals
        """
        _, chunked_t, n, _ = x.shape
        bias_size = chunked_t * n + self.num_register_tokens
        bias_size_multiple = int((bias_size // 8 + 1) * 8)  # cutlassF needs size to be multiple of 8
        attn_bias = torch.ones(
            (1, self.num_heads, bias_size, bias_size_multiple),
            dtype=torch.float32,
            device=x.device,
        )[..., :bias_size]

        # Mask out the future tokens
        attn_bias[..., self.num_register_tokens :, self.num_register_tokens :] = attn_bias[
            ..., self.num_register_tokens :, self.num_register_tokens :
        ].tril()

        # Prevent patch tokens from piggybacking on register tokens
        attn_bias[..., self.num_register_tokens :, : self.num_register_tokens] = 0

        # Create block causal mask
        for i in range(chunked_t):
            start = i * n + self.num_register_tokens
            end = (i + 1) * n + self.num_register_tokens
            attn_bias[..., start:end, start:end] = 1

        # Convert to additive bias
        attn_bias.masked_fill_(attn_bias == 0, float("-inf"))
        attn_bias.masked_fill_(attn_bias == 1, 0)

        return attn_bias


def brainco_tiny(
    in_dim: int,
    in_chans: int,
    sequence_length,
    depth=8,
    num_register_tokens=1,
    time_chunk_size=1,
    **kwargs,
):
    model = BraincoTransformer(
        in_dim=in_dim,
        in_chans=in_chans,
        sequence_length=sequence_length,
        time_chunk_size=time_chunk_size,
        embed_dim=192,
        depth=depth,
        num_heads=3,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )
    return model
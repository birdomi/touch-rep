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
from tactile_ssl.model import SignalTransformer

from .layers import PatchEmbed1d, PatchEmbed
from .layers import MemEffAttention, Mlp
from .layers import NestedTensorBlock as Block
from tactile_ssl.utils import apply_masks

log = get_pylogger(__name__)


class NullAwarePatchEmbed(nn.Module):
    """Drop-in replacement for nn.Linear used in BraincoTransformer.patch_embed.

    Standard nn.Linear behaviour for valid channels.
    For channels marked as null (null_mask=True):
      - their linear contribution is zeroed after input normalization
      - a learned per-channel null vector is added instead

    Args:
        in_chans  : number of input channels (e.g. 4 or 10)
        embed_dim : output embedding dimension
        num_null_chans : how many leading channels can be null (default = in_chans).
                        Set to 4 when in_chans=10 (sensor+pos cat) so position
                        channels are never treated as null.
    """

    def __init__(self, in_chans: int, embed_dim: int, num_null_chans: Optional[int] = None):
        super().__init__()
        self.in_chans       = in_chans
        self.embed_dim      = embed_dim
        self.num_null_chans = in_chans if num_null_chans is None else num_null_chans

        self.linear    = nn.Linear(in_chans, embed_dim)
        # Learned null contribution, one vector per nullable channel: (num_null_chans, embed_dim)
        self.null_embed = nn.Parameter(torch.zeros(self.num_null_chans, embed_dim))

    def forward(
        self,
        x: torch.Tensor,
        null_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x         : (..., in_chans)        — normalized and zero-filled at invalid positions
            null_mask : (..., num_null_chans)  — True where the original value was invalid
        Returns:
            (..., embed_dim)
        """
        out = self.linear(x)
        if null_mask is not None:
            # (..., num_null_chans) @ (num_null_chans, embed_dim) → (..., embed_dim)
            out = out + null_mask.float() @ self.null_embed
        return out


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
        self.pre_fusion_block_idx = 4
        
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
        self.sensor_block = nn.ModuleList([
            Block(
                attn_class=MemEffAttention,
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=0.0,
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=Mlp,
                init_values=init_values,
            )
            for _ in range(self.pre_fusion_block_idx)
        ])
        
        if normalization is not None:
            self.register_buffer("signal_mean", torch.tensor(normalization.mean))
            self.register_buffer("signal_std", torch.tensor(normalization.std))
        else:
            self.register_buffer("signal_mean", torch.tensor([0.0]*4))
            self.register_buffer("signal_std", torch.tensor([1.0]*4))
        print(f"signal mean: {self.signal_mean}, signal std: {self.signal_std}")
        
        if self.input_type == "signal":
            # num_null_chans=4: only the 4 sensor channels can be null;
            # when in_chans=10 (sensor+pos cat) position channels are never null.
            self.patch_embed = NullAwarePatchEmbed(
                in_chans=in_chans,
                embed_dim=self.embed_dim,
                num_null_chans=min(4, in_chans),
            )
            self.position_embed = nn.Linear(6, self.embed_dim)
        else:
            raise ValueError(f"Unknown input_type: {self.input_type}")
        
        self.head = nn.Identity() if head is None else head

        self.init_weights()

    def update_stats(self, signal_mean, signal_std, pos_mean=None, pos_std=None):
        assert isinstance(signal_mean, torch.Tensor) and isinstance(signal_std, torch.Tensor)

        if pos_mean is not None and pos_std is not None:
            assert isinstance(pos_mean, torch.Tensor) and isinstance(pos_std, torch.Tensor)
            signal_mean = signal_mean.reshape(-1)
            signal_std = signal_std.reshape(-1)
            pos_mean = pos_mean.reshape(-1)
            pos_std = pos_std.reshape(-1)

            if signal_mean.shape[-1] + pos_mean.shape[-1] == self.in_chans:
                signal_mean = torch.cat([signal_mean, pos_mean], dim=0)
                signal_std = torch.cat([signal_std, pos_std], dim=0)

        assert signal_mean.shape[-1] == signal_std.shape[-1] == self.in_chans
        self.signal_mean = signal_mean
        self.signal_std = signal_std

    def normalize(self, x: torch.Tensor):
        if hasattr(self, "signal_mean") and hasattr(self, "signal_std"):
            if self.in_chans == 4:
                x = (x - self.signal_mean) / self.signal_std
            elif self.in_chans == 10:
                x_norm = (x[..., :4] - self.signal_mean) / self.signal_std
                x = torch.cat([x_norm, x[..., 4:]], dim=-1)
            else:
                # General case: per-channel normalization, broadcast over last dim
                x = (x - self.signal_mean) / self.signal_std
                # print(self.signal_mean, self.signal_std)

        return x

    def pre_embed(self, x: torch.Tensor):
        # Detect null BEFORE normalization (invalid values are stored as -1)
        num_null = self.patch_embed.num_null_chans           # 4 (or fewer)
        null_mask = (x[..., :num_null] < 0)                 # (..., num_null_chans) bool

        # Normalize valid values first, then force invalid positions to zero so
        # they make no contribution through the linear projection.
        x = self.normalize(x)
        x = x.clone()
        x[..., :num_null] = x[..., :num_null].masked_fill(null_mask, 0.0)
        sensor_embed = self.patch_embed(x, null_mask)
        return sensor_embed

    def pre_pos_embed(self, x: torch.Tensor):
        position_embed_ = self.position_embed(x)
        return position_embed_

    def create_causal_mask(self, x):
        """
        Create lower triangular block mask for temporal BrainCo signals.
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

    def sensor_transform(self, x, bias):
        for blk in self.sensor_block:
            x = blk(x, bias) 
        out_norm = x
        return x, out_norm

    def transform(self, x, x_pos, bias):
        for i, blk in enumerate(self.blocks):
            if i == self.pre_fusion_block_idx:
                x_pos = x_pos + x
            x_pos = blk(x_pos, bias)
        x_norm = self.norm(x_pos)
        return x, x_norm
    
    def forward_features(
        self,
        x,
        pos,
        masks: Optional[List[torch.Tensor]] = None,
        mask_type: Optional[Literal["block", "tubelet"]] = None,
        masktoken_masks: Optional[List[torch.Tensor]] = None,
    ):
        x = self.pre_embed(x)
        pos = self.pre_pos_embed(pos)

        # print(x.shape, pos.shape)
        x, bias = self.prepare_tokens_with_mask(x, masks, mask_type, masktoken_masks)
        pos, pos_bias = self.prepare_tokens_with_mask(pos, masks, mask_type, masktoken_masks)
        sen, sen_norm = self.sensor_transform(x, bias)
        x_prenorm, x_postnorm = self.transform(sen, pos, bias)

        reg_tokens = x_postnorm[:, : self.num_register_tokens]
        patch_tokens = x_postnorm[:, self.num_register_tokens :]    
        patch_tokens_prenorm = x_prenorm[:, self.num_register_tokens :]
        out = {
            "x_norm_regtokens": reg_tokens,
            "x_norm_patchtokens": patch_tokens,
            "x_prenorm": patch_tokens_prenorm,
            "x_tokens": x_postnorm,
        }
        return out

class BraincoCatTransformer(BraincoTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Avoid retaining sensor_block parameters if it is completely unused
        self.sensor_block = nn.ModuleList([])

    def forward_features(
        self,
        x,
        pos,
        masks: Optional[List[torch.Tensor]] = None,
        mask_type: Optional[Literal["block", "tubelet"]] = None,
        masktoken_masks: Optional[List[torch.Tensor]] = None,
    ):
        x_cat = torch.cat([x, pos], dim=-1)
        x_embed = self.pre_embed(x_cat)

        x_tokens, bias = self.prepare_tokens_with_mask(x_embed, masks, mask_type, masktoken_masks)
        
        for blk in self.blocks:
            x_tokens = blk(x_tokens, bias)
            
        x_postnorm = self.norm(x_tokens)
        x_prenorm = x_tokens

        reg_tokens = x_postnorm[:, : self.num_register_tokens]
        patch_tokens = x_postnorm[:, self.num_register_tokens :]
        patch_tokens_prenorm = x_prenorm[:, self.num_register_tokens :]
        
        out = {
            "x_norm_regtokens": reg_tokens,
            "x_norm_patchtokens": patch_tokens,
            "x_prenorm": patch_tokens_prenorm,
            "x_tokens": x_postnorm,
        }
        return out

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

def brainco_cat_tiny(
    in_dim: int,
    in_chans: int,
    sequence_length,
    depth=8,
    num_register_tokens=1,
    time_chunk_size=1,
    **kwargs,
):
    model = BraincoCatTransformer(
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

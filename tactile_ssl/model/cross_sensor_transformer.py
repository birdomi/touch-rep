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
from .layers import MemEffAttention, Mlp
from .layers import NestedTensorBlock as Block
from tactile_ssl.utils import apply_masks

log = get_pylogger(__name__)

class SensorSpecificEmbed(nn.Module):
    def __init__(self, num_sensors, input_dim, hidden_dim):
        super().__init__()
        self.num_sensors = num_sensors
        self.input_dim = input_dim

        self.W = nn.Parameter(0.02 * torch.randn(num_sensors, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_sensors, hidden_dim))
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, x, sensor_ids):
        # x: (B, T, N, C)        
        ids = sensor_ids.unique()
        selected_W_ = self.W[sensor_ids]
        selected_b_ = self.b[sensor_ids]
        b, t, n, c = x.shape

        out_ = torch.zeros(b, n, self.W.shape[-1], device=x.device)

        for id in ids:            
            selected_x = x[sensor_ids == id]
            selected_W = selected_W_[sensor_ids==id]
            selected_b = selected_b_[sensor_ids==id]

            # Flatten T and N dimensions for matrix multiplication

            if id == 0: # sensor 1 == xela
                selected_x = selected_x[:, :, :, :3] # use only 3 xela mag channels -> 10 * 3 -> c = 30
                selected_x = einops.rearrange(selected_x, "b t n c -> b n (t c)") 
                
                # Project using sensor-specific weights
                out = torch.bmm(selected_x, selected_W) + selected_b.unsqueeze(1)

            if id == 1: # sensor 2 == actionsense dataset
                # print(selected_x.shape, selected_W.shape)
                selected_x = selected_x[:, -1, :, :] #remove temporal padding
                out = torch.bmm(selected_x, selected_W) + selected_b.unsqueeze(1)

            # Reshape back to (B, T, N, hidden)
            out_[sensor_ids == id] = out
            
            if id > self.num_sensors:
                raise ValueError(f"currently sensor {id} is not implemented")

        out_ = self.ln(out_)
        out_ = out_.unsqueeze(1) # b, 1, n, c
            
        return out_

class CrossSensorTransformer(SignalTransformer):
    def __init__(
        self,
        in_dim: int,
        in_chans: int,
        time_chunk_size: int,
        sequence_length: int,
        embed_dim: int,
        depth: int = 8,
        num_heads: int = 3,
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
        num_register_tokens: int = 1,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        with_masktoken: bool = False,
        causal: bool = False,
        normalization: Optional[DictConfig] = None,
        input_type: str = "xela",
        patch_size: int = 16,
        num_sensors: int = 2,
    ):
        self.in_dim: int = in_dim
        self.in_chans: int = in_chans
        self.sequence_length: int = sequence_length
        self.time_chunk_size: int = time_chunk_size
        self.num_chunks: int = int(sequence_length // time_chunk_size)
        self.input_type = input_type
        self.num_sensors = num_sensors
        
        if self.input_type == "xela":
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
        self.pre_fusion_block_idx = 4
        pos_blocks_list = [
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
            for i in range(4)
        ]
        self.pos_blocks = nn.ModuleList(pos_blocks_list)
        self.pos_norm = norm_layer(embed_dim)
        self.pos_linear = nn.Linear(7, embed_dim)

        self.pos_embed = nn.Parameter(
            torch.zeros(
                self.num_sensors,
                1,
                self.in_dim,
                self.embed_dim,
            )
        )

        self.sensor_block = nn.ModuleList([
            nn.ModuleList([
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
            for _ in range(self.num_sensors)
        ])
        self.sensor_norm = norm_layer(embed_dim)

        if hasattr(self, 'blocks'):
            self.fusion_block = self.blocks[:self.pre_fusion_block_idx]


        if normalization is not None:
            means = normalization.mean
            stds = normalization.std

            means_tensor = torch.zeros(self.num_sensors, 1, self.in_dim, self.in_chans)
            stds_tensor = torch.ones(self.num_sensors, 1, self.in_dim, self.in_chans)

            for i in range(min(len(means), self.num_sensors)):
                m = torch.tensor(means[i])
                s = torch.tensor(stds[i])
                c_i = m.shape[0]
                # Pad to in_chans and expand to in_dim
                means_tensor[i, 0, :, :c_i] = m
                stds_tensor[i, 0, :, :c_i] = s
            self.register_buffer("xela_mean", means_tensor)
            self.register_buffer("xela_std", stds_tensor)
        else:
            self.register_buffer("xela_mean", torch.zeros(self.num_sensors, 1, self.in_dim, self.in_chans))
            self.register_buffer("xela_std", torch.ones(self.num_sensors, 1, self.in_dim, self.in_chans))
        
        self.patch_embed = SensorSpecificEmbed(
            num_sensors=num_sensors,
            input_dim=in_chans,
            hidden_dim=self.embed_dim,
        )
        
        self.head = nn.Identity() if head is None else head
        self.init_weights()

    def update_stats(self, xela_mean, xela_std):
        assert isinstance(xela_mean, torch.Tensor) and isinstance(xela_std, torch.Tensor)
        assert xela_mean.shape[-1] == xela_std.shape[-1] == 3
        self.xela_mean = xela_mean
        self.xela_std = xela_std

    def normalize(self, x: torch.Tensor, sensor_ids: Optional[torch.Tensor] = None):
        if hasattr(self, "xela_mean") and hasattr(self, "xela_std"):
            # x: (B, T, N, C_total)
            x = einops.rearrange(x, "b t n (k c) -> b t n k c", c=self.in_chans)
            # print(x.shape)
            
            if sensor_ids is not None:
                # self.xela_mean: (S, 1, N, C)
                # target_means: (B, 1, N, C)
                target_means = self.xela_mean[sensor_ids]
                target_stds = self.xela_std[sensor_ids]
                
                # Reshape for broadcasting with (B, T, N, K, C)
                target_means = target_means.unsqueeze(-2) # (B, 1, N, 1, C)
                target_stds = target_stds.unsqueeze(-2)   # (B, 1, N, 1, C)
                # print(target_means.shape, target_stds.shape)
                # print(target_means[:,:,:,:,:6])
                
                x = (x - target_means) / target_stds
            else:
                # Fallback to first sensor stats if no sensor_ids
                x = (x - self.xela_mean[0:1].unsqueeze(-2)) / self.xela_std[0:1].unsqueeze(-2)
            
            x = einops.rearrange(x, "b t n k c -> b t n (k c)")
        return x

    def pre_embed(self, x: torch.Tensor, sensor_ids: Optional[torch.Tensor] = None):
        b = x.shape[0]

        x = self.normalize(x, sensor_ids=sensor_ids)

        sensor_embed = self.patch_embed(x, sensor_ids)

        return sensor_embed

    def pos_transform(self, x, bias):
        # calculate embedding from poses
        x = self.pos_linear(x)
        for blk in self.pos_blocks:
            x = blk(x, bias)
        x_norm = self.pos_norm(x)
        if self.num_register_tokens > 0:
            zero_tokens = torch.zeros(
                x_norm.shape[0], self.num_register_tokens, x_norm.shape[-1], device=x_norm.device, dtype=x_norm.dtype
            )
            x_norm = torch.cat([zero_tokens, x_norm], dim=1)
        return x, x_norm

    def apply_tubelet_masks(self, x, masks, sensor_idx = None, concat=True):
        all_x = []
        all_idx = []
        _, t, _, c = x.shape
        # print(x.shape, masks.shape)
        for mask in masks:
            mask_keep = einops.repeat(mask, "b n -> b t n c", c=c, t=t)
            masked_x = torch.gather(x, dim=-2, index=mask_keep)
            all_x.append(masked_x)
            if sensor_idx is not None:
                all_idx.append(sensor_idx)
        if not concat:
            return all_x, all_idx
        if sensor_idx is None:
            return torch.cat(all_x, dim=0)
        return torch.cat(all_x, dim=0), torch.cat(all_idx, dim=0)

    def prepare_tokens_with_mask(
        self,
        x,
        sensor_ids,
        masks,
        mask_type: Optional[Literal["block", "tubelet"]],
        masktoken_masks: Optional[List[torch.Tensor]],
    ):

        if self.pos_embed_fn == "sinusoidal":
            pos_embed = self.pos_embed(x.device).float().unsqueeze(0)
        elif self.pos_embed_fn == "learned":
            pos_embed = self.pos_embed.float()
        else:
            raise NotImplementedError("Unknown position embeding function")
        pos_embed = pos_embed[sensor_ids]
        x = x + pos_embed

        if masks is not None:
            if mask_type == "tubelet":
                x, sensor_ids = self.apply_tubelet_masks(x, masks, sensor_ids)
            elif mask_type == "block":
                x, sensor_ids = self.apply_block_masks(x, masks, sensor_ids)
        if self.causal:
            attn_bias = self.create_causal_mask(x)
        else:
            attn_bias = None

        if masktoken_masks is not None:
            x = self.apply_masktokens(x, masktoken_masks)
        x = einops.rearrange(x, "b t n c -> b (t n) c")

        if self.register_tokens is not None:
            x = torch.cat([self.register_tokens.expand(x.shape[0], -1, -1), x], dim=1)
        # print(x.shape)
        return x, attn_bias, sensor_ids

    def prepare_tokens_with_mask_pos(
        self,
        x,
        masks,        
        mask_type: Optional[Literal["block", "tubelet"]],
        masktoken_masks: Optional[List[torch.Tensor]],
    ):
        x = x[:, -1:, :, :] # select the last position
        t, n = x.shape[-3], x.shape[-2]

        # print(x.shape)

        assert t <= self.sequence_length, (
            f"Input sequence length {t} is greater than model sequence length {self.sequence_length}"
        )

        if masks is not None:
            if mask_type == "tubelet":
                x = self.apply_tubelet_masks(x, masks)
            elif mask_type == "block":
                x = apply_masks(x, masks)
            else:
                raise NotImplementedError(f"Unknown mask type {mask_type}")
        if self.causal:
            attn_bias = self.create_causal_mask(x)
        else:
            attn_bias = None
        
        x = x.squeeze(1)
        return x, attn_bias

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

    def sensor_transform(self, x, sensor_ids, bias):
        out = torch.zeros_like(x)
        ids = sensor_ids.unique()
        for sensor_id in ids:
            mask = sensor_ids == sensor_id
            curr_x = x[mask]
            
            for blk in self.sensor_block[sensor_id]:
                curr_x = blk(curr_x, bias)
                
            out[mask] = curr_x
        out_norm = self.sensor_norm(out)
        return out, out_norm

    def transform(self, x, x_pos, bias):
        x = x + x_pos
        for i, blk in enumerate(self.blocks):
            x = blk(x, bias)
        x_norm = self.norm(x)
        return x, x_norm
    
    def forward_features(
        self,
        x,
        pos,
        sensor_ids: Optional[torch.Tensor] = None,
        masks: Optional[List[torch.Tensor]] = None,
        mask_type: Optional[Literal["block", "tubelet"]] = None,
        masktoken_masks: Optional[List[torch.Tensor]] = None,
    ):
        x = self.pre_embed(x, sensor_ids=sensor_ids)
        x, bias, sensor_ids = self.prepare_tokens_with_mask(x, sensor_ids, masks, mask_type, masktoken_masks)
        pos, pos_bias = self.prepare_tokens_with_mask_pos(pos, masks, mask_type, masktoken_masks)
        pos, pos_norm = self.pos_transform(pos, pos_bias)
        sen, sen_norm = self.sensor_transform(x, sensor_ids, bias)
        x_prenorm, x_postnorm = self.transform(sen, pos_norm, bias)

        reg_tokens = x_postnorm[:, : self.num_register_tokens]
        patch_tokens = x_postnorm[:, self.num_register_tokens :]
        patch_tokens_prenorm = x_prenorm[:, self.num_register_tokens :]
        out = {
            "x_norm_regtokens": reg_tokens,
            "x_norm_patchtokens": patch_tokens,
            "x_prenorm": patch_tokens_prenorm,
        }
        return out

    def forward(self, x, pos, masks=None, mask_type=None, masktoken_masks=None, sensor_ids=None):
        out = self.forward_features(x, pos, masks, mask_type, masktoken_masks, sensor_ids=sensor_ids)
        return self.head(out["x_norm_patchtokens"])

def cross_sensor_tiny(
    in_dim: int,
    in_chans: int,
    sequence_length,
    depth=8,
    num_register_tokens=1,
    time_chunk_size=5,
    **kwargs,
):
    model = CrossSensorTransformer(
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
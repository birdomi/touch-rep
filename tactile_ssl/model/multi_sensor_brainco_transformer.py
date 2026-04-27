# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MultiSensorBraincoTransformer: BraincoTransformer with per-sensor normalization and embedding.

Supports two sensor types fed into the same model:
  sensor_id=0  BrainCo  — raw input (B, 1, 10, 4),  padded to (B, 1, 10, 40) by data wrapper
  sensor_id=1  XELA     — raw input (B, 1, 10, 40) = 10 frames × 4 channels

Per-sensor components:
  patch_embed[s]  : Linear(in_chans_s, embed_dim)
  sensor_block[s] : pre-fusion transformer blocks for sensor s
  signal_mean[s]  : (in_chans,) normalization mean for sensor s
  signal_std[s]   : (in_chans,) normalization std  for sensor s

Both sensors share the same positional embedding (position_embed) and the
shared transformer blocks (self.blocks) for cross-sensor fusion.
"""

from functools import partial
from typing import List, Literal, Optional

import einops
import torch
import torch.nn as nn
from omegaconf import DictConfig

from tactile_ssl.utils.logging import get_pylogger

from .brainco_transformer import BraincoTransformer
from .layers import MemEffAttention, Mlp
from .layers import NestedTensorBlock as Block

log = get_pylogger(__name__)

NUM_SENSORS = 2
SENSOR_ID_BRAINCO = 0
SENSOR_ID_XELA = 1


class MultiSensorBraincoTransformer(BraincoTransformer):
    """BraincoTransformer extended with per-sensor normalization and embedding.

    The collated batch has a uniform channel size of `xela_num_frames * in_chans`
    (e.g. 10 * 4 = 40).  BrainCo samples are zero-padded to this size by the
    SensorIdDatasetWrapper before collation; only the first `in_chans` channels
    carry valid data.

    Args:
        xela_num_frames: Number of XELA frames stacked into the channel dim (default 10).
        normalization:   DictConfig with ``mean`` and ``std`` fields each of shape
                         (num_sensors, in_chans) — i.e. a 2×4 list of lists.
        **kwargs:        All remaining kwargs forwarded to BraincoTransformer.
    """

    def __init__(
        self,
        xela_num_frames: int = 10,
        normalization: Optional[DictConfig] = None,
        **kwargs,
    ):
        # Capture block hyper-params before super().__init__ consumes them
        mlp_ratio = kwargs.get("mlp_ratio", 4.0)
        num_heads_kwarg = kwargs.get("num_heads", 3)
        qkv_bias = kwargs.get("qkv_bias", True)
        proj_bias = kwargs.get("proj_bias", True)
        ffn_bias = kwargs.get("ffn_bias", True)
        init_values = kwargs.get("init_values", None)
        norm_layer = kwargs.get("norm_layer", partial(nn.LayerNorm, eps=1e-6))
        act_layer = kwargs.get("act_layer", nn.GELU)

        # Parent initialises signal_mean/std as (in_chans,) buffers.
        # We pass normalization=None so the parent uses defaults; we override below.
        super().__init__(normalization=None, **kwargs)

        xela_in_chans = xela_num_frames * self.in_chans   # e.g. 10 * 4 = 40
        self.xela_in_chans = xela_in_chans
        self.xela_num_frames = xela_num_frames

        # ── Per-sensor patch embedding ────────────────────────────────────
        self.patch_embed = nn.ModuleList([
            nn.Linear(self.in_chans, self.embed_dim),    # sensor 0: BrainCo
            nn.Linear(xela_in_chans, self.embed_dim),    # sensor 1: XELA
        ])

        # ── Per-sensor pre-fusion blocks ─────────────────────────────────
        def _make_blocks():
            return nn.ModuleList([
                Block(
                    attn_class=MemEffAttention,
                    dim=self.embed_dim,
                    num_heads=num_heads_kwarg,
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

        self.sensor_block = nn.ModuleList([_make_blocks() for _ in range(NUM_SENSORS)])

        # ── Per-sensor normalization buffers: (num_sensors, in_chans) ────
        if normalization is not None:
            m = torch.tensor(normalization.mean, dtype=torch.float32)   # (2, 4)
            s = torch.tensor(normalization.std,  dtype=torch.float32)   # (2, 4)
            if m.dim() == 1:   # single-sensor stats — expand for both sensors
                m = m.unsqueeze(0).expand(NUM_SENSORS, -1).clone()
                s = s.unsqueeze(0).expand(NUM_SENSORS, -1).clone()
        else:
            m = torch.zeros(NUM_SENSORS, self.in_chans)
            s = torch.ones(NUM_SENSORS, self.in_chans)
        self.register_buffer("signal_mean", m)
        self.register_buffer("signal_std",  s)

        log.info(
            f"MultiSensorBraincoTransformer: embed_dim={self.embed_dim}, "
            f"xela_in_chans={xela_in_chans}, signal_mean={self.signal_mean.tolist()}"
        )

    # ── Stats update ─────────────────────────────────────────────────────────

    def update_stats(self, signal_mean, signal_std):
        """signal_mean / signal_std: tensors of shape (num_sensors, in_chans)."""
        assert signal_mean.shape[-1] == signal_std.shape[-1] == self.in_chans
        self.signal_mean = signal_mean
        self.signal_std  = signal_std

    # ── Per-sensor normalize ──────────────────────────────────────────────────

    def normalize(self, x: torch.Tensor, sensor_ids: Optional[torch.Tensor] = None):
        """Normalize x using per-sensor statistics.

        Args:
            x:          (B, T, N, C_max) where C_max = xela_in_chans (padded)
            sensor_ids: (B,) long tensor with sensor type per batch item

        Returns:
            Normalized tensor, same shape as x.
        """
        if sensor_ids is None:
            # Fallback: normalise first in_chans channels as BrainCo
            x = x.clone()
            mean = self.signal_mean[SENSOR_ID_BRAINCO]
            std  = self.signal_std[SENSOR_ID_BRAINCO]
            x[..., :self.in_chans] = (x[..., :self.in_chans] - mean) / std
            return x

        out = x.clone()
        for sid in sensor_ids.unique().tolist():
            mask = (sensor_ids == sid)
            xi   = x[mask]
            mean = self.signal_mean[sid]   # (in_chans,)
            std  = self.signal_std[sid]    # (in_chans,)

            if sid == SENSOR_ID_XELA:
                # xi: (B_s, T, N, T_f*in_chans) — reshape to apply 4-ch stats
                xi_r = einops.rearrange(xi, "b t n (f c) -> b t n f c",
                                        c=self.in_chans)   # (B_s, T, N, T_f, 4)
                xi_r = (xi_r - mean) / std
                out[mask] = einops.rearrange(xi_r, "b t n f c -> b t n (f c)")
            else:
                # BrainCo: only first in_chans channels are valid
                xi = xi.clone()
                xi[..., :self.in_chans] = (xi[..., :self.in_chans] - mean) / std
                out[mask] = xi
        return out

    # ── Per-sensor patch embedding ────────────────────────────────────────────

    def pre_embed(self, x: torch.Tensor, sensor_ids: Optional[torch.Tensor] = None):
        """Normalise and embed x using per-sensor patch_embed.

        Args:
            x:          (B, T, N, C_max)
            sensor_ids: (B,) long tensor

        Returns:
            (B, T, N, embed_dim)
        """
        x = self.normalize(x, sensor_ids)

        out = torch.zeros(*x.shape[:-1], self.embed_dim, device=x.device, dtype=x.dtype)

        if sensor_ids is None:
            out = self.patch_embed[SENSOR_ID_BRAINCO](x[..., :self.in_chans])
            return out

        for sid in sensor_ids.unique().tolist():
            mask = (sensor_ids == sid)
            if sid == SENSOR_ID_XELA:
                out[mask] = self.patch_embed[SENSOR_ID_XELA](x[mask])
            else:
                out[mask] = self.patch_embed[SENSOR_ID_BRAINCO](
                    x[mask][..., :self.in_chans]
                )
        return out

    # ── Per-sensor pre-fusion transformer ────────────────────────────────────

    def sensor_transform(self, x: torch.Tensor, sensor_ids, bias):
        """Apply per-sensor pre-fusion blocks.

        Args:
            x:          (B_eff, seq, embed_dim)  after prepare_tokens_with_mask
            sensor_ids: (B_eff,) long tensor, or None (falls back to BrainCo blocks)
            bias:       attention bias (None when causal=False)

        Returns:
            (x, x) — pre-norm and post-norm (identical; fusion norm is applied later)
        """
        if sensor_ids is None:
            # Fallback: use BrainCo blocks for all items
            for blk in self.sensor_block[SENSOR_ID_BRAINCO]:
                x = blk(x, bias)
            return x, x

        out = torch.zeros_like(x)
        for sid in sensor_ids.unique().tolist():
            mask = (sensor_ids == sid)
            curr = x[mask]
            for blk in self.sensor_block[sid]:
                curr = blk(curr, bias)
            out[mask] = curr
        return out, out

    # ── forward_features ─────────────────────────────────────────────────────

    def forward_features(
        self,
        x,
        pos,
        sensor_ids: Optional[torch.Tensor] = None,
        masks: Optional[List[torch.Tensor]] = None,
        mask_type: Optional[Literal["block", "tubelet"]] = None,
        masktoken_masks: Optional[List[torch.Tensor]] = None,
    ):
        x   = self.pre_embed(x,   sensor_ids=sensor_ids)
        pos = self.pre_pos_embed(pos)

        x,   bias     = self.prepare_tokens_with_mask(x,   masks, mask_type, masktoken_masks)
        pos, pos_bias = self.prepare_tokens_with_mask(pos, masks, mask_type, masktoken_masks)

        # Expand sensor_ids from (B,) → (B_eff,) = (M*B,) after tubelet masking
        if sensor_ids is not None and masks is not None:
            num_masks = masks.shape[0]   # M
            sensor_ids_exp = sensor_ids.unsqueeze(0).expand(num_masks, -1).flatten()
        else:
            sensor_ids_exp = sensor_ids

        sen, sen_norm = self.sensor_transform(x, sensor_ids_exp, bias)
        x_prenorm, x_postnorm = self.transform(sen, pos, bias)

        reg_tokens          = x_postnorm[:, :self.num_register_tokens]
        patch_tokens        = x_postnorm[:, self.num_register_tokens:]
        patch_tokens_prenorm = x_prenorm[:, self.num_register_tokens:]

        return {
            "x_norm_regtokens":  reg_tokens,
            "x_norm_patchtokens": patch_tokens,
            "x_prenorm":          patch_tokens_prenorm,
            "x_tokens":           x_postnorm,
        }


class CrossAttentionBlock(nn.Module):
    """One cross-attention layer with pre-norm, residual, and FFN.

    Q   = sensor tokens          (B, reg+N, D)  — updated each layer
    K/V = cat([sen_cur, x_pos])  (B, 2*(reg+N), D)  — pos fixed, sen portion updates

    Layout per block:
        q = q + CrossAttn(norm_q(q), norm_kv(kv))
        q = q + FFN(norm_ff(q))
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q  = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn    = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(embed_dim, eps=1e-6)
        hidden = int(embed_dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        kv_n = self.norm_kv(kv)
        out, _ = self.attn(self.norm_q(q), kv_n, kv_n, need_weights=False)
        q = q + out
        q = q + self.ff(self.norm_ff(q))
        return q


class MultiSensorBraincoTransformerCrossAttn(MultiSensorBraincoTransformer):
    """Multi-layer cross-attention fusion variant of MultiSensorBraincoTransformer.

    Fusion layout
    -------------
    sensor stream : sensor_block[0..pre_fusion_block_idx-1]  → sen    (B, reg+N, D)
    pos stream    : self.blocks[0..pre_fusion_block_idx-1]   → x_pos  (B, reg+N, D)  [fixed]

    for each CrossAttentionBlock (num_crossattn_layers):
        kv      = cat([sen_current, x_pos])              (B, 2*(reg+N), D)
        sen     = CrossAttentionBlock(Q=sen, K/V=kv)     (B, reg+N, D)

    fusion blocks : self.blocks[pre_fusion_block_idx..]  on sen only
    output        : norm(sen) → reg_tokens + patch_tokens

    Args:
        num_crossattn_layers: number of stacked cross-attention blocks (default 4).
    """

    def __init__(self, num_crossattn_layers: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=kwargs.get("mlp_ratio", 4.0),
            )
            for _ in range(num_crossattn_layers)
        ])

    def transform_crossattn(self, sen: torch.Tensor, pos: torch.Tensor, bias):
        """Fuse sensor and pos via stacked cross-attention, then run fusion blocks.

        Args:
            sen:  (B_eff, reg+N, D) — sensor_transform output
            pos:  (B_eff, reg+N, D) — raw positional embedding tokens
            bias: attention bias (None when causal=False)

        Returns:
            (x_prenorm, x_postnorm) both (B_eff, reg+N, D)
        """
        # Step 1: pos through pre-fusion blocks (fixed across all cross-attn layers)
        x_pos = pos
        for i, blk in enumerate(self.blocks):
            if i >= self.pre_fusion_block_idx:
                break
            x_pos = blk(x_pos, bias)

        # Step 2: stacked cross-attention
        #   pos is fixed; sen updates each layer.
        #   KV = cat([sen_current, x_pos]) so sensor's self-context also updates.
        sen_fused = sen
        for ca_blk in self.cross_attn_blocks:
            kv = torch.cat([sen_fused, x_pos], dim=1)   # (B_eff, 2*(reg+N), D)
            sen_fused = ca_blk(sen_fused, kv)

        # Step 3: shared fusion blocks on sensor-length sequence
        for i, blk in enumerate(self.blocks):
            if i >= self.pre_fusion_block_idx:
                sen_fused = blk(sen_fused, bias)

        x_norm = self.norm(sen_fused)
        return sen_fused, x_norm

    def forward_features(
        self,
        x,
        pos,
        sensor_ids: Optional[torch.Tensor] = None,
        masks: Optional[List[torch.Tensor]] = None,
        mask_type: Optional[Literal["block", "tubelet"]] = None,
        masktoken_masks: Optional[List[torch.Tensor]] = None,
    ):
        x   = self.pre_embed(x,   sensor_ids=sensor_ids)
        pos = self.pre_pos_embed(pos)

        x,   bias     = self.prepare_tokens_with_mask(x,   masks, mask_type, masktoken_masks)
        pos, pos_bias = self.prepare_tokens_with_mask(pos, masks, mask_type, masktoken_masks)

        if sensor_ids is not None and masks is not None:
            num_masks = masks.shape[0]
            sensor_ids_exp = sensor_ids.unsqueeze(0).expand(num_masks, -1).flatten()
        else:
            sensor_ids_exp = sensor_ids

        sen, sen_norm = self.sensor_transform(x, sensor_ids_exp, bias)
        x_prenorm, x_postnorm = self.transform_crossattn(sen, pos, bias)

        reg_tokens           = x_postnorm[:, :self.num_register_tokens]
        patch_tokens         = x_postnorm[:, self.num_register_tokens:]
        patch_tokens_prenorm = x_prenorm[:, self.num_register_tokens:]

        return {
            "x_norm_regtokens":   reg_tokens,
            "x_norm_patchtokens": patch_tokens,
            "x_prenorm":          patch_tokens_prenorm,
            "x_tokens":           x_postnorm,
        }


class MultiSensorBraincoTransformerConcat(MultiSensorBraincoTransformer):
    """Concat fusion variant of MultiSensorBraincoTransformer.

    Instead of summing sensor and pos transformer outputs at the fusion point,
    this model concatenates them along the sequence dimension and feeds the
    combined sequence into the fusion blocks.  The final output is taken only
    from the sensor portion (sensor query tokens + CLS/register token).

    Fusion layout
    -------------
    sensor stream : sensor_block[0..pre_fusion_block_idx-1]  → sen   (B, reg+N, D)
    pos stream    : self.blocks[0..pre_fusion_block_idx-1]   → x_pos (B, reg+N, D)
    fusion input  : cat([sen, x_pos], dim=1)                 (B, 2*(reg+N), D)
    fusion blocks : self.blocks[pre_fusion_block_idx..]
    output        : sensor portion fused[:, :reg+N]  → norm → reg_tokens + patch_tokens
    """

    def transform_concat(self, sen: torch.Tensor, pos: torch.Tensor, bias):
        """Run pos through pre-fusion blocks, concat with sen, run fusion blocks.

        Args:
            sen:  (B_eff, reg+N, D)  — output of sensor_transform
            pos:  (B_eff, reg+N, D)  — positional embedding tokens (raw, no blocks yet)
            bias: attention bias (None when causal=False)

        Returns:
            (x_prenorm, x_postnorm) both of shape (B_eff, reg+N, D) — sensor portion only
        """
        # Step 1: run pos through pre-fusion blocks (shared with summation branch)
        x_pos = pos
        for i, blk in enumerate(self.blocks):
            if i >= self.pre_fusion_block_idx:
                break
            x_pos = blk(x_pos, bias)

        # Step 2: concat sensor and processed pos along sequence dim
        fused = torch.cat([sen, x_pos], dim=1)  # (B_eff, 2*(reg+N), D)

        # Step 3: run fusion blocks; use unrestricted attention (no causal bias)
        for i, blk in enumerate(self.blocks):
            if i >= self.pre_fusion_block_idx:
                fused = blk(fused, None)

        # Step 4: take only the sensor portion
        sen_len = sen.shape[1]
        x_sen = fused[:, :sen_len]                # (B_eff, reg+N, D)
        x_norm = self.norm(x_sen)
        return x_sen, x_norm

    def forward_features(
        self,
        x,
        pos,
        sensor_ids: Optional[torch.Tensor] = None,
        masks: Optional[List[torch.Tensor]] = None,
        mask_type: Optional[Literal["block", "tubelet"]] = None,
        masktoken_masks: Optional[List[torch.Tensor]] = None,
    ):
        x   = self.pre_embed(x,   sensor_ids=sensor_ids)
        pos = self.pre_pos_embed(pos)

        x,   bias     = self.prepare_tokens_with_mask(x,   masks, mask_type, masktoken_masks)
        pos, pos_bias = self.prepare_tokens_with_mask(pos, masks, mask_type, masktoken_masks)

        if sensor_ids is not None and masks is not None:
            num_masks = masks.shape[0]
            sensor_ids_exp = sensor_ids.unsqueeze(0).expand(num_masks, -1).flatten()
        else:
            sensor_ids_exp = sensor_ids

        sen, sen_norm = self.sensor_transform(x, sensor_ids_exp, bias)
        x_prenorm, x_postnorm = self.transform_concat(sen, pos, bias)

        reg_tokens           = x_postnorm[:, :self.num_register_tokens]
        patch_tokens         = x_postnorm[:, self.num_register_tokens:]
        patch_tokens_prenorm = x_prenorm[:, self.num_register_tokens:]

        return {
            "x_norm_regtokens":   reg_tokens,
            "x_norm_patchtokens": patch_tokens,
            "x_prenorm":          patch_tokens_prenorm,
            "x_tokens":           x_postnorm,
        }


# ── Factory functions ─────────────────────────────────────────────────────────

def multi_sensor_brainco_tiny(
    in_dim: int,
    in_chans: int,
    sequence_length: int,
    depth: int = 8,
    num_register_tokens: int = 1,
    time_chunk_size: int = 1,
    xela_num_frames: int = 10,
    **kwargs,
):
    return MultiSensorBraincoTransformer(
        in_dim=in_dim,
        in_chans=in_chans,
        sequence_length=sequence_length,
        time_chunk_size=time_chunk_size,
        embed_dim=192,
        depth=depth,
        num_heads=3,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        xela_num_frames=xela_num_frames,
        **kwargs,
    )


def multi_sensor_brainco_crossattn_tiny(
    in_dim: int,
    in_chans: int,
    sequence_length: int,
    depth: int = 8,
    num_register_tokens: int = 1,
    time_chunk_size: int = 1,
    xela_num_frames: int = 10,
    num_crossattn_layers: int = 4,
    **kwargs,
):
    return MultiSensorBraincoTransformerCrossAttn(
        in_dim=in_dim,
        in_chans=in_chans,
        sequence_length=sequence_length,
        time_chunk_size=time_chunk_size,
        embed_dim=192,
        depth=depth,
        num_heads=3,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        xela_num_frames=xela_num_frames,
        num_crossattn_layers=num_crossattn_layers,
        **kwargs,
    )


def multi_sensor_brainco_cat_tiny(
    in_dim: int,
    in_chans: int,
    sequence_length: int,
    depth: int = 8,
    num_register_tokens: int = 1,
    time_chunk_size: int = 1,
    xela_num_frames: int = 10,
    **kwargs,
):
    return MultiSensorBraincoTransformerConcat(
        in_dim=in_dim,
        in_chans=in_chans,
        sequence_length=sequence_length,
        time_chunk_size=time_chunk_size,
        embed_dim=192,
        depth=depth,
        num_heads=3,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        xela_num_frames=xela_num_frames,
        **kwargs,
    )


def multi_sensor_brainco_cat_temporal_tiny(
    in_dim: int,
    in_chans: int,
    sequence_length: int = 5,
    depth: int = 8,
    num_register_tokens: int = 1,
    time_chunk_size: int = 1,
    xela_num_frames: int = 10,
    **kwargs,
):
    """multi_sensor_brainco_cat_tiny with temporal context (default sequence_length=5).

    The encoder processes ``sequence_length`` consecutive frames jointly so that
    intra-window temporal dynamics are captured inside the transformer, not just
    in the downstream probe.

    Downstream encode() groups the W window frames into (W // sequence_length)
    temporal chunks and produces (B, W // sequence_length, embed_dim) tokens.
    """
    return MultiSensorBraincoTransformerConcat(
        in_dim=in_dim,
        in_chans=in_chans,
        sequence_length=sequence_length,
        time_chunk_size=time_chunk_size,
        embed_dim=192,
        depth=depth,
        num_heads=3,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        xela_num_frames=xela_num_frames,
        **kwargs,
    )

"""FingerAngleTransformer: single-stream transformer for finger angles.

Architecture
------------
One stream:
  pos : finger_angles  (B, T, N_fingers=10, C=4) — finger angle vectors

Input is patch-embedded via PatchEmbed1d, positional embedding is applied
(per-finger with left/right hand bias), then processed by shared transformer
blocks. No contact stream, no sensor_block.
"""

from functools import partial
from typing import Callable, List, Literal, Optional

import einops
import torch
import torch.nn as nn

from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.model import SignalTransformer
from .layers import MemEffAttention, Mlp, PatchEmbed1d
from .layers import NestedTensorBlock as Block

log = get_pylogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_embed1d(in_chans: int, seq_len: int, chunk_size: int, embed_dim: int) -> PatchEmbed1d:
    return PatchEmbed1d(
        modal_chans=in_chans,
        modal_lens=seq_len,
        chunk_size=chunk_size,
        embed_dim=embed_dim,
        padding=0,
    )


def _apply_embed1d(x: torch.Tensor, embed: PatchEmbed1d, B: int) -> torch.Tensor:
    """(B, T, N, C) → PatchEmbed1d per sensor → (B, num_chunks, N, D)."""
    x = einops.rearrange(x, "b t n c -> (b n) c t")
    x = embed(x)                              # (B*N, D, num_chunks)
    x = einops.rearrange(x, "(b n) c t -> b t n c", b=B)
    return x


# ── main class ────────────────────────────────────────────────────────────────

class FingerAngleTransformer(SignalTransformer):
    """Single-stream Transformer for finger angles only.

    No contact stream, no sensor_block.
    Input: finger angles (B, T, N_fingers, pos_in_chans).
    Positional embedding is per-finger with left/right hand bias.
    Register tokens serve as the output embedding.

    Args:
        pos_in_dim      : N_fingers (must be even — left/right split)
        pos_in_chans    : C per finger (e.g. 4 for angle vectors)
        sequence_length : input time length T
        time_chunk_size : Conv1d stride (T → T//chunk tokens)
        embed_dim       : transformer hidden dim
        depth           : transformer block count
        normalization   : DictConfig(mean, std) for input normalization
    """

    def __init__(
        self,
        pos_in_dim: int = 10,
        pos_in_chans: int = 4,
        sequence_length: int = 1,
        time_chunk_size: int = 1,
        embed_dim: int = 192,
        depth: int = 8,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        ffn_layer: str = "mlp",
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        init_values: Optional[float] = None,
        num_register_tokens: int = 1,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        with_masktoken: bool = False,
        causal: bool = False,
        normalization=None,
    ):
        assert sequence_length % time_chunk_size == 0
        assert pos_in_dim % 2 == 0, f"pos_in_dim({pos_in_dim}) must be even (left/right hand)"

        super().__init__(
            in_dim=pos_in_dim,
            in_chans=pos_in_chans,
            sequence_length=sequence_length,
            time_chunk_size=time_chunk_size,
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
            pos_embed_fn="learned",
            init_values=init_values,
            num_register_tokens=num_register_tokens,
            drop_path_rate=drop_path_rate,
            drop_path_uniform=drop_path_uniform,
            with_masktoken=with_masktoken,
            causal=causal,
        )

        self.pos_in_dim   = pos_in_dim
        self.pos_in_chans = pos_in_chans

        D = embed_dim

        # Per-finger positional embedding: (2, N//2, D) for left/right hand
        self.angle_pos_embed = nn.Parameter(torch.zeros(2, pos_in_dim // 2, D))
        self.hand_embed      = nn.Parameter(torch.zeros(2, D))

        # PatchEmbed1d for angle stream
        self.angle_embed = _make_embed1d(pos_in_chans, sequence_length, time_chunk_size, D)

        if normalization is not None:
            m = torch.tensor(normalization.mean, dtype=torch.float32)
            s = torch.tensor(normalization.std,  dtype=torch.float32)
        else:
            m = torch.zeros(pos_in_chans)
            s = torch.ones(pos_in_chans)
        self.register_buffer("signal_mean", m)
        self.register_buffer("signal_std",  s)

        nn.init.trunc_normal_(self.hand_embed,      std=0.02)
        nn.init.trunc_normal_(self.angle_pos_embed, std=0.02)

        num_chunks = sequence_length // time_chunk_size
        log.info(
            f"FingerAngleTransformer: T={sequence_length}, chunk={time_chunk_size}, "
            f"num_chunks={num_chunks}, N_angles={pos_in_dim}, embed_dim={D}"
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _full_embed(self, per_hand: torch.Tensor) -> torch.Tensor:
        """(2, N//2, D) → (N, D) with left/right hand bias."""
        h  = self.hand_embed.float()
        lh = per_hand[0].float() + h[0]
        rh = per_hand[1].float() + h[1]
        return torch.cat([lh, rh], dim=0)

    # ── embedding ─────────────────────────────────────────────────────────────

    def pre_embed(self, pos: torch.Tensor) -> torch.Tensor:
        """PatchEmbed1d for angle stream. (B, T, N, C) → (B, T', N, D)"""
        return _apply_embed1d(pos, self.angle_embed, pos.shape[0])

    # ── token preparation ─────────────────────────────────────────────────────

    def prepare_tokens_with_mask(
        self,
        x,
        masks,
        mask_type: Optional[Literal["block", "tubelet"]],
        masktoken_masks: Optional[List[torch.Tensor]],
        joint_embed: Optional[torch.Tensor] = None,
        skip_register: bool = False,
    ):
        if joint_embed is not None:
            n = x.shape[-2]
            x = x + joint_embed.view(1, 1, n, -1)

        if masks is not None:
            if mask_type == "tubelet":
                x = self.apply_tubelet_masks(x, masks)
            elif mask_type == "block":
                from tactile_ssl.utils.masking import apply_masks
                x = apply_masks(x, masks)
            else:
                raise NotImplementedError(f"Unknown mask type: {mask_type}")

        attn_bias = self.create_causal_mask(x) if self.causal else None

        if masktoken_masks is not None:
            x = self.apply_masktokens(x, masktoken_masks)

        x = einops.rearrange(x, "b t n c -> b (t n) c")
        if self.register_tokens is not None and not skip_register:
            x = torch.cat([self.register_tokens.expand(x.shape[0], -1, -1), x], dim=1)

        return x, attn_bias

    # ── forward ───────────────────────────────────────────────────────────────

    def forward_features(
        self,
        pos: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        mask_type: Optional[Literal["block", "tubelet"]] = None,
        masktoken_masks: Optional[List[torch.Tensor]] = None,
    ) -> dict:
        """Forward pass.

        Args:
            pos             : (B, T, N_fingers, pos_in_chans)
            masks           : masking indices (or None)
            mask_type       : "block" or "tubelet"
            masktoken_masks : mask token positions

        Returns:
            dict with keys:
              x_norm_regtokens   : (B, num_register_tokens, D)
              x_norm_patchtokens : (B, num_patches, D)
              x_prenorm          : (B, num_patches, D)
              x_tokens           : (B, reg+patches, D)
        """
        pos = self.pre_embed(pos)

        ang_pe = self._full_embed(self.angle_pos_embed)

        pos, bias = self.prepare_tokens_with_mask(
            pos, masks, mask_type, masktoken_masks,
            joint_embed=ang_pe, skip_register=False,
        )

        x_prenorm, x_postnorm = self.transform(pos, bias)

        r = self.num_register_tokens
        return {
            "x_norm_regtokens":   x_postnorm[:, :r],
            "x_norm_patchtokens": x_postnorm[:, r:],
            "x_prenorm":          x_prenorm[:, r:],
            "x_tokens":           x_postnorm,
        }

    def forward(self, pos, **kwargs):
        return self.forward_features(pos, **kwargs)["x_norm_patchtokens"]

    def update_stats(
        self,
        signal_mean: torch.Tensor,
        signal_std: torch.Tensor,
        **_kwargs,
    ):
        self.signal_mean.copy_(signal_mean)
        self.signal_std.copy_(signal_std)


# ── factory functions ─────────────────────────────────────────────────────────

def finger_angle_tiny(
    pos_in_dim: int = 10,
    pos_in_chans: int = 4,
    sequence_length: int = 1,
    time_chunk_size: int = 1,
    depth: int = 8,
    num_register_tokens: int = 1,
    **kwargs,
) -> FingerAngleTransformer:
    """FingerAngleTransformer tiny (embed_dim=192, depth=8)."""
    return FingerAngleTransformer(
        pos_in_dim=pos_in_dim,
        pos_in_chans=pos_in_chans,
        sequence_length=sequence_length,
        time_chunk_size=time_chunk_size,
        embed_dim=192,
        depth=depth,
        num_heads=3,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def finger_angle_small(
    pos_in_dim: int = 10,
    pos_in_chans: int = 4,
    sequence_length: int = 1,
    time_chunk_size: int = 1,
    depth: int = 12,
    num_register_tokens: int = 1,
    **kwargs,
) -> FingerAngleTransformer:
    """FingerAngleTransformer small (embed_dim=384, depth=12)."""
    return FingerAngleTransformer(
        pos_in_dim=pos_in_dim,
        pos_in_chans=pos_in_chans,
        sequence_length=sequence_length,
        time_chunk_size=time_chunk_size,
        embed_dim=384,
        depth=depth,
        num_heads=6,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def finger_angle_base(
    pos_in_dim: int = 10,
    pos_in_chans: int = 4,
    sequence_length: int = 1,
    time_chunk_size: int = 1,
    depth: int = 12,
    num_register_tokens: int = 1,
    **kwargs,
) -> FingerAngleTransformer:
    """FingerAngleTransformer base (embed_dim=768, depth=12)."""
    return FingerAngleTransformer(
        pos_in_dim=pos_in_dim,
        pos_in_chans=pos_in_chans,
        sequence_length=sequence_length,
        time_chunk_size=time_chunk_size,
        embed_dim=768,
        depth=depth,
        num_heads=12,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )

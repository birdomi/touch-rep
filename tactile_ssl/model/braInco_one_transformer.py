from functools import partial
from typing import Callable, List, Literal, Optional

import einops
import torch
import torch.nn as nn
from omegaconf import DictConfig

from .layers import PatchEmbed1d
from .layers import init_weights_vit_timm
from .signal_transformer import SignalTransformer


def _make_embed1d(in_chans: int, seq_len: int, chunk_size: int, embed_dim: int) -> PatchEmbed1d:
    return PatchEmbed1d(
        modal_chans=in_chans,
        modal_lens=seq_len,
        chunk_size=chunk_size,
        embed_dim=embed_dim,
        padding=0,
    )


def _apply_embed1d(x: torch.Tensor, embed: PatchEmbed1d, b: int) -> torch.Tensor:
    x = einops.rearrange(x, "b t n c -> (b n) c t")
    x = embed(x)
    return einops.rearrange(x, "(b n) c t -> b t n c", b=b)


class BraincoOneTransformer(SignalTransformer):
    """One-stream Brainco transformer. sensor(4) + pos(3) share one embedding."""

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
        pos_in_chans: int = 3,
    ):
        if input_type != "signal":
            raise ValueError(f"Unknown input_type: {input_type}")
        if in_dim % 2 != 0:
            raise ValueError(f"in_dim({in_dim}) must be even for left/right hand embedding")

        self.num_chunks = int(sequence_length // time_chunk_size)
        self.input_type = input_type
        self.patch_size = patch_size
        self.pos_in_chans = pos_in_chans
        self.input_chans = in_chans + pos_in_chans

        super().__init__(
            in_dim=in_dim,
            in_chans=in_chans,
            time_chunk_size=time_chunk_size,
            sequence_length=sequence_length,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            ffn_layer=ffn_layer,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            head=head,
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
            self.register_buffer("signal_mean", torch.tensor(normalization.mean, dtype=torch.float32))
            self.register_buffer("signal_std", torch.tensor(normalization.std, dtype=torch.float32))
        else:
            self.register_buffer("signal_mean", torch.zeros(in_chans))
            self.register_buffer("signal_std", torch.ones(in_chans))

        self.patch_embed = _make_embed1d(self.input_chans, sequence_length, time_chunk_size, embed_dim)
        self.joint_pos_embed = nn.Parameter(torch.zeros(2, in_dim // 2, embed_dim))
        self.hand_embed = nn.Parameter(torch.zeros(2, embed_dim))
        self.head = nn.Identity() if head is None else head

        self.init_weights()
        nn.init.trunc_normal_(self.joint_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.hand_embed, std=0.02)

    def init_weights(self):
        super().init_weights()
        if hasattr(self, "joint_pos_embed"):
            nn.init.trunc_normal_(self.joint_pos_embed, std=0.02)
        if hasattr(self, "hand_embed"):
            nn.init.trunc_normal_(self.hand_embed, std=0.02)
        if hasattr(self, "patch_embed"):
            self.patch_embed.apply(init_weights_vit_timm)

    def update_stats(self, signal_mean, signal_std, pos_mean=None, pos_std=None):
        assert isinstance(signal_mean, torch.Tensor) and isinstance(signal_std, torch.Tensor)
        signal_mean = signal_mean.reshape(-1)
        signal_std = signal_std.reshape(-1)
        assert signal_mean.shape[-1] == signal_std.shape[-1] == self.in_chans
        self.signal_mean = signal_mean
        self.signal_std = signal_std

    def _full_embed(self) -> torch.Tensor:
        left = self.joint_pos_embed[0].float() + self.hand_embed[0].float()
        right = self.joint_pos_embed[1].float() + self.hand_embed[1].float()
        return torch.cat([left, right], dim=0)

    def normalize(self, x: torch.Tensor):
        return (x - self.signal_mean) / self.signal_std.clamp(min=1e-6)

    def pre_embed(self, x: torch.Tensor, pos: torch.Tensor):
        if x.shape[-1] != self.in_chans:
            raise ValueError(f"sensor channels must be {self.in_chans}, got {x.shape[-1]}")
        if pos.shape[-1] != self.pos_in_chans:
            raise ValueError(f"pos channels must be {self.pos_in_chans}, got {pos.shape[-1]}")

        b = x.shape[0]
        x = x.clone()
        x[x < 0] = 0.0
        x = self.normalize(x)

        x = torch.cat([x, pos], dim=-1)
        x = _apply_embed1d(x, self.patch_embed, b)

        joint_embed = self._full_embed().view(1, 1, self.in_dim, self.embed_dim)
        return x + joint_embed

    def create_causal_mask(self, x):
        _, chunked_t, n, _ = x.shape
        bias_size = chunked_t * n + self.num_register_tokens
        bias_size_multiple = int((bias_size // 8 + 1) * 8)
        attn_bias = torch.ones(
            (1, self.num_heads, bias_size, bias_size_multiple),
            dtype=torch.float32,
            device=x.device,
        )[..., :bias_size]

        attn_bias[..., self.num_register_tokens :, self.num_register_tokens :] = attn_bias[
            ..., self.num_register_tokens :, self.num_register_tokens :
        ].tril()
        attn_bias[..., self.num_register_tokens :, : self.num_register_tokens] = 0

        for i in range(chunked_t):
            start = i * n + self.num_register_tokens
            end = (i + 1) * n + self.num_register_tokens
            attn_bias[..., start:end, start:end] = 1

        attn_bias.masked_fill_(attn_bias == 0, float("-inf"))
        attn_bias.masked_fill_(attn_bias == 1, 0)
        return attn_bias

    def forward_features(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        masks: Optional[List[torch.Tensor]] = None,
        mask_type: Optional[Literal["block", "tubelet"]] = None,
        masktoken_masks: Optional[List[torch.Tensor]] = None,
    ):
        x = self.pre_embed(x, pos)
        x, bias = self.prepare_tokens_with_mask(x, masks, mask_type, masktoken_masks)
        x_prenorm, x_postnorm = self.transform(x, bias)

        r = self.num_register_tokens
        return {
            "x_norm_regtokens": x_postnorm[:, :r],
            "x_norm_patchtokens": x_postnorm[:, r:],
            "x_prenorm": x_prenorm[:, r:],
            "x_tokens": x_postnorm,
        }

    def forward(self, x, pos, **kwargs):
        out = self.forward_features(x, pos, **kwargs)
        return self.head(out["x_norm_patchtokens"])


def brainco_one_tiny(
    in_dim: int,
    in_chans: int = 4,
    sequence_length=1,
    depth=8,
    num_register_tokens=1,
    time_chunk_size=1,
    pos_in_chans: int = 3,
    **kwargs,
):
    return BraincoOneTransformer(
        in_dim=in_dim,
        in_chans=in_chans,
        sequence_length=sequence_length,
        time_chunk_size=time_chunk_size,
        embed_dim=192,
        depth=depth,
        num_heads=3,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        pos_in_chans=pos_in_chans,
        **kwargs,
    )

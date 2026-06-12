"""AngleAdaLNTransformer: angle stream conditioned by contact sensor tokens.

This variant keeps the dual-stream structure from :mod:`angle_transformer`:

  sensor : joint/contact stream processed by ``sensor_block``
  angle  : finger-angle stream processed by ``self.blocks``

The difference from ``AngleTransformer`` is the fusion stage. Sensor tokens are
not concatenated and then fed through the shared blocks. Instead, the pre-fusion
sensor output is pooled into a global condition and used by AdaLayerNorm inside
the angle-stream transformer blocks. The final token sequence still appends the
sensor tokens after the conditioned angle tokens so downstream code that pools
``x_tokens`` can consume both streams.
"""

from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn

from tactile_ssl.utils.logging import get_pylogger

from .angle_transformer import AngleTransformer
from .layers import MemEffAttention, Mlp, SwiGLUFFNFused, init_weights_vit_timm
from .layers.drop_path import DropPath
from .layers.layer_scale import LayerScale

log = get_pylogger(__name__)


class AdaLayerNorm(nn.Module):
    """LayerNorm modulated by a per-sample global condition."""

    def __init__(
        self,
        dim: int,
        cond_dim: Optional[int] = None,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ) -> None:
        super().__init__()
        cond_dim = dim if cond_dim is None else cond_dim
        self.norm = norm_layer(dim)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.dim() == 2:
            cond = cond[:, None, :]
        shift, scale = self.modulation(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class AdaLNBlock(nn.Module):
    """Transformer block whose two norms are conditioned by AdaLayerNorm."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_class: Callable[..., nn.Module] = MemEffAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        cond_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        cond_dim = dim if cond_dim is None else cond_dim

        self.norm1 = AdaLayerNorm(dim, cond_dim=cond_dim, norm_layer=norm_layer)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = AdaLayerNorm(dim, cond_dim=cond_dim, norm_layer=norm_layer)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.gate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_bias=None,
        return_attn: bool = False,
    ) -> torch.Tensor:
        if cond.dim() == 2:
            gate_cond = cond[:, None, :]
        else:
            gate_cond = cond
        gate_attn, gate_mlp = self.gate(gate_cond).chunk(2, dim=-1)

        x_attn = self.norm1(x, cond)
        if return_attn:
            return self.attn(x_attn, attn_bias=attn_bias, return_attn=True)

        x = x + self.drop_path1(gate_attn * self.ls1(self.attn(x_attn, attn_bias=attn_bias)))
        x = x + self.drop_path2(gate_mlp * self.ls2(self.mlp(self.norm2(x, cond))))
        return x


def _resolve_ffn_layer(ffn_layer: str):
    if ffn_layer == "mlp":
        return Mlp
    if ffn_layer in {"swiglufused", "swiglu"}:
        return SwiGLUFFNFused
    if ffn_layer == "identity":
        def f(*args, **kwargs):
            return nn.Identity()

        return f
    raise NotImplementedError(f"Unknown FFN layer: {ffn_layer}")


class AngleAdaLNTransformer(AngleTransformer):
    """AngleTransformer variant using sensor-conditioned AdaLayerNorm.

    ``sensor_block`` is identical to ``AngleTransformer``. ``self.blocks`` are
    replaced with ``AdaLNBlock`` and operate on the angle/register stream only.
    The sensor output is pooled to ``(B, D)`` and passed as the AdaLN condition.
    """

    def __init__(
        self,
        *args,
        ffn_layer: str = "mlp",
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            ffn_layer=ffn_layer,
            drop_path_rate=drop_path_rate,
            drop_path_uniform=drop_path_uniform,
            **kwargs,
        )

        if drop_path_uniform:
            dpr = [drop_path_rate] * self.depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]

        ffn_layer_ = _resolve_ffn_layer(ffn_layer)
        self.blocks = nn.ModuleList([
            AdaLNBlock(
                attn_class=MemEffAttention,
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=kwargs.get("mlp_ratio", 4.0),
                qkv_bias=kwargs.get("qkv_bias", True),
                proj_bias=kwargs.get("proj_bias", True),
                ffn_bias=kwargs.get("ffn_bias", True),
                drop_path=dpr[i],
                norm_layer=kwargs.get("norm_layer", partial(nn.LayerNorm, eps=1e-6)),
                act_layer=kwargs.get("act_layer", nn.GELU),
                ffn_layer=ffn_layer_,
                init_values=kwargs.get("init_values", None),
                cond_dim=self.embed_dim,
            )
            for i in range(self.depth)
        ])
        self.blocks.apply(init_weights_vit_timm)
        self._init_adaln_weights()

        if self.fine_tune_sensor:
            self._apply_fine_tune_sensor()

        log.info(
            f"AngleAdaLNTransformer: replaced fusion blocks with sensor-conditioned "
            f"AdaLN blocks, depth={self.depth}, embed_dim={self.embed_dim}"
        )

    def _init_adaln_weights(self) -> None:
        for block in self.blocks:
            nn.init.zeros_(block.norm1.modulation[-1].weight)
            nn.init.zeros_(block.norm1.modulation[-1].bias)
            nn.init.zeros_(block.norm2.modulation[-1].weight)
            nn.init.zeros_(block.norm2.modulation[-1].bias)
            nn.init.zeros_(block.gate[-1].weight)
            nn.init.zeros_(block.gate[-1].bias)

    def sensor_condition(self, sen: torch.Tensor) -> torch.Tensor:
        """Pool sensor tokens into the global AdaLN condition."""
        return sen.mean(dim=1)

    def transform_concat(
        self,
        sen: torch.Tensor,
        pos: torch.Tensor,
        sen_mask,
        pos_mask,
        bias,
    ):
        """Condition angle blocks with sensor output, then append sensor tokens.

        Sensor tokens are not inputs to ``self.blocks``. They only produce the
        global AdaLN condition for the angle/register stream.
        """
        sen_pe = self._full_embed(self.contact_pos_embed)
        ang_pe = self._full_embed(self.angle_pos_embed)

        if sen_mask is not None:
            ns = sen_mask.shape[-1]
            se = sen_pe[sen_mask.view(-1, ns)]
            sen = sen + se

            np_ = pos_mask.shape[-1]
            ae = ang_pe[pos_mask.view(-1, np_)]
            pos[:, 1:] = pos[:, 1:] + ae
        else:
            sen = sen + sen_pe
            pos[:, 1:] = pos[:, 1:] + ang_pe

        cond = self.sensor_condition(sen)
        for blk in self.blocks:
            pos = blk(pos, cond, attn_bias=None)

        fused = torch.cat([pos, sen], dim=1)
        x_norm = self.norm(fused)
        return fused, x_norm


def angle_adaln_tiny(
    in_dim: int = 42,
    in_chans: int = 1,
    pos_in_dim: int = 10,
    pos_in_chans: int = 4,
    sequence_length: int = 1,
    time_chunk_size: int = 1,
    depth: int = 8,
    num_register_tokens: int = 1,
    **kwargs,
) -> AngleAdaLNTransformer:
    """AngleAdaLNTransformer tiny (embed_dim=192, depth=8)."""
    return AngleAdaLNTransformer(
        in_dim=in_dim,
        in_chans=in_chans,
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


def angle_adaln_small(
    in_dim: int = 42,
    in_chans: int = 1,
    pos_in_dim: int = 10,
    pos_in_chans: int = 4,
    sequence_length: int = 1,
    time_chunk_size: int = 1,
    depth: int = 12,
    num_register_tokens: int = 1,
    **kwargs,
) -> AngleAdaLNTransformer:
    """AngleAdaLNTransformer small (embed_dim=384, depth=12)."""
    return AngleAdaLNTransformer(
        in_dim=in_dim,
        in_chans=in_chans,
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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Transformer architecture for multisensory data. To align with the DINOv2 architecture,
this module implements a multimodal transformer that can handle multiple sensing modalities
(e.g., vision, audio, IMU) and supports various fusion strategies (vanilla concatenation of
tokens or bottleneck fusion).

Please refer to this paper for more details about the bottleneck fusion strategy:
@article{nagrani2021attention,
  title={Attention bottlenecks for multimodal fusion},
  author={Nagrani, Arsha and Yang, Shan and Arnab, Anurag and Jansen, Aren and Schmid, Cordelia and Sun, Chen},
  journal={Advances in neural information processing systems},
  volume={34},
  pages={14200--14213},
  year={2021}
}
Also github repo: https://github.com/google-research/scenic/tree/main/scenic/projects/mbt

It includes a multimodal decoder for tasks such as image reconstruction.
"""

import math
from functools import partial
from typing import Callable, Optional, List, Literal, Dict, Any
import numpy as np

import einops
import torch
import torch.nn as nn

from tactile_ssl.utils.logging import get_pylogger

from .layers import MemEffAttention, Mlp
from .layers import NestedTensorBlock as Block
from .layers import DecoderBlock
from .layers import SinusoidalEmbed, SwiGLUFFNFused
from .layers import init_weights_vit_timm
from abc import abstractmethod

log = get_pylogger(__name__)


class MultimodalTransformer(nn.Module):
    def __init__(
        self,
        modals: List[str],
        modal_shapes: List[List[int]],
        embed_dim: int,
        depth: int = 12,
        block_class: Callable[..., nn.Module] = partial(Block, attn_class=MemEffAttention),
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        ffn_layer: str = "mlp",
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        pos_embed_fn: Literal["sinusoidal", "learned"] = "learned",
        init_values: Optional[int] = 1,
        num_register_tokens: int = 0,
        fusion_type: Literal["bottleneck", "vanilla"] = "vanilla",
        fusion_layer: int = 0,
        num_bottlenecks: int = 4,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        reversed: bool = False,
    ):
        super().__init__()
        self.num_modals = len(modals)
        self.modals = modals
        self.modal_shapes = {modal: modal_shape for modal, modal_shape in zip(modals, modal_shapes)}
        assert len(modal_shapes) == len(modals)

        self.modal_sizes = {modal: np.prod(modal_shape).astype(int) for modal, modal_shape in self.modal_shapes.items()}
        self.pos_embed_sizes = self.modal_sizes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        assert fusion_layer >= 0 and fusion_layer <= depth
        self.fusion_layer = fusion_layer
        self.fusion_type = fusion_type
        self.num_bottlenecks = num_bottlenecks if self.fusion_type == "bottleneck" else 0
        self.bottleneck = (
            nn.Parameter(torch.zeros(1, self.num_bottlenecks, embed_dim)) if self.fusion_type == "bottleneck" else None
        )

        assert num_register_tokens >= 0
        self.num_register_tokens = num_register_tokens
        self.register_tokens = (
            nn.Parameter(torch.zeros(self.num_modals, num_register_tokens, embed_dim))
            if num_register_tokens > 0
            else None
        )

        self.init_pos_embed(pos_embed_fn)

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        ffn_layer_ = None
        if ffn_layer == "mlp":
            log.info("using MLP layer as FFN")
            ffn_layer_ = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            log.info("using SwiGLU layer as FFN")
            ffn_layer_ = SwiGLUFFNFused
        elif ffn_layer == "identity":
            log.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer_ = f
        else:
            raise NotImplementedError

        block_range = range(0, self.fusion_layer) if not reversed else range(depth - self.fusion_layer, depth)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        modal: block_class(
                            dim=embed_dim,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            proj_bias=proj_bias,
                            ffn_bias=ffn_bias,
                            drop_path=dpr[i],
                            norm_layer=norm_layer,
                            act_layer=act_layer,
                            ffn_layer=ffn_layer_,
                            init_values=init_values,
                            depth=i,
                        )
                        for modal in self.modals
                    }
                )
                for i in block_range
            ]
        )

        fusion_block_range = range(self.fusion_layer, depth) if not reversed else range(0, depth - self.fusion_layer)
        if self.fusion_type == "vanilla":
            self.fusion_blocks = nn.ModuleList(
                [
                    block_class(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        drop_path=dpr[i],
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                        ffn_layer=ffn_layer_,
                        init_values=init_values,
                        depth=i,
                    )
                    for i in fusion_block_range
                ]
            )

            self.norm = norm_layer(embed_dim)
        elif self.fusion_type == "bottleneck":
            self.fusion_blocks = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            modal: block_class(
                                dim=embed_dim,
                                num_heads=num_heads,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                proj_bias=proj_bias,
                                ffn_bias=ffn_bias,
                                drop_path=dpr[i],
                                norm_layer=norm_layer,
                                act_layer=act_layer,
                                ffn_layer=ffn_layer_,
                                init_values=init_values,
                                depth=i,
                            )
                            for modal in self.modals
                        }
                    )
                    for i in fusion_block_range
                ]
            )
            self.norm = nn.ModuleDict({modal: norm_layer(embed_dim) for modal in self.modals})
        else:
            raise NotImplementedError

        self.init_weights()
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layers in enumerate(self.blocks):
            for _, layer in layers.items():
                if layer is Block:
                    rescale(layer.attn.proj.weight.data, layer_id + 1)
                    rescale(layer.mlp.fc2.weight.data, layer_id + 1)
                elif layer is DecoderBlock:
                    rescale(layer.self_attn.proj.weight.data, layer_id + 1)
                    rescale(layer.cross_attn.proj.weight.data, layer_id + 1)
                    rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def init_pos_embed(self, pos_embed_fn):
        self.pos_embed_fn = pos_embed_fn
        if pos_embed_fn == "sinusoidal":
            offset = 0
            pos_embeds = []
            for modal in self.modals:
                modal_shape = self.modal_shapes[modal]
                modal_size = self.modal_sizes[modal]
                pos_embeds.append(SinusoidalEmbed(modal_shape, [1 for _ in modal_shape], self.embed_dim, offset))
                offset += modal_size
            self.pos_embeds = nn.ModuleDict({modal: pos_embed for modal, pos_embed in zip(self.modals, pos_embeds)})
        elif (
            pos_embed_fn == "learned"
        ):  # NOTE: Different from DINOv2, we don't add learned positional embedding to cls / register tokens
            pos_embeds = nn.ParameterDict(
                {
                    modal: nn.Parameter(
                        torch.zeros(
                            1,
                            pos_embed_size,
                            self.embed_dim,
                        )
                    )
                    for modal, pos_embed_size in self.pos_embed_sizes.items()
                }
            )
            self.pos_embeds = pos_embeds
        else:
            raise NotImplementedError

    def init_weights(self):
        if self.pos_embed_fn == "learned":
            for _, pos_embed in self.pos_embeds.items():
                nn.init.trunc_normal_(pos_embed, std=0.02)
        if self.register_tokens is not None:
            nn.init.trunc_normal_(self.register_tokens, std=1e-6)
        if self.bottleneck is not None:
            nn.init.trunc_normal_(self.bottleneck, std=1e-2)
        self.apply(init_weights_vit_timm)

    def apply_masks(self, x: torch.Tensor, masks: List[torch.Tensor], concat: bool = True, *args, **kwargs):
        assert len(x.shape) == 3
        all_x = []
        for mask in masks:
            ids_keep = einops.repeat(mask, "b n -> b n c", c=x.shape[-1])
            all_x.append(torch.gather(x, dim=1, index=ids_keep))
        return torch.cat(all_x, dim=0) if concat else all_x

    @abstractmethod
    def pre_embed(self, xs: Dict[str, torch.Tensor], *args, **kwargs):
        raise NotImplementedError

    def embed(self, xs: Dict[str, torch.Tensor], *args, **kwargs):
        if self.pos_embed_fn == "sinusoidal":
            pos_embeds = {
                modal: pos_embed(xs[modal].device).float().unsqueeze(0) for modal, pos_embed in self.pos_embeds.items()
            }
        elif self.pos_embed_fn == "learned":
            pos_embeds = {modal: pos_embed.float() for modal, pos_embed in self.pos_embeds.items()}
        else:
            raise NotImplementedError("Unknown position embedding function")
        for x in xs.values():
            # x is in the shape of (b, n, c)
            assert len(x.shape) == 3 and x.shape[-1] == self.embed_dim
        xs = {modal: x + pos_embeds[modal] for modal, x in xs.items()}
        return xs

    def prepare_tokens(
        self,
        xs: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, List[torch.Tensor]]] = None,
        *args,
        **kwargs,
    ):
        if masks is not None:
            assert len(masks) == self.num_modals
            xs = {modal: self.apply_masks(x, masks[modal], *args, **kwargs) for modal, x in xs.items()}

        if self.register_tokens is not None:
            xs = {
                modal: torch.cat([register_token.expand(x.shape[0], -1, -1), x], dim=1)
                for (modal, x), register_token in zip(xs.items(), self.register_tokens)
            }
        return xs

    def transcode(self, xs: Dict[str, torch.Tensor], *args, **kwargs):
        assert len(xs) == self.num_modals

        xs = {modal: xs[modal] for modal in self.modals}
        split_sizes = [x.shape[1] for _, x in xs.items()]

        for blks in self.blocks:
            xs = {modal: blks[modal](x) for modal, x in xs.items()}

        if self.fusion_type == "vanilla":
            x = torch.cat([x for _, x in xs.items()], dim=1)
            bottleneck = None
            for blk in self.fusion_blocks:
                x = blk(x)
            x_norm = self.norm(x)
            xs = torch.split(x, split_sizes, dim=1)
            xs = {modal: x for modal, x in zip(self.modals, xs)}
            xs_norm = torch.split(x_norm, split_sizes, dim=1)
            xs_norm = {modal: x_norm for modal, x_norm in zip(self.modals, xs_norm)}
        elif self.fusion_type == "bottleneck":
            bottleneck = self.bottleneck.expand(xs[self.modals[0]].shape[0], -1, -1)
            for blks in self.fusion_blocks:
                xs = {modal: blks[modal](torch.cat([bottleneck, x], dim=1)) for modal, x in xs.items()}
                bottleneck = torch.stack([x[:, : self.num_bottlenecks] for _, x in xs.items()], dim=-1).mean(dim=-1)
                xs = {modal: x[:, self.num_bottlenecks :] for modal, x in xs.items()}
            xs_norm = {modal: self.norm[modal](x) for modal, x in xs.items()}
        else:
            raise NotImplementedError

        return xs, xs_norm, bottleneck

    def post_transcode(
        self,
        xs: Dict[str, torch.Tensor],
        xs_norm: Dict[str, torch.Tensor],
        bottleneck: Optional[torch.Tensor],
        *args,
        **kwargs,
    ):
        reg_tokens = {modal: x_norm[:, : self.num_register_tokens] for modal, x_norm in xs_norm.items()}
        patch_tokens = {modal: x_norm[:, self.num_register_tokens :] for modal, x_norm in xs_norm.items()}
        patch_tokens_prenorm = xs

        out = {
            "x_norm_regtokens": reg_tokens,
            "x_norm_patchtokens": patch_tokens,
            "x_prenorm": patch_tokens_prenorm,
            "bottleneck": bottleneck,
        }
        return out

    def forward_features(
        self,
        xs: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, List[torch.Tensor]]] = None,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        xs = self.pre_embed(xs, *args, **kwargs)
        xs = self.embed(xs, *args, **kwargs)
        xs = self.prepare_tokens(xs, masks, *args, **kwargs)
        xs, xs_norm, bottleneck = self.transcode(xs, *args, **kwargs)
        outputs = self.post_transcode(xs, xs_norm, bottleneck, *args, **kwargs)
        return outputs

    def forward(
        self,
        xs: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, List[torch.Tensor]]] = None,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.forward_features(xs, masks, *args, **kwargs)
        return outputs["x_norm_patchtokens"]


class MultimodalDecoder(MultimodalTransformer):
    def __init__(
        self,
        modal_chans: int,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        fusion_type: Literal["bottleneck", "vanilla"] = "vanilla",
        fusion_layer: int = 0,
        num_bottlenecks: int = 4,
        *args,
        **kwargs,
    ):
        block_class = partial(Block, attn_class=MemEffAttention)
        super().__init__(
            *args,
            block_class=block_class,
            norm_layer=norm_layer,
            fusion_type=fusion_type,
            fusion_layer=fusion_layer,
            num_bottlenecks=num_bottlenecks,
            reversed=True,
            **kwargs,
        )

        assert self.num_register_tokens == 0

        self.modal_chans = modal_chans
        self.input_projection = nn.Linear(self.modal_chans, self.embed_dim)  # TODO: one projection per modality
        self.norm = nn.ModuleDict({modal: norm_layer(self.embed_dim) for modal in self.modals})
        super().init_weights()

    def pre_embed(
        self,
        xs: Dict[str, torch.Tensor],
        *args,
        **kwargs,
    ):
        xs = {modal: self.input_projection(x) for modal, x in xs.items()}
        return xs

    def transcode(self, xs: Dict[str, torch.Tensor], *args, **kwargs):
        # Note that dict() keeps the insertion order after Python 3.7
        assert len(xs) == self.num_modals

        xs = {modal: xs[modal] for modal in self.modals}
        split_sizes = [x.shape[1] for _, x in xs.items()]

        if self.fusion_type == "vanilla":
            x = torch.cat([x for _, x in xs.items()], dim=1)
            bottleneck = None
            for blk in self.fusion_blocks:
                x = blk(x)
            xs = torch.split(x, split_sizes, dim=1)
            xs = {modal: x for modal, x in zip(self.modals, xs)}
        elif self.fusion_type == "bottleneck":
            bottleneck = self.bottleneck.expand(xs[self.modals[0]].shape[0], -1, -1)
            for blks in self.fusion_blocks:
                xs = {modal: blks[modal](torch.cat([bottleneck, x], dim=1)) for modal, x in xs.items()}
                bottleneck = torch.stack([x[:, : self.num_bottlenecks] for _, x in xs.items()], dim=-1).mean(dim=-1)
                xs = {modal: x[:, self.num_bottlenecks :] for modal, x in xs.items()}
        else:
            raise NotImplementedError

        for blks in self.blocks:
            xs = {modal: blks[modal](x) for modal, x in xs.items()}

        xs_norm = {modal: self.norm[modal](x) for modal, x in xs.items()}

        return xs, xs_norm, bottleneck

    @abstractmethod
    def post_transcode(self, xs: List[torch.Tensor], *args, **kwargs):
        raise NotImplementedError

    def forward(
        self,
        xs: Dict[str, torch.Tensor],
        *args,
        **kwargs,
    ):
        assert len(xs) == self.num_modals

        xs = self.pre_embed(xs, *args, **kwargs)
        xs = self.embed(xs, *args, **kwargs)
        xs = self.prepare_tokens(xs, *args, **kwargs)
        xs, xs_norm, bottleneck = self.transcode(xs, *args, **kwargs)
        output = self.post_transcode(xs_norm, *args, **kwargs)
        return output

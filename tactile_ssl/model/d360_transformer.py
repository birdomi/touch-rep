# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Transformer architecture for D360 multisensory data.
"""

from functools import partial
from typing import Callable, Optional, List, Literal, Tuple, Dict, Any, Union

import einops
import torch
import torch.nn as nn

from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.model import MultimodalTransformer, MultimodalDecoder
from omegaconf import DictConfig

from .layers import MemEffAttention, PatchEmbed1d, PatchEmbed
from .layers import NestedTensorBlock as Block

log = get_pylogger(__name__)


class D360Transformer(MultimodalTransformer):
    def __init__(
        self,
        use_img: bool,
        use_mic: bool,
        use_imu: bool,
        use_pressure: bool,
        sensor_sizes: Dict[str, Union[int, Tuple[int, int]]],
        sensor_chans: Dict[str, int],
        patch_sizes: Dict[str, int],
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
        init_values: Optional[int] = 1,
        num_register_tokens: int = 0,
        fusion_type: Literal["bottleneck", "vanilla"] = "vanilla",
        fusion_layer: int = 0,
        num_bottlenecks: int = 4,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        normalization: Optional[DictConfig] = None,
        attn_class: nn.Module = MemEffAttention,
    ):
        self.use_img = use_img
        self.use_mic = use_mic
        self.use_imu = use_imu
        self.use_pressure = use_pressure

        self.sensor_sizes = sensor_sizes
        self.sensor_chans = sensor_chans
        self.patch_sizes = patch_sizes

        sensors = self.init_sensors(
            use_img=use_img,
            use_mic=use_mic,
            use_imu=use_imu,
            use_pressure=use_pressure,
        )

        modal_shapes = self.init_modal_shapes(sensors=sensors, sensor_sizes=sensor_sizes, patch_sizes=patch_sizes)

        self.modal_shapes = modal_shapes
        self.sensors = sensors

        super().__init__(
            modals=sensors,
            modal_shapes=self.modal_shapes,
            embed_dim=embed_dim,
            depth=depth,
            block_class=partial(Block, attn_class=attn_class),
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
            fusion_type=fusion_type,
            fusion_layer=fusion_layer,
            num_bottlenecks=num_bottlenecks,
            drop_path_rate=drop_path_rate,
            drop_path_uniform=drop_path_uniform,
        )

        D360Transformer.init_normalization(self, normalization)

        patch_embed = {}
        for sensor in self.sensors:
            if sensor == "img" or sensor == "mic":
                patch_embed[sensor] = PatchEmbed(
                    img_size=list(self.sensor_sizes[sensor]),
                    patch_size=self.patch_sizes[sensor],
                    in_chans=self.sensor_chans[sensor],
                    embed_dim=embed_dim,
                )
            elif sensor == "imu" or sensor == "pressure":
                patch_embed[sensor] = PatchEmbed1d(
                    modal_chans=self.sensor_chans[sensor],
                    modal_lens=self.sensor_sizes[sensor],
                    chunk_size=self.patch_sizes[sensor],
                    embed_dim=embed_dim,
                    padding=0,
                )
            else:
                raise NotImplementedError

        self.patch_embed = nn.ModuleDict(patch_embed)

        self.head = nn.Identity() if head is None else head
        self.init_weights()
        self._rescale_blocks()

    def normalize_img(self, x: torch.Tensor):
        return (x - self.img_avg) / self.img_div

    def normalize_mic(self, x: torch.Tensor):
        x = (x - self.mic_avg) / self.mic_div
        x = torch.clamp(x, self.mic_range[0], self.mic_range[1])
        return x[:, None]

    def normalize_imu(self, x: torch.Tensor):
        return x / self.imu_div

    def normalize_pressure(self, x: torch.Tensor):
        x = (x - self.pressure_avg) / self.pressure_div
        x = torch.clamp(x, self.pressure_range[0], self.pressure_range[1])
        return x

    def pre_embed(self, xs: Dict[str, torch.Tensor], *args, **kwargs):
        xs_embed = {}
        for sensor, x in xs.items():
            x = getattr(self, f"normalize_{sensor}")(x)
            if sensor == "img" or sensor == "mic":
                xs_embed[sensor] = self.patch_embed[sensor](x)
            elif sensor == "imu" or sensor == "pressure":
                x = einops.rearrange(x, "b n c -> b c n") if x.shape[-1] != 1 else x[:, None, :, 0]
                x = self.patch_embed[sensor](x)
                xs_embed[sensor] = einops.rearrange(x, "b c n -> b n c")
            else:
                raise NotImplementedError
        xs_embed = {sensor: x.contiguous() for sensor, x in xs_embed.items()}
        return xs_embed

    @staticmethod
    def init_normalization(transformer, normalization: Optional[DictConfig] = None):
        def handle_exception(e, modal):
            log.error(e)
            log.warning(f"No valid normalization for {modal}. The default normalization is used")

        if transformer.use_img:
            try:
                img_chan = transformer.sensor_chans["img"]
                img_avg = torch.tensor(normalization["img"]["avg"])
                img_div = torch.tensor(normalization["img"]["std"]) * normalization["img"]["div"]

                assert img_chan % len(img_avg) == 0 and len(img_avg) == len(img_div)

                transformer.register_buffer(
                    "img_avg",
                    torch.vstack([img_avg[:, None, None] for _ in range(img_chan // len(img_avg))]),
                    persistent=True,
                )
                transformer.register_buffer(
                    "img_div",
                    torch.vstack([img_div[:, None, None] for _ in range(img_chan // len(img_div))]),
                    persistent=True,
                )
            except Exception as e:
                handle_exception(e, "img")
                transformer.register_buffer(
                    "img_avg", torch.zeros(transformer.sensor_chans["img"], 1, 1), persistent=True
                )
                transformer.register_buffer(
                    "img_div", torch.ones(transformer.sensor_chans["img"], 1, 1), persistent=True
                )

        if transformer.use_mic:
            try:
                transformer.register_buffer("mic_avg", torch.tensor(normalization["mic_fbank"]["avg"]), persistent=True)
                transformer.register_buffer(
                    "mic_div",
                    torch.tensor(normalization["mic_fbank"]["std"]) * normalization["mic_fbank"]["div"],
                    persistent=True,
                )
                transformer.register_buffer(
                    "mic_range", torch.tensor(normalization["mic_fbank"]["range"]), persistent=True
                )
            except Exception as e:
                handle_exception(e, "mic")
                transformer.register_buffer("mic_avg", torch.zeros(transformer.sensor_sizes["mic"][1]), persistent=True)
                transformer.register_buffer(
                    "mic_div", 2 * torch.ones(transformer.sensor_sizes["mic"][1]), persistent=True
                )
                transformer.register_buffer("mic_range", torch.tensor([-32, 32]), persistent=True)

        if transformer.use_imu:
            try:
                transformer.register_buffer("imu_div", torch.tensor(normalization["imu_acc"]["div"]), persistent=True)
                transformer.register_buffer("imu_avg", torch.tensor(0), persistent=True)
            except Exception as e:
                handle_exception(e, "imu")
                transformer.register_buffer("imu_div", torch.tensor(4), persistent=True)
                transformer.register_buffer("imu_avg", torch.tensor(0), persistent=True)

        if transformer.use_pressure:
            try:
                transformer.register_buffer(
                    "pressure_avg", torch.tensor(normalization["pressure"]["avg"]), persistent=True
                )
                transformer.register_buffer(
                    "pressure_div",
                    torch.tensor(normalization["pressure"]["std"]) * normalization["pressure"]["div"],
                    persistent=True,
                )
                transformer.register_buffer(
                    "pressure_range", torch.tensor(normalization["pressure"]["range"]), persistent=True
                )
            except Exception as e:
                handle_exception(e, "pressure")
                if transformer.sensor_chans["pressure"] == 1:
                    transformer.register_buffer("pressure_avg", torch.tensor(0), persistent=True)
                    transformer.register_buffer("pressure_div", torch.tensor(2), persistent=True)
                else:
                    transformer.register_buffer(
                        "pressure_avg", torch.zeros(transformer.sensor_chans["pressure"]), persistent=True
                    )
                    transformer.register_buffer(
                        "pressure_div", 2 * torch.ones(transformer.sensor_chans["pressure"]), persistent=True
                    )
                transformer.register_buffer("pressure_range", torch.tensor([-8, 8]), persistent=True)

        if transformer.use_imu:
            assert len(transformer.imu_div.shape) == 0, "imu_div must be a scalar"

    @staticmethod
    def init_sensors(
        use_img: bool,
        use_mic: bool,
        use_imu: bool,
        use_pressure: bool,
    ):
        sensors = []
        if use_img:
            sensors.append("img")

        if use_mic:
            sensors.append("mic")

        if use_imu:
            sensors.append("imu")

        if use_pressure:
            sensors.append("pressure")

        return sensors

    @staticmethod
    def init_modal_shapes(
        sensors: List[str], sensor_sizes: Dict[str, Union[int, Tuple[int, int]]], patch_sizes: Dict[str, int]
    ):
        modal_shapes = []
        for sensor in sensors:
            sensor_size = sensor_sizes[sensor]
            patch_size = patch_sizes[sensor]
            if sensor == "img" or sensor == "mic":
                assert (
                    sensor_size[0] % patch_size == 0 and sensor_size[1] % patch_size == 0
                ), f"{sensor} sensor size must be divisible by patch size"
                modal_shapes.append([sensor_size[0] // patch_size, sensor_size[1] // patch_size])
            elif sensor == "imu" or sensor == "pressure":
                assert sensor_size % patch_size == 0, f"{sensor} sensor size must be divisible by patch size"
                modal_shapes.append([sensor_size // patch_size])
            else:
                raise NotImplementedError

        return modal_shapes


class D360Decoder(MultimodalDecoder):
    def __init__(
        self,
        use_img: bool,
        use_mic: bool,
        use_imu: bool,
        use_pressure: bool,
        sensor_sizes: Dict[str, Union[int, Tuple[int, int]]],
        sensor_chans: Dict[str, int],
        patch_sizes: Dict[str, int],
        modal_chans: int,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        fusion_type: Literal["bottleneck", "vanilla"] = "vanilla",
        fusion_layer: int = 0,
        num_bottlenecks: int = 4,
        normalization: Optional[DictConfig] = None,
        *args,
        **kwargs,
    ):
        sensors = D360Transformer.init_sensors(
            use_img=use_img, use_mic=use_mic, use_imu=use_imu, use_pressure=use_pressure
        )
        modal_shapes = D360Transformer.init_modal_shapes(
            sensors=sensors, sensor_sizes=sensor_sizes, patch_sizes=patch_sizes
        )

        super().__init__(
            *args,
            modals=sensors,
            modal_shapes=modal_shapes,
            modal_chans=modal_chans,
            norm_layer=norm_layer,
            fusion_type=fusion_type,
            fusion_layer=fusion_layer,
            num_bottlenecks=num_bottlenecks,
            **kwargs,
        )

        self.sensors = sensors

        self.use_img = use_img
        self.use_mic = use_mic
        self.use_imu = use_imu
        self.use_pressure = use_pressure

        self.sensor_sizes = sensor_sizes
        self.sensor_chans = sensor_chans
        self.patch_sizes = patch_sizes

        decoder_preds = self.init_preds(
            sensors=sensors,
            sensor_chans=sensor_chans,
            patch_sizes=patch_sizes,
            embed_dim=self.embed_dim,
        )
        self.decoder_pred = nn.ModuleDict(decoder_preds)

        D360Transformer.init_normalization(self, normalization)

        super().init_weights()

    def post_transcode(self, xs: Dict[str, torch.Tensor], *args, **kwargs):
        return D360Decoder.decode_outputs(self, xs)

    @staticmethod
    def init_preds(
        sensors: List[str],
        sensor_chans: Dict[str, Union[int, Tuple[int, int]]],
        patch_sizes: Dict[str, int],
        embed_dim: int,
    ):
        decoder_preds = {}
        for sensor in sensors:
            patch_size = patch_sizes[sensor]
            if sensor == "img" or sensor == "mic":
                decoder_preds[sensor] = nn.Linear(
                    embed_dim,
                    patch_size**2 * sensor_chans[sensor],
                )
            elif sensor == "imu" or sensor == "pressure":
                decoder_preds[sensor] = nn.Linear(
                    embed_dim,
                    patch_size * sensor_chans[sensor],
                )
            else:
                raise NotImplementedError

        return decoder_preds

    @staticmethod
    def decode_outputs(decoder: nn.Module, xs: List[torch.Tensor]):
        xs = {sensor: decoder.decoder_pred[sensor](x) for (sensor, x) in xs.items()}
        xs = {sensor: x[:, None] for sensor, x in xs.items()}
        outputs = {}
        for sensor, x in xs.items():
            if sensor == "img":
                outputs[sensor] = x
            elif sensor == "mic" or sensor == "imu" or sensor == "pressure":
                div = decoder.get_buffer(f"{sensor}_div")
                avg = decoder.get_buffer(f"{sensor}_avg")
                outputs[sensor] = x * div + avg
            else:
                raise NotImplementedError
        return outputs


def dit_tinier(
    use_img: bool,
    use_mic: bool,
    use_imu: bool,
    use_pressure: bool,
    sensor_sizes: Dict[str, Union[int, Tuple[int, int]]],
    sensor_chans: Dict[str, int],
    patch_sizes: Dict[str, int],
    depth: int = 8,
    num_register_tokens=0,
    fusion_type: Literal["bottleneck", "vanilla"] = "bottleneck",
    fusion_layer: int = 6,
    num_bottlenecks: int = 4,
    **kwargs,
):
    model = D360Transformer(
        use_img=use_img,
        use_mic=use_mic,
        use_imu=use_imu,
        use_pressure=use_pressure,
        sensor_sizes=sensor_sizes,
        sensor_chans=sensor_chans,
        patch_sizes=patch_sizes,
        embed_dim=96,
        depth=depth,
        num_heads=3,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        fusion_type=fusion_type,
        fusion_layer=fusion_layer,
        num_bottlenecks=num_bottlenecks,
        **kwargs,
    )
    return model


def dit_tiny(
    use_img: bool,
    use_mic: bool,
    use_imu: bool,
    use_pressure: bool,
    sensor_sizes: Dict[str, Union[int, Tuple[int, int]]],
    sensor_chans: Dict[str, int],
    patch_sizes: Dict[str, int],
    depth: int = 12,
    num_register_tokens=0,
    fusion_type: Literal["bottleneck", "vanilla"] = "bottleneck",
    fusion_layer: int = 8,
    num_bottlenecks: int = 4,
    **kwargs,
):
    model = D360Transformer(
        use_img=use_img,
        use_mic=use_mic,
        use_imu=use_imu,
        use_pressure=use_pressure,
        sensor_sizes=sensor_sizes,
        sensor_chans=sensor_chans,
        patch_sizes=patch_sizes,
        embed_dim=192,
        depth=depth,
        num_heads=3,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        fusion_type=fusion_type,
        fusion_layer=fusion_layer,
        num_bottlenecks=num_bottlenecks,
        **kwargs,
    )
    return model


def dit_small(
    use_img: bool,
    use_mic: bool,
    use_imu: bool,
    use_pressure: bool,
    sensor_sizes: Dict[str, Union[int, Tuple[int, int]]],
    sensor_chans: Dict[str, int],
    patch_sizes: Dict[str, int],
    depth: int = 12,
    num_register_tokens=0,
    fusion_type: Literal["bottleneck", "vanilla"] = "bottleneck",
    fusion_layer: int = 8,
    num_bottlenecks: int = 4,
    **kwargs,
):
    model = D360Transformer(
        use_img=use_img,
        use_mic=use_mic,
        use_imu=use_imu,
        use_pressure=use_pressure,
        sensor_sizes=sensor_sizes,
        sensor_chans=sensor_chans,
        patch_sizes=patch_sizes,
        embed_dim=384,
        depth=depth,
        num_heads=6,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        fusion_type=fusion_type,
        fusion_layer=fusion_layer,
        num_bottlenecks=num_bottlenecks,
        **kwargs,
    )
    return model


def dit_base(
    use_img: bool,
    use_mic: bool,
    use_imu: bool,
    use_pressure: bool,
    sensor_sizes: Dict[str, Union[int, Tuple[int, int]]],
    sensor_chans: Dict[str, int],
    patch_sizes: Dict[str, int],
    depth: int = 12,
    num_register_tokens=0,
    fusion_type: Literal["bottleneck", "vanilla"] = "bottleneck",
    fusion_layer: int = 8,
    num_bottlenecks: int = 4,
    **kwargs,
):
    model = D360Transformer(
        use_img=use_img,
        use_mic=use_mic,
        use_imu=use_imu,
        use_pressure=use_pressure,
        sensor_sizes=sensor_sizes,
        sensor_chans=sensor_chans,
        patch_sizes=patch_sizes,
        embed_dim=768,
        depth=depth,
        num_heads=12,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        fusion_type=fusion_type,
        fusion_layer=fusion_layer,
        num_bottlenecks=num_bottlenecks,
        **kwargs,
    )
    return model


def dit_decoder(
    use_img: bool,
    use_mic: bool,
    use_imu: bool,
    use_pressure: bool,
    sensor_sizes: Dict[str, Union[int, Tuple[int, int]]],
    sensor_chans: Dict[str, int],
    patch_sizes: Dict[str, int],
    modal_chans: int,
    depth: int = 12,
    num_heads: int = 12,
    embed_dim: int = 192,
    fusion_type: Literal["bottleneck", "vanilla"] = "bottleneck",
    fusion_layer: int = 8,
    num_bottlenecks: int = 4,
    **kwargs,
):
    return D360Decoder(
        use_img=use_img,
        use_mic=use_mic,
        use_imu=use_imu,
        use_pressure=use_pressure,
        sensor_sizes=sensor_sizes,
        sensor_chans=sensor_chans,
        patch_sizes=patch_sizes,
        modal_chans=modal_chans,
        depth=depth,
        num_heads=num_heads,
        embed_dim=embed_dim,
        mlp_ratio=4,
        ffn_layer="mlp",
        num_register_tokens=0,
        fusion_type=fusion_type,
        fusion_layer=fusion_layer,
        num_bottlenecks=num_bottlenecks,
        **kwargs,
    )
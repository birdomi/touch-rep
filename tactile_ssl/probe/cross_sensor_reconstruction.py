# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Decoders for input reconstruction from learned representations.

This module provides decoder architectures for reconstructing original inputs
(images or time series) from latent representations. These decoders help evaluate
how well the learned representations preserve information from the input space.

The module includes:
- SignalDecoder: Time series reconstruction decoder
"""

import einops
import torch
import torch.nn as nn

from tactile_ssl.model.signal_transformer import SignalTransformer

class SensorSpecificLinear(nn.Module):
    def __init__(self, num_sensors, input_dim, hidden_dim):
        super().__init__()
        self.num_sensors = num_sensors
        self.input_dim = input_dim

        self.W = nn.Parameter(0.02 * torch.randn(num_sensors, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_sensors, hidden_dim))
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, x, sensor_ids):  
        ids = sensor_ids.unique()
        selected_W = self.W[sensor_ids]
        selected_b = self.b[sensor_ids]
        b, n, c = x.shape

        out_ = torch.zeros(b, n, self.W.shape[-1], device=x.device)
        out = torch.bmm(x, selected_W) + selected_b.unsqueeze(1)

        outs =[]

        for id in ids: # sensor-wise decomposition
            # Flatten T and N dimensions for matrix multiplication

            if id == 0: # sensor 1 == xela -> c = 30
                out_ = einops.rearrange(out[sensor_ids==id], "b n (t c) -> b t n c", c=3) 
            if id == 1: # sensor 2 == actionsense dataset
                out_ = out[sensor_ids==id][:, :, :25]
            # Reshape back to (B, T, N, hidden)
            outs.append(out_)
        
            if id > self.num_sensors:
                raise ValueError(f"currently sensor {id} is not implemented")
            
        return outs


# TODO CrossSensor -> MultiSensor
class CrossSensorSignalDecoder(SignalTransformer):
    def __init__(self, input_embed_dim: int = 768, num_sensors=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder_embed = nn.Linear(input_embed_dim, self.embed_dim, bias=True)
        output_dim = self.time_chunk_size * self.in_chans
        self.decoder_pred = SensorSpecificLinear(num_sensors = num_sensors, input_dim=self.embed_dim, hidden_dim=output_dim)
        self.init_weights()

    def forward(self, x, sensor_ids, **kwargs):
        x = self.decoder_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        x = self.decoder_pred(x_norm, sensor_ids)
        return x


def CrossSensorSignalDecoderBase(**kwargs):
    model = CrossSensorSignalDecoder(num_heads=12, mlp_ratio=4, qkv_bias=True, pos_embed_fn="sinusoidal", **kwargs)
    return model
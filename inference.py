# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import torch
import torch.nn as nn
import torch.nn.functional as F
from tactile_ssl.build_encoder import build_encoder

# Example inference script for DIGIT360 SparshX ###########################################################
config = "config/encoder/digit360_sparshx.yaml"
ckpt_path = "checkpoints/d360_sparshx_img_mic_imu_pressure_tiny.pth"

encoder = build_encoder(config, ckpt_path=ckpt_path, device="cuda", mode="eval")

with torch.inference_mode():
    # Dummy input
    tactile_img = torch.randn(1, 6, 224, 224)
    tactile_audio = torch.randn(1, 224, 256)
    tactile_imu = torch.randn(1, 224, 3)
    tactile_pressure = torch.randn(1, 224, 1)

    input_dict = {
        "img": tactile_img.to("cuda"),
        "mic": tactile_audio.to("cuda"),
        "imu": tactile_imu.to("cuda"),
        "pressure": tactile_pressure.to("cuda"),
    }
    # Forward pass
    tactile_rep = encoder(input_dict)

    # organize output
    tactile_embeddings = []
    for k, v in tactile_rep.items():
        print(f"{k}: {v.shape}")
        rep = F.layer_norm(v, (v.shape[-1],))
        tactile_embeddings.append(rep)
    tactile_embeddings = torch.cat(tactile_embeddings, dim=1)

# Example inference script for Xela Sparsh-skin ###############################################################
config = "config/encoder/xela_sparshskin.yaml"
ckpt_path = "checkpoints/xela_sparshskin_tiny.pth"

encoder = build_encoder(config, ckpt_path=ckpt_path, device="cuda", mode="eval")

with torch.inference_mode():
    # Dummy input
    input = torch.randn(1, 100, 368, 6).to("cuda")  # batch, sequence, num_sensors, channels (xyz + position)

    # Forward pass
    tactile_rep = encoder.forward_features(input)

    # organize output
    tactile_embeddings = []
    for k, v in tactile_rep.items():
        print(f"{k}: {v.shape}")
        rep = F.layer_norm(v, (v.shape[-1],))
        tactile_embeddings.append(rep)
    tactile_embeddings = torch.cat(tactile_embeddings, dim=1)

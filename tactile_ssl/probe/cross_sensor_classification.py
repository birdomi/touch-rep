# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



from typing import List
import torch
import torch.nn as nn
from tactile_ssl.downstream_task.attentive_pooler import AttentivePooler



class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for Adversarial Alignment.
    Forward pass is identity, backward pass negates the gradients and scales by alpha.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)


class CrossSensorAttentiveClassifier(nn.Module):
    """
    Attentive pooling that extracts sensor-invariant features 
    using adversarial alignment (Gradient Reversal Layer).
    """
    def __init__(
        self,
        input_embed_dim: int = 768,
        num_sensors: int = 2,
        num_heads: int = 12,
        depth: int = 1,
        alpha: float = 1.0,
        class_weights = None,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.input_embed_dim = input_embed_dim
        self.alpha = alpha
        self.class_weights = class_weights
        
        # Attentive Pooler
        self.pooler = AttentivePooler(
            num_queries=1,
            embed_dim=input_embed_dim,
            num_heads=num_heads,
            depth=depth,
        )
        
        # Domain Discriminator (Sensor Classifier) for Adversarial Alignment
        self.domain_discriminator = nn.Sequential(
            nn.Linear(input_embed_dim, input_embed_dim // 2),
            nn.ReLU(),
            nn.Linear(input_embed_dim // 2, num_sensors)
        )

    def forward(self, x, sensor_ids=None, return_domain_preds=True):
        """
        Args:
            x: Input feature representation. Shape (B, C) or (B, N, C).
            sensor_ids: Tensor of sensor labels for adversarial alignment. Shape (B,). Optional, unused directly but kept for compatibility.
            return_domain_preds: If True or if model.training is True, returns (features, sensor_preds).
        """
        # Ensure x is 3D (B, N, C) for the attentive pooler
        if x.dim() == 2:
            x = x.unsqueeze(1) # (B, 1, C)
            
        # 1. Attentive Pooling
        # AttentivePooler returns (B, 1, C), we squeeze it to (B, C)
        # Reverse gradients propagating from domain discriminator
        x = grad_reverse(x, self.alpha)
        features = self.pooler(x).squeeze(1)
        
        # 2. Adversarial Domain Classification
        if return_domain_preds:
            sensor_preds = self.domain_discriminator(features)
            
        return sensor_preds

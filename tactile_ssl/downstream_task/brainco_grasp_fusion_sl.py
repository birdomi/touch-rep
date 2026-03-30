"""
Fusion grasp detection module — ResNet18 (vision) + BraincoTransformer (tactile).

Architecture:
  Tactile branch:
    BraincoTransformer → AttentivePooler → (B, W, embed_dim)
    → temporal mean pool  → (B, embed_dim)

  Vision branch:
    ResNet18 (ImageNet pretrained, FC removed)
    per-frame features → (B, T, 512)
    → temporal mean pool  → (B, 512)

  Fusion:
    concat → (B, embed_dim + 512)
    → LayerNorm → Linear(2)
"""

import io
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt

from tactile_ssl.downstream_task.brainco_grasp_sl import BraincoGraspDetectionSLModule
from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)


class BraincoGraspFusionSLModule(BraincoGraspDetectionSLModule):
    """Fusion of ResNet18 vision features and BraincoTransformer tactile features."""

    def __init__(
        self,
        pretrained_vision: bool = True,
        freeze_vision: bool = False,
        **kwargs,   # passed to BraincoGraspDetectionSLModule
    ):
        super().__init__(**kwargs)

        # ── Vision branch (ResNet18, FC removed) ──────────────────────────
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained_vision else None
        backbone = tvm.resnet18(weights=weights)
        self.vision_backbone = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 512, 1, 1)
        self.vision_feat_dim = 512

        if freeze_vision:
            self.vision_backbone.requires_grad_(False)

        # ── Fusion head ────────────────────────────────────────────────────
        tactile_dim = self.model_encoder.embed_dim
        fused_dim   = tactile_dim + self.vision_feat_dim

        # Replace existing classifier with a fusion head
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 2),
        )

    # ── forward ───────────────────────────────────────────────────────────

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Args:
            batch["sensor"]       : (B, W, num_sensors, 4)
            batch["sensor_poses"] : (B, W, 10, 6)
            batch["frames"]       : (B, T, C, H, W)
        Returns:
            logits: (B, 2)
        """
        # ── tactile branch ─────────────────────────────────────────────────
        sensor       = batch["sensor"]
        sensor_poses = batch["sensor_poses"]
        mask         = batch.get("mask", None)

        tactile_tokens = self.encode(sensor, sensor_poses, mask)   # (B, W, embed_dim)
        if self.train_encoder:
            tactile_feat = self.classifier(tactile_tokens)         # uses parent's classifier as temporal model
        else:
            tactile_feat_seq = self.classifier(tactile_tokens.detach())  # (B, 2) — not what we want

        # Re-derive temporal embedding properly
        with torch.no_grad() if not self.train_encoder else torch.enable_grad():
            tactile_emb = tactile_tokens if self.train_encoder else tactile_tokens.detach()

        # Mean pool over windows  →  (B, embed_dim)
        tactile_emb = tactile_emb.mean(dim=1)

        # ── vision branch ─────────────────────────────────────────────────
        frames = batch["frames"]                    # (B, T, C, H, W)
        B, T, C, H, W = frames.shape
        vision_feat = self.vision_backbone(frames.view(B * T, C, H, W))   # (B*T, 512, 1, 1)
        vision_feat = vision_feat.view(B, T, self.vision_feat_dim).mean(dim=1)  # (B, 512)

        # ── fusion ─────────────────────────────────────────────────────────
        fused  = torch.cat([tactile_emb, vision_feat], dim=-1)  # (B, embed_dim + 512)
        logits = self.fusion_head(fused)                         # (B, 2)
        return logits

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        logits = self.forward(batch)
        labels = batch["label"]
        loss   = self.loss_fn(logits, labels)
        preds  = logits.argmax(dim=-1).detach()
        accuracy = (preds == labels).float().mean()
        return {"loss": loss, "accuracy": accuracy.item(), "preds": preds, "labels": labels}

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)

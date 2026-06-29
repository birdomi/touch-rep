"""BraincoAngleGraspSLModule: grasp prediction with AngleTransformer encoder."""

from functools import partial
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .brainco_grasp_sl import BraincoGraspDetectionSLModule


class BraincoAngleGraspSLModule(BraincoGraspDetectionSLModule):
    """Grasp success/fail prediction using AngleTransformer.

    Reads ``joint_contact`` and ``finger_angles`` from the batch instead of
    ``sensor`` and ``sensor_poses``.

    Batch keys:
        joint_contact  : (B, W, 10, 4)  — tactile contact per sensor
        finger_angles  : (B, W, 10, 4)  — finger angle encoding
        label          : (B,)  long
    """

    def encode(self, joint_contact: torch.Tensor, finger_angles: torch.Tensor,
               mask=None) -> torch.Tensor:
        """Encode windowed angle+contact data through AngleTransformer.

        Args:
            joint_contact : (B, W, 10, 4)
            finger_angles : (B, W, 10, 4)
            mask          : optional (B, W) bool

        Returns:
            window_tokens : (B, W, embed_dim)
        """
        B, W, N, C = joint_contact.shape

        xs  = joint_contact.view(B * W, 1, N, C)                  # (B*W, T=1, 10, 4)
        pos = finger_angles.view(B * W, 1, finger_angles.shape[2],
                                 finger_angles.shape[3])           # (B*W, T=1, 10, 4)

        ctx = torch.no_grad() if not self.train_encoder else torch.enable_grad()
        with ctx:
            out      = self.model_encoder.forward_features(xs, pos)
            x_tokens = out["x_tokens"]                             # (B*W, reg+N, D)

        pooled        = self.pooler(x_tokens)                      # (B*W, 1, D)
        window_tokens = pooled.view(B, W, -1)                      # (B, W, D)

        if mask is not None:
            window_tokens = window_tokens * mask.unsqueeze(-1).float()

        return window_tokens

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        joint_contact = batch["joint_contact"]   # (B, W, 10, 4)
        finger_angles = batch["finger_angles"]   # (B, W, 10, 4)
        mask          = batch.get("mask", None)

        embeddings = self.encode(joint_contact, finger_angles, mask)   # (B, W, D)

        if self.train_encoder:
            logits = self.classifier(embeddings)
        else:
            logits = self.classifier(embeddings.detach())
        # print(logits.shape)

        return logits

    def on_validation_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.val_preds.append(outputs["preds"])
        self.val_labels.append(outputs["labels"])
        self.log_metrics(outputs, trainer_instance.global_val_step, trainer_instance, "val")

        preds      = outputs["preds"]
        labels     = outputs["labels"]
        ep_paths   = batch.get("episode_path",       [None] * len(preds))
        win_starts = batch.get("window_start_frame", [None] * len(preds))
        contact    = batch["joint_contact"].detach().cpu()

        for i in range(len(preds)):
            if preds[i] != labels[i]:
                ws = win_starts[i]
                self.val_failed_samples.append({
                    "sensor":              contact[i],
                    "label":               labels[i].item(),
                    "pred":                preds[i].item(),
                    "episode_path":        ep_paths[i] if isinstance(ep_paths[i], str) else None,
                    "window_start_frame":  ws.item() if hasattr(ws, "item") else ws,
                })

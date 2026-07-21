"""BrainCo grasp prediction with an AngleTransformer-style dual encoder."""

from functools import partial
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .brainco_grasp_sl import BraincoGraspDetectionSLModule


class BraincoAngleGraspSLModule(BraincoGraspDetectionSLModule):
    """Grasp success/fail prediction using AngleTransformer.

    Reads ``joint_contact`` plus either ``finger_angles`` or ``finger_xyz``.

    Batch keys:
        joint_contact  : (B, W, 10, 4)  — tactile contact per sensor
        finger_angles  : (B, W, 10, 4)  — optional finger angle encoding
        finger_xyz     : (B, W, 10, 3)  — optional wrist-local fingertip XYZ
        label          : (B,)  long
    """

    def encode(self, joint_contact: torch.Tensor, position_input: torch.Tensor,
               mask=None) -> torch.Tensor:
        """Encode windowed contact and angle/XYZ data through AngleTransformer.

        Args:
            joint_contact : (B, W, 10, 4)
            position_input : (B, W, 10, C_pos)
            mask          : optional (B, W) bool

        Returns:
            window_tokens : (B, W, embed_dim)
        """
        B, W, N, C = joint_contact.shape

        xs  = joint_contact.view(B * W, 1, N, C)                  # (B*W, T=1, 10, 4)
        pos = position_input.view(B * W, 1, position_input.shape[2],
                                  position_input.shape[3])

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
        if "finger_xyz" in batch:
            position_input = batch["finger_xyz"]
        else:
            position_input = batch["finger_angles"]
        mask          = batch.get("mask", None)

        embeddings = self.encode(joint_contact, position_input, mask)   # (B, W, D)

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

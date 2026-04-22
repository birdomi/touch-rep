"""
BrainCo grasp detection SL module for concatenated world-frame sensor input.

This module is designed for `BraincoGraspDetectionWorldDataset`, where
`sensor` already contains tactile(4) + world xyz(3). Only `sensor` is passed
to the encoder; `sensor_poses` is ignored.
"""

from functools import partial
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tactile_ssl.downstream_task.sl_module import SLModule
from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)


class BraincoCatGraspDetectionSLModule(SLModule):
    """Supervised grasp detection using concatenated BrainCo world-frame input."""

    def __init__(
        self,
        model_encoder: nn.Module,
        model_task: nn.Module,
        optim_cfg: partial,
        scheduler_cfg: Optional[partial],
        checkpoint_encoder: Optional[str] = None,
        checkpoint_task: Optional[str] = None,
        train_encoder: bool = False,
        encoder_type: str = "jepa",
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        lora_target_modules: tuple = ("qkv", "proj"),
    ):
        super().__init__(
            model_encoder=model_encoder,
            model_task=model_task,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            checkpoint_encoder=checkpoint_encoder,
            checkpoint_task=checkpoint_task,
            train_encoder=train_encoder,
            encoder_type=encoder_type,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )
        self.val_label_gt = []
        self.val_label_pred = []
        self.last_val_metrics = {}
        self.best_val_metrics = {}

    def encode(self, sensor: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode windowed tactile+world-position input.

        Args:
            sensor: (B, W, N, C)
            mask:   (B, W) optional validity mask

        Returns:
            window_tokens: (B, G, D), where G = W // sequence_length
        """
        B, W, N, C = sensor.shape
        T = self.model_encoder.sequence_length
        assert W % T == 0, f"Window size {W} must be divisible by sequence_length {T}"
        G = W // T

        sensor_input = sensor.contiguous().view(B * G, T, N, C)
        empty_pos = sensor.new_zeros(B * G, T, N, 0)

        with torch.no_grad() if not self.train_encoder else torch.enable_grad():
            if not self.train_encoder:
                self.model_encoder.eval()
            out = self.model_encoder.forward_features(sensor_input, empty_pos)

        if out["x_norm_regtokens"].shape[1] > 0:
            group_tokens = out["x_norm_regtokens"].mean(dim=1)
        else:
            group_tokens = out["x_norm_patchtokens"].mean(dim=1)

        window_tokens = group_tokens.view(B, G, -1)

        if mask is not None:
            mask_grouped = mask.view(B, G, T).any(dim=-1).float()
            window_tokens = window_tokens * mask_grouped.unsqueeze(-1)

        return window_tokens

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        sensor = batch["sensor"]
        mask = batch.get("mask", None)
        embeddings = self.encode(sensor, mask=mask)

        if self.train_encoder:
            return self.model_task(embeddings)
        return self.model_task(embeddings.detach())

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        logits_pred = self.forward(batch)
        labels_gt = batch["label"].long()
        loss = F.cross_entropy(logits_pred, labels_gt)

        labels_pred = logits_pred.argmax(dim=-1).detach()
        accuracy = (labels_pred == labels_gt).float().mean()

        return {
            "loss": loss,
            "accuracy": accuracy.item(),
            "logits_pred": logits_pred.detach(),
            "label_pred": labels_pred,
            "label_gt": labels_gt.detach(),
        }

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        return self.training_step(batch, batch_idx)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        return self.training_step(batch, batch_idx)

    def log_metrics(self, outputs: Dict[str, Any], step: int, trainer_instance=None, label: str = "train"):
        if trainer_instance is not None and trainer_instance.should_log:
            trainer_instance.wandb.log(
                {
                    f"{label}/loss": outputs["loss"],
                    f"global_{label}_step": step,
                }
            )
            trainer_instance.wandb.log(
                {
                    f"{label}/accuracy": outputs["accuracy"],
                    f"global_{label}_step": step,
                }
            )

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.log_metrics(outputs, trainer_instance.global_step, trainer_instance)

    def on_validation_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.val_label_gt.append(outputs["label_gt"])
        self.val_label_pred.append(outputs["label_pred"])
        self.log_metrics(outputs, trainer_instance.global_val_step, trainer_instance, "val")

    def on_validation_epoch_end(self, trainer_instance=None):
        if not self.val_label_gt:
            return

        gt = torch.cat(self.val_label_gt, dim=0)
        pred = torch.cat(self.val_label_pred, dim=0)
        acc = (pred == gt).float().mean().item()
        gt_pos = (gt == 1)
        pred_pos = (pred == 1)
        tp = (gt_pos & pred_pos).sum().item()
        fp = ((~gt_pos) & pred_pos).sum().item()
        fn = (gt_pos & (~pred_pos)).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

        self.last_val_metrics = {
            "accuracy": acc,
            "f1": f1,
        }
        if (not self.best_val_metrics) or (f1 > self.best_val_metrics.get("f1", float("-inf"))):
            self.best_val_metrics = {
                "accuracy": acc,
                "f1": f1,
            }

        print(
            f"[Validation] epoch={getattr(trainer_instance, 'current_epoch', -1)} "
            f"accuracy={acc:.4f} f1={f1:.4f}"
        )

        if trainer_instance is not None and trainer_instance.should_log:
            trainer_instance.wandb.log(
                {
                    "val/overall_accuracy": acc,
                    "val/overall_f1": f1,
                    "epoch": trainer_instance.current_epoch,
                }
            )

        self.val_label_gt = []
        self.val_label_pred = []

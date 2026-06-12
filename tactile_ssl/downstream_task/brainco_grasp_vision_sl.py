from functools import partial
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torchvision import models

from tactile_ssl.algorithm.module import Module
from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)


class ResNet18GraspModule(Module, nn.Module):
    """RGB-only grasp success/fail classifier over sampled episode frames."""

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        optim_cfg: Optional[partial] = None,
        scheduler_cfg: Optional[partial] = None,
        num_classes: int = 2,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optim_partial = optim_cfg or partial(torch.optim.AdamW, lr=3e-4, weight_decay=1e-4)
        self.scheduler_partial = scheduler_cfg

        self.model = self._build_resnet18(pretrained=pretrained, num_classes=self.num_classes)
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                param.requires_grad = name.startswith("fc.")

        self.val_preds = []
        self.val_labels = []
        self.last_val_metrics = {}
        self.best_val_metrics = {}

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        log.info(
            f"ResNet18GraspModule(pretrained={pretrained}, freeze_backbone={freeze_backbone}): "
            f"{trainable:,} / {total:,} trainable params"
        )

    def _build_resnet18(self, pretrained: bool, num_classes: int) -> nn.Module:
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        except AttributeError:
            model = models.resnet18(pretrained=pretrained)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        frames = batch["frames"]
        if frames.ndim != 5:
            raise ValueError(f"Expected frames shape (B, T, C, H, W), got {tuple(frames.shape)}")

        bsz, num_frames, channels, height, width = frames.shape
        logits = self.model(frames.view(bsz * num_frames, channels, height, width))
        logits = logits.view(bsz, num_frames, self.num_classes).mean(dim=1)
        return logits

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        logits = self.forward(batch)
        labels = batch["label"]
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=-1)
        accuracy = (preds == labels).float().mean()
        return {
            "loss": loss,
            "accuracy": accuracy.item(),
            "preds": preds.detach(),
            "labels": labels.detach(),
        }

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)

    def _log_step_metrics(self, outputs: Dict, step: int, trainer_instance=None, label: str = "train") -> None:
        if trainer_instance is None or not trainer_instance.should_log:
            return

        loss = outputs["loss"].item() if torch.is_tensor(outputs["loss"]) else outputs["loss"]
        trainer_instance.wandb.log({
            f"{label}/loss": loss,
            f"{label}/accuracy": outputs["accuracy"],
            f"global_{label}_step": step,
        })

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self._log_step_metrics(outputs, trainer_instance.global_step, trainer_instance, "train")

    def on_validation_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.val_preds.append(outputs["preds"])
        self.val_labels.append(outputs["labels"])
        self._log_step_metrics(outputs, trainer_instance.global_val_step, trainer_instance, "val")

    def on_validation_epoch_end(self, trainer_instance=None):
        if not self.val_preds:
            return

        preds = torch.cat(self.val_preds, dim=0).cpu().numpy()
        labels = torch.cat(self.val_labels, dim=0).cpu().numpy()
        accuracy = float((preds == labels).sum() / len(preds))
        average = "binary" if self.num_classes == 2 else "macro"
        f1 = float(f1_score(labels, preds, average=average, zero_division=0))

        self.last_val_metrics = {"accuracy": accuracy, "f1": f1}
        if accuracy > self.best_val_metrics.get("accuracy", -1.0):
            self.best_val_metrics = {"accuracy": accuracy, "f1": f1}

        log.info("=" * 40)
        log.info("Vision Validation Results")
        log.info(f"Accuracy: {accuracy:.4f}")
        log.info(f"F1 Score: {f1:.4f}")
        log.info("=" * 40)

        if trainer_instance is not None:
            trainer_instance.wandb.log({
                "val/overall_accuracy": accuracy,
                "val/f1_score": f1,
                "epoch": trainer_instance.current_epoch,
            })

        self.val_preds = []
        self.val_labels = []

    def configure_optimizers(
        self,
        num_iterations_per_epoch: int,
        num_epochs: int,
    ) -> Tuple[torch.optim.Optimizer, Optional[Dict], Optional[Dict]]:
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = self.optim_partial(params)

        if self.scheduler_partial is None:
            return optimizer, None, None

        scheduler = self.scheduler_partial(optimizer)
        return (
            optimizer,
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": None,
            },
            None,
        )

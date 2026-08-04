"""ResNet18 slip-detection baseline over RGB windows.

Reports the same metrics as the tactile slip modules — balanced accuracy plus
binary/macro F1 — because the slip class is a small minority of the windows and
raw accuracy cannot be told apart from a majority-class predictor there.
"""

from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torchvision import models

from tactile_ssl.algorithm.module import Module
from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)


class ResNet18VisionWindowModule(Module, nn.Module):
    """RGB-only slip / non-slip classifier over a short window of frames.

    Batch keys:
        frames : (B, T, 3, H, W) — ImageNet-normalized RGB window
        label  : (B,) long       — 0 = non-slip, 1 = slip
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        optim_cfg: Optional[partial] = None,
        scheduler_cfg: Optional[partial] = None,
        num_classes: int = 2,
        num_input_frames: int = 3,
        frame_aggregation: str = "mean",
        dropout: float = 0.0,
        class_weights: Optional[Sequence[float]] = None,
        best_val_metric: str = "balanced_accuracy",
        val_average_epochs: Sequence[int] = (10, 20, 30, 40, 50),
        log_confusion_matrix_image: bool = False,
    ):
        super().__init__()
        if frame_aggregation not in {"mean", "concat", "diff"}:
            raise ValueError(
                "frame_aggregation must be 'mean', 'concat', or 'diff', "
                f"got {frame_aggregation!r}"
            )
        if best_val_metric == "accuracy":
            best_val_metric = "balanced_accuracy"
        if best_val_metric not in {"balanced_accuracy", "f1", "f1_macro"}:
            raise ValueError(
                "best_val_metric must be 'balanced_accuracy', 'f1', or 'f1_macro', "
                f"got {best_val_metric!r}"
            )

        self.num_classes = int(num_classes)
        self.num_input_frames = int(num_input_frames)
        self.frame_aggregation = frame_aggregation
        self.best_val_metric = best_val_metric
        self.log_confusion_matrix_image = bool(log_confusion_matrix_image)
        self.val_average_epochs = tuple(
            sorted({int(epoch) for epoch in val_average_epochs if int(epoch) > 0})
        )
        if not self.val_average_epochs:
            raise ValueError("val_average_epochs must contain at least one positive epoch")

        self.optim_partial = optim_cfg or partial(
            torch.optim.AdamW, lr=3e-4, weight_decay=1e-4
        )
        self.scheduler_partial = scheduler_cfg

        if class_weights is None:
            loss_weights = None
        else:
            loss_weights = torch.as_tensor(class_weights, dtype=torch.float32)
            if loss_weights.ndim != 1 or loss_weights.numel() != self.num_classes:
                raise ValueError(
                    "class_weights must contain one value per class: expected "
                    f"{self.num_classes}, got {loss_weights.tolist()}"
                )
            if not torch.isfinite(loss_weights).all() or (loss_weights <= 0).any():
                raise ValueError("class_weights must be finite and positive")
        self.loss_fn = nn.CrossEntropyLoss(weight=loss_weights)

        self.backbone, feature_dim = self._build_backbone(pretrained)
        if freeze_backbone:
            self.backbone.requires_grad_(False)

        head_dim = {
            "mean": feature_dim,
            "concat": feature_dim * self.num_input_frames,
            # Window mean plus the last-minus-first feature difference, so the
            # head sees within-window motion and not just appearance.
            "diff": feature_dim * 2,
        }[frame_aggregation]
        self.head = nn.Sequential(
            nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(head_dim, self.num_classes),
        )

        self.val_preds = []
        self.val_labels = []
        self.last_val_metrics = {}
        self.val_metrics_by_epoch = {}
        self.epoch_avg_val_metrics = {}
        self.best_val_metrics = {}

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        log.info(
            f"ResNet18VisionWindowModule(pretrained={pretrained}, "
            f"freeze_backbone={freeze_backbone}, aggregation={frame_aggregation}, "
            f"frames={self.num_input_frames}): {trainable:,} / {total:,} trainable params"
        )

    def _build_backbone(self, pretrained: bool) -> Tuple[nn.Module, int]:
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        except AttributeError:
            model = models.resnet18(pretrained=pretrained)

        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim

    # ── forward / steps ───────────────────────────────────────────────────────

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        frames = batch["frames"]
        if frames.ndim != 5:
            raise ValueError(
                f"Expected frames shape (B, T, C, H, W), got {tuple(frames.shape)}"
            )

        bsz, num_frames, channels, height, width = frames.shape
        if (
            self.frame_aggregation in {"concat", "diff"}
            and num_frames != self.num_input_frames
        ):
            raise ValueError(
                f"frame_aggregation={self.frame_aggregation!r} was configured for "
                f"{self.num_input_frames} frames but the batch has {num_frames}"
            )

        features = self.backbone(frames.view(bsz * num_frames, channels, height, width))
        features = features.view(bsz, num_frames, -1)

        if self.frame_aggregation == "mean":
            pooled = features.mean(dim=1)
        elif self.frame_aggregation == "concat":
            pooled = features.flatten(1)
        else:
            pooled = torch.cat(
                (features.mean(dim=1), features[:, -1] - features[:, 0]), dim=-1
            )
        return self.head(pooled)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        logits = self.forward(batch)
        labels = batch["label"]
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=-1).detach()
        accuracy = (preds == labels).float().mean()
        return {
            "loss": loss,
            "accuracy": accuracy.item(),
            "preds": preds,
            "labels": labels.detach(),
        }

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)

    # ── logging hooks ─────────────────────────────────────────────────────────

    def log_metrics(self, outputs: Dict, step: int, trainer_instance=None, label: str = "train"):
        if trainer_instance is None or not trainer_instance.should_log:
            return
        loss = outputs["loss"]
        trainer_instance.wandb.log({
            f"{label}/loss": loss.item() if torch.is_tensor(loss) else loss,
            f"{label}/accuracy": outputs["accuracy"],
            f"global_{label}_step": step,
        })

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.log_metrics(outputs, trainer_instance.global_step, trainer_instance, "train")

    def on_validation_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.val_preds.append(outputs["preds"])
        self.val_labels.append(outputs["labels"])
        self.log_metrics(outputs, trainer_instance.global_val_step, trainer_instance, "val")

    def on_validation_epoch_end(self, trainer_instance=None):
        if not self.val_preds:
            return

        preds = torch.cat(self.val_preds, dim=0).cpu().numpy()
        labels = torch.cat(self.val_labels, dim=0).cpu().numpy()
        class_labels = list(range(self.num_classes))

        balanced_accuracy = float(balanced_accuracy_score(labels, preds))
        f1_average = "binary" if self.num_classes == 2 else "macro"
        f1 = float(f1_score(labels, preds, average=f1_average, zero_division=0))
        f1_macro = float(
            f1_score(labels, preds, labels=class_labels, average="macro", zero_division=0)
        )
        cm = confusion_matrix(labels, preds, labels=class_labels)

        log.info("=" * 40)
        log.info("Vision Slip Validation Results")
        log.info(f"Balanced Accuracy: {balanced_accuracy:.4f}")
        log.info(f"F1 Score: {f1:.4f}")
        log.info(f"F1 Macro: {f1_macro:.4f}")
        log.info(f"Confusion Matrix:\n{cm}")
        log.info("=" * 40)

        self.last_val_metrics = {
            "balanced_accuracy": balanced_accuracy,
            "f1": f1,
            "f1_macro": f1_macro,
        }

        if trainer_instance is not None:
            epoch = int(trainer_instance.current_epoch)
            self.val_metrics_by_epoch[epoch] = dict(self.last_val_metrics)
            average_epochs = sorted(
                recorded_epoch
                for recorded_epoch in self.val_metrics_by_epoch
                if recorded_epoch in self.val_average_epochs
            )
            if tuple(average_epochs) == self.val_average_epochs:
                self.epoch_avg_val_metrics = {
                    key: float(np.mean([
                        self.val_metrics_by_epoch[recorded_epoch][key]
                        for recorded_epoch in average_epochs
                    ]))
                    for key in ("balanced_accuracy", "f1", "f1_macro")
                }
                self.epoch_avg_val_metrics["epochs"] = average_epochs

        selected_metric = self.last_val_metrics[self.best_val_metric]
        if selected_metric > self.best_val_metrics.get(self.best_val_metric, -1.0):
            self.best_val_metrics = dict(self.last_val_metrics)
            log.info(f"New best val {self.best_val_metric}: {selected_metric:.4f}")

        if trainer_instance is not None:
            val_log = {
                "val/balanced_accuracy": balanced_accuracy,
                "val/f1_score": f1,
                "val/f1_macro": f1_macro,
                "epoch": trainer_instance.current_epoch,
            }
            image = self._confusion_matrix_image(cm, class_labels)
            if image is not None:
                val_log["val/confusion_matrix"] = trainer_instance.wandb.Image(image)
            trainer_instance.wandb.log(val_log)

        self.val_preds = []
        self.val_labels = []

    def _confusion_matrix_image(self, cm, class_labels):
        if not self.log_confusion_matrix_image:
            return None

        import io

        import matplotlib.pyplot as plt
        from PIL import Image
        from sklearn.metrics import ConfusionMatrixDisplay

        display_labels = (
            ["non-slip", "slip"] if self.num_classes == 2
            else [str(c) for c in class_labels]
        )
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(cmap="Blues")
        figure = disp.ax_.get_figure()
        figure.set_figwidth(6)
        figure.set_figheight(6)
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close("all")
        return Image.open(buffer)

    # ── optim ─────────────────────────────────────────────────────────────────

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

"""
ResNet18-based grasp detection module — vision only (RGB frames).

Architecture:
  - ResNet18 backbone (ImageNet pretrained, FC removed)
  - Per-frame feature extraction  →  (B, T, 512)
  - Temporal average pooling      →  (B, 512)
  - Linear classifier             →  (B, 2)
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

from tactile_ssl.algorithm.module import Module
from tactile_ssl.model.custom_scheduler import WarmupCosineScheduler
from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)


class ResNet18GraspModule(Module, nn.Module):
    """Binary grasp success/fail classifier using pretrained ResNet18 on RGB frames."""

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        optim_cfg: Optional[partial] = None,
        scheduler_cfg: Optional[partial] = None,
    ):
        super().__init__()

        # ResNet18 backbone — remove the final FC layer
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # (B, 512, 1, 1)
        self.feat_dim = 512

        if freeze_backbone:
            self.backbone.requires_grad_(False)

        self.classifier = nn.Linear(self.feat_dim, 2)
        self.loss_fn = nn.CrossEntropyLoss()

        self.optim_partial = optim_cfg
        self.scheduler_partial = scheduler_cfg

        # Validation state
        self.val_preds: list = []
        self.val_labels: list = []
        self.val_failed_samples: list = []
        self.last_val_metrics: dict = {}

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W)
        Returns:
            logits: (B, 2)
        """
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)
        feats = self.backbone(x).view(B, T, self.feat_dim)  # (B, T, 512)
        feats = feats.mean(dim=1)                            # (B, 512)
        return self.classifier(feats)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        logits = self.forward(batch["frames"])
        labels = batch["label"]
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=-1).detach()
        accuracy = (preds == labels).float().mean()
        return {"loss": loss, "accuracy": accuracy.item(), "preds": preds, "labels": labels}

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)

    # ── batch-end hooks ───────────────────────────────────────────────────

    def _log_metrics(self, outputs, step, trainer, label):
        if trainer is not None and trainer.should_log:
            trainer.wandb.log({
                f"{label}/loss": outputs["loss"].item() if torch.is_tensor(outputs["loss"]) else outputs["loss"],
                f"{label}/accuracy": outputs["accuracy"],
                f"global_{label}_step": step,
            })

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer=None):
        self._log_metrics(outputs, trainer.global_step, trainer, "train")

    def on_validation_batch_end(self, outputs, batch, batch_idx, trainer=None):
        self.val_preds.append(outputs["preds"])
        self.val_labels.append(outputs["labels"])
        self._log_metrics(outputs, trainer.global_val_step, trainer, "val")

        # Collect misclassified samples
        preds  = outputs["preds"]
        labels = outputs["labels"]
        frames = batch["frames"].detach().cpu()          # (B, T, C, H, W)
        ep_paths   = batch.get("episode_path", [None] * len(preds))
        win_starts = batch.get("window_start_frame", [None] * len(preds))
        for i in range(len(preds)):
            if preds[i] != labels[i]:
                ws = win_starts[i]
                self.val_failed_samples.append({
                    "frames": frames[i],                 # (T, C, H, W)  — for fallback
                    "label": labels[i].item(),
                    "pred": preds[i].item(),
                    "episode_path": ep_paths[i] if isinstance(ep_paths[i], str) else None,
                    "window_start_frame": ws.item() if hasattr(ws, "item") else ws,
                })

    def on_validation_epoch_end(self, trainer=None):
        if not self.val_preds:
            return

        preds  = torch.cat(self.val_preds,  dim=0).cpu().numpy()
        labels = torch.cat(self.val_labels, dim=0).cpu().numpy()

        accuracy = (preds == labels).sum() / len(preds)
        f1 = f1_score(labels, preds, average="binary", zero_division=0)
        cm = confusion_matrix(labels, preds, labels=[0, 1])

        log.info("=" * 40)
        log.info("Validation Results (Vision):")
        log.info(f"  Accuracy : {accuracy:.4f}")
        log.info(f"  F1 Score : {f1:.4f}")
        log.info(f"  Confusion Matrix:\n{cm}")
        log.info("=" * 40)

        # Confusion matrix image
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail", "Success"])
        disp.plot(cmap="Blues")
        fig = disp.ax_.get_figure()
        fig.set_figwidth(6)
        fig.set_figheight(6)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close("all")
        cm_img = Image.open(buf)

        self.last_val_metrics = {"accuracy": accuracy, "f1": f1}

        if trainer is not None:
            trainer.wandb.log({
                "val/confusion_matrix": trainer.wandb.Image(cm_img),
                "val/overall_accuracy": accuracy,
                "val/f1_score": f1,
                "epoch": trainer.current_epoch,
            })

            is_last_epoch = (trainer.current_epoch >= trainer.max_epochs - 1)
            if self.val_failed_samples and is_last_epoch:
                # Print misclassified episodes
                log.info(f"--- Misclassified samples ({len(self.val_failed_samples)}) ---")
                ep_errors = Counter()
                for s in self.val_failed_samples:
                    ep_path = s.get("episode_path") or ""
                    ep_name = Path(ep_path).parent.name + "/" + Path(ep_path).name if ep_path else "unknown"
                    ep_errors[ep_name] += 1
                for ep_name, cnt in sorted(ep_errors.items()):
                    log.info(f"  {ep_name}  ({cnt} windows)")
                log.info("---")

                failed_img = self._visualize_failed_samples(self.val_failed_samples)
                trainer.wandb.log({
                    "val/failed_samples": trainer.wandb.Image(failed_img),
                    "epoch": trainer.current_epoch,
                })

        self.val_preds = []
        self.val_labels = []
        self.val_failed_samples = []

    # ── visualization ─────────────────────────────────────────────────────

    def _load_rgb_frame(self, episode_path: str, frame_idx: int):
        import cv2
        try:
            colors_dir = Path(episode_path) / "colors"
            if not colors_dir.exists():
                return None
            img_files = sorted(
                f for f in colors_dir.iterdir()
                if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            frame_idx = min(frame_idx, len(img_files) - 1)
            bgr = cv2.imread(str(img_files[frame_idx]))
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
        except Exception:
            return None

    def _visualize_failed_samples(self, failed_samples, max_samples: int = 16):
        CLASS_NAMES = {0: "Fail", 1: "Success"}
        samples = failed_samples[:max_samples]
        n = len(samples)
        fig, axes = plt.subplots(n, 1, figsize=(6, 3.5 * n), squeeze=False)
        fig.suptitle(f"Misclassified Validation Samples — last epoch  ({n} shown)", fontsize=11)

        for row, s in enumerate(samples):
            ax = axes[row][0]
            gt   = CLASS_NAMES[s["label"]]
            pred = CLASS_NAMES[s["pred"]]
            ep_path   = s.get("episode_path")
            win_start = s.get("window_start_frame") or 0
            # Use middle frame of window
            mid_frame = int(win_start + s["frames"].shape[0] // 2)
            rgb = self._load_rgb_frame(ep_path, mid_frame) if ep_path else None
            if rgb is not None:
                ax.imshow(rgb)
            else:
                ax.text(0.5, 0.5, "No image", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                ax.set_facecolor("#eeeeee")
            ax.set_title(f"GT: {gt}  →  Pred: {pred}", fontsize=9, color="red")
            ax.axis("off")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=80)
        plt.close("all")
        return Image.open(buf)

    # ── optimizer ─────────────────────────────────────────────────────────

    def configure_optimizers(self, num_iterations_per_epoch, num_epochs):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = self.optim_partial(params)
        if self.scheduler_partial is None:
            return optimizer, None, None
        scheduler = self.scheduler_partial(
            optimizer=optimizer,
            T_max=int(num_epochs * num_iterations_per_epoch),
        )
        return optimizer, {"scheduler": scheduler, "interval": "step", "monitor": None}, None

    # ── no-op hooks required by Module interface ──────────────────────────

    def on_fit_start(self, *args, **kwargs): pass
    def on_train_epoch_start(self, *args, **kwargs): pass
    def on_train_epoch_end(self, *args, **kwargs): pass
    def on_train_batch_start(self, *args, **kwargs): pass

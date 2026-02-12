# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Multi-class classification training module using Sparsh-X embeddings for the D360 touch sensor.

This module:
- Requires a D360Transformer (Sparsh-X) as the encoder to handle multi-sensory tactile inputs
- Supports customizable classification tasks through configurable model_task and loss functions
- Provides performance evaluation with confusion matrices

Both the encoder architecture and classification task parameters can be specified in the configuration file.
"""

from functools import partial
from typing import Any, Dict, Optional, List

import numpy as np
import torch

from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.downstream_task.d360_sl import D360SLModule

from tactile_ssl.model.d360_transformer import D360Transformer

import matplotlib.pyplot as plt
from PIL import Image
import io
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


log = get_pylogger(__name__)


class D360ClassificationSLModule(D360SLModule):
    def __init__(
        self,
        model_encoder: D360Transformer,
        model_task: torch.nn.Module,
        optim_cfg: partial,
        scheduler_cfg: Optional[partial],
        label: str,
        loss_fn: partial,
        sensors: Optional[List[str]] = None,
        checkpoint_encoder: Optional[str] = None,
        checkpoint_task: Optional[str] = None,
        train_encoder: bool = False,
        encoder_type: str = "dino",
    ):
        super().__init__(
            model_encoder=model_encoder,
            model_task=model_task,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            sensors=sensors,
            checkpoint_encoder=checkpoint_encoder,
            checkpoint_task=checkpoint_task,
            train_encoder=train_encoder,
            encoder_type=encoder_type,
        )
        self.label = label
        self.loss_fn = loss_fn(self.class_weights)

        self.val_label_pred = []
        self.val_label_gt = []

    @property
    def classes(self):
        return self.model_task.classes

    @property
    def num_classes(self):
        return self.model_task.num_classes

    @property
    def class_weights(self):
        return self.model_task.class_weights

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        xs, xs_gt = self.prepare_data_multimodal(batch)
        labels_gt = xs_gt[self.label]
        logits_pred = self.forward(xs)
        loss = self.loss_fn(logits_pred, labels_gt)
        labels_pred = logits_pred.argmax(dim=-1).detach()
        accuracy = (labels_pred == labels_gt)[labels_gt >= 0].float().mean()
        return {
            "loss": loss,
            "accuracy": accuracy.item(),
            "logits_pred": logits_pred.detach(),
            "label_pred": labels_pred,
            "label_gt": labels_gt,
        }

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)

    def log_metrics(self, outputs, step, trainer_instance=None, label="train"):
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
        self.log_metrics(outputs, trainer_instance.global_step, trainer_instance)  # type: ignore

    def on_validation_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.val_label_gt.append(outputs["label_gt"])
        self.val_label_pred.append(outputs["label_pred"])
        self.log_metrics(outputs, trainer_instance.global_val_step, trainer_instance, "val")  # type: ignore

    def on_validation_epoch_end(self, trainer_instance=None):
        def plot_confusion_matrix(gt, pred, labels, var_name):
            pred = torch.cat(pred, dim=0)
            pred = pred.int().cpu().numpy()
            gt = torch.cat(gt, dim=0).int().cpu().numpy()
            acc = (pred == gt).sum() / pred.shape[0]

            cm = confusion_matrix(gt, pred, normalize="true", labels=labels)
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=labels,
            )
            disp.plot(xticks_rotation="vertical", cmap="Blues", include_values=False)
            fig = disp.ax_.get_figure()
            fig.set_figwidth(10)
            fig.set_figheight(10)
            disp.im_.set_clim(0, 1)
            plt.tight_layout()
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format="png")
            plt.close("all")
            im = Image.open(img_buf)

            if trainer_instance is not None:
                trainer_instance.wandb.log(
                    {
                        f"val/{var_name}_cm": trainer_instance.wandb.Image(im),
                    }
                )
                trainer_instance.wandb.log(
                    {
                        "val/overall_accuracy": acc,
                        "epoch": trainer_instance.current_epoch,
                    }
                )

        plot_confusion_matrix(self.val_label_gt, self.val_label_pred, self.classes, f"{self.label} classification")

        self.val_label_gt = []
        self.val_label_pred = []

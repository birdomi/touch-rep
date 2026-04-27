"""
MultiSensor Grasp Detection SLModule.

Extends BraincoGraspDetectionSLModule to work with MultiSensorBraincoTransformer.
The encoder expects sensor_ids and (B,1,N,40) input; BrainCo data (C=4) is
zero-padded to C=40 inside encode() so the dataloader stays unchanged.

Partial fine-tuning
-------------------
Set ``finetune_modules`` to a list of encoder submodule names to train only
those modules (plus the task head). All other encoder parameters are frozen.

Example config:
    finetune_modules: [sensor_block, patch_embed]   # sensor-specific parts only
    finetune_modules: ~                              # train all (if train_encoder=true)
"""

from typing import List, Optional
import io
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from tactile_ssl.downstream_task.brainco_grasp_sl import BraincoGraspDetectionSLModule, BraincoGraspProbe
from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)

SENSOR_ID_BRAINCO = 0


class MultiSensorGraspDetectionSLModule(BraincoGraspDetectionSLModule):
    """BraincoGraspDetectionSLModule adapted for MultiSensorBraincoTransformer.

    Args:
        xela_num_frames:  Number of XELA frames stacked into channel dim (default 10).
        finetune_modules: List of encoder submodule name patterns to keep trainable.
                          Only effective when train_encoder=True.
                          e.g. ["sensor_block", "patch_embed"]
        **kwargs:         Forwarded to BraincoGraspDetectionSLModule.
    """

    def __init__(
        self,
        xela_num_frames: int = 10,
        finetune_modules: Optional[List[str]] = None,
        mask_finetune: bool = False,
        mask_ratio: float = 0.1,
        num_classes: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mask_finetune = mask_finetune
        self.mask_ratio = mask_ratio
        self.num_classes = num_classes

        # BraincoGraspDetectionSLModule hardcodes num_classes=2; override when needed
        if num_classes != 2:
            embed_dim = self.model_encoder.embed_dim
            num_heads = getattr(self.model_encoder, "num_heads", 12)
            self.classifier = BraincoGraspProbe(
                num_classes=num_classes,
                embed_dim=embed_dim,
                num_heads=num_heads,
            )
            log.info(f"Classifier overridden: num_classes={num_classes}")

        if finetune_modules is not None and self.train_encoder:
            self._apply_partial_finetune(finetune_modules)

    def _apply_partial_finetune(self, finetune_modules: List[str]):
        """Freeze entire encoder, then selectively unfreeze listed submodules and LoRA params."""
        # Step 1: freeze everything
        self.model_encoder.requires_grad_(False)

        # Step 2: unfreeze each specified submodule
        unfrozen_params = 0
        for mod_name in finetune_modules:
            try:
                submod = self.model_encoder
                for part in mod_name.split("."):
                    submod = getattr(submod, part)
                submod.requires_grad_(True)
                n = sum(p.numel() for p in submod.parameters())
                unfrozen_params += n
                log.info(f"Partial finetune: unfrozen encoder.{mod_name}  ({n:,} params)")
            except AttributeError:
                log.warning(f"Partial finetune: '{mod_name}' not found in encoder, skipping")

        # Step 3: always keep LoRA params trainable if present
        lora_params = 0
        for name, param in self.model_encoder.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)
                lora_params += param.numel()
        if lora_params:
            log.info(f"Partial finetune: also keeping {lora_params:,} LoRA params trainable")

        total_enc = sum(p.numel() for p in self.model_encoder.parameters())
        log.info(
            f"Partial finetune: {unfrozen_params + lora_params:,} / {total_enc:,} encoder params trainable"
        )

    def encode(self, sensor, sensor_poses, wrist_poses=None, mask=None):
        """Encode windowed BrainCo tactile data through MultiSensorBraincoTransformer.

        Supports both single-timestep (sequence_length=1) and multi-timestep
        (sequence_length=T>1) encoders.  When T>1 the W window frames are grouped
        into (W // T) temporal chunks; each chunk is processed jointly so the
        encoder can capture intra-window temporal dynamics.

        Args:
            sensor:       (B, W, N, C)   C=4 for BrainCo grasp data
            sensor_poses: (B, W, N, 3)   fingertip 3D position (wrist-local)
            wrist_poses:  (B, W, 2, 9)   wrist poses [translation(3)+rot6d(6)],
                          or None to fall back to learned register token
            mask:         (B, W) boolean mask for valid windows

        Returns:
            window_tokens: (B, W // T, embed_dim)  where T = encoder sequence_length
        """
        B, W, N, C = sensor.shape
        _, _, N_pos, _ = sensor_poses.shape
        T = self.model_encoder.sequence_length   # 1 for standard, 5 for temporal
        # print(B, W, N, C, T)

        assert W % T == 0, (
            f"Window size {W} must be divisible by encoder sequence_length {T}"
        )
        G = W // T   # number of temporal groups per sample

        sensor_input = sensor.view(B * G, T, N, C)
        poses_input  = sensor_poses.view(B * G, T, N_pos, 3)

        wrist_input = None
        if wrist_poses is not None:
            wrist_input = wrist_poses.view(B * G, T, 2, 9)

        # Pad channels from in_chans → max_channels (e.g. 4 → 40)
        # if C < self._max_channels:
        #     sensor_input = F.pad(sensor_input, (0, self._max_channels - C))

        # All samples are BrainCo (sensor_id=0)
        sensor_ids = torch.zeros(B * G, dtype=torch.long, device=sensor.device)

        # mask_finetune: 학습 시에만 랜덤하게 fingertip 센서를 mask token으로 교체
        finetune_mask_ratio = self.mask_ratio if (self.mask_finetune and self.training) else None

        with torch.no_grad() if not self.train_encoder else torch.enable_grad():
            self.model_encoder.eval()
            out = self.model_encoder.forward_features(
                sensor_input, poses_input,
                wrist_poses=wrist_input,
                sensor_ids=sensor_ids,
                finetune_mask_ratio=finetune_mask_ratio,
            )
            x_tokens = out["x_tokens"]   # (B*G, num_tokens, embed_dim)

        pooled_tokens = self.pooler(x_tokens)          # (B*G, 1, embed_dim)
        window_tokens = pooled_tokens.view(B, G, -1)   # (B, G, embed_dim)

        if mask is not None:
            mask_grouped = mask.view(B, G, T).any(dim=-1).float()  # (B, G)
            window_tokens = window_tokens * mask_grouped.unsqueeze(-1)

        return window_tokens

    def forward(self, batch):
        sensor       = batch["sensor"]
        sensor_poses = batch["sensor_poses"]
        wrist_poses  = batch.get("wrist_poses", None)
        mask         = batch.get("mask", None)

        embeddings = self.encode(sensor, sensor_poses, wrist_poses=wrist_poses, mask=mask)

        if self.train_encoder:
            logits = self.classifier(embeddings)
        else:
            logits = self.classifier(embeddings.detach())

        return logits

    def on_validation_epoch_end(self, trainer_instance=None):
        # binary case → 부모 클래스가 처리
        if self.num_classes == 2:
            super().on_validation_epoch_end(trainer_instance)
            return

        # multi-class case
        if not self.val_preds:
            return

        preds  = torch.cat(self.val_preds,  dim=0).cpu().numpy()
        labels = torch.cat(self.val_labels, dim=0).cpu().numpy()

        accuracy = float((preds == labels).sum() / len(preds))
        f1       = float(f1_score(labels, preds, average="macro", zero_division=0))

        class_labels = list(range(self.num_classes))
        cm = confusion_matrix(labels, preds, labels=class_labels)

        if self._in_test:
            log.info("=" * 40)
            log.info("Test Results (best val weights):")
            log.info(f"Accuracy: {accuracy:.4f}  Macro-F1: {f1:.4f}")
            log.info(f"Confusion Matrix:\n{cm}")
            log.info("=" * 40)
            self.last_test_metrics = {"accuracy": accuracy, "f1": f1}
            if trainer_instance is not None:
                trainer_instance.wandb.log({
                    "test/overall_accuracy": accuracy,
                    "test/f1_score": f1,
                })
            self.val_preds  = []
            self.val_labels = []
            return

        log.info("=" * 40)
        log.info("Validation Results:")
        log.info(f"Accuracy: {accuracy:.4f}  Macro-F1: {f1:.4f}")
        log.info(f"Confusion Matrix:\n{cm}")
        log.info("=" * 40)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(cmap="Blues")
        fig = disp.ax_.get_figure()
        fig.set_figwidth(8)
        fig.set_figheight(8)
        plt.tight_layout()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")
        plt.close("all")

        self.last_val_metrics = {"accuracy": accuracy, "f1": f1}
        if accuracy > self.best_val_metrics.get("accuracy", -1.0):
            self.best_val_metrics = {"accuracy": accuracy, "f1": f1}
            import copy
            self._best_state_dict = copy.deepcopy(
                {k: v.cpu() for k, v in self.state_dict().items()}
            )
            log.info(f"New best val accuracy: {accuracy:.4f} — weights saved.")

        if trainer_instance is not None:
            trainer_instance.wandb.log({
                "val/accuracy": accuracy,
                "val/f1_macro": f1,
            })

        self.val_preds  = []
        self.val_labels = []

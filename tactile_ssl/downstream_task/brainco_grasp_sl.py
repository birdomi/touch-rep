"""
Grasp Detection Supervised Learning module for BrainCo tactile sensor.

Uses a pre-trained BraincoTransformer encoder with AttentiveClassifier
for binary grasp success/fail classification.
"""

from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional
import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.downstream_task.sl_module import SLModule
from tactile_ssl.model.brainco_transformer import BraincoTransformer
from tactile_ssl.downstream_task.attentive_pooler import AttentivePooler
from tactile_ssl.model.layers import NestedTensorBlock as Block
from tactile_ssl.model.layers import SinusoidalEmbed

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score

log = get_pylogger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class RotaryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope_base: float = 10000.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {self.head_dim}")

        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        inv_freq = 1.0 / (rope_base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        positions = torch.arange(seq_len, device=q.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("n,d->nd", positions, self.inv_freq)
        cos = freqs.cos().repeat_interleave(2, dim=-1).to(dtype=q.dtype)[None, None, :, :]
        sin = freqs.sin().repeat_interleave(2, dim=-1).to(dtype=q.dtype)[None, None, :, :]
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin

    def forward(self, x: torch.Tensor, attn_bias=None, return_attn=False) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self._rope(q, k)
        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        if return_attn:
            return attn
        return out


class BraincoGraspProbe(nn.Module):
    def __init__(
        self,
        num_classes=2,
        embed_dim=192,
        num_heads=3,
        mlp_ratio=4.0,
        depth=0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        max_seq_len=100,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    drop_path=0.1,
                )
                for _ in range(depth)
            ]
        )
        self.init_std = init_std
        self.probe = nn.Linear(embed_dim, self.num_classes)

        self.pos_embed_fn = SinusoidalEmbed(max_seq_len, 1, embed_dim)
        attn_bias = torch.ones(1, 1, max_seq_len, max_seq_len)
        attn_bias = attn_bias.tril()
        attn_bias.masked_fill_(attn_bias == 0, float("-inf"))
        attn_bias.masked_fill_(attn_bias == 1, 0)
        self.register_buffer("attn_bias", attn_bias)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _causal_attn_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len <= self.attn_bias.shape[-1]:
            return self.attn_bias[:, :, :seq_len, :seq_len]

        attn_bias = torch.ones(1, 1, seq_len, seq_len, device=device)
        attn_bias = attn_bias.tril()
        attn_bias.masked_fill_(attn_bias == 0, float("-inf"))
        attn_bias.masked_fill_(attn_bias == 1, 0)
        return attn_bias

    def forward(self, z):
        b, t, c = z.shape

        pos_embed = self.pos_embed_fn(z.device).float().unsqueeze(0)
        z = z + pos_embed[:, :t, :]

        attn_bias = self._causal_attn_bias(t, z.device)
        for block in self.blocks:
            z = block(z, attn_bias)
        
        y = self.probe(z[:, -1, :])
        return y


class BraincoGraspRoPEProbe(BraincoGraspProbe):
    def __init__(
        self,
        num_classes=2,
        embed_dim=192,
        num_heads=3,
        mlp_ratio=4.0,
        depth=0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        qkv_bias=True,
        max_seq_len=100,
    ):
        super().__init__(
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            depth=0,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            max_seq_len=max_seq_len,
        )
        del self.pos_embed_fn
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    drop_path=0.1,
                    attn_class=RotaryAttention,
                )
                for _ in range(depth)
            ]
        )
        self.apply(self._init_weights)

    def forward(self, z):
        b, t, c = z.shape

        attn_bias = self._causal_attn_bias(t, z.device)
        for block in self.blocks:
            z = block(z, attn_bias)

        y = self.probe(z[:, -1, :])
        return y


class MeanPoolProbe(nn.Module):
    def __init__(
        self,
        num_classes=2,
        embed_dim=192,
        init_std=0.02,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.init_std = init_std
        self.probe = nn.Linear(embed_dim, self.num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        z = z.mean(dim=1)
        return self.probe(z)


class BraincoGraspDetectionSLModule(SLModule):
    """Grasp detection (success/fail) using attentive pooling on BrainCo encoder embeddings."""

    def __init__(
        self,
        model_encoder: BraincoTransformer,
        model_task: nn.Module,
        optim_cfg: partial,
        scheduler_cfg: Optional[partial] = None,
        checkpoint_encoder: Optional[str] = None,
        checkpoint_task: Optional[str] = None,
        train_encoder: bool = False,
        encoder_type: str = "dino",
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        lora_target_modules: tuple = ("qkv", "proj"),
        encoder_warmup_fine_tune_sensor_epochs: int = 0,
        encoder_warmup_fine_tune_sensor_shallow_blocks: Optional[int] = None,
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
        self.encoder_warmup_fine_tune_sensor_epochs = int(encoder_warmup_fine_tune_sensor_epochs)
        self.encoder_warmup_fine_tune_sensor_shallow_blocks = encoder_warmup_fine_tune_sensor_shallow_blocks
        self._encoder_warmup_mode = None
        self._encoder_warmup_epoch_idx = 0
        if self.encoder_warmup_fine_tune_sensor_epochs < 0:
            raise ValueError(
                "encoder_warmup_fine_tune_sensor_epochs must be >= 0, "
                f"got {self.encoder_warmup_fine_tune_sensor_epochs}"
            )
        if self.encoder_warmup_fine_tune_sensor_epochs > 0:
            if not hasattr(self.model_encoder, "_apply_fine_tune_sensor"):
                raise ValueError(
                    "encoder_warmup_fine_tune_sensor_epochs requires an encoder "
                    "with _apply_fine_tune_sensor()."
                )
            if not self.train_encoder:
                log.warning(
                    "encoder_warmup_fine_tune_sensor_epochs is set but train_encoder=False; "
                    "the encoder will stay frozen."
                )
            else:
                # Optimizer groups are built before the first train epoch. Keep the
                # full encoder eligible here, then apply the warmup freeze in the hook.
                self.model_encoder.requires_grad_(True)
                self.model_encoder.train()
                log.info(
                    "Encoder warmup fine_tune_sensor schedule enabled for "
                    f"{self.encoder_warmup_fine_tune_sensor_epochs} epochs."
                )
        self.loss_fn = nn.CrossEntropyLoss()
        self.val_preds = []
        self.val_labels = []
        self.val_failed_samples = []   # list of {sensor, label, pred}
        self.last_val_metrics = {}
        self.best_val_metrics = {}
        self._best_state_dict = None   # saved when val accuracy peaks
        self._in_test = False          # flag: running test eval loop
        self.last_test_metrics = {}
        
        embed_dim = self.model_encoder.embed_dim
        num_heads = getattr(self.model_encoder, "num_heads", 12)
        # num_classes comes from model_task config; default to 2 (binary grasp).
        self.num_classes = int(getattr(model_task, "num_classes", 2))
        self.pooler = AttentivePooler(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        if isinstance(model_task, (BraincoGraspProbe, MeanPoolProbe)):
            self.classifier = model_task
        else:
            self.classifier = BraincoGraspProbe(
                num_classes=self.num_classes,
                embed_dim=embed_dim,
                num_heads=num_heads,
            )
        print(self.classifier)

    def _log_encoder_trainable_params(self, label: str):
        n_trainable = sum(p.numel() for p in self.model_encoder.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model_encoder.parameters())
        log.info(f"{label}: encoder trainable params {n_trainable:,} / {n_total:,}")

    def _set_encoder_warmup_mode(self, mode: str):
        if self._encoder_warmup_mode == mode:
            return

        if mode == "fine_tune_sensor":
            if self.encoder_warmup_fine_tune_sensor_shallow_blocks is not None and hasattr(
                self.model_encoder, "fine_tune_sensor_shallow_blocks"
            ):
                self.model_encoder.fine_tune_sensor_shallow_blocks = int(
                    self.encoder_warmup_fine_tune_sensor_shallow_blocks
                )
            self.model_encoder._apply_fine_tune_sensor()
            self.model_encoder.train()
            self._log_encoder_trainable_params("Encoder warmup fine_tune_sensor")
        elif mode == "full":
            self.model_encoder.requires_grad_(True)
            self.model_encoder.train()
            self._log_encoder_trainable_params("Encoder warmup finished, full fine-tune")
        else:
            raise ValueError(f"Unknown encoder warmup mode: {mode}")

        self._encoder_warmup_mode = mode

    def on_train_epoch_start(self, trainer_instance=None):
        super().on_train_epoch_start(trainer_instance)
        if not self.train_encoder or self.encoder_warmup_fine_tune_sensor_epochs <= 0:
            return

        if trainer_instance is not None:
            epoch = int(trainer_instance.current_epoch)
        else:
            epoch = self._encoder_warmup_epoch_idx
            self._encoder_warmup_epoch_idx += 1

        if epoch < self.encoder_warmup_fine_tune_sensor_epochs:
            self._set_encoder_warmup_mode("fine_tune_sensor")
        else:
            self._set_encoder_warmup_mode("full")

    def encode(self, sensor, sensor_poses, mask=None):
        """Encode windowed tactile data through BraincoTransformer.
        
        Args:
            sensor: (B, num_windows, W, num_sensors, 4) tactile data
            sensor_poses: (B, num_windows, W, 10, 6) fingertip positions
            mask: (B, num_windows) boolean mask for valid windows
            
        Returns:
            embeddings: (B, num_valid_tokens, embed_dim)
        """
        B, W, N, C = sensor.shape
        # Reshape to process all windows at once: (B * num_windows, num_sensors, C)
        sensor_input = sensor.view(B * W, 1, N, C)
        poses_input = sensor_poses.view(B * W, 1, N, 3)

        # Run encoder
        with torch.no_grad() if not self.train_encoder else torch.enable_grad():
            out = self.model_encoder.forward_features(sensor_input, poses_input)
            x_tokens = out["x_tokens"]  # (B*num_windows, num_patches + 1, embed_dim)
            
        # Apply attentive pooler to convert multiple tokens per window into 1 token
        pooled_tokens = self.pooler(x_tokens)  # (B*num_windows, 1, embed_dim)
        
        window_tokens = pooled_tokens.view(B, W, -1)  # (B, num_windows, embed_dim)

        # Apply mask: zero out padding windows
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # (B, num_windows, 1)
            window_tokens = window_tokens * mask_expanded

        return window_tokens

    def forward(self, batch):
        sensor = batch["sensor"]
        sensor_poses = batch["sensor_poses"]
        mask = batch.get("mask", None)

        # Encode: (B, num_windows, embed_dim)
        embeddings = self.encode(sensor, sensor_poses, mask)

        # Apply classifier sequence modeling: expects (B, seq_len, embed_dim)
        if self.train_encoder:
            logits = self.classifier(embeddings)
        else:
            logits = self.classifier(embeddings.detach())

        return logits

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        logits = self.forward(batch)  # (B, num_classes)
        labels = batch["label"]       # (B,)
        
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=-1).detach()
        accuracy = (preds == labels).float().mean()

        return {
            "loss": loss,
            "accuracy": accuracy.item(),
            "preds": preds,
            "labels": labels,
        }

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        # print(batch['sensor_poses'].shape)
        # print(batch['sensor'].shape)

        val_result = self.training_step(batch, batch_idx)
        # print(val_result)
        return val_result

    def log_metrics(self, outputs, step, trainer_instance=None, label="train"):
        if trainer_instance is not None and trainer_instance.should_log:
            trainer_instance.wandb.log({
                f"{label}/loss": outputs["loss"].item() if torch.is_tensor(outputs["loss"]) else outputs["loss"],
                f"global_{label}_step": step,
            })
            trainer_instance.wandb.log({
                f"{label}/accuracy": outputs["accuracy"],
                f"global_{label}_step": step,
            })

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.log_metrics(outputs, trainer_instance.global_step, trainer_instance, "train")

    def on_validation_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        self.val_preds.append(outputs["preds"])
        self.val_labels.append(outputs["labels"])
        self.log_metrics(outputs, trainer_instance.global_val_step, trainer_instance, "val")

        # Collect misclassified samples (sensor data kept on CPU)
        preds = outputs["preds"]   # (B,)
        labels = outputs["labels"] # (B,)
        sensor = batch["sensor"].detach().cpu()  # (B, W, num_sensors, 4)
        ep_paths   = batch.get("episode_path", [None] * len(preds))
        win_starts = batch.get("window_start_frame", [None] * len(preds))
        for i in range(len(preds)):
            if preds[i] != labels[i]:
                ws = win_starts[i]
                self.val_failed_samples.append({
                    "sensor": sensor[i],
                    "label": labels[i].item(),
                    "pred": preds[i].item(),
                    "episode_path": ep_paths[i] if isinstance(ep_paths[i], str) else None,
                    "window_start_frame": ws.item() if hasattr(ws, "item") else ws,
                })

    def on_validation_epoch_end(self, trainer_instance=None):
        if not self.val_preds:
            return

        preds = torch.cat(self.val_preds, dim=0).cpu().numpy()
        labels = torch.cat(self.val_labels, dim=0).cpu().numpy()

        accuracy = (preds == labels).sum() / len(preds)
        class_labels = list(range(self.num_classes))
        f1_average = "binary" if self.num_classes == 2 else "macro"
        f1 = f1_score(labels, preds, average=f1_average, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(labels, preds, labels=class_labels)

        # ── test-mode: just record metrics and return ──────────────────────────
        if self._in_test:
            log.info("="*40)
            log.info(f"Test Results (best val weights):")
            log.info(f"Accuracy: {accuracy:.4f}")
            log.info(f"F1 Score: {f1:.4f}")
            log.info(f"Confusion Matrix:\n{cm}")
            log.info("="*40)
            self.last_test_metrics = {"accuracy": float(accuracy), "f1": float(f1)}
            if trainer_instance is not None:
                trainer_instance.wandb.log({
                    "test/overall_accuracy": accuracy,
                    "test/f1_score": f1,
                })
            self.val_preds = []
            self.val_labels = []
            return

        log.info("="*40)
        log.info(f"Validation Results:")
        log.info(f"Accuracy: {accuracy:.4f}")
        log.info(f"F1 Score: {f1:.4f}")
        log.info(f"Confusion Matrix (Normalized):\n{cm}")
        log.info("="*40)

        display_labels = (
            ["Fail", "Success"] if self.num_classes == 2 else [str(c) for c in class_labels]
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=display_labels,
        )
        disp.plot(cmap="Blues")
        fig = disp.ax_.get_figure()
        fig.set_figwidth(6)
        fig.set_figheight(6)
        plt.tight_layout()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")
        plt.close("all")
        im = Image.open(img_buf)

        self.last_val_metrics = {"accuracy": accuracy, "f1": f1}
        if accuracy > self.best_val_metrics.get("accuracy", -1.0):
            self.best_val_metrics = {"accuracy": float(accuracy), "f1": float(f1)}
            import copy
            self._best_state_dict = copy.deepcopy(
                {k: v.cpu() for k, v in self.state_dict().items()}
            )
            log.info(f"New best val accuracy: {accuracy:.4f} — weights saved.")

        if trainer_instance is not None:
            trainer_instance.wandb.log({
                "val/confusion_matrix": trainer_instance.wandb.Image(im),
                "val/overall_accuracy": accuracy,
                "val/f1_score": f1,
                "epoch": trainer_instance.current_epoch,
            })

            # Print + visualize misclassified samples — only on the last epoch
            is_last_epoch = (trainer_instance.current_epoch >= trainer_instance.max_epochs - 1)
            if self.val_failed_samples and is_last_epoch:
                from collections import Counter
                log.info(f"--- Misclassified samples ({len(self.val_failed_samples)}) ---")
                episode_errors = Counter()
                for s in self.val_failed_samples:
                    ep_path = s.get("episode_path") or ""
                    ep_name = Path(ep_path).parent.name + "/" + Path(ep_path).name if ep_path else "unknown"
                    episode_errors[ep_name] += 1
                for ep_name, cnt in sorted(episode_errors.items()):
                    log.info(f"  {ep_name}  ({cnt} windows misclassified)")
                log.info("---")

                failed_img = self._visualize_failed_samples(self.val_failed_samples)
                trainer_instance.wandb.log({
                    "val/failed_samples": trainer_instance.wandb.Image(failed_img),
                    "epoch": trainer_instance.current_epoch,
                })

        self.val_preds = []
        self.val_labels = []
        self.val_failed_samples = []

    def evaluate_test(self, test_loader, trainer_instance):
        """Restore best-val weights and evaluate on test_loader. Returns {accuracy, f1}."""
        if self._best_state_dict is not None:
            self.load_state_dict(self._best_state_dict)
            log.info("Restored best-val weights for test evaluation.")
        else:
            log.warning("No best_state_dict saved — using final weights for test.")

        self._in_test = True
        trainer_instance.val_loop(self, test_loader)
        self._in_test = False
        return self.last_test_metrics

    def _load_rgb_frame(self, episode_path: str, frame_idx: int):
        """Load a single RGB frame from {episode_path}/colors/. Returns HxWx3 uint8 or None."""
        import cv2
        try:
            colors_dir = Path(episode_path) / "colors"
            if not colors_dir.exists():
                return None
            img_files = sorted(
                f for f in colors_dir.iterdir()
                if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            if frame_idx >= len(img_files):
                frame_idx = len(img_files) - 1
            bgr = cv2.imread(str(img_files[frame_idx]))
            if bgr is None:
                return None
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

    def _visualize_failed_samples(self, failed_samples, max_samples=16):
        """Plot misclassified validation samples.

        Each row:  RGB (mid-window frame) | Left-hand tactile | Right-hand tactile
        Tactile shows normal force (ch0) and proximity (ch3) over time as heatmaps.
        Falls back to a blank panel when colors/ is unavailable.
        """
        if self.num_classes == 2:
            CLASS_NAMES = {0: "Fail", 1: "Success"}
        else:
            CLASS_NAMES = {i: str(i) for i in range(self.num_classes)}
        FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

        samples = failed_samples[:max_samples]
        n = len(samples)

        # 3 columns: RGB | Left tactile | Right tactile
        fig, axes = plt.subplots(n, 3, figsize=(13, 3.2 * n), squeeze=False)
        fig.suptitle(f"Misclassified Validation Samples — last epoch  ({n} shown)", fontsize=12)

        for row, s in enumerate(samples):
            sensor = s["sensor"].numpy()   # (W, 10, 4)  — 5 left + 5 right fingers
            gt   = CLASS_NAMES[s["label"]]
            pred_name = CLASS_NAMES[s["pred"]]
            W = sensor.shape[0]

            # ── col 0: RGB ──────────────────────────────────────────────
            ax_rgb = axes[row][0]
            ep_path = s.get("episode_path")
            win_start = s.get("window_start_frame") or 0
            mid_frame = int(win_start + W // 2)
            rgb = self._load_rgb_frame(ep_path, mid_frame) if ep_path else None
            if rgb is not None:
                ax_rgb.imshow(rgb)
            else:
                ax_rgb.text(0.5, 0.5, "No image", ha="center", va="center",
                            transform=ax_rgb.transAxes, fontsize=9)
                ax_rgb.set_facecolor("#eeeeee")
            ax_rgb.set_title(f"GT: {gt}  →  Pred: {pred_name}", fontsize=9, color="red")
            ax_rgb.axis("off")

            # ── col 1 & 2: tactile heatmaps ─────────────────────────────
            for col, (hand_slice, hand_name) in enumerate([(slice(0, 5), "Left"), (slice(5, 10), "Right")]):
                ax = axes[row][col + 1]
                # normal force (ch0) as the primary signal  (W, 5)
                normal_force = sensor[:, hand_slice, 0].T   # (5, W)
                # proximity  (ch3) drawn as a semi-transparent overlay  (W, 5)
                proximity = sensor[:, hand_slice, 3].T      # (5, W)

                vmin = 0.0
                vmax = max(np.percentile(normal_force, 95), 1e-6)
                im = ax.imshow(normal_force, aspect="auto", origin="lower",
                               vmin=vmin, vmax=vmax, cmap="hot")
                # overlay proximity as contour-like alpha mask
                p_norm = np.clip(proximity / (np.percentile(proximity, 95) + 1e-6), 0, 1)
                ax.imshow(p_norm, aspect="auto", origin="lower",
                          cmap="Blues", alpha=0.3, vmin=0, vmax=1)

                ax.set_yticks(range(5))
                ax.set_yticklabels(FINGER_NAMES, fontsize=7)
                ax.set_xlabel("Time step", fontsize=7)
                ax.set_title(f"{hand_name}  (red=force, blue=prox)", fontsize=8)
                plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=80)
        plt.close("all")
        return Image.open(buf)

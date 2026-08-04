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

    def __init__(self, *args, pos_normalization: Optional[Any] = None, **kwargs):
        """``pos_normalization``: {mean: [...], std: [...]} over the pose channels.

        The pose stream is otherwise fed raw, and for BrainCo that means metres of
        wrist-local fingertip XYZ: std 0.056 overall and 0.009 on the weakest
        channel. A randomly initialised encoder gets almost no per-sample
        variation from an input that small and sits at ln(2) forever -- measured
        directly: the same model reaches train accuracy 1.0 on std-1.0 input and
        0.64 (loss stuck at 0.6931) on std-0.03 input, same steps and lr.

        Off by default. Every checkpoint pretrained before this existed learned on
        the raw pose scale, so switching it on changes the input distribution
        those weights expect -- only enable it for arms trained under the same
        setting.
        """
        super().__init__(*args, **kwargs)
        if pos_normalization is None:
            self.pos_norm = None
        else:
            mean = torch.tensor(list(pos_normalization["mean"]), dtype=torch.float32)
            std = torch.tensor(list(pos_normalization["std"]), dtype=torch.float32)
            if (std <= 0).any():
                raise ValueError(f"pos_normalization std must be positive, got {std}")
            self.register_buffer("pos_norm_mean", mean, persistent=False)
            self.register_buffer("pos_norm_std", std, persistent=False)
            self.pos_norm = True

    def encode(self, joint_contact: torch.Tensor, position_input: torch.Tensor,
               mask=None) -> torch.Tensor:
        """Encode windowed contact and angle/XYZ data through AngleTransformer.

        Args:
            joint_contact : (B, W, 10, 4)
            position_input : (B, W, 10, C_pos)
            mask          : optional (B, W) bool

        Returns:
            window_tokens : (B, num_encoder_windows, embed_dim)
        """
        B, W, N, C = joint_contact.shape
        encoder_sequence_length = int(
            getattr(self.model_encoder, "sequence_length", 1)
        )

        if encoder_sequence_length == 1:
            # Frame-wise encoder: independently encode each downstream frame,
            # then let the task probe pool over the resulting window tokens.
            xs = joint_contact.reshape(B * W, 1, N, C)
            pos = position_input.reshape(
                B * W, 1, position_input.shape[2], position_input.shape[3]
            )
            num_encoder_windows = W
        elif W % encoder_sequence_length == 0:
            # Reuse the temporal encoder on consecutive, non-overlapping
            # windows. For example, a nine-frame downstream sample with a
            # three-frame pretrained encoder produces three task tokens.
            num_encoder_windows = W // encoder_sequence_length
            xs = joint_contact.reshape(
                B * num_encoder_windows, encoder_sequence_length, N, C
            )
            pos = position_input.reshape(
                B * num_encoder_windows,
                encoder_sequence_length,
                position_input.shape[2],
                position_input.shape[3],
            )
        else:
            raise ValueError(
                "Downstream window length must be divisible by the temporal "
                f"encoder sequence_length: got W={W}, "
                f"encoder sequence_length={encoder_sequence_length}"
            )

        encoder_grad_enabled = self.encoder_grad_enabled()
        ctx = torch.enable_grad() if encoder_grad_enabled else torch.no_grad()
        with ctx:
            out      = self.model_encoder.forward_features(xs, pos)

        if self.pooling == "cls":
            # The encoder's own register/CLS token, as trained by DINO.
            pooled = out["x_norm_regtokens"][:, 0]
        else:
            pooled = self.pooler(out["x_tokens"])                  # (B*W, 1, D)
        window_tokens = pooled.reshape(B, num_encoder_windows, -1)

        if mask is not None:
            if mask.ndim != 2 or mask.shape[0] != B:
                raise ValueError(
                    f"Expected mask shape (B, W), got {tuple(mask.shape)}"
                )
            if mask.shape[1] == W:
                mask = mask.reshape(
                    B, num_encoder_windows, encoder_sequence_length
                ).all(dim=-1)
            elif mask.shape[1] != num_encoder_windows:
                raise ValueError(
                    "Mask length must match either the input frame count or "
                    f"the encoder-window count: got {mask.shape[1]}, expected "
                    f"{W} or {num_encoder_windows}"
                )
            window_tokens = window_tokens * mask.unsqueeze(-1).float()

        return window_tokens

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        joint_contact = batch["joint_contact"]   # (B, W, 10, 4)
        if self.sensor_channels is not None:
            joint_contact = joint_contact[..., self.sensor_channels]
        if "finger_xyz" in batch:
            position_input = batch["finger_xyz"]
        else:
            position_input = batch["finger_angles"]
        if self.pos_norm is not None:
            position_input = (
                position_input - self.pos_norm_mean.to(position_input.device)
            ) / self.pos_norm_std.to(position_input.device)
        mask          = batch.get("mask", None)

        embeddings = self.encode(joint_contact, position_input, mask)   # (B, W, D)

        if self.encoder_grad_enabled():
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

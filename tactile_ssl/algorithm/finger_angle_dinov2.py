"""FingerAngleDinov2Module: DINOv2 SSL for finger angle data (single stream).

Input
-----
pos : finger_angles  (B, T, N_fingers=10, C=4) — finger angle vectors only.
No joint_contact stream.
"""

from typing import Any, Dict, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from xformers.ops import fmha

from .brainco_dinov2 import BraincoDINOv2Module


class FingerAngleDinov2Module(BraincoDINOv2Module):
    """DINOv2 SSL module for finger angle pretraining (single stream).

    Simplified AngleDinov2Module: no joint_contact stream, no dual-stream mask
    sampling. Masking and iBOT operate solely on the angle token sequence.

    Batch key: ``finger_angles`` → pos  (B, T, N_fingers, 4).
    """

    def __init__(
        self,
        classification_loss: bool = False,
        classification_num_classes: int = 6,
        classification_target_key: str = "classification_id",
        classification_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_classification_loss = classification_loss
        self.classification_target_key = classification_target_key
        self.classification_loss_weight = classification_loss_weight

        if self.use_classification_loss:
            embed_dim = self.student_encoder_dict["backbone"].embed_dim
            self.classification_head = nn.Linear(embed_dim, classification_num_classes)

    # ── logging ───────────────────────────────────────────────────────────────

    def log_on_batch_end(self, outputs, stage: str = "train", trainer_instance=None):
        super().log_on_batch_end(outputs, stage=stage, trainer_instance=trainer_instance)
        if trainer_instance is None or (stage == "train" and not trainer_instance.should_log):
            return
        step = trainer_instance.step
        for key in ("classification_loss", "classification_accuracy"):
            if key in outputs:
                trainer_instance.wandb.log({
                    f"{stage}/{key}": outputs[key],
                    f"global_{stage}_step": step,
                })

    # ── masking ───────────────────────────────────────────────────────────────

    def sample_masks(self, pos: torch.Tensor):
        """Sample masks for the single angle stream.

        Args:
            pos : (B, T, N_fingers, C)

        Returns:
            global_masks, local_masks, ibot_masks  — same as BraincoDINOv2Module
        """
        return super().sample_masks(pos)

    # ── forward ───────────────────────────────────────────────────────────────

    def _classification_loss(self, cls_tokens: torch.Tensor, labels: torch.Tensor):
        logits = self.classification_head(cls_tokens)
        loss   = F.cross_entropy(logits, labels)
        acc    = (logits.argmax(dim=1) == labels).float().mean()
        return loss, acc

    def forward(
        self,
        pos: torch.Tensor,
        global_masks: torch.Tensor,
        local_masks: torch.Tensor,
        ibot_masks: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        assert global_masks is not None and local_masks is not None

        n_total = global_masks.shape[-1]

        ibot_masks_flat   = ibot_masks.flatten(0, 1)
        ibot_mask_indices = torch.nonzero(ibot_masks_flat).flatten()
        num_ibot_tokens   = len(ibot_mask_indices)

        student_global_dict = self.student_encoder_dict["backbone"].forward_features(
            pos, masks=global_masks, mask_type="tubelet", masktoken_masks=ibot_masks,
        )
        student_local_dict = self.student_encoder_dict["backbone"].forward_features(
            pos, masks=local_masks, mask_type="tubelet",
        )

        student_global_cls_tokens   = student_global_dict["x_norm_regtokens"][:, 0]
        student_local_cls_tokens    = student_local_dict["x_norm_regtokens"][:, 0]
        student_global_patch_tokens = student_global_dict["x_norm_patchtokens"]

        student_global_patch_tokens = einops.rearrange(
            student_global_patch_tokens, "b (t n) c -> (b n) t c", n=n_total,
        )
        student_masked_patch_tokens = student_global_patch_tokens.new_zeros(
            (num_ibot_tokens,
             student_global_patch_tokens.shape[-2],
             student_global_patch_tokens.shape[-1])
        )
        student_masked_patch_tokens.copy_(student_global_patch_tokens[ibot_mask_indices])
        student_masked_patch_tokens = student_masked_patch_tokens.flatten(0, 1)

        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list([
            student_global_cls_tokens.unsqueeze(0),
            student_local_cls_tokens.unsqueeze(0),
            student_masked_patch_tokens.unsqueeze(0),
        ])
        after_head_list = _attn_bias.split(
            self.student_encoder_dict["dino_head"](cat_inputs)
        )
        (
            student_global_cls_tokens_after_head,
            student_local_cls_tokens_after_head,
            student_patch_tokens_after_head,
        ) = (
            after_head_list[0].squeeze(0),
            after_head_list[1].squeeze(0),
            after_head_list[2].squeeze(0),
        )
        student_cls_tokens_after_head = torch.cat(
            [student_global_cls_tokens_after_head,
             student_local_cls_tokens_after_head],
            dim=0,
        )

        with torch.no_grad():
            teacher_global_dict = self.teacher_encoder_dict["backbone"].forward_features(
                pos, masks=global_masks, mask_type="tubelet",
            )
            teacher_global_cls_tokens = teacher_global_dict["x_norm_regtokens"][:, 0]
            teacher_global_cls_tokens = teacher_global_cls_tokens.chunk(self.num_global_masks)
            assert self.num_global_masks == 2
            teacher_global_cls_tokens = torch.cat(
                (teacher_global_cls_tokens[1], teacher_global_cls_tokens[0])
            )

            teacher_global_patch_tokens = teacher_global_dict["x_norm_patchtokens"]
            teacher_global_patch_tokens = einops.rearrange(
                teacher_global_patch_tokens, "b (t n) c -> (b n) t c", n=n_total,
            )
            teacher_masked_patch_tokens = teacher_global_patch_tokens.new_zeros(
                (num_ibot_tokens,
                 student_global_patch_tokens.shape[-2],
                 student_global_patch_tokens.shape[-1])
            )
            teacher_masked_patch_tokens.copy_(teacher_global_patch_tokens[ibot_mask_indices])
            teacher_masked_patch_tokens = teacher_masked_patch_tokens.flatten(0, 1)

            teacher_cls_tokens_after_head = self.teacher_encoder_dict["dino_head"](
                teacher_global_cls_tokens
            )
            teacher_masked_patch_tokens_after_head = self.teacher_encoder_dict["dino_head"](
                teacher_masked_patch_tokens
            )

            if self.centering == "centering":
                teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head,
                    teacher_temp=self.current_teacher_temp,
                ).view(self.num_global_masks, -1, *teacher_cls_tokens_after_head.shape[1:])
                teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                    teacher_masked_patch_tokens_after_head.unsqueeze(0),
                    teacher_temp=self.current_teacher_temp,
                ).squeeze(0)
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                self.ibot_patch_loss.update_center(teacher_masked_patch_tokens_after_head)
            elif self.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head,
                    teacher_temp=self.current_teacher_temp,
                ).view(self.num_global_masks, -1, *teacher_cls_tokens_after_head.shape[1:])
                teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                    teacher_masked_patch_tokens_after_head,
                    teacher_temp=self.current_teacher_temp,
                    n_masked_patches_tensor=torch.tensor(
                        num_ibot_tokens, dtype=int,
                        device=teacher_masked_patch_tokens.device,
                    ),
                )
            else:
                raise NotImplementedError

        n_local_crops_loss_terms  = max(self.num_local_masks * self.num_global_masks, 1)
        n_global_crops_loss_terms = (self.num_global_masks - 1) * self.num_global_masks

        dino_loss = self.dino_loss(
            student_cls_tokens_after_head.chunk(self.num_global_masks + self.num_local_masks),
            teacher_dino_softmaxed_centered_list,
        ) / (n_local_crops_loss_terms + n_global_crops_loss_terms)

        koleo_loss = self.koleo_weight * sum(
            self.koleo_loss(p.squeeze(dim=-2))
            for p in student_global_cls_tokens.chunk(2, dim=1)
        )

        ibot_loss_scale = 1.0 / self.num_global_masks
        patch_loss = ibot_loss_scale * self.ibot_patch_loss(
            student_patch_tokens_after_head, teacher_ibot_softmaxed_centered
        )
        ssl_loss = dino_loss + patch_loss + koleo_loss
        result   = {"ssl_loss": ssl_loss, "loss": ssl_loss}

        if self.use_classification_loss and labels is not None:
            labels_repeated = labels.repeat(self.num_local_masks)
            cls_loss, acc   = self._classification_loss(
                student_local_cls_tokens, labels_repeated
            )
            result["loss"]                    = ssl_loss + self.classification_loss_weight * cls_loss
            result["classification_loss"]     = cls_loss
            result["classification_accuracy"] = acc

        return result

    # ── training / validation ─────────────────────────────────────────────────

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        self.step += 1
        self.generator.manual_seed(self.step)

        pos    = batch["finger_angles"]   # (B, T, N_fingers, 4)
        labels = batch.get(self.classification_target_key)

        if self.use_classification_loss and labels is None:
            raise KeyError(
                f"classification_loss=True requires batch['{self.classification_target_key}']"
            )

        global_masks, local_masks, ibot_masks = self.sample_masks(pos)
        fwd  = self.forward(pos, global_masks, local_masks, ibot_masks, labels=labels)
        loss = fwd["loss"]

        output = {"ssl_loss": fwd["ssl_loss"].item()}
        if "classification_loss" in fwd:
            output["classification_loss"]     = fwd["classification_loss"].item()
            output["classification_accuracy"] = fwd["classification_accuracy"]

        online_probes_loss = 0.0
        if len(self.online_probes) > 0:
            with torch.no_grad():
                teacher_dict  = self.teacher_encoder_dict["backbone"].forward_features(pos)
                cls_embedding = teacher_dict["x_norm_regtokens"].squeeze(1)
                embedding     = teacher_dict["x_norm_patchtokens"]
                embedding     = F.layer_norm(embedding, (embedding.size(-1),))

            for probe in self.online_probes:
                probe_name = str(probe.probe_name)
                if "classification" in probe_name:
                    gt_labels   = batch[probe_name]
                    probe_loss, pred_logits = probe(cls_embedding, target=gt_labels)
                    pred_labels = torch.argmax(pred_logits, dim=1)
                    accuracy    = (pred_labels == gt_labels).float().mean()
                    online_probes_loss += probe_loss
                    output[f"{probe_name}_loss"]     = probe_loss.item()
                    output[f"{probe_name}_accuracy"] = accuracy
                else:
                    raise NotImplementedError(
                        f"Probe '{probe_name}' not supported for FingerAngleDinov2Module"
                    )

        loss += online_probes_loss
        output["loss"]               = loss
        output["online_probes_loss"] = online_probes_loss
        return output

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)

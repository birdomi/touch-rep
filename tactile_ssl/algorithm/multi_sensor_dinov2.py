"""MultiSensorDINOv2Module: BraincoDINOv2 extended for mixed-sensor batches.

Adds ``sensor_id`` routing so that each batch item is embedded and normalised
by the corresponding per-sensor components of MultiSensorBraincoTransformer.

sensor_id=0  →  BrainCo  (in_chans=4, padded to xela_in_chans)
sensor_id=1  →  XELA     (in_chans=xela_num_frames * 4 = 40)

When ``use_grl=True``, a domain-adversarial Gradient Reversal Layer is added:
student CLS tokens are routed through a domain classifier via a GRL, forcing
the encoder to produce sensor-invariant representations (DANN-style).
"""

from typing import Any, Dict, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from .brainco_dinov2 import BraincoDINOv2Module
from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)


# ── Gradient Reversal Layer ────────────────────────────────────────────────────

class _GRLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha, dtype=torch.float32))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (alpha,) = ctx.saved_tensors
        return -alpha.item() * grad_output, None


class GradientReversalLayer(nn.Module):
    """Passes input forward unchanged; reverses and scales gradients on backward."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GRLFunction.apply(x, self.alpha)


# ── Domain Classifier ──────────────────────────────────────────────────────────

class DomainClassifier(nn.Module):
    """Small MLP predicting sensor domain (0=BrainCo, 1=XELA)."""

    def __init__(self, embed_dim: int, num_domains: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── MultiSensorDINOv2Module ────────────────────────────────────────────────────

class MultiSensorDINOv2Module(BraincoDINOv2Module):
    """DINOv2 SSL module for multi-sensor (BrainCo + XELA) joint training.

    The only behavioural difference from BraincoDINOv2Module is that the
    ``sensor_id`` field present in every batch item is extracted and forwarded
    to the encoder's ``forward_features``, enabling per-sensor normalisation
    and embedding.

    Args:
        use_grl:        Enable domain-adversarial GRL training (default False).
        grl_weight:     Gradient reversal scale α (default 1.0).
        grl_hidden_dim: Hidden size of the domain classifier MLP (default 64).
        **kwargs:       Forwarded to BraincoDINOv2Module.
    """

    def __init__(
        self,
        use_grl: bool = False,
        grl_weight: float = 1.0,
        grl_hidden_dim: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_grl = use_grl
        self.grl_weight = grl_weight

        if use_grl:
            embed_dim = self.student_encoder_dict["backbone"].embed_dim
            self.grl = GradientReversalLayer(alpha=grl_weight)
            self.domain_classifier = DomainClassifier(
                embed_dim=embed_dim,
                num_domains=2,
                hidden_dim=grl_hidden_dim,
            )
            log.info(
                f"MultiSensorDINOv2Module: GRL enabled  "
                f"alpha={grl_weight}, embed_dim={embed_dim}, hidden={grl_hidden_dim}"
            )

    def sample_masks(self, x):
        """x/pos 각각 독립 마스크 생성 + full_ibot(sensor+pos) 반환.

        Returns:
            global_masks:     (num_global, B, n_keep_x)
            local_masks:      (num_local,  B, n_keep_x)
            ibot_masks:       (num_global, B, n_keep_x) bool  — sensor 전용
            pos_global_masks: (num_global, B, n_keep_p)
            pos_local_masks:  (num_local,  B, n_keep_p)
            full_ibot:        (num_global, B, n_keep_x+n_keep_p) bool
        """
        global_masks, local_masks, ibot_masks = super().sample_masks(x)
        pos_global_masks, pos_local_masks, _  = super().sample_masks(x)

        n_p = pos_global_masks.shape[-1]
        ibot_ratio = ibot_masks.float().mean()
        pos_ibot = (
            torch.rand(ibot_masks.shape[0], ibot_masks.shape[1], n_p,
                       device=ibot_masks.device) < ibot_ratio
        )
        full_ibot = torch.cat([ibot_masks, pos_ibot], dim=-1)

        return global_masks, local_masks, ibot_masks, pos_global_masks, pos_local_masks, full_ibot

    def forward(
        self,
        xs: torch.Tensor,
        pos: torch.Tensor,
        sensor_ids: torch.Tensor,
        global_masks: torch.Tensor,
        local_masks: torch.Tensor,
        ibot_masks: torch.Tensor,
        pos_global_masks: torch.Tensor,
        pos_local_masks: torch.Tensor,
        full_ibot: torch.Tensor,
    ):
        assert global_masks is not None and local_masks is not None

        n_x     = global_masks.shape[-1]
        n_p     = pos_global_masks.shape[-1]
        n_total = n_x + n_p

        ibot_masks_flat   = full_ibot.flatten(0, 1)            # (B_eff, n_total)
        ibot_mask_indices = torch.nonzero(ibot_masks_flat).flatten()
        num_ibot_tokens   = len(ibot_mask_indices)

        student_global_dict = self.student_encoder_dict["backbone"].forward_features(
            xs, pos, sensor_ids=sensor_ids,
            masks=global_masks, mask_type="tubelet", masktoken_masks=ibot_masks,
            pos_masks=pos_global_masks,
        )
        student_local_dict = self.student_encoder_dict["backbone"].forward_features(
            xs, pos, sensor_ids=sensor_ids,
            masks=local_masks, mask_type="tubelet",
            pos_masks=pos_local_masks,
        )

        student_global_cls_tokens  = student_global_dict["x_norm_regtokens"][:, 0]
        student_local_cls_tokens   = student_local_dict["x_norm_regtokens"][:, 0]
        # patch_tokens: (B_eff, n_total, D) — sensor patch 앞, pos patch 뒤
        student_global_patch_tokens = student_global_dict["x_norm_patchtokens"]


        student_global_patch_tokens = einops.rearrange(
            student_global_patch_tokens,
            "b (t n) c -> (b n) t c",
            n=n_total,
        )
        student_masked_patch_tokens = student_global_patch_tokens.new_zeros(
            (num_ibot_tokens, student_global_patch_tokens.shape[-2],
             student_global_patch_tokens.shape[-1])
        )
        student_masked_patch_tokens.copy_(student_global_patch_tokens[ibot_mask_indices])
        student_masked_patch_tokens = student_masked_patch_tokens.flatten(0, 1)

        from xformers.ops import fmha
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
        ) = (after_head_list[0].squeeze(0),
             after_head_list[1].squeeze(0),
             after_head_list[2].squeeze(0))

        student_cls_tokens_after_head = torch.cat(
            [student_global_cls_tokens_after_head,
             student_local_cls_tokens_after_head],
            dim=0,
        )

        with torch.no_grad():
            teacher_global_dict = self.teacher_encoder_dict["backbone"].forward_features(
                xs, pos, sensor_ids=sensor_ids,
                masks=global_masks, mask_type="tubelet",
                pos_masks=pos_global_masks,
            )
            teacher_global_cls_tokens   = teacher_global_dict["x_norm_regtokens"][:, 0]
            teacher_global_cls_tokens   = teacher_global_cls_tokens.chunk(self.num_global_masks)
            assert self.num_global_masks == 2
            teacher_global_cls_tokens   = torch.cat(
                (teacher_global_cls_tokens[1], teacher_global_cls_tokens[0])
            )

            teacher_global_patch_tokens = teacher_global_dict["x_norm_patchtokens"]
            teacher_global_patch_tokens = einops.rearrange(
                teacher_global_patch_tokens,
                "b (t n) c -> (b n) t c",
                n=n_total,
            )
            teacher_masked_patch_tokens = teacher_global_patch_tokens.new_zeros(
                (num_ibot_tokens, student_global_patch_tokens.shape[-2],
                 student_global_patch_tokens.shape[-1])
            )
            teacher_masked_patch_tokens.copy_(
                teacher_global_patch_tokens[ibot_mask_indices]
            )
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
        return dino_loss + patch_loss + koleo_loss

    def _grl_domain_loss(
        self, x: torch.Tensor, pos: torch.Tensor, sensor_ids: torch.Tensor
    ) -> torch.Tensor:
        """student CLS token → GRL → domain CE loss.

        The GRL multiplies gradients by -alpha before they reach the backbone,
        pushing the encoder to produce indistinguishable sensor-type features.
        """
        cls_tokens = self.student_encoder_dict["backbone"].forward_sensor_cls(
            x, sensor_ids=sensor_ids
        )
        #cls_tokens = student_dict["x_norm_regtokens"][:, 0]   # (B, embed_dim)
        domain_logits = self.domain_classifier(self.grl(cls_tokens))
        return F.cross_entropy(domain_logits, sensor_ids)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        self.step += 1
        self.generator.manual_seed(self.step)

        x          = batch["sensor"]        # (B, T, N, C_max)
        pos        = batch["sensor_poses"]  # (B, T, N, 6)
        sensor_ids = batch["sensor_id"]     # (B,)
        
        # print(x.shape, pos.shape, sensor_ids.shape)

        global_masks, local_masks, ibot_masks, \
            pos_global_masks, pos_local_masks, full_ibot = self.sample_masks(x)
        loss = self.forward(
            x, pos, sensor_ids,
            global_masks, local_masks, ibot_masks,
            pos_global_masks, pos_local_masks, full_ibot,
        )

        output = {"ssl_loss": loss.item()}

        # ── GRL domain-adversarial loss ────────────────────────────────────────
        if self.use_grl:
            grl_loss = self._grl_domain_loss(x, pos, sensor_ids)
            loss = loss + grl_loss
            output["grl_loss"] = grl_loss.item()

        # ── online probes ──────────────────────────────────────────────────────
        embedding     = None
        cls_embedding = None
        if len(self.online_probes) > 0:
            with torch.no_grad():
                teacher_dict = self.teacher_encoder_dict["backbone"].forward_features(
                    x, pos, sensor_ids=sensor_ids
                )
                cls_embedding = teacher_dict["x_norm_regtokens"].squeeze(1)
                embedding     = teacher_dict["x_norm_patchtokens"]
                embedding     = F.layer_norm(embedding, (embedding.size(-1),))
                target        = self.teacher_encoder_dict["backbone"].normalize(
                    x, sensor_ids=sensor_ids
                )

        online_probes_loss = 0.0
        for probe in self.online_probes:
            probe_name = str(probe.probe_name)
            if probe_name == "reconstruction":
                if not self.teacher_encoder_dict["backbone"].input_type == "image":
                    target = einops.rearrange(
                        target,
                        "b (t k) n c -> b (t n) (c k)",
                        k=self.student_encoder_dict["backbone"].time_chunk_size,
                    )
                probe_loss, decoded_x = probe(embedding, target=target)
                online_probes_loss += probe_loss
                output[f"{probe_name}_loss"] = probe_loss.item()
                output[f"{probe_name}_img"]  = decoded_x.detach()
            elif "classification" in probe_name:
                gt_labels  = batch[probe_name]
                probe_loss, pred_logits = probe(cls_embedding, target=gt_labels)
                pred_labels = torch.argmax(pred_logits, dim=1)
                accuracy    = (pred_labels == gt_labels).float().mean()
                online_probes_loss += probe_loss
                output[f"{probe_name}_loss"]     = probe_loss.item()
                output[f"{probe_name}_accuracy"] = accuracy
            else:
                raise NotImplementedError(f"Probe {probe_name} missing target")

        loss += online_probes_loss
        output["loss"]               = loss
        output["online_probes_loss"] = online_probes_loss
        return output

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)

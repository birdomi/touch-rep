import random
from typing import Any, Dict, List, Optional

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from xformers.ops import fmha

from tactile_ssl.algorithm import DINOv2Module
from tactile_ssl.data.xela.utils import xela_sensor_layout
from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.utils.masking import sample_block_mask, sample_block_size_1d

log = get_pylogger(__name__)


class CrossSensorDINOv2Module(DINOv2Module):
    def __init__(
        self,
        ibot_mask_ratio: List[float] = [0.1, 0.5],
        num_sensor: int = 1,
        sensor_means: Optional[List[List[float]]] = None,
        sensor_stds: Optional[List[List[float]]] = None,
        *args,
        **kwargs,
    ): 
        super().__init__(*args, **kwargs)
        # TODO: Load this in a different way
        # This is valid only when the baseline is subtracted in the xela dataset
        self.ibot_mask_ratio = ibot_mask_ratio
        self.num_sensor = num_sensor
        self.sensor_means = sensor_means
        self.sensor_stds = sensor_stds

    def on_validation_batch_end(self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None):
        self.log_on_batch_end(outputs, stage="val", trainer_instance=trainer_instance)
        # Plot online probe predictions
        if trainer_instance is not None:
            step = trainer_instance.global_val_step
            if step is None:
                return
            if (step % self.log_freq_img == 0) and "reconstruction_img" in outputs.keys():
                X_pred = outputs["reconstruction_img"]
                X_orig = batch["sensor"]

                encoder = self.student_encoder_dict["backbone"]
                
                if X_pred.ndim == 4:
                    X_pred = einops.rearrange(
                        X_pred,
                        "b t n (c k) -> b (t k) n c",
                        k=encoder.time_chunk_size,
                        n=encoder.in_dim,
                    )
                else:
                    X_pred = einops.rearrange(
                        X_pred,
                        "b (t n) (c k) -> b (t k) n c",
                        k=encoder.time_chunk_size,
                        n=encoder.in_dim,
                    )
                # in_dims corresponds to num sensors
                print(X_pred.shape)
                # X_pred is likely just Xela (3 channels) based on training_step logic
                if X_pred.shape[-1] == 3:
                     # Already in Xela format, just add dim for consistency if needed, or skip rearrange
                     X_pred = X_pred.unsqueeze(-2) # (b, t, n, 1, 3)
                else:
                     X_pred = einops.rearrange(X_pred, "b t n (c l) -> b t n c l", n=encoder.in_dim, l=encoder.in_chans)
                
                # X_orig is full sensor input (padded), so we rearrange it
                X_orig = einops.rearrange(X_orig, "b t n (c l) -> b t n c l", n=encoder.in_dim, l=encoder.in_chans)
                
                # Take first channel/component (Xela) for visualization
                X_pred = X_pred[0].cpu().numpy()[..., 0, :3]
                X_orig = X_orig[0].cpu().numpy()[..., 0, :3]
                xela_mean = self.teacher_encoder_dict["backbone"].xela_mean.detach().cpu().numpy()
                xela_std = self.teacher_encoder_dict["backbone"].xela_std.detach().cpu().numpy()
                # X_pred = xela_sensor_layout(X_pred, xela_mean, xela_std)
                # X_orig = xela_sensor_layout(X_orig)
                # print(X_pred.shape, X_orig.shape)

                # trainer_instance.wandb.log(
                #     {
                #         "val/pred_signal": trainer_instance.wandb.Video(X_pred, fps=5, format="gif"),
                #         "val/target_signal": trainer_instance.wandb.Video(X_orig, fps=5, format="gif"),
                #     }
                # )

    def sample_masks(self, x, valid_masks):
        batch_size, _, num_sensors, _ = x.shape

        collated_local_masks, collated_global_masks, collated_ibot_masks = [], [], []
        min_keep_local_patches, min_keep_global_patches = (num_sensors, num_sensors)
        for i in range(batch_size):
            # Calculate valid N for this sample
            # valid_masks: (B, T, N, C)
            valid_n = int(valid_masks[i, 0, :, 0].sum().item())
            
            local_maskblock_sizes = sample_block_size_1d(valid_n, self.local_mask_scale)[0]
            global_maskblock_sizes = sample_block_size_1d(valid_n, self.global_mask_scale)[0]

            masks_encoder, masks_complement = [], []
            ibot_masks = []
            for _ in range(self.num_global_masks):
                mask, mask_complement = sample_block_mask(
                    [valid_n],
                    [global_maskblock_sizes],
                    min_mask_size=self.min_keep,
                    generator=self.generator,
                )
                
                # Expand/Pad masks to full num_sensors
                # Generate IBOT mask on the valid/unpadded mask first
                ibot_mask = torch.zeros(len(mask), dtype=torch.bool)
                num_masked_tokens = int(random.uniform(*self.ibot_mask_ratio) * valid_n)
                # Ensure we don't try to mask more than available in the block mask
                num_masked_tokens = min(num_masked_tokens, len(mask))
                
                if len(mask) > 0:
                    ibot_mask_idx = torch.randperm(len(mask))[:num_masked_tokens]
                    ibot_mask[ibot_mask_idx] = 1

                    # Pad mask (indices) with the first masked token (to avoid masking new valid tokens)
                    padding_size = num_sensors - len(mask)
                    if padding_size > 0:
                        # Pad with mask[0]
                        pad_val = mask[0].item()
                        mask = F.pad(torch.as_tensor(mask), (0, padding_size), value=pad_val)
                        # Pad ibot_mask with False (don't predict the padded duplicates)
                        ibot_mask = F.pad(ibot_mask, (0, padding_size), value=False)
                        # Pad mask_complement (map) with 0 (False/Masked region)
                        if padding_size > 0:
                            mask_complement = F.pad(torch.as_tensor(mask_complement), (0, padding_size), value=0)

                else:
                    # Edge case: empty mask? Should not happen with min_mask_size >= 1
                    # But if it does, pad with something safe (e.g. 0 or dummy)
                    padding_size = num_sensors
                    mask = torch.zeros(num_sensors, dtype=torch.long) # Pad with 0?
                    ibot_mask = torch.zeros(num_sensors, dtype=torch.bool)
                    mask_complement = torch.zeros(num_sensors, dtype=torch.long) # Pad with 0?

                ibot_masks.append(ibot_mask)
                masks_encoder.append(mask)
                masks_complement.append(mask_complement)
                # Note: min_keep logic tracks unpadded length?
                # If we pad, len(mask) is now num_sensors.
                # So min_keep becomes num_sensors.
                # This effectively disables the min_keep cropping logic, which is what we want (fixed batch size).
                min_keep_global_patches = num_sensors

            collated_global_masks.append(masks_encoder)
            collated_ibot_masks.append(ibot_masks)

            acceptable_regions = masks_complement
            if self.allow_mask_overlap:
                acceptable_regions = None

            masks_local = []
            for _ in range(self.num_local_masks):
                mask, _ = sample_block_mask(
                    [valid_n],
                    [local_maskblock_sizes],
                    min_mask_size=self.min_keep,
                    acceptable_regions=acceptable_regions,
                    generator=self.generator,
                )
                
                # Pad local masks too
                if len(mask) > 0:
                    padding_size = num_sensors - len(mask)
                    if padding_size > 0:
                         pad_val = mask[0].item()
                         mask = F.pad(torch.as_tensor(mask), (0, padding_size), value=pad_val)
                else:
                    mask = torch.zeros(num_sensors, dtype=torch.long)

                masks_local.append(mask)
                # min_keep becomes num_sensors because we padded.
                min_keep_local_patches = num_sensors
            collated_local_masks.append(masks_local)

        collated_global_masks = [[cm[:min_keep_global_patches] for cm in masks] for masks in collated_global_masks]
        collated_local_masks = [[cm[:min_keep_local_patches] for cm in masks] for masks in collated_local_masks]

        local_masks = torch.stack(data.default_collate(collated_local_masks), dim=0).to(x.device)
        global_masks = torch.stack(data.default_collate(collated_global_masks), dim=0).to(x.device)
        ibot_masks = torch.stack(data.default_collate(collated_ibot_masks), dim=0).to(x.device)
        # print(local_masks.shape, global_masks.shape, ibot_masks.shape)

        return global_masks, local_masks, ibot_masks

    def forward(
        self,
        xs: torch.Tensor,
        pos: torch.Tensor,
        sensor_ids: torch.Tensor,
        global_masks: torch.Tensor,
        local_masks: torch.Tensor,
        ibot_masks: torch.Tensor,
    ):
        assert global_masks is not None and local_masks is not None, "Masks are required for DINOModule during training"

        ibot_masks_flat = ibot_masks.flatten(0, 1)
        ibot_mask_indices = torch.nonzero(ibot_masks_flat).flatten()
        num_ibot_tokens = len(ibot_mask_indices)

        # TODO: @Akash Sharma - Raise to make sure context encoder implements taking masks as an argument
        student_global_dict = self.student_encoder_dict["backbone"].forward_features(
            xs, pos, sensor_ids = sensor_ids, masks=global_masks, mask_type="tubelet", masktoken_masks=ibot_masks
        )
        student_local_dict = self.student_encoder_dict["backbone"].forward_features(
            xs, pos, sensor_ids = sensor_ids, masks=local_masks, mask_type="tubelet"
        )

        student_global_cls_tokens = student_global_dict["x_norm_regtokens"][:, 0]
        student_local_cls_tokens = student_local_dict["x_norm_regtokens"][:, 0]
        student_global_patch_tokens = student_global_dict["x_norm_patchtokens"]

        # Here we ensure that we select every mask token in the time series
        student_global_patch_tokens = einops.rearrange(
            student_global_patch_tokens,
            "b (t n) c -> (b n) t c",
            n=global_masks.shape[-1],
        )
        student_masked_patch_tokens = student_global_patch_tokens.new_zeros(
            (
                num_ibot_tokens,
                student_global_patch_tokens.shape[-2],
                student_global_patch_tokens.shape[-1],
            )
        )
        student_masked_patch_tokens.copy_(student_global_patch_tokens[ibot_mask_indices])
        student_masked_patch_tokens = student_masked_patch_tokens.flatten(0, 1)

        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(
            [
                student_global_cls_tokens.unsqueeze(0),
                student_local_cls_tokens.unsqueeze(0),
                student_masked_patch_tokens.unsqueeze(0),
            ]
        )
        after_head_list = _attn_bias.split(self.student_encoder_dict["dino_head"](cat_inputs))
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
            [student_global_cls_tokens_after_head, student_local_cls_tokens_after_head],
            dim=0,
        )

        with torch.no_grad():
            teacher_global_dict = self.teacher_encoder_dict["backbone"].forward_features(
                xs, pos, sensor_ids=sensor_ids, masks=global_masks, mask_type="tubelet"
            )
            teacher_global_cls_tokens = teacher_global_dict["x_norm_regtokens"][:, 0]

            teacher_global_cls_tokens = teacher_global_cls_tokens.chunk(self.num_global_masks)
            # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
            assert self.num_global_masks == 2, "Only 2 global masks are supported"
            teacher_global_cls_tokens = torch.cat((teacher_global_cls_tokens[1], teacher_global_cls_tokens[0]))

            teacher_global_patch_tokens = teacher_global_dict["x_norm_patchtokens"]
            teacher_global_patch_tokens = einops.rearrange(
                teacher_global_patch_tokens,
                "b (t n) c -> (b n) t c",
                n=global_masks.shape[-1],
            )
            teacher_masked_patch_tokens = teacher_global_patch_tokens.new_zeros(
                (
                    num_ibot_tokens,
                    student_global_patch_tokens.shape[-2],
                    student_global_patch_tokens.shape[-1],
                )
            )
            teacher_masked_patch_tokens.copy_(teacher_global_patch_tokens[ibot_mask_indices])
            teacher_masked_patch_tokens = teacher_masked_patch_tokens.flatten(0, 1)

            teacher_cls_tokens_after_head = self.teacher_encoder_dict["dino_head"](teacher_global_cls_tokens)
            teacher_masked_patch_tokens_after_head = self.teacher_encoder_dict["dino_head"](teacher_masked_patch_tokens)

            if self.centering == "centering":
                teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head,
                    teacher_temp=self.current_teacher_temp,
                ).view(
                    self.num_global_masks,
                    -1,
                    *teacher_cls_tokens_after_head.shape[1:],
                )
                teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                    teacher_masked_patch_tokens_after_head.unsqueeze(0),
                    teacher_temp=self.current_teacher_temp,
                )
                teacher_ibot_softmaxed_centered = teacher_ibot_softmaxed_centered.squeeze(0)
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                self.ibot_patch_loss.update_center(teacher_masked_patch_tokens_after_head)

            elif self.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head,
                    teacher_temp=self.current_teacher_temp,
                ).view(
                    self.num_global_masks,
                    -1,
                    *teacher_cls_tokens_after_head.shape[1:],
                )
                teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                    teacher_masked_patch_tokens_after_head,
                    teacher_temp=self.current_teacher_temp,
                    n_masked_patches_tensor=torch.tensor(
                        num_ibot_tokens,
                        dtype=int,
                        device=teacher_masked_patch_tokens.device,
                    ),
                )
            else:
                raise NotImplementedError

        n_local_crops_loss_terms = max(self.num_local_masks * self.num_global_masks, 1)
        n_global_crops_loss_terms = (self.num_global_masks - 1) * self.num_global_masks

        dino_loss = self.dino_loss(
            student_cls_tokens_after_head.chunk(self.num_global_masks + self.num_local_masks),
            teacher_dino_softmaxed_centered_list,
        ) / (n_local_crops_loss_terms + n_global_crops_loss_terms)

        koleo_loss = self.koleo_weight * sum(
            self.koleo_loss(p.squeeze(dim=-2)) for p in student_global_cls_tokens.chunk(2, dim=1)
        )  # we don't apply koleo loss between cls tokens of a same image

        ibot_loss_scale = 1.0 / self.num_global_masks
        patch_loss = ibot_loss_scale * self.ibot_patch_loss(
            student_patch_tokens_after_head, teacher_ibot_softmaxed_centered
        )
        loss = dino_loss + patch_loss + koleo_loss

        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        self.step = self.step + 1
        self.generator.manual_seed(self.step)
        x = batch["sensor"]
        pos = batch["sensor_poses"]
        sensor_ids = batch["sensor_ids"]
        valid_masks = batch["valid_mask"]
        global_masks, local_masks, ibot_masks = self.sample_masks(x, valid_masks)

        loss = self.forward(x, pos, sensor_ids, global_masks, local_masks, ibot_masks)

        output = {
            "ssl_loss": loss.item(),
        }

        # online probes
        embedding = None
        cls_embedding = None
        if len(self.online_probes) > 0:
            with torch.no_grad():
                teacher_dict = self.teacher_encoder_dict["backbone"].forward_features(x, pos, sensor_ids=sensor_ids)
                cls_embedding = teacher_dict["x_norm_regtokens"].squeeze(1)
                embedding = teacher_dict["x_norm_patchtokens"]
                embedding = F.layer_norm(embedding, (embedding.size(-1),))
                target = self.teacher_encoder_dict["backbone"].normalize(x)

        online_probes_loss = 0.0
        for probe in self.online_probes:
            probe_name: str = str(probe.probe_name)
            if probe_name == "reconstruction":
                targets = []
                for sensor_idx in range(self.num_sensor):
                    target_ = target[sensor_ids == sensor_idx]
                    if sensor_idx == 0:
                        target_ = target_[:, :, :, :3]
                    elif sensor_idx == 1:
                        target_ = target_[:, -1, :, :25]
                    # print(target.shape, target_.shape, sensor_idx)
                    targets.append(target_)

                probe_loss, decoded_x = probe(embedding, target=targets, sensor_ids=sensor_ids, sensor_wise=True)
                online_probes_loss += probe_loss
                output[f"{probe_name}_loss"] = probe_loss.item()
                output[f"{probe_name}_img"] = decoded_x[0].detach()

            elif "classification" in probe_name:
                gt_labels = batch[probe_name]
                probe_loss, pred_logits = probe(cls_embedding, target=gt_labels)
                pred_labels = torch.argmax(pred_logits, dim=1)
                accuracy = (pred_labels == gt_labels).float().mean()
                online_probes_loss += probe_loss
                output[f"{probe_name}_loss"] = probe_loss.item()
                output[f"{probe_name}_accuracy"] = accuracy
            else:
                raise NotImplementedError(f"Probe {probe_name} missing target")

        loss += online_probes_loss
        output["loss"] = loss  # type: ignore
        output["online_probes_loss"] = online_probes_loss

        return output

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)

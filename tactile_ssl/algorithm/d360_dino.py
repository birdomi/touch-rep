# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
SSL trainign module for D360 multisensory representation learning following 
the teacher-student self-distillation paradigm, (e.g. DINO loss).
"""

from typing import Any, Dict, Tuple, Optional, List, Union, Literal
from functools import partial

import torch
import torch.nn as nn
from tactile_ssl.model.d360_transformer import D360Transformer

import einops

from tactile_ssl.utils.logging import get_pylogger, img_logger, imu_logger, pressure_logger
from tactile_ssl.algorithm.multimodal_dino import MultimodalDINOModule
from tactile_ssl.utils.masking import sample_block_size_1d, sample_block_size_2d

log = get_pylogger(__name__)


class D360DINOModule(MultimodalDINOModule):
    def __init__(
        self,
        encoder: D360Transformer,
        dino_head: partial,
        optim_cfg: partial,
        lr_scheduler_cfg: Optional[partial],
        wd_scheduler_cfg: Optional[partial],
        global_mask_scales: Dict[str, Tuple[float, float]],
        local_mask_scales: Dict[str, Tuple[float, float]],
        online_probes: Optional[List[nn.Module]] = None,
        online_probes_lrs: List[float] = [],
        num_global_masks: int = 1,
        num_local_masks: int = 4,
        min_keep_num_patches: Dict[str, int] = {"img": 4, "mic": 4, "imu": 4, "pressure": 4},
        allow_mask_overlap: bool = False,
        moving_average_decay: Union[float, Tuple[float, ...]] = 0.99,
        teacher_temp: Union[float, Tuple[float, ...]] = (0.04, 0.07),
        teacher_warmup_epochs: int = 10,
        use_momentum=True,
        loss_weight_reconstruction: Dict[str, int] = {"img": 1, "mic": 1, "imu": 1, "pressure": 1e-4},
        log_freq_reconstruction: int = 1000,
    ):
        self.use_img = encoder.use_img
        self.use_mic = encoder.use_mic
        self.use_imu = encoder.use_imu
        self.use_pressure = encoder.use_pressure

        self.sensor_sizes: Dict[str, Union[int, Tuple[int, int]]] = encoder.sensor_sizes
        self.sensor_chans: Dict[str, int] = encoder.sensor_chans
        self.patch_sizes: Dict[str, int] = encoder.patch_sizes

        assert len(global_mask_scales) == 4
        assert len(local_mask_scales) == 4
        assert len(min_keep_num_patches) == 4

        self.sensors = encoder.sensors
        self.embed_shapes = encoder.modal_shapes

        global_m_scales = {}
        local_m_scales = {}
        min_keeps = {}
        for sensor in self.sensors:
            global_m_scales[sensor] = global_mask_scales[sensor]
            local_m_scales[sensor] = local_mask_scales[sensor]
            min_keeps[sensor] = min_keep_num_patches[sensor]

        self.loss_weight_reconstruction = loss_weight_reconstruction

        super().__init__(
            encoder=encoder,
            dino_head=dino_head,
            optim_cfg=optim_cfg,
            lr_scheduler_cfg=lr_scheduler_cfg,
            wd_scheduler_cfg=wd_scheduler_cfg,
            global_mask_scales=global_m_scales,
            local_mask_scales=local_m_scales,
            num_global_masks=num_global_masks,
            num_local_masks=num_local_masks,
            min_keep_num_patches=min_keeps,
            allow_mask_overlap=allow_mask_overlap,
            online_probes=online_probes,
            online_probes_lrs=online_probes_lrs,
            moving_average_decay=moving_average_decay,
            teacher_temp=teacher_temp,
            teacher_warmup_epochs=teacher_warmup_epochs,
            use_momentum=use_momentum,
            log_freq_reconstruction=log_freq_reconstruction,
        )

    def get_mask_shapes(self, mask_scales: Dict[str, float], *args, **kwargs):
        mask_shapes = {}
        for sensor in self.sensors:
            embed_shape = self.embed_shapes[sensor]
            mask_scale = mask_scales[sensor]
            if sensor == "img" or sensor == "mic":
                mask_shape = sample_block_size_2d(
                    embed_shape[0], embed_shape[1], mask_scale, aspect_ratio_scale=(1.0, 1.0), generator=self.generator
                )
            elif sensor == "imu" or sensor == "pressure":
                mask_shape = sample_block_size_1d(embed_shape[0], mask_scale, generator=self.generator)
            else:
                raise NotImplementedError
            mask_shapes[sensor] = mask_shape
        return mask_shapes

    def get_global_mask_shapes(self, *args, **kwargs):
        return self.get_mask_shapes(self.global_mask_scales)

    def get_local_mask_shapes(self, *args, **kwargs):
        return self.get_mask_shapes(self.local_mask_scales)

    def get_embed_shapes(self, *args, **kwargs):
        return self.embed_shapes

    def log_on_batch_end(
        self, outputs: Dict[str, torch.Tensor], stage: Literal["train", "val"] = "train", trainer_instance=None
    ):
        if trainer_instance is not None and trainer_instance.should_log:
            step = trainer_instance.global_val_step if stage == "val" else trainer_instance.global_step

            for k, v in outputs.items():
                if "loss" in k or "accuracy" in k:
                    trainer_instance.wandb.log({f"{stage}/{k}": v, f"global_{stage}_step": step})

            trainer_instance.wandb.log(
                {
                    f"{stage}/teacher_temperature": self.current_teacher_temp,
                    f"global_{stage}_step": step,
                }
            )

    def on_validation_batch_end(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
        batch_idx: int,
        trainer_instance=None,
    ):
        self.log_on_batch_end(outputs, stage="val", trainer_instance=trainer_instance)
        # Plot online probe predictions
        step = trainer_instance.global_val_step
        if trainer_instance is not None and trainer_instance.should_log:
            if step % self.log_freq_img == 0:
                for sensor in self.sensors:
                    pred_sensor = f"pred_{sensor}"
                    gt_sensor = f"gt_{sensor}"
                    if pred_sensor in outputs.keys():
                        x_pred = outputs[pred_sensor]
                        x_orig = outputs[gt_sensor] if gt_sensor in outputs.keys() else None

                        if sensor == "mic":
                            x_pred = x_pred[:, None]
                            x_orig = x_orig[:, None] if x_orig is not None else None

                        if sensor == "img" or sensor == "mic":
                            img_logger(
                                wandb=trainer_instance.wandb,
                                global_step=step,
                                predictions=x_pred,
                                X=x_orig,
                                label="val",
                                type=sensor,
                            )
                        elif sensor == "imu":
                            imu_logger(
                                wandb=trainer_instance.wandb,
                                global_step=step,
                                predictions=x_pred,
                                origs=x_orig,
                                label="val",
                            )
                        elif sensor == "pressure":
                            pressure_logger(
                                wandb=trainer_instance.wandb,
                                global_step=step,
                                predictions=x_pred,
                                origs=x_orig,
                                label="val",
                            )
                        else:
                            raise NotImplementedError

    def prepare_data(self, batch: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        info = {"mic_fbank": "mic", "imu_acc": "imu"}
        xs_orig = {info.get(key, key): val for key, val in batch.items()}
        xs = {}

        for sensor in self.sensors:
            if sensor == "img":
                xs_orig["img"] = einops.rearrange(xs_orig["img"], "b n c h w -> b c n h w")
                xs["img"] = einops.rearrange(xs_orig["img"], "b c n h w -> b (c n) h w")
            elif sensor == "mic" or sensor == "imu" or sensor == "pressure":
                xs[sensor] = xs_orig[sensor]
            else:
                raise NotImplementedError

        return xs, xs_orig

    def postprocess(
        self,
        losses: Dict[str, torch.Tensor],
        *args,
        **kwargs,
    ):
        assert len(self.sensors) == len(losses)
        ssl_loss = sum(losses.values())
        ssl_losses = {"ssl_loss": ssl_loss.item()}
        ssl_losses.update({f"{sensor}_ssl_loss": losses[sensor].item() for sensor in self.sensors})
        return ssl_loss, ssl_losses

    def online_probe_reconstruction(
        self, xs_gt: Dict[str, torch.Tensor], embeddings: Dict[str, torch.Tensor], probe: nn.Module, *args, **kwargs
    ):
        outputs = {}
        xs_gt = {sensor: xs_gt[sensor] for sensor in self.sensors}
        decoded_x = probe.forward_decoder(embeddings)
        xs_pred = {}
        for sensor, x in decoded_x.items():
            sensor_size = self.sensor_sizes[sensor]
            num_chans = self.sensor_chans[sensor]
            patch_size = self.patch_sizes[sensor]
            if sensor == "img":
                xs_pred[sensor] = einops.rearrange(
                    x,
                    "b 1 (h w) (p q c n) -> b c n (h p) (w q)",
                    h=sensor_size[0] // patch_size,
                    w=sensor_size[1] // patch_size,
                    p=patch_size,
                    q=patch_size,
                    n=2,
                    c=num_chans // 2,
                )
            elif sensor == "mic":
                xs_pred[sensor] = einops.rearrange(
                    x,
                    "b 1 (h w) (p q) -> b (h p) (w q)",
                    h=sensor_size[0] // patch_size,
                    w=sensor_size[1] // patch_size,
                    p=patch_size,
                    q=patch_size,
                )
            elif sensor == "imu" or sensor == "pressure":
                xs_pred[sensor] = einops.rearrange(x, "b 1 t (p c) -> b (t p) c", p=patch_size, c=num_chans)
            else:
                raise NotImplementedError

        loss = 0
        for sensor in self.sensors:
            recon_loss = probe.loss_fn(xs_pred[sensor], xs_gt[sensor])
            loss += recon_loss * self.loss_weight_reconstruction[sensor]
            output = {
                f"pred_{sensor}": xs_pred[sensor],
                f"gt_{sensor}": xs_gt[sensor],
                f"recon_{sensor}_loss": recon_loss,
            }
            outputs.update(output)

        return loss, outputs

    def online_probe_classification(
        self, xs_gt: Dict[str, torch.Tensor], embeddings: Dict[str, torch.Tensor], probe: nn.Module, *args, **kwargs
    ):
        assert "classification" in probe.probe_name

        probe_name = probe.probe_name
        output = {}

        gt_labels = xs_gt[probe_name.split("_")[0]]
        probe_loss, pred_logits = probe(
            torch.cat([embeddings[sensor] for sensor in self.sensors], dim=1), target=gt_labels
        )
        pred_labels = torch.sigmoid(pred_logits).argmax(dim=-1)
        accuracy = (pred_labels == gt_labels)[gt_labels >= 0].float().mean()
        output[f"{probe_name}_loss"] = probe_loss.item()
        output[f"{probe_name}_accuracy"] = accuracy.item()
        return probe_loss, output


    def online_probe(
        self, xs_gt: Dict[str, Dict[str, torch.Tensor]], embeddings: Dict[str, Dict[str, torch.Tensor]], *args, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
       
        """
        Online probes are small networks that are trained to perform auxiliary tasks, 
        such as reconstruction or classification, on the learned representations.

        The losses from these probes are not propagated back to the main model, 
        but are used for monitoring the quality of the learned representations.
        """
       
        online_probe_loss = torch.tensor(0.0)
        online_probe_outputs = {}
        if self.online_probes is None:
            return online_probe_loss, online_probe_outputs

        if "supp" not in xs_gt:
            xs_gt["supp"] = xs_gt["main"]
            embeddings["supp"] = embeddings["main"]

        for probe in self.online_probes:
            if "reconstruction" in probe.probe_name:
                probe_loss, probe_output = self.online_probe_reconstruction(xs_gt["main"], embeddings["main"], probe)
            elif "classification" in probe.probe_name:
                probe_loss, probe_output = self.online_probe_classification(xs_gt["supp"], embeddings["supp"], probe)
            else:
                raise NotImplementedError

            online_probe_loss = online_probe_loss + probe_loss
            online_probe_outputs.update(probe_output)
        return online_probe_loss, online_probe_outputs

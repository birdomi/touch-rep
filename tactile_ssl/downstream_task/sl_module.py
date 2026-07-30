# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Reference:
#   https://github.com/facebookresearch/sparsh/blob/main/tactile_ssl/downstream_task/sl_module.py


from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from tactile_ssl.algorithm.module import Module
from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)


class SLModule(Module, nn.Module):
    def __init__(
        self,
        model_encoder: nn.Module,
        model_task: nn.Module,
        optim_cfg: partial,
        scheduler_cfg: Optional[partial],
        checkpoint_encoder: Optional[str] = None,
        checkpoint_task: Optional[str] = None,
        train_encoder: bool = False,
        train_input_conv1d: bool = False,
        encoder_type: str = "jepa",
        lora_rank: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        lora_target_modules: tuple = ("qkv", "proj"),
        encoder_lr: Optional[float] = None,
        task_lr: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.model_task: nn.Module = model_task
        self.model_encoder: nn.Module = model_encoder
        self.train_encoder: bool = train_encoder
        self.train_input_conv1d: bool = train_input_conv1d
        self.encoder_type: str = encoder_type
        self.encoder_lr = encoder_lr
        self.task_lr = task_lr

        if checkpoint_encoder is not None:
            log.info("Loading encoder ONLY from checkpoint.")
            self.load_encoder(checkpoint_encoder)
        else:
            log.info("No checkpoint provided. Training from scratch.")

        if checkpoint_task is not None:
            log.info("Loading task decoder from checkpoint.")
            self.load_task(checkpoint_task)

        # ── LoRA ──────────────────────────────────────────────────────────────
        if lora_rank > 0:
            from tactile_ssl.model.lora import apply_lora, lora_trainable_params
            n = apply_lora(
                self.model_encoder,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=tuple(lora_target_modules),
            )
            log.info(f"LoRA applied to {n} layers (rank={lora_rank}, alpha={lora_alpha})")

        # Freeze encoder; if LoRA, unfreeze only lora_ params
        if not self.train_encoder:
            self.model_encoder.requires_grad_(False)
            self.model_encoder.eval()

        # Keep only the temporal input projections trainable while the
        # Transformer encoder itself is frozen. AngleTransformer exposes its
        # two PatchEmbed1d modules as direct children (sensor_embed and
        # angle_embed); only their Conv1d projections are unfrozen, not their
        # LayerNorms or any Transformer parameters.
        if self.train_input_conv1d:
            trainable_input_convs = []
            # The temporal composite nests its AngleTransformer one level down
            # (model_encoder.frame_encoder); search both containers.
            containers = [("", self.model_encoder)]
            frame_encoder = getattr(self.model_encoder, "frame_encoder", None)
            if frame_encoder is not None:
                containers.append(("frame_encoder.", frame_encoder))
            for prefix, container in containers:
                for module_name, module in container.named_children():
                    projection = getattr(module, "proj", None)
                    if isinstance(projection, nn.Conv1d):
                        projection.requires_grad_(True)
                        trainable_input_convs.append(f"{prefix}{module_name}.proj")
            if not trainable_input_convs:
                raise ValueError(
                    "train_input_conv1d=True, but the encoder has no direct "
                    "PatchEmbed1d-style Conv1d projection"
                )
            log.info(
                "Trainable encoder input Conv1d projections: "
                + ", ".join(trainable_input_convs)
            )

        if lora_rank > 0:
            for name, param in self.model_encoder.named_parameters():
                if "lora_" in name:
                    param.requires_grad_(True)
            self.model_encoder.train()  # enable dropout in LoRA layers
            n_lora = sum(p.numel() for n, p in self.model_encoder.named_parameters()
                         if p.requires_grad and "lora_" in n)
            log.info(f"LoRA trainable encoder params: {n_lora:,}")

        self.scheduler_partial = scheduler_cfg
        self.optim_partial = optim_cfg

    def encoder_grad_enabled(self) -> bool:
        """Return whether any encoder parameter should receive gradients."""
        return any(param.requires_grad for param in self.model_encoder.parameters())

    def load_task(self, checkpoint_task: str):
        try:
            state_dict = torch.load(checkpoint_task, weights_only=False)
            if "ckpt" in checkpoint_task:
                state_dict = state_dict["model"]
            # check if there are keys starting with "model_encoder."
            if any([key.startswith("model_encoder.") for key in state_dict.keys()]):
                log.info("Found encoder in task checkpoint. Loading encoder and decoder from task checkpoint.")
                self.load_state_dict(state_dict, strict=True)
                return
            else:
                state_dict = {key.replace("model_task.", ""): value for key, value in state_dict.items()}
                self.model_task.load_state_dict(state_dict, strict=True)
                log.info(f"Loaded task model from {checkpoint_task}")
        except Exception as e:
            log.info(f"{e} Could not load task model from {checkpoint_task}")
            raise ValueError()

    def load_encoder(self, checkpoint_encoder: str):
        log.info(f"Loading encoder from {checkpoint_encoder}")
        checkpoint = torch.load(checkpoint_encoder, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
            checkpoint_state = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            checkpoint_state = checkpoint["state_dict"]
        else:
            checkpoint_state = checkpoint

        if self.encoder_type == "tactile_jepa_teacher":
            # SpatiotemporalTactileJEPAModule stores the EMA target directly
            # under teacher_encoder.*. Do not silently fall back to the student.
            encoder_keys = ["teacher_encoder"]
        elif "jepa" in self.encoder_type:
            encoder_keys = ["target_encoder", "context_encoder", "encoder", "model_encoder"]
        elif "dino" in self.encoder_type:
            encoder_keys = [
                "teacher_encoder.backbone",
                "student_encoder.backbone",
                "teacher_encoder",
                "student_encoder",
                "encoder",
                "model_encoder",
            ]
        else:
            encoder_keys = ["encoder", "model_encoder"]

        model_state = self.model_encoder.state_dict()

        filtered_state_dict = {}
        matched_encoder_key = None
        skipped = []
        for encoder_key in encoder_keys:
            prefix = f"{encoder_key}."
            target_keys = [key for key in checkpoint_state.keys() if prefix in key]
            new_state_dict = {
                key.split(prefix, 1)[1]: checkpoint_state[key]
                for key in target_keys
                if hasattr(checkpoint_state[key], "shape")
            }
            filtered_state_dict = {
                k: v for k, v in new_state_dict.items()
                if k in model_state and v.shape == model_state[k].shape
            }
            skipped = [k for k in new_state_dict if k not in filtered_state_dict]
            if filtered_state_dict:
                matched_encoder_key = encoder_key
                break

        if not filtered_state_dict:
            tensor_keys = [k for k, v in checkpoint_state.items() if hasattr(v, "shape")]
            raise RuntimeError(
                f"No compatible encoder parameters were found in checkpoint: {checkpoint_encoder}. "
                f"Tried prefixes {encoder_keys}. First tensor keys: {tensor_keys[:12]}"
            )
        if skipped:
            log.info(f"Skipped {len(skipped)} missing/shape-mismatched keys: {skipped[:20]}")
        self.model_encoder.load_state_dict(filtered_state_dict, strict=False)
        print(
            f"Loaded encoder from {checkpoint_encoder} "
            f"(prefix={matched_encoder_key}, tensors={len(filtered_state_dict)})"
        )

    def forward(self, x, *args, **kwargs):  # noqa
        raise NotImplementedError

    def training_step(self, batch: Dict[str, Any], batch_idx: int, *args, **kwargs):  # noqa
        raise NotImplementedError

    def validation_step(self, *args, **kwargs):  # noqa
        raise NotImplementedError

    def test_step(self, *args, **kwargs):  # noqa
        raise NotImplementedError

    def configure_optimizers(
        self, num_iterations_per_epoch, num_epochs, *args, **kwargs
    ) -> Tuple[torch.optim.Optimizer, Optional[Dict], Optional[Dict]]:
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        use_discriminative_lr = self.encoder_lr is not None or self.task_lr is not None
        if use_discriminative_lr:
            if self.encoder_lr is None or self.task_lr is None:
                raise ValueError(
                    "encoder_lr and task_lr must be set together for discriminative LR"
                )
            encoder_params = {
                name: param
                for name, param in param_dict.items()
                if name.startswith("model_encoder.")
            }
            task_params = {
                name: param
                for name, param in param_dict.items()
                if not name.startswith("model_encoder.")
            }
            optim_groups = []
            for params, lr in (
                (encoder_params, float(self.encoder_lr)),
                (task_params, float(self.task_lr)),
            ):
                decay = [param for param in params.values() if param.dim() >= 2]
                nodecay = [param for param in params.values() if param.dim() < 2]
                if decay:
                    optim_groups.append({"params": decay, "lr": lr})
                if nodecay:
                    optim_groups.append({
                        "params": nodecay,
                        "lr": lr,
                        "WD_exclude": True,
                        "weight_decay": 0.0,
                    })
            log.info(
                f"Discriminative LR: encoder={float(self.encoder_lr):.2e}, "
                f"task={float(self.task_lr):.2e}"
            )
        else:
            decay_params = [p for p in param_dict.values() if p.dim() >= 2]
            nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
            optim_groups = [
                {"params": decay_params},
                {"params": nodecay_params, "WD_exclude": True, "weight_decay": 0.0},
            ]
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        log.info(
            f"Total parameters: {sum(p.numel() for p in param_dict.values())}, "
            f"Decay parameters: {num_decay_params}, "
            f"Nodecay parameters: {num_nodecay_params}"
        )
        optimizer = self.optim_partial(optim_groups)

        if self.scheduler_partial is not None:
            lr_scheduler = self.scheduler_partial(
                optimizer=optimizer,
                T_max=int(num_epochs * num_iterations_per_epoch),
                steps_per_epoch=num_iterations_per_epoch,
            )
            return (
                optimizer,
                {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "monitor": None,
                },
                None,
            )

        return optimizer, None, None

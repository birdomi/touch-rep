"""Spatial and non-overlapping temporal JEPA for tactile hand sequences.

Each training sample contains six frames. Frames 0:3 form the current window
and frames 3:6 form the non-overlapping future window. A temporal Conv1d in the
encoder compresses each three-frame signal into one token per spatial entity.

The shared token layout is always

    [10 fingertip XYZ tokens][42 tactile/force tokens].

The student sees a masked current window. Tactile and XYZ finger-block masks
are sampled independently. The EMA teacher sees both windows without masks.
"""

from __future__ import annotations

import copy
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from tactile_ssl.algorithm.module import Module
from tactile_ssl.utils.ema import update_moving_average
from tactile_ssl.utils.logging import get_pylogger


log = get_pylogger(__name__)

NUM_FINGERS = 10
NUM_XYZ_TOKENS = 10
NUM_TACTILE_TOKENS = 42
NUM_TOKEN_POSITIONS = NUM_XYZ_TOKENS + NUM_TACTILE_TOKENS

# MediaPipe-style skeleton: wrist followed by four joints for each finger.
# Wrist tokens 0 and 21 remain visible because they do not belong to a finger.
TACTILE_FINGER_GROUPS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
    (22, 23, 24, 25),
    (26, 27, 28, 29),
    (30, 31, 32, 33),
    (34, 35, 36, 37),
    (38, 39, 40, 41),
)


class SpatiotemporalTactileJEPAModule(Module, nn.Module):
    """Patch-only JEPA with independent tactile and XYZ finger masks."""

    def __init__(
        self,
        encoder: nn.Module,
        predictor: partial,
        optim_cfg: partial,
        lr_scheduler_cfg: Optional[partial],
        wd_scheduler_cfg: Optional[partial],
        sensor_input_key: str = "joint_force",
        pos_input_key: str = "finger_xyz",
        context_window_size: int = 3,
        tactile_mask_ratio: float = 0.4,
        xyz_mask_ratio: float = 0.4,
        spatial_tactile_loss_weight: float = 1.0,
        spatial_xyz_loss_weight: float = 1.0,
        temporal_tactile_loss_weight: float = 1.0,
        moving_average_decay: Union[float, Sequence[float]] = (0.994, 1.0),
        loss_type: str = "smooth_l1",
    ) -> None:
        super().__init__()
        if encoder.num_register_tokens != 0:
            raise ValueError(
                "SpatiotemporalTactileJEPAModule requires num_register_tokens=0"
            )
        if encoder.sequence_length != context_window_size:
            raise ValueError(
                "encoder.sequence_length must equal context_window_size, got "
                f"{encoder.sequence_length} and {context_window_size}"
            )
        if encoder.time_chunk_size != context_window_size:
            raise ValueError(
                "time_chunk_size must equal context_window_size so Conv1d produces "
                "one token per spatial entity"
            )
        if encoder.in_dim != NUM_TACTILE_TOKENS:
            raise ValueError(
                f"Expected {NUM_TACTILE_TOKENS} tactile tokens, got {encoder.in_dim}"
            )
        if encoder.pos_in_dim != NUM_XYZ_TOKENS:
            raise ValueError(
                f"Expected {NUM_XYZ_TOKENS} XYZ tokens, got {encoder.pos_in_dim}"
            )
        for name, ratio in (
            ("tactile_mask_ratio", tactile_mask_ratio),
            ("xyz_mask_ratio", xyz_mask_ratio),
        ):
            if not 0.0 < ratio < 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {ratio}")
        if loss_type not in {"smooth_l1", "cosine"}:
            raise ValueError("loss_type must be 'smooth_l1' or 'cosine'")

        self.sensor_input_key = sensor_input_key
        self.pos_input_key = pos_input_key
        self.context_window_size = int(context_window_size)
        self.required_window_size = 2 * self.context_window_size
        self.tactile_mask_ratio = float(tactile_mask_ratio)
        self.xyz_mask_ratio = float(xyz_mask_ratio)
        self.spatial_tactile_loss_weight = float(spatial_tactile_loss_weight)
        self.spatial_xyz_loss_weight = float(spatial_xyz_loss_weight)
        self.temporal_tactile_loss_weight = float(temporal_tactile_loss_weight)
        self.loss_type = loss_type

        self.student_encoder = encoder
        self.teacher_encoder = copy.deepcopy(encoder)
        self.teacher_encoder.requires_grad_(False)
        self.teacher_encoder.eval()

        predictor_kwargs = {
            "encoder_dim": encoder.embed_dim,
            "num_token_positions": NUM_TOKEN_POSITIONS,
        }
        self.spatial_predictor = predictor(**predictor_kwargs)
        self.temporal_predictor = predictor(**predictor_kwargs)

        self.optim_partial = optim_cfg
        self.lr_scheduler_partial = lr_scheduler_cfg
        self.wd_scheduler_partial = wd_scheduler_cfg

        if isinstance(moving_average_decay, (float, int)):
            self.moving_average_decay: Union[float, Tuple[float, float]] = float(
                moving_average_decay
            )
        else:
            decay_values = list(moving_average_decay)
            if len(decay_values) != 2:
                raise ValueError("moving_average_decay sequence must contain [start, end]")
            self.moving_average_decay = (
                float(decay_values[0]),
                float(decay_values[1]),
            )
        self.momentum_scheduler = None
        self.current_momentum = (
            self.moving_average_decay[0]
            if isinstance(self.moving_average_decay, tuple)
            else self.moving_average_decay
        )

        self.register_buffer(
            "tactile_finger_groups",
            torch.tensor(TACTILE_FINGER_GROUPS, dtype=torch.long),
            persistent=False,
        )

    def train(self, mode: bool = True):
        result = super().train(mode)
        # The EMA target must remain deterministic even while the student uses
        # drop-path during training.
        self.teacher_encoder.eval()
        return result

    @staticmethod
    def _gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        gather_indices = indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        return torch.gather(tokens, dim=1, index=gather_indices)

    @staticmethod
    def _visible_complement(masked: torch.Tensor, num_tokens: int) -> torch.Tensor:
        batch_size = masked.shape[0]
        visible = torch.ones(
            batch_size, num_tokens, dtype=torch.bool, device=masked.device
        )
        visible.scatter_(1, masked, False)
        all_indices = torch.arange(num_tokens, device=masked.device).expand(
            batch_size, -1
        )
        return all_indices[visible].view(batch_size, -1)

    @staticmethod
    def _num_masked_fingers(mask_ratio: float) -> int:
        return max(1, min(NUM_FINGERS - 1, int(round(mask_ratio * NUM_FINGERS))))

    def _sample_finger_masks(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Sample independent tactile and XYZ finger-block masks."""
        num_tactile_fingers = self._num_masked_fingers(self.tactile_mask_ratio)
        num_xyz_fingers = self._num_masked_fingers(self.xyz_mask_ratio)

        tactile_fingers = torch.rand(batch_size, NUM_FINGERS, device=device).topk(
            num_tactile_fingers, dim=-1, largest=False
        ).indices
        xyz_fingers = torch.rand(batch_size, NUM_FINGERS, device=device).topk(
            num_xyz_fingers, dim=-1, largest=False
        ).indices

        tactile_masked = self.tactile_finger_groups[tactile_fingers].flatten(1)
        tactile_masked = tactile_masked.sort(dim=-1).values
        xyz_masked = xyz_fingers.sort(dim=-1).values

        tactile_visible = self._visible_complement(
            tactile_masked, NUM_TACTILE_TOKENS
        )
        xyz_visible = self._visible_complement(xyz_masked, NUM_XYZ_TOKENS)
        return {
            "tactile_visible": tactile_visible,
            "tactile_masked": tactile_masked,
            "xyz_visible": xyz_visible,
            "xyz_masked": xyz_masked,
        }

    def _latent_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(prediction, target)
        return (1.0 - F.cosine_similarity(prediction, target, dim=-1)).mean()

    def _split_windows(
        self, sensor: torch.Tensor, pos: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if sensor.ndim != 4 or pos.ndim != 4:
            raise ValueError(
                "Expected sensor and pos inputs with shape (B,T,N,C), got "
                f"{tuple(sensor.shape)} and {tuple(pos.shape)}"
            )
        if (
            sensor.shape[1] != self.required_window_size
            or pos.shape[1] != self.required_window_size
        ):
            raise ValueError(
                f"JEPA requires exactly {self.required_window_size} frames per sample, "
                f"got sensor T={sensor.shape[1]} and pos T={pos.shape[1]}"
            )
        split = self.context_window_size
        return sensor[:, :split], pos[:, :split], sensor[:, split:], pos[:, split:]

    def forward(self, sensor: torch.Tensor, pos: torch.Tensor) -> Dict[str, torch.Tensor]:
        current_sensor, current_pos, future_sensor, future_pos = self._split_windows(
            sensor, pos
        )
        batch_size = sensor.shape[0]
        masks = self._sample_finger_masks(batch_size, sensor.device)

        # A leading singleton mask dimension creates one masked view and keeps
        # the effective student batch size equal to B.
        student_dict = self.student_encoder.forward_features(
            current_sensor,
            current_pos,
            masks=masks["tactile_visible"].unsqueeze(0),
            mask_type="tubelet",
            pos_masks=masks["xyz_visible"].unsqueeze(0),
        )
        student_tokens = student_dict["x_norm_patchtokens"]

        # AngleTransformer emits XYZ first and tactile second.
        context_indices = torch.cat(
            (
                masks["xyz_visible"],
                masks["tactile_visible"] + NUM_XYZ_TOKENS,
            ),
            dim=1,
        )
        spatial_target_indices = torch.cat(
            (
                masks["xyz_masked"],
                masks["tactile_masked"] + NUM_XYZ_TOKENS,
            ),
            dim=1,
        )
        spatial_predictions = self.spatial_predictor(
            student_tokens, context_indices, spatial_target_indices
        )

        future_tactile_indices = torch.arange(
            NUM_XYZ_TOKENS,
            NUM_TOKEN_POSITIONS,
            device=sensor.device,
            dtype=torch.long,
        ).expand(batch_size, -1)
        temporal_predictions = self.temporal_predictor(
            student_tokens, context_indices, future_tactile_indices
        )

        with torch.no_grad():
            # Encode both unmasked teacher windows in one call.
            teacher_dict = self.teacher_encoder.forward_features(
                torch.cat((current_sensor, future_sensor), dim=0),
                torch.cat((current_pos, future_pos), dim=0),
            )
            teacher_current, teacher_future = teacher_dict[
                "x_norm_patchtokens"
            ].chunk(2, dim=0)

        num_xyz_targets = masks["xyz_masked"].shape[1]
        spatial_xyz_predictions = spatial_predictions[:, :num_xyz_targets]
        spatial_tactile_predictions = spatial_predictions[:, num_xyz_targets:]

        teacher_current_xyz = teacher_current[:, :NUM_XYZ_TOKENS]
        teacher_current_tactile = teacher_current[:, NUM_XYZ_TOKENS:]
        spatial_xyz_targets = self._gather_tokens(
            teacher_current_xyz, masks["xyz_masked"]
        )
        spatial_tactile_targets = self._gather_tokens(
            teacher_current_tactile, masks["tactile_masked"]
        )
        temporal_tactile_targets = teacher_future[:, NUM_XYZ_TOKENS:]

        spatial_xyz_loss = self._latent_loss(
            spatial_xyz_predictions, spatial_xyz_targets
        )
        spatial_tactile_loss = self._latent_loss(
            spatial_tactile_predictions, spatial_tactile_targets
        )
        temporal_tactile_loss = self._latent_loss(
            temporal_predictions, temporal_tactile_targets
        )
        total_loss = (
            self.spatial_tactile_loss_weight * spatial_tactile_loss
            + self.spatial_xyz_loss_weight * spatial_xyz_loss
            + self.temporal_tactile_loss_weight * temporal_tactile_loss
        )
        return {
            "loss": total_loss,
            "spatial_tactile_loss": spatial_tactile_loss,
            "spatial_xyz_loss": spatial_xyz_loss,
            "temporal_tactile_loss": temporal_tactile_loss,
        }

    def training_step(self, batch: Dict, batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.forward(batch[self.sensor_input_key], batch[self.pos_input_key])

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.forward(batch[self.sensor_input_key], batch[self.pos_input_key])

    def _log_outputs(self, outputs: Dict, stage: str, trainer_instance=None) -> None:
        if trainer_instance is None:
            return
        if stage == "train" and not trainer_instance.should_log:
            return
        step = (
            trainer_instance.step
            if stage == "train"
            else trainer_instance.global_val_step
        )
        payload = {
            f"{stage}/{name}": value
            for name, value in outputs.items()
            if name.endswith("loss")
        }
        payload[f"global_{stage}_step"] = step
        if stage == "train":
            payload["train/moving_average_decay"] = self.current_momentum
        trainer_instance.wandb.log(payload)

    def on_train_batch_end(
        self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None
    ) -> None:
        if self.momentum_scheduler is not None:
            self.current_momentum = next(self.momentum_scheduler)
        elif isinstance(self.moving_average_decay, tuple):
            self.current_momentum = self.moving_average_decay[0]
        else:
            self.current_momentum = self.moving_average_decay
        with torch.no_grad():
            update_moving_average(
                self.teacher_encoder, self.student_encoder, self.current_momentum
            )
        self._log_outputs(outputs, "train", trainer_instance)

    def on_validation_batch_end(
        self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None
    ) -> None:
        self._log_outputs(outputs, "val", trainer_instance)

    def configure_optimizers(
        self, num_iterations_per_epoch: int, num_epochs: int
    ):
        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad]
        decay_params = [parameter for parameter in trainable if parameter.dim() >= 2]
        no_decay_params = [parameter for parameter in trainable if parameter.dim() < 2]
        optimizer = self.optim_partial(
            [
                {"params": decay_params},
                {
                    "params": no_decay_params,
                    "WD_exclude": True,
                    "weight_decay": 0.0,
                },
            ]
        )

        total_steps = int(num_epochs * num_iterations_per_epoch)
        if isinstance(self.moving_average_decay, tuple):
            start, end = self.moving_average_decay
            self.momentum_scheduler = (
                start + step * (end - start) / max(total_steps, 1)
                for step in range(total_steps + 1)
            )

        if self.lr_scheduler_partial is None:
            return optimizer, None, None
        lr_scheduler = self.lr_scheduler_partial(
            optimizer=optimizer,
            T_max=total_steps,
            steps_per_epoch=num_iterations_per_epoch,
        )
        lr_cfg = {"scheduler": lr_scheduler, "interval": "step", "monitor": None}

        if self.wd_scheduler_partial is None:
            return optimizer, lr_cfg, None
        wd_scheduler = self.wd_scheduler_partial(optimizer, T_max=total_steps)
        wd_cfg = {"wd_scheduler": wd_scheduler, "interval": "step", "frequency": 1}
        return optimizer, lr_cfg, wd_cfg

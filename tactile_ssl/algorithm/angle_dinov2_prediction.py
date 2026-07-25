"""DINOv2 with future tactile-latent prediction."""

from functools import partial
from typing import Any, Dict

import torch
import torch.nn.functional as F

from .angle_dinov2 import AngleDinov2Module


class AngleDINOv2PredictionModule(AngleDinov2Module):
    """Add the temporal tactile objective from SpatiotemporalTactileJEPA."""

    def __init__(
        self,
        predictor: partial,
        context_window_size: int = 3,
        temporal_tactile_loss_weight: float = 1.0,
        prediction_loss_type: str = "smooth_l1",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        encoder = self.student_encoder_dict["backbone"]
        if encoder.sequence_length != context_window_size:
            raise ValueError(
                "encoder.sequence_length must equal context_window_size, got "
                f"{encoder.sequence_length} and {context_window_size}"
            )
        if encoder.time_chunk_size != context_window_size:
            raise ValueError(
                "encoder.time_chunk_size must equal context_window_size so the "
                "position stream produces one token per finger"
            )
        if prediction_loss_type not in {"smooth_l1", "cosine"}:
            raise ValueError(
                "prediction_loss_type must be 'smooth_l1' or 'cosine'"
            )

        self.context_window_size = int(context_window_size)
        self.required_window_size = 2 * self.context_window_size
        self.temporal_tactile_loss_weight = float(
            temporal_tactile_loss_weight
        )
        self.prediction_loss_type = prediction_loss_type
        self.num_pos_tokens = int(encoder.pos_in_dim)
        self.num_tactile_tokens = int(encoder.in_dim)
        self.num_token_positions = (
            self.num_pos_tokens + self.num_tactile_tokens
        )
        self.temporal_predictor = predictor(
            encoder_dim=encoder.embed_dim,
            num_token_positions=self.num_token_positions,
        )

    def train(self, mode: bool = True):
        result = super().train(mode)
        # Keep the EMA prediction target deterministic.
        self.teacher_encoder.eval()
        return result

    def _split_windows(
        self, sensor: torch.Tensor, pos: torch.Tensor
    ):
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
                f"Prediction requires exactly {self.required_window_size} "
                f"frames, got sensor T={sensor.shape[1]} and pos T={pos.shape[1]}"
            )
        split = self.context_window_size
        return (
            sensor[:, :split],
            pos[:, :split],
            sensor[:, split:],
            pos[:, split:],
        )

    def _prediction_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.prediction_loss_type == "smooth_l1":
            return F.smooth_l1_loss(prediction, target)
        return (1.0 - F.cosine_similarity(prediction, target, dim=-1)).mean()

    def _temporal_tactile_loss(
        self,
        current_sensor: torch.Tensor,
        current_pos: torch.Tensor,
        future_sensor: torch.Tensor,
        future_pos: torch.Tensor,
    ) -> torch.Tensor:
        student_tokens = self.student_encoder_dict[
            "backbone"
        ].forward_features(current_sensor, current_pos)["x_norm_patchtokens"]
        with torch.no_grad():
            future_tokens = self.teacher_encoder_dict[
                "backbone"
            ].forward_features(future_sensor, future_pos)["x_norm_patchtokens"]
            tactile_targets = future_tokens[:, self.num_pos_tokens :]

        batch_size = current_sensor.shape[0]
        context_indices = torch.arange(
            self.num_token_positions,
            device=current_sensor.device,
            dtype=torch.long,
        ).expand(batch_size, -1)
        tactile_indices = torch.arange(
            self.num_pos_tokens,
            self.num_token_positions,
            device=current_sensor.device,
            dtype=torch.long,
        ).expand(batch_size, -1)
        predictions = self.temporal_predictor(
            student_tokens, context_indices, tactile_indices
        )
        return self._prediction_loss(predictions, tactile_targets)

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Dict:
        sensor = batch[self.sensor_input_key]
        pos = batch[self.pos_input_key]
        current_sensor, current_pos, future_sensor, future_pos = (
            self._split_windows(sensor, pos)
        )

        current_batch = dict(batch)
        current_batch[self.sensor_input_key] = current_sensor
        current_batch[self.pos_input_key] = current_pos
        output = super().training_step(current_batch, batch_idx)

        temporal_tactile_loss = self._temporal_tactile_loss(
            current_sensor, current_pos, future_sensor, future_pos
        )
        weighted_prediction_loss = (
            self.temporal_tactile_loss_weight * temporal_tactile_loss
        )
        output["loss"] = output["loss"] + weighted_prediction_loss
        output["dino_loss"] = output["ssl_loss"]
        output["ssl_loss"] = (
            output["ssl_loss"] + weighted_prediction_loss.detach().item()
        )
        output["temporal_tactile_loss"] = temporal_tactile_loss.detach().item()
        return output

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Dict:
        return self.training_step(batch, batch_idx)

    def log_on_batch_end(
        self, outputs, stage: str = "train", trainer_instance=None
    ):
        super().log_on_batch_end(
            outputs, stage=stage, trainer_instance=trainer_instance
        )
        if trainer_instance is None:
            return
        if stage == "train" and not trainer_instance.should_log:
            return
        step = (
            trainer_instance.step
            if stage == "train"
            else trainer_instance.global_val_step
        )
        trainer_instance.wandb.log(
            {
                f"{stage}/dino_loss": outputs["dino_loss"],
                f"{stage}/temporal_tactile_loss": outputs[
                    "temporal_tactile_loss"
                ],
                f"global_{stage}_step": step,
            }
        )


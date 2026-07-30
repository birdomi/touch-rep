"""DINOv2 with masked force reconstruction.

DINO, iBOT and KoLeo are all invariance objectives: they reward representations
that stay stable under masking and cropping. Force *magnitude* is exactly the
kind of low-dimensional, high-variance signal such objectives are free to
discard, which is a plausible reason the pretrained encoders help grasp
prediction but not slip detection (slip is a force magnitude / rate problem).

This module adds a regression head that reconstructs the normalized force
vector of a sensor token from the student's fused representation.
Reconstruction cannot be solved by a magnitude-invariant feature, so the latent
is forced to keep force information that DINO alone would throw away.

``reconstruction_token_set`` selects what is reconstructed:

``visible`` (default)
    The kept slots the student *can* see. A read-out objective: every token
    must still expose its own force after 8 fusion blocks. Well posed, and the
    most direct counterweight to the invariance objectives.
``masked``
    The iBOT-masked slots, MAE style. Much harder here: force is spatially
    sparse and a masked joint's force is only weakly predictable from its
    neighbours, so this mostly fits the marginal (verified experimentally --
    it converged no better than a deliberately misaligned control target).
``all``
    Both sets.

The head reads the same student global-view forward pass DINO already runs
(via ``AngleDinov2Module._extra_losses``), so the only added cost is the head
itself -- there is no second encoder pass.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .angle_dinov2 import AngleDinov2Module


class AngleDINOv2ReconstructionModule(AngleDinov2Module):
    """Add masked force reconstruction to the DINOv2 objective."""

    extra_log_keys = (
        "reconstruction_loss",
        "reconstruction_mae",
        "reconstruction_contact_mae",
        "reconstruction_contact_frac",
    )

    #: Which of the kept sensor slots the head is asked to reconstruct.
    TOKEN_SETS = ("visible", "masked", "all")

    def __init__(
        self,
        reconstruction_loss_weight: float = 1.0,
        reconstruction_loss_type: str = "smooth_l1",
        reconstruction_token_set: str = "visible",
        reconstruction_hidden_dim: Optional[int] = None,
        reconstruction_contact_weight: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if reconstruction_loss_type not in {"smooth_l1", "mse"}:
            raise ValueError(
                "reconstruction_loss_type must be 'smooth_l1' or 'mse', got "
                f"{reconstruction_loss_type!r}"
            )
        if reconstruction_token_set not in self.TOKEN_SETS:
            raise ValueError(
                f"reconstruction_token_set must be one of {self.TOKEN_SETS}, got "
                f"{reconstruction_token_set!r}"
            )
        if reconstruction_contact_weight < 0.0:
            raise ValueError("reconstruction_contact_weight must be >= 0")

        backbone = self.student_encoder_dict["backbone"]
        if getattr(backbone, "use_null_token", False):
            raise NotImplementedError(
                "Force reconstruction assumes sensor tokens map 1:1 onto input "
                "joints; use_null_token=True expands the joint axis"
            )

        self.reconstruction_loss_weight = float(reconstruction_loss_weight)
        self.reconstruction_loss_type = reconstruction_loss_type
        self.reconstruction_token_set = reconstruction_token_set
        self.reconstruction_contact_weight = float(reconstruction_contact_weight)

        # `sensor_transform` keeps only the last temporal chunk, so one sensor
        # token carries `sensor_time_chunk_size` frames of one joint.
        self.reconstruction_chunk = int(backbone.sensor_time_chunk_size)
        out_dim = int(backbone.in_chans) * self.reconstruction_chunk

        embed_dim = int(backbone.embed_dim)
        if reconstruction_hidden_dim:
            self.reconstruction_head = nn.Sequential(
                nn.Linear(embed_dim, int(reconstruction_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(reconstruction_hidden_dim), out_dim),
            )
        else:
            self.reconstruction_head = nn.Linear(embed_dim, out_dim)

    def _reconstruction_targets(
        self, xs: torch.Tensor, global_masks: torch.Tensor
    ) -> torch.Tensor:
        """Normalized force of the joints kept by each global view.

        Args:
            xs: raw sensor input, ``(B, T, N, C)``.
            global_masks: kept joint indices, ``(G, B, n_keep)``.

        Returns:
            ``(G * B, n_keep, C * sensor_time_chunk_size)``
        """
        backbone = self.student_encoder_dict["backbone"]
        num_global, batch_size, num_keep = global_masks.shape
        chunk = self.reconstruction_chunk

        # (B, T, N, C) -> last chunk only -> (B, N, chunk * C)
        target = backbone.normalize(xs)[:, -chunk:]
        target = target.permute(0, 2, 1, 3).reshape(batch_size, target.shape[2], -1)

        target = target.unsqueeze(0).expand(num_global, -1, -1, -1).flatten(0, 1)
        index = global_masks.flatten(0, 1).unsqueeze(-1).expand(-1, -1, target.shape[-1])
        return torch.gather(target, 1, index)

    def _contact_indicator(
        self, xs: torch.Tensor, global_masks: torch.Tensor
    ) -> torch.Tensor:
        """Whether each kept joint has non-zero raw normal force, ``(G*B, n_keep)``."""
        chunk = self.reconstruction_chunk
        # Channel 0 is normal_force; invalid readings are already zero-filled.
        contact = (xs[:, -chunk:, :, 0] > 0).any(dim=1)
        contact = contact.unsqueeze(0).expand(global_masks.shape[0], -1, -1).flatten(0, 1)
        return torch.gather(contact, 1, global_masks.flatten(0, 1))

    def _extra_losses(
        self,
        xs: torch.Tensor,
        student_global_dict: Dict[str, torch.Tensor],
        global_masks: torch.Tensor,
        ibot_masks: torch.Tensor,
        **_context,
    ) -> Dict[str, tuple]:
        tokens = student_global_dict["x_norm_patchtokens"]
        num_keep = global_masks.shape[-1]
        # transform_concat concatenates [pos tokens, sensor tokens], so the
        # sensor stream is the trailing `num_keep` positions.
        sensor_tokens = tokens[:, -num_keep:]

        # `ibot_masks` marks the kept slots the encoder replaced with the mask
        # token, i.e. the positions whose force the student cannot see.
        ibot = ibot_masks.flatten(0, 1)
        if self.reconstruction_token_set == "masked":
            masked = ibot
        elif self.reconstruction_token_set == "visible":
            masked = ~ibot
        else:
            masked = torch.ones_like(ibot)
        selected = sensor_tokens[masked]
        if selected.numel() == 0:
            zero = tokens.new_zeros(())
            return {
                "reconstruction_loss": (zero, self.reconstruction_loss_weight),
                "reconstruction_mae": (zero, 0.0),
                "reconstruction_contact_mae": (zero, 0.0),
                "reconstruction_contact_frac": (zero, 0.0),
            }

        target = self._reconstruction_targets(xs, global_masks)[masked]
        prediction = self.reconstruction_head(selected)

        if self.reconstruction_loss_type == "smooth_l1":
            per_token = F.smooth_l1_loss(prediction, target, reduction="none").mean(-1)
        else:
            per_token = F.mse_loss(prediction, target, reduction="none").mean(-1)

        contact = self._contact_indicator(xs, global_masks)[masked]
        if self.reconstruction_contact_weight > 0.0:
            # Contact tokens are a minority; upweight them so the loss is not
            # dominated by predicting "no force".
            weights = 1.0 + self.reconstruction_contact_weight * contact.to(per_token.dtype)
            loss = (per_token * weights).sum() / weights.sum()
        else:
            loss = per_token.mean()

        with torch.no_grad():
            absolute_error = (prediction - target).abs().mean(-1)
            contact_mae = (
                absolute_error[contact].mean()
                if contact.any()
                else absolute_error.new_zeros(())
            )

        return {
            "reconstruction_loss": (loss, self.reconstruction_loss_weight),
            "reconstruction_mae": (absolute_error.mean(), 0.0),
            "reconstruction_contact_mae": (contact_mae, 0.0),
            "reconstruction_contact_frac": (contact.to(loss.dtype).mean(), 0.0),
        }

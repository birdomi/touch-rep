"""
MultiSensor Grasp Detection SLModule.

Extends BraincoGraspDetectionSLModule to work with MultiSensorBraincoTransformer.
The encoder expects sensor_ids and (B,1,N,40) input; BrainCo data (C=4) is
zero-padded to C=40 inside encode() so the dataloader stays unchanged.

Partial fine-tuning
-------------------
Set ``finetune_modules`` to a list of encoder submodule names to train only
those modules (plus the task head). All other encoder parameters are frozen.

Example config:
    finetune_modules: [sensor_block, patch_embed]   # sensor-specific parts only
    finetune_modules: ~                              # train all (if train_encoder=true)
"""

from typing import List, Optional
import torch
import torch.nn.functional as F

from tactile_ssl.downstream_task.brainco_grasp_sl import BraincoGraspDetectionSLModule
from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)

SENSOR_ID_BRAINCO = 0


class MultiSensorGraspDetectionSLModule(BraincoGraspDetectionSLModule):
    """BraincoGraspDetectionSLModule adapted for MultiSensorBraincoTransformer.

    Args:
        xela_num_frames:  Number of XELA frames stacked into channel dim (default 10).
        finetune_modules: List of encoder submodule name patterns to keep trainable.
                          Only effective when train_encoder=True.
                          e.g. ["sensor_block", "patch_embed"]
        **kwargs:         Forwarded to BraincoGraspDetectionSLModule.
    """

    def __init__(
        self,
        xela_num_frames: int = 10,
        finetune_modules: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.xela_num_frames = xela_num_frames
        self._max_channels = xela_num_frames * getattr(self.model_encoder, "in_chans", 4)

        if finetune_modules is not None and self.train_encoder:
            self._apply_partial_finetune(finetune_modules)

    def _apply_partial_finetune(self, finetune_modules: List[str]):
        """Freeze entire encoder, then selectively unfreeze listed submodules."""
        # Step 1: freeze everything
        self.model_encoder.requires_grad_(False)

        # Step 2: unfreeze each specified submodule
        unfrozen_params = 0
        for mod_name in finetune_modules:
            try:
                submod = self.model_encoder
                for part in mod_name.split("."):
                    submod = getattr(submod, part)
                submod.requires_grad_(True)
                n = sum(p.numel() for p in submod.parameters())
                unfrozen_params += n
                log.info(f"Partial finetune: unfrozen encoder.{mod_name}  ({n:,} params)")
            except AttributeError:
                log.warning(f"Partial finetune: '{mod_name}' not found in encoder, skipping")

        total_enc = sum(p.numel() for p in self.model_encoder.parameters())
        log.info(
            f"Partial finetune: {unfrozen_params:,} / {total_enc:,} encoder params trainable"
        )

    def encode(self, sensor, sensor_poses, mask=None):
        """Encode windowed BrainCo tactile data through MultiSensorBraincoTransformer.

        Supports both single-timestep (sequence_length=1) and multi-timestep
        (sequence_length=T>1) encoders.  When T>1 the W window frames are grouped
        into (W // T) temporal chunks; each chunk is processed jointly so the
        encoder can capture intra-window temporal dynamics.

        Args:
            sensor:       (B, W, N, C)  C=4 for BrainCo grasp data
            sensor_poses: (B, W, N, 6)
            mask:         (B, W) boolean mask for valid windows

        Returns:
            window_tokens: (B, W // T, embed_dim)  where T = encoder sequence_length
        """
        B, W, N, C = sensor.shape
        T = self.model_encoder.sequence_length   # 1 for standard, 5 for temporal

        assert W % T == 0, (
            f"Window size {W} must be divisible by encoder sequence_length {T}"
        )
        G = W // T   # number of temporal groups per sample

        sensor_input = sensor.view(B * G, T, N, C)
        poses_input  = sensor_poses.view(B * G, T, N, 6)

        # Pad channels from in_chans → max_channels (e.g. 4 → 40)
        if C < self._max_channels:
            sensor_input = F.pad(sensor_input, (0, self._max_channels - C))

        # All samples are BrainCo (sensor_id=0)
        sensor_ids = torch.zeros(B * G, dtype=torch.long, device=sensor.device)

        with torch.no_grad() if not self.train_encoder else torch.enable_grad():
            out = self.model_encoder.forward_features(
                sensor_input, poses_input, sensor_ids=sensor_ids
            )
            x_tokens = out["x_tokens"]   # (B*G, num_tokens, embed_dim)

        pooled_tokens = self.pooler(x_tokens)          # (B*G, 1, embed_dim)
        window_tokens = pooled_tokens.view(B, G, -1)   # (B, G, embed_dim)

        if mask is not None:
            # mask: (B, W) → aggregate to (B, G) by taking any-valid per group
            mask_grouped = mask.view(B, G, T).any(dim=-1).float()  # (B, G)
            window_tokens = window_tokens * mask_grouped.unsqueeze(-1)

        return window_tokens

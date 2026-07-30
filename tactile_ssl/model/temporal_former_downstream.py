"""Downstream-facing view of the temporal frame-encoder + former stack.

``BraincoAngleGraspSLModule.encode`` expects an encoder that

* exposes ``sequence_length``, used to split the downstream window into
  consecutive encoder windows, and
* implements ``forward_features(xs, pos)`` on raw ``(B, T, N, C)`` input,
  returning a dict containing ``x_tokens``.

The pretraining stack deliberately splits those two steps
(``encode_frames`` then ``forward_features`` on tokens) so temporal crops can
be gathered without re-encoding frames. This subclass re-joins them behind the
interface the downstream module already uses.

Parameter names are unchanged (``frame_encoder.*`` / ``former.*``), so
``SLModule.load_encoder`` matches every tensor of a temporal pretraining
checkpoint under the ``teacher_encoder.backbone.`` prefix with no silent
shape-mismatch skips.
"""

from typing import Dict

import torch

from .temporal_former import FrameTemporalEncoder, TemporalTokenFormer


class TemporalDownstreamEncoder(FrameTemporalEncoder):
    """``FrameTemporalEncoder`` with the SLModule encoder interface."""

    def __init__(self, *args, sequence_length: int = 15, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if int(sequence_length) <= 0:
            raise ValueError("sequence_length must be positive")
        # encode() divides the downstream window by this, so the whole window
        # reaches the former at once when they are equal.
        self.sequence_length = int(sequence_length)

    def forward_features(  # type: ignore[override]
        self, xs: torch.Tensor, pos: torch.Tensor, **_unused
    ) -> Dict[str, torch.Tensor]:
        """Raw ``(B, T, N, C)`` in, standard token dict out."""
        tokens = self.encode_frames(xs, pos)
        frame_index = (
            torch.arange(tokens.shape[1], device=tokens.device)
            .unsqueeze(0)
            .expand(tokens.shape[0], -1)
        )
        out = self.former.forward_features(tokens, frame_index)
        out["x_tokens"] = torch.cat(
            [out["x_norm_regtokens"], out["x_norm_patchtokens"]], dim=1
        )
        out["x_prenorm"] = out["x_norm_patchtokens"]
        return out


def temporal_tiny_downstream(
    frame_encoder,
    depth: int = 4,
    num_heads: int = 3,
    mlp_ratio: float = 4.0,
    num_register_tokens: int = 1,
    drop_path_rate: float = 0.0,
    frame_token_set: str = "all",
    sequence_length: int = 15,
    **_stats,
) -> TemporalDownstreamEncoder:
    """Factory mirroring ``temporal_tiny`` for downstream task configs."""
    former = TemporalTokenFormer(
        embed_dim=frame_encoder.embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        num_register_tokens=num_register_tokens,
        drop_path_rate=drop_path_rate,
    )
    return TemporalDownstreamEncoder(
        frame_encoder,
        former,
        frame_token_set=frame_token_set,
        sequence_length=sequence_length,
    )

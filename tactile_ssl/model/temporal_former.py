"""Frame encoder + temporal former, trained jointly end to end.

The frame encoder (an ``AngleTransformer`` with ``sequence_length=1``) turns
each frame into a small set of tokens. The temporal former then attends over
the whole ``(frame, token)`` grid and emits its own CLS/register tokens plus
per-position patch tokens.

Nothing here is frozen: both halves are plain submodules of one
``nn.Module``, so ``DINOv2Module`` picks up every parameter in
``configure_optimizers`` and the EMA teacher tracks the whole composite.

Independent of the existing frame-wise stack -- this module only *imports*
from ``tactile_ssl.model.layers`` and never modifies it. The former uses the
2-axis ``(frame_index, token_index)`` RoPE that ``layers.attention._apply_rope``
already supports for N-axis positions, so no attention code changes either.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .layers.block import Block

# Register/CLS positions are excluded from RoPE with a negative coordinate,
# matching the convention in angle_transformer._flatten_rope_positions.
_ROPE_NULL = -1.0


class TemporalTokenFormer(nn.Module):
    """Transformer over a ``(frames x tokens-per-frame)`` grid."""

    def __init__(
        self,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 1,
        drop_path_rate: float = 0.0,
        qkv_bias: bool = True,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_register_tokens = int(num_register_tokens)

        self.register_tokens = nn.Parameter(
            torch.zeros(1, self.num_register_tokens, self.embed_dim)
        )
        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim))
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=drop_path_rate,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

        self.init_std = init_std
        nn.init.trunc_normal_(self.register_tokens, std=init_std)
        nn.init.trunc_normal_(self.mask_token, std=init_std)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _rope_positions(
        self, frame_index: torch.Tensor, tokens_per_frame: int
    ) -> torch.Tensor:
        """``(B, R + L*K, 2)`` of (frame index, token index) coordinates."""
        batch, num_frames = frame_index.shape
        device = frame_index.device
        token_index = torch.arange(tokens_per_frame, device=device, dtype=torch.float32)

        frames = frame_index.to(torch.float32).unsqueeze(-1).expand(-1, -1, tokens_per_frame)
        tokens = token_index.view(1, 1, -1).expand(batch, num_frames, -1)
        grid = torch.stack([frames, tokens], dim=-1).flatten(1, 2)

        register = grid.new_full((batch, self.num_register_tokens, 2), _ROPE_NULL)
        return torch.cat([register, grid], dim=1)

    def forward_features(
        self,
        tokens: torch.Tensor,
        frame_index: torch.Tensor,
        masktoken_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Attend over the frame/token grid.

        Args:
            tokens: ``(B, L, K, D)`` frame-encoder tokens for the L frames of
                this crop.
            frame_index: ``(B, L)`` index of each frame in the source window.
                Absolute indices keep crops of different lengths on a common
                time axis.
            masktoken_masks: ``(B, L, K)`` bool; True positions are replaced by
                the learned mask token before attention (iBOT).

        Returns:
            ``x_norm_regtokens`` ``(B, R, D)`` and ``x_norm_patchtokens``
            ``(B, L*K, D)``, matching the keys the DINOv2 modules consume.
        """
        if tokens.dim() != 4:
            raise ValueError(f"Expected tokens (B, L, K, D), got {tuple(tokens.shape)}")
        batch, num_frames, tokens_per_frame, _ = tokens.shape

        if masktoken_masks is not None:
            tokens = torch.where(
                masktoken_masks.unsqueeze(-1),
                self.mask_token.to(tokens.dtype).view(1, 1, 1, -1),
                tokens,
            )

        rope_positions = self._rope_positions(frame_index, tokens_per_frame)
        x = tokens.flatten(1, 2)
        x = torch.cat([self.register_tokens.expand(batch, -1, -1), x], dim=1)

        for block in self.blocks:
            x = block(x, None, rope_positions=rope_positions)
        x = self.norm(x)

        r = self.num_register_tokens
        return {"x_norm_regtokens": x[:, :r], "x_norm_patchtokens": x[:, r:]}


class FrameTemporalEncoder(nn.Module):
    """Per-frame encoder followed by :class:`TemporalTokenFormer`.

    ``encode_frames`` is deliberately separate from ``forward_features``: the
    frame encoder is frame-independent, so a caller can encode a whole window
    once and then gather several temporal crops from the result instead of
    re-encoding overlapping frames. That is exact, not an approximation.
    """

    def __init__(
        self,
        frame_encoder: nn.Module,
        former: TemporalTokenFormer,
        frame_token_set: str = "all",
    ) -> None:
        super().__init__()
        if frame_token_set not in {"all", "sensor", "cls"}:
            raise ValueError("frame_token_set must be 'all', 'sensor', or 'cls'")
        if int(getattr(frame_encoder, "sequence_length", 1)) != 1:
            raise ValueError(
                "FrameTemporalEncoder expects a frame-wise encoder "
                f"(sequence_length=1), got {frame_encoder.sequence_length}"
            )
        if frame_encoder.embed_dim != former.embed_dim:
            raise ValueError(
                f"embed_dim mismatch: frame encoder {frame_encoder.embed_dim} "
                f"vs former {former.embed_dim}"
            )

        self.frame_encoder = frame_encoder
        self.former = former
        self.frame_token_set = frame_token_set
        self.embed_dim = int(frame_encoder.embed_dim)
        self.num_register_tokens = former.num_register_tokens
        # patchtokens are laid out [pos tokens, sensor tokens]; with
        # sequence_length=1 there is one pos token per finger.
        self.num_pos_tokens = int(frame_encoder.pos_in_dim)
        # A joint-only frame encoder emits no sensor tokens at all.
        self.num_sensor_tokens = (
            0 if getattr(frame_encoder, "pos_only", False) else int(frame_encoder.in_dim)
        )
        if frame_token_set == "sensor" and self.num_sensor_tokens == 0:
            raise ValueError(
                "frame_token_set='sensor' needs a sensor stream, but the frame "
                "encoder was built with input_streams='pos'"
            )
        if frame_token_set == "cls":
            # One token per frame: the frame encoder's own register/CLS
            # summary. The former then attends over a (frames x 1) grid, so
            # iBOT span masking becomes "recover the missing frames' summaries
            # from the surrounding time".
            self.tokens_per_frame = 1
        elif frame_token_set == "sensor":
            self.tokens_per_frame = self.num_sensor_tokens
        else:
            self.tokens_per_frame = self.num_pos_tokens + self.num_sensor_tokens

    def encode_frames(self, xs: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """``(B, W, N, C)`` and ``(B, W, P, Cp)`` -> ``(B, W, K, D)``."""
        batch, window = xs.shape[:2]
        out = self.frame_encoder.forward_features(
            xs.reshape(batch * window, 1, *xs.shape[2:]),
            pos.reshape(batch * window, 1, *pos.shape[2:]),
        )
        if self.frame_token_set == "cls":
            flat = out["x_norm_regtokens"][:, :1]
        else:
            flat = out["x_norm_patchtokens"]
            if self.frame_token_set == "sensor":
                flat = flat[:, self.num_pos_tokens:]
        return flat.view(batch, window, flat.shape[-2], flat.shape[-1])

    def forward_features(
        self,
        tokens: torch.Tensor,
        frame_index: torch.Tensor,
        masktoken_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.former.forward_features(tokens, frame_index, masktoken_masks)

    def forward(self, xs: torch.Tensor, pos: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full-window encode, for downstream use."""
        tokens = self.encode_frames(xs, pos)
        frame_index = torch.arange(
            tokens.shape[1], device=tokens.device
        ).unsqueeze(0).expand(tokens.shape[0], -1)
        return self.forward_features(tokens, frame_index)

    def update_stats(self, *args, **kwargs) -> None:
        """Delegate dataset normalization stats to the frame encoder.

        train.py's ``_maybe_update_encoder_stats`` calls this on the DINOv2
        backbone, which here is the composite; the tensors belong one level
        down.
        """
        self.frame_encoder.update_stats(*args, **kwargs)


def temporal_tiny(
    frame_encoder: nn.Module,
    depth: int = 4,
    num_heads: int = 3,
    mlp_ratio: float = 4.0,
    num_register_tokens: int = 1,
    drop_path_rate: float = 0.0,
    frame_token_set: str = "all",
    **_stats,
) -> FrameTemporalEncoder:
    """Factory used by the Hydra config.

    ``**_stats`` swallows the ``normalization`` / ``pos_normalization`` keys
    that train.py writes onto ``cfg.algorithm.encoder`` before instantiation.
    They belong to the frame encoder, which already receives them through its
    own ``normalization: ${data.normalization}`` interpolation and then
    authoritatively via :meth:`FrameTemporalEncoder.update_stats`.
    """
    former = TemporalTokenFormer(
        embed_dim=frame_encoder.embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        num_register_tokens=num_register_tokens,
        drop_path_rate=drop_path_rate,
    )
    return FrameTemporalEncoder(frame_encoder, former, frame_token_set=frame_token_set)

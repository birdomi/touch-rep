"""AngleTransformer: dual-stream transformer for hand angle + joint contact data.

Architecture
------------
Two asymmetric streams:
  sensor : joint_contact  (B, T, N_joints=42,  C=1) — contact per joint
  pos    : finger_angles  (B, T, N_fingers=10, C=4) — finger angle vectors

Both streams are independently compressed via PatchEmbed1d (Conv1d), then:
  1. sensor_block  : pre-fusion self-attention on sensor tokens
  2. Concat fusion : cat([pos, sen]) → shared blocks → output

No wrist_poses or sensor_id routing.
"""

from functools import partial
from typing import Callable, List, Literal, Optional

import einops
import torch
import torch.nn as nn
from omegaconf import DictConfig

from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.model import SignalTransformer
from .layers import MemEffAttention, Mlp, PatchEmbed1d
from .layers import NestedTensorBlock as Block

log = get_pylogger(__name__)

# Full skeleton layout (42 joints) — same as MultiSensorTransformer
# Fingertip touch-link indices where tactile sensors are physically placed
# Left hand:  thumb=4, index=8,  middle=12, ring=16, pinky=20
# Right hand: thumb=25,index=29, middle=33, ring=37, pinky=41
FULL_SKELETON_SIZE = 42
TACTILE_SENSOR_IDXS = [4, 8, 12, 16, 20, 25, 29, 33, 37, 41]


def _make_embed1d(in_chans: int, seq_len: int, chunk_size: int, embed_dim: int) -> PatchEmbed1d:
    return PatchEmbed1d(
        modal_chans=in_chans,
        modal_lens=seq_len,
        chunk_size=chunk_size,
        embed_dim=embed_dim,
        padding=0,
    )


def _apply_embed1d(x: torch.Tensor, embed: PatchEmbed1d, B: int) -> torch.Tensor:
    """(B, T, N, C) → PatchEmbed1d per sensor → (B, num_chunks, N, D)."""
    x = einops.rearrange(x, "b t n c -> (b n) c t")
    x = embed(x)                              # (B*N, D, num_chunks)
    x = einops.rearrange(x, "(b n) c t -> b t n c", b=B)
    return x


class AngleTransformer(SignalTransformer):
    """Dual-stream Transformer for hand angle + contact pretraining.

    Args:
        in_dim          : N_joints for contact stream (e.g. 42)
        in_chans        : C for contact stream (e.g. 1)
        pos_in_dim      : N_fingers for angle stream (e.g. 10)
        pos_in_chans    : C for angle stream (e.g. 4)
        sequence_length : input time length T
        time_chunk_size : Conv1d stride (T → T//chunk tokens)
        embed_dim       : transformer hidden dim
        depth           : shared fusion block count
        pre_fusion_depth: sensor_block depth (default depth//4)
        normalization   : DictConfig(mean, std) for contact stream, shape (in_chans,)
        pos_normalization: DictConfig(mean, std) for angle stream, shape (pos_in_chans,)
        fine_tune_sensor_shallow_blocks:
            number of shallow shared fusion blocks to train with fine_tune_sensor
    """

    def __init__(
        self,
        in_dim: int = 42,
        in_chans: int = 1,
        pos_in_dim: int = 10,
        pos_in_chans: int = 4,
        sequence_length: int = 1,
        time_chunk_size: int = 1,
        embed_dim: int = 192,
        depth: int = 8,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        ffn_layer: str = "mlp",
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        pos_embed_fn: Literal["sinusoidal", "learned", "rope"] = "learned",
        init_values: Optional[float] = None,
        num_register_tokens: int = 1,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        with_masktoken: bool = False,
        use_null_token: bool = False,
        causal: bool = False,
        pre_fusion_depth: Optional[int] = None,
        normalization: Optional[DictConfig] = None,
        fine_tune_sensor: bool = False,
        fine_tune_sensor_shallow_blocks: Optional[int] = 0,
    ):
        assert sequence_length % time_chunk_size == 0, (
            f"sequence_length({sequence_length}) must be divisible by time_chunk_size({time_chunk_size})"
        )
        assert in_dim % 2 == 0, f"in_dim({in_dim}) must be even (left/right hand)"
        assert pos_in_dim % 2 == 0, f"pos_in_dim({pos_in_dim}) must be even (left/right hand)"

        self.use_null_token = use_null_token
        if use_null_token:
            with_masktoken = True

        super().__init__(
            in_dim=in_dim,
            in_chans=in_chans,
            sequence_length=sequence_length,
            time_chunk_size=time_chunk_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            ffn_layer=ffn_layer,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            act_layer=act_layer,
            norm_layer=norm_layer,
            pos_embed_fn=pos_embed_fn,
            init_values=init_values,
            num_register_tokens=num_register_tokens,
            drop_path_rate=drop_path_rate,
            drop_path_uniform=drop_path_uniform,
            with_masktoken=with_masktoken,
            causal=causal,
        )

        self.in_dim      = in_dim
        self.in_chans    = in_chans
        self.pos_in_dim  = pos_in_dim
        self.pos_in_chans = pos_in_chans
        self.use_rope = pos_embed_fn == "rope"
        self.pre_fusion_depth = pre_fusion_depth if pre_fusion_depth is not None else depth // 4
        self.fine_tune_sensor_shallow_blocks = (
            self.pre_fusion_depth
            if fine_tune_sensor_shallow_blocks is None
            else int(fine_tune_sensor_shallow_blocks)
        )
        if self.fine_tune_sensor_shallow_blocks < 0:
            raise ValueError(
                "fine_tune_sensor_shallow_blocks must be >= 0, "
                f"got {self.fine_tune_sensor_shallow_blocks}"
            )

        D     = embed_dim
        chunk = time_chunk_size
        seq   = sequence_length

        # ── Positional embeddings ──────────────────────────────────────────────
        if self.use_rope:
            self.contact_pos_embed = None
            self.angle_pos_embed = None
            self.hand_embed = None
        else:
            self.contact_pos_embed = nn.Parameter(torch.zeros(2, in_dim // 2, D))
            self.angle_pos_embed = nn.Parameter(torch.zeros(2, pos_in_dim // 2, D))
            self.hand_embed = nn.Parameter(torch.zeros(2, D))

        # ── PatchEmbed1d ───────────────────────────────────────────────────────
        self.sensor_embed    = _make_embed1d(in_chans,     seq, chunk, D)  # contact
        self.angle_embed = _make_embed1d(pos_in_chans, seq, chunk, D)  # angles

        # ── Pre-fusion sensor blocks ───────────────────────────────────────────
        self.sensor_block = nn.ModuleList([
            Block(
                attn_class=MemEffAttention,
                dim=D,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=0.0,
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=Mlp,
                init_values=init_values,
            )
            for _ in range(self.pre_fusion_depth)
        ])

        # ── Normalization buffers ──────────────────────────────────────────────
        if normalization is not None:
            m = torch.tensor(normalization.mean, dtype=torch.float32)
            s = torch.tensor(normalization.std,  dtype=torch.float32)
        else:
            m = torch.zeros(in_chans)
            s = torch.ones(in_chans)
        self.register_buffer("signal_mean", m)  # (in_chans,)
        self.register_buffer("signal_std",  s)  # (in_chans,)


        self.init_weights()
        if self.hand_embed is not None:
            nn.init.trunc_normal_(self.hand_embed, std=0.02)

        self.fine_tune_sensor = fine_tune_sensor
        if fine_tune_sensor:
            self._apply_fine_tune_sensor()

        num_chunks = seq // chunk
        log.info(
            f"AngleTransformer: T={seq}, chunk={chunk}, num_chunks={num_chunks}, "
            f"N_contact={in_dim}, N_angles={pos_in_dim}, "
            f"pre_fusion_depth={self.pre_fusion_depth}, embed_dim={D}, "
            f"pos_embed_fn={pos_embed_fn}, "
            f"use_null_token={self.use_null_token}, "
            f"fine_tune_sensor={self.fine_tune_sensor}, "
            f"fine_tune_sensor_shallow_blocks={self.fine_tune_sensor_shallow_blocks}"
        )

    # ── fine-tune mode ────────────────────────────────────────────────────────

    def _apply_fine_tune_sensor(self) -> None:
        """Freeze all parameters except sensor path and shallow fusion blocks."""
        trainable_modules = {"sensor_embed", "sensor_block"}
        trainable_params = {"contact_pos_embed"}
        shallow_blocks = min(self.fine_tune_sensor_shallow_blocks, len(self.blocks))

        for name, param in self.named_parameters():
            top = name.split(".", 1)[0]
            is_shallow_block = False
            if top == "blocks":
                parts = name.split(".", 2)
                is_shallow_block = len(parts) > 1 and int(parts[1]) < shallow_blocks

            if top in trainable_modules or top in trainable_params or is_shallow_block:
                param.requires_grad = True
            else:
                param.requires_grad = False

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        log.info(
            f"fine_tune_sensor=True: unfroze contact_pos_embed, sensor_embed, "
            f"sensor_block, blocks[0:{shallow_blocks}] "
            f"({n_trainable:,} / {n_total:,} params trainable)"
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _full_embed(self, per_hand: torch.Tensor) -> torch.Tensor:
        """Combine per-hand positional embed with hand bias.

        Args:
            per_hand : (2, N//2, D) — index 0=lh, 1=rh

        Returns:
            (N, D) full positional embedding
        """
        h  = self.hand_embed.float()              # (2, D)
        lh = per_hand[0].float() + h[0]           # (N//2, D)
        rh = per_hand[1].float() + h[1]           # (N//2, D)
        return torch.cat([lh, rh], dim=0)         # (N, D)

    def _rope_positions(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        b, t, n = x.shape[:3]
        ids = torch.arange(offset, offset + n, device=x.device, dtype=torch.long)
        return ids.view(1, 1, n).expand(b, t, n)

    def _apply_position_masks(self, pos_ids: torch.Tensor, masks) -> torch.Tensor:
        all_pos = []
        t = pos_ids.shape[1]
        for mask in masks:
            mask = mask.to(device=pos_ids.device, dtype=torch.long)
            mask_keep = einops.repeat(mask, "b n -> b t n", t=t)
            all_pos.append(torch.gather(pos_ids, dim=-1, index=mask_keep))
        return torch.cat(all_pos, dim=0)

    def _flatten_rope_positions(self, pos_ids: Optional[torch.Tensor], skip_register: bool) -> Optional[torch.Tensor]:
        if pos_ids is None:
            return None
        pos_ids = einops.rearrange(pos_ids, "b t n -> b (t n)")
        if self.register_tokens is not None and not skip_register:
            reg = pos_ids.new_full((pos_ids.shape[0], self.num_register_tokens), -1)
            pos_ids = torch.cat([reg, pos_ids], dim=1)
        return pos_ids

    # ── null token expansion ──────────────────────────────────────────────────

    def expand_to_skeleton(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand tactile-only contact input to full skeleton space.

        Args:
            x: (B, T, 10, C)  — tactile contact for the 10 fingertips

        Returns:
            x_full        : (B, T, 42, C)   — zeros at non-tactile positions
            masktoken_mask: (1, B, 42) bool  — True = null position (no sensor)
        """
        B, T, N, C = x.shape
        assert N == len(TACTILE_SENSOR_IDXS), (
            f"Expected {len(TACTILE_SENSOR_IDXS)} tactile sensors, got {N}"
        )
        x_full = torch.zeros(B, T, FULL_SKELETON_SIZE, C, device=x.device, dtype=x.dtype)
        x_full[:, :, TACTILE_SENSOR_IDXS, :] = x

        null_mask = torch.ones(B, FULL_SKELETON_SIZE, dtype=torch.bool, device=x.device)
        null_mask[:, TACTILE_SENSOR_IDXS] = False
        masktoken_mask = null_mask.unsqueeze(0)  # (1, B, 42)

        return x_full, masktoken_mask

    # ── normalization ─────────────────────────────────────────────────────────

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize contact stream in-place clone."""
        x = x.clone()
        x = (x - self.signal_mean) / self.signal_std.clamp(min=1e-6)
        return x

    # ── embedding ─────────────────────────────────────────────────────────────

    def pre_sensor_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Zero-fill negatives → normalize → PatchEmbed1d.

        Args:
            x : (B, T, N_joints, C)

        Returns:
            (B, num_chunks, N_joints, D)
        """
        B = x.shape[0]
        x = x.clone()
        # x[x < 0] = 0.0
        # print(x, self.signal_mean, self.signal_std)
        x = self.normalize(x)
        return _apply_embed1d(x, self.sensor_embed, B)

    def pre_pos_embed(self, pos: torch.Tensor) -> torch.Tensor:
        """PatchEmbed1d for angle stream (no normalization).

        Args:
            pos : (B, T, N_fingers, pos_in_chans)

        Returns:
            (B, num_chunks, N_fingers, D)
        """
        return _apply_embed1d(pos, self.angle_embed, pos.shape[0])

    # ── token preparation ─────────────────────────────────────────────────────

    def prepare_tokens_with_mask(
        self,
        x,
        masks,
        mask_type: Optional[Literal["block", "tubelet"]],
        masktoken_masks: Optional[List[torch.Tensor]],
        joint_embed: Optional[torch.Tensor] = None,
        skip_register: bool = False,
        rope_offset: int = 0,
    ):
        """Flatten + optional learned PE/RoPE ids + masking + register token."""
        rope_positions = self._rope_positions(x, rope_offset) if self.use_rope else None

        if joint_embed is not None:
            n = x.shape[-2]
            x = x + joint_embed.view(1, 1, n, -1)

        if masks is not None:
            if mask_type in ("tubelet", "block"):
                x = self.apply_tubelet_masks(x, masks)
                if rope_positions is not None:
                    rope_positions = self._apply_position_masks(rope_positions, masks)
            else:
                raise NotImplementedError(f"Unknown mask type: {mask_type}")

        attn_bias = self.create_causal_mask(x) if self.causal else None

        if masktoken_masks is not None:
            x = self.apply_masktokens(x, masktoken_masks)

        x = einops.rearrange(x, "b t n c -> b (t n) c")
        rope_positions = self._flatten_rope_positions(rope_positions, skip_register)
        if self.register_tokens is not None and not skip_register:
            x = torch.cat([self.register_tokens.expand(x.shape[0], -1, -1), x], dim=1)

        return x, attn_bias, rope_positions

    # ── pre-fusion sensor blocks ───────────────────────────────────────────────

    def sensor_transform(self, x: torch.Tensor, _bias, rope_positions=None) -> torch.Tensor:
        """Pre-fusion self-attention over contact tokens."""
        for blk in self.sensor_block:
            kwargs = {} if rope_positions is None else {"rope_positions": rope_positions}
            x = blk(x, _bias, **kwargs)
        return x

    # ── fusion ────────────────────────────────────────────────────────────────

    def transform_concat(
        self,
        sen: torch.Tensor,
        pos: torch.Tensor,
        sen_mask,
        pos_mask,
        bias,
        sen_rope_positions=None,
        pos_rope_positions=None,
    ):
        """Concat sensor and angle streams → fusion blocks → norm.

        Args:
            sen     : (B_eff, N_contact_keep,  D)  — after sensor_transform
            pos     : (B_eff, reg+N_angle_keep, D)  — with register token
            sen_mask: sensor stream mask indices (or None)
            pos_mask: angle stream mask indices (or None)
            bias    : attention bias

        Returns:
            x_prenorm, x_postnorm : both (B_eff, reg+N_contact+N_angle, D)
        """
        rope_positions = None
        if self.use_rope:
            rope_positions = torch.cat([pos_rope_positions, sen_rope_positions], dim=1)
        else:
            sen_pe = self._full_embed(self.contact_pos_embed)
            ang_pe = self._full_embed(self.angle_pos_embed)

            if sen_mask is not None:
                ns = sen_mask.shape[-1]
                se = sen_pe[sen_mask.view(-1, ns)]
                sen = sen + se

                np_ = pos_mask.shape[-1]
                ae = ang_pe[pos_mask.view(-1, np_)]
                pos[:, 1:] = pos[:, 1:] + ae
            else:
                sen = sen + sen_pe
                pos[:, 1:] = pos[:, 1:] + ang_pe

        fused = torch.cat([pos, sen], dim=1)     # (B_eff, reg+N_a+N_s, D)

        # pos only
        # fused = pos[:, 1:]
        
        # sen only
        # fused = sen

        for blk in self.blocks:
            kwargs = {} if rope_positions is None else {"rope_positions": rope_positions}
            fused = blk(fused, None, **kwargs)

        x_norm = self.norm(fused)
        return x_norm, x_norm

    # ── forward ───────────────────────────────────────────────────────────────

    def forward_features(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        mask_type: Optional[Literal["block", "tubelet"]] = None,
        masktoken_masks: Optional[List[torch.Tensor]] = None,
        pos_masks: Optional[torch.Tensor] = None,
    ) -> dict:
        """Full forward pass.

        Args:
            x         : (B, T, N_joints=42,  C=1)  — joint contact
            pos       : (B, T, N_fingers=10, C=4)  — finger angles
            masks     : sensor stream masking
            mask_type : "block" or "tubelet"
            masktoken_masks: mask token positions
            pos_masks : angle stream masking (defaults to masks if None)

        Returns:
            dict with keys:
              x_norm_regtokens   : (B_eff, num_register_tokens, D)
              x_norm_patchtokens : (B_eff, num_patches, D)
              x_prenorm          : (B_eff, num_patches, D)
              x_tokens           : (B_eff, reg+patches, D)
        """
        # --- null token expansion: (B,T,10,C) → (B,T,42,C) + mask -----------
        if self.use_null_token:
            x, null_mask = self.expand_to_skeleton(x)
            if masktoken_masks is None:
                masktoken_masks = null_mask
            else:
                masktoken_masks = masktoken_masks | null_mask

        # --- embed streams ---------------------------------------------------
        x   = self.pre_sensor_embed(x)
        pos = self.pre_pos_embed(pos)

        # --- positional embeddings (pre-fusion stage) ------------------------
        sen_pe = None if self.use_rope else self._full_embed(self.contact_pos_embed)
        ang_pe = None if self.use_rope else self._full_embed(self.angle_pos_embed)

        _pos_masks = pos_masks if pos_masks is not None else masks

        x, bias, sen_rope_positions = self.prepare_tokens_with_mask(
            x, masks, mask_type, masktoken_masks,
            joint_embed=sen_pe, skip_register=True, rope_offset=0,
        )
        pos, _, pos_rope_positions = self.prepare_tokens_with_mask(
            pos, _pos_masks, mask_type, None,
            joint_embed=ang_pe, skip_register=False, rope_offset=self.in_dim,
        )

        # --- pre-fusion sensor attention -------------------------------------
        sen = self.sensor_transform(x, bias, rope_positions=sen_rope_positions)

        # --- fusion ----------------------------------------------------------
        x_prenorm, x_postnorm = self.transform_concat(
            sen, pos, masks, _pos_masks, bias,
            sen_rope_positions=sen_rope_positions,
            pos_rope_positions=pos_rope_positions,
        )

        r = self.num_register_tokens
        return {
            "x_norm_regtokens":   x_postnorm[:, :r],
            "x_norm_patchtokens": x_postnorm[:, r:],
            "x_prenorm":          x_prenorm[:, r:],
            "x_tokens":           x_postnorm,
        }

    def forward(self, x, pos, **kwargs):
        return self.forward_features(x, pos, **kwargs)["x_norm_patchtokens"]

    def update_stats(
        self,
        signal_mean: torch.Tensor,
        signal_std: torch.Tensor,
        **_kwargs,  # ignore pos_mean/pos_std — angles are not normalized
    ):
        self.signal_mean = signal_mean
        self.signal_std  = signal_std


# ── factory functions ─────────────────────────────────────────────────────────

def angle_tiny(
    in_dim: int = 42,
    in_chans: int = 1,
    pos_in_dim: int = 10,
    pos_in_chans: int = 4,
    sequence_length: int = 1,
    time_chunk_size: int = 1,
    depth: int = 8,
    num_register_tokens: int = 1,
    **kwargs,
) -> AngleTransformer:
    """AngleTransformer tiny (embed_dim=192, depth=8)."""
    return AngleTransformer(
        in_dim=in_dim,
        in_chans=in_chans,
        pos_in_dim=pos_in_dim,
        pos_in_chans=pos_in_chans,
        sequence_length=sequence_length,
        time_chunk_size=time_chunk_size,
        embed_dim=192,
        depth=depth,
        num_heads=3,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def angle_small(
    in_dim: int = 42,
    in_chans: int = 1,
    pos_in_dim: int = 10,
    pos_in_chans: int = 4,
    sequence_length: int = 1,
    time_chunk_size: int = 1,
    depth: int = 12,
    num_register_tokens: int = 1,
    **kwargs,
) -> AngleTransformer:
    """AngleTransformer small (embed_dim=384, depth=12)."""
    return AngleTransformer(
        in_dim=in_dim,
        in_chans=in_chans,
        pos_in_dim=pos_in_dim,
        pos_in_chans=pos_in_chans,
        sequence_length=sequence_length,
        time_chunk_size=time_chunk_size,
        embed_dim=384,
        depth=depth,
        num_heads=6,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )

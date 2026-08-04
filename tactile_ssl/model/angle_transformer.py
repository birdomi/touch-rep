"""AngleTransformer: dual-stream transformer for hand angle + joint contact data.

Architecture
------------
Two asymmetric streams:
  sensor : joint_contact (B, T, N_joints=42, C=4) — force per joint
  pos    : finger_xyz   (B, T, N_fingers=10, C=3) — fingertip positions

The sensor stream keeps one token per frame and joint. Pre-fusion uses the last
frame as queries and all frames as keys/values, then concatenates with XYZ.

When ``pos_embed_fn="rope"``, the 10 angle tokens are encoded with multimodal
RoPE coordinates ``(hand, finger, -1)`` while the 4 joint values stay as token
channels.

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
from .layers import CrossAttentionBlock, PatchEmbed1d

log = get_pylogger(__name__)

# Full skeleton layout (42 joints) — same as MultiSensorTransformer
# Fingertip touch-link indices where tactile sensors are physically placed
# Left hand:  thumb=4, index=8,  middle=12, ring=16, pinky=20
# Right hand: thumb=25,index=29, middle=33, ring=37, pinky=41
FULL_SKELETON_SIZE = 42
SKELETON_JOINTS_PER_FINGER = 4
TACTILE_SENSOR_IDXS = [4, 8, 12, 16, 20, 25, 29, 33, 37, 41]
ROPE_AXIS_NAMES = ("hand", "finger", "joint")


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
        sensor_time_chunk_size: Optional[int] = None,
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
        pos_embed_fn: Literal["sinusoidal", "learned", "rope", "multimodal_rope", "mrope"] = "learned",
        init_values: Optional[float] = None,
        num_register_tokens: int = 1,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        with_masktoken: bool = False,
        use_null_token: bool = False,
        causal: bool = False,
        pre_fusion_depth: Optional[int] = None,
        normalization: Optional[DictConfig] = None,
        pos_normalization: Optional[DictConfig] = None,
        fine_tune_sensor: bool = False,
        fine_tune_sensor_shallow_blocks: Optional[int] = 0,
        rope_hand_offset: int = 100,
        debug_rope: bool = False,
        input_streams: Literal["both", "pos"] = "both",
    ):
        assert sequence_length % time_chunk_size == 0, (
            f"sequence_length({sequence_length}) must be divisible by time_chunk_size({time_chunk_size})"
        )
        sensor_time_chunk_size = (
            time_chunk_size
            if sensor_time_chunk_size is None
            else int(sensor_time_chunk_size)
        )
        assert sequence_length % sensor_time_chunk_size == 0, (
            f"sequence_length({sequence_length}) must be divisible by "
            f"sensor_time_chunk_size({sensor_time_chunk_size})"
        )
        assert in_dim % 2 == 0, f"in_dim({in_dim}) must be even (left/right hand)"
        assert pos_in_dim % 2 == 0, f"pos_in_dim({pos_in_dim}) must be even (left/right hand)"

        if input_streams not in {"both", "pos"}:
            raise ValueError(
                f"input_streams must be 'both' or 'pos', got {input_streams!r}"
            )
        # "pos": joint-only ablation. The sensor stream is never embedded, so
        # no force/proximity token ever enters the network -- the sequence is
        # registers + pose tokens and nothing else. sensor_embed and
        # sensor_block still exist (so state dicts stay loadable both ways) but
        # receive no input and therefore no gradient.
        self.pos_only = input_streams == "pos"
        if self.pos_only and use_null_token:
            raise ValueError("use_null_token applies to the sensor stream; "
                             "it cannot be combined with input_streams='pos'")

        self.use_null_token = use_null_token
        self.debug_rope = bool(debug_rope)
        self._debug_rope_printed = False
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
        self.use_rope = pos_embed_fn in {"rope", "multimodal_rope", "mrope"}
        self.rope_hand_offset = rope_hand_offset
        self.pos_token_dim = pos_in_dim
        self.sensor_time_chunk_size = sensor_time_chunk_size
        self.sensor_num_chunks = sequence_length // sensor_time_chunk_size
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
        self.sensor_embed = _make_embed1d(
            in_chans, seq, sensor_time_chunk_size, D
        )
        self.angle_embed = _make_embed1d(pos_in_chans, seq, chunk, D)
        self.sensor_time_embed = nn.Parameter(
            torch.zeros(1, self.sensor_num_chunks, 1, D)
        )

        # ── Pre-fusion: last-frame queries attend to all sensor frames ─────────
        self.sensor_block = nn.ModuleList([
            CrossAttentionBlock(
                dim=D,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            for _ in range(self.pre_fusion_depth)
        ])
        if self.sensor_num_chunks > 1 and not self.sensor_block:
            raise ValueError(
                "pre_fusion_depth must be >= 1 when sensor tokens span "
                "multiple time chunks"
            )

        # ── Normalization buffers ──────────────────────────────────────────────
        if normalization is not None:
            m = torch.tensor(normalization.mean, dtype=torch.float32)
            s = torch.tensor(normalization.std,  dtype=torch.float32)
        else:
            m = torch.zeros(in_chans)
            s = torch.ones(in_chans)
        self.register_buffer("signal_mean", m)  # (in_chans,)
        self.register_buffer("signal_std",  s)  # (in_chans,)

        # Position stream. Identity unless the dataloader supplies stats, so a
        # config that does not ask for pos standardization behaves exactly as
        # before. HOI and BrainCo fingertip frames only line up once HOI is
        # converted to BrainCo's convention (pseudo_force_tactile.
        # align_xyz_to_brainco); this z-score then removes the residual
        # per-axis offset and scale between the two corpora.
        if pos_normalization is not None:
            pm = torch.tensor(pos_normalization.mean, dtype=torch.float32)
            ps = torch.tensor(pos_normalization.std,  dtype=torch.float32)
        else:
            pm = torch.zeros(pos_in_chans)
            ps = torch.ones(pos_in_chans)
        self.register_buffer("pos_mean", pm)  # (pos_in_chans,)
        self.register_buffer("pos_std",  ps)  # (pos_in_chans,)


        self.init_weights()
        nn.init.trunc_normal_(self.sensor_time_embed, std=0.02)
        if self.hand_embed is not None:
            nn.init.trunc_normal_(self.hand_embed, std=0.02)

        self.fine_tune_sensor = fine_tune_sensor
        if fine_tune_sensor:
            self._apply_fine_tune_sensor()

        num_chunks = seq // chunk
        log.info(
            f"AngleTransformer: T={seq}, pos_chunk={chunk}, num_chunks={num_chunks}, "
            f"sensor_chunk={self.sensor_time_chunk_size}, "
            f"sensor_num_chunks={self.sensor_num_chunks}, "
            f"N_contact={in_dim}, N_angles={pos_in_dim}, "
            f"N_angle_tokens={self.pos_token_dim}, "
            f"pre_fusion_depth={self.pre_fusion_depth}, embed_dim={D}, "
            f"pos_embed_fn={pos_embed_fn}, "
            f"rope_hand_offset={self.rope_hand_offset}, "
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

    def _finger_coords(self, n: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(n, device=device, dtype=torch.float32)
        hand = torch.div(idx, n // 2, rounding_mode="floor") * self.rope_hand_offset
        finger = idx.remainder(n // 2)
        joint = torch.full_like(finger, -1.0)
        return torch.stack([hand, finger, joint], dim=-1)

    def _skeleton_coords(self, n: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(n, device=device, dtype=torch.float32)
        per_hand = n // 2
        local = idx.remainder(per_hand)
        hand = torch.div(idx, per_hand, rounding_mode="floor") * self.rope_hand_offset
        finger = torch.full_like(local, -1.0)
        joint = torch.full_like(local, -1.0)

        is_finger_joint = local > 0
        local_finger = local[is_finger_joint] - 1
        finger[is_finger_joint] = torch.div(
            local_finger,
            SKELETON_JOINTS_PER_FINGER,
            rounding_mode="floor",
        )
        joint[is_finger_joint] = local_finger.remainder(SKELETON_JOINTS_PER_FINGER)
        return torch.stack([hand, finger, joint], dim=-1)

    def _sparse_tactile_coords(self, device: torch.device) -> torch.Tensor:
        """Return full-skeleton RoPE coordinates for the 10 sensed fingertips.

        BrainCo provides tactile measurements only at the fingertip links. When
        null-token expansion is disabled, retain those 10 tokens and select
        their coordinates from the same 42-joint skeleton used in pretraining.
        """
        full_coords = self._skeleton_coords(FULL_SKELETON_SIZE, device)
        fingertip_indices = torch.as_tensor(
            TACTILE_SENSOR_IDXS, dtype=torch.long, device=device
        )
        return full_coords.index_select(0, fingertip_indices)

    def _rope_positions(self, x: torch.Tensor, stream: str) -> torch.Tensor:
        b, t, n = x.shape[:3]
        if stream == "angle":
            expected = self.pos_in_dim
            if n != expected:
                raise ValueError(f"Angle RoPE expected {expected} tokens, got {n}")
            coords = self._finger_coords(n, x.device)
        elif not self.use_null_token and n == len(TACTILE_SENSOR_IDXS):
            coords = self._sparse_tactile_coords(x.device)
        elif n == FULL_SKELETON_SIZE:
            coords = self._skeleton_coords(n, x.device)
        else:
            # Fallback for unusual inputs: keep a single ordered finger axis.
            finger = torch.arange(n, device=x.device, dtype=torch.float32)
            coords = torch.stack([
                torch.zeros_like(finger),
                finger,
                torch.full_like(finger, -1.0),
            ], dim=-1)

        return coords.view(1, 1, n, len(ROPE_AXIS_NAMES)).expand(b, t, n, -1)

    def _apply_position_masks(self, pos_ids: torch.Tensor, masks) -> torch.Tensor:
        all_pos = []
        t = pos_ids.shape[1]
        for mask in masks:
            mask = mask.to(device=pos_ids.device, dtype=torch.long)
            if pos_ids.dim() == 4:
                a = pos_ids.shape[-1]
                mask_keep = einops.repeat(mask, "b n -> b t n a", t=t, a=a)
                gather_dim = -2
            else:
                mask_keep = einops.repeat(mask, "b n -> b t n", t=t)
                gather_dim = -1
            all_pos.append(torch.gather(pos_ids, dim=gather_dim, index=mask_keep))
        return torch.cat(all_pos, dim=0)

    def _flatten_rope_positions(self, pos_ids: Optional[torch.Tensor], skip_register: bool) -> Optional[torch.Tensor]:
        if pos_ids is None:
            return None
        if pos_ids.dim() == 4:
            pos_ids = einops.rearrange(pos_ids, "b t n a -> b (t n) a")
        else:
            pos_ids = einops.rearrange(pos_ids, "b t n -> b (t n)")
        if self.register_tokens is not None and not skip_register:
            reg_shape = (pos_ids.shape[0], self.num_register_tokens, *pos_ids.shape[2:])
            reg = pos_ids.new_full(reg_shape, -1)
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
        """Apply the configured signal transform, then embed.

        BrainCo adapters zero-fill invalid readings using their explicit
        pre-baseline validity mask. Negative values can therefore be legitimate
        baseline deltas and must not be treated as invalid here.

        Args:
            x : (B, T, N_joints, C)

        Returns:
            (B, num_chunks, N_joints, D)
        """
        B = x.shape[0]
        x = self.normalize(x)
        x = _apply_embed1d(x, self.sensor_embed, B)
        return x + self.sensor_time_embed

    def pre_pos_embed(self, pos: torch.Tensor) -> torch.Tensor:
        """Z-score the position stream, then embed.

        The buffers default to mean 0 / std 1, so this is a no-op unless the
        dataloader supplied per-axis statistics. RoPE is unaffected either way:
        `_rope_positions` builds its coordinates from finger indices, never from
        these values.

        Args:
            pos : (B, T, N_fingers, pos_in_chans)

        Returns:
            learned PE: (B, num_chunks, N_fingers, D)
            RoPE:       (B, num_chunks, N_fingers, D)
        """
        if self.use_rope and (pos.shape[-2] != self.pos_in_dim or pos.shape[-1] != self.pos_in_chans):
            raise ValueError(
                f"Expected angle input (B,T,{self.pos_in_dim},{self.pos_in_chans}), "
                f"got {tuple(pos.shape)}"
            )
        pos = (pos - self.pos_mean) / self.pos_std
        return _apply_embed1d(pos, self.angle_embed, pos.shape[0])

    def _maybe_print_rope_debug(
        self,
        raw_x_shape,
        raw_pos_shape,
        embedded_x: torch.Tensor,
        embedded_pos: torch.Tensor,
        contact_tokens: torch.Tensor,
        angle_tokens: torch.Tensor,
        sen_rope_positions: Optional[torch.Tensor],
        pos_rope_positions: Optional[torch.Tensor],
        fused_tokens: torch.Tensor,
    ) -> None:
        if not (self.debug_rope and self.use_rope) or self._debug_rope_printed:
            return

        self._debug_rope_printed = True

        def _preview(t: Optional[torch.Tensor], n: int = 8):
            if t is None:
                return None
            item = t[0, : min(n, t.shape[1])].detach().cpu()
            return item.tolist()

        print("[AngleTransformer RoPE debug]")
        print(f"  raw joint_contact shape      : {tuple(raw_x_shape)}")
        print(f"  raw finger_angles shape      : {tuple(raw_pos_shape)}")
        print(f"  embedded contact shape       : {tuple(embedded_x.shape)}")
        print(f"  embedded angle shape         : {tuple(embedded_pos.shape)}")
        print(f"  angle tokens                 : {self.pos_token_dim}")
        print(f"  angle joint channels         : {self.pos_in_chans}")
        print(f"  contact tokens after masking : {tuple(contact_tokens.shape)}")
        print(f"  angle tokens after masking   : {tuple(angle_tokens.shape)}")
        print(f"  contact rope coords preview  : {_preview(sen_rope_positions)}")
        print(f"  angle rope coords preview    : {_preview(pos_rope_positions)}")
        print(f"  fused token shape            : {tuple(fused_tokens.shape)}")

    # ── token preparation ─────────────────────────────────────────────────────

    def prepare_tokens_with_mask(
        self,
        x,
        masks,
        mask_type: Optional[Literal["block", "tubelet"]],
        masktoken_masks: Optional[List[torch.Tensor]],
        joint_embed: Optional[torch.Tensor] = None,
        skip_register: bool = False,
        rope_stream: str = "contact",
    ):
        """Flatten + optional learned PE/RoPE ids + masking + register token."""
        rope_positions = self._rope_positions(x, rope_stream) if self.use_rope else None

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

    def sensor_transform(self, x: torch.Tensor, _bias, rope_positions=None):
        """Last-frame queries cross-attend to all temporal sensor tokens."""
        if _bias is not None:
            raise NotImplementedError("Causal bias is not supported in sensor cross-attention")
        if x.shape[1] % self.sensor_num_chunks != 0:
            raise ValueError(
                f"Sensor token count {x.shape[1]} is not divisible by "
                f"sensor_num_chunks={self.sensor_num_chunks}"
            )
        tokens_per_frame = x.shape[1] // self.sensor_num_chunks
        q = x[:, -tokens_per_frame:]
        q_rope_positions = (
            None
            if rope_positions is None
            else rope_positions[:, -tokens_per_frame:]
        )
        for blk in self.sensor_block:
            q = blk(
                q,
                x,
                q_rope_positions=q_rope_positions,
                x_rope_positions=rope_positions,
            )
        return q, q_rope_positions

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
        if sen is None:
            # Joint-only: registers + pose tokens are the whole sequence.
            rope_positions = pos_rope_positions if self.use_rope else None
            if not self.use_rope:
                ang_pe = self._full_embed(self.angle_pos_embed)
                if pos_mask is not None:
                    np_ = pos_mask.shape[-1]
                    pos[:, 1:] = pos[:, 1:] + ang_pe[pos_mask.view(-1, np_)]
                else:
                    pos[:, 1:] = pos[:, 1:] + ang_pe

            fused = pos
            for blk in self.blocks:
                kwargs = {} if rope_positions is None else {"rope_positions": rope_positions}
                fused = blk(fused, None, **kwargs)
            x_norm = self.norm(fused)
            return x_norm, x_norm

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
        pos_masktoken_masks: Optional[List[torch.Tensor]] = None,
    ) -> dict:
        """Full forward pass.

        Args:
            x         : (B, T, N_joints=42,  C=1)  — joint contact
            pos       : (B, T, N_fingers=10, C=4)  — finger angles
            masks     : sensor stream masking
            mask_type : "block" or "tubelet"
            masktoken_masks: mask token positions
            pos_masks : angle stream masking (defaults to masks if None)
            pos_masktoken_masks: mask token positions for the angle/XYZ stream

        Returns:
            dict with keys:
              x_norm_regtokens   : (B_eff, num_register_tokens, D)
              x_norm_patchtokens : (B_eff, num_patches, D)
              x_prenorm          : (B_eff, num_patches, D)
              x_tokens           : (B_eff, reg+patches, D)
        """
        raw_x_shape = x.shape
        raw_pos_shape = pos.shape

        # --- null token expansion: (B,T,10,C) → (B,T,42,C) + mask -----------
        if self.use_null_token:
            x, null_mask = self.expand_to_skeleton(x)
            if masktoken_masks is None:
                masktoken_masks = null_mask
            else:
                masktoken_masks = masktoken_masks | null_mask

        # --- embed streams ---------------------------------------------------
        # Joint-only: the sensor stream is not embedded at all, so ``x`` never
        # becomes tokens. The masks that select sensor joints are meaningless
        # here and are ignored.
        if not self.pos_only:
            x = self.pre_sensor_embed(x)
        pos = self.pre_pos_embed(pos)
        embedded_x = x
        embedded_pos = pos

        # --- positional embeddings (pre-fusion stage) ------------------------
        sen_pe = None if self.use_rope else self._full_embed(self.contact_pos_embed)
        ang_pe = None if self.use_rope else self._full_embed(self.angle_pos_embed)

        _pos_masks = pos_masks if pos_masks is not None else masks

        if self.pos_only:
            bias, sen_rope_positions = None, None
        else:
            x, bias, sen_rope_positions = self.prepare_tokens_with_mask(
                x, masks, mask_type, masktoken_masks,
                joint_embed=sen_pe, skip_register=True, rope_stream="contact",
            )
        pos, _, pos_rope_positions = self.prepare_tokens_with_mask(
            pos, _pos_masks, mask_type, pos_masktoken_masks,
            joint_embed=ang_pe, skip_register=False, rope_stream="angle",
        )

        # --- pre-fusion sensor attention -------------------------------------
        if self.pos_only:
            sen = None
        else:
            sen, sen_rope_positions = self.sensor_transform(
                x, bias, rope_positions=sen_rope_positions
            )

        # --- fusion ----------------------------------------------------------
        x_prenorm, x_postnorm = self.transform_concat(
            sen, pos, masks, _pos_masks, bias,
            sen_rope_positions=sen_rope_positions,
            pos_rope_positions=pos_rope_positions,
        )

        self._maybe_print_rope_debug(
            raw_x_shape=raw_x_shape,
            raw_pos_shape=raw_pos_shape,
            embedded_x=embedded_x,
            embedded_pos=embedded_pos,
            contact_tokens=x,
            angle_tokens=pos,
            sen_rope_positions=sen_rope_positions,
            pos_rope_positions=pos_rope_positions,
            fused_tokens=x_postnorm,
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
        pos_mean: Optional[torch.Tensor] = None,
        pos_std: Optional[torch.Tensor] = None,
        **_kwargs,
    ):
        self.signal_mean = signal_mean
        self.signal_std  = signal_std
        # Left untouched when the caller has no position statistics, so the
        # buffers keep whatever the checkpoint or the identity default holds.
        if pos_mean is not None and pos_std is not None:
            self.pos_mean = pos_mean.to(self.pos_mean.device, self.pos_mean.dtype)
            self.pos_std  = pos_std.to(self.pos_std.device, self.pos_std.dtype)


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

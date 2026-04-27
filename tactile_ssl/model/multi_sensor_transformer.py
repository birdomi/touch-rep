"""MultiSensorTransformer: PatchEmbed1d 기반 멀티센서 dual-stream transformer.

SignalTransformer를 직접 상속해 처음부터 구현.

Architecture
------------
두 스트림(sensor, position)을 각각 PatchEmbed1d로 시간 압축 후:
  1. sensor_block  : 센서 타입별 독립 pre-fusion Self-Attention
  2. Concat fusion : cat([sen, pos]) → shared blocks → sensor 부분만 출력

Temporal compression (PatchEmbed1d)
------------------------------------
  sensor : (B, T, N, C)  → Conv1d(C→D, kernel=chunk) → (B, T//chunk, N, D)
  pos    : (B, T, N, 3)  → Conv1d(3→D, kernel=chunk) → (B, T//chunk, N, D)

  총 토큰 = num_register_tokens + num_chunks × N
  e.g. T=50, chunk=5, N=10 → 1 + 10×10 = 101 tokens

Sensor types
------------
  sensor_id = 0 : BrainCo  — in_chans = 4,  sequence_length = brainco_seq
  sensor_id = 1 : XELA     — in_chans = 4,  sequence_length = xela_seq
  두 타입 모두 N=10 (손가락/센서 그룹 수), in_chans=4 동일
  seq_len만 다를 수 있으며 PatchEmbed1d(Conv1d)는 임의 길이 처리 가능

Null handling
-------------
  BrainCo ch2(depth)는 거의 항상 -1. Conv1d 전에 zero-fill 후 normalization.
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
from .layers import init_weights_vit_timm

log = get_pylogger(__name__)

NUM_SENSORS = 1
SENSOR_ID_BRAINCO = 0
SENSOR_ID_XELA = 1

# Full skeleton layout (42 joints per SKELETON_JOINTS definition)
# Touch-link indices (0-indexed) where tactile sensors are physically placed
# Left hand:  thumb=4, index=8,  middle=12, ring=16, pinky=20
# Right hand: thumb=25,index=29, middle=33, ring=37, pinky=41
FULL_SKELETON_SIZE = 42
TACTILE_SENSOR_IDXS = [4, 8, 12, 16, 20, 25, 29, 33, 37, 41]


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_embed1d(in_chans: int, seq_len: int, chunk_size: int, embed_dim: int) -> PatchEmbed1d:
    """padding=0으로 출력 길이가 정확히 seq_len // chunk_size가 되도록."""
    return PatchEmbed1d(
        modal_chans=in_chans,
        modal_lens=seq_len,
        chunk_size=chunk_size,
        embed_dim=embed_dim,
        padding=0,
    )


def _apply_embed1d(x: torch.Tensor, embed: PatchEmbed1d, B: int, N: int) -> torch.Tensor:
    """각 센서(또는 손가락)를 독립적인 시계열로 취급해 PatchEmbed1d 적용.

    Args:
        x    : (B, T, N, C)
        embed: PatchEmbed1d  — Conv1d(C → D, kernel=chunk, stride=chunk)
        B, N : batch size, 센서 수

    Returns:
        (B, num_chunks, N, D)
    """
    x = einops.rearrange(x, "b t n c -> (b n) c t")   # 각 센서를 1-D 시계열로 분리
    x = embed(x)                                        # (B*N, D, num_chunks)
    x = einops.rearrange(x, "(b n) c t -> b t n c", b=B)
    return x


# ── main class ────────────────────────────────────────────────────────────────

class MultiSensorTransformer(SignalTransformer):
    """PatchEmbed1d 기반 멀티센서 dual-stream Transformer.

    SignalTransformer에서 제공하는 것:
      - self.blocks         : depth개의 shared transformer block (pos pre-fusion + fusion)
      - self.norm           : final LayerNorm
      - self.pos_embed      : learned (1, num_chunks*N, D) — 시퀀스 위치 인코딩
      - self.register_tokens, self.mask_token
      - prepare_tokens_with_mask()

    이 클래스가 추가하는 것:
      - self.patch_embed    : ModuleList[PatchEmbed1d × NUM_SENSORS]
      - self.position_embed : PatchEmbed1d for fingertip 6D pose
      - self.sensor_block   : ModuleList[pre-fusion blocks × NUM_SENSORS]
      - self.signal_mean/std: per-sensor normalization buffers

    Args:
        in_dim         : 센서 수 N (e.g. 10 — 손가락 수)
        in_chans       : 채널 수 (e.g. 4, BrainCo/XELA 공통)
        sequence_length: 입력 시계열 길이 T (기준값; Conv1d는 임의 길이 처리)
        time_chunk_size: Conv1d stride (T → T//chunk 토큰)
        embed_dim      : transformer hidden dim
        depth          : shared blocks 총 수
        num_heads      : attention head 수
        mlp_ratio      : FFN hidden dim 비율
        pre_fusion_depth: sensor_block depth. None이면 depth//4 사용
        normalization  : DictConfig(mean, std) — shape (NUM_SENSORS, in_chans)
    """

    def __init__(
        self,
        in_dim: int,
        in_chans: int,
        sequence_length: int,
        time_chunk_size: int,
        embed_dim: int,
        depth: int = 8,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        ffn_layer: str = "mlp",
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        pos_embed_fn: Literal["sinusoidal", "learned"] = "learned",
        init_values: Optional[float] = None,
        num_register_tokens: int = 1,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        with_masktoken: bool = False,
        use_null_token: bool = False,
        use_partial_tactile: bool = False,
        mask_finetune: bool = False,
        causal: bool = False,
        pre_fusion_depth: Optional[int] = None,
        normalization: Optional[DictConfig] = None,
        pos_normalization: Optional[DictConfig] = None,
    ):
        assert sequence_length % time_chunk_size == 0, (
            f"sequence_length({sequence_length}) must be divisible by time_chunk_size({time_chunk_size})"
        )
        self.use_partial_tactile = use_partial_tactile
        self.use_null_token = use_null_token
        self.mask_finetune = mask_finetune
        if self.use_null_token or self.mask_finetune:
            with_masktoken = True  # null token 또는 mask_finetune 사용 시 mask token도 필요
        # ── SignalTransformer 초기화 ──────────────────────────────────────
        # blocks, norm, pos_embed, register_tokens, mask_tok en 생성
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

        self.in_dim = in_dim
        self.in_chans = in_chans
        self.xela_in_chans = in_chans       # BrainCo와 동일하게 in_chans=4
        self.pre_fusion_depth = pre_fusion_depth if pre_fusion_depth is not None else depth // 4

        chunk = time_chunk_size
        seq   = sequence_length
        D     = embed_dim
        num_chunks = seq // chunk

        # ── Positional Embedding ───────────────────────────────────
        self.pos_embed = nn.Parameter(torch.zeros(2, 21, D)) # 21 joints embedding for each trasnformer (sensor + fuse)
        self.hand_embed = nn.Parameter(torch.zeros(2, 2, D)) # left or right hand embedding for each trasnformer (pose + fuse)


        # ── PatchEmbed1d: sensor stream ───────────────────────────────────
        # BrainCo: Conv1d(4 → D, kernel=chunk, stride=chunk)
        # XELA:    Conv1d(4 → D, kernel=chunk, stride=chunk)  — seq_len만 다름
        self.patch_embed = nn.ModuleList([
            _make_embed1d(self.in_chans,      seq, chunk, D),  # sensor_id = 0 (BrainCo)
        ])

        # ── PatchEmbed1d: position stream (fingertip 3D pose) ─────────────
        # Conv1d(3 → D, kernel=chunk, stride=chunk)
        self.position_embed = _make_embed1d(3, seq, chunk, D)

        # ── Wrist register token embedding: concat(left_9d, right_9d) → D ──
        # Input: (B, T, 2, 9) → reshape (B, T, 18) → Linear → mean over T → (B, 1, D)
        self.wrist_embed = nn.Linear(18, D)

        # ── sensor_block: per-sensor pre-fusion transformer blocks ─────────
        def _make_sensor_blocks() -> nn.ModuleList:
            return nn.ModuleList([
                Block(
                    attn_class=MemEffAttention,
                    dim=D,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop_path=0.0,           # sensor_block은 stochastic depth 없음
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    ffn_layer=Mlp,
                    init_values=init_values,
                )
                for _ in range(self.pre_fusion_depth)
            ])

        # sensor_block[0] = BrainCo 전용, sensor_block[1] = XELA 전용
        self.sensor_block = nn.ModuleList([
            _make_sensor_blocks() for _ in range(NUM_SENSORS)
        ])


        # ── Per-sensor normalization buffers: (NUM_SENSORS, in_chans) ─────
        if normalization is not None:
            m = torch.tensor(normalization.mean, dtype=torch.float32)
            s = torch.tensor(normalization.std,  dtype=torch.float32)
            if m.dim() == 1:  # 단일 센서 stats → 두 센서에 복제
                m = m.unsqueeze(0).expand(NUM_SENSORS, -1).clone()
                s = s.unsqueeze(0).expand(NUM_SENSORS, -1).clone()
        else:
            m = torch.zeros(in_chans)
            s = torch.ones(in_chans)
        self.register_buffer("signal_mean", m)   # (2, 4)
        self.register_buffer("signal_std",  s)   # (2, 4)

        # ── Per-channel pose normalization buffers: (3,) ──────────────────
        if pos_normalization is not None:
            pm = torch.tensor(pos_normalization.mean, dtype=torch.float32)
            ps = torch.tensor(pos_normalization.std,  dtype=torch.float32)
        else:
            pm = torch.zeros(3)
            ps = torch.ones(3)

        self.register_buffer("pos_mean", pm)   # (3,)
        self.register_buffer("pos_std",  ps)   # (3,)

        self.init_weights()
        # Other parameters (pos_embed, hand_embed)
        nn.init.trunc_normal_(self.hand_embed, std=0.02)
        

        log.info(
            f"MultiSensorTransformer: T={seq}, chunk={chunk}, num_chunks={num_chunks}, "
            f"N={in_dim}, pre_fusion_depth={self.pre_fusion_depth}, "
            f"total_tokens={1 + num_chunks * in_dim}, embed_dim={D}"
        )


    # ── normalization ─────────────────────────────────────────────────────────

    def normalize(
        self,
        x: torch.Tensor,
        sensor_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-sensor 정규화.

        Args:
            x:          (B, T, N, C_max)  — C_max = xela_in_chans (0-padded)
            sensor_ids: (B,) long,  None이면 전부 BrainCo로 처리

        Returns:
            정규화된 텐서, 같은 shape
        """
        # if sensor_ids is None:
        x = x.clone()
        m = self.signal_mean[SENSOR_ID_BRAINCO]   # (4,)
        s = self.signal_std[SENSOR_ID_BRAINCO]    # (4,)
        x[..., :self.in_chans] = (x[..., :self.in_chans] - m) / s
        return x

        # out = x.clone()
        # for sid in sensor_ids.unique().tolist():
        #     mask = sensor_ids == sid
        #     xi   = x[mask]
        #     m    = self.signal_mean[sid]   # (4,)
        #     s    = self.signal_std[sid]    # (4,)

        #     # BrainCo/XELA 모두 in_chans=4로 동일 처리
        #     xi = xi.clone()
        #     xi[..., :self.in_chans] = (xi[..., :self.in_chans] - m) / s
        #     out[mask] = xi

        return out

    # ── null token expansion ──────────────────────────────────────────────────

    def expand_to_skeleton(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand tactile-only input to full skeleton space and generate null mask.

        Tactile sensors exist only at TACTILE_SENSOR_IDXS (10 positions) out of
        FULL_SKELETON_SIZE (42) joints.  All other positions are zero-filled and
        marked in the returned mask so they can later be replaced with mask_token.

        Args:
            x: (B, T, 10, C)  — tactile sensor data for the 10 fingertips

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

        # True  → null position (no tactile sensor, will be replaced with mask_token)
        # False → valid tactile position
        null_mask = torch.ones(B, FULL_SKELETON_SIZE, dtype=torch.bool, device=x.device)
        null_mask[:, TACTILE_SENSOR_IDXS] = False
        masktoken_mask = null_mask.unsqueeze(0)  # (1, B, 42)

        return x_full, masktoken_mask

    # ── sensor embedding ──────────────────────────────────────────────────────

    def pre_sensor_embed(
        self,
        x: torch.Tensor,
        sensor_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Null-fill → per-sensor normalize → PatchEmbed1d.

        Args:
            x:          (B, T, N, C_max)
            sensor_ids: (B,) long,  None이면 BrainCo fallback, 0 for hand, 1 for brainco, 2 for xela

        Returns:
            (B, num_chunks, N, embed_dim)
        """
        B, T, N, _ = x.shape

        # --- BrainCo ch2(depth) -1(invalid) → zero-fill before Conv1d ------
        x = x.clone()
        x[..., :self.in_chans][x[..., :self.in_chans] < 0] = 0.0

        # --- per-sensor normalization ----------------------------------------
        x = self.normalize(x, sensor_ids)

        # --- PatchEmbed1d per sensor type ------------------------------------
        out = _apply_embed1d(x, self.patch_embed[0], B, N
        )
        # for sid in sensor_ids.unique().tolist():
        #     print(sid, )
        #     mask = sensor_ids == sid
        #     B_s  = int(mask.sum())
        #     # BrainCo/XELA 모두 in_chans=4로 동일 처리, sensor별 patch_embed 사용
        #     out[mask] = _apply_embed1d(
        #         x[mask][..., :self.in_chans],
        #         self.patch_embed[sid],
        #         B_s, N,
        #     )

        return out

    # ── position embedding ────────────────────────────────────────────────────

    def pre_pos_embed(self, pos: torch.Tensor) -> torch.Tensor:
        """Normalize + PatchEmbed1d for fingertip 3D pose.

        Args:
            pos: (B, T, N, 3) or (B, T, N, 6) — only first 3 channels (xyz) used

        Returns:
            (B, num_chunks, N, embed_dim)
        """
        B, _, N, _ = pos.shape
        pos_xyz = pos[..., :3].clone()
        pos_xyz = (pos_xyz - self.pos_mean) / self.pos_std
        # print(self.pos_mean, self.pos_std)

        return _apply_embed1d(pos_xyz, self.position_embed, B, N)

    # ── per-sensor pre-fusion blocks ──────────────────────────────────────────

    def sensor_transform(
        self,
        x: torch.Tensor,
        sensor_ids: Optional[torch.Tensor],
        bias,
    ) -> torch.Tensor:
        """센서 타입별 독립 pre-fusion self-attention.

        Args:
            x:          (B_eff, seq, D)  — prepare_tokens_with_mask 출력
            sensor_ids: (B_eff,) long,   None이면 BrainCo blocks 적용
            bias:       causal attention bias (causal=False이면 None)

        Returns:
            (B_eff, seq, D)
        """
        # if sensor_ids is None:

        sid = sensor_ids[0].item() if sensor_ids is not None else SENSOR_ID_BRAINCO
        # print(sensor_ids, sid)
        for blk in self.sensor_block[0]:
            x = blk(x, bias)
        return x

        out = torch.zeros_like(x)
        for sid in sensor_ids.unique().tolist():
            mask = sensor_ids == sid
            curr = x[mask]
            for blk in self.sensor_block[sid]:
                curr = blk(curr, bias)
            out[mask] = curr
        return out

    # ── concat fusion ─────────────────────────────────────────────────────────

    def transform_concat(
        self,
        sen: torch.Tensor,
        pos: torch.Tensor,
        sen_mask,
        pos_mask,
        bias
    ):
        """Concat → Fusion blocks → sensor portion.

        Layout:
          cat([sen, pos], dim=1)               (B_eff, 2*(reg+N), D)
          → blocks (fusion, no causal bias)
          → sensor portion [:reg+N]
          → norm

        Args:
            sen: (B_eff, reg+N, D)  — sensor_transform 출력
            pos: (B_eff, reg+N, D)  — prepare_tokens_with_mask 출력
            bias: attention bias

        Returns:
            (x_prenorm, x_postnorm) both (B_eff, reg+N, D)
        """
        # Step 1: sensor + pos 토큰을 sequence 방향으로 concat하기 전에 pos_embed + hand_embed로 위치 인코딩 보정
        pos_embed = self.pos_embed[1,].float()
        hand_embed = self.hand_embed[1,].float()
        pos_embed = torch.cat([pos_embed + hand_embed[0], pos_embed + hand_embed[1]], dim=0)  # (2, 42, D) → (4, 42, D) for sensor and fusion blocks

        if not sen_mask is None:
            _, b, n = sen_mask.shape
            sen_mask = sen_mask.view(-1, n)
            sen_pos_embed = pos_embed[sen_mask]  # (n, D)


            _, b, n = pos_mask.shape
            pos_mask = pos_mask.view(-1, n)
            pos_pos_embed = pos_embed[pos_mask]  # (n, D)

            sen = sen + sen_pos_embed
            pos[:, 1:, :] = pos[:, 1:, :] + pos_pos_embed
        else:
            if sen.shape[1] != pos_embed.shape[0]:
                sen = sen + pos_embed[TACTILE_SENSOR_IDXS]
            else:
                sen = sen + pos_embed

            if (pos.shape[1]-1) != pos_embed.shape[0]:
                 pos[:, 1:, :] = pos[:, 1:, :] + pos_embed[TACTILE_SENSOR_IDXS]
            else:
                pos[:, 1:, :] = pos[:, 1:, :] + pos_embed

        # Step 2: sensor + pos 토큰을 sequence 방향으로 concat
        fused = torch.cat([pos, sen], dim=1)   # (B_eff, 2*(N) + reg, D) 
        # fused = pos # debugging


        # Step 3: fusion blocks — sensor와 position이 cross-attend
        for blk in self.blocks:
            fused = blk(fused, None)

        # Step 4: LayerNorm
        x = fused
        x_norm = self.norm(fused)
        return x, x_norm

    def prepare_tokens_with_mask(
        self,
        x,
        masks,
        mask_type: Optional[Literal["block", "tubelet"]],
        masktoken_masks: Optional[List[torch.Tensor]],
        skip_register: bool = False,
        mask_token_append: bool = False,
        finetune_mask_ratio: Optional[float] = None,
    ):
        t, n = x.shape[-3], x.shape[-2]
        # print('##', x.shape, self.pos_embed.shape, self.pos_embed_fn)

        assert t <= self.sequence_length, (
            f"Input sequence length {t} is greater than model sequence length {self.sequence_length}"
        )

        pos_embed = self.pos_embed[0,].float()
        hand_embed = self.hand_embed[0,].float()
        pos_embed = torch.cat([pos_embed + hand_embed[0], pos_embed + hand_embed[1]], dim=0)  # (2, 42, D) → (4, 42, D) for sensor and fusion blocks

        # print(pos_embed.shape, hand_embed.shape)
        if n == 42:        
            pos_embed = pos_embed.view(1, 1, 42, -1)
            x = x + pos_embed
        else:            
            pos_embed = pos_embed[TACTILE_SENSOR_IDXS]
            pos_embed = pos_embed.view(1, 1, n, -1)
            x = x + pos_embed

        if masks is not None:
            if mask_type == "tubelet":
                x = self.apply_tubelet_masks(x, masks)
            elif mask_type == "block":
                x = apply_masks(x, masks)
            else:
                raise NotImplementedError(f"Unknown mask type {mask_type}")
        if self.causal:
            attn_bias = self.create_causal_mask(x)
        else:
            attn_bias = None


        if finetune_mask_ratio is not None:
            B_fm, _, n_fm, _ = x.shape
            rand = torch.rand(B_fm, n_fm, device=x.device)
            random_mask = (rand < finetune_mask_ratio).unsqueeze(0)  # (1, B, N)
            masktoken_masks = random_mask if masktoken_masks is None else (masktoken_masks | random_mask)

        if masktoken_masks is not None:
            x = self.apply_masktokens(x, masktoken_masks)

        if mask_token_append and self.use_null_token:  # mask_finetune만 true일 때는 skip
            B, T, _, C = x.shape
            full = self.mask_token.expand(B, T, FULL_SKELETON_SIZE, C).clone()
            full[:, :, TACTILE_SENSOR_IDXS, :] = x
            x = full

    
        x = einops.rearrange(x, "b t n c -> b (t n) c")
        if self.register_tokens is not None and not skip_register:
            x = torch.cat([self.register_tokens.expand(x.shape[0], -1, -1), x], dim=1)

        return x, attn_bias

    # ── forward ───────────────────────────────────────────────────────────────

    def forward_features(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        wrist_poses: Optional[torch.Tensor] = None,
        sensor_ids: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        mask_type: Optional[Literal["block", "tubelet"]] = None,
        masktoken_masks: Optional[List[torch.Tensor]] = None,
        pos_masks: Optional[torch.Tensor] = None,
        finetune_mask_ratio: Optional[float] = None,
    ) -> dict:
        """전체 forward pass.

        Args:
            x:           (B, T, N, C_max)  sensor data
            pos:         (B, T, N, 3)      fingertip 3D pose (wrist-local)
            wrist_poses: (B, T, 2, 9)      wrist poses [translation(3)+rot6d(6)].
                         When provided, the two wrists are concatenated (18D),
                         embedded via wrist_embed, and used as the register token
                         instead of the learned register_tokens parameter.
            sensor_ids:  (B,) long         sensor type per sample
            masks:       sensor stream masking (x 전용)
            mask_type:   "block" or "tubelet"
            masktoken_masks: mask token 위치
            pos_masks:   position stream masking (pos 전용, None이면 masks와 동일)

        Returns:
            dict with keys:
              x_norm_regtokens   : (B_eff, num_register_tokens, D)
              x_norm_patchtokens : (B_eff, num_patches, D)
              x_prenorm          : (B_eff, num_patches, D)
              x_tokens           : (B_eff, reg+patches, D)
        """
        # --- Null token expansion: (B,T,10,C) → (B,T,42,C) + mask ----------
        # print(wrist_poses, self.pos_mean)
        # if self.use_null_token:
        #     x, null_masktoken_mask = self.expand_to_skeleton(x)
        #     # Combine with any external masktoken_masks (SSL iBOT masks)
        #     if masktoken_masks is None:
        #         masktoken_masks = null_masktoken_mask
        #     else:
        #         # OR: positions that are either null OR iBOT-masked get mask_token
        #         masktoken_masks = masktoken_masks | null_masktoken_mask
        # print('###', x.shape, pos.shape)

        # --- Embedding: (B, T, N, C) → (B, num_chunks, N, D) ---------------
        x   = self.pre_sensor_embed(x,   sensor_ids=sensor_ids)
        pos = self.pre_pos_embed(pos)

        # --- Wrist register token: concat(left_9d, right_9d) → embed → (B, 1, D) ---
        if wrist_poses is not None:
            # print(wrist_poses)
            B_orig = x.shape[0]
            # (B, T, 2, 9) → (B, T, 18) → Linear → mean over T → (B, 1, D)
            wrist_token = self.wrist_embed(
                wrist_poses.reshape(B_orig, wrist_poses.shape[1], 18)
            ).mean(dim=1, keepdim=True)   # (B, 1, D)
        else:
            wrist_token = None

        # --- prepare_tokens_with_mask: flatten + reg token + pos_embed ------
        #   x:   (B_eff, n_keep_x, D)   — no register token (skip_register=True)
        #   pos: (B_eff, reg+n_keep_p, D) — register token prepended (or wrist token)
        _pos_masks = pos_masks if pos_masks is not None else masks
        x,   bias     = self.prepare_tokens_with_mask(x,   masks,     mask_type, masktoken_masks, skip_register=True, mask_token_append=self.use_null_token, finetune_mask_ratio=finetune_mask_ratio)
        if wrist_token is not None:
            pos, pos_bias = self.prepare_tokens_with_mask(pos, _pos_masks, mask_type, None, skip_register=True)
            # print(_pos_masks)
            # Expand wrist_token to match batch expansion from masks
            B_eff = pos.shape[0]
            if B_eff != wrist_token.shape[0]:
                repeat = B_eff // wrist_token.shape[0]
                wrist_token = wrist_token.repeat_interleave(repeat, dim=0)
            pos = torch.cat([wrist_token, pos], dim=1)
        else:
            pos, pos_bias = self.prepare_tokens_with_mask(pos, _pos_masks, mask_type, None, skip_register=False)

        # --- sensor-specific pre-fusion blocks ------------------------------
        sen = self.sensor_transform(x, sensor_ids, bias)

        # --- concat fusion: fused (B_eff, reg+n_keep_x+n_keep_p, D) --------
        x_prenorm, x_postnorm = self.transform_concat(sen, pos, masks, _pos_masks, bias)

        r = self.num_register_tokens
        reg_tokens           = x_postnorm[:, :r]
        patch_tokens         = x_postnorm[:, r:]
        patch_tokens_prenorm = x_prenorm[:, r:]

        return {
            "x_norm_regtokens":   reg_tokens,
            "x_norm_patchtokens": patch_tokens,
            "x_prenorm":          patch_tokens_prenorm,
            "x_tokens":           x_postnorm,
        }

    def forward(self, x, pos, sensor_ids=None, **kwargs):
        out = self.forward_features(x, pos, sensor_ids=sensor_ids, **kwargs)
        return out["x_norm_patchtokens"]

    # ── stats update ─────────────────────────────────────────────────────────

    def update_stats(
        self,
        signal_mean: torch.Tensor,
        signal_std: torch.Tensor,
        pos_mean: Optional[torch.Tensor] = None,
        pos_std: Optional[torch.Tensor] = None,
    ):
        """Update normalization buffers.

        Args:
            signal_mean/std: (NUM_SENSORS, in_chans)
            pos_mean/std:    (3,) — optional pose normalization stats
        """
        # print(signal_mean.shape, signal_mean)
        # assert signal_mean.shape == signal_std.shape == (NUM_SENSORS, self.in_chans)
        self.signal_mean = signal_mean
        self.signal_std  = signal_std
        if pos_mean is not None and pos_std is not None:
            assert pos_mean.shape == pos_std.shape == (3,)
            self.pos_mean = pos_mean
            self.pos_std  = pos_std


# ── factory functions ─────────────────────────────────────────────────────────

def multi_sensor_tiny(
    in_dim: int,
    in_chans: int,
    sequence_length: int,
    time_chunk_size: int = 5,
    depth: int = 8,
    num_register_tokens: int = 1,
    **kwargs,
) -> MultiSensorTransformer:
    """MultiSensorTransformer tiny (embed_dim=192, depth=8).

    추천 설정:
      sequence_length=50,  time_chunk_size=5  → 10 chunks/sensor  (0.5s @100Hz)
      sequence_length=100, time_chunk_size=10 → 10 chunks/sensor  (1.0s @100Hz)

    token 수 = 1 + 10 × 10 = 101
    """
    return MultiSensorTransformer(
        in_dim=in_dim,
        in_chans=in_chans,
        sequence_length=sequence_length,
        time_chunk_size=time_chunk_size,
        embed_dim=192,
        depth=depth,
        num_heads=3,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def multi_sensor_small(
    in_dim: int,
    in_chans: int,
    sequence_length: int,
    time_chunk_size: int = 5,
    depth: int = 12,
    num_register_tokens: int = 1,
    **kwargs,
) -> MultiSensorTransformer:
    """MultiSensorTransformer small (embed_dim=384, depth=12)."""
    return MultiSensorTransformer(
        in_dim=in_dim,
        in_chans=in_chans,
        sequence_length=sequence_length,
        time_chunk_size=time_chunk_size,
        embed_dim=384,
        depth=depth,
        num_heads=6,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )

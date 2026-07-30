"""ACT — Action Chunking with Transformers.

Reference:
    Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware" (RSS 2023) — https://github.com/tonyzhaozh/act

A CVAE whose decoder is a DETR-style transformer:

    observation tokens = [latent, qpos] (+ tactile) (+ per-camera ResNet features)
    decoder queries    = ``chunk_size`` learned queries → one action each

The CVAE encoder runs only during training: it compresses the ground-truth
action chunk into a style latent ``z`` so the decoder can be trained on
multi-modal demonstrations without averaging them away. At inference ``z`` is
set to the prior mean (zeros).

This implementation differs from the reference in one detail: positional
embeddings are added once to the encoder input instead of being re-added at
every attention layer, which lets it use ``nn.TransformerEncoder`` directly.
"""

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
from torchvision import models

from tactile_ssl.utils.logging import get_pylogger

log = get_pylogger(__name__)


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with frozen affine + running statistics (DETR-style).

    ACT fine-tunes ImageNet backbones with small batches, where updating the
    running statistics destabilizes training.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        scale = weight * (running_var + self.eps).rsqrt()
        return x * scale + (bias - running_mean * scale)


def _freeze_batchnorm(module: nn.Module) -> nn.Module:
    """Recursively replace every BatchNorm2d with a FrozenBatchNorm2d."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            frozen.weight.data.copy_(child.weight.data)
            frozen.bias.data.copy_(child.bias.data)
            frozen.running_mean.data.copy_(child.running_mean.data)
            frozen.running_var.data.copy_(child.running_var.data)
            setattr(module, name, frozen)
        else:
            _freeze_batchnorm(child)
    return module


class PositionEmbeddingSine2d(nn.Module):
    """DETR's 2D sine/cosine position embedding over a feature map grid."""

    def __init__(self, embed_dim: int, temperature: float = 10000.0, scale: float = 2 * math.pi):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim must be even, got {embed_dim}")
        self.num_pos_feats = embed_dim // 2
        self.temperature = temperature
        self.scale = scale

    def forward(self, height: int, width: int, device, dtype) -> torch.Tensor:
        """Returns ``(1, height * width, embed_dim)``."""
        y_embed = torch.arange(1, height + 1, device=device, dtype=torch.float32)
        x_embed = torch.arange(1, width + 1, device=device, dtype=torch.float32)
        y_embed = (y_embed / (height + 1e-6) * self.scale).view(-1, 1).expand(height, width)
        x_embed = (x_embed / (width + 1e-6) * self.scale).view(1, -1).expand(height, width)

        dim_t = torch.arange(self.num_pos_feats, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos = torch.cat((pos_y, pos_x), dim=-1)  # (H, W, embed_dim)
        return pos.flatten(0, 1).unsqueeze(0).to(dtype)


def build_vision_backbone(name: str, pretrained: bool, frozen_bn: bool):
    """ResNet trunk up to layer4, returning ``(module, num_channels)``."""
    if not hasattr(models, name):
        raise ValueError(f"Unknown torchvision backbone: {name}")
    try:
        weights = "DEFAULT" if pretrained else None
        resnet = getattr(models, name)(weights=weights)
    except TypeError:  # torchvision < 0.13
        resnet = getattr(models, name)(pretrained=pretrained)

    num_channels = resnet.fc.in_features
    trunk = nn.Sequential(*list(resnet.children())[:-2])  # drop avgpool + fc
    if frozen_bn:
        trunk = _freeze_batchnorm(trunk)
    return trunk, num_channels


def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|a, qpos) || N(0, I)), summed over latent dims, averaged over batch."""
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=-1).mean()


class ACTPolicy(nn.Module):
    """Action Chunking Transformer policy.

    Args:
        state_dim:   proprioceptive input width (26 for BrainCo arms + hands).
        action_dim:  per-timestep action width (26 by default).
        chunk_size:  number of future actions predicted in one forward pass.
        num_cameras: RGB streams; each gets its own ResNet backbone.
        hidden_dim:  transformer width.
        latent_dim:  CVAE style-variable width.
        use_tactile: append one token per tactile sensor to the observation.
    """

    def __init__(
        self,
        state_dim: int = 26,
        action_dim: int = 26,
        chunk_size: int = 60,
        num_cameras: int = 1,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 3200,
        enc_layers: int = 4,
        dec_layers: int = 7,
        cvae_enc_layers: int = 4,
        dropout: float = 0.1,
        latent_dim: int = 32,
        backbone: str = "resnet18",
        pretrained_backbone: bool = True,
        frozen_backbone_bn: bool = True,
        use_tactile: bool = True,
        num_tactile_sensors: int = 10,
        tactile_channels: int = 4,
        camera_names: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_tactile = bool(use_tactile)
        self.num_tactile_sensors = int(num_tactile_sensors)
        self.camera_names = list(camera_names) if camera_names is not None else None
        self.num_cameras = (
            len(self.camera_names) if self.camera_names is not None else int(num_cameras)
        )

        # ── vision backbones (one per camera, as in the reference) ──────────
        backbones, projections = [], []
        for _ in range(self.num_cameras):
            trunk, num_channels = build_vision_backbone(
                backbone, pretrained_backbone, frozen_backbone_bn
            )
            backbones.append(trunk)
            projections.append(nn.Conv2d(num_channels, hidden_dim, kernel_size=1))
        self.backbones = nn.ModuleList(backbones)
        self.input_proj_images = nn.ModuleList(projections)
        self.image_pos_embed = PositionEmbeddingSine2d(hidden_dim)
        self.camera_embed = nn.Embedding(self.num_cameras, hidden_dim)

        # ── observation tokens ──────────────────────────────────────────────
        self.input_proj_qpos = nn.Linear(self.state_dim, hidden_dim)
        self.input_proj_latent = nn.Linear(self.latent_dim, hidden_dim)
        # [latent, qpos] get learned positional embeddings.
        self.proprio_pos_embed = nn.Embedding(2, hidden_dim)

        if self.use_tactile:
            self.input_proj_tactile = nn.Linear(int(tactile_channels), hidden_dim)
            self.tactile_pos_embed = nn.Embedding(self.num_tactile_sensors, hidden_dim)

        # ── CVAE encoder (training only) ────────────────────────────────────
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.cvae_proj_qpos = nn.Linear(self.state_dim, hidden_dim)
        self.cvae_proj_action = nn.Linear(self.action_dim, hidden_dim)
        self.cvae_pos_embed = nn.Embedding(self.chunk_size + 2, hidden_dim)
        self.cvae_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="relu",
                batch_first=True,
            ),
            num_layers=int(cvae_enc_layers),
            enable_nested_tensor=False,
        )
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)

        # ── main transformer ────────────────────────────────────────────────
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="relu",
                batch_first=True,
            ),
            num_layers=int(enc_layers),
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="relu",
                batch_first=True,
            ),
            num_layers=int(dec_layers),
            norm=nn.LayerNorm(hidden_dim),
        )
        self.query_embed = nn.Embedding(self.chunk_size, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, self.action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)

        # ── normalization stats (filled in from the training split) ─────────
        self.register_buffer("qpos_mean", torch.zeros(self.state_dim))
        self.register_buffer("qpos_std", torch.ones(self.state_dim))
        self.register_buffer("action_mean", torch.zeros(self.action_dim))
        self.register_buffer("action_std", torch.ones(self.action_dim))
        self.register_buffer("tactile_mean", torch.zeros(int(tactile_channels)))
        self.register_buffer("tactile_std", torch.ones(int(tactile_channels)))

        num_params = sum(p.numel() for p in self.parameters())
        log.info(
            f"ACTPolicy: {num_params/1e6:.1f}M params, state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, chunk_size={self.chunk_size}, "
            f"cameras={self.num_cameras}, use_tactile={self.use_tactile}"
        )

    # ── normalization ───────────────────────────────────────────────────────

    @torch.no_grad()
    def set_norm_stats(self, stats: Dict[str, torch.Tensor]):
        """Store the dataset's normalization stats so checkpoints carry them."""
        for key, value in stats.items():
            if hasattr(self, key):
                buffer = getattr(self, key)
                buffer.copy_(torch.as_tensor(value, dtype=buffer.dtype, device=buffer.device))

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions * self.action_std + self.action_mean

    # ── forward ─────────────────────────────────────────────────────────────

    def encode_latent(self, qpos, actions, is_pad):
        """CVAE encoder: ground-truth chunk → style latent posterior."""
        batch_size = qpos.shape[0]
        tokens = torch.cat([
            self.cls_embed.weight.unsqueeze(0).expand(batch_size, -1, -1),
            self.cvae_proj_qpos(qpos).unsqueeze(1),
            self.cvae_proj_action(actions),
        ], dim=1)  # (B, 2 + chunk, hidden)
        tokens = tokens + self.cvae_pos_embed.weight[: tokens.shape[1]].unsqueeze(0)

        padding_mask = None
        if is_pad is not None:
            cls_qpos_mask = torch.zeros(batch_size, 2, dtype=torch.bool, device=qpos.device)
            padding_mask = torch.cat([cls_qpos_mask, is_pad], dim=1)

        encoded = self.cvae_encoder(tokens, src_key_padding_mask=padding_mask)
        latent_params = self.latent_proj(encoded[:, 0])
        return latent_params[:, : self.latent_dim], latent_params[:, self.latent_dim :]

    def _observation_tokens(self, qpos, images, tactile, latent):
        batch_size = qpos.shape[0]
        tokens = torch.stack([self.input_proj_latent(latent), self.input_proj_qpos(qpos)], dim=1)
        tokens = tokens + self.proprio_pos_embed.weight.unsqueeze(0)
        parts = [tokens]

        if self.use_tactile:
            if tactile is None:
                raise ValueError("use_tactile=True but no tactile tensor was provided")
            tactile_tokens = self.input_proj_tactile(tactile)  # (B, num_sensors, hidden)
            parts.append(tactile_tokens + self.tactile_pos_embed.weight.unsqueeze(0))

        for camera_index in range(self.num_cameras):
            features = self.backbones[camera_index](images[:, camera_index])
            features = self.input_proj_images[camera_index](features)  # (B, hidden, h, w)
            _, _, height, width = features.shape
            image_tokens = features.flatten(2).transpose(1, 2)  # (B, h*w, hidden)
            pos = self.image_pos_embed(height, width, features.device, features.dtype)
            image_tokens = image_tokens + pos + self.camera_embed.weight[camera_index].view(1, 1, -1)
            parts.append(image_tokens)

        return torch.cat(parts, dim=1).reshape(batch_size, -1, self.hidden_dim)

    def forward(
        self,
        qpos: torch.Tensor,
        images: torch.Tensor,
        tactile: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Predict an action chunk.

        Args:
            qpos:    ``(B, state_dim)`` normalized robot state.
            images:  ``(B, num_cameras, 3, H, W)`` normalized RGB.
            tactile: ``(B, num_sensors, channels)`` normalized tactile reading.
            actions: ``(B, chunk_size, action_dim)`` ground truth — training only.
            is_pad:  ``(B, chunk_size)`` True where ``actions`` is padding.

        Returns:
            ``a_hat`` ``(B, chunk_size, action_dim)``, plus ``mu``/``logvar``
            (None at inference) and the auxiliary ``is_pad_hat`` logits.
        """
        if actions is not None:
            mu, logvar = self.encode_latent(qpos, actions, is_pad)
            latent = reparametrize(mu, logvar)
        else:
            mu = logvar = None
            latent = torch.zeros(
                qpos.shape[0], self.latent_dim, dtype=qpos.dtype, device=qpos.device
            )

        memory = self.encoder(self._observation_tokens(qpos, images, tactile, latent))
        queries = self.query_embed.weight.unsqueeze(0).expand(qpos.shape[0], -1, -1)
        hidden = self.decoder(queries, memory)

        return {
            "a_hat": self.action_head(hidden),
            "is_pad_hat": self.is_pad_head(hidden).squeeze(-1),
            "mu": mu,
            "logvar": logvar,
        }

    @torch.no_grad()
    def predict(
        self,
        qpos: torch.Tensor,
        images: torch.Tensor,
        tactile: Optional[torch.Tensor] = None,
        normalized_inputs: bool = False,
    ) -> torch.Tensor:
        """Inference helper: raw observations in, raw action chunk out."""
        if not normalized_inputs:
            qpos = (qpos - self.qpos_mean) / self.qpos_std
            if tactile is not None:
                tactile = (tactile - self.tactile_mean) / self.tactile_std
        out = self.forward(qpos, images, tactile)
        return self.denormalize_actions(out["a_hat"])


def act_tiny(**kwargs) -> ACTPolicy:
    defaults = dict(hidden_dim=256, num_heads=8, dim_feedforward=1024, enc_layers=2, dec_layers=4)
    defaults.update(kwargs)
    return ACTPolicy(**defaults)


def act_base(**kwargs) -> ACTPolicy:
    defaults = dict(hidden_dim=512, num_heads=8, dim_feedforward=3200, enc_layers=4, dec_layers=7)
    defaults.update(kwargs)
    return ACTPolicy(**defaults)

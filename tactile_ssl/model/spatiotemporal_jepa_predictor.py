"""Token predictors used by spatial and temporal tactile JEPA objectives."""

from __future__ import annotations

import torch
import torch.nn as nn


class JEPATokenPredictor(nn.Module):
    """Predict latent tokens at requested spatial positions.

    Context tokens may be an arbitrary visible subset. ``context_indices`` and
    ``target_indices`` use the shared multimodal layout

        [10 XYZ tokens][42 tactile tokens].

    Learned position embeddings make the subset ordering explicit to the
    predictor. Spatial and temporal objectives instantiate separate predictors,
    so their query tokens and transformer weights are not shared.
    """

    def __init__(
        self,
        encoder_dim: int,
        num_token_positions: int = 52,
        predictor_dim: int = 192,
        depth: int = 2,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if predictor_dim % num_heads != 0:
            raise ValueError(
                f"predictor_dim ({predictor_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.num_token_positions = int(num_token_positions)
        self.context_projection = nn.Linear(encoder_dim, predictor_dim)
        self.position_embedding = nn.Embedding(self.num_token_positions, predictor_dim)
        self.context_type_embedding = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        self.query_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            dim_feedforward=int(predictor_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(predictor_dim, eps=1e-6)
        self.output_projection = nn.Linear(predictor_dim, encoder_dim)

        nn.init.trunc_normal_(self.position_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.context_type_embedding, std=0.02)
        nn.init.trunc_normal_(self.query_token, std=0.02)

    def _validate_indices(self, indices: torch.Tensor, name: str) -> None:
        if indices.ndim != 2:
            raise ValueError(f"{name} must have shape (B, N), got {tuple(indices.shape)}")
        if indices.numel() == 0:
            raise ValueError(f"{name} must contain at least one token index")
        if indices.min() < 0 or indices.max() >= self.num_token_positions:
            raise ValueError(
                f"{name} values must be in [0, {self.num_token_positions}), "
                f"got min={indices.min().item()} max={indices.max().item()}"
            )

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted target latents with shape ``(B, N_target, D)``."""
        if context_tokens.ndim != 3:
            raise ValueError(
                "context_tokens must have shape (B, N_visible, D), "
                f"got {tuple(context_tokens.shape)}"
            )
        self._validate_indices(context_indices, "context_indices")
        self._validate_indices(target_indices, "target_indices")
        if context_tokens.shape[:2] != context_indices.shape:
            raise ValueError(
                "context token/index shape mismatch: "
                f"{tuple(context_tokens.shape[:2])} vs {tuple(context_indices.shape)}"
            )
        if target_indices.shape[0] != context_tokens.shape[0]:
            raise ValueError("target_indices batch size must match context_tokens")

        context = self.context_projection(context_tokens)
        context = (
            context
            + self.position_embedding(context_indices)
            + self.context_type_embedding
        )

        queries = self.query_token.expand(
            target_indices.shape[0], target_indices.shape[1], -1
        )
        queries = queries + self.position_embedding(target_indices)

        num_targets = target_indices.shape[1]
        tokens = torch.cat((context, queries), dim=1)
        tokens = self.blocks(tokens)
        predictions = self.norm(tokens[:, -num_targets:])
        return self.output_projection(predictions)


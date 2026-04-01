"""Minimal LoRA (Low-Rank Adaptation) for nn.Linear layers.

Usage:
    from tactile_ssl.model.lora import apply_lora

    n = apply_lora(model, rank=8, alpha=16.0, target_modules=("qkv", "proj"))
    # → replaces matching Linear layers in-place, returns count

    # Freeze base weights, unfreeze only LoRA params:
    model.requires_grad_(False)
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad_(True)
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear that adds a low-rank adapter.

    output = W(x) + (lora_B @ lora_A)(x) * scale
           = base_linear(x) + lora_B(lora_A(dropout(x))) * scale

    Base weights (W) are kept frozen; only lora_A and lora_B are trained.

    Args:
        linear   : original nn.Linear to wrap (weights are NOT copied — same tensor)
        rank     : LoRA rank r
        alpha    : LoRA scaling factor  (scale = alpha / rank)
        dropout  : dropout applied to x before lora_A (default 0)
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_linear = linear           # original frozen weights
        in_features  = linear.in_features
        out_features = linear.out_features
        self.scale = alpha / rank

        self.lora_A = nn.Linear(in_features,  rank,         bias=False)
        self.lora_B = nn.Linear(rank,          out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Init: A ~ kaiming_uniform, B = 0  (so adapter starts at zero)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_linear(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scale

    def extra_repr(self) -> str:
        return (
            f"in={self.base_linear.in_features}, "
            f"out={self.base_linear.out_features}, "
            f"rank={self.lora_A.out_features}, "
            f"scale={self.scale:.4f}"
        )


def apply_lora(
    model: nn.Module,
    rank: int,
    alpha: float = 1.0,
    dropout: float = 0.0,
    target_modules: tuple = ("qkv", "proj"),
) -> int:
    """Replace nn.Linear children whose attribute name is in target_modules with LoRALinear.

    Traverses the full module tree and replaces matching layers in-place.

    Args:
        model          : model to modify in-place
        rank           : LoRA rank r
        alpha          : LoRA scaling (scale = alpha / rank)
        dropout        : input dropout before LoRA adapter
        target_modules : attribute names to target (default: attention qkv + proj)

    Returns:
        Number of layers replaced.
    """
    replaced = 0
    for module in model.modules():
        for attr_name in list(vars(module).keys()):
            child = getattr(module, attr_name, None)
            if attr_name in target_modules and isinstance(child, nn.Linear):
                setattr(module, attr_name, LoRALinear(child, rank, alpha, dropout))
                replaced += 1
    return replaced


def lora_trainable_params(model: nn.Module) -> int:
    """Count trainable LoRA parameters in the model."""
    return sum(p.numel() for name, p in model.named_parameters()
               if p.requires_grad and "lora_" in name)

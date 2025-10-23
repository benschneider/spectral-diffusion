from typing import Any, Dict

import torch
from torch import nn


class DiffusionLoss(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config or {}
        self.reduction = self.config.get("reduction", "mean")
        if self.reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")

    def forward(self, residual: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        loss = residual**2
        if weight is not None:
            loss = loss * weight
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


def get_loss_fn(config: Dict[str, Any]) -> DiffusionLoss:
    return DiffusionLoss(config=config)

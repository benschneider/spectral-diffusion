from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn


class DiffusionLoss(nn.Module):
    """Minimal diffusion loss supporting MSE/MAE with optional log-SNR weighting."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        cfg = config or {}
        mode = cfg.get("mode", cfg.get("type", "mse")).lower()
        if mode not in {"mse", "mae"}:
            raise ValueError("DiffusionLoss.mode must be 'mse' or 'mae'")
        reduction = str(cfg.get("reduction", "mean")).lower()
        if reduction not in {"mean", "sum"}:
            raise ValueError("DiffusionLoss.reduction must be 'mean' or 'sum'")
        self.mode = mode
        self.reduction = reduction
        self.log_snr_weighting = bool(cfg.get("log_snr_weighting", False))

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        *_: torch.Tensor,
        snr_rel: Optional[torch.Tensor] = None,
        **__: Any,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        residual = prediction - target
        if self.mode == "mae":
            base = residual.abs()
        else:
            base = residual.pow(2)

        weight_diag: Dict[str, float] = {}
        if self.log_snr_weighting and snr_rel is not None:
            weight = torch.log(snr_rel.clamp(min=1e-6, max=1e6)).abs() + 1.0
            base = base * weight
            weight_diag = {
                "snr_weight_min": float(weight.min().item()),
                "snr_weight_max": float(weight.max().item()),
                "snr_weight_mean": float(weight.mean().item()),
            }

        if self.reduction == "sum":
            loss = base.sum()
        else:
            loss = base.mean()

        per_sample = base.view(base.shape[0], -1).mean(dim=1)
        diag: Dict[str, Any] = {
            "mae": float(residual.abs().mean().detach().item()),
            "per_sample_loss": per_sample.detach(),
        }
        diag.update(weight_diag)
        return loss, diag


def get_loss_fn(config: Optional[Dict[str, Any]] = None) -> DiffusionLoss:
    return DiffusionLoss(config=config)


__all__ = ["DiffusionLoss", "get_loss_fn"]

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from src.core.functional.diffusion import compute_snr_weight
from src.spectral.adapter import SpectralAdapter


class DiffusionLoss(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config or {}
        self.reduction = self.config.get("reduction", "mean")
        if self.reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")
        spectral_cfg = self.config.get("spectral_weighting", None)
        if spectral_cfg and spectral_cfg != "none":
            inner = self.config.get("bandpass_inner", 0.1)
            outer = self.config.get("bandpass_outer", 0.6)
            self.spectral_adapter = SpectralAdapter(
                enabled=True,
                weighting=spectral_cfg,
                normalize=True,
                bandpass_inner=inner,
                bandpass_outer=outer,
            )
        else:
            self.spectral_adapter = None

    def forward(
        self,
        residual: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.spectral_adapter is not None:
            residual = self.spectral_adapter(residual)
        loss = residual**2
        if weight is not None:
            loss = loss * weight
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


def get_loss_fn(config: Dict[str, Any]) -> DiffusionLoss:
    return DiffusionLoss(config=config)

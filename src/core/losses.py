from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from src.core.numeric import compute_snr, safe_clamp
from src.core.residuals import AdaptiveSNRWeight, compute_residual, weighted_residual_loss
from src.spectral.adapter import SpectralAdapter


class DiffusionLoss(nn.Module):
    """Unified diffusion loss with optional adaptive SNR weighting."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config or {}
        self.mode = str(self.config.get("residual_mode", "pixel"))
        if self.mode not in {"pixel", "spectral"}:
            raise ValueError("residual_mode must be 'pixel' or 'spectral'")

        self.reduction = str(self.config.get("reduction", "mean"))
        if self.reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")

        spectral_cfg = self.config.get("spectral_weighting")
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

        self.use_weighting = bool(self.config.get("use_weighting", True))
        adaptive_default = bool(self.config.get("adaptive_snr", self.use_weighting))
        self._adaptive_params = {
            "beta": float(self.config.get("adaptive_beta", 0.3)),
            "ema_decay": float(self.config.get("adaptive_ema_decay", 0.99)),
            "eps": float(self.config.get("adaptive_eps", 1e-8)),
        }
        self._adaptive_requested = adaptive_default
        self.adaptive = (
            AdaptiveSNRWeight(**self._adaptive_params)
            if self.use_weighting and self._adaptive_requested
            else None
        )

    def set_weighting_enabled(self, enabled: bool, adaptive: Optional[bool] = None) -> None:
        """Toggle SNR weighting and optionally control the adaptive strategy."""

        self.use_weighting = bool(enabled)
        if adaptive is not None:
            self._adaptive_requested = bool(adaptive)
        if not self.use_weighting:
            self.adaptive = None
        else:
            if self._adaptive_requested:
                self.adaptive = AdaptiveSNRWeight(**self._adaptive_params)
            else:
                self.adaptive = None

    def _apply_spectral_weighting(self, residual: torch.Tensor) -> torch.Tensor:
        if self.spectral_adapter is None:
            return residual
        return self.spectral_adapter(residual)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        sqrt_alpha_t: torch.Tensor,
        sqrt_one_minus_alpha_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        sigma_t = safe_clamp(sqrt_one_minus_alpha_t, min_value=1e-6)
        snr = compute_snr(safe_clamp(sqrt_alpha_t, min_value=1e-6), sigma_t)

        residual = compute_residual(prediction, target, mode=self.mode)
        residual = self._apply_spectral_weighting(residual)

        loss, diag = weighted_residual_loss(
            prediction,
            target,
            snr,
            adaptive=self.adaptive if self.use_weighting else None,
            mode="pixel",
            enable_weighting=self.use_weighting,
            residual=residual,
            reduction=self.reduction,
        )

        diag["spectral_weighting"] = 1.0 if self.spectral_adapter is not None else 0.0
        diag["mode_code"] = 0.0 if self.mode == "pixel" else 1.0
        return loss, diag

    def residual_marker(self) -> str:
        """Return a human-readable descriptor for logging."""

        adaptive_flag = 1 if self.adaptive is not None else 0
        return (
            "[Residuals] mode=adaptive_snr v1.2 "
            f"beta={self._adaptive_params['beta']:.3f} ema={self._adaptive_params['ema_decay']:.3f} "
            f"quant_safe=True scale_norm=True residual_mode={self.mode} "
            f"weighting={'on' if self.use_weighting else 'off'} "
            f"adaptive={bool(adaptive_flag)}"
        )


def get_loss_fn(config: Dict[str, Any]) -> DiffusionLoss:
    return DiffusionLoss(config=config)

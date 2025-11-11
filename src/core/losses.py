from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from src.core.adaptive_weight import AdaptiveSNRWeight
from src.core.diffusion_step import describe_regime, predict_x0, select_regime
from src.core.numeric import safe_clamp
from src.core.residuals import compute_residual, weighted_residual_loss
from src.core.snr_scheduler import compute_snr_stats
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
        beta_value = float(self.config.get("adaptive_beta", 0.3))
        self._adaptive_params = {
            "beta": beta_value,
            "ema_decay": float(self.config.get("adaptive_ema_decay", 0.98)),
            "gamma_alpha": float(self.config.get("adaptive_gamma_alpha", 0.25)),
            "kappa_floor": float(self.config.get("adaptive_kappa_floor", 1e-6)),
            "freeze_ratio": float(self.config.get("adaptive_freeze_ratio", 5.0)),
            "delta": float(self.config.get("adaptive_delta", self.config.get("adaptive_eps", 1e-3))),
            "delta_growth": float(self.config.get("adaptive_delta_growth", 1.0)),
            "overflow_target": float(self.config.get("adaptive_overflow_target", 0.01)),
            "overflow_decay": float(self.config.get("adaptive_overflow_decay", 0.9)),
            "change_threshold": float(self.config.get("adaptive_change_threshold", 0.1)),
            "log_interval": int(self.config.get("adaptive_log_interval", 200)),
            "snr_clip": float(
                self.config.get("adaptive_snr_clip", self.config.get("snr_clip", 250.0))
            ),
        }
        self._adaptive_requested = adaptive_default
        self.snr_clip = float(self.config.get("snr_clip", 250.0))
        self.log_snr_smooth = bool(self.config.get("log_snr_smooth", True))
        self.use_overflow_bridge = bool(self.config.get("overflow_bridge", False))
        self.overflow_renorm = bool(self.config.get("overflow_renorm", False))
        self.adaptive = (
            AdaptiveSNRWeight(**self._adaptive_params)
            if self.use_weighting and self._adaptive_requested
            else None
        )
        if self.adaptive:
            import logging
            self.adaptive.set_log_fn(lambda msg, diag: logging.getLogger("AdaptiveSNR").warning(msg))

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
        *,
        x_t: Optional[torch.Tensor] = None,
        x0: Optional[torch.Tensor] = None,
        prediction_type: str = "eps",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        sqrt_alpha = safe_clamp(sqrt_alpha_t, min_value=1e-6)
        sigma_t = safe_clamp(sqrt_one_minus_alpha_t, min_value=1e-6)
        stats = compute_snr_stats(sqrt_alpha, sigma_t, snr_clip=self.snr_clip)
        snr_raw = stats.snr_raw

        snr_clip = self.snr_clip
        regimes = select_regime(snr_raw, snr_clip)
        regime_mode, loss_mode = describe_regime(regimes)
        overflow_mask = regimes["overflow"]

        snr_soft = stats.snr_weight if self.log_snr_smooth else snr_raw
        snr_for_weight = stats.snr_clamped if self.log_snr_smooth else snr_raw

        alpha_t = sqrt_alpha ** 2
        input_std = float(target.detach().std().item()) if target.numel() else 0.0

        mask_broadcast = overflow_mask
        while mask_broadcast.ndim < prediction.ndim:
            mask_broadcast = mask_broadcast.unsqueeze(-1)

        combined_prediction = prediction
        combined_target = target
        raw_loss_override: Optional[torch.Tensor] = None
        mae_tensor: Optional[torch.Tensor] = None

        if (
            self.use_overflow_bridge
            and torch.any(overflow_mask)
            and x_t is not None
            and x0 is not None
        ):
            x0_pred = predict_x0(prediction, prediction_type, x_t, sqrt_alpha, sigma_t)
            combined_prediction = torch.where(mask_broadcast, x0_pred, prediction)
            combined_target = torch.where(mask_broadcast, x0, target)

            residual_pixel = combined_target - combined_prediction
            l2_loss = residual_pixel.pow(2)
            l1_loss = residual_pixel.abs()
            raw_loss_override = torch.where(mask_broadcast, l1_loss, l2_loss)
            mae_tensor = residual_pixel.abs()
        else:
            residual_pixel = combined_target - combined_prediction
            mae_tensor = residual_pixel.abs()

        residual = compute_residual(combined_prediction, combined_target, mode=self.mode)
        residual = self._apply_spectral_weighting(residual)

        overflow_ratio = float(overflow_mask.float().mean().item())
        diag_extra = {"overflow_mask_ratio": overflow_ratio}
        if self.adaptive is not None:
            diag_extra["overflow_ema"] = getattr(self.adaptive, "_overflow_ema", 0.0)

        loss, diag = weighted_residual_loss(
            combined_prediction,
            combined_target,
            snr_for_weight,
            alpha_t=alpha_t,
            adaptive=self.adaptive if self.use_weighting else None,
            mode="pixel",
            enable_weighting=self.use_weighting,
            residual=residual,
            raw_loss=raw_loss_override,
            reduction=self.reduction,
            input_std=input_std,
            overflow_mask=overflow_mask,
            diag_extra=diag_extra,
        )

        diag.update(
            {
                "spectral_weighting": 1.0 if self.spectral_adapter is not None else 0.0,
                "mode_code": 0.0 if self.mode == "pixel" else 1.0,
                "snr_raw_max": float(snr_raw.max().detach().item()),
                "snr_soft_mean": float(snr_soft.mean().detach().item()),
                "snr_log_max": float(stats.log_snr.max().detach().item()),
                "overflow_fraction": float(overflow_mask.float().mean().detach().item()),
                "regime_mode_code": 2.0
                if regime_mode == "deterministic"
                else (1.0 if regime_mode == "hybrid" else 0.0),
                "loss_mode_code": 1.0 if loss_mode == "x0" else 0.0,
            }
        )
        diag["loss_mode_str"] = loss_mode
        diag["regime_mode_str"] = regime_mode
        if mae_tensor is not None:
            diag["mae"] = float(mae_tensor.mean().detach().item())
        return loss, diag

    def residual_marker(self) -> str:
        """Return a human-readable descriptor for logging."""

        adaptive_flag = 1 if self.adaptive is not None else 0
        return (
            "[Residuals] mode=adaptive_snr v1.4 "
            f"beta={self._adaptive_params['beta']:.3f} "
            f"ema={self._adaptive_params['ema_decay']:.3f} "
            f"gamma_alpha={self._adaptive_params['gamma_alpha']:.2f} "
            f"snr_clip={self.snr_clip:.1f} "
            f"freeze={self._adaptive_params['freeze_ratio']:.1f}x "
            f"kappa_floor={self._adaptive_params['kappa_floor']:.2e} "
            f"residual_mode={self.mode} "
            f"weighting={'on' if self.use_weighting else 'off'} "
            f"adaptive={bool(adaptive_flag)}"
        )


def get_loss_fn(config: Dict[str, Any]) -> DiffusionLoss:
    return DiffusionLoss(config=config)

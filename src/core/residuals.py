"""Residual helpers built on the adaptive SNR weighting primitives."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from .adaptive_weight import AdaptiveSNRWeight


def compute_residual(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str = "pixel",
) -> torch.Tensor:
    """Return residuals in the requested domain."""

    if mode == "pixel":
        return target - prediction
    if mode == "spectral":
        fft_pred = torch.fft.fft2(prediction)
        fft_target = torch.fft.fft2(target)
        return torch.fft.ifft2(fft_target - fft_pred).real
    raise ValueError(f"Unsupported residual mode: {mode}")


def weighted_residual_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    snr_weight: torch.Tensor,
    *,
    alpha_t: Optional[torch.Tensor] = None,
    adaptive: Optional[AdaptiveSNRWeight] = None,
    mode: str = "pixel",
    residual: Optional[torch.Tensor] = None,
    raw_loss: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    enable_weighting: bool = True,
    input_std: Optional[float] = None,
    overflow_mask: Optional[torch.Tensor] = None,
    diag_extra: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute a weighted loss and return diagnostic statistics."""

    def _merge_diag(diag: Dict[str, float], extra: Optional[Dict[str, float]]) -> None:
        if extra:
            diag.update(extra)

    if residual is None:
        residual = compute_residual(prediction, target, mode=mode)
    if raw_loss is None:
        raw_loss = residual.pow(2)

    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'")

    if not enable_weighting:
        loss_val = raw_loss.sum() if reduction == "sum" else raw_loss.mean()
        diag = {
            "mean_weight": 1.0,
            "max_weight": 1.0,
            "kappa": 0.0,
            "ema": 0.0,
            "alpha_fac": 1.0,
            "overflow": 0.0,
            "overflow_ema": 0.0,
            "adaptive": 0.0,
            "log_event": False,
            "frozen": False,
            "delta": 0.0,
        }
        _merge_diag(diag, diag_extra)
        return loss_val, diag

    if adaptive is not None and input_std is not None:
        baseline = max(input_std, 1e-6)
        pred_std = float(prediction.detach().std().item()) if prediction.numel() else 0.0
        if pred_std > adaptive.freeze_ratio * baseline:
            ema_tensor = getattr(adaptive, "_ema_val", None)
            ema_value = float(ema_tensor.detach().item()) if torch.is_tensor(ema_tensor) else 0.0
            loss_val = prediction.sum() * 0.0
            diag = {
                "mean_weight": 0.0,
                "max_weight": 0.0,
                "kappa": 0.0,
                "ema": ema_value,
                "alpha_fac": 1.0,
                "overflow": 0.0,
                "overflow_ema": getattr(adaptive, "_overflow_ema", 0.0),
                "adaptive": 1.0,
                "log_event": True,
                "frozen": True,
                "delta": adaptive.delta,
            }
            _merge_diag(diag, diag_extra)
            return loss_val, diag

    if adaptive is not None:
        if alpha_t is None:
            raise ValueError("alpha_t must be provided when using adaptive weighting")
        adaptive.to(prediction.device)
        # Clamp snr_weight to avoid overflow propagation
        snr_clamped = snr_weight.clamp(min=0.0, max=1e6)
        weight, diag = adaptive.update(
            snr_clamped,
            raw_loss,
            alpha_t,
            overflow_mask=overflow_mask,
            diag_extra=diag_extra,
        )
        mean_weight_raw = float(weight.detach().mean().item())
        max_weight_raw = float(weight.detach().max().item())
        # Normalize weights so their mean is approx 1.0 for stable loss scaling
        mean_weight = mean_weight_raw
        if mean_weight > 0:
            weight = weight / mean_weight
            diag["mean_weight"] = 1.0
            diag["max_weight"] = float(weight.detach().max().item())
            diag["mean_weight_normalized"] = float(weight.detach().mean().item())
            diag["max_weight_raw"] = max_weight_raw
            diag["mean_weight_raw"] = mean_weight_raw
        else:
            diag["mean_weight"] = 0.0
            diag["max_weight"] = 0.0
            diag["mean_weight_raw"] = 0.0
            diag["max_weight_raw"] = 0.0
        diag["adaptive"] = 1.0
        diag.setdefault("frozen", False)
    else:
        weight = snr_weight.to(dtype=raw_loss.dtype)
        diag = {
            "mean_weight": float(weight.detach().mean().item()),
            "max_weight": float(weight.detach().max().item()),
            "kappa": 0.0,
            "ema": 0.0,
            "alpha_fac": 1.0,
            "overflow": 0.0,
            "overflow_ema": 0.0,
            "adaptive": 0.0,
            "log_event": False,
            "frozen": False,
            "delta": 0.0,
        }
        diag["mean_weight_raw"] = diag["mean_weight"]
        diag["max_weight_raw"] = diag["max_weight"]
        _merge_diag(diag, diag_extra)

    weighted = weight * raw_loss
    loss_val = weighted.sum() if reduction == "sum" else weighted.mean()
    diag.setdefault("mean_weight", float(weight.detach().mean().item()))
    diag.setdefault("max_weight", float(weight.detach().max().item()))
    return loss_val, diag


# Backwards-compatibility ----------------------------------------------------

AdaptiveSNRWeightAlias = AdaptiveSNRWeight

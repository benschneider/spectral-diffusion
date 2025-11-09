"""Residual handling utilities for diffusion training."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


class AdaptiveSNRWeight:
    """Adaptive SNR weighting with resolution-aware normalisation.

    The module tracks an exponential moving average (EMA) of ``SNR * loss`` so
    that timesteps with extremely confident predictions do not dominate the
    optimisation signal.  The running value is turned into a balance term
    (``kappa``) that shrinks the effective weight when the instantaneous SNR
    is much larger than the EMA estimate.

    The raw SNR is normalised by the square-root of the image area which keeps
    the scale comparable across different resolutions.  The class is safe to
    use with mixed precision because all internal buffers stay in ``float32``
    and the caller can re-create the state if devices change.
    """

    def __init__(
        self,
        *,
        beta: float = 0.3,
        ema_decay: float = 0.99,
        eps: float = 1e-8,
        device: Optional[torch.device] = None,
    ) -> None:
        self.beta = float(beta)
        self.ema_decay = float(ema_decay)
        self.eps = float(eps)
        self.device = device
        self._ema_val: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> None:
        """Attach the adaptive state to ``device`` when known."""

        self.device = device
        if self._ema_val is not None:
            self._ema_val = self._ema_val.to(device)

    def reset(self) -> None:
        """Forget the accumulated EMA statistics."""

        self._ema_val = None

    def update(
        self,
        snr: torch.Tensor,
        raw_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return an adaptive weight tensor and diagnostic statistics."""

        if raw_loss.ndim >= 3:
            area = raw_loss.shape[-2] * raw_loss.shape[-1]
        else:
            area = 1
        norm_factor = torch.tensor(area, dtype=snr.dtype, device=snr.device).sqrt()
        snr_norm = snr / (norm_factor + self.eps)

        detached_loss = raw_loss.detach()
        while detached_loss.ndim > snr_norm.ndim:
            detached_loss = detached_loss.mean(dim=-1)
        val = (snr_norm * detached_loss).mean()

        if self._ema_val is None:
            self._ema_val = val.detach()
        else:
            self._ema_val = self.ema_decay * self._ema_val + (1.0 - self.ema_decay) * val.detach()

        kappa = self.beta * self._ema_val
        adaptive_weight = snr_norm / (snr_norm + kappa + self.eps)

        diag = {
            "kappa": float(kappa.detach().item()),
            "ema": float(self._ema_val.detach().item()),
            "norm": float(norm_factor.detach().item()),
            "mean_weight": float(adaptive_weight.detach().mean().item()),
            "max_weight": float(adaptive_weight.detach().max().item()),
        }
        return adaptive_weight, diag


def compute_residual(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str = "pixel",
) -> torch.Tensor:
    """Return residuals in ``mode`` while keeping pixel alignment."""

    if mode == "pixel":
        return target - pred
    if mode == "spectral":
        fft_pred = torch.fft.fft2(pred)
        fft_target = torch.fft.fft2(target)
        return torch.fft.ifft2(fft_target - fft_pred).real
    raise ValueError(f"Unsupported residual mode: {mode}")


def weighted_residual_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    snr: torch.Tensor,
    *,
    adaptive: Optional[AdaptiveSNRWeight] = None,
    mode: str = "pixel",
    enable_weighting: bool = True,
    residual: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return a weighted residual MSE and associated diagnostics."""

    if residual is None:
        residual = compute_residual(pred, target, mode=mode)
    raw_loss = residual.pow(2)

    if not enable_weighting:
        if reduction == "sum":
            loss_val = raw_loss.sum()
        else:
            loss_val = raw_loss.mean()
        diag = {
            "mean_weight": 1.0,
            "max_weight": 1.0,
            "kappa": 0.0,
            "ema": 0.0,
            "norm": float(raw_loss.shape[-2] * raw_loss.shape[-1]) if raw_loss.ndim >= 3 else 1.0,
            "adaptive": 0.0,
        }
        return loss_val, diag

    if adaptive is not None:
        adaptive.to(pred.device)
        weight, diag = adaptive.update(snr, raw_loss)
        diag["adaptive"] = 1.0
    else:
        weight = snr / (1.0 + snr)
        diag = {
            "mean_weight": float(weight.detach().mean().item()),
            "max_weight": float(weight.detach().max().item()),
            "kappa": 0.0,
            "ema": 0.0,
            "norm": float(raw_loss.shape[-2] * raw_loss.shape[-1]) if raw_loss.ndim >= 3 else 1.0,
            "adaptive": 0.0,
        }

    weighted = weight * raw_loss
    if reduction == "sum":
        loss_val = weighted.sum()
    else:
        loss_val = weighted.mean()
    return loss_val, diag


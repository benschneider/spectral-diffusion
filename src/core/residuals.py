"""Residual handling utilities for diffusion training."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


class AdaptiveSNRWeight:
    """Adaptive SNR weighting with resolution-aware normalisation.

    The module tracks an exponential moving average (EMA) of ``SNR * loss`` in
    ``float32`` so that timesteps with extremely confident predictions do not
    dominate the optimisation signal.  The running value is turned into a
    balance term (``kappa``) that shrinks the effective weight when the
    instantaneous SNR is much larger than the EMA estimate.  A configurable
    floor keeps ``kappa`` from collapsing to zero when losses become tiny.

    The raw SNR is normalised by the square-root of the image area which keeps
    the scale comparable across different resolutions.  Logging occurs only
    when the EMA changes substantially or when a periodic interval elapses,
    preventing per-step spam while still surfacing notable events.  The class
    remains safe to use with mixed precision because all internal buffers stay
    in ``float32`` and the caller can re-create the state if devices change.
    """

    def __init__(
        self,
        *,
        beta: float = 0.3,
        ema_decay: float = 0.99,
        eps: float = 1e-8,
        kappa_floor: float = 1e-4,
        ref_sqrt_area: float = 512.0,
        log_interval: int = 0,
        change_threshold: float = 1e-4,
        device: Optional[torch.device] = None,
    ) -> None:
        self.beta = float(beta)
        self.ema_decay = float(ema_decay)
        self.eps = float(eps)
        self.kappa_floor = float(kappa_floor)
        self.ref_sqrt_area = float(ref_sqrt_area)
        self.log_interval = int(log_interval)
        self.change_threshold = float(change_threshold)
        self.device = device
        self._ema_val: Optional[torch.Tensor] = None
        self._step = 0
        self._last_log_step = 0
        self._last_logged_ema: Optional[float] = None

    def to(self, device: torch.device) -> None:
        """Attach the adaptive state to ``device`` when known."""

        self.device = device
        if self._ema_val is not None:
            self._ema_val = self._ema_val.to(device=device, dtype=torch.float32)

    def reset(self) -> None:
        """Forget the accumulated EMA statistics."""

        self._ema_val = None

    def update(
        self,
        snr: torch.Tensor,
        raw_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return an adaptive weight tensor and diagnostic statistics."""

        target_device = self.device or snr.device

        loss_detached = raw_loss.detach().to(dtype=torch.float32, device=target_device)
        snr_detached = snr.detach().to(dtype=torch.float32, device=target_device)
        if snr_detached.ndim < loss_detached.ndim:
            view_shape = list(snr_detached.shape) + [1] * (loss_detached.ndim - snr_detached.ndim)
            snr_detached = snr_detached.view(*view_shape)

        if loss_detached.ndim >= 3:
            area = float(loss_detached.shape[-2] * loss_detached.shape[-1])
        else:
            area = 1.0
        sqrt_area = area**0.5
        scale = sqrt_area / max(self.ref_sqrt_area, 1.0)

        snr_norm = snr_detached / (scale + self.eps)

        product = snr_norm * loss_detached
        val = product.mean()

        if self._ema_val is None or torch.isnan(self._ema_val):
            self._ema_val = val.detach()
        else:
            self._ema_val = (
                self._ema_val * self.ema_decay
                + val.detach() * (1.0 - self.ema_decay)
            )

        self._ema_val = self._ema_val.to(dtype=torch.float32, device=target_device)

        kappa = torch.clamp(self.beta * self._ema_val, min=self.kappa_floor)
        adaptive_weight = (snr_norm / (snr_norm + kappa + self.eps)).to(raw_loss.dtype)

        self._step += 1
        ema_float = float(self._ema_val.item())
        should_log = self._last_logged_ema is None
        if self.log_interval > 0 and (self._step - self._last_log_step) >= self.log_interval:
            should_log = True
        if (
            self.change_threshold > 0.0
            and self._last_logged_ema is not None
            and abs(ema_float - self._last_logged_ema)
            >= self.change_threshold * max(abs(self._last_logged_ema), 1.0)
        ):
            should_log = True

        if should_log:
            self._last_log_step = self._step
            self._last_logged_ema = ema_float

        diag = {
            "kappa": float(kappa.detach().item()),
            "ema": ema_float,
            "norm": float(sqrt_area),
            "scale": float(scale),
            "val": float(val.detach().item()),
            "mean_weight": float(adaptive_weight.detach().mean().item()),
            "max_weight": float(adaptive_weight.detach().max().item()),
            "log_event": bool(should_log),
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
        if raw_loss.ndim >= 3:
            norm_val = float((raw_loss.shape[-2] * raw_loss.shape[-1]) ** 0.5)
        else:
            norm_val = 1.0
        diag = {
            "mean_weight": 1.0,
            "max_weight": 1.0,
            "kappa": 0.0,
            "ema": 0.0,
            "norm": norm_val,
            "adaptive": 0.0,
            "scale": 1.0,
            "val": 0.0,
            "log_event": False,
        }
        return loss_val, diag

    if adaptive is not None:
        adaptive.to(pred.device)
        weight, diag = adaptive.update(snr, raw_loss)
        diag["adaptive"] = 1.0
    else:
        weight = snr / (1.0 + snr)
        if raw_loss.ndim >= 3:
            norm_val = float((raw_loss.shape[-2] * raw_loss.shape[-1]) ** 0.5)
        else:
            norm_val = 1.0
        diag = {
            "mean_weight": float(weight.detach().mean().item()),
            "max_weight": float(weight.detach().max().item()),
            "kappa": 0.0,
            "ema": 0.0,
            "norm": norm_val,
            "adaptive": 0.0,
            "scale": 1.0,
            "val": 0.0,
            "log_event": False,
        }

    weighted = weight * raw_loss
    if reduction == "sum":
        loss_val = weighted.sum()
    else:
        loss_val = weighted.mean()
    return loss_val, diag


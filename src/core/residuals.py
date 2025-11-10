"""Residual handling utilities for diffusion training."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


class AdaptiveSNRWeight:
    """Self-tuning adaptive SNR weighting.

    The implementation keeps all running statistics in ``float32`` and adjusts
    the effective gain so that the balance term ``kappa`` remains numerically
    meaningful (roughly ``1e-3`` – ``1e-1``).  The normalisation uses the
    current sample's spatial extent instead of a static reference and the raw
    SNR is clipped before weighting which prevents overflow gradients.  Logging
    happens only when the internal state changes significantly or a periodic
    interval elapses, keeping diagnostics informative without spamming output.
    """

    def __init__(
        self,
        *,
        beta: float = 0.3,
        beta_init: Optional[float] = None,
        ema_decay: float = 0.99,
        running_decay: float = 0.98,
        eps: float = 1e-8,
        target_val: float = 1e-2,
        snr_clip: float = 250.0,
        kappa_floor: float = 1e-6,
        log_interval: int = 200,
        change_threshold: float = 0.1,
        device: Optional[torch.device] = None,
        **_unused: object,
    ) -> None:
        base_beta = beta_init if beta_init is not None else beta
        self.base_beta = float(base_beta)
        self.beta = float(base_beta)
        self.ema_decay = float(ema_decay)
        self.running_decay = float(running_decay)
        self.eps = float(eps)
        self.target_val = float(target_val)
        self.snr_clip = float(snr_clip)
        self.kappa_floor = float(kappa_floor)
        self.log_interval = int(log_interval)
        self.change_threshold = float(change_threshold)
        self.device = device

        self._ema_val: Optional[torch.Tensor] = None
        self._running_mean: Optional[torch.Tensor] = None
        self._running_std: Optional[torch.Tensor] = None
        self._step = 0
        self._last_log_step = 0
        self._last_logged: Dict[str, float] = {}

    def to(self, device: torch.device) -> None:
        """Attach the adaptive state to ``device`` when known."""

        self.device = device
        if self._ema_val is not None:
            self._ema_val = self._ema_val.to(device=device, dtype=torch.float32)

    def reset(self) -> None:
        """Forget the accumulated EMA statistics."""

        self._ema_val = None
        self._running_mean = None
        self._running_std = None

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
            view_shape = list(snr_detached.shape) + [1] * (
                loss_detached.ndim - snr_detached.ndim
            )
            snr_detached = snr_detached.view(*view_shape)

        if loss_detached.ndim >= 3:
            area = float(loss_detached.shape[-2] * loss_detached.shape[-1])
        else:
            area = 1.0
        sqrt_area = area**0.5

        snr_clamped = torch.clamp(snr_detached, max=self.snr_clip)
        snr_norm = snr_clamped / (sqrt_area + self.eps)

        product = snr_norm * loss_detached
        if product.ndim > 1:
            per_example = product.reshape(product.shape[0], -1).mean(dim=1)
        else:
            per_example = product
        val_mean = per_example.mean()

        if self._running_mean is None:
            self._running_mean = val_mean.detach()
            self._running_std = torch.abs(val_mean.detach()) + self.eps
        else:
            self._running_mean = (
                self._running_mean * self.running_decay
                + val_mean.detach() * (1.0 - self.running_decay)
            )
            deviation = torch.abs(val_mean.detach() - self._running_mean)
            self._running_std = (
                self._running_std * self.running_decay
                + deviation * (1.0 - self.running_decay)
            )
            self._running_std = torch.clamp(self._running_std, min=self.eps)

        normed_val = val_mean.detach() / (self._running_std + self.eps)

        if self._ema_val is None or torch.isnan(self._ema_val):
            self._ema_val = normed_val
        else:
            self._ema_val = (
                self._ema_val * self.ema_decay
                + normed_val * (1.0 - self.ema_decay)
            )

        self._ema_val = self._ema_val.to(dtype=torch.float32, device=target_device)
        self._running_mean = self._running_mean.to(dtype=torch.float32, device=target_device)
        self._running_std = self._running_std.to(dtype=torch.float32, device=target_device)

        beta_adjust = self.target_val / (torch.abs(self._ema_val) + self.eps)
        beta_eff = torch.clamp(
            self.base_beta * beta_adjust,
            min=0.1 * self.base_beta,
            max=10.0 * self.base_beta,
        )

        kappa = torch.clamp(beta_eff * torch.abs(self._ema_val), min=self.kappa_floor)
        adaptive_weight = (snr_norm / (snr_norm + kappa + self.eps)).to(raw_loss.dtype)

        self._step += 1
        diag = {
            "kappa": float(kappa.detach().mean().item()),
            "ema": float(self._ema_val.detach().item()),
            "running_std": float(self._running_std.detach().item()),
            "beta_eff": float(beta_eff.detach().item()),
            "norm": float(sqrt_area),
            "scale": float(sqrt_area),
            "val": float(val_mean.detach().item()),
            "mean_weight": float(adaptive_weight.detach().mean().item()),
            "max_weight": float(adaptive_weight.detach().max().item()),
        }

        should_log = False
        if not self._last_logged:
            should_log = True
        elif self.change_threshold > 0.0:
            for key in ("kappa", "beta_eff", "ema"):
                previous = self._last_logged.get(key)
                current = diag[key]
                if previous is None:
                    should_log = True
                    break
                baseline = max(abs(previous), 1e-6)
                if abs(current - previous) / baseline >= self.change_threshold:
                    should_log = True
                    break
        if not should_log and self.log_interval > 0:
            if (self._step - self._last_log_step) >= self.log_interval:
                should_log = True

        if should_log:
            self._last_log_step = self._step
            self._last_logged = {key: diag[key] for key in ("kappa", "beta_eff", "ema")}

        diag["log_event"] = bool(should_log)
        diag["snr_clipped"] = bool((snr_detached > self.snr_clip).any().item())
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
            "beta_eff": 0.0,
            "running_std": 0.0,
            "snr_clipped": False,
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
            "beta_eff": 0.0,
            "running_std": 0.0,
            "snr_clipped": False,
        }

    weighted = weight * raw_loss
    if reduction == "sum":
        loss_val = weighted.sum()
    else:
        loss_val = weighted.mean()
    return loss_val, diag


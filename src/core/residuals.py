"""Residual handling utilities for diffusion training."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch


def _expand_like(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast ``tensor`` to match ``reference`` dimensionality."""

    if tensor.ndim >= reference.ndim:
        return tensor
    view = list(tensor.shape) + [1] * (reference.ndim - tensor.ndim)
    return tensor.view(*view)


class AdaptiveSNRv14:
    """Log-SNR adaptive weighting with α-residual stabilisation and auto-freeze."""

    def __init__(
        self,
        *,
        beta: float = 0.3,
        ema_decay: float = 0.98,
        gamma_alpha: float = 0.25,
        eps: float = 1e-8,
        snr_clip: float = 250.0,
        kappa_floor: float = 1e-6,
        freeze_ratio: float = 5.0,
        log_interval: int = 200,
        change_threshold: float = 0.1,
    ) -> None:
        self.beta = float(beta)
        self.ema_decay = float(ema_decay)
        self.gamma_alpha = float(gamma_alpha)
        self.eps = float(eps)
        self.snr_clip = float(snr_clip)
        self.kappa_floor = float(kappa_floor)
        self.freeze_ratio = float(freeze_ratio)
        self.log_interval = int(log_interval)
        self.change_threshold = float(change_threshold)

        self._ema_val: Optional[torch.Tensor] = None
        self._step = 0
        self._last_log_step = 0
        self._last_logged: Dict[str, float] = {}

    def to(self, device: torch.device) -> None:  # pragma: no cover - helper
        if self._ema_val is not None:
            self._ema_val = self._ema_val.to(device=device, dtype=torch.float32)

    def reset(self) -> None:
        """Forget accumulated EMA statistics."""

        self._ema_val = None
        self._step = 0
        self._last_log_step = 0
        self._last_logged.clear()

    def _centre(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim <= 1:
            return tensor - tensor.mean()
        dims: Iterable[int] = tuple(range(1, tensor.ndim))
        return tensor - tensor.mean(dim=dims, keepdim=True)

    def _should_log(self, diag: Dict[str, float]) -> bool:
        if not self._last_logged:
            return True
        if self.change_threshold > 0.0:
            for key in ("kappa", "ema", "mean_weight"):
                previous = self._last_logged.get(key)
                if previous is None:
                    return True
                baseline = max(abs(previous), 1e-6)
                if abs(diag[key] - previous) / baseline >= self.change_threshold:
                    return True
        if self.log_interval > 0 and (self._step - self._last_log_step) >= self.log_interval:
            return True
        return False

    def update(
        self,
        snr: torch.Tensor,
        raw_loss: torch.Tensor,
        alpha_t: torch.Tensor,
        diag_extra: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return adaptive weights and diagnostics for the provided tensors."""

        snr_detached = torch.clamp(
            snr.detach().to(dtype=torch.float32), min=self.eps, max=self.snr_clip
        )
        loss_detached = raw_loss.detach().to(dtype=torch.float32)
        alpha_detached = alpha_t.detach().to(dtype=torch.float32)

        snr_detached = _expand_like(snr_detached, loss_detached)
        alpha_detached = _expand_like(alpha_detached, snr_detached)

        logsnr = self._centre(torch.log(snr_detached + self.eps))
        loss_scale = loss_detached.abs().mean().clamp_min(self.eps)
        loss_norm = loss_detached / loss_scale

        exp_logsnr = torch.exp(logsnr)
        if loss_norm.ndim > 1:
            dims: Iterable[int] = tuple(range(1, loss_norm.ndim))
            per_example = (loss_norm * exp_logsnr).mean(dim=dims)
        else:
            per_example = (loss_norm * exp_logsnr).mean()

        val_mean = per_example.mean()
        if self._ema_val is None or torch.isnan(self._ema_val):
            self._ema_val = val_mean
        else:
            self._ema_val = (
                self._ema_val * self.ema_decay + val_mean * (1.0 - self.ema_decay)
            )
        self._ema_val = self._ema_val.to(dtype=torch.float32)

        alpha_fac = 1.0 + self.gamma_alpha * (1.0 - alpha_detached.clamp(0.0, 1.0)).mean()
        kappa = torch.clamp(self.beta * self._ema_val * alpha_fac, min=self.kappa_floor)

        kappa_broadcast = _expand_like(kappa, snr_detached)
        weight = snr_detached / (snr_detached + kappa_broadcast + self.eps)

        overflow_mask = snr_detached >= (self.snr_clip * 0.9)
        if overflow_mask.any():
            weight = torch.where(overflow_mask, torch.zeros_like(weight), weight)

        weight = weight.to(dtype=raw_loss.dtype)
        self._step += 1

        diag = {
            "kappa": float(kappa.detach().item()),
            "ema": float(self._ema_val.detach().item()),
            "alpha_fac": float(alpha_fac.detach().item()),
            "overflow": float(overflow_mask.float().mean().item()),
            "mean_weight": float(weight.detach().mean().item()),
            "max_weight": float(weight.detach().max().item()),
        }
        if diag_extra:
            diag.update(diag_extra)

        log_event = self._should_log(diag)
        diag["log_event"] = log_event
        if log_event:
            self._last_log_step = self._step
            self._last_logged = {key: diag[key] for key in ("kappa", "ema", "mean_weight")}
            print(
                "[AdaptiveSNRv14] step="
                f"{self._step:04d} κ={diag['kappa']:.3e} ema={diag['ema']:.3e} "
                f"α_fac={diag['alpha_fac']:.2f} overflow={diag['overflow']:.3f}"
            )

        return weight, diag


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
    alpha_t: Optional[torch.Tensor] = None,
    adaptive: Optional[AdaptiveSNRv14] = None,
    mode: str = "pixel",
    enable_weighting: bool = True,
    residual: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    input_std: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return a weighted residual MSE and associated diagnostics."""

    if residual is None:
        residual = compute_residual(pred, target, mode=mode)
    raw_loss = residual.pow(2)

    if not enable_weighting:
        loss_val = raw_loss.sum() if reduction == "sum" else raw_loss.mean()
        diag = {
            "mean_weight": 1.0,
            "max_weight": 1.0,
            "kappa": 0.0,
            "ema": 0.0,
            "alpha_fac": 1.0,
            "overflow": 0.0,
            "adaptive": 0.0,
            "log_event": False,
            "frozen": False,
        }
        return loss_val, diag

    if adaptive is not None and input_std is not None:
        baseline = max(input_std, 1e-6)
        if float(pred.detach().std().item()) > adaptive.freeze_ratio * baseline:
            loss_val = pred.sum() * 0.0
            ema_val = 0.0
            if getattr(adaptive, "_ema_val", None) is not None:  # pylint: disable=protected-access
                ema_tensor = adaptive._ema_val  # type: ignore[attr-defined]  # pylint: disable=protected-access
                ema_val = float(ema_tensor.detach().item())
            diag = {
                "mean_weight": 0.0,
                "max_weight": 0.0,
                "kappa": 0.0,
                "ema": ema_val,
                "alpha_fac": 1.0,
                "overflow": 0.0,
                "adaptive": 1.0,
                "log_event": False,
                "frozen": True,
            }
            return loss_val, diag

    if adaptive is not None:
        if alpha_t is None:
            raise ValueError("alpha_t must be provided when using adaptive weighting")
        adaptive.to(pred.device)
        weight, diag = adaptive.update(snr, raw_loss, alpha_t)
        diag["adaptive"] = 1.0
        diag.setdefault("frozen", False)
    else:
        weight = (snr / (1.0 + snr)).to(dtype=raw_loss.dtype)
        diag = {
            "mean_weight": float(weight.detach().mean().item()),
            "max_weight": float(weight.detach().max().item()),
            "kappa": 0.0,
            "ema": 0.0,
            "alpha_fac": 1.0,
            "overflow": 0.0,
            "adaptive": 0.0,
            "log_event": False,
            "frozen": False,
        }

    weighted = weight * raw_loss
    loss_val = weighted.sum() if reduction == "sum" else weighted.mean()
    diag.setdefault("mean_weight", float(weight.detach().mean().item()))
    diag.setdefault("max_weight", float(weight.detach().max().item()))
    return loss_val, diag


# Backwards compatibility alias -------------------------------------------------

AdaptiveSNRWeight = AdaptiveSNRv14


"""Adaptive weighting helpers for diffusion training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch

EPS = 1e-8


def _expand_to(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast ``tensor`` so that it matches ``reference`` dimensions."""

    if tensor.ndim >= reference.ndim:
        return tensor
    view = list(tensor.shape) + [1] * (reference.ndim - tensor.ndim)
    return tensor.view(*view)


def _mean_over_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim <= 1:
        return tensor
    dims: Iterable[int] = tuple(range(1, tensor.ndim))
    return tensor.mean(dim=dims)


@dataclass
class AdaptiveDiagnostics:
    """Lightweight container for adaptive-weight statistics."""

    mean_weight: float
    max_weight: float
    kappa: float
    ema: float
    alpha_gap: float
    overflow: float
    overflow_ema: float
    delta: float
    log_event: bool

    def as_dict(self) -> Dict[str, float]:
        return {
            "mean_weight": self.mean_weight,
            "max_weight": self.max_weight,
            "kappa": self.kappa,
            "ema": self.ema,
            "alpha_fac": self.alpha_gap,
            "overflow": self.overflow,
            "overflow_ema": self.overflow_ema,
            "delta": self.delta,
            "log_event": float(self.log_event),
        }


class AdaptiveSNRWeight:
    """Self-tuning SNR weighting with log-SNR smoothing and α-damping."""

    def __init__(
        self,
        *,
        beta: float = 0.3,
        ema_decay: float = 0.98,
        gamma_alpha: float = 0.25,
        kappa_floor: float = 1e-6,
        freeze_ratio: float = 5.0,
        delta: float = 1e-3,
        overflow_target: float = 0.01,
        overflow_decay: float = 0.9,
        change_threshold: float = 0.1,
        log_interval: int = 200,
    ) -> None:
        self.beta = float(beta)
        self.ema_decay = float(ema_decay)
        self.gamma_alpha = float(gamma_alpha)
        self.kappa_floor = float(kappa_floor)
        self.freeze_ratio = float(freeze_ratio)
        self.delta = float(delta)
        self.overflow_target = float(overflow_target)
        self.overflow_decay = float(overflow_decay)
        self.change_threshold = float(change_threshold)
        self.log_interval = int(log_interval)

        self._ema_val: Optional[torch.Tensor] = None
        self._overflow_ema: float = 0.0
        self._step = 0
        self._last_log_step = 0
        self._last_diag: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # housekeeping helpers
    def to(self, device: torch.device) -> None:  # pragma: no cover - helper
        if self._ema_val is not None:
            self._ema_val = self._ema_val.to(device=device, dtype=torch.float32)

    def reset(self) -> None:
        self._ema_val = None
        self._overflow_ema = 0.0
        self._step = 0
        self._last_log_step = 0
        self._last_diag.clear()

    # ------------------------------------------------------------------
    def _compute_weight_core(
        self,
        snr: torch.Tensor,
        raw_loss: torch.Tensor,
        alpha_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, float, float, float]:
        snr_detached = snr.detach().to(dtype=torch.float32)
        loss_detached = raw_loss.detach().to(dtype=torch.float32)
        alpha_detached = alpha_t.detach().to(dtype=torch.float32)

        snr_detached = _expand_to(snr_detached, loss_detached)
        alpha_detached = _expand_to(alpha_detached, snr_detached)

        per_example_snr = _mean_over_batch(snr_detached)
        per_example_loss = _mean_over_batch(loss_detached)

        loss_scale = per_example_loss.mean().clamp_min(EPS)
        norm_loss = per_example_loss / loss_scale

        log_snr = torch.log(per_example_snr.clamp_min(EPS))
        soft_weight = torch.tanh(0.5 * log_snr).clamp(0.0, 1.0)
        soft_weight = soft_weight + EPS

        value = (soft_weight * norm_loss).mean()
        if self._ema_val is None or torch.isnan(self._ema_val):
            self._ema_val = value.detach()
        else:
            self._ema_val = (
                self._ema_val * self.ema_decay + value.detach() * (1.0 - self.ema_decay)
            )
        self._ema_val = self._ema_val.to(dtype=torch.float32)

        alpha_gap = (1.0 - alpha_detached.clamp(0.0, 1.0)).mean()
        alpha_term = 1.0 + self.gamma_alpha * float(alpha_gap.item())

        base = torch.abs(self._ema_val) * self.beta
        kappa = torch.clamp(base * alpha_term + self.delta, min=self.kappa_floor)

        expanded_soft = _expand_to(soft_weight, loss_detached).to(raw_loss.dtype)
        kappa_expanded = _expand_to(kappa, expanded_soft)
        weight = expanded_soft / (expanded_soft + kappa_expanded + EPS)

        return weight, float(kappa.item()), float(self._ema_val.item()), alpha_term

    def _should_log(self, diag: Dict[str, float]) -> bool:
        if not self._last_diag:
            return True
        if self.change_threshold > 0:
            for key in ("kappa", "ema", "mean_weight"):
                previous = self._last_diag.get(key)
                if previous is None:
                    return True
                baseline = max(abs(previous), EPS)
                if abs(diag[key] - previous) / baseline >= self.change_threshold:
                    return True
        if self.log_interval > 0 and (self._step - self._last_log_step) >= self.log_interval:
            return True
        return False

    # ------------------------------------------------------------------
    def update(
        self,
        snr: torch.Tensor,
        raw_loss: torch.Tensor,
        alpha_t: torch.Tensor,
        *,
        overflow_mask: Optional[torch.Tensor] = None,
        diag_extra: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return adaptive weights and diagnostics."""

        weight, kappa_value, ema_value, alpha_fac = self._compute_weight_core(
            snr, raw_loss, alpha_t
        )

        overflow_ratio = 0.0
        if overflow_mask is not None:
            overflow_mask = _expand_to(overflow_mask.detach().float(), weight)
            if torch.any(overflow_mask > 0):
                weight = torch.where(overflow_mask > 0, torch.zeros_like(weight), weight)
                overflow_ratio = float(overflow_mask.mean().item())

        self._overflow_ema = (
            self.overflow_decay * self._overflow_ema
            + (1.0 - self.overflow_decay) * overflow_ratio
        )

        mean_weight = float(weight.detach().mean().item())
        max_weight = float(weight.detach().max().item())

        diag = {
            "mean_weight": mean_weight,
            "max_weight": max_weight,
            "kappa": kappa_value,
            "ema": ema_value,
            "alpha_fac": alpha_fac,
            "overflow": overflow_ratio,
            "overflow_ema": self._overflow_ema,
            "delta": self.delta,
        }
        if diag_extra:
            diag.update(diag_extra)

        self._step += 1
        should_log = self._should_log(diag)
        diag["log_event"] = should_log
        if should_log:
            self._last_log_step = self._step
            self._last_diag = {key: diag[key] for key in ("kappa", "ema", "mean_weight")}
            print(
                "[AdaptiveSNR] step="
                f"{self._step:04d} κ={diag['kappa']:.3e} ema={diag['ema']:.3e} "
                f"α_fac={diag['alpha_fac']:.2f} overflow={diag['overflow']:.3f} "
                f"overflow_ema={diag['overflow_ema']:.3f}"
            )

        return weight, diag


# Backwards compatibility ----------------------------------------------------

AdaptiveSNRv14 = AdaptiveSNRWeight


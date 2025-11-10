"""Adaptive controller for progressive SNR scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from src.analysis.learning_efficiency import compute_efficiency
from src.analysis.trend_filters import EWMA


@dataclass
class _ControllerState:
    ratio: float
    step: int = 0
    prev_loss: Optional[float] = None


class AdaptiveSNRController:
    """Regulate the effective SNR ratio using diagnostics from training."""

    def __init__(
        self,
        min_snr: float,
        max_snr: float,
        inc: float,
        dec: float,
        kappa_thresh: float,
        alpha_fac_high: float,
        overflow_high: float,
        *,
        initial_ratio: Optional[float] = None,
        loss_beta: float = 0.9,
        efficiency_beta: float = 0.6,
    ) -> None:
        if min_snr <= 0:
            raise ValueError("min_snr must be positive")
        if max_snr <= min_snr:
            raise ValueError("max_snr must exceed min_snr")
        self.min_snr = float(min_snr)
        self.max_snr = float(max_snr)
        self.inc = float(inc)
        self.dec = float(dec)
        self.kappa_thresh = float(kappa_thresh)
        self.alpha_fac_high = float(alpha_fac_high)
        self.overflow_high = float(overflow_high)
        start_ratio = float(initial_ratio) if initial_ratio is not None else self.min_snr
        self._state = _ControllerState(ratio=self._clamp_ratio(start_ratio))
        self._loss_filter = EWMA(beta=loss_beta)
        self._efficiency_filter = EWMA(beta=efficiency_beta)
        self._last_metrics: Dict[str, float] = {}

    @property
    def ratio(self) -> float:
        return self._state.ratio

    @property
    def latest_metrics(self) -> Dict[str, float]:
        return dict(self._last_metrics)

    def _clamp_ratio(self, value: float) -> float:
        return float(min(max(value, self.min_snr), self.max_snr))

    def _compute_overflow(self, snr_vals: torch.Tensor) -> Tuple[float, float, float]:
        if snr_vals.numel() == 0:
            return 0.0, 0.0, 0.0
        snr_cpu = snr_vals.detach().float()
        snr_max = float(snr_cpu.max().item())
        snr_mean = float(snr_cpu.mean().item())
        overflow_ratio = float((snr_cpu > self.max_snr).float().mean().item())
        return snr_mean, snr_max, overflow_ratio

    def _should_decay(self, adaptive_diag: Optional[Dict[str, float]], overflow_ratio: float) -> bool:
        if overflow_ratio > self.overflow_high:
            return True
        if not adaptive_diag:
            return False
        if adaptive_diag.get("overflow", 0.0) > self.overflow_high:
            return True
        if adaptive_diag.get("kappa", 0.0) > self.kappa_thresh:
            return True
        if adaptive_diag.get("alpha_fac", 0.0) > self.alpha_fac_high:
            return True
        return False

    def _should_accelerate(
        self,
        efficiency_value: Optional[float],
        fft_feedback: Dict[str, float],
    ) -> bool:
        if efficiency_value is None or efficiency_value <= 0.0:
            return False
        if self._efficiency_filter.slope <= 0.0:
            return False
        high_band = fft_feedback.get("amplitude_high_mae")
        if high_band is None:
            return True
        return high_band < fft_feedback.get("amplitude_mid_mae", high_band)

    def update(
        self,
        loss: float,
        grad_norm: float,
        fft_feedback: Dict[str, float],
        adaptive_diag: Optional[Dict[str, float]],
        snr_vals: torch.Tensor,
    ) -> Tuple[float, Optional[str]]:
        """Return the next SNR ratio and an optional log message."""

        loss = float(loss)
        grad_norm = float(grad_norm)
        self._state.step += 1

        smoothed_loss = self._loss_filter.update(loss)
        efficiency = compute_efficiency(self._state.prev_loss, loss, grad_norm)
        efficiency_smoothed = None
        if efficiency is not None:
            efficiency_smoothed = self._efficiency_filter.update(efficiency)
        self._state.prev_loss = loss

        snr_mean, snr_max, overflow_ratio = self._compute_overflow(snr_vals)
        headroom = float(self.max_snr - snr_max)

        prev_ratio = self._state.ratio
        ratio = prev_ratio
        reasons: list[str] = []

        if self._should_decay(adaptive_diag, overflow_ratio):
            ratio = self._clamp_ratio(ratio * self.dec)
            reasons.append("decay")
        elif self._should_accelerate(efficiency_smoothed, fft_feedback):
            ratio = self._clamp_ratio(ratio * self.inc)
            reasons.append("boost")

        changed = ratio != prev_ratio
        self._state.ratio = ratio

        self._last_metrics = {
            "snr_ratio": ratio,
            "snr_headroom": headroom,
            "snr_max": snr_max,
            "snr_mean": snr_mean,
            "overflow_ratio": overflow_ratio,
            "loss_ema": smoothed_loss,
            "efficiency": efficiency_smoothed if efficiency_smoothed is not None else 0.0,
            "efficiency_slope": self._efficiency_filter.slope,
        }

        if changed:
            reason_str = ",".join(reasons) if reasons else "update"
            note = (
                "step={step} ratio={ratio:.3f} reason={reason} headroom={headroom:.3f} "
                "overflow={overflow:.3f} eff={eff:.3e}"
            ).format(
                step=self._state.step,
                ratio=ratio,
                reason=reason_str,
                headroom=headroom,
                overflow=overflow_ratio,
                eff=self._last_metrics["efficiency"],
            )
            return ratio, note
        return ratio, None

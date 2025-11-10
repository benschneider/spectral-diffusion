"""Shared adaptive regulator helpers for SNR controllers and weighting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional


@dataclass
class AdaptiveRegulatorMetrics:
    """Container for adaptive regulator telemetry."""

    kappa: float = 0.0
    ema: float = 0.0
    overflow: float = 0.0
    overflow_ema: float = 0.0
    alpha_fac: float = 1.05
    snr_target: float = 0.0

    def update_from_diag(self, diag: Optional[Mapping[str, float]]) -> None:
        if not diag:
            return
        if "kappa" in diag:
            self.kappa = float(diag["kappa"])
        if "ema" in diag:
            self.ema = float(diag["ema"])
        if "overflow" in diag:
            self.overflow = float(diag["overflow"])
        if "overflow_ema" in diag:
            self.overflow_ema = float(diag["overflow_ema"])
        if "alpha_fac" in diag:
            self.alpha_fac = float(diag["alpha_fac"])
        if "snr_target" in diag:
            self.snr_target = float(diag["snr_target"])

    def as_dict(self) -> Dict[str, float]:
        return {
            "kappa": self.kappa,
            "ema": self.ema,
            "overflow": self.overflow,
            "overflow_ema": self.overflow_ema,
            "alpha_fac": self.alpha_fac,
            "snr_target": self.snr_target,
        }


def compute_alpha_fac(kappa: Optional[float], ema: Optional[float]) -> float:
    """Compute adaptive alpha factor with responsiveness clamp."""

    base = 1.05
    if kappa is None or ema is None:
        return base
    value = base + 0.4 * abs(float(kappa) - float(ema))
    return float(min(max(value, 1.0), 1.3))


def blend_overflow_ema(
    prev: float,
    overflow_ratio: float,
    diag_overflow: Optional[float],
    diag_overflow_ema: Optional[float],
) -> float:
    """Combine overflow signals with the regulator smoothing rules."""

    overflow_signal = overflow_ratio
    if diag_overflow is not None:
        overflow_signal = max(overflow_signal, float(diag_overflow))
    ema = 0.8 * prev + 0.2 * overflow_signal
    if diag_overflow_ema is not None:
        ema = 0.5 * ema + 0.5 * float(diag_overflow_ema)
    return ema

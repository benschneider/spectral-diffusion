"""Centralised adaptive SNR governor and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch


def _to_float(value: Optional[float], default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


class _EWMA:
    """Simple exponential moving average tracker."""

    def __init__(self, beta: float) -> None:
        if not 0.0 < beta < 1.0:
            raise ValueError("beta must lie in (0, 1)")
        self.beta = beta
        self.value: Optional[float] = None

    def update(self, sample: float) -> float:
        sample = float(sample)
        if self.value is None:
            self.value = sample
        else:
            self.value = self.beta * self.value + (1.0 - self.beta) * sample
        return self.value


@dataclass
class AdaptiveRegulatorMetrics:
    """Container for adaptive regulator telemetry."""

    kappa: float = 0.0
    ema: float = 0.0
    overflow: float = 0.0
    overflow_ema: float = 0.0
    alpha_fac: float = 1.05
    snr_target: float = 0.0
    micro_reset: float = 0.0
    variance_ratio: float = 1.0
    hard_fraction: float = 0.0
    delta_loss: float = 0.0
    band_hard: float = 0.35
    band_medium: float = 0.4
    band_easy: float = 0.25
    lambda_var: float = 0.0

    def update_from_diag(self, diag: Optional[Mapping[str, float]]) -> None:
        if not diag:
            return
        self.kappa = _to_float(diag.get("kappa"), self.kappa)
        self.ema = _to_float(diag.get("ema"), self.ema)
        self.overflow = _to_float(diag.get("overflow"), self.overflow)
        self.overflow_ema = _to_float(diag.get("overflow_ema"), self.overflow_ema)
        self.alpha_fac = _to_float(diag.get("alpha_fac"), self.alpha_fac)
        self.snr_target = _to_float(diag.get("snr_target"), self.snr_target)
        self.micro_reset = _to_float(diag.get("micro_reset"), self.micro_reset)

    def as_dict(self) -> Dict[str, float]:
        return {
            "kappa": self.kappa,
            "ema": self.ema,
            "overflow": self.overflow,
            "overflow_ema": self.overflow_ema,
            "alpha_fac": self.alpha_fac,
            "snr_target": self.snr_target,
            "micro_reset": self.micro_reset,
            "variance_ratio": self.variance_ratio,
            "hard_fraction": self.hard_fraction,
            "delta_loss": self.delta_loss,
            "band_hard": self.band_hard,
            "band_medium": self.band_medium,
            "band_easy": self.band_easy,
            "lambda_var": self.lambda_var,
        }


@dataclass(frozen=True)
class MicroResetPolicy:
    """Encapsulate the periodic micro-reset behaviour."""

    period: int = 200
    kappa_scale: float = 1.2
    overflow_scale: float = 0.5

    def should_reset(self, step: int) -> bool:
        return step > 0 and step % self.period == 0

    def factors(self, step: int) -> Tuple[float, float, bool]:
        if self.should_reset(step):
            return self.kappa_scale, self.overflow_scale, True
        return 1.0, 1.0, False

    def apply(self, metrics: AdaptiveRegulatorMetrics, step: int) -> bool:
        if not self.should_reset(step):
            metrics.micro_reset = 0.0
            return False
        metrics.kappa *= self.kappa_scale
        metrics.overflow_ema *= self.overflow_scale
        metrics.micro_reset = 1.0
        return True


@dataclass
class _BandTracker:
    window: int = 64
    eta: float = 0.30
    hard_bounds: Tuple[float, float] = (0.25, 0.65)
    easy_bounds: Tuple[float, float] = (0.1, 0.5)

    counts: Dict[str, float] = field(default_factory=lambda: {"hard": 0.0, "medium": 0.0, "easy": 0.0})
    total: int = 0
    probs: Dict[str, float] = field(default_factory=lambda: {"hard": 0.35, "medium": 0.4, "easy": 0.25})

    def register(self, hard: float, medium: float, easy: float) -> None:
        self.counts["hard"] += hard
        self.counts["medium"] += medium
        self.counts["easy"] += easy
        self.total += 1

    def _renorm(self) -> None:
        total = sum(self.probs.values())
        if total <= 0.0:
            self.probs = {"hard": 0.35, "medium": 0.4, "easy": 0.25}
            total = 1.0
        for key in self.probs:
            self.probs[key] = max(0.0, self.probs[key])
        total = sum(self.probs.values())
        if total <= 0.0:
            self.probs = {"hard": 0.35, "medium": 0.4, "easy": 0.25}
            total = 1.0
        for key in self.probs:
            self.probs[key] /= total

    def set_probs(self, probs: Mapping[str, float]) -> None:
        hard = float(probs.get("hard", self.probs["hard"]))
        easy = float(probs.get("easy", self.probs["easy"]))
        medium = float(probs.get("medium", 1.0 - hard - easy))
        hard = float(min(max(hard, self.hard_bounds[0]), self.hard_bounds[1]))
        easy = float(min(max(easy, self.easy_bounds[0]), self.easy_bounds[1]))
        medium = float(max(0.0, medium))
        total = hard + medium + easy
        if total <= 0.0:
            self.probs = {"hard": 0.35, "medium": 0.4, "easy": 0.25}
            return
        self.probs = {"hard": hard / total, "medium": medium / total, "easy": easy / total}
        self.probs["hard"] = float(
            min(max(self.probs["hard"], self.hard_bounds[0]), self.hard_bounds[1]))
        self.probs["easy"] = float(
            min(max(self.probs["easy"], self.easy_bounds[0]), self.easy_bounds[1]))
        remainder = max(0.0, 1.0 - self.probs["hard"] - self.probs["easy"])
        self.probs["medium"] = remainder
        self._renorm()

    def maybe_update(self, overflow_gap: float) -> Dict[str, float]:
        if self.total < self.window:
            return dict(self.probs)
        hard_frac = self.counts["hard"] / max(self.total, 1)
        easy_frac = self.counts["easy"] / max(self.total, 1)
        target_hard = 0.35
        target_easy = 0.25

        self.probs["hard"] += self.eta * (target_hard - hard_frac)
        self.probs["hard"] = float(
            min(max(self.probs["hard"], self.hard_bounds[0]), self.hard_bounds[1]))

        self.probs["easy"] += self.eta * (target_easy - easy_frac)
        self.probs["easy"] = float(
            min(max(self.probs["easy"], self.easy_bounds[0]), self.easy_bounds[1]))

        if overflow_gap > 0.0:
            self.probs["hard"] = float(
                min(self.hard_bounds[1], self.probs["hard"] + self.eta * overflow_gap))
            self.probs["easy"] = float(
                max(self.easy_bounds[0], self.probs["easy"] - self.eta * overflow_gap))

        self.probs["medium"] = max(0.0, 1.0 - self.probs["hard"] - self.probs["easy"])
        self._renorm()
        self.counts = {"hard": 0.0, "medium": 0.0, "easy": 0.0}
        self.total = 0
        return dict(self.probs)


def compute_alpha_fac(kappa: Optional[float], ema: Optional[float]) -> float:
    base = 1.05
    if kappa is None or ema is None:
        return base
    value = base + 0.4 * abs(float(kappa) - float(ema))
    return float(min(max(value, 1.0), 1.3))


def blend_overflow_ema(prev: float, overflow: float, *, diag_overflow: Optional[float] = None,
                       diag_overflow_ema: Optional[float] = None) -> float:
    overflow_signal = float(max(overflow, _to_float(diag_overflow, overflow)))
    ema = 0.8 * prev + 0.2 * overflow_signal
    if diag_overflow_ema is not None:
        ema = 0.5 * ema + 0.5 * float(diag_overflow_ema)
    return ema


def normalise_weights(weights: torch.Tensor, snr: torch.Tensor) -> torch.Tensor:
    """Normalise per-sample weights with band-aware scaling."""

    if weights.numel() == 0:
        return weights

    per_example = weights
    if weights.ndim > 1:
        dims: Iterable[int] = tuple(range(1, weights.ndim))
        per_example = weights.mean(dim=dims)

    snr_detached = snr.detach().to(dtype=weights.dtype)
    while snr_detached.ndim < per_example.ndim:
        snr_detached = snr_detached.unsqueeze(-1)

    hard_mask = (snr_detached >= 0.4) & (snr_detached < 0.8)
    easy_mask = (snr_detached >= 1.4) & (snr_detached <= 2.4)
    scale = torch.ones_like(per_example)
    scale = torch.where(hard_mask, scale * 1.1, scale)
    scale = torch.where(easy_mask, scale * 0.9, scale)

    expanded_scale = scale
    if weights.ndim > scale.ndim:
        expanded_scale = scale.view(scale.shape + (1,) * (weights.ndim - scale.ndim))
    weights = weights * expanded_scale

    mean_val = weights.mean()
    if torch.isfinite(mean_val) and mean_val.abs() > 0:
        weights = weights / (mean_val + 1e-6)

    return weights.clamp(0.5, 1.5)


def predicted_noise_from_output(
    prediction: torch.Tensor,
    *,
    prediction_type: str,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    sqrt_alpha_t: torch.Tensor,
    sqrt_one_minus_alpha_t: torch.Tensor,
) -> torch.Tensor:
    """Convert model outputs into epsilon-space for variance diagnostics."""

    if prediction_type == "eps":
        return prediction
    if prediction_type == "x0":
        return (noisy - sqrt_alpha_t * prediction) / (sqrt_one_minus_alpha_t + 1e-6)
    if prediction_type == "v":
        return (prediction + sqrt_one_minus_alpha_t * clean) / (sqrt_alpha_t + 1e-6)
    raise ValueError(f"Unsupported prediction_type '{prediction_type}' for variance diagnostics")


@dataclass
class SNRGovernorUpdate:
    ratio: float
    snr_target: float
    metrics: Dict[str, float]
    log_message: Optional[str]


class AdaptiveSNRGovernor:
    """Self-regulating adaptive SNR governor."""

    def __init__(
        self,
        *,
        min_ratio: float,
        max_ratio: float,
        initial_ratio: Optional[float] = None,
        overflow_target: float = 0.05,
        loss_beta: float = 0.9,
        metric_beta: float = 0.8,
        lambda_var: float = 7e-4,
        band_report_interval: int = 50,
    ) -> None:
        if min_ratio <= 0.0:
            raise ValueError("min_ratio must be positive")
        if max_ratio <= min_ratio:
            raise ValueError("max_ratio must exceed min_ratio")
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)
        start = float(initial_ratio) if initial_ratio is not None else self.min_ratio
        self._ratio = self._clamp(start)
        self.overflow_target = float(overflow_target)
        self.lambda_var = float(lambda_var)
        self.metrics = AdaptiveRegulatorMetrics(snr_target=self._ratio, lambda_var=self.lambda_var)
        self._loss_filter = _EWMA(loss_beta)
        self._metric_filter = _EWMA(metric_beta)
        self._overflow_ema = 0.0
        self._kappa_ema = _EWMA(metric_beta)
        self._variance_ema = _EWMA(metric_beta)
        self._hard_ema = _EWMA(metric_beta)
        self._band_tracker = _BandTracker()
        self._micro_reset = MicroResetPolicy()
        self._step = 0
        self._prev_loss: Optional[float] = None
        self._noise_low = 0
        self._noise_high = 0
        self._band_log_interval = max(1, int(band_report_interval))
        self._lambda_logged = False

    def _clamp(self, value: float) -> float:
        return float(min(max(value, self.min_ratio), self.max_ratio))

    @property
    def ratio(self) -> float:
        return self._ratio

    def _band_fractions(self, snr: torch.Tensor) -> Tuple[float, float, float]:
        if snr.numel() == 0:
            return 0.0, 0.0, 0.0
        snr_cpu = snr.detach().float()
        hard = float(((snr_cpu >= 0.4) & (snr_cpu < 0.8)).float().mean().item())
        medium = float(((snr_cpu >= 0.8) & (snr_cpu < 1.4)).float().mean().item())
        easy = float(((snr_cpu >= 1.4) & (snr_cpu <= 2.4)).float().mean().item())
        return hard, medium, easy

    def update(
        self,
        *,
        loss: float,
        grad_norm: float,
        snr_raw: torch.Tensor,
        snr_clamped: torch.Tensor,
        adaptive_diag: Optional[Mapping[str, float]],
        predicted_noise: torch.Tensor,
        true_noise: torch.Tensor,
        std_ratio: Optional[float] = None,
    ) -> SNRGovernorUpdate:
        loss = float(loss)
        grad_norm = float(grad_norm)
        self._step += 1

        self._loss_filter.update(loss)
        delta_loss = 0.0
        if self._prev_loss is not None:
            delta_loss = float(self._prev_loss - loss)
        self._prev_loss = loss

        diag = dict(adaptive_diag) if adaptive_diag else {}

        diag_kappa = diag.get("kappa")
        diag_overflow = diag.get("overflow")
        diag_overflow_ema = diag.get("overflow_ema")
        diag_ema = diag.get("ema")

        alpha_fac = compute_alpha_fac(diag_kappa, diag_ema)

        overflow_ratio = 0.0
        snr_mean = 0.0
        snr_max = 0.0
        if snr_raw.numel() > 0:
            overflow_ratio = float((snr_raw > self.max_ratio).float().mean().item())
            snr_mean = float(snr_clamped.detach().float().mean().item())
            snr_max = float(snr_clamped.detach().float().max().item())

        self._overflow_ema = blend_overflow_ema(
            self._overflow_ema,
            overflow_ratio,
            diag_overflow=diag_overflow,
            diag_overflow_ema=diag_overflow_ema,
        )

        kappa_val = self._kappa_ema.update(_to_float(diag_kappa, 0.0))
        variance_ratio_raw = 1.0
        pred_std_val = 0.0
        true_std_val = 0.0
        if predicted_noise.numel() and true_noise.numel():
            pred_centered = predicted_noise.detach() - predicted_noise.detach().mean()
            true_centered = true_noise.detach() - true_noise.detach().mean()
            pred_std_val = float(pred_centered.std(unbiased=False).item())
            true_std_val = float(true_centered.std(unbiased=False).item())
            if true_std_val > 0:
                variance_ratio_raw = pred_std_val / max(true_std_val, 1e-6)
        variance_ratio = self._variance_ema.update(variance_ratio_raw)

        hard_frac, med_frac, easy_frac = self._band_fractions(snr_clamped)
        hard_fraction = self._hard_ema.update(hard_frac)
        self._band_tracker.register(hard_frac, med_frac, easy_frac)
        overflow_gap = max(0.0, self._overflow_ema - self.overflow_target)
        band_probs = self._band_tracker.maybe_update(overflow_gap)
        r_feedback_note: Optional[str] = None
        if variance_ratio_raw < 0.9:
            updated = min(
                self._band_tracker.hard_bounds[1], band_probs["hard"] + 0.05
            )
            if updated > band_probs["hard"] + 1e-6:
                band_probs["hard"] = updated
                r_feedback_note = (
                    f"[SNR-GOV] r-feedback step={self._step} action=harder "
                    f"value={variance_ratio_raw:.3f}"
                )
        elif variance_ratio_raw > 1.1:
            updated = max(
                self._band_tracker.hard_bounds[0], band_probs["hard"] - 0.05
            )
            if updated < band_probs["hard"] - 1e-6:
                band_probs["hard"] = updated
                r_feedback_note = (
                    f"[SNR-GOV] r-feedback step={self._step} action=easier "
                    f"value={variance_ratio_raw:.3f}"
                )

        band_probs["easy"] = float(
            min(max(band_probs["easy"], self._band_tracker.easy_bounds[0]), self._band_tracker.easy_bounds[1])
        )
        band_probs["medium"] = max(0.0, 1.0 - band_probs["hard"] - band_probs["easy"])
        self._band_tracker.set_probs(band_probs)
        band_probs = dict(self._band_tracker.probs)

        snr_target = self._ratio
        snr_target *= 1.0 + 0.3 * (self._overflow_ema - 0.02)
        if std_ratio is not None and std_ratio < 0.8:
            snr_target *= 1.1
        if overflow_gap > 0.0:
            snr_target *= max(0.7, 1.0 - 0.25 * overflow_gap)
        snr_target = self._clamp(snr_target)
        if not snr_target > 0:
            raise AssertionError("Invalid SNR target")

        ratio = snr_target

        micro_reset = self._micro_reset.apply(self.metrics, self._step)
        if micro_reset:
            ratio = self._clamp(ratio)

        noise_std = 0.0
        if true_noise.numel():
            noise_std = float(true_noise.detach().std().item())

        if noise_std < 0.28:
            self._noise_low = getattr(self, "_noise_low", 0) + 1
        else:
            self._noise_low = 0
        if noise_std > 0.35:
            self._noise_high = getattr(self, "_noise_high", 0) + 1
        else:
            self._noise_high = 0

        hysteresis = 3
        if getattr(self, "_noise_low", 0) >= hysteresis:
            snr_target = self._clamp(snr_target * 0.95)
            self._noise_low = 0
        elif getattr(self, "_noise_high", 0) >= hysteresis:
            snr_target = self._clamp(snr_target * 1.05)
            self._noise_high = 0
        ratio = snr_target

        self.metrics.kappa = kappa_val
        self.metrics.ema = _to_float(diag_ema, self.metrics.ema)
        self.metrics.overflow = overflow_ratio
        self.metrics.overflow_ema = self._overflow_ema
        self.metrics.alpha_fac = alpha_fac
        self.metrics.snr_target = snr_target
        self.metrics.variance_ratio = variance_ratio
        self.metrics.hard_fraction = hard_fraction
        self.metrics.delta_loss = delta_loss
        self.metrics.band_hard = band_probs["hard"]
        self.metrics.band_medium = band_probs["medium"]
        self.metrics.band_easy = band_probs["easy"]
        self.metrics.lambda_var = self.lambda_var

        micro_reset = self._micro_reset.apply(self.metrics, self._step)
        if micro_reset:
            ratio = self._clamp(ratio)

        self.metrics.micro_reset = 1.0 if micro_reset else 0.0

        notes: List[str] = [
            (
                "[SNR-GOV] step={step} ratio={ratio:.3f} target={target:.3f} "
                "overflow={overflow:.3f} hard={hard:.3f} r={ratio_var:.3f}"
            ).format(
                step=self._step,
                ratio=ratio,
                target=snr_target,
                overflow=self._overflow_ema,
                hard=hard_fraction,
                ratio_var=variance_ratio,
            )
        ]
        if not self._lambda_logged:
            notes.append(f"[SNR-GOV] lambda_var={self.lambda_var:.3e}")
            self._lambda_logged = True
        if r_feedback_note:
            notes.append(r_feedback_note)
        if self._step % self._band_log_interval == 0:
            notes.append(
                (
                    "[SNR-GOV] bands hard={hard:.2f} med={med:.2f} easy={easy:.2f}"
                ).format(
                    hard=band_probs["hard"],
                    med=band_probs["medium"],
                    easy=band_probs["easy"],
                )
            )

        self._ratio = ratio

        metrics = self.metrics.as_dict()
        metrics.update(
            {
                "snr_ratio": ratio,
                "overflow_ratio": overflow_ratio,
                "delta_loss": delta_loss,
                "snr_mean": snr_mean,
                "snr_max": snr_max,
                "snr_headroom": max(self.max_ratio - snr_max, 0.0),
                "noise_rms": noise_std,
                "variance_ratio_raw": variance_ratio_raw,
                "pred_noise_std": pred_std_val,
                "true_noise_std": true_std_val,
            }
        )
        log_message = "\n".join(notes)
        return SNRGovernorUpdate(ratio=ratio, snr_target=snr_target, metrics=metrics, log_message=log_message)

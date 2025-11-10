"""Adaptive weighting helpers for diffusion training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch

from src.training.regulators import MicroResetPolicy

EPS = 1e-8


def _default_log_fn(message: str, diag: Dict[str, float]) -> None:
    print(message)


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
        delta_growth: float = 1.0,
        overflow_target: float = 0.01,
        overflow_decay: float = 0.9,
        change_threshold: float = 0.1,
        log_interval: int = 200,
        snr_clip: Optional[float] = None,
        log_fn: Optional[Callable[[str, Dict[str, float]], None]] = _default_log_fn,
    ) -> None:
        self.beta = float(beta)
        self.ema_decay = float(ema_decay)
        self.gamma_alpha = float(gamma_alpha)
        self.kappa_floor = float(kappa_floor)
        self.freeze_ratio = float(freeze_ratio)
        self._delta_base = float(delta)
        self.delta = float(delta)
        self.delta_growth = float(delta_growth)
        self.overflow_target = float(overflow_target)
        self.overflow_decay = float(overflow_decay)
        self.change_threshold = float(change_threshold)
        self.log_interval = int(log_interval)
        self.snr_clip = float(snr_clip) if snr_clip is not None else None

        self._ema_val: Optional[torch.Tensor] = None
        self._overflow_ema: float = 0.0
        self._overflow_actual_ema: float = 0.0
        self._step = 0
        self._last_log_step = 0
        self._last_diag: Dict[str, float] = {}
        self._log_fn = log_fn
        self._micro_reset = MicroResetPolicy()

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

    def set_log_fn(self, log_fn: Optional[Callable[[str, Dict[str, float]], None]]) -> None:
        self._log_fn = log_fn

    # ------------------------------------------------------------------
    def _compute_weight_core(
        self,
        snr: torch.Tensor,
        raw_loss: torch.Tensor,
        alpha_t: torch.Tensor,
        *,
        kappa_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, float, float, float, float]:
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

        per_example_value = (soft_weight * norm_loss).clamp_min(EPS)
        value = per_example_value.mean()
        if self._ema_val is None or torch.isnan(self._ema_val):
            self._ema_val = value.detach()
        else:
            self._ema_val = (
                self._ema_val * self.ema_decay + value.detach() * (1.0 - self.ema_decay)
            )
        self._ema_val = self._ema_val.to(dtype=torch.float32)

        alpha_gap = (1.0 - alpha_detached.clamp(0.0, 1.0)).mean()
        alpha_term = 1.0 + self.gamma_alpha * float(alpha_gap.item())

        overflow_adj = 1.0
        if self.overflow_target > 0:
            excess = max(0.0, self._overflow_actual_ema - self.overflow_target) / self.overflow_target
            overflow_adj += excess * max(self.delta_growth, 0.0)

        delta_eff = self._delta_base * overflow_adj
        self.delta = delta_eff

        kappa_per_example = torch.clamp(
            torch.abs(per_example_value) * self.beta * alpha_term,
            min=self.kappa_floor,
        )
        kappa_per_example = kappa_per_example + 0.05 * soft_weight + delta_eff
        if kappa_scale != 1.0:
            kappa_per_example = kappa_per_example * float(kappa_scale)

        expanded_soft = _expand_to(soft_weight, loss_detached).to(raw_loss.dtype)
        kappa_expanded = _expand_to(kappa_per_example, expanded_soft).to(raw_loss.dtype)
        weight = expanded_soft / (expanded_soft + kappa_expanded + EPS)

        kappa_mean = float(kappa_per_example.mean().item())

        return weight, kappa_mean, float(self._ema_val.item()), alpha_term, float(delta_eff)

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

        snr_in = snr
        clip_mask: Optional[torch.Tensor] = None
        if self.snr_clip is not None:
            clip_mask = (snr_in > self.snr_clip).to(snr_in.dtype)
            snr_in = snr_in.clamp(min=0.0, max=self.snr_clip)
        overflow_tensor: Optional[torch.Tensor] = None
        if overflow_mask is not None:
            overflow_tensor = overflow_mask.detach().float()
        ratio_mask: Optional[torch.Tensor] = None
        if clip_mask is not None and overflow_tensor is not None:
            ratio_mask = torch.maximum(
                _expand_to(clip_mask, snr_in), _expand_to(overflow_tensor, snr_in)
            )
        elif clip_mask is not None:
            ratio_mask = _expand_to(clip_mask, snr_in)
        elif overflow_tensor is not None:
            ratio_mask = _expand_to(overflow_tensor, snr_in)

        next_step = self._step + 1
        kappa_scale, overflow_scale, micro_reset = self._micro_reset.factors(next_step)
        weight, kappa_value, ema_value, alpha_fac, delta_eff = self._compute_weight_core(
            snr_in, raw_loss, alpha_t, kappa_scale=kappa_scale
        )

        combined_ratio = 0.0
        if ratio_mask is not None:
            combined_ratio = float(ratio_mask.detach().float().mean().item())

        actual_ratio = 0.0
        if overflow_tensor is not None:
            overflow_expanded = _expand_to(overflow_tensor, weight)
            actual_ratio = float(overflow_expanded.detach().float().mean().item())
            if torch.any(overflow_expanded > 0):
                weight = torch.where(overflow_expanded > 0, torch.zeros_like(weight), weight)

        self._overflow_actual_ema = (
            self.overflow_decay * self._overflow_actual_ema
            + (1.0 - self.overflow_decay) * actual_ratio
        )
        self._overflow_ema = (
            self.overflow_decay * self._overflow_ema
            + (1.0 - self.overflow_decay) * combined_ratio
        )

        if micro_reset:
            self._overflow_ema *= overflow_scale

        mean_weight = float(weight.detach().mean().item())
        max_weight = float(weight.detach().max().item())

        clip_ratio = float(clip_mask.mean().item()) if clip_mask is not None else 0.0
        diag = {
            "mean_weight": mean_weight,
            "max_weight": max_weight,
            "kappa": kappa_value,
            "ema": ema_value,
            "alpha_fac": alpha_fac,
            "overflow": combined_ratio,
            "overflow_actual": actual_ratio,
            "overflow_ema": self._overflow_ema,
            "overflow_actual_ema": self._overflow_actual_ema,
            "delta": delta_eff,
            "clip_overflow": clip_ratio,
            "micro_reset": 1.0 if micro_reset else 0.0,
        }
        if micro_reset:
            diag["micro_reset"] = 1.0
        if diag_extra:
            for key, value in diag_extra.items():
                if key not in diag:
                    diag[key] = value

        self._step += 1
        diag["step"] = self._step
        should_log = self._should_log(diag)
        diag["log_event"] = should_log
        if should_log:
            self._last_log_step = self._step
            self._last_diag = {key: diag[key] for key in ("kappa", "ema", "mean_weight")}
            message = (
                "[AdaptiveSNR] step="
                f"{self._step:04d} κ={diag['kappa']:.3e} ema={diag['ema']:.3e} "
                f"α_fac={diag['alpha_fac']:.2f} overflow={diag['overflow']:.3f} "
                f"overflow_ema={diag['overflow_ema']:.3f}"
            )
            if self._log_fn is not None:
                self._log_fn(message, diag)

        return weight, diag


# Backwards compatibility ----------------------------------------------------

AdaptiveSNRv14 = AdaptiveSNRWeight


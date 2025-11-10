from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.core.diffusion_step import describe_regime, select_regime
from src.core.functional import compute_snr_weight, compute_target
from src.core.numeric import safe_clamp
from src.core.fft_feedback import compute_fft_feedback, fft_high_mean
from src.core.overflow_handler import OverflowHandler
from src.core.snr_scheduler import compute_snr_stats, measure_batch_snr
from src.training.noise import NoiseBatch


ALPHA_MIN = 0.01
ALPHA_MAX = 0.999
SIGMA_MIN = 1e-4
SNR_CLIP = 250.0
PRED_STD_WARN_FACTOR = 5.0
SPECTRAL_WARN_THRESHOLD = 0.8

@dataclass
class StepOutcome:
    """Scalar metrics emitted by a single training step."""

    loss: float
    mae: float
    grad_norm: Optional[float]
    fft_feedback: Dict[str, float]
    coeff_stats: Dict[str, float]
    batch_stats: Dict[str, float]
    weight_stats: Optional[Dict[str, float]] = None
    residual_mode: Optional[str] = None


class TrainingStepExecutor:
    """Encapsulate the forward/backward pass for a training iteration."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[..., Tensor],
        *,
        prediction_type: str,
        snr_weighting: Optional[bool],
        snr_transform: str,
        fft_norm: str,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.prediction_type = prediction_type
        default_weighting = getattr(self.loss_fn, "use_weighting", True)
        if hasattr(self.loss_fn, "set_weighting_enabled"):
            enabled = default_weighting if snr_weighting is None else bool(snr_weighting)
            self.loss_fn.set_weighting_enabled(enabled)
            self.snr_weighting = enabled
        else:
            self.snr_weighting = bool(snr_weighting) if snr_weighting is not None else default_weighting
        self.snr_transform = snr_transform
        self.fft_norm = fft_norm
        self.snr_clip = getattr(self.loss_fn, "snr_clip", SNR_CLIP)
        self._overflow_decay = 0.9
        self._overflow_ema = 0.0
        self.overflow_handler = OverflowHandler(snr_clip=self.snr_clip, ema_decay=self._overflow_decay)

    def run_step(
        self,
        clean_batch: Tensor,
        noise_batch: NoiseBatch,
        timesteps: Tensor,
        grad_callback: Optional[Callable[[], Optional[float]]] = None,
    ) -> StepOutcome:
        sqrt_alpha_t = safe_clamp(
            noise_batch.sqrt_alpha_t,
            min_value=ALPHA_MIN,
            max_value=ALPHA_MAX,
        )
        sqrt_one_minus_alpha_t = safe_clamp(
            noise_batch.sqrt_one_minus_alpha_t, min_value=SIGMA_MIN
        )

        prediction = self.model(noise_batch.noisy, timesteps)
        target = compute_target(
            self.prediction_type,
            clean_batch,
            noise_batch.noisy,
            noise_batch.eps,
            sqrt_alpha_t,
            sqrt_one_minus_alpha_t,
        )
        snr_stats = compute_snr_stats(
            sqrt_alpha_t,
            sqrt_one_minus_alpha_t,
            snr_clip=self.snr_clip,
            min_sigma=SIGMA_MIN**2,
        )
        snr_raw = snr_stats.snr_raw
        regimes = select_regime(snr_raw, self.snr_clip)
        overflow_mask = regimes["overflow"]
        prediction = self.overflow_handler.renormalise(prediction, overflow_mask)

        residual = prediction - target

        loss_diag: Optional[Dict[str, Any]] = None
        fallback_weight: Optional[Tensor] = None
        try:
            loss_result = self.loss_fn(
                prediction,
                target,
                sqrt_alpha_t,
                sqrt_one_minus_alpha_t,
                x_t=noise_batch.noisy,
                x0=clean_batch,
                prediction_type=self.prediction_type,
            )
        except TypeError:
            if self.snr_weighting:
                fallback_weight = compute_snr_weight(
                    sqrt_alpha_t,
                    sqrt_one_minus_alpha_t,
                    transform=self.snr_transform,
                )
            loss = self.loss_fn(residual, fallback_weight)
        else:
            if isinstance(loss_result, tuple):
                loss, loss_diag = loss_result
            else:
                loss = loss_result

        if loss_diag and "mae" in loss_diag:
            mae = torch.tensor(loss_diag["mae"], device=loss.device)
        else:
            mae = F.l1_loss(prediction, target)

        weight_stats: Optional[Dict[str, float]] = None
        if loss_diag:
            weight_stats = {
                key: float(value)
                for key, value in loss_diag.items()
                if isinstance(value, (int, float))
            }
        elif fallback_weight is not None:
            weight = fallback_weight
            weight_stats = {
                "min": float(weight.min().item()),
                "max": float(weight.max().item()),
                "mean": float(weight.mean().item()),
            }
        elif self.snr_weighting:
            weight = compute_snr_weight(
                sqrt_alpha_t,
                sqrt_one_minus_alpha_t,
                transform=self.snr_transform,
            )
            weight_stats = {
                "min": float(weight.min().item()),
                "max": float(weight.max().item()),
                "mean": float(weight.mean().item()),
            }

        fft_feedback = compute_fft_feedback(
            prediction,
            target,
            fft_norm=self.fft_norm,
        )
        snr = snr_stats.snr_clamped
        overflow_stats = self.overflow_handler.update(overflow_mask)
        self.overflow_handler.log(snr_stats.snr_clamped, regimes)

        batch_snr = measure_batch_snr(clean_batch, noise_batch.noisy, sqrt_alpha_t)

        coeff_stats = {
            "timestep_min": float(timesteps.min().item()),
            "timestep_max": float(timesteps.max().item()),
            "timestep_mean": float(timesteps.float().mean().item()),
            "sqrt_alpha_min": float(sqrt_alpha_t.min().item()),
            "sqrt_alpha_max": float(sqrt_alpha_t.max().item()),
            "sqrt_alpha_mean": float(sqrt_alpha_t.mean().item()),
            "sqrt_one_minus_min": float(sqrt_one_minus_alpha_t.min().item()),
            "sqrt_one_minus_max": float(sqrt_one_minus_alpha_t.max().item()),
            "sqrt_one_minus_mean": float(sqrt_one_minus_alpha_t.mean().item()),
            "snr_min": float(snr.min().item()),
            "snr_max": float(snr.max().item()),
            "snr_mean": float(snr.mean().item()),
            "snr_raw_max": float(snr_raw.max().item()),
            "overflow_count": float(overflow_stats.count),
            "overflow_ema": float(overflow_stats.ema),
            "signal_rms": float(batch_snr.signal_rms.mean().item()),
            "noise_rms": float(batch_snr.noise_rms.mean().item()),
            "snr_measured": float(batch_snr.snr_measured.mean().item()),
        }

        prediction_mean = float(prediction.detach().mean().item())
        prediction_std = float(prediction.detach().std().item())
        input_std = float(clean_batch.detach().std().item())
        if input_std > 0:
            std_ratio = prediction_std / max(input_std, 1e-8)
            if std_ratio > PRED_STD_WARN_FACTOR:
                print(
                    "[WARN] Prediction std drift: "
                    f"std={prediction_std:.3f}, input_std={input_std:.3f}, "
                    f"ratio={std_ratio:.2f}"
                )
        else:
            std_ratio = float("inf")

        fft_high = fft_high_mean(prediction.detach())
        if not math.isnan(fft_high) and fft_high > SPECTRAL_WARN_THRESHOLD:
            print(
                "[WARN] Spectral blowup suspected: "
                f"fft_high_mean={fft_high:.3f}"
            )

        batch_stats = {
            "prediction_mean": prediction_mean,
            "prediction_std": prediction_std,
            "prediction_abs_max": float(prediction.detach().abs().max().item()),
            "target_mean": float(target.detach().mean().item()),
            "target_std": float(target.detach().std().item()),
            "target_abs_max": float(target.detach().abs().max().item()),
            "residual_mean": float(residual.detach().mean().item()),
            "residual_std": float(residual.detach().std().item()),
            "residual_abs_max": float(residual.detach().abs().max().item()),
            "residual_mse": float(residual.detach().pow(2).mean().item()),
            "input_std": input_std,
            "prediction_std_ratio": std_ratio,
            "prediction_fft_high": fft_high,
        }

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = grad_callback() if grad_callback else None
        self.optimizer.step()

        residual_mode = getattr(self.loss_fn, "mode", None)

        return StepOutcome(
            loss=float(loss.detach().cpu()),
            mae=float(mae.detach().cpu()),
            grad_norm=grad_norm,
            fft_feedback=fft_feedback,
            coeff_stats=coeff_stats,
            batch_stats=batch_stats,
            weight_stats=weight_stats,
            residual_mode=residual_mode,
        )

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
from src.core.fft_feedback import compute_fft_feedback
from src.core.overflow_handler import OverflowHandler
from src.core.snr_scheduler import compute_snr_stats, measure_batch_snr
from src.training.noise import NoiseBatch
from src.utils.adaptive_snr import predicted_noise_from_output
from src.utils.debug_helpers import fft_band_means


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
    per_example_mse: Optional[torch.Tensor] = None


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
        lambda_var: float = 7e-4,
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
        enable_overflow_renorm = bool(getattr(self.loss_fn, "overflow_renorm", False))
        self._overflow_decay = 0.9
        self._overflow_ema = 0.0
        self.overflow_handler = OverflowHandler(
            snr_clip=self.snr_clip,
            ema_decay=self._overflow_decay,
            enable_renorm=enable_overflow_renorm,
        )
        self.lambda_var = float(lambda_var)

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
        predicted_noise = predicted_noise_from_output(
            prediction,
            prediction_type=self.prediction_type,
            clean=clean_batch,
            noisy=noise_batch.noisy,
            sqrt_alpha_t=sqrt_alpha_t,
            sqrt_one_minus_alpha_t=sqrt_one_minus_alpha_t,
        )
        true_noise = noise_batch.eps
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
        B = residual.shape[0]
        per_example_mse = (
            residual.detach().view(B, -1).pow(2).mean(dim=1).to(device=residual.device)
        )

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

        if torch.isnan(loss.detach()).any():
            raise AssertionError("NaN loss detected")

        if loss_diag and "mae" in loss_diag:
            mae = torch.tensor(loss_diag["mae"], device=loss.device)
        else:
            mae = F.l1_loss(prediction, target)

        per_sample_loss_tensor: Optional[Tensor] = None
        if loss_diag:
            candidate = loss_diag.get("per_sample_loss")
            if isinstance(candidate, torch.Tensor):
                per_sample_loss_tensor = candidate
        if per_sample_loss_tensor is None:
            per_sample_loss_tensor = residual.view(B, -1).pow(2).mean(dim=1)

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

        weight_stats = weight_stats or {}
        fft_feedback = compute_fft_feedback(
            prediction,
            target,
            fft_norm=self.fft_norm,
        )
        snr = snr_stats.snr_clamped
        overflow_stats = self.overflow_handler.update(overflow_mask)
        self.overflow_handler.log(snr_stats.snr_clamped, regimes)

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
            "signal_rms": float(measure_batch_snr(clean_batch, noise_batch.noisy, sqrt_alpha_t).signal_rms.mean().item()),
            "noise_rms": float(measure_batch_snr(clean_batch, noise_batch.noisy, sqrt_alpha_t).noise_rms.mean().item()),
            "snr_measured": float(measure_batch_snr(clean_batch, noise_batch.noisy, sqrt_alpha_t).snr_measured.mean().item()),
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

        band_means = fft_band_means(prediction.detach())
        fft_low = band_means.get("fft_low", float("nan"))
        fft_mid = band_means.get("fft_mid", float("nan"))
        fft_high = band_means.get("fft_high", float("nan"))
        spectral_term = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        if not math.isnan(fft_low) and not math.isnan(fft_high):
            low_tensor = torch.tensor(fft_low, device=loss.device, dtype=loss.dtype)
            high_tensor = torch.tensor(fft_high, device=loss.device, dtype=loss.dtype)
            spectral_pressure = ((high_tensor / (low_tensor + 1e-6)) - 1.0).abs()
            spectral_term = 0.05 * spectral_pressure
            fft_feedback["spectral_pressure"] = float(spectral_pressure.detach().cpu())
        else:
            spectral_pressure = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        fft_feedback.setdefault(
            "spectral_pressure", float(spectral_pressure.detach().cpu())
        )

        lambda_var_tensor = loss.new_tensor(self.lambda_var)
        variance_penalty = loss.new_tensor(0.0)
        variance_ratio = float("nan")
        pred_std_value = float("nan")
        true_std_value = float("nan")
        if predicted_noise.numel() and true_noise.numel():
            pred_centered = predicted_noise - predicted_noise.mean()
            true_centered = true_noise - true_noise.mean()
            std_pred = torch.sqrt(torch.mean(pred_centered.pow(2)))
            std_true = torch.sqrt(torch.mean(true_centered.pow(2)))
            variance_penalty = lambda_var_tensor * (std_pred - std_true.detach()) ** 2
            pred_std_detached = std_pred.detach()
            true_std_detached = std_true.detach().clamp(min=1e-6)
            pred_std_value = float(pred_std_detached.item())
            true_std_value = float(true_std_detached.item())
            variance_ratio = float((pred_std_detached / true_std_detached).item())

        if per_sample_loss_tensor is not None and per_sample_loss_tensor.shape[0] == B:
            loss = (per_sample_loss_tensor + variance_penalty).mean()
        else:
            loss = loss + variance_penalty

        loss = loss + spectral_term

        variance_penalty_value = float(variance_penalty.detach().item())
        fft_feedback["variance_ratio"] = variance_ratio
        fft_feedback["variance_penalty"] = variance_penalty_value
        if not math.isnan(fft_high) and fft_high > SPECTRAL_WARN_THRESHOLD:
            print(
                "[WARN] Spectral blowup suspected: "
                f"fft_high_mean={fft_high:.3f}"
            )

        fft_ratio = float("nan")
        if not math.isnan(fft_low) and fft_low > 0.0 and not math.isnan(fft_high):
            fft_ratio = float(fft_high / (fft_low + 1e-6))

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
            "prediction_fft_low": fft_low,
            "prediction_fft_mid": fft_mid,
            "prediction_fft_high": fft_high,
            "prediction_fft_ratio": fft_ratio,
            "variance_ratio": variance_ratio,
            "variance_penalty": variance_penalty_value,
            "pred_noise_std_centered": pred_std_value,
            "true_noise_std_centered": true_std_value,
            "spectral_pressure": float(spectral_pressure.detach().cpu()),
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
            per_example_mse=per_example_mse.detach().cpu(),
        )

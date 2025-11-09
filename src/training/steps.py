from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.core.functional import compute_snr_weight, compute_target
from src.core.numeric import safe_clamp, safe_ratio
from src.training.noise import NoiseBatch


ALPHA_MIN = 0.01
ALPHA_MAX = 0.999
SIGMA_MIN = 1e-4
SNR_CLIP = 250.0
PRED_STD_WARN_FACTOR = 5.0
SPECTRAL_WARN_THRESHOLD = 0.8

def _radial_frequency_masks(
    height: int, width: int, device: torch.device, boundaries: tuple[float, ...]
) -> Dict[str, torch.Tensor]:
    """Return radial frequency masks for low/mid/high bands."""

    if len(boundaries) != 4:
        raise ValueError("Expected boundaries to describe three bands")

    fy = torch.fft.fftfreq(height, device=device)
    fx = torch.fft.fftfreq(width, device=device)
    yy = fy[:, None]
    xx = fx[None, :]
    radius = torch.sqrt(xx**2 + yy**2)

    labels = ("low", "mid", "high")
    masks: Dict[str, torch.Tensor] = {}
    for label, lower, upper in zip(labels, boundaries[:-1], boundaries[1:]):
        mask = (radius >= lower) & (radius < upper)
        if torch.any(mask):
            masks[label] = mask
    return masks


def compute_fft_feedback(
    prediction: Tensor, target: Tensor, *, fft_norm: str = "ortho"
) -> Dict[str, float]:
    """Return amplitude/phase residual statistics between prediction and target."""

    pred_fft = torch.fft.fftn(prediction.detach(), dim=(-2, -1), norm=fft_norm)
    target_fft = torch.fft.fftn(target.detach(), dim=(-2, -1), norm=fft_norm)

    amplitude_delta = pred_fft.abs() - target_fft.abs()
    amplitude_error = amplitude_delta.abs()
    amplitude_mae = float(amplitude_error.mean().item())

    phase_pred = torch.angle(pred_fft)
    phase_target = torch.angle(target_fft)
    phase_delta = torch.atan2(
        torch.sin(phase_pred - phase_target),
        torch.cos(phase_pred - phase_target),
    )
    phase_error = phase_delta.abs()
    phase_mae = float(phase_error.mean().item())

    real_mae = float((pred_fft.real - target_fft.real).abs().mean().item())
    imag_mae = float((pred_fft.imag - target_fft.imag).abs().mean().item())
    complex_delta = pred_fft - target_fft
    complex_error = complex_delta.abs()
    complex_mae = float(complex_error.mean().item())

    height, width = prediction.shape[-2:]
    boundaries = (0.0, 0.12, 0.28, float("inf"))
    masks = _radial_frequency_masks(height, width, prediction.device, boundaries)

    band_metrics: Dict[str, float] = {}
    for label, mask in masks.items():
        amplitude_band = amplitude_error[..., mask]
        phase_band = phase_error[..., mask]
        complex_band = complex_error[..., mask]
        if amplitude_band.numel() > 0:
            band_metrics[f"amplitude_{label}_mae"] = float(amplitude_band.mean().item())
        if phase_band.numel() > 0:
            band_metrics[f"phase_{label}_mae"] = float(phase_band.mean().item())
        if complex_band.numel() > 0:
            band_metrics[f"complex_{label}_mae"] = float(complex_band.mean().item())

    dc_error = amplitude_error[..., 0, 0]
    band_metrics["amplitude_dc_mae"] = float(dc_error.mean().item())

    return {
        "amplitude_mae": amplitude_mae,
        "phase_mae": phase_mae,
        "real_mae": real_mae,
        "imag_mae": imag_mae,
        "complex_mae": complex_mae,
        **band_metrics,
    }


def _fft_high_mean(tensor: Tensor) -> float:
    """Return the mean magnitude of the high-frequency FFT band."""

    if tensor.ndim < 3:
        return float("nan")
    complex_tensor = (
        tensor
        if tensor.is_complex()
        else torch.complex(tensor, torch.zeros_like(tensor))
    )
    fft = torch.fft.fftn(complex_tensor, dim=(-2, -1), norm="ortho")
    fft = torch.fft.fftshift(fft, dim=(-2, -1))
    magnitude = fft.abs()
    height, width = magnitude.shape[-2:]
    fy = torch.fft.fftfreq(height, device=tensor.device)
    fx = torch.fft.fftfreq(width, device=tensor.device)
    yy = fy[:, None]
    xx = fx[None, :]
    radius = torch.sqrt(xx**2 + yy**2)
    mask_high = radius >= 0.25
    if not torch.any(mask_high):
        return float("nan")
    return float(magnitude[..., mask_high].mean().item())


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
        residual = prediction - target

        loss_diag: Optional[Dict[str, Any]] = None
        fallback_weight: Optional[Tensor] = None
        try:
            loss_result = self.loss_fn(
                prediction,
                target,
                sqrt_alpha_t,
                sqrt_one_minus_alpha_t,
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

        snr_raw = safe_ratio(
            sqrt_alpha_t**2,
            sqrt_one_minus_alpha_t**2,
            min_den=SIGMA_MIN**2,
        )
        snr = safe_clamp(snr_raw, max_value=SNR_CLIP)
        if torch.any(snr_raw > SNR_CLIP):
            overflow = float(snr_raw.max().item())
            count = int(torch.count_nonzero(snr_raw > SNR_CLIP).item())
            print(
                f"[WARN] SNR overflow detected: max={overflow:.1f} (>{SNR_CLIP}), "
                f"count={count}"
            )
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

        fft_high = _fft_high_mean(prediction.detach())
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

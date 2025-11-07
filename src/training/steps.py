from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.core.functional import compute_snr_weight, compute_target
from src.training.noise import NoiseBatch


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


class TrainingStepExecutor:
    """Encapsulate the forward/backward pass for a training iteration."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[Tensor, Optional[Tensor]], Tensor],
        *,
        prediction_type: str,
        snr_weighting: bool,
        snr_transform: str,
        fft_norm: str,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.prediction_type = prediction_type
        self.snr_weighting = snr_weighting
        self.snr_transform = snr_transform
        self.fft_norm = fft_norm

    def run_step(
        self,
        clean_batch: Tensor,
        noise_batch: NoiseBatch,
        timesteps: Tensor,
        grad_callback: Optional[Callable[[], Optional[float]]] = None,
    ) -> StepOutcome:
        prediction = self.model(noise_batch.noisy, timesteps)
        target = compute_target(
            self.prediction_type,
            clean_batch,
            noise_batch.noisy,
            noise_batch.eps,
            noise_batch.sqrt_alpha_t,
            noise_batch.sqrt_one_minus_alpha_t,
        )
        residual = prediction - target
        weight: Optional[Tensor] = None
        if self.snr_weighting:
            weight = compute_snr_weight(
                noise_batch.sqrt_alpha_t,
                noise_batch.sqrt_one_minus_alpha_t,
                transform=self.snr_transform,
            )
        loss = self.loss_fn(residual, weight)
        mae = F.l1_loss(prediction, target)

        weight_stats: Optional[Dict[str, float]] = None
        if weight is not None:
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

        snr = (noise_batch.sqrt_alpha_t**2) / (
            noise_batch.sqrt_one_minus_alpha_t**2 + 1e-8
        )
        coeff_stats = {
            "timestep_min": float(timesteps.min().item()),
            "timestep_max": float(timesteps.max().item()),
            "timestep_mean": float(timesteps.float().mean().item()),
            "sqrt_alpha_min": float(noise_batch.sqrt_alpha_t.min().item()),
            "sqrt_alpha_max": float(noise_batch.sqrt_alpha_t.max().item()),
            "sqrt_alpha_mean": float(noise_batch.sqrt_alpha_t.mean().item()),
            "sqrt_one_minus_min": float(
                noise_batch.sqrt_one_minus_alpha_t.min().item()
            ),
            "sqrt_one_minus_max": float(
                noise_batch.sqrt_one_minus_alpha_t.max().item()
            ),
            "sqrt_one_minus_mean": float(
                noise_batch.sqrt_one_minus_alpha_t.mean().item()
            ),
            "snr_min": float(snr.min().item()),
            "snr_max": float(snr.max().item()),
            "snr_mean": float(snr.mean().item()),
        }

        batch_stats = {
            "prediction_mean": float(prediction.detach().mean().item()),
            "prediction_std": float(prediction.detach().std().item()),
            "prediction_abs_max": float(prediction.detach().abs().max().item()),
            "target_mean": float(target.detach().mean().item()),
            "target_std": float(target.detach().std().item()),
            "target_abs_max": float(target.detach().abs().max().item()),
            "residual_mean": float(residual.detach().mean().item()),
            "residual_std": float(residual.detach().std().item()),
            "residual_abs_max": float(residual.detach().abs().max().item()),
            "residual_mse": float(residual.detach().pow(2).mean().item()),
        }

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = grad_callback() if grad_callback else None
        self.optimizer.step()

        return StepOutcome(
            loss=float(loss.detach().cpu()),
            mae=float(mae.detach().cpu()),
            grad_norm=grad_norm,
            fft_feedback=fft_feedback,
            coeff_stats=coeff_stats,
            batch_stats=batch_stats,
            weight_stats=weight_stats,
        )

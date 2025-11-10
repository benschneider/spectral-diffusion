"""FFT diagnostics shared between the recorder and training executor."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor


def _radial_frequency_masks(
    height: int, width: int, device: torch.device, boundaries: Tuple[float, ...]
) -> Dict[str, torch.Tensor]:
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


def compute_fft_feedback(prediction: Tensor, target: Tensor, *, fft_norm: str = "ortho") -> Dict[str, float]:
    """Return amplitude/phase residual statistics between prediction and target."""

    pred_fft = torch.fft.fftn(prediction.detach(), dim=(-2, -1), norm=fft_norm)
    target_fft = torch.fft.fftn(target.detach(), dim=(-2, -1), norm=fft_norm)

    amplitude_delta = pred_fft.abs() - target_fft.abs()
    amplitude_error = amplitude_delta.abs()
    phase_pred = torch.angle(pred_fft)
    phase_target = torch.angle(target_fft)
    phase_delta = torch.atan2(
        torch.sin(phase_pred - phase_target),
        torch.cos(phase_pred - phase_target),
    )
    phase_error = phase_delta.abs()

    complex_delta = pred_fft - target_fft
    complex_error = complex_delta.abs()

    metrics = {
        "amplitude_mae": float(amplitude_error.mean().item()),
        "phase_mae": float(phase_error.mean().item()),
        "real_mae": float((pred_fft.real - target_fft.real).abs().mean().item()),
        "imag_mae": float((pred_fft.imag - target_fft.imag).abs().mean().item()),
        "complex_mae": float(complex_error.mean().item()),
    }

    height, width = prediction.shape[-2:]
    boundaries = (0.0, 0.12, 0.28, float("inf"))
    masks = _radial_frequency_masks(height, width, prediction.device, boundaries)
    for label, mask in masks.items():
        amplitude_band = amplitude_error[..., mask]
        phase_band = phase_error[..., mask]
        complex_band = complex_error[..., mask]
        if amplitude_band.numel() > 0:
            metrics[f"amplitude_{label}_mae"] = float(amplitude_band.mean().item())
        if phase_band.numel() > 0:
            metrics[f"phase_{label}_mae"] = float(phase_band.mean().item())
        if complex_band.numel() > 0:
            metrics[f"complex_{label}_mae"] = float(complex_band.mean().item())

    dc_error = amplitude_error[..., 0, 0]
    metrics["amplitude_dc_mae"] = float(dc_error.mean().item())
    return metrics


def fft_high_mean(tensor: Tensor) -> float:
    """Return the mean magnitude of the high-frequency FFT band."""

    if tensor.ndim < 3:
        return float("nan")
    complex_tensor = (
        tensor if tensor.is_complex() else torch.complex(tensor, torch.zeros_like(tensor))
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


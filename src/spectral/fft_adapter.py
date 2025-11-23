from __future__ import annotations

from typing import Optional, Tuple

import torch

from .operator import spectral_operator


def _resolve_scaling(snr_ratio: Optional[float]) -> float:
    if snr_ratio is None:
        return 1.0
    ratio = float(max(snr_ratio, 1e-6))
    return 1.0 / ratio


def add_uniform_frequency_noise(
    x0: torch.Tensor,
    noise: torch.Tensor,
    *,
    sqrt_alpha_t: torch.Tensor,
    sqrt_one_minus_alpha_t: torch.Tensor,
    operator_mode: str = "none",
    mask_params: Optional[dict] = None,
    snr_ratio: Optional[float] = None,
    return_noise: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Inject spectrally-shaped noise into ``x0`` using a single operator."""

    eps_shaped = spectral_operator(noise, mode=operator_mode, mask_params=mask_params)
    scale = _resolve_scaling(snr_ratio)
    eps = eps_shaped * scale
    noise_component = sqrt_one_minus_alpha_t * eps
    noisy = sqrt_alpha_t * x0 + noise_component

    if return_noise:
        return noisy, eps
    return noisy


__all__ = ["add_uniform_frequency_noise"]

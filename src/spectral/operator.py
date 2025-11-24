from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import torch

from .fast_fft import fft2 as _fast_fft2
from .fast_fft import ifft2 as _fast_ifft2


def _per_sample_rms(tensor: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, tensor.ndim)) if tensor.ndim > 1 else ()
    rms = tensor.pow(2).mean(dim=dims, keepdim=True).sqrt()
    return rms.clamp_min(1e-8)


@lru_cache(maxsize=32)
def _radial_mask(shape: Tuple[int, int], power: float) -> torch.Tensor:
    height, width = shape
    fy = torch.fft.fftfreq(height, d=1.0)
    fx = torch.fft.fftfreq(width, d=1.0)
    yy = fy[:, None]
    xx = fx[None, :]
    radius = torch.sqrt(xx**2 + yy**2)
    radius[0, 0] = 1.0
    mask = torch.pow(radius, -power)
    mask[0, 0] = 1.0
    mask = mask / mask.pow(2).mean().sqrt()
    return mask.to(torch.float32)


def _normalize_spatial(noise: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, noise.ndim)) if noise.ndim > 1 else ()
    centered = noise - noise.mean(dim=dims, keepdim=True)
    rms = _per_sample_rms(centered)
    return centered / rms


def spectral_operator(
    eps_raw: torch.Tensor,
    mode: str = "none",
    mask_params: Optional[dict] = None,
) -> torch.Tensor:
    """Apply spectral shaping to Gaussian noise while preserving RMS=1.

    Args:
        eps_raw: Input noise with shape (B, C, H, W).
        mode: "none", "radial", or "radial_squared".
        mask_params: optional dict reserved for future parameterisations.
    Returns:
        Tensor with the same shape as ``eps_raw`` and per-sample RMS=1.
    """

    if eps_raw.ndim < 4:
        raise ValueError("spectral_operator expects noise with shape (B, C, H, W)")

    eps = eps_raw.to(torch.float32)
    mode_normalized = (mode or "none").lower()
    if mode_normalized == "none":
        return _normalize_spatial(eps)

    if mode_normalized not in {"radial", "radial_squared"}:
        raise ValueError(f"Unsupported spectral operator mode: {mode}")

    power = 1.0 if mode_normalized == "radial" else 2.0
    mask = _radial_mask((eps.shape[-2], eps.shape[-1]), power=power)
    mask = mask.to(device=eps.device, dtype=eps.dtype)
    mask = mask.view(1, 1, *mask.shape)

    noise_fft = _fast_fft2(eps, norm="ortho")
    shaped_fft = noise_fft * mask
    shaped = _fast_ifft2(shaped_fft, norm="ortho").real
    return _normalize_spatial(shaped)


__all__ = ["spectral_operator"]

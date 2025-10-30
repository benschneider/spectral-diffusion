from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import torch


@lru_cache(maxsize=32)
def _radial_mask_base(shape: Tuple[int, int]) -> torch.Tensor:
    """
    Pre-compute a reciprocal radial weighting mask for a given spatial size.

    The mask up-weights high frequencies during noise injection so that
    overall signal-to-noise decay remains approximately uniform across bands.
    """
    height, width = shape
    fy = torch.fft.fftfreq(height, d=1.0 / float(height))
    fx = torch.fft.fftfreq(width, d=1.0 / float(width))
    yy = fy[:, None]
    xx = fx[None, :]
    radius = torch.sqrt(xx**2 + yy**2)
    mask = 1.0 / (radius + 1e-4)
    return mask.to(torch.float32)


def add_uniform_frequency_noise(
    x0: torch.Tensor,
    noise: torch.Tensor,
    sqrt_alpha_t: torch.Tensor,
    sqrt_one_minus_alpha_t: torch.Tensor,
    uniform_corruption: bool = False,
) -> torch.Tensor:
    """
    Apply diffusion forward noise with optional uniform frequency corruption.

    When ``uniform_corruption`` is True, the noise is injected in the frequency
    domain with a reciprocal-radius weighting so that higher frequencies receive
    proportionally more energy, balancing SNR decay across the spectrum.
    """
    if not uniform_corruption:
        return sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * noise

    dims = x0.dim()
    if dims < 3:
        raise ValueError("Expected image tensor with at least 3 dimensions (C, H, W).")

    height, width = x0.shape[-2], x0.shape[-1]
    base_mask = _radial_mask_base((height, width)).to(device=x0.device, dtype=x0.dtype)
    mask = base_mask.unsqueeze(0).unsqueeze(0)

    x_fft = torch.fft.fftn(x0, dim=(-2, -1))
    noise_fft = torch.fft.fftn(noise, dim=(-2, -1)) * mask
    x_t_fft = x_fft * sqrt_alpha_t + noise_fft * sqrt_one_minus_alpha_t

    x_t = torch.fft.ifftn(x_t_fft, dim=(-2, -1)).real
    return x_t

from __future__ import annotations

import logging
import math
import random
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F


def _gaussian_kernel(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(int(3.0 * sigma), 1)
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def _apply_gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return image
    kernel_1d = _gaussian_kernel(sigma, image.device, image.dtype)
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.expand(image.shape[-3], 1, -1, -1)
    padding = kernel_2d.shape[-1] // 2
    return F.conv2d(image, kernel_2d, padding=padding, groups=image.shape[-3])


def _normalize_channels(img: torch.Tensor) -> torch.Tensor:
    max_val = img.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return img / max_val


def _make_mesh(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return yy, xx


def _piecewise_constant_sample(height: int, width: int, params: Dict) -> torch.Tensor:
    pattern_types = params.get(
        "pattern_types",
        ["checkerboard", "stripes", "circles", "mondrian"],
    )
    freq_range = params.get("frequency_range", [2, 8])
    blur_sigma = float(params.get("edge_blur_sigma", 0.0))
    contrast = float(params.get("contrast", 1.0))
    yy, xx = _make_mesh(height, width, torch.device("cpu"), torch.float32)
    choice = random.choice(pattern_types)

    if choice == "checkerboard":
        freq = random.uniform(*freq_range)
        pattern = torch.sign(torch.sin(math.pi * freq * yy) * torch.sin(math.pi * freq * xx))
    elif choice == "stripes":
        freq = random.uniform(*freq_range)
        theta = random.uniform(0.0, math.pi)
        projection = math.cos(theta) * xx + math.sin(theta) * yy
        pattern = torch.sign(torch.sin(math.pi * freq * projection))
    elif choice == "circles":
        radius = torch.sqrt(xx**2 + yy**2)
        threshold = random.uniform(0.2, 0.6)
        pattern = (radius <= threshold).float() * 2.0 - 1.0
    elif choice == "mondrian":
        pattern = torch.zeros_like(xx)
        num_rects = random.randint(3, 7)
        for _ in range(num_rects):
            x0 = random.randint(0, width - 2)
            y0 = random.randint(0, height - 2)
            x1 = random.randint(x0 + 1, width)
            y1 = random.randint(y0 + 1, height)
            value = random.uniform(-1.0, 1.0)
            pattern[y0:y1, x0:x1] = value
        pattern = torch.clamp(pattern, -1.0, 1.0)
    else:
        pattern = torch.randn(height, width)

    pattern = pattern * contrast
    pattern = pattern.unsqueeze(0)
    if blur_sigma > 0:
        pattern = _apply_gaussian_blur(pattern.unsqueeze(0), blur_sigma).squeeze(0)
    pattern = torch.clamp(pattern, -1.0, 1.0)
    return pattern


def _parametric_texture_sample(height: int, width: int, params: Dict) -> torch.Tensor:
    freq_range = params.get("frequency_range", [1, 10])
    bandwidth = params.get("bandwidth", 0.25)
    phase_jitter = params.get("phase_jitter", math.pi)
    amplitude = params.get("amplitude", 1.0)
    yy, xx = _make_mesh(height, width, torch.device("cpu"), torch.float32)
    theta = random.uniform(0.0, math.pi)
    freq = random.uniform(*freq_range)
    projection = torch.cos(torch.tensor(theta)) * xx + torch.sin(torch.tensor(theta)) * yy
    phase = random.uniform(-phase_jitter, phase_jitter)
    carrier = torch.sin(2 * math.pi * freq * projection + phase)
    envelope = torch.exp(-(projection**2) / (2 * (bandwidth**2)))
    texture = amplitude * carrier * envelope
    return texture.unsqueeze(0)


def _random_field_sample(height: int, width: int, params: Dict) -> torch.Tensor:
    alpha_range = params.get("alpha_range", [0.0, 2.0])
    alpha = random.uniform(*alpha_range)
    noise = torch.randn(1, height, width)
    fft = torch.fft.fft2(noise)
    fy = torch.fft.fftfreq(height, d=1.0).view(-1, 1)
    fx = torch.fft.fftfreq(width, d=1.0).view(1, -1)
    radius = torch.sqrt(fx**2 + fy**2)
    radius[0, 0] = 1.0
    filter_mag = radius.pow(-alpha / 2.0)
    filtered = fft * filter_mag
    field = torch.fft.ifft2(filtered).real
    field = _normalize_channels(field)
    return field


def generate_synthetic_samples(
    count: int,
    channels: int,
    height: int,
    width: int,
    data_cfg: Dict,
) -> torch.Tensor:
    family = str(data_cfg.get("family", "noise")).lower()
    params = data_cfg.get(family, {})

    images: Iterable[torch.Tensor]
    if family == "piecewise":
        images = [_piecewise_constant_sample(height, width, params) for _ in range(count)]
    elif family in {"texture", "grating"}:
        images = [_parametric_texture_sample(height, width, params) for _ in range(count)]
    elif family in {"random_field", "powerlaw"}:
        images = [_random_field_sample(height, width, params) for _ in range(count)]
    elif family == "noise":
        noise = torch.randn(count, channels, height, width)
        return noise
    else:
        logging.warning("Unknown synthetic family '%s'; falling back to Gaussian noise", family)
        return torch.randn(count, channels, height, width)

    batch = torch.stack([img.repeat(channels, 1, 1) for img in images], dim=0)
    batch = torch.clamp(batch, -1.0, 1.0)
    return batch


__all__ = ["generate_synthetic_samples"]

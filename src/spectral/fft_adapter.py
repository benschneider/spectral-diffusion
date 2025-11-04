from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, Optional, Tuple

import torch


@lru_cache(maxsize=32)
def _radial_mask_base(shape: Tuple[int, int]) -> torch.Tensor:
    """
    Pre-compute a reciprocal radial weighting mask for a given spatial size.

    The mask up-weights high frequencies during noise injection so that
    overall signal-to-noise decay remains approximately uniform across bands.
    """
    height, width = shape
    fy = torch.fft.fftfreq(height, d=1.0)
    fx = torch.fft.fftfreq(width, d=1.0)
    yy = fy[:, None]
    xx = fx[None, :]
    radius = torch.sqrt(xx**2 + yy**2)

    nonzero = radius[radius > 0]
    if nonzero.numel() == 0:
        mask = torch.ones_like(radius)
    else:
        base_radius = torch.min(nonzero)
        scaled = radius / base_radius
        mask = torch.sqrt(scaled**2 + 1.0)
        mask[0, 0] = 1.0
    return mask.to(torch.float32)


def _compute_similarity_metrics(x: torch.Tensor, y: torch.Tensor) -> dict:
    b = x.shape[0]
    x_flat = x.view(b, -1)
    y_flat = y.view(b, -1)
    x_center = x_flat - x_flat.mean(dim=1, keepdim=True)
    y_center = y_flat - y_flat.mean(dim=1, keepdim=True)
    numerator = (x_center * y_center).sum(dim=1)
    denominator = torch.sqrt(
        (x_center.pow(2).sum(dim=1) * y_center.pow(2).sum(dim=1)) + 1e-8
    )
    corr = numerator / denominator
    mse = (x_flat - y_flat).pow(2).mean(dim=1)
    return {
        "corr": float(corr.mean().item()),
        "mse": float(mse.mean().item()),
    }


def _compute_fft_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    x_fft = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
    y_fft = torch.fft.fftshift(torch.fft.fft2(y, dim=(-2, -1)), dim=(-2, -1))
    x_mag = torch.log1p(torch.abs(x_fft))
    y_mag = torch.log1p(torch.abs(y_fft))
    b = x_mag.shape[0]
    x_flat = x_mag.view(b, -1)
    y_flat = y_mag.view(b, -1)
    cosine = torch.cosine_similarity(x_flat, y_flat, dim=1)
    return float(cosine.mean().item())


def _per_sample_rms(tensor: torch.Tensor) -> torch.Tensor:
    """Compute RMS per sample over all non-batch dimensions."""
    dims = tuple(range(1, tensor.ndim))
    rms = tensor.abs().pow(2).mean(dim=dims, keepdim=True).sqrt()
    return rms.clamp_min(1e-8)


def _scale_fft_noise_for_snr(
    signal_fft: torch.Tensor,
    noise_fft: torch.Tensor,
    snr_ratio: float,
    fft_norm: str = "ortho",
) -> torch.Tensor:
    """
    Scale FFT-space noise so that RMS(signal)/RMS(noise) == snr_ratio in spatial domain.
    """
    signal_energy = signal_fft.abs().pow(2).mean(dim=(-2, -1), keepdim=True)
    noise_energy = noise_fft.abs().pow(2).mean(dim=(-2, -1), keepdim=True).clamp_min(1e-12)

    height, width = signal_fft.shape[-2:]
    parseval_corr = 1.0 if fft_norm == "ortho" else 1.0 / math.sqrt(height * width)

    scale = torch.sqrt(signal_energy / noise_energy) / snr_ratio * parseval_corr
    return noise_fft * scale


def _normalize_fft_noise(
    signal_fft: torch.Tensor,
    noise_fft: torch.Tensor,
    fft_norm: str = "ortho",
) -> torch.Tensor:
    """Match noise energy to signal energy (SNR=1) with Parseval correction."""
    return _scale_fft_noise_for_snr(signal_fft, noise_fft, snr_ratio=1.0, fft_norm=fft_norm)


def add_uniform_frequency_noise(
    x0: torch.Tensor,
    noise: torch.Tensor,
    sqrt_alpha_t: torch.Tensor,
    sqrt_one_minus_alpha_t: torch.Tensor,
    uniform_corruption: bool = False,
    strength: float = 1.0,
    mode: str = "magnitude",
    phase_std: float = 0.0,
    target_corr: Optional[float] = None,
    adaptive_rescale: bool = False,
    stats: Optional[Dict[str, float]] = None,
    fft_norm: str = "ortho",
    snr_ratio: Optional[float] = None,
    dc_scale_factor: float = 0.1,
) -> torch.Tensor:
    """
    Apply diffusion forward noise with optional uniform frequency corruption.

    When ``uniform_corruption`` is True, the noise is injected in the frequency
    domain with a reciprocal-radius weighting so that higher frequencies receive
    proportionally more energy, balancing SNR decay across the spectrum.
    """
    if not uniform_corruption:
        x_t = sqrt_alpha_t * x0 + sqrt_one_minus_alpha_t * noise
        if stats is not None:
            sim = _compute_similarity_metrics(x0, x_t)
            stats["structure_corr_pre"] = sim["corr"]
            stats["mse_pre"] = sim["mse"]
            stats["structure_corr_post"] = sim["corr"]
            stats["mse_post"] = sim["mse"]
            stats["fft_corr"] = _compute_fft_correlation(x0, x_t)
            if snr_ratio is not None:
                stats["snr_ratio"] = snr_ratio
            stats["noisy_mean"] = float(x_t.detach().mean().item())
            stats["noisy_std"] = float(x_t.detach().std().item())
        return x_t

    dims = x0.dim()
    if dims < 3:
        raise ValueError("Expected image tensor with at least 3 dimensions (C, H, W).")

    baseline_offset = 0.5
    height, width = x0.shape[-2], x0.shape[-1]
    base_mask = _radial_mask_base((height, width)).to(device=x0.device, dtype=x0.dtype)
    mask = base_mask.unsqueeze(0).unsqueeze(0)
    rms = torch.sqrt(torch.mean(mask**2)) + 1e-8
    mask = mask / rms

    strength = float(strength)
    mode = mode.lower()
    phase_std = float(phase_std)

    x_ref = x0 - baseline_offset

    X = torch.fft.fftn(x_ref, dim=(-2, -1), norm=fft_norm)
    N = torch.fft.fftn(noise, dim=(-2, -1), norm=fft_norm)

    magnitude = torch.abs(X)
    phase = torch.angle(X)

    strength_scaled = strength
    sqrt_alpha_t_complex = sqrt_alpha_t
    sqrt_one_minus_alpha_t_complex = sqrt_one_minus_alpha_t

    if stats is not None:
        stats["signal_energy"] = float(X.abs().pow(2).mean().item())

    if mode == "magnitude":
        mag_noise = (N * mask).abs()
        mag_noise = mag_noise / _per_sample_rms(mag_noise)
        dA = mag_noise * _per_sample_rms(magnitude)
        base_noise_fft = dA * torch.exp(1j * phase)
    elif mode == "phase":
        phase_noise = torch.randn_like(phase) * phase_std
        base_noise_fft = X * (torch.exp(1j * phase_noise) - 1.0)
    else:  # "complex"
        base_noise_fft = N * mask

    noise_fft = _scale_fft_noise_for_snr(X, base_noise_fft, 1.0, fft_norm=fft_norm)
    noise_fft = noise_fft * strength_scaled

    if snr_ratio is not None:
        signal_rms = _per_sample_rms(x_ref)
        scale_accum = torch.ones_like(signal_rms)
        for _ in range(2):
            noise_spatial = torch.fft.ifftn(noise_fft, dim=(-2, -1), norm=fft_norm).real
            noise_spatial_final = noise_spatial * sqrt_one_minus_alpha_t_complex
            noise_rms_final = _per_sample_rms(noise_spatial_final)
            adjust = torch.clamp(
                (signal_rms / torch.clamp(noise_rms_final, min=1e-8)) / float(snr_ratio),
                min=1e-6,
            )
            noise_fft = noise_fft * adjust
            scale_accum = scale_accum * adjust
        if stats is not None:
            stats["snr_scale_factor"] = float(scale_accum.mean().item())
    elif stats is not None:
        stats["snr_scale_factor"] = 1.0

    if stats is not None:
        stats["noise_energy"] = float(noise_fft.abs().pow(2).mean().item())

    X_t = X * sqrt_alpha_t_complex + noise_fft * sqrt_one_minus_alpha_t_complex

    if snr_ratio is not None and uniform_corruption:
        dc_signal = X[..., 0, 0]
        dc_noise = noise_fft[..., 0, 0]
        effective_dc_scale = float(dc_scale_factor) / max(float(snr_ratio), 1e-6)
        effective_dc_scale = max(0.0, min(1.0, effective_dc_scale))
        blended = dc_signal + effective_dc_scale * dc_noise
        X_t[..., 0, 0] = blended
        if stats is not None:
            stats["dc_scale_factor"] = float(dc_scale_factor)
            stats["dc_scale_effective"] = float(effective_dc_scale)
            stats["dc_mean_pre"] = float(dc_signal.real.mean().item())
            stats["dc_mean_post"] = float(blended.real.mean().item())

    x_t = torch.fft.ifftn(X_t, dim=(-2, -1), norm=fft_norm).real
    if uniform_corruption:
        x_t = x_t + baseline_offset

    sim_pre = _compute_similarity_metrics(x0, x_t)
    fft_corr = _compute_fft_correlation(x0, x_t)

    if adaptive_rescale and target_corr is not None and sim_pre["corr"] < target_corr:
        scale_factor = max(0.0, min(1.0, (sim_pre["corr"] / target_corr) ** 0.5))
        x_t = x0 * (1.0 - scale_factor) + x_t * scale_factor

    sim_post = _compute_similarity_metrics(x0, x_t)
    if stats is not None:
        stats["structure_corr_pre"] = sim_pre["corr"]
        stats["mse_pre"] = sim_pre["mse"]
        stats["structure_corr_post"] = sim_post["corr"]
        stats["mse_post"] = sim_post["mse"]
        stats["fft_corr"] = fft_corr
        if snr_ratio is not None:
            stats["snr_ratio"] = snr_ratio
        stats["noisy_mean"] = float(x_t.detach().mean().item())
        stats["noisy_std"] = float(x_t.detach().std().item())

    return x_t

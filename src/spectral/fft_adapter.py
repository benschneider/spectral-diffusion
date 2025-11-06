from __future__ import annotations

import math
from functools import lru_cache
import warnings
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


def _compute_fft_correlation(x: torch.Tensor, y: torch.Tensor, norm: str = "ortho") -> float:
    x_fft = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1), norm=norm), dim=(-2, -1))
    y_fft = torch.fft.fftshift(torch.fft.fft2(y, dim=(-2, -1), norm=norm), dim=(-2, -1))
    x_mag = torch.log1p(torch.abs(x_fft))
    y_mag = torch.log1p(torch.abs(y_fft))
    b = x_mag.shape[0]
    x_flat = x_mag.view(b, -1)
    y_flat = y_mag.view(b, -1)
    cosine = torch.cosine_similarity(x_flat, y_flat, dim=1)
    return float(cosine.mean().item())


def _per_sample_rms(tensor: torch.Tensor) -> torch.Tensor:
    """Compute RMS per sample while preserving channel dimension when present."""
    if tensor.ndim <= 2:
        dims = tuple(range(1, tensor.ndim))
    elif tensor.ndim >= 3:
        dims = tuple(range(2, tensor.ndim))
    else:
        dims = ()
    rms = tensor.abs().pow(2).mean(dim=dims, keepdim=True).sqrt()
    return rms.clamp_min(1e-8)


def _scale_fft_noise_for_snr(
    signal_spatial: torch.Tensor,
    noise_fft: torch.Tensor,
    snr_ratio: float,
    sqrt_one_minus_alpha_t: torch.Tensor,
    fft_norm: str = "ortho",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scale noise_fft so that RMS(signal) / RMS(noise) == snr_ratio in spatial domain.
    """
    signal_rms = _per_sample_rms(signal_spatial)
    noise_spatial = torch.fft.ifftn(noise_fft, dim=(-2, -1), norm=fft_norm).real
    noise_spatial = noise_spatial * sqrt_one_minus_alpha_t
    noise_rms = _per_sample_rms(noise_spatial)
    adjust = torch.clamp(
        (signal_rms / (noise_rms + 1e-8)) / snr_ratio,
        min=1e-6,
    )
    return noise_fft * adjust, adjust


def _normalize_fft_noise(
    noise_fft: torch.Tensor,
    fft_norm: str = "ortho",
) -> torch.Tensor:
    """
    Normalize noise_fft so that the corresponding spatial noise has unit RMS.

    This leverages Parseval by measuring energy after inverse FFT with the
    same normalization used during synthesis.
    """
    noise_spatial = torch.fft.ifftn(noise_fft, dim=(-2, -1), norm=fft_norm).real
    noise_rms = _per_sample_rms(noise_spatial)
    return noise_fft / noise_rms


def _check_parseval_consistency(
    spatial: torch.Tensor,
    spectrum: torch.Tensor,
    fft_norm: str,
    context: str,
) -> None:
    if not __debug__:
        return
    spatial_energy = spatial.abs().pow(2).sum()
    freq_energy = spectrum.abs().pow(2).sum()
    hw = spatial.shape[-2] * spatial.shape[-1]
    if fft_norm == "backward":
        freq_energy = freq_energy / hw
    elif fft_norm == "forward":
        freq_energy = freq_energy * hw
    diff = torch.abs(spatial_energy - freq_energy)
    if spatial_energy.abs() > 0:
        rel = diff / spatial_energy.abs()
    else:
        rel = diff
    if rel > 1e-4:
        warnings.warn(
            f"Parseval mismatch ({context}): rel_err={rel.item():.2e} "
            f"(spatial={spatial_energy.item():.4e}, freq={freq_energy.item():.4e}, norm={fft_norm})"
        )


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
    dc_scale_factor: float = 0.0,
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

    # ``dc_scale_factor`` remains part of the signature for compatibility with
    # older configs, but the corresponding DC offset manipulation has been
    # disabled while we reassess the colouring strategy.

    channel_dims = tuple(range(2, x0.ndim))
    signal_channel_mean = x0.mean(dim=channel_dims, keepdim=True)

    height, width = x0.shape[-2], x0.shape[-1]
    base_mask = _radial_mask_base((height, width)).to(device=x0.device, dtype=x0.dtype)
    mask = base_mask.unsqueeze(0).unsqueeze(0)
    rms = torch.sqrt(torch.mean(mask**2)) + 1e-8
    mask = mask / rms

    strength = float(strength)
    mode = mode.lower()
    phase_std = float(phase_std)

    x_ref = x0 - signal_channel_mean

    X = torch.fft.fftn(x_ref, dim=(-2, -1), norm=fft_norm)
    _check_parseval_consistency(x_ref, X, fft_norm, "signal")
    N = torch.fft.fftn(noise, dim=(-2, -1), norm=fft_norm)

    magnitude = torch.abs(X)
    phase = torch.angle(X)

    sqrt_alpha_t_complex = sqrt_alpha_t
    sqrt_one_minus_alpha_t_complex = sqrt_one_minus_alpha_t


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

    noise_fft = _normalize_fft_noise(base_noise_fft, fft_norm=fft_norm)

    signal_component_fft = X * sqrt_alpha_t_complex

    signal_spatial = sqrt_alpha_t * x_ref

    snr_scale_tensor = torch.ones_like(sqrt_one_minus_alpha_t_complex)
    if snr_ratio is not None:
        noise_fft, snr_scale_tensor = _scale_fft_noise_for_snr(
            signal_spatial,
            noise_fft,
            float(snr_ratio),
            sqrt_one_minus_alpha_t_complex,
            fft_norm=fft_norm,
        )

    noise_fft = noise_fft * strength
    noise_component_fft = noise_fft * sqrt_one_minus_alpha_t_complex

    if snr_ratio is not None:
        noise_component_spatial = torch.fft.ifftn(
            noise_component_fft, dim=(-2, -1), norm=fft_norm
        ).real
        signal_rms = _per_sample_rms(x_ref)
        target_noise_rms = signal_rms / float(snr_ratio)
        bias_component = (sqrt_alpha_t - 1.0) * x_ref
        dims = tuple(range(1, noise_component_spatial.ndim))
        A = bias_component.pow(2).mean(dim=dims, keepdim=True)
        B = (bias_component * noise_component_spatial).mean(dim=dims, keepdim=True)
        C = noise_component_spatial.pow(2).mean(dim=dims, keepdim=True)
        target_sq = target_noise_rms.pow(2)
        discriminant = torch.clamp(B.pow(2) - C * (A - target_sq), min=0.0)
        sqrt_disc = torch.sqrt(discriminant + 1e-12)
        scale = torch.where(
            C > 1e-12,
            (-B + sqrt_disc) / (C + 1e-12),
            torch.ones_like(C),
        )
        scale = torch.clamp(scale, min=1e-6)
        noise_component_fft = noise_component_fft * scale
        snr_scale_tensor = snr_scale_tensor * scale

    X_t = signal_component_fft + noise_component_fft

    x_t_complex = torch.fft.ifftn(X_t, dim=(-2, -1), norm=fft_norm)
    _check_parseval_consistency(x_t_complex, X_t, fft_norm, "noised")
    x_t = x_t_complex.real + signal_channel_mean

    sim_pre = _compute_similarity_metrics(x0, x_t)
    fft_corr = _compute_fft_correlation(x0, x_t, norm=fft_norm)

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
            signal_rms_measured = _per_sample_rms(x_ref)
            noise_rms_measured = _per_sample_rms(x_t - x0)
            stats["snr_measured"] = float(
                (signal_rms_measured / (noise_rms_measured + 1e-8)).mean().item()
            )
            stats["snr_scale_factor"] = float(snr_scale_tensor.mean().item())
        stats["noisy_mean"] = float(x_t.detach().mean().item())
        stats["noisy_std"] = float(x_t.detach().std().item())
        stats["signal_energy"] = float(signal_component_fft.abs().pow(2).mean().item())
        stats["noise_energy"] = float(noise_component_fft.abs().pow(2).mean().item())
        if uniform_corruption:
            noise_term = (x_t - x0).detach()
            stats["noise_channel_std_min"] = float(
                noise_term.std(dim=channel_dims, unbiased=False).min().item()
            )
            stats["noise_channel_std_max"] = float(
                noise_term.std(dim=channel_dims, unbiased=False).max().item()
            )

    return x_t

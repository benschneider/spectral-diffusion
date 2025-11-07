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
    *,
    return_noise: bool = False,
) -> torch.Tensor:
    """
    Apply diffusion forward noise with optional uniform frequency corruption.

    When ``uniform_corruption`` is True, the noise is injected in the frequency
    domain with a reciprocal-radius weighting so that higher frequencies receive
    proportionally more energy, balancing SNR decay across the spectrum.
    """
    sqrt_alpha_t_complex = sqrt_alpha_t
    sqrt_one_minus_alpha_t_complex = sqrt_one_minus_alpha_t
    signal_component = sqrt_alpha_t_complex * x0

    if not uniform_corruption:
        noise_component = sqrt_one_minus_alpha_t_complex * noise
        x_t = signal_component + noise_component
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
        if return_noise:
            effective_noise = noise_component / (sqrt_one_minus_alpha_t_complex + 1e-8)
            return x_t, effective_noise
        return x_t

    dims = x0.dim()
    if dims < 3:
        raise ValueError("Expected image tensor with at least 3 dimensions (C, H, W).")

    # ``dc_scale_factor`` remains part of the signature for compatibility with
    # older configs, but the corresponding DC offset manipulation has been
    # disabled while we reassess the colouring strategy.

    channel_dims = tuple(range(2, x0.ndim))
    height, width = x0.shape[-2], x0.shape[-1]
    base_mask = _radial_mask_base((height, width)).to(device=x0.device, dtype=x0.dtype)
    mask = base_mask.unsqueeze(0).unsqueeze(0)
    mask = mask / (torch.sqrt(torch.mean(mask**2)) + 1e-8)

    mode = mode.lower()
    strength = float(strength)
    phase_std = float(phase_std)

    signal_component_fft = torch.fft.fftn(signal_component, dim=(-2, -1), norm=fft_norm)
    _check_parseval_consistency(signal_component, signal_component_fft, fft_norm, "signal")

    base_noise_fft = torch.fft.fftn(noise, dim=(-2, -1), norm=fft_norm)

    if mode == "phase":
        phase_noise = torch.randn_like(signal_component_fft.real) * phase_std
        coloured_fft = signal_component_fft * (torch.exp(1j * phase_noise) - 1.0)
    else:  # "magnitude" and "complex" collapse to coloured random noise
        coloured_fft = base_noise_fft * mask

    coloured_fft = _normalize_fft_noise(coloured_fft, fft_norm=fft_norm)

    coloured_spatial = torch.fft.ifftn(coloured_fft, dim=(-2, -1), norm=fft_norm).real
    coloured_spatial = coloured_spatial * strength

    noise_component = sqrt_one_minus_alpha_t_complex * coloured_spatial

    snr_scale_tensor = torch.ones_like(sqrt_one_minus_alpha_t_complex)
    if snr_ratio is not None:
        signal_center = x0 - x0.mean(dim=channel_dims, keepdim=True)
        signal_rms = _per_sample_rms(signal_center)
        noise_rms = _per_sample_rms(noise_component)
        scale = (signal_rms / (noise_rms + 1e-8)) / float(snr_ratio)
        noise_component = noise_component * scale
        snr_scale_tensor = scale

    x_t_pre = signal_component + noise_component
    sim_pre = _compute_similarity_metrics(x0, x_t_pre)

    if adaptive_rescale and target_corr is not None and sim_pre["corr"] < target_corr:
        scale_factor = max(0.0, min(1.0, (sim_pre["corr"] / target_corr) ** 0.5))
        noise_component = noise_component * scale_factor
        snr_scale_tensor = snr_scale_tensor * scale_factor

    x_t = signal_component + noise_component

    noise_component_fft = torch.fft.fftn(noise_component, dim=(-2, -1), norm=fft_norm)
    _check_parseval_consistency(noise_component, noise_component_fft, fft_norm, "noise")

    X_t = torch.fft.fftn(x_t, dim=(-2, -1), norm=fft_norm)
    _check_parseval_consistency(x_t, X_t, fft_norm, "noised")

    fft_corr = _compute_fft_correlation(x0, x_t, norm=fft_norm)
    sim_post = _compute_similarity_metrics(x0, x_t)
    if stats is not None:
        stats["structure_corr_pre"] = sim_pre["corr"]
        stats["mse_pre"] = sim_pre["mse"]
        stats["structure_corr_post"] = sim_post["corr"]
        stats["mse_post"] = sim_post["mse"]
        stats["fft_corr"] = fft_corr
        if snr_ratio is not None:
            stats["snr_ratio"] = snr_ratio
            signal_rms_measured = _per_sample_rms(x0 - x0.mean(dim=channel_dims, keepdim=True))
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
            channel_std = noise_term.std(dim=channel_dims, unbiased=False)
            stats["noise_channel_std_min"] = float(channel_std.min().item())
            stats["noise_channel_std_max"] = float(channel_std.max().item())

    if return_noise:
        effective_noise = noise_component / (sqrt_one_minus_alpha_t_complex + 1e-8)
        return x_t, effective_noise

    return x_t

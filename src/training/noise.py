from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch

from src.core.numeric import safe_clamp
from src.spectral.fft_adapter import add_uniform_frequency_noise


ALPHA_MIN = 0.01
ALPHA_MAX = 0.999
SIGMA_MIN = 1e-4


@dataclass
class NoiseBatch:
    """Container for noisy samples and associated statistics."""

    noisy: torch.Tensor
    eps: torch.Tensor
    sqrt_alpha_t: torch.Tensor
    sqrt_one_minus_alpha_t: torch.Tensor
    stats: Dict[str, float]
    eps_norm: float


class NoisePreparer:
    """Prepare noisy samples for diffusion training steps."""

    def __init__(
        self,
        *,
        uniform_corruption: bool,
        uniform_corruption_scale: float,
        corruption_mode: str,
        phase_std: float,
        target_corr: Optional[float],
        adaptive_rescale: bool,
        fft_norm: str,
        snr_ratio: Optional[float],
        freq_equalized_noise: bool,
        snr_scale_min: Optional[float],
        snr_scale_max: Optional[float],
    ) -> None:
        self.uniform_corruption = bool(uniform_corruption)
        self.uniform_corruption_scale = float(uniform_corruption_scale)
        self.corruption_mode = str(corruption_mode)
        self.phase_std = float(phase_std)
        self.target_corr = target_corr
        self.adaptive_rescale = bool(adaptive_rescale)
        self.fft_norm = str(fft_norm)
        self.snr_ratio = snr_ratio
        self.freq_equalized_noise = bool(freq_equalized_noise)
        self.snr_scale_min = snr_scale_min
        self.snr_scale_max = snr_scale_max

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "NoisePreparer":
        diffusion_cfg: Mapping[str, Any] = config.get("diffusion", {}) or {}
        spectral_cfg: Mapping[str, Any] = config.get("spectral", {}) or {}

        uniform_corruption = diffusion_cfg.get(
            "uniform_corruption",
            spectral_cfg.get("uniform_corruption", False),
        )
        strength = diffusion_cfg.get(
            "uniform_corruption_scale",
            spectral_cfg.get("uniform_corruption_scale", 1.0),
        )
        corruption_mode = diffusion_cfg.get(
            "corruption_mode",
            spectral_cfg.get("corruption_mode", "magnitude"),
        )
        phase_std = diffusion_cfg.get("phase_std", spectral_cfg.get("phase_std", 0.0))
        target_corr = diffusion_cfg.get(
            "similarity_target",
            spectral_cfg.get("similarity_target"),
        )
        adaptive_rescale = diffusion_cfg.get(
            "adaptive_rescale",
            spectral_cfg.get("adaptive_rescale", False),
        )
        fft_norm = diffusion_cfg.get("fft_norm", spectral_cfg.get("fft_norm", "ortho"))
        snr_ratio = diffusion_cfg.get("snr_ratio", spectral_cfg.get("snr_ratio"))
        if snr_ratio is not None:
            snr_ratio = float(snr_ratio)
        freq_equalized = bool(spectral_cfg.get("freq_equalized_noise", False))
        snr_scale_min = diffusion_cfg.get("snr_scale_min", None)
        snr_scale_max = diffusion_cfg.get("snr_scale_max", None)
        snr_scale_min = float(snr_scale_min) if snr_scale_min is not None else None
        snr_scale_max = float(snr_scale_max) if snr_scale_max is not None else None

        return cls(
            uniform_corruption=bool(uniform_corruption),
            uniform_corruption_scale=float(strength),
            corruption_mode=str(corruption_mode),
            phase_std=float(phase_std),
            target_corr=float(target_corr) if target_corr is not None else None,
            adaptive_rescale=bool(adaptive_rescale),
            fft_norm=str(fft_norm),
            snr_ratio=snr_ratio,
            freq_equalized_noise=freq_equalized,
            snr_scale_min=snr_scale_min,
            snr_scale_max=snr_scale_max,
        )

    def prepare(
        self,
        clean_batch: torch.Tensor,
        coeffs: Any,
        timesteps: torch.Tensor,
        base_noise: Optional[torch.Tensor] = None,
    ) -> NoiseBatch:
        device = clean_batch.device
        batch_size = clean_batch.shape[0]
        sqrt_alpha_t = (
            coeffs.sqrt_alphas_cumprod[timesteps].view(batch_size, 1, 1, 1).to(device)
        )
        sqrt_alpha_t = safe_clamp(sqrt_alpha_t, min_value=ALPHA_MIN, max_value=ALPHA_MAX)
        sqrt_one_minus_alpha_t = (
            coeffs.sqrt_one_minus_alphas_cumprod[timesteps]
            .view(batch_size, 1, 1, 1)
            .to(device)
        )
        sqrt_one_minus_alpha_t = safe_clamp(
            sqrt_one_minus_alpha_t, min_value=SIGMA_MIN
        )

        noise = base_noise if base_noise is not None else torch.randn_like(clean_batch)
        stats: Dict[str, float] = {}
        noisy, eps = add_uniform_frequency_noise(
            clean_batch,
            noise,
            sqrt_alpha_t=sqrt_alpha_t,
            sqrt_one_minus_alpha_t=sqrt_one_minus_alpha_t,
            uniform_corruption=self.uniform_corruption,
            strength=self.uniform_corruption_scale,
            mode=self.corruption_mode,
            phase_std=self.phase_std,
            target_corr=self.target_corr,
            adaptive_rescale=self.adaptive_rescale,
            stats=stats,
            fft_norm=self.fft_norm,
            snr_ratio=self.snr_ratio,
            freq_equalized_noise=self.freq_equalized_noise,
            snr_scale_min=self.snr_scale_min,
            snr_scale_max=self.snr_scale_max,
            return_noise=True,
        )

        eps_norm = float(
            eps.view(eps.shape[0], -1).norm(dim=1).mean().detach().cpu().item()
            if eps.numel() > 0
            else 0.0
        )

        return NoiseBatch(
            noisy=noisy,
            eps=eps,
            sqrt_alpha_t=sqrt_alpha_t,
            sqrt_one_minus_alpha_t=sqrt_one_minus_alpha_t,
            stats=stats,
            eps_norm=eps_norm,
        )

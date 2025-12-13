from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch

from src.spectral.fft_adapter import add_uniform_frequency_noise


def _per_sample_variance(tensor: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, tensor.ndim)) if tensor.ndim > 1 else ()
    centered = tensor - tensor.mean(dim=dims, keepdim=True)
    return centered.pow(2).mean(dim=dims, keepdim=True)


@dataclass
class NoiseBatch:
    """Container for noisy samples and associated statistics."""

    noisy: torch.Tensor
    eps: torch.Tensor
    sqrt_alpha_t: torch.Tensor
    sqrt_one_minus_alpha_t: torch.Tensor
    stats: Dict[str, float]
    eps_norm: float
    snr_theory: Optional[torch.Tensor] = None
    snr_emp: Optional[torch.Tensor] = None
    snr_rel: Optional[torch.Tensor] = None


class NoisePreparer:
    """Prepare noisy samples for diffusion training steps."""

    def __init__(
        self,
        *,
        operator_mode: str,
        snr_ratio: float,
        mask_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.operator_mode = str(operator_mode or "none")
        self.snr_ratio = 1.0 if snr_ratio is None else float(snr_ratio)
        self.mask_params = dict(mask_params or {})

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "NoisePreparer":
        diffusion_cfg: Mapping[str, Any] = config.get("diffusion", {}) or {}
        operator_mode = diffusion_cfg.get("spectral_operator_mode", "none")
        snr_ratio = diffusion_cfg.get("snr_ratio", 1.0)
        mask_params = None

        return cls(
            operator_mode=str(operator_mode or "none"),
            snr_ratio=float(snr_ratio),
            mask_params=mask_params,
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
        sqrt_one_minus_alpha_t = (
            coeffs.sqrt_one_minus_alphas_cumprod[timesteps]
            .view(batch_size, 1, 1, 1)
            .to(device)
        )

        noise = base_noise if base_noise is not None else torch.randn_like(clean_batch)
        noisy, eps = add_uniform_frequency_noise(
            clean_batch,
            noise,
            sqrt_alpha_t=sqrt_alpha_t,
            sqrt_one_minus_alpha_t=sqrt_one_minus_alpha_t,
            operator_mode=self.operator_mode,
            mask_params=self.mask_params,
            snr_ratio=self.snr_ratio,
            return_noise=True,
        )

        stats, snr_theory_tensor, snr_emp_tensor, snr_rel_tensor = self._compute_stats(
            clean_batch,
            sqrt_alpha_t,
            sqrt_one_minus_alpha_t,
            eps,
            noisy,
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
            snr_theory=snr_theory_tensor,
            snr_emp=snr_emp_tensor,
            snr_rel=snr_rel_tensor,
        )

    def _compute_stats(
        self,
        clean_batch: torch.Tensor,
        sqrt_alpha_t: torch.Tensor,
        sqrt_one_minus_alpha_t: torch.Tensor,
        eps: torch.Tensor,
        noisy: torch.Tensor,
    ) -> tuple[Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha_bar = sqrt_alpha_t.pow(2)
        sigma_sq = (1.0 - alpha_bar).clamp_min(1e-8)
        snr_theory = alpha_bar / sigma_sq

        signal_component = sqrt_alpha_t * clean_batch
        noise_component = noisy - signal_component

        signal_var = _per_sample_variance(signal_component)
        noise_var = _per_sample_variance(noise_component)

        snr_emp = signal_var / (noise_var + 1e-8)
        snr_rel = snr_emp / (snr_theory + 1e-8)
        variance_sum = signal_var + noise_var

        channel_dims = tuple(range(2, noise_component.ndim))
        if channel_dims:
            channel_std = noise_component.std(dim=channel_dims, unbiased=False)
        else:
            channel_std = noise_component

        stats = {
            "snr_theory": float(snr_theory.mean().item()),
            "snr_emp": float(snr_emp.mean().item()),
            "snr_rel": float(snr_rel.mean().item()),
            "variance_sum": float(variance_sum.mean().item()),
            "noise_channel_std_min": float(channel_std.min().item()),
            "noise_channel_std_max": float(channel_std.max().item()),
        }
        return stats, snr_theory, snr_emp, snr_rel

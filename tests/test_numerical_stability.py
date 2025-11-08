from __future__ import annotations

import torch
from torch import nn

from src.core.functional.diffusion import compute_snr_weight
from src.training.sampling import DDPMSampler
from src.training.scheduler import MIN_SIGMA_THRESHOLD, build_diffusion, make_beta_schedule


class ZeroModel(nn.Module):
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.zeros_like(x)


def test_diffusion_coeffs_have_safe_sqrt() -> None:
    coeffs = build_diffusion(1000, "linear")
    assert torch.isfinite(coeffs.sqrt_one_minus_alphas_cumprod).all()
    assert coeffs.sqrt_one_minus_alphas_cumprod.min().item() >= 1e-3


def test_compute_snr_weight_handles_extremes() -> None:
    alpha = torch.ones(4, 1, 1, 1)
    sigma = torch.full_like(alpha, 1e-8)
    snr = compute_snr_weight(alpha, sigma, transform="snr")
    snr_sqrt = compute_snr_weight(alpha, sigma, transform="snr_sqrt")
    assert torch.isfinite(snr).all()
    assert torch.isfinite(snr_sqrt).all()
    assert snr.min().item() >= 0.0
    assert snr_sqrt.min().item() >= 0.0


def test_build_diffusion_trims_unstable_prefix() -> None:
    total_steps = 1000
    coeffs = build_diffusion(total_steps, "linear")
    assert coeffs.num_timesteps <= total_steps
    assert coeffs.min_safe_sigma >= MIN_SIGMA_THRESHOLD
    if coeffs.trim_offset > 0:
        original_betas = make_beta_schedule(total_steps, "linear")
        original_alphas = 1.0 - original_betas
        original_a_bar = torch.cumprod(original_alphas, dim=0)
        original_sigma = torch.sqrt(1.0 - original_a_bar)
        unsafe_prefix = original_sigma[: coeffs.trim_offset]
        assert unsafe_prefix.max().item() < MIN_SIGMA_THRESHOLD


def test_ddpm_sampler_outputs_remain_finite() -> None:
    coeffs = build_diffusion(1000, "linear")
    model = ZeroModel()
    sampler = DDPMSampler(model=model, coeffs=coeffs)
    samples = sampler.sample(
        num_samples=2,
        shape=(3, 8, 8),
        num_steps=10,
        device=torch.device("cpu"),
    )
    assert torch.isfinite(samples).all()

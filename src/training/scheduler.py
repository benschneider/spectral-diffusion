from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.core.numeric import safe_clamp, safe_sqrt


MIN_SIGMA_THRESHOLD = 0.03


@dataclass
class DiffusionCoeffs:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    min_safe_timestep: int
    min_safe_sigma: float


def make_beta_schedule(T: int, kind: str = "linear") -> torch.Tensor:
    if kind == "linear":
        start, end = 1e-4, 0.02
        return torch.linspace(start, end, T, dtype=torch.float32)
    if kind == "cosine":
        s = 0.008
        steps = torch.arange(T + 1, dtype=torch.float32) / T
        alphas_cumprod = torch.cos((steps + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-8, 0.999)
    raise ValueError(f"Unknown beta_schedule '{kind}'")


def build_diffusion(T: int, kind: str) -> DiffusionCoeffs:
    betas = make_beta_schedule(T, kind)
    alphas = 1.0 - betas
    a_bar = torch.cumprod(alphas, dim=0)
    a_bar_prev = torch.cat([torch.tensor([1.0], dtype=a_bar.dtype), a_bar[:-1]], dim=0)
    sqrt_alphas = safe_sqrt(safe_clamp(a_bar, min_value=1e-12, max_value=1.0))
    sqrt_one_minus = safe_sqrt(safe_clamp(1.0 - a_bar, min_value=1e-6))

    eligible = torch.nonzero(
        sqrt_one_minus >= MIN_SIGMA_THRESHOLD, as_tuple=False
    )
    min_safe_timestep = int(eligible[0].item()) if eligible.numel() > 0 else 0
    min_safe_sigma = float(sqrt_one_minus[min_safe_timestep].item())

    return DiffusionCoeffs(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=a_bar,
        alphas_cumprod_prev=a_bar_prev,
        sqrt_alphas_cumprod=sqrt_alphas,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus,
        min_safe_timestep=min_safe_timestep,
        min_safe_sigma=min_safe_sigma,
    )


def sample_timesteps(
    B: int,
    T: int,
    device: torch.device,
    *,
    min_timestep: int = 0,
) -> torch.Tensor:
    if not 0 <= min_timestep < T:
        raise ValueError(
            f"min_timestep={min_timestep} is outside the valid range [0, {T - 1}]"
        )
    return torch.randint(min_timestep, T, (B,), device=device, dtype=torch.long)

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.core.numeric import safe_clamp, safe_ratio, safe_sqrt


MIN_SIGMA_THRESHOLD = 0.03


@dataclass
class DiffusionCoeffs:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    min_safe_sigma: float
    num_timesteps: int
    trim_offset: int


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
    original_betas = make_beta_schedule(T, kind)
    original_alphas = 1.0 - original_betas
    original_a_bar = torch.cumprod(original_alphas, dim=0)
    original_sigma = safe_sqrt(safe_clamp(1.0 - original_a_bar, min_value=1e-8))

    eligible = torch.nonzero(original_sigma >= MIN_SIGMA_THRESHOLD, as_tuple=False)
    trim_offset = int(eligible[0].item()) if eligible.numel() > 0 else 0

    if trim_offset > 0:
        a_bar = original_a_bar[trim_offset:]
    else:
        a_bar = original_a_bar

    alphas = torch.empty_like(a_bar)
    if a_bar.numel() == 0:
        raise ValueError("Diffusion schedule produced no valid timesteps")
    alphas[0] = a_bar[0]
    if a_bar.numel() > 1:
        alphas[1:] = safe_ratio(a_bar[1:], a_bar[:-1])

    betas = 1.0 - alphas
    a_bar_prev = torch.cat([torch.tensor([1.0], dtype=a_bar.dtype), a_bar[:-1]], dim=0)
    sqrt_alphas = safe_sqrt(safe_clamp(a_bar, min_value=1e-12, max_value=1.0))
    sqrt_one_minus = safe_sqrt(safe_clamp(1.0 - a_bar, min_value=1e-6))

    min_safe_sigma = float(sqrt_one_minus.min().item())

    return DiffusionCoeffs(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=a_bar,
        alphas_cumprod_prev=a_bar_prev,
        sqrt_alphas_cumprod=sqrt_alphas,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus,
        min_safe_sigma=min_safe_sigma,
        num_timesteps=int(betas.shape[0]),
        trim_offset=trim_offset,
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

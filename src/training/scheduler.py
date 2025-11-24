from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from src.core.numeric import safe_clamp, safe_ratio, safe_sqrt


@dataclass
class DiffusionCoeffs:
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    num_timesteps: int


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


def logsnr_cosine_schedule(
    num_steps: int,
    lambda_min: float = -13.0,
    lambda_max: float = 13.0,
    delta: float = 0.008,
) -> Dict[str, torch.Tensor]:
    """Return ᾱ, σ, and log-SNR for the SD3/Flux cosine schedule.

    Implements λ(t) = λ_min + (λ_max − λ_min)·f(t) with
    f(t) = cos²(((t + δ)/(1 + δ))·π/2) normalised by its value at t = 0.
    The cumulative signal weight is ᾱ(t) = sigmoid(λ(t)), and σ(t) = sqrt(1 − ᾱ(t)).
    """

    if num_steps <= 0:
        raise ValueError("num_steps must be positive for log-SNR cosine schedule")

    t = torch.linspace(0.0, 1.0, num_steps, dtype=torch.float32)
    base = math.pi * 0.5
    denom = math.cos(base * delta / (1.0 + delta)) ** 2
    f_t = torch.cos(base * (t + delta) / (1.0 + delta)) ** 2
    f_t = (f_t / denom).clamp(0.0, 1.0)

    log_snr = torch.lerp(
        torch.full_like(f_t, float(lambda_min)),
        torch.full_like(f_t, float(lambda_max)),
        f_t,
    )
    alpha = torch.sigmoid(log_snr)
    sigma = safe_sqrt(safe_clamp(1.0 - alpha, min_value=1e-12))

    return {"alpha": alpha, "sigma": sigma, "log_snr": log_snr}


def build_diffusion(
    T: int,
    kind: str,
    schedule_kwargs: Optional[Dict[str, float]] = None,
) -> DiffusionCoeffs:
    schedule_kwargs = dict(schedule_kwargs or {})
    lower_kind = kind.replace("-", "_").lower()

    if lower_kind == "logsnr_cosine":
        schedule = logsnr_cosine_schedule(
            T,
            lambda_min=float(schedule_kwargs.get("lambda_min", -13.0)),
            lambda_max=float(schedule_kwargs.get("lambda_max", 13.0)),
            delta=float(schedule_kwargs.get("delta", 0.008)),
        )
        a_bar = schedule["alpha"].to(torch.float32)
        betas = None
    else:
        betas = make_beta_schedule(T, lower_kind)
        alphas = 1.0 - betas
        a_bar = torch.cumprod(alphas, dim=0)

    if a_bar.numel() == 0:
        raise ValueError("Diffusion schedule produced no valid timesteps")
    alphas = torch.empty_like(a_bar)
    alphas[0] = a_bar[0]
    if a_bar.numel() > 1:
        alphas[1:] = safe_ratio(a_bar[1:], a_bar[:-1])

    if betas is None:
        betas = 1.0 - alphas
    a_bar_prev = torch.cat([torch.tensor([1.0], dtype=a_bar.dtype), a_bar[:-1]], dim=0)
    sqrt_alphas = safe_sqrt(a_bar.clamp(min=0.0, max=1.0))
    sqrt_one_minus = safe_sqrt((1.0 - a_bar).clamp(min=0.0, max=1.0))

    return DiffusionCoeffs(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=a_bar,
        alphas_cumprod_prev=a_bar_prev,
        sqrt_alphas_cumprod=sqrt_alphas,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus,
        num_timesteps=int(betas.shape[0]),
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


def loss_aware_timesteps(
    B: int,
    loss_landscape: torch.Tensor,
    *,
    device: torch.device,
    temperature: float = 1.0,
    min_timestep: int = 0,
) -> torch.Tensor:
    """Sample timesteps with probabilities proportional to loss magnitudes."""

    if loss_landscape.ndim != 1:
        raise ValueError("loss_landscape must be a 1D tensor")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if min_timestep < 0 or min_timestep >= loss_landscape.numel():
        raise ValueError("min_timestep is outside the valid range")

    slice_ = loss_landscape[min_timestep:].detach().float()
    slice_ = slice_.clamp_min(1e-6)
    logits = torch.log(slice_)
    logits = logits / float(temperature)
    probs = torch.softmax(logits, dim=0)
    indices = torch.multinomial(probs, num_samples=B, replacement=True)
    timesteps = indices + min_timestep
    return timesteps.to(device=device, dtype=torch.long)

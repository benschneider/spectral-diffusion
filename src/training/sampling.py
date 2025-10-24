from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Sequence, Type

import torch

from src.training.scheduler import DiffusionCoeffs


def _make_timesteps(num_steps: int, total_steps: int, device: torch.device) -> torch.Tensor:
    if num_steps >= total_steps:
        return torch.arange(total_steps - 1, -1, -1, device=device, dtype=torch.long)
    step_indices = torch.linspace(
        total_steps - 1, 0, steps=num_steps, device=device, dtype=torch.float32
    ).round().long()
    step_indices = torch.unique(step_indices, sorted=False)
    if step_indices[-1] != 0:
        step_indices = torch.cat([step_indices, torch.zeros(1, device=device, dtype=torch.long)])
    return step_indices


class Sampler(ABC):
    """Base class for diffusion samplers."""

    def __init__(self, model: torch.nn.Module, coeffs: DiffusionCoeffs) -> None:
        self.model = model
        self.coeffs = coeffs

    @abstractmethod
    def sample(
        self,
        num_samples: int,
        shape: Sequence[int],
        num_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        raise NotImplementedError


class DDPMSampler(Sampler):
    """Reference DDPM sampler following the training-forward process."""

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        shape: Sequence[int],
        num_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        self.model.eval()
        coeffs = self.coeffs
        total_steps = coeffs.betas.shape[0]
        timesteps = _make_timesteps(num_steps, total_steps, device)

        x = torch.randn(num_samples, *shape, device=device)

        for t in timesteps:
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            eps = self.model(x, t_batch)

            beta_t = coeffs.betas[t]
            alpha_t = coeffs.alphas[t]
            alpha_bar_t = coeffs.alphas_cumprod[t]

            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)

            pred_x0 = (x - sqrt_one_minus * eps) / torch.sqrt(alpha_bar_t)

            if t > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt(beta_t)
                x = (
                    (1.0 / sqrt_alpha_t) * (x - beta_t / sqrt_one_minus * eps)
                    + sigma_t * noise
                )
            else:
                x = (1.0 / sqrt_alpha_t) * (x - beta_t / sqrt_one_minus * eps)

            x = torch.clamp(x, -1.0, 1.0)
        return x


SAMPLER_REGISTRY: Dict[str, Type[Sampler]] = {
    "ddpm": DDPMSampler,
}


def register_sampler(name: str, sampler_cls: Type[Sampler]) -> None:
    SAMPLER_REGISTRY[name.lower()] = sampler_cls


def build_sampler(name: str, model: torch.nn.Module, coeffs: DiffusionCoeffs) -> Sampler:
    sampler_cls = SAMPLER_REGISTRY.get(name.lower())
    if sampler_cls is None:
        raise ValueError(f"Unknown sampler '{name}'. Available: {sorted(SAMPLER_REGISTRY.keys())}")
    return sampler_cls(model=model, coeffs=coeffs)


@torch.no_grad()
def sample_ddpm(
    model: torch.nn.Module,
    coeffs: DiffusionCoeffs,
    num_samples: int,
    shape: Sequence[int],
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Compatibility shim for legacy imports."""
    sampler = DDPMSampler(model=model, coeffs=coeffs)
    return sampler.sample(
        num_samples=num_samples,
        shape=shape,
        num_steps=num_steps,
        device=device,
    )

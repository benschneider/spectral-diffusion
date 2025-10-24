from __future__ import annotations

from typing import Iterable, Sequence

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


@torch.no_grad()
def sample_ddpm(
    model: torch.nn.Module,
    coeffs: DiffusionCoeffs,
    num_samples: int,
    shape: Sequence[int],
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    total_steps = coeffs.betas.shape[0]
    timesteps = _make_timesteps(num_steps, total_steps, device)

    x = torch.randn(num_samples, *shape, device=device)

    for t in timesteps:
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        eps = model(x, t_batch)

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

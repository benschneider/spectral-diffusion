from __future__ import annotations

import torch

from src.core.numeric import compute_snr, safe_clamp, safe_sqrt


def compute_target(
    prediction_type: str,
    x0: torch.Tensor,
    xt: torch.Tensor,
    eps: torch.Tensor,
    alpha_t: torch.Tensor,
    sigma_t: torch.Tensor,
) -> torch.Tensor:
    if prediction_type == "eps":
        return eps
    if prediction_type == "x0":
        return x0
    if prediction_type == "v":
        return alpha_t * eps - sigma_t * x0
    raise ValueError(f"Unknown prediction_type '{prediction_type}'")


def compute_snr_weight(
    alpha_t: torch.Tensor,
    sigma_t: torch.Tensor,
    transform: str = "snr",
) -> torch.Tensor:
    snr = compute_snr(alpha_t, sigma_t)
    if transform == "snr":
        return torch.clamp(snr, max=1e4)
    if transform == "snr_sqrt":
        return safe_sqrt(safe_clamp(snr, min_value=0.0))
    if transform == "snr_clamped":
        return safe_clamp(snr, max_value=10.0)
    raise ValueError(f"Unknown SNR transform '{transform}'")

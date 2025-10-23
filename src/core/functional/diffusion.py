from __future__ import annotations

from typing import Optional

import torch


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
    snr = (alpha_t**2) / (sigma_t**2 + 1e-8)
    if transform == "snr":
        return snr
    if transform == "snr_sqrt":
        return torch.sqrt(snr)
    if transform == "snr_clamped":
        return torch.minimum(snr, torch.full_like(snr, 10.0))
    raise ValueError(f"Unknown SNR transform '{transform}'")

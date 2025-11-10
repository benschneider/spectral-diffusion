"""Core helpers for diffusion regime selection and prediction utilities."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor

from .numeric import safe_ratio


def predict_x0(
    prediction: Tensor,
    prediction_type: str,
    x_t: Tensor,
    sqrt_alpha_t: Tensor,
    sqrt_one_minus_alpha_t: Tensor,
) -> Tensor:
    """Return the implied clean sample given a diffusion prediction."""

    prediction_type = prediction_type.lower()
    if prediction_type == "eps":
        return safe_ratio(
            x_t - sqrt_one_minus_alpha_t * prediction,
            sqrt_alpha_t,
            min_den=1e-8,
        )
    if prediction_type == "x0":
        return prediction
    if prediction_type == "v":
        return sqrt_alpha_t * prediction + sqrt_one_minus_alpha_t * x_t
    raise ValueError(f"Unsupported prediction_type '{prediction_type}'")


def select_regime(snr: Tensor, snr_clip: float) -> Dict[str, Tensor]:
    """Return boolean masks that partition the batch by SNR regime."""

    noise_dom = snr < 1.0
    overflow = snr > snr_clip
    balanced = (~noise_dom) & (~overflow)
    return {"noise": noise_dom, "balanced": balanced, "overflow": overflow}


def describe_regime(masks: Dict[str, Tensor]) -> Tuple[str, str]:
    """Return human readable regime and loss descriptions for logging."""

    if torch.any(masks["overflow"]):
        return "deterministic", "x0"
    if torch.any(masks["balanced"]):
        return "hybrid", "eps"
    return "stochastic", "eps"


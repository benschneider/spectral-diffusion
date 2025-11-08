from __future__ import annotations

from typing import Optional

import torch

EPS = 1e-6


def safe_clamp(tensor: torch.Tensor, *, min_value: Optional[float] = None, max_value: Optional[float] = None) -> torch.Tensor:
    """Clamp ``tensor`` with optional minimum/maximum bounds.

    ``torch.clamp`` does not accept ``None`` for keyword-only arguments in all
    versions, so this wrapper keeps the call-sites uniform while still allowing
    callers to skip either bound.
    """

    kwargs = {}
    if min_value is not None:
        kwargs["min"] = float(min_value)
    if max_value is not None:
        kwargs["max"] = float(max_value)
    if not kwargs:
        return tensor
    return torch.clamp(tensor, **kwargs)


def safe_sqrt(tensor: torch.Tensor, *, min_value: float = EPS) -> torch.Tensor:
    """Return ``sqrt`` of ``tensor`` after clamping it away from zero."""

    return torch.sqrt(safe_clamp(tensor, min_value=min_value))


def safe_reciprocal(tensor: torch.Tensor, *, min_value: float = EPS) -> torch.Tensor:
    """Return the reciprocal while preventing divisions by tiny magnitudes."""

    return 1.0 / safe_clamp(tensor, min_value=min_value)


def safe_ratio(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    *,
    min_den: float = EPS,
    max_value: Optional[float] = None,
) -> torch.Tensor:
    """Compute ``numerator / denominator`` with clamping safeguards."""

    result = numerator / safe_clamp(denominator, min_value=min_den)
    if max_value is not None:
        result = safe_clamp(result, max_value=max_value)
    return result


def compute_snr(
    alpha_t: torch.Tensor,
    sigma_t: torch.Tensor,
    *,
    min_sigma: float = EPS,
    max_value: float = 1e3,
) -> torch.Tensor:
    """Return a diffusion signal-to-noise ratio with shared safeguards."""

    return safe_ratio(alpha_t**2, sigma_t**2, min_den=min_sigma, max_value=max_value)

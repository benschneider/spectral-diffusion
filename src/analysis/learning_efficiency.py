"""Learning efficiency diagnostics for diffusion training."""

from __future__ import annotations

from typing import Optional


def compute_efficiency(
    prev_loss: Optional[float],
    curr_loss: float,
    grad_norm: float,
    *,
    eps: float = 1e-8,
) -> Optional[float]:
    """Return the normalized learning signal Δloss/‖grad‖."""

    if prev_loss is None:
        return None
    if not float(grad_norm) or abs(grad_norm) <= eps:
        return None
    delta = prev_loss - curr_loss
    return float(delta / max(abs(grad_norm), eps))

"""Signal-to-noise utilities shared across training and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor

from .numeric import compute_snr, safe_clamp, safe_ratio

EPS = 1e-8


@dataclass
class SNRStats:
    snr_raw: Tensor
    snr_clamped: Tensor
    snr_weight: Tensor
    log_snr: Tensor


@dataclass
class BatchSNR:
    signal_rms: Tensor
    noise_rms: Tensor
    snr_measured: Tensor


def soft_snr_weight(snr: Tensor) -> Tensor:
    """Return a bounded weighting factor derived from log-SNR."""

    return torch.tanh(0.5 * torch.log(snr.clamp_min(EPS))).clamp(0.0, 1.0)


def compute_snr_stats(
    sqrt_alpha_t: Tensor,
    sqrt_one_minus_alpha_t: Tensor,
    *,
    snr_clip: float,
    min_sigma: float = EPS,
) -> SNRStats:
    snr_raw = compute_snr(sqrt_alpha_t, sqrt_one_minus_alpha_t, min_sigma=min_sigma)
    snr_clamped = safe_clamp(snr_raw, max_value=snr_clip)
    snr_weight = soft_snr_weight(snr_raw)
    log_snr = torch.log(snr_raw.clamp_min(EPS))
    return SNRStats(snr_raw=snr_raw, snr_clamped=snr_clamped, snr_weight=snr_weight, log_snr=log_snr)


def measure_batch_snr(
    clean_batch: Tensor,
    noisy_batch: Tensor,
    sqrt_alpha_t: Tensor,
) -> BatchSNR:
    """Measure signal and noise RMS along with SNR for a batch."""

    dims: Tuple[int, ...] = tuple(range(1, clean_batch.ndim))
    centered_signal = clean_batch - clean_batch.mean(dim=dims, keepdim=True)
    signal_rms = torch.sqrt(centered_signal.pow(2).mean(dim=dims))

    broadcast_alpha = sqrt_alpha_t
    while broadcast_alpha.ndim < clean_batch.ndim:
        broadcast_alpha = broadcast_alpha.unsqueeze(-1)
    noise_component = noisy_batch - broadcast_alpha * clean_batch
    noise_rms = torch.sqrt(noise_component.pow(2).mean(dim=dims))
    snr_measured = safe_ratio(signal_rms.pow(2), noise_rms.pow(2), min_den=EPS)
    return BatchSNR(signal_rms=signal_rms, noise_rms=noise_rms, snr_measured=snr_measured)


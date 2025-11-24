"""Overflow management utilities for diffusion training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import logging
import torch
from torch import Tensor

from .diffusion_step import describe_regime


@dataclass
class OverflowStats:
    ratio: float
    ema: float
    count: int


class OverflowHandler:
    """Track and mitigate extremely high-SNR regimes."""

    def __init__(
        self,
        *,
        snr_clip: float = 250.0,
        ema_decay: float = 0.9,
        enable_renorm: bool = False,
        log_interval: Optional[int] = None,
    ) -> None:
        self.snr_clip = float(snr_clip)
        self.ema_decay = float(ema_decay)
        self.enable_renorm = bool(enable_renorm)
        self.log_interval = None
        if log_interval is not None:
            value = int(log_interval)
            self.log_interval = value if value > 0 else 1
        self._ema = 0.0
        self._log_counter = 0
        self._emitted_once = False

    def renormalise(self, prediction: Tensor, overflow_mask: Tensor) -> Tensor:
        """Renormalise overflowing predictions to limit variance growth."""

        if not self.enable_renorm:
            return prediction

        if not torch.any(overflow_mask):
            return prediction

        dims = tuple(range(1, prediction.ndim))
        mean = prediction.mean(dim=dims, keepdim=True)
        std = prediction.std(dim=dims, unbiased=False, keepdim=True)
        renormed = (prediction - mean) / (std + 1e-6)

        mask = overflow_mask
        while mask.ndim < prediction.ndim:
            mask = mask.unsqueeze(-1)
        return torch.where(mask, renormed, prediction)

    def update(self, overflow_mask: Tensor) -> OverflowStats:
        if overflow_mask.numel() == 0:
            return OverflowStats(ratio=0.0, ema=self._ema, count=0)
        ratio = float(overflow_mask.float().mean().item())
        self._ema = self.ema_decay * self._ema + (1.0 - self.ema_decay) * ratio
        count = int(torch.count_nonzero(overflow_mask).item())
        return OverflowStats(ratio=ratio, ema=self._ema, count=count)

    def log(self, snr_display: Tensor, regimes: Dict[str, Tensor]) -> None:
        overflow = regimes.get("overflow")
        if overflow is None or overflow.numel() == 0:
            return
        if not torch.any(overflow):
            return
        self._log_counter += 1
        if self.log_interval is not None:
            if self._log_counter % self.log_interval != 0:
                return
        elif self._emitted_once:
            return
        ratio = float(overflow.float().mean().item())
        max_snr = float(snr_display.max().item()) if snr_display.numel() > 0 else 0.0
        msg = (
            f"[OverflowHandler] overflow_ratio={ratio:.4f} "
            f"max_snr={max_snr:.2f} enable_renorm={self.enable_renorm}"
        )
        print(msg)
        logging.warning(msg)
        self._emitted_once = True

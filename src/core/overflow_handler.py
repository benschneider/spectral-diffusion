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
    ) -> None:
        self.snr_clip = float(snr_clip)
        self.ema_decay = float(ema_decay)
        self.enable_renorm = bool(enable_renorm)
        self._ema = 0.0

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
        if not torch.any(regimes["overflow"]):
            return
        mode, loss_mode = describe_regime(regimes)
        message = (
            "[OverflowHandler] "
            f"mode={mode} "
            f"snr={float(snr_display.max().item()):.1f} "
            f"loss_mode={loss_mode} "
            f"count={int(torch.count_nonzero(regimes['overflow']).item())}"
        )
        logging.getLogger("OverflowHandler").error(message)
        print(message)

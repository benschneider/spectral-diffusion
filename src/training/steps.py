from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.core.functional import compute_snr_weight, compute_target
from src.training.noise import NoiseBatch


@dataclass
class StepOutcome:
    """Scalar metrics emitted by a single training step."""

    loss: float
    mae: float
    grad_norm: Optional[float]


class TrainingStepExecutor:
    """Encapsulate the forward/backward pass for a training iteration."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[Tensor, Optional[Tensor]], Tensor],
        *,
        prediction_type: str,
        snr_weighting: bool,
        snr_transform: str,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.prediction_type = prediction_type
        self.snr_weighting = snr_weighting
        self.snr_transform = snr_transform

    def run_step(
        self,
        clean_batch: Tensor,
        noise_batch: NoiseBatch,
        timesteps: Tensor,
        grad_callback: Optional[Callable[[], Optional[float]]] = None,
    ) -> StepOutcome:
        prediction = self.model(noise_batch.noisy, timesteps)
        target = compute_target(
            self.prediction_type,
            clean_batch,
            noise_batch.noisy,
            noise_batch.eps,
            noise_batch.sqrt_alpha_t,
            noise_batch.sqrt_one_minus_alpha_t,
        )
        residual = prediction - target
        weight: Optional[Tensor] = None
        if self.snr_weighting:
            weight = compute_snr_weight(
                noise_batch.sqrt_alpha_t,
                noise_batch.sqrt_one_minus_alpha_t,
                transform=self.snr_transform,
            )
        loss = self.loss_fn(residual, weight)
        mae = F.l1_loss(prediction, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = grad_callback() if grad_callback else None
        self.optimizer.step()

        return StepOutcome(
            loss=float(loss.detach().cpu()),
            mae=float(mae.detach().cpu()),
            grad_norm=grad_norm,
        )

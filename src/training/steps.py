from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.core.fft_feedback import compute_fft_feedback
from src.core.functional import compute_target
from src.training.noise import NoiseBatch


@dataclass
class StepOutcome:
    """Scalar metrics emitted by a single training step."""

    loss: float
    mae: float
    grad_norm: Optional[float]
    fft_feedback: Dict[str, float]
    coeff_stats: Dict[str, float]
    batch_stats: Dict[str, float]
    per_example_mse: Optional[torch.Tensor] = None


class TrainingStepExecutor:
    """Encapsulate the forward/backward pass for a training iteration."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[..., Tensor],
        *,
        prediction_type: str,
        fft_norm: str,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.prediction_type = prediction_type
        self.fft_norm = fft_norm

    def run_step(
        self,
        clean_batch: Tensor,
        noise_batch: NoiseBatch,
        timesteps: Tensor,
        grad_callback: Optional[Callable[[], Optional[float]]] = None,
    ) -> StepOutcome:
        sqrt_alpha_t = noise_batch.sqrt_alpha_t
        sqrt_one_minus_alpha_t = noise_batch.sqrt_one_minus_alpha_t

        prediction = self.model(noise_batch.noisy, timesteps)
        target = compute_target(
            self.prediction_type,
            clean_batch,
            noise_batch.noisy,
            noise_batch.eps,
            sqrt_alpha_t,
            sqrt_one_minus_alpha_t,
        )

        residual = prediction - target
        B = residual.shape[0]
        per_example_mse = (
            residual.detach().view(B, -1).pow(2).mean(dim=1).to(device=residual.device)
        )

        loss_result = self.loss_fn(
            prediction,
            target,
            sqrt_alpha_t,
            sqrt_one_minus_alpha_t,
            snr_rel=noise_batch.snr_rel,
        )
        if isinstance(loss_result, tuple):
            loss, loss_diag = loss_result
        else:
            loss = loss_result
            loss_diag: Dict[str, float] = {}

        mae_value = loss_diag.get("mae")
        if mae_value is None:
            mae = F.l1_loss(prediction, target)
        elif isinstance(mae_value, torch.Tensor):
            mae = mae_value
        else:
            mae = loss.new_tensor(float(mae_value))

        per_sample_loss_tensor = loss_diag.get("per_sample_loss")
        if isinstance(per_sample_loss_tensor, torch.Tensor) and per_sample_loss_tensor.shape[0] == B:
            per_example_loss = per_sample_loss_tensor.detach().cpu()
        else:
            per_example_loss = per_example_mse.detach().cpu()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = grad_callback() if grad_callback else None
        self.optimizer.step()

        fft_feedback = compute_fft_feedback(
            prediction,
            target,
            fft_norm=self.fft_norm,
        )

        coeff_stats = {
            "timestep_min": float(timesteps.min().item()),
            "timestep_max": float(timesteps.max().item()),
            "timestep_mean": float(timesteps.float().mean().item()),
            "sqrt_alpha_min": float(sqrt_alpha_t.min().item()),
            "sqrt_alpha_max": float(sqrt_alpha_t.max().item()),
            "sqrt_alpha_mean": float(sqrt_alpha_t.mean().item()),
            "sqrt_one_minus_min": float(sqrt_one_minus_alpha_t.min().item()),
            "sqrt_one_minus_max": float(sqrt_one_minus_alpha_t.max().item()),
            "sqrt_one_minus_mean": float(sqrt_one_minus_alpha_t.mean().item()),
        }
        coeff_stats.update({key: float(value) for key, value in (noise_batch.stats or {}).items()})

        prediction_mean = float(prediction.detach().mean().item())
        prediction_std = float(prediction.detach().std().item())
        target_mean = float(target.detach().mean().item())
        target_std = float(target.detach().std().item())
        residual_mean = float(residual.detach().mean().item())
        residual_std = float(residual.detach().std().item())

        batch_stats = {
            "prediction_mean": prediction_mean,
            "prediction_std": prediction_std,
            "prediction_abs_max": float(prediction.detach().abs().max().item()),
            "target_mean": target_mean,
            "target_std": target_std,
            "target_abs_max": float(target.detach().abs().max().item()),
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "residual_abs_max": float(residual.detach().abs().max().item()),
            "residual_mse": float(residual.detach().pow(2).mean().item()),
        }

        return StepOutcome(
            loss=float(loss.detach().cpu()),
            mae=float(mae.detach().cpu()),
            grad_norm=grad_norm,
            fft_feedback=fft_feedback,
            coeff_stats=coeff_stats,
            batch_stats=batch_stats,
            per_example_mse=per_example_loss,
        )

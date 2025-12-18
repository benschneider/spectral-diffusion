import torch
from torch import nn

from src.training.noise import NoiseBatch
from src.training.steps import TrainingStepExecutor


class ConstantLoss:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prediction, target, *_, snr_rel=None, **__):
        self.calls += 1
        residual = target - prediction
        loss = residual.pow(2).mean()
        diag = {
            "mae": float(residual.abs().mean().item()),
            "per_sample_loss": residual.view(residual.shape[0], -1).pow(2).mean(dim=1),
        }
        return loss, diag


def _noise_batch() -> NoiseBatch:
    tensor = torch.ones((2, 1, 4, 4))
    stats = {"snr_theory": 10.0, "snr_emp": 9.5, "snr_rel": 0.95, "variance_sum": 1.0}
    return NoiseBatch(
        noisy=tensor.clone(),
        eps=tensor.clone(),
        sqrt_alpha_t=torch.full((2, 1, 1, 1), 0.8),
        sqrt_one_minus_alpha_t=torch.full((2, 1, 1, 1), 0.6),
        stats=stats,
        eps_norm=1.0,
        snr_theory=torch.full((2, 1, 1, 1), 10.0),
        snr_emp=torch.full((2, 1, 1, 1), 9.5),
        snr_rel=torch.full((2, 1, 1, 1), 0.95),
    )


def test_training_step_executor_runs_backward_and_logs_stats(monkeypatch):
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.0))

        def forward(self, x, timesteps):  # type: ignore[override]
            return self.weight * torch.ones_like(x)

    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    loss_fn = ConstantLoss()
    executor = TrainingStepExecutor(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        prediction_type="eps",
        fft_norm="ortho",
    )

    clean = torch.zeros_like(_noise_batch().noisy)
    noise_batch = _noise_batch()
    timesteps = torch.tensor([0, 1])

    outcome = executor.run_step(clean, noise_batch, timesteps)

    assert loss_fn.calls == 1
    assert outcome.loss >= 0.0
    assert outcome.grad_norm is None or outcome.grad_norm >= 0.0
    assert "snr_theory" in outcome.coeff_stats
    assert "prediction_mean" in outcome.batch_stats
    assert outcome.per_example_mse is not None


def test_training_step_executor_uses_loss_diag_per_sample(monkeypatch):
    class ZeroModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.0))

        def forward(self, x, timesteps):  # type: ignore[override]
            return self.weight * torch.ones_like(x)

    model = ZeroModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss_fn = ConstantLoss()
    executor = TrainingStepExecutor(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        prediction_type="eps",
        fft_norm="backward",
    )

    clean = torch.randn(2, 1, 4, 4)
    noise_batch = _noise_batch()
    timesteps = torch.tensor([2, 3])

    outcome = executor.run_step(clean, noise_batch, timesteps)

    assert outcome.per_example_mse.shape[0] == clean.shape[0]
    assert outcome.fft_feedback["amplitude_mae"] >= 0.0

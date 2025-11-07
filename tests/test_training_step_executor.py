import pytest
import torch
from torch import nn

from src.training.noise import NoiseBatch
from src.training.steps import TrainingStepExecutor


def test_step_executor_runs_backward_and_invokes_callback(monkeypatch):
    class ConstantModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, x, timesteps):  # pylint: disable=unused-argument
            return self.weight * torch.ones_like(x)

    model = ConstantModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    captured = {}

    def fake_compute_target(pred_type, clean_batch, noisy_batch, eps, sqrt_alpha, sqrt_one_minus):
        captured["target_args"] = (pred_type, clean_batch.shape)
        return torch.zeros_like(noisy_batch)

    def fake_compute_snr_weight(sqrt_alpha, sqrt_one_minus, transform="snr"):
        captured["weight_args"] = transform
        return torch.ones_like(sqrt_alpha)

    def fake_fft_feedback(prediction, target, fft_norm="ortho"):
        captured["fft_norm"] = fft_norm
        return {
            "amplitude_mae": 1.0,
            "phase_mae": 0.5,
            "real_mae": 0.25,
            "imag_mae": 0.75,
            "complex_mae": 0.9,
        }

    monkeypatch.setattr("src.training.steps.compute_target", fake_compute_target)
    monkeypatch.setattr("src.training.steps.compute_snr_weight", fake_compute_snr_weight)
    monkeypatch.setattr("src.training.steps.compute_fft_feedback", fake_fft_feedback)

    def loss_fn(residual, weight):
        captured["loss_weight"] = weight
        return residual.mean()

    executor = TrainingStepExecutor(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        prediction_type="eps",
        snr_weighting=True,
        snr_transform="logsnr",
        fft_norm="ortho",
    )

    clean = torch.zeros((1, 1, 2, 2))
    noise_batch = NoiseBatch(
        noisy=torch.ones_like(clean),
        eps=torch.ones_like(clean),
        sqrt_alpha_t=torch.ones((1, 1, 1, 1)),
        sqrt_one_minus_alpha_t=torch.ones((1, 1, 1, 1)),
        stats={},
        eps_norm=1.0,
    )
    timesteps = torch.tensor([0])

    callback_called = []

    def grad_callback():
        callback_called.append(True)
        return 3.0

    outcome = executor.run_step(clean, noise_batch, timesteps, grad_callback=grad_callback)

    assert callback_called
    assert captured["target_args"][0] == "eps"
    assert captured["weight_args"] == "logsnr"
    assert captured["loss_weight"].shape == (1, 1, 1, 1)
    assert outcome.grad_norm == pytest.approx(3.0)
    assert outcome.loss > 0
    assert outcome.fft_feedback["amplitude_mae"] == pytest.approx(1.0)
    assert captured["fft_norm"] == "ortho"


def test_step_executor_handles_absent_weight(monkeypatch):
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0))

        def forward(self, x, timesteps):  # pylint: disable=unused-argument
            return self.weight * torch.ones_like(x)

    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    monkeypatch.setattr(
        "src.training.steps.compute_target", lambda *args, **kwargs: torch.zeros_like(args[1])
    )

    loss_weights = []

    def loss_fn(residual, weight):
        loss_weights.append(weight)
        return residual.pow(2).mean()

    executor = TrainingStepExecutor(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        prediction_type="eps",
        snr_weighting=False,
        snr_transform="snr",
        fft_norm="backward",
    )

    clean = torch.zeros((1, 1, 2, 2))
    noise_batch = NoiseBatch(
        noisy=torch.ones_like(clean),
        eps=torch.ones_like(clean),
        sqrt_alpha_t=torch.ones((1, 1, 1, 1)),
        sqrt_one_minus_alpha_t=torch.ones((1, 1, 1, 1)),
        stats={},
        eps_norm=1.0,
    )
    timesteps = torch.tensor([0])

    executor.run_step(clean, noise_batch, timesteps)

    assert loss_weights[-1] is None

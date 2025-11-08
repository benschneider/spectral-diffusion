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
    assert "timestep_mean" in outcome.coeff_stats
    assert "prediction_mean" in outcome.batch_stats
    assert outcome.weight_stats is not None
    assert outcome.weight_stats["min"] == pytest.approx(1.0)
    assert outcome.weight_stats["max"] == pytest.approx(1.0)
    assert outcome.weight_stats["mean"] == pytest.approx(1.0)


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

    outcome = executor.run_step(clean, noise_batch, timesteps)

    assert loss_weights[-1] is None
    assert outcome.weight_stats is None


def test_step_executor_clamps_and_reports_snr(monkeypatch, capsys):
    class ZeroModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.0))

        def forward(self, x, timesteps):  # pylint: disable=unused-argument
            return self.weight * torch.ones_like(x)

    model = ZeroModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    monkeypatch.setattr(
        "src.training.steps.compute_target",
        lambda *args, **kwargs: torch.zeros_like(args[1]),
    )

    executor = TrainingStepExecutor(
        model=model,
        optimizer=optimizer,
        loss_fn=lambda residual, weight: residual.pow(2).mean(),
        prediction_type="eps",
        snr_weighting=False,
        snr_transform="snr",
        fft_norm="ortho",
    )

    clean = torch.zeros((1, 1, 2, 2))
    tiny = torch.full((1, 1, 1, 1), 1e-5)
    noise_batch = NoiseBatch(
        noisy=torch.zeros_like(clean),
        eps=torch.zeros_like(clean),
        sqrt_alpha_t=torch.ones((1, 1, 1, 1)),
        sqrt_one_minus_alpha_t=tiny,
        stats={},
        eps_norm=0.0,
    )
    timesteps = torch.tensor([0])

    outcome = executor.run_step(clean, noise_batch, timesteps)

    captured = capsys.readouterr()
    assert "SNR overflow detected" in captured.out
    assert outcome.coeff_stats["snr_max"] <= 1e3 + 1e-6
    assert outcome.coeff_stats["snr_raw_max"] > 1e3

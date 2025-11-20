import pytest
import torch

from src.training.noise import NoisePreparer


def test_noise_preparer_from_config_prefers_diffusion_over_spectral():
    config = {
        "diffusion": {
            "uniform_corruption": True,
            "uniform_corruption_scale": 2.5,
            "corruption_mode": "phase",
            "phase_std": 0.3,
            "similarity_target": 0.75,
            "adaptive_rescale": True,
            "fft_norm": "backward",
            "snr_ratio": 1.25,
        },
        "spectral": {
            "uniform_corruption": False,
            "uniform_corruption_scale": 0.5,
            "corruption_mode": "magnitude",
            "phase_std": 0.1,
            "similarity_target": 0.2,
            "adaptive_rescale": False,
            "fft_norm": "ortho",
            "snr_ratio": 2.0,
        },
    }

    preparer = NoisePreparer.from_config(config)

    assert preparer.uniform_corruption is True
    assert preparer.uniform_corruption_scale == 2.5
    assert preparer.corruption_mode == "phase"
    assert preparer.phase_std == 0.3
    assert preparer.target_corr == 0.75
    assert preparer.adaptive_rescale is True
    assert preparer.fft_norm == "backward"
    assert preparer.snr_ratio == 1.25


def test_noise_preparer_prepare_uses_adapter(monkeypatch):
    config = {"diffusion": {"uniform_corruption": True}}
    preparer = NoisePreparer.from_config(config)

    class DummyCoeffs:
        def __init__(self):
            values = torch.linspace(0.1, 0.9, steps=5)
            self.sqrt_alphas_cumprod = values
            self.sqrt_one_minus_alphas_cumprod = 1 - values

    coeffs = DummyCoeffs()
    clean = torch.zeros((2, 1, 2, 2))
    timesteps = torch.tensor([1, 3])
    captured = {}

    def fake_adapter(clean_batch, noise_batch, **kwargs):
        captured.update(kwargs)
        stats = kwargs["stats"]
        stats["noisy_mean"] = 1.23
        return clean_batch + 1.0, torch.full_like(clean_batch, 0.5)

    monkeypatch.setattr("src.training.noise.add_uniform_frequency_noise", fake_adapter)

    batch = preparer.prepare(clean, coeffs, timesteps)

    assert torch.allclose(batch.noisy, torch.ones_like(clean))
    assert torch.allclose(batch.eps, torch.full_like(clean, 0.5))
    assert captured["uniform_corruption"] is True
    assert captured["strength"] == preparer.uniform_corruption_scale
    assert batch.stats["noisy_mean"] == 1.23
    assert batch.eps_norm == pytest.approx(1.0)
    assert batch.sqrt_alpha_t.shape == (2, 1, 1, 1)
    assert batch.sqrt_one_minus_alpha_t.shape == (2, 1, 1, 1)


class _DummyCoeffs:
    def __init__(self):
        values = torch.linspace(0.15, 0.95, steps=10)
        self.sqrt_alphas_cumprod = values
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - values**2)


@pytest.mark.parametrize(
    "uniform_corruption, adaptive_rescale",
    [(False, False), (True, False), (True, True)],
)
def test_eps_mode_reconstruction_matches_clean(uniform_corruption, adaptive_rescale):
    torch.manual_seed(0)

    config = {
        "diffusion": {
            "uniform_corruption": uniform_corruption,
            "adaptive_rescale": adaptive_rescale,
            "similarity_target": 0.95 if adaptive_rescale else None,
            "snr_ratio": 1.0,
        }
    }

    preparer = NoisePreparer.from_config(config)
    coeffs = _DummyCoeffs()
    clean = torch.rand((2, 3, 16, 16))
    timesteps = torch.tensor([2, 7])

    batch = preparer.prepare(clean, coeffs, timesteps)

    reconstructed = (batch.noisy - batch.sqrt_one_minus_alpha_t * batch.eps) / (
        batch.sqrt_alpha_t + 1e-8
    )

    assert torch.allclose(reconstructed, clean, atol=1e-5)


@pytest.mark.parametrize("snr_ratio", [0.5, 1.0, 2.0])
def test_snr_ratio_matches_signal_energy(snr_ratio):
    torch.manual_seed(1)

    config = {
        "diffusion": {
            "uniform_corruption": True,
            "snr_ratio": snr_ratio,
            "adaptive_rescale": False,
            "uniform_corruption_scale": 0.2,
        }
    }

    preparer = NoisePreparer.from_config(config)
    coeffs = _DummyCoeffs()
    clean = torch.rand((4, 3, 16, 16))
    timesteps = torch.tensor([1, 3, 6, 8])

    batch = preparer.prepare(clean, coeffs, timesteps)

    signal_component = batch.sqrt_alpha_t * clean
    noise_component = batch.noisy - signal_component

    channel_dims = tuple(range(2, clean.ndim))
    signal_center = signal_component - signal_component.mean(dim=channel_dims, keepdim=True)
    signal_rms = signal_center.pow(2).mean(dim=channel_dims).sqrt()
    noise_rms = noise_component.pow(2).mean(dim=channel_dims).sqrt()

    measured = signal_rms / (noise_rms + 1e-8)
    target = measured.mean().new_tensor(snr_ratio)

    assert torch.allclose(measured.mean(), target, atol=5e-3, rtol=5e-2)

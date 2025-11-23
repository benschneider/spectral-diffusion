import torch

from src.training.noise import NoisePreparer
from src.training.scheduler import build_diffusion


def test_noise_preparer_from_config_prefers_diffusion_fields() -> None:
    config = {
        "diffusion": {
            "spectral_operator_mode": "radial",
            "snr_ratio": 0.75,
        },
        "spectral": {
            "operator_mode": "radial_squared",
            "snr_ratio": 2.0,
        },
    }
    preparer = NoisePreparer.from_config(config)
    assert preparer.operator_mode == "radial"
    assert abs(preparer.snr_ratio - 0.75) < 1e-6


def test_noise_statistics_match_invariants_when_unshaped() -> None:
    config = {"diffusion": {"spectral_operator_mode": "none", "snr_ratio": 1.0}}
    preparer = NoisePreparer.from_config(config)
    coeffs = build_diffusion(10, "linear")
    clean = torch.randn(4, 3, 8, 8)
    timesteps = torch.tensor([0, 1, 2, 3])
    batch = preparer.prepare(clean, coeffs, timesteps)

    stats = batch.stats
    assert abs(stats["variance_sum"] - 1.0) < 1e-3
    assert abs(stats["snr_rel"] - 1.0) < 1e-2
    assert "noise_channel_std_min" in stats and "noise_channel_std_max" in stats

    signal = batch.sqrt_alpha_t * clean
    noise = batch.noisy - signal
    dims = tuple(range(1, clean.ndim))
    signal_var = (signal - signal.mean(dim=dims, keepdim=True)).pow(2).mean(dim=dims)
    noise_var = (noise - noise.mean(dim=dims, keepdim=True)).pow(2).mean(dim=dims)
    assert torch.allclose(signal_var.mean() + noise_var.mean(), torch.tensor(1.0), atol=1e-3)


def test_snr_rel_stays_near_unity_across_timesteps() -> None:
    config = {"diffusion": {"spectral_operator_mode": "none", "snr_ratio": 1.0}}
    preparer = NoisePreparer.from_config(config)
    coeffs = build_diffusion(20, "linear")
    clean = torch.randn(6, 3, 8, 8)
    timesteps = torch.tensor([0, 3, 7, 11, 15, 19])
    batch = preparer.prepare(clean, coeffs, timesteps)
    snr_rel = batch.stats.get("snr_rel")
    assert snr_rel is not None
    assert abs(snr_rel - 1.0) < 0.1

import math

import pytest
import torch

from src.spectral.fft_adapter import add_uniform_frequency_noise


TORCH_DEVICE = torch.device("cpu")


def _run_frequency_noise(
    shape,
    *,
    snr_ratio: float = 1.0,
    fft_norm: str = "ortho",
    mode: str = "magnitude",
    phase_std: float = 0.0,
    freq_equalized_noise: bool = False,
):
    torch.manual_seed(0)
    b, c, (h, w) = 2, 3, shape
    x = torch.rand(b, c, h, w, device=TORCH_DEVICE)
    noise = torch.randn_like(x)
    sqrt_alpha = torch.tensor(0.9, device=TORCH_DEVICE).view(1, 1, 1, 1)
    sqrt_one_minus = torch.sqrt(1.0 - sqrt_alpha.pow(2))
    stats = {}
    kwargs = {"phase_std": phase_std} if mode == "phase" else {}
    x_t = add_uniform_frequency_noise(
        x,
        noise,
        sqrt_alpha_t=sqrt_alpha,
        sqrt_one_minus_alpha_t=sqrt_one_minus,
        uniform_corruption=True,
        strength=1.0,
        fft_norm=fft_norm,
        snr_ratio=snr_ratio,
        freq_equalized_noise=freq_equalized_noise,
        mode=mode,
        stats=stats,
        **kwargs,
    )
    stats["sqrt_alpha"] = float(sqrt_alpha.item())
    return x, x_t, stats


def _spatial_fft_energy(x_t, fft_norm):
    centered = x_t - x_t.mean(dim=(-2, -1), keepdim=True)
    fft = torch.fft.fftn(centered, dim=(-2, -1), norm=fft_norm)
    spatial_energy = centered.pow(2).sum().item()
    freq_energy = fft.abs().pow(2).sum().item()
    hw = x_t.shape[-2] * x_t.shape[-1]
    if fft_norm == "backward":
        freq_energy /= hw
    elif fft_norm == "forward":
        freq_energy *= hw
    rel_err = abs(spatial_energy - freq_energy) / max(abs(spatial_energy), 1e-12)
    return rel_err


@pytest.mark.parametrize("shape", [(16, 16), (28, 40), (32, 32), (48, 24)])
@pytest.mark.parametrize("snr_ratio", [0.8, 1.0, 1.4])
@pytest.mark.parametrize("fft_norm", ["ortho", "backward"])
def test_snr_targets_respected(shape, snr_ratio, fft_norm):
    x, x_t, stats = _run_frequency_noise(shape, snr_ratio=snr_ratio, fft_norm=fft_norm)

    signal_center = x - x.mean(dim=(-2, -1), keepdim=True)
    signal_rms = signal_center.pow(2).mean().sqrt().item()
    noise_rms = (x_t - x).pow(2).mean().sqrt().item()
    measured = signal_rms / max(noise_rms, 1e-8)

    assert abs(measured - snr_ratio) / snr_ratio < 0.1
    assert "snr_measured" in stats
    assert math.isfinite(stats["snr_measured"])
    assert abs(stats["snr_measured"] - snr_ratio) / snr_ratio < 0.1

    mean_val = float(x_t.mean())
    std_val = float(x_t.std())
    assert 0.4 < mean_val < 0.6
    assert 0.1 < std_val < 0.65

    rel_err = _spatial_fft_energy(x_t, fft_norm)
    assert rel_err < 1e-5


@pytest.mark.parametrize("mode", ["magnitude", "phase", "complex"])
def test_modes_share_scaler(mode):
    phase_std = 0.1 if mode == "phase" else 0.0
    x, x_t, _ = _run_frequency_noise((32, 32), mode=mode, snr_ratio=1.0, phase_std=phase_std)

    signal_center = x - x.mean(dim=(-2, -1), keepdim=True)
    signal_rms = signal_center.pow(2).mean().sqrt().item()
    noise_rms = (x_t - x).pow(2).mean().sqrt().item()
    snr_measured = signal_rms / max(noise_rms, 1e-8)

    low, high = (0.8, 1.2) if mode == "phase" else (0.9, 1.1)
    assert low <= snr_measured <= high


@pytest.mark.parametrize("fft_norm", ["ortho", "backward", "forward"])
def test_parseval_consistency_signal_and_noised(fft_norm):
    x, x_t, _ = _run_frequency_noise((32, 32), snr_ratio=1.0, fft_norm=fft_norm)

    rel_signal = _spatial_fft_energy(x, fft_norm)
    rel_noised = _spatial_fft_energy(x_t, fft_norm)
    assert rel_signal < 1e-5
    assert rel_noised < 1e-5


def test_freq_equalized_noise_increases_high_band_energy():
    x_base, x_t_base, stats_base = _run_frequency_noise((32, 32), snr_ratio=1.0)
    x_eq, x_t_eq, stats_eq = _run_frequency_noise(
        (32, 32), snr_ratio=1.0, freq_equalized_noise=True
    )
    sqrt_alpha = stats_base["sqrt_alpha"]
    base_noise = x_t_base - sqrt_alpha * x_base
    eq_noise = x_t_eq - stats_eq["sqrt_alpha"] * x_eq
    base_ratio = _band_energy_ratio(base_noise)
    eq_ratio = _band_energy_ratio(eq_noise)
    assert eq_ratio > base_ratio * 1.05


def _band_energy_ratio(noise: torch.Tensor) -> float:
    noise_centered = noise - noise.mean(dim=(-2, -1), keepdim=True)
    fft = torch.fft.fftshift(torch.fft.fftn(noise_centered, dim=(-2, -1)), dim=(-2, -1))
    magnitude = fft.abs().mean(dim=(0, 1))  # average across batch/channel
    h, w = noise.shape[-2], noise.shape[-1]
    fy = torch.fft.fftfreq(h, d=1.0)
    fx = torch.fft.fftfreq(w, d=1.0)
    yy = fy[:, None]
    xx = fx[None, :]
    radius = torch.sqrt(xx**2 + yy**2)
    radius = torch.fft.fftshift(radius)
    low_mask = radius < 0.15
    high_mask = radius > 0.35
    low_energy = magnitude[low_mask].mean().item()
    high_energy = magnitude[high_mask].mean().item()
    return high_energy / max(low_energy, 1e-6)


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
def test_return_noise_matches_residual(uniform, adaptive):
    torch.manual_seed(123)
    b, c, h, w = 2, 3, 16, 16
    x = torch.rand(b, c, h, w)
    base_noise = torch.randn_like(x)
    sqrt_alpha = torch.tensor(0.85).view(1, 1, 1, 1)
    sqrt_one_minus = torch.sqrt(1.0 - sqrt_alpha.pow(2))

    kwargs = {
        "uniform_corruption": uniform,
        "strength": 1.0,
        "fft_norm": "ortho",
        "snr_ratio": 1.0,
        "return_noise": True,
    }
    if adaptive:
        kwargs.update({"adaptive_rescale": True, "target_corr": 0.95})

    x_t, eps = add_uniform_frequency_noise(
        x,
        base_noise,
        sqrt_alpha_t=sqrt_alpha,
        sqrt_one_minus_alpha_t=sqrt_one_minus,
        **kwargs,
    )

    reconstructed = (x_t - sqrt_alpha * x) / (sqrt_one_minus + 1e-8)
    assert torch.allclose(eps, reconstructed, atol=1e-5)

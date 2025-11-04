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
    dc_scale_factor: float = 0.1,
    phase_std: float = 0.0,
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
        dc_scale_factor=dc_scale_factor,
        mode=mode,
        stats=stats,
        **kwargs,
    )
    return x, x_t, stats


def _spatial_fft_energy(x_t, fft_norm):
    fft = torch.fft.fftn(x_t, dim=(-2, -1), norm=fft_norm)
    spatial_energy = x_t.pow(2).sum().item()
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

    signal_rms = (x - 0.5).pow(2).mean().sqrt().item()
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

    rel_err = _spatial_fft_energy(x_t - 0.5, fft_norm)
    assert rel_err < 1e-5


@pytest.mark.parametrize("mode", ["magnitude", "phase", "complex"])
def test_modes_share_scaler(mode):
    phase_std = 0.1 if mode == "phase" else 0.0
    x, x_t, _ = _run_frequency_noise((32, 32), mode=mode, snr_ratio=1.0, phase_std=phase_std)

    signal_rms = (x - 0.5).pow(2).mean().sqrt().item()
    noise_rms = (x_t - x).pow(2).mean().sqrt().item()
    snr_measured = signal_rms / max(noise_rms, 1e-8)

    low, high = (0.8, 1.2) if mode == "phase" else (0.9, 1.1)
    assert low <= snr_measured <= high


def test_dc_scale_factor_controls_mean():
    factors = [0.05, 0.1, 0.2]
    outputs = []
    for factor in factors:
        x, x_t, stats = _run_frequency_noise((32, 32), snr_ratio=1.0, dc_scale_factor=factor)
        outputs.append((x, x_t, stats))

    for x, x_t, stats in outputs:
        mean_diff = abs(float(x_t.mean() - x.mean()))
        assert mean_diff < 1e-3

    for _, _, stats in outputs:
        assert "dc_scale_effective" in stats
        assert 0.0 <= stats["dc_scale_effective"] <= 1.0
        assert stats.get("dc_mean_shift", 0.0) < 1e-3


@pytest.mark.parametrize("fft_norm", ["ortho", "backward", "forward"])
def test_parseval_consistency_signal_and_noised(fft_norm):
    x, x_t, _ = _run_frequency_noise((32, 32), snr_ratio=1.0, fft_norm=fft_norm)

    rel_signal = _spatial_fft_energy(x - 0.5, fft_norm)
    rel_noised = _spatial_fft_energy(x_t - 0.5, fft_norm)
    assert rel_signal < 1e-5
    assert rel_noised < 1e-5


def test_dc_shift_tracks_noise_energy():
    x, x_t, _ = _run_frequency_noise((32, 32), snr_ratio=1.0, fft_norm="ortho")
    X = torch.fft.fftn(x - 0.5, dim=(-2, -1), norm="ortho")
    X_t = torch.fft.fftn(x_t - 0.5, dim=(-2, -1), norm="ortho")
    dc_shift = (X_t[..., 0, 0] - X[..., 0, 0]).abs().mean().item()
    total_noise_energy = (X_t - X).abs().pow(2).mean().sqrt().item()
    assert total_noise_energy > 0.0
    assert 0.01 * total_noise_energy <= dc_shift <= 1.0 * total_noise_energy

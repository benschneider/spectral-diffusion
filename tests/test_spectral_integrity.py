import math

import pytest
import torch

from src.core import build_model
from src.spectral.fft_adapter import add_uniform_frequency_noise


TORCH_DEVICE = torch.device("cpu")


def _structure_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean Pearson correlation between two batched tensors."""
    b = x.shape[0]
    x_flat = x.view(b, -1)
    y_flat = y.view(b, -1)
    x_center = x_flat - x_flat.mean(dim=1, keepdim=True)
    y_center = y_flat - y_flat.mean(dim=1, keepdim=True)
    numerator = (x_center * y_center).sum(dim=1)
    denominator = torch.sqrt(
        (x_center.pow(2).sum(dim=1) * y_center.pow(2).sum(dim=1)) + 1e-8
    )
    corr = numerator / denominator
    return float(corr.mean().item())


def _phase_rms(a: torch.Tensor, b: torch.Tensor, norm: str = "ortho") -> float:
    pa = torch.angle(torch.fft.fftn(a, dim=(-2, -1), norm=norm))
    pb = torch.angle(torch.fft.fftn(b, dim=(-2, -1), norm=norm))
    d = torch.atan2(torch.sin(pb - pa), torch.cos(pb - pa))
    return float(d.std().item())


@pytest.fixture(scope="module")
def random_batch():
    torch.manual_seed(0)
    return torch.randn(4, 3, 32, 32, device=TORCH_DEVICE)


@pytest.fixture(scope="module")
def diffusion_coeffs():
    sqrt_alpha = torch.tensor(0.95, device=TORCH_DEVICE)
    sqrt_one_minus = torch.tensor(math.sqrt(1.0 - float(sqrt_alpha) ** 2), device=TORCH_DEVICE)
    sqrt_alpha_t = sqrt_alpha.view(1, 1, 1, 1)
    sqrt_one_minus_t = sqrt_one_minus.view(1, 1, 1, 1)
    return sqrt_alpha_t, sqrt_one_minus_t


@pytest.fixture(scope="module")
def noisy_batch(random_batch, diffusion_coeffs):
    sqrt_alpha_t, sqrt_one_minus_t = diffusion_coeffs
    noise = torch.randn_like(random_batch)
    stats = {}
    x_t = add_uniform_frequency_noise(
        random_batch,
        noise,
        sqrt_alpha_t=sqrt_alpha_t,
        sqrt_one_minus_alpha_t=sqrt_one_minus_t,
        uniform_corruption=True,
        strength=0.15,
        mode="magnitude",
        fft_norm="ortho",
        stats=stats,
    )
    return x_t, stats


def test_fft_round_trip(random_batch):
    X = torch.fft.fftn(random_batch, dim=(-2, -1), norm="ortho")
    recon = torch.fft.ifftn(X, dim=(-2, -1), norm="ortho").real
    err = (random_batch - recon).abs().mean().item()
    assert err < 1e-5, f"FFT/IFFT round-trip error too large: {err:.2e}"


def test_uniform_noise_preserves_rms(random_batch, noisy_batch):
    x_t, stats = noisy_batch
    rms_in = random_batch.pow(2).mean().sqrt().item()
    rms_out = x_t.pow(2).mean().sqrt().item()
    ratio = rms_out / max(rms_in, 1e-8)
    assert 0.75 < ratio < 1.25, f"Spectral RMS changed drastically (ratio={ratio:.2f})"
    if stats:
        signal_energy = stats.get("signal_energy")
        noise_energy = stats.get("noise_energy")
        if signal_energy and noise_energy:
            assert noise_energy > 0.0


def test_uniform_noise_phase_stability(random_batch, noisy_batch):
    x_t, _ = noisy_batch
    drift = _phase_rms(random_batch, x_t, norm="ortho")
    assert drift < 0.25, f"Phase decorrelation too high ({drift:.2f})"


def test_uniform_noise_structure_correlation(random_batch, noisy_batch):
    x_t, stats = noisy_batch
    corr = _structure_correlation(random_batch, x_t)
    assert corr > 0.9, f"Structure correlation dropped too low ({corr:.2f})"
    if "structure_corr_post" in stats:
        assert stats["structure_corr_post"] > 0.9


def test_uniform_noise_fft_variance(random_batch, noisy_batch):
    x_t, _ = noisy_batch
    mag_in = torch.log1p(torch.fft.fftn(random_batch, dim=(-2, -1), norm="ortho").abs())
    mag_out = torch.log1p(torch.fft.fftn(x_t, dim=(-2, -1), norm="ortho").abs())
    var_ratio = mag_out.var() / (mag_in.var() + 1e-8)
    assert 0.8 < float(var_ratio) < 1.25, f"FFT magnitude variance out of range ({float(var_ratio):.2f})"


def test_snr_ratio_scaling(random_batch):
    snr_target = 1.4
    strength = 0.2
    batch = random_batch.to(TORCH_DEVICE)
    B, _, H, W = batch.shape
    sqrt_alpha = torch.tensor(0.9, device=TORCH_DEVICE)
    sqrt_one_minus = torch.sqrt(1.0 - sqrt_alpha.pow(2))
    sqrt_alpha_t = sqrt_alpha.view(1, 1, 1, 1).repeat(B, 1, 1, 1)
    sqrt_one_minus_t = sqrt_one_minus.view(1, 1, 1, 1).repeat(B, 1, 1, 1)
    noise = torch.randn_like(batch)
    stats = {}
    x_t = add_uniform_frequency_noise(
        batch,
        noise,
        sqrt_alpha_t=sqrt_alpha_t,
        sqrt_one_minus_alpha_t=sqrt_one_minus_t,
        uniform_corruption=True,
        strength=strength,
        mode="complex",
        fft_norm="ortho",
        stats=stats,
        snr_ratio=snr_target,
    )

    assert stats.get("snr_ratio") == pytest.approx(snr_target, rel=1e-5)
    x_fft = torch.fft.fftn(batch - 0.5, dim=(-2, -1), norm="ortho")
    x_t_fft = torch.fft.fftn(x_t - 0.5, dim=(-2, -1), norm="ortho")

    strength_scaled = strength
    residual_fft = x_t_fft - x_fft * sqrt_alpha_t
    scale_factor = stats.get("snr_scale_factor", 1.0)
    mix = sqrt_one_minus_t * strength_scaled * scale_factor
    noise_fft = residual_fft / mix

    signal_rms = torch.sqrt(x_fft.abs().pow(2).mean(dim=(-3, -2, -1)))
    noise_rms = torch.sqrt(noise_fft.abs().pow(2).mean(dim=(-3, -2, -1)))
    ratios = signal_rms / (noise_rms + 1e-8)
    for ratio in ratios:
        value = float(ratio.item())
        assert 0.5 * snr_target <= value <= 2.0 * snr_target, (
            f"SNR ratio {value:.2f} outside tolerance for target {snr_target}"
        )


def test_model_spectral_amplification(random_batch):
    model_cfg = {
        "type": "unet_spectral",
        "channels": 3,
        "base_channels": 16,
        "enable_phase_attention": False,
        "enable_amp_residual": False,
        "amp_hidden_dim": 16,
        "diffusion": {"time_embed_dim": 64},
        "data": {"channels": 3},
    }
    torch.manual_seed(0)
    model = build_model(model_cfg).to(TORCH_DEVICE)
    model.eval()
    t = torch.zeros(random_batch.shape[0], dtype=torch.long, device=TORCH_DEVICE)
    with torch.no_grad():
        out = model(random_batch, t)

    assert torch.isfinite(out).all(), "Model produced NaN or Inf outputs."

    fft_in = torch.fft.fftn(random_batch, dim=(-2, -1), norm="ortho").abs().mean().item()
    fft_out = torch.fft.fftn(out, dim=(-2, -1), norm="ortho").abs().mean().item()
    ratio = fft_out / max(fft_in, 1e-8)
    assert 0.35 < ratio < 2.0, f"Model amplifies spectrum excessively (ratio={ratio:.2f})"

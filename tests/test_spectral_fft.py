import math

import torch

from src.spectral.fft_adapter import add_uniform_frequency_noise
from src.spectral.fft_utils import (
    apply_weight_map,
    fft_transform,
    inverse_fft_transform,
)


def test_fft_round_trip_identity():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 16, 16)
    x_fft = fft_transform(x)
    x_rec = inverse_fft_transform(x_fft)
    assert torch.allclose(x, x_rec, atol=1e-6, rtol=1e-6), (
        f"Round-trip mismatch: max diff {torch.max(torch.abs(x - x_rec))}"
    )


def test_fft_round_trip_without_normalization():
    torch.manual_seed(1)
    x = torch.randn(1, 1, 8, 8)
    x_fft = fft_transform(x, normalize=False)
    x_rec = inverse_fft_transform(x_fft, normalize=False)
    assert torch.allclose(x, x_rec, atol=1e-6, rtol=1e-6)


def test_apply_weight_map_radial_behavior():
    torch.manual_seed(2)
    x = torch.randn(1, 1, 8, 8)
    x_fft = fft_transform(x)

    # Construct a simple radial-like weight that zeros out half the spectrum.
    weight = torch.zeros(8, 8)
    weight[:4, :4] = 1.0  # keep low-frequency quadrant
    weighted_fft = apply_weight_map(x_fft, weight)

    # Ensure masked frequencies are zeroed out
    masked_region = weighted_fft[..., 4:, 4:, :]
    assert masked_region.abs().sum() == 0


def test_apply_weight_map_broadcasting():
    torch.manual_seed(3)
    x = torch.randn(1, 3, 8, 8)
    x_fft = fft_transform(x)
    weight = torch.linspace(0, 1, steps=8).unsqueeze(0).repeat(8, 1)
    weighted = apply_weight_map(x_fft, weight)

    # Weight map should broadcast across channels without error
    assert weighted.shape == x_fft.shape


def test_fft_is_linear():
    torch.manual_seed(4)
    x = torch.randn(1, 1, 8, 8)
    y = torch.randn(1, 1, 8, 8)
    alpha = torch.tensor(1.2345)
    fft_x = fft_transform(x)
    fft_y = fft_transform(y)
    fft_combined = fft_transform(alpha * x + y)
    assert torch.allclose(fft_combined, alpha * fft_x + fft_y, atol=1e-6, rtol=1e-6)


def test_uniform_frequency_noise_changes_distribution():
    torch.manual_seed(5)
    x0 = torch.randn(2, 3, 8, 8)
    noise = torch.randn_like(x0)
    sqrt_alpha = torch.full((2, 1, 1, 1), 0.9)
    sqrt_one_minus = torch.full((2, 1, 1, 1), 0.1)

    standard = add_uniform_frequency_noise(
        x0, noise, sqrt_alpha, sqrt_one_minus, uniform_corruption=False
    )
    uniform = add_uniform_frequency_noise(
        x0, noise, sqrt_alpha, sqrt_one_minus, uniform_corruption=True
    )

    assert standard.shape == x0.shape
    assert uniform.shape == x0.shape
    assert torch.isfinite(uniform).all()
    assert not torch.allclose(standard, uniform)

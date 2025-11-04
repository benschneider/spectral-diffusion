import pytest
import torch

from src.spectral.fft_adapter import add_uniform_frequency_noise


TORCH_DEVICE = torch.device("cpu")


@pytest.mark.parametrize("size", [16, 32, 64])
@pytest.mark.parametrize("snr_target", [0.8, 1.0, 1.4])
@pytest.mark.parametrize("fft_norm", ["ortho", "backward"])
def test_fft_noise_scaling_accuracy(size, snr_target, fft_norm):
    torch.manual_seed(0)
    batch_size, channels, height, width = 2, 3, size, size
    x = torch.rand(batch_size, channels, height, width, device=TORCH_DEVICE)
    noise = torch.randn_like(x)

    sqrt_alpha = torch.tensor(0.9, device=TORCH_DEVICE)
    sqrt_one_minus = torch.sqrt(1.0 - sqrt_alpha.pow(2))
    sqrt_alpha_t = sqrt_alpha.view(1, 1, 1, 1)
    sqrt_one_minus_t = sqrt_one_minus.view(1, 1, 1, 1)

    x_t = add_uniform_frequency_noise(
        x,
        noise,
        sqrt_alpha_t=sqrt_alpha_t,
        sqrt_one_minus_alpha_t=sqrt_one_minus_t,
        uniform_corruption=True,
        strength=1.0,
        fft_norm=fft_norm,
        snr_ratio=snr_target,
    )

    X_t = torch.fft.fftn(x_t, dim=(-2, -1), norm=fft_norm)
    signal_rms = (x - 0.5).pow(2).mean().sqrt().item()
    noise_rms = (x_t - x).pow(2).mean().sqrt().item()
    measured = signal_rms / max(noise_rms, 1e-8)

    assert abs(measured - snr_target) / snr_target < 0.1, (
        f"SNR mismatch: target={snr_target}, measured={measured:.3f}"
    )

    mean_val = float(x_t.mean())
    std_val = float(x_t.std())
    assert 0.4 < mean_val < 0.6, f"mean drifted: {mean_val:.3f}"
    assert 0.1 < std_val < 0.65, f"std out of range: {std_val:.3f}"

    spatial_energy = x_t.pow(2).sum().item()
    freq_energy = X_t.abs().pow(2).sum().item()
    if fft_norm == "backward":
        freq_energy /= (size * size)
    elif fft_norm == "forward":
        freq_energy *= (size * size)
    if spatial_energy != 0.0:
        rel_err = abs(spatial_energy - freq_energy) / abs(spatial_energy)
    else:
        rel_err = abs(freq_energy)
    assert rel_err < 1e-5


def test_fft_scaling_parseval_consistency():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 32, device=TORCH_DEVICE)
    X = torch.fft.fftn(x, dim=(-2, -1), norm="ortho")
    spatial_energy = x.pow(2).sum().item()
    freq_energy = X.abs().pow(2).sum().item()
    assert abs(spatial_energy - freq_energy) / spatial_energy < 1e-5, "Parseval energy mismatch"


def test_dc_component_preserved():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 32, 32, device=TORCH_DEVICE)
    noise = torch.randn_like(x)

    sqrt_alpha = torch.tensor(0.9, device=TORCH_DEVICE)
    sqrt_one_minus = torch.sqrt(1.0 - sqrt_alpha.pow(2))
    sqrt_alpha_t = sqrt_alpha.view(1, 1, 1, 1)
    sqrt_one_minus_t = sqrt_one_minus.view(1, 1, 1, 1)

    x_t = add_uniform_frequency_noise(
        x,
        noise,
        sqrt_alpha_t=sqrt_alpha_t,
        sqrt_one_minus_alpha_t=sqrt_one_minus_t,
        uniform_corruption=True,
        fft_norm="ortho",
        snr_ratio=1.0,
    )

    mean_diff = abs(float(x_t.mean() - x.mean()))
    assert mean_diff < 0.05, f"Mean brightness drift too high ({mean_diff:.3f})"


def test_dc_scale_factor_effect():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 32, 32, device=TORCH_DEVICE)
    noise = torch.randn_like(x)

    sqrt_alpha = torch.tensor(0.9, device=TORCH_DEVICE)
    sqrt_one_minus = torch.sqrt(1.0 - sqrt_alpha.pow(2))
    sqrt_alpha_t = sqrt_alpha.view(1, 1, 1, 1)
    sqrt_one_minus_t = sqrt_one_minus.view(1, 1, 1, 1)

    means = []
    for factor in [0.05, 0.1, 0.2]:
        x_t = add_uniform_frequency_noise(
            x,
            noise,
            sqrt_alpha_t=sqrt_alpha_t,
            sqrt_one_minus_alpha_t=sqrt_one_minus_t,
            uniform_corruption=True,
            fft_norm="ortho",
            snr_ratio=1.0,
            dc_scale_factor=factor,
        )
        means.append(float(x_t.mean()))

    assert means[0] < means[1] < means[2] or means[0] > means[1] > means[2], "DC scaling not monotonic"


@pytest.mark.parametrize("shape", [(16, 16), (28, 40), (32, 32), (48, 24)])
@pytest.mark.parametrize("snr_target", [0.8, 1.0, 1.4])
@pytest.mark.parametrize("fft_norm", ["ortho", "backward"])
def test_fft_noise_scaling_nonsquare(shape, snr_target, fft_norm):
    H, W = shape
    torch.manual_seed(0)
    x = torch.rand(2, 3, H, W, device=TORCH_DEVICE)
    noise = torch.randn_like(x)

    sqrt_alpha = torch.tensor(0.9, device=TORCH_DEVICE).view(1, 1, 1, 1)
    sqrt_one_minus = torch.sqrt(1.0 - sqrt_alpha.pow(2))

    x_t = add_uniform_frequency_noise(
        x,
        noise,
        sqrt_alpha_t=sqrt_alpha,
        sqrt_one_minus_alpha_t=sqrt_one_minus,
        uniform_corruption=True,
        fft_norm=fft_norm,
        snr_ratio=snr_target,
    )

    X_t = torch.fft.fftn(x_t, dim=(-2, -1), norm=fft_norm)
    signal_rms = (x - 0.5).pow(2).mean().sqrt().item()
    noise_rms = (x_t - x).pow(2).mean().sqrt().item()
    measured = signal_rms / max(noise_rms, 1e-8)
    assert abs(measured - snr_target) / snr_target < 0.1
    assert 0.4 < float(x_t.mean()) < 0.6
    spatial_energy = x_t.pow(2).sum().item()
    freq_energy = X_t.abs().pow(2).sum().item()
    if fft_norm == "backward":
        freq_energy /= (H * W)
    elif fft_norm == "forward":
        freq_energy *= (H * W)
    if spatial_energy != 0.0:
        rel_err = abs(spatial_energy - freq_energy) / abs(spatial_energy)
    else:
        rel_err = abs(freq_energy)
    assert rel_err < 1e-5


@pytest.mark.parametrize("mode", ["magnitude", "phase", "complex"])
def test_modes_share_scaler(mode):
    torch.manual_seed(1)
    x = torch.rand(1, 3, 32, 32, device=TORCH_DEVICE)
    noise = torch.randn_like(x)

    sqrt_alpha = torch.tensor(0.9, device=TORCH_DEVICE).view(1, 1, 1, 1)
    sqrt_one_minus = torch.sqrt(1.0 - sqrt_alpha.pow(2))

    kwargs = {}
    if mode == "phase":
        kwargs["phase_std"] = 0.1
    x_t = add_uniform_frequency_noise(
        x,
        noise,
        sqrt_alpha_t=sqrt_alpha,
        sqrt_one_minus_alpha_t=sqrt_one_minus,
        uniform_corruption=True,
        fft_norm="ortho",
        snr_ratio=1.0,
        dc_scale_factor=0.1,
        mode=mode,
        **kwargs,
    )

    signal_rms = (x - 0.5).pow(2).mean().sqrt().item()
    noise_rms = (x_t - x).pow(2).mean().sqrt().item()
    snr_measured = signal_rms / max(noise_rms, 1e-8)
    if mode == "phase":
        low, high = 0.8, 1.2
    else:
        low, high = 0.9, 1.1
    assert low <= float(snr_measured) <= high


def test_dc_follows_noise_energy():
    torch.manual_seed(0)
    x = torch.rand(1, 3, 32, 32, device=TORCH_DEVICE)
    noise = torch.randn_like(x)
    sqrt_alpha = torch.tensor(0.9, device=TORCH_DEVICE).view(1, 1, 1, 1)
    sqrt_one_minus = torch.sqrt(1.0 - sqrt_alpha.pow(2))

    x_t = add_uniform_frequency_noise(
        x,
        noise,
        sqrt_alpha_t=sqrt_alpha,
        sqrt_one_minus_alpha_t=sqrt_one_minus,
        uniform_corruption=True,
        fft_norm="ortho",
        snr_ratio=1.0,
        dc_scale_factor=0.1,
    )

    X = torch.fft.fftn(x - 0.5, dim=(-2, -1), norm="ortho")
    X_t = torch.fft.fftn(x_t - 0.5, dim=(-2, -1), norm="ortho")
    dc_shift = (X_t[..., 0, 0] - X[..., 0, 0]).abs().mean().item()
    total_noise_energy = (X_t - X).abs().pow(2).mean().sqrt().item()
    assert total_noise_energy > 0.0
    assert 0.01 * total_noise_energy <= dc_shift <= 1.0 * total_noise_energy

import torch


def test_fft_reconstruction_integrity(cifar_batch):
    """FFT followed by IFFT should reconstruct the spatial input within tolerance."""
    Xf = torch.fft.fft2(cifar_batch, norm="ortho")
    x_rec = torch.fft.ifft2(Xf, norm="ortho").real
    err = (cifar_batch - x_rec).abs().mean().item()
    assert err < 1e-3, f"FFT reconstruction error too high: {err:.2e}"


def test_fft_channel_consistency(cifar_batch):
    """FFT complex split should double the channel count (real + imaginary)."""
    Xf = torch.fft.fft2(cifar_batch, norm="ortho")
    stacked = torch.cat([Xf.real, Xf.imag], dim=1)
    expected_channels = cifar_batch.shape[1] * 2
    assert (
        stacked.shape[1] == expected_channels
    ), f"FFT channel doubling mismatch: expected {expected_channels}, got {stacked.shape[1]}"

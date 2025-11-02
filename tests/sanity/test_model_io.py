import torch


def _reconstruct_from_fft(stacked: torch.Tensor) -> torch.Tensor:
    """Convert stacked real/imag channels back to spatial domain."""
    channels = stacked.shape[1] // 2
    real = stacked[:, :channels]
    imag = stacked[:, channels:]
    complex_fft = torch.complex(real, imag)
    spatial = torch.fft.ifft2(complex_fft, norm="ortho").real
    return spatial


def test_model_input_output_shapes(spectral_model, cifar_batch):
    """Spectral model should preserve batch and spatial dimensions."""
    spectral_model.eval()
    with torch.no_grad():
        out = spectral_model(cifar_batch)
    assert out.shape[0] == cifar_batch.shape[0], "Batch dimension changed by spectral model."
    assert (
        out.shape[2:] == cifar_batch.shape[2:]
    ), f"Spatial dimensions mismatch: {out.shape[2:]} vs {cifar_batch.shape[2:]}"


def test_model_accepts_complex_input(spectral_model, cifar_batch):
    """
    Simulate the FFT pipeline (real/imag stacking) and ensure the model
    processes the reconstructed spatial tensor without error.
    """
    spectral_model.eval()
    Xf = torch.fft.fft2(cifar_batch, norm="ortho")
    stacked = torch.cat([Xf.real, Xf.imag], dim=1)
    spatial = _reconstruct_from_fft(stacked)
    with torch.no_grad():
        out = spectral_model(spatial)
    assert out.shape == cifar_batch.shape, "Spectral model failed to process FFT reconstructed input."

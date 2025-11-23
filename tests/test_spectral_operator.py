import torch

from src.spectral.operator import spectral_operator


def _per_sample_rms(tensor: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, tensor.ndim))
    return tensor.pow(2).mean(dim=dims).sqrt()


def test_spectral_operator_identity_mode_normalises_noise() -> None:
    torch.manual_seed(0)
    eps = torch.randn(3, 2, 8, 8)
    shaped = spectral_operator(eps, mode="none")
    rms = _per_sample_rms(shaped)
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)
    expected = eps / _per_sample_rms(eps).view(-1, 1, 1, 1)
    assert torch.allclose(shaped, expected, atol=1e-4)


def test_spectral_operator_modes_preserve_rms() -> None:
    torch.manual_seed(1)
    eps = torch.randn(4, 3, 16, 16)
    for mode in ("radial", "radial_squared"):
        shaped = spectral_operator(eps, mode=mode)
        rms = _per_sample_rms(shaped)
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

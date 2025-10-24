import torch

from src.core.model_unet_spectral import SpectralUNet


def _config() -> dict:
    return {
        "channels": 3,
        "base_channels": 8,
        "diffusion": {"time_embed_dim": 32},
        "data": {"channels": 3, "height": 16, "width": 16},
    }


def test_spectral_unet_forward_matches_input_shape():
    model = SpectralUNet(_config())
    x = torch.randn(2, 3, 16, 16)
    t = torch.randint(0, 10, (2,))
    out = model(x, t)
    assert out.shape == x.shape


def test_spectral_unet_backward_pass():
    model = SpectralUNet(_config())
    x = torch.randn(1, 3, 16, 16, requires_grad=True)
    t = torch.randint(0, 10, (1,))
    out = model(x, t)
    out.sum().backward()
    assert x.grad is not None

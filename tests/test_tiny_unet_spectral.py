import torch

from src.core.model_unet_tiny import TinyUNet


def _build_config(enabled: bool, weighting: str = "none"):
    return {
        "channels": 3,
        "base_channels": 8,
        "depth": 1,
        "data": {"channels": 3, "height": 16, "width": 16},
        "spectral": {
            "enabled": enabled,
            "normalize": True,
            "weighting": weighting,
        },
    }


def test_tiny_unet_forward_shape_matches_input():
    config = _build_config(enabled=False)
    model = TinyUNet(config)
    x = torch.randn(2, 3, 16, 16)
    out = model(x)
    assert out.shape == x.shape
    assert model._spectral_calls == 0


def test_tiny_unet_spectral_toggle_identity():
    config = _build_config(enabled=True, weighting="none")
    model = TinyUNet(config)
    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        out = model._apply_spectral_roundtrip(x)
    assert torch.allclose(out, x, atol=1e-5)
    assert model._spectral_calls >= 1
    stats = model.spectral_stats()
    assert stats["spectral_calls"] >= 1


def test_tiny_unet_spectral_weighting_changes_tensor():
    config = _build_config(enabled=True, weighting="radial")
    model = TinyUNet(config)
    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        freq = model._apply_spectral_roundtrip(x)
    diff = (freq - x).abs().mean()
    assert diff > 1e-5
    assert model._spectral_calls >= 1

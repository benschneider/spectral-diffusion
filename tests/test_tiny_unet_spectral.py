import torch

from src.core.model_unet_tiny import TinyUNet


def _build_config(enabled: bool, weighting: str = "none", apply_to=None, per_block: bool = False):
    if apply_to is None:
        apply_to = ["input"]
    return {
        "channels": 3,
        "base_channels": 8,
        "depth": 1,
        "data": {"channels": 3, "height": 16, "width": 16},
        "spectral": {
            "enabled": enabled,
            "normalize": True,
            "weighting": weighting,
            "apply_to": apply_to,
            "per_block": per_block,
            "learnable": False,
        },
    }


def test_tiny_unet_forward_shape_matches_input():
    config = _build_config(enabled=False)
    model = TinyUNet(config)
    x = torch.randn(2, 3, 16, 16)
    out = model(x)
    assert out.shape == x.shape
    assert model.spectral_stats()["spectral_calls"] == 0


def test_tiny_unet_spectral_toggle_identity():
    config = _build_config(enabled=True, weighting="none")
    model = TinyUNet(config)
    x = torch.randn(1, 3, 16, 16)
    adapter = model.spectral_input
    assert adapter is not None
    with torch.no_grad():
        out = adapter(x)
    assert torch.allclose(out, x, atol=1e-5)
    stats = model.spectral_stats()
    assert stats["spectral_calls"] >= 1


def test_tiny_unet_spectral_weighting_changes_tensor():
    config = _build_config(enabled=True, weighting="radial")
    model = TinyUNet(config)
    x = torch.randn(1, 3, 16, 16)
    adapter = model.spectral_input
    assert adapter is not None
    with torch.no_grad():
        freq = adapter(x)
    diff = (freq - x).abs().mean()
    assert diff > 1e-5
    stats = model.spectral_stats()
    assert stats["spectral_calls"] >= 1


def test_tiny_unet_per_block_adapter_runs():
    config = _build_config(enabled=True, weighting="radial", per_block=True, apply_to=["input"])
    model = TinyUNet(config)
    x = torch.randn(1, 3, 16, 16)
    out = model(x)
    assert out.shape == x.shape
    stats = model.spectral_stats()
    assert stats["spectral_calls"] >= 1


def test_tiny_unet_learnable_output_adapter_backward():
    config = _build_config(enabled=True, weighting="bandpass", apply_to=["output"], per_block=False)
    config["spectral"]["learnable"] = True
    config["spectral"]["condition"] = "time"
    model = TinyUNet(config)
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    t = torch.randint(0, 1000, (2,), dtype=torch.long)
    out = model(x, t)
    out.mean().backward()
    assert any(
        (p.grad is not None) for p in model.spectral_output.parameters() if p.requires_grad
    )

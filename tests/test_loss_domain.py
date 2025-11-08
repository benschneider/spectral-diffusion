import pytest

torch = pytest.importorskip("torch")

from src.core.losses import DiffusionLoss
from src.spectral.adapter import SpectralAdapter


def test_diffusion_loss_default_matches_spatial_mse():
    residual = torch.randn(2, 3, 16, 16, dtype=torch.float32)
    loss_fn = DiffusionLoss(config={})
    loss = loss_fn(residual.clone())
    expected = residual.pow(2).mean()
    assert torch.allclose(loss, expected)


def test_diffusion_loss_spectral_weighting_matches_adapter_mse():
    residual = torch.randn(2, 3, 16, 16, dtype=torch.float32, requires_grad=True)
    loss_fn = DiffusionLoss(config={"spectral_weighting": "radial"})
    loss = loss_fn(residual)
    loss.backward()
    grad_from_loss = residual.grad.detach().clone()

    residual_manual = residual.detach().clone().requires_grad_()
    adapter = SpectralAdapter(
        enabled=True,
        weighting="radial",
        normalize=True,
        bandpass_inner=loss_fn.config.get("bandpass_inner", 0.1),
        bandpass_outer=loss_fn.config.get("bandpass_outer", 0.6),
    )
    filtered = adapter(residual_manual)
    manual_loss = filtered.pow(2).mean()
    manual_loss.backward()

    assert torch.allclose(loss.detach(), manual_loss.detach(), atol=1e-6, rtol=1e-5)
    assert torch.allclose(grad_from_loss, residual_manual.grad, atol=1e-6, rtol=1e-5)

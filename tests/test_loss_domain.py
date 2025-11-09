import pytest

torch = pytest.importorskip("torch")

from src.core.losses import DiffusionLoss
from src.spectral.adapter import SpectralAdapter


def _coef(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.full((batch, 1, 1, 1), 0.85, dtype=torch.float32),
        torch.full((batch, 1, 1, 1), 0.5, dtype=torch.float32),
    )


def test_diffusion_loss_default_matches_spatial_mse():
    prediction = torch.randn(2, 3, 16, 16, dtype=torch.float32)
    target = torch.randn_like(prediction)
    sqrt_alpha, sqrt_one_minus = _coef(prediction.shape[0])
    loss_fn = DiffusionLoss(config={"use_weighting": False})
    loss, _ = loss_fn(prediction, target, sqrt_alpha, sqrt_one_minus)
    expected = (prediction - target).pow(2).mean()
    assert torch.allclose(loss, expected)


def test_diffusion_loss_spectral_weighting_matches_adapter_mse():
    prediction = torch.randn(2, 3, 16, 16, dtype=torch.float32, requires_grad=True)
    target = torch.randn_like(prediction)
    sqrt_alpha, sqrt_one_minus = _coef(prediction.shape[0])

    loss_fn = DiffusionLoss(config={"spectral_weighting": "radial", "use_weighting": False})
    loss, _ = loss_fn(prediction, target, sqrt_alpha, sqrt_one_minus)
    loss.backward()
    grad_from_loss = prediction.grad.detach().clone()

    prediction_manual = prediction.detach().clone().requires_grad_()
    residual = target.detach() - prediction_manual
    adapter = SpectralAdapter(
        enabled=True,
        weighting="radial",
        normalize=True,
        bandpass_inner=loss_fn.config.get("bandpass_inner", 0.1),
        bandpass_outer=loss_fn.config.get("bandpass_outer", 0.6),
    )
    filtered = adapter(residual)
    manual_loss = filtered.pow(2).mean()
    manual_loss.backward()

    assert torch.allclose(loss.detach(), manual_loss.detach(), atol=1e-6, rtol=1e-5)
    assert torch.allclose(grad_from_loss, prediction_manual.grad, atol=1e-6, rtol=1e-5)

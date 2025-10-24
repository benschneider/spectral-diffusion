import torch

from src.core.losses import DiffusionLoss


def test_diffusion_loss_no_weighting_matches_mse():
    residual = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    loss = DiffusionLoss({"spectral_weighting": "none", "reduction": "mean"})
    mse = (residual**2).mean()
    assert torch.allclose(loss(residual), mse)


def test_diffusion_loss_with_weighting_changes_value():
    residual = torch.randn(2, 3, 8, 8)
    loss_none = DiffusionLoss({"spectral_weighting": "none", "reduction": "mean"})
    baseline = loss_none(residual)

    loss_radial = DiffusionLoss({"spectral_weighting": "radial", "reduction": "mean"})
    weighted = loss_radial(residual)
    assert not torch.allclose(baseline, weighted)

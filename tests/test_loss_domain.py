import pytest

torch = pytest.importorskip("torch")

from src.core.losses import DiffusionLoss


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


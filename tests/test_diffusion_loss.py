import torch

from src.core.losses import DiffusionLoss


def _coef(shape):
    return (
        torch.full((shape[0], 1, 1, 1), 0.9),
        torch.full((shape[0], 1, 1, 1), 0.4),
    )


def test_diffusion_loss_no_weighting_matches_mse():
    prediction = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float32).unsqueeze(0)
    target = torch.zeros_like(prediction)
    sqrt_alpha, sqrt_one_minus = _coef(prediction.shape)
    loss_fn = DiffusionLoss({"log_snr_weighting": False})
    loss, _ = loss_fn(
        prediction,
        target,
        sqrt_alpha,
        sqrt_one_minus,
        x_t=target,  # dummy noisy input not used when clip not exceeded
        x0=target,
    )
    expected = (prediction - target).pow(2).mean()
    assert torch.allclose(loss, expected)


def test_diffusion_loss_with_weighting_changes_value():
    prediction = torch.randn(2, 3, 8, 8)
    target = torch.randn_like(prediction)
    sqrt_alpha, sqrt_one_minus = _coef(prediction.shape)

    loss_none = DiffusionLoss({"log_snr_weighting": False})
    baseline, _ = loss_none(
        prediction,
        target,
        sqrt_alpha,
        sqrt_one_minus,
        x_t=target,
        x0=target,
    )

    loss_weighted = DiffusionLoss({"log_snr_weighting": True})
    weighted, diag = loss_weighted(
        prediction,
        target,
        sqrt_alpha,
        sqrt_one_minus,
        x_t=target,
        x0=target,
        snr_rel=torch.tensor([0.5, 2.0]).view(2, 1, 1, 1),
    )

    assert not torch.allclose(baseline, weighted)
    assert "snr_weight_mean" in diag

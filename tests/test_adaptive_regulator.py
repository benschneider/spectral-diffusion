import pytest

torch = pytest.importorskip("torch")

from src.training.regulators import AdaptiveSNRGovernor, MicroResetPolicy


def test_adaptive_snr_governor_micro_reset_metrics():
    governor = AdaptiveSNRGovernor(min_ratio=0.5, max_ratio=2.5)
    governor._micro_reset = MicroResetPolicy(period=1, kappa_scale=1.4, overflow_scale=0.4)  # pylint: disable=protected-access

    snr_raw = torch.full((4,), 1.0)
    adaptive_diag = {"kappa": 0.5, "ema": 0.3, "overflow": 0.1, "overflow_ema": 0.4}
    predicted_noise = torch.randn(4, 2)
    true_noise = torch.randn(4, 2)

    update = governor.update(
        loss=0.5,
        grad_norm=1.0,
        snr_raw=snr_raw,
        snr_clamped=snr_raw,
        adaptive_diag=adaptive_diag,
        predicted_noise=predicted_noise,
        true_noise=true_noise,
    )

    metrics = update.metrics
    assert update.ratio > 0.0
    assert metrics["micro_reset"] == 1.0
    assert metrics["kappa"] == pytest.approx(0.5 * 1.4)
    assert metrics["overflow_ema"] == pytest.approx(governor.metrics.overflow_ema)
    assert "[SNR-GOV]" in update.log_message
    assert metrics["lambda_var"] == pytest.approx(governor.lambda_var)
    assert "variance_ratio_raw" in metrics

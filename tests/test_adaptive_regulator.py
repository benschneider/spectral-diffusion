import pytest

torch = pytest.importorskip("torch")

from src.training.regulators import AdaptiveSNRController, MicroResetPolicy


def test_adaptive_snr_controller_micro_reset_metrics():
    controller = AdaptiveSNRController(
        min_snr=0.5,
        max_snr=2.5,
        inc=1.1,
        dec=0.9,
        kappa_thresh=0.8,
        alpha_fac_high=1.25,
        overflow_high=0.3,
    )
    controller._micro_reset = MicroResetPolicy(period=1, kappa_scale=1.4, overflow_scale=0.4)  # pylint: disable=protected-access

    snr_vals = torch.full((4,), 1.0)
    fft_feedback = {"amplitude_high_mae": 0.3, "amplitude_mid_mae": 0.2}
    adaptive_diag = {"kappa": 0.5, "ema": 0.3, "overflow": 0.1, "overflow_ema": 0.4}

    ratio, note = controller.update(
        loss=0.5,
        grad_norm=1.0,
        fft_feedback=fft_feedback,
        adaptive_diag=adaptive_diag,
        snr_vals=snr_vals,
    )

    metrics = controller.latest_metrics
    assert ratio > 0.0
    assert metrics["micro_reset"] == 1.0
    assert metrics["kappa"] == pytest.approx(0.5 * 1.4)
    assert metrics["overflow_ema"] == pytest.approx(0.21 * 0.4)
    assert "micro_reset" in note

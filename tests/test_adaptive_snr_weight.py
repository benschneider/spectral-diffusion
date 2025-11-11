import pytest

torch = pytest.importorskip("torch")

from src.core.residuals import AdaptiveSNRWeight
from src.training.regulators import MicroResetPolicy


def test_adaptive_snr_weight_maintains_fp32_state_and_floor():
    adaptive = AdaptiveSNRWeight(
        beta=0.5,
        ema_decay=0.0,
        kappa_floor=1e-3,
        log_interval=0,
        change_threshold=0.0,
        snr_clip=100.0,
        delta=1e-3,
    )

    snr = torch.full((2,), 10.0, dtype=torch.float16)
    raw_loss = torch.full((2, 1, 4, 4), 0.1, dtype=torch.float16)
    alpha = torch.full((2,), 0.5, dtype=torch.float16)

    weight, diag = adaptive.update(snr, raw_loss, alpha)

    assert weight.dtype == raw_loss.dtype
    assert adaptive._ema_val.dtype == torch.float32  # pylint: disable=protected-access
    assert diag["kappa"] >= 1e-3
    assert diag["mean_weight"] == pytest.approx(1.0, rel=0.2)
    assert 0.0 <= diag["overflow"] <= 1.0
    assert "alpha_fac" in diag
    assert "overflow_ema" in diag
    assert diag["delta"] >= 1e-3


def test_adaptive_snr_weight_handles_overflow_and_delta_growth():
    adaptive = AdaptiveSNRWeight(
        snr_clip=100.0,
        log_interval=0,
        change_threshold=0.0,
        delta=1e-3,
        overflow_target=0.05,
        delta_growth=2.0,
    )

    snr = torch.tensor([150.0, 10.0])
    raw_loss = torch.ones((2, 1, 2, 2))
    alpha = torch.tensor([0.99, 0.5])

    _, diag = adaptive.update(snr, raw_loss, alpha)

    assert diag["overflow"] > 0.0
    assert diag["mean_weight"] == pytest.approx(1.0, rel=0.2)
    assert diag["delta"] >= adaptive._delta_base  # pylint: disable=protected-access
    assert diag["overflow_ema"] > 0.0


def test_adaptive_snr_weight_logs_periodically():
    adaptive = AdaptiveSNRWeight(log_interval=2, change_threshold=10.0)

    snr = torch.ones(1)
    raw_loss = torch.ones((1, 1, 2, 2))
    alpha = torch.full((1,), 0.5)

    events = []
    for _ in range(3):
        _, diag = adaptive.update(snr, raw_loss, alpha)
        events.append(diag["log_event"])

    assert events[0] is True
    assert events[1] is False
    assert events[2] is True


def test_adaptive_snr_weight_micro_reset_scaling():
    adaptive = AdaptiveSNRWeight()
    adaptive._micro_reset = MicroResetPolicy(period=2, kappa_scale=1.5, overflow_scale=0.25)  # pylint: disable=protected-access

    snr = torch.ones(1)
    raw_loss = torch.ones((1, 1, 2, 2))
    alpha = torch.full((1,), 0.5)
    overflow_mask = torch.ones_like(raw_loss)

    _, first_diag = adaptive.update(snr, raw_loss, alpha, overflow_mask=overflow_mask)
    overflow_before = adaptive._overflow_ema  # pylint: disable=protected-access

    _, second_diag = adaptive.update(snr, raw_loss, alpha, overflow_mask=overflow_mask)

    assert first_diag["micro_reset"] == 0.0
    assert second_diag["micro_reset"] == 1.0
    expected_overflow = (
        adaptive.overflow_decay * overflow_before + (1.0 - adaptive.overflow_decay) * 1.0
    ) * 0.25
    assert second_diag["overflow_ema"] == pytest.approx(expected_overflow)
    assert second_diag["kappa"] >= first_diag["kappa"] * 1.5 - 1e-6

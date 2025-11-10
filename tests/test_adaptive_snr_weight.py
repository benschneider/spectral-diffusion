import pytest

torch = pytest.importorskip("torch")

from src.core.residuals import AdaptiveSNRWeight


def test_adaptive_snr_weight_maintains_fp32_state_and_floor():
    adaptive = AdaptiveSNRWeight(
        beta=0.5,
        ema_decay=0.0,
        kappa_floor=1e-3,
        log_interval=0,
        change_threshold=0.0,
        snr_clip=100.0,
    )

    snr = torch.full((2,), 10.0, dtype=torch.float16)
    raw_loss = torch.full((2, 1, 4, 4), 0.1, dtype=torch.float16)
    alpha = torch.full((2,), 0.5, dtype=torch.float16)

    weight, diag = adaptive.update(snr, raw_loss, alpha)

    assert weight.dtype == raw_loss.dtype
    assert adaptive._ema_val.dtype == torch.float32  # pylint: disable=protected-access
    assert diag["kappa"] >= 1e-3
    assert 0.0 < diag["mean_weight"] < 1.0
    assert 0.0 <= diag["overflow"] <= 1.0
    assert "alpha_fac" in diag


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

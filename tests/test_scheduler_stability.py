import torch

from src.training.scheduler import logsnr_cosine_schedule


def test_logsnr_cosine_stability() -> None:
    sched = logsnr_cosine_schedule(1000)
    alpha = sched["alpha"]
    sigma = sched["sigma"]
    snr = alpha / (1.0 - alpha)

    assert torch.all(alpha > 0) and torch.all(alpha < 1)
    assert torch.all(torch.isfinite(sigma))
    assert torch.all(snr < 1e6)

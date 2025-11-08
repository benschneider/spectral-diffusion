import runpy
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "debug" / "record_training_steps.py"
MODULE = runpy.run_path(str(MODULE_PATH), run_name="__test__")
summarise_snr_spikes = MODULE["_summarise_snr_spikes"]


def test_snr_spike_summary_returns_none_when_threshold_not_exceeded():
    snr = torch.tensor([10.0, 20.0]).view(-1, 1, 1, 1)
    sqrt_alpha = torch.full_like(snr, 0.9)
    sqrt_one_minus = torch.full_like(snr, 0.4)
    timesteps = torch.tensor([100, 200])
    batch = torch.zeros(2, 3, 4, 4)

    summary = summarise_snr_spikes(
        snr_vals=snr,
        sqrt_alpha_t=sqrt_alpha,
        sqrt_one_minus_t=sqrt_one_minus,
        timesteps=timesteps,
        clean=batch,
        noisy=batch,
        noise=batch,
        target=batch,
        prediction=batch,
        threshold=1_000.0,
    )

    assert summary is None


def test_snr_spike_summary_reports_top_entries_sorted():
    snr = torch.tensor([500.0, 1500.0, 2500.0]).view(-1, 1, 1, 1)
    sqrt_alpha = torch.tensor([0.9, 0.95, 0.99]).view(-1, 1, 1, 1)
    sqrt_one_minus = torch.tensor([0.3, 0.2, 0.1]).view(-1, 1, 1, 1)
    timesteps = torch.tensor([10, 20, 30])

    clean = torch.randn(3, 3, 4, 4)
    noise = torch.randn(3, 3, 4, 4)
    noisy = clean + noise
    target = torch.randn(3, 3, 4, 4)
    prediction = torch.randn(3, 3, 4, 4)

    summary = summarise_snr_spikes(
        snr_vals=snr,
        sqrt_alpha_t=sqrt_alpha,
        sqrt_one_minus_t=sqrt_one_minus,
        timesteps=timesteps,
        clean=clean,
        noisy=noisy,
        noise=noise,
        target=target,
        prediction=prediction,
        threshold=1_000.0,
        top_k=2,
    )

    assert summary is not None
    assert summary["count"] == 2
    assert summary["max_snr"] == 2500.0
    assert summary["top_timesteps"] == [30, 20]
    assert [entry["timestep"] for entry in summary["entries"]] == [30, 20]
    assert summary["entries"][0]["sample_index"] == 2
    assert summary["entries"][0]["snr"] == 2500.0

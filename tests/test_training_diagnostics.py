from types import SimpleNamespace

import json
import subprocess
import sys

import pytest
import torch
from torch import nn

from src.training.diagnostics import TaguchiAggregator, TrainingDiagnostics


def test_taguchi_aggregator_uses_runs_root(tmp_path):
    root = tmp_path / "project"
    run_dir = root / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    factor_levels = {"alpha": {"level_label": "hi"}}

    aggregator = TaguchiAggregator(run_dir, factor_levels)

    assert aggregator.aggregate_base == root
    factor_dir = aggregator.get_factor_dir("alpha")
    assert factor_dir == root / "factors" / "alpha" / "hi"
    assert factor_dir.exists()
    sanity = aggregator.sanity_dir
    assert sanity == root / "sanity"
    aggregate_dir = aggregator.aggregate_dir
    assert aggregate_dir == root / "diagnostics"


def test_training_diagnostics_captures_and_finalises(monkeypatch, tmp_path):
    factor_levels = {
        "spectral_loss_weighting": {"level_label": "strong"},
        "sampler_type": {"level_index": 1},
        "phase_attention_capacity": {"level_index": 2},
    }
    aggregator = TaguchiAggregator(tmp_path, factor_levels)

    def fake_sanity(batch, dataset_name, output_dir, prefix):
        output_dir.mkdir(parents=True, exist_ok=True)
        base = output_dir / f"{prefix}stats.npz"
        base.write_text("stats")
        (output_dir / f"{base.stem}_spatial.png").write_text("spatial")
        (output_dir / f"{base.stem}_fft_mag.png").write_text("fft")
        return base

    monkeypatch.setattr("src.training.diagnostics.check_fft_sanity", fake_sanity)

    class PlotterStub:
        def __init__(self):
            self.loss_calls = []
            self.tail_calls = []
            self.noise_calls = []
            self.phase_calls = []

        def loss_and_gradients(self, loss_steps, loss_history, grad_steps, grad_history, output_dir, run_id, filename="loss_gradients.png"):
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / filename
            path.write_text("loss")
            self.loss_calls.append((tuple(loss_steps), tuple(loss_history), tuple(grad_steps), tuple(grad_history)))
            return path

        def recent_loss_tail(self, loss_steps, loss_history, target_dir, run_id, window=50, filename=None):
            if target_dir is None:
                return None
            target_dir.mkdir(parents=True, exist_ok=True)
            path = target_dir / (filename or f"demo_loss_tail_{run_id}.png")
            path.write_text("tail")
            self.tail_calls.append(tuple(loss_steps[-window:]))
            return path

        def noise_norm(self, noise_steps, noise_history, target_dir, run_id, filename=None):
            if target_dir is None:
                return None
            target_dir.mkdir(parents=True, exist_ok=True)
            path = target_dir / (filename or f"demo_noise_norm_{run_id}.png")
            path.write_text("noise")
            self.noise_calls.append(tuple(noise_steps))
            return path

        def phase_attention(self, attention, target_path):
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text("phase")
            self.phase_calls.append(attention.shape)
            return target_path

    plotter = PlotterStub()
    diagnostics = TrainingDiagnostics(
        run_id="demo",
        dataset_name="cifar",
        work_dir=tmp_path,
        aggregator=aggregator,
        plotter=plotter,
    )

    batch = torch.zeros((1, 3, 2, 2))
    diagnostics.capture_initial_batch(batch)
    diagnostics.capture_noisy_example(batch)

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0]))
            self.pcm = SimpleNamespace(last_attention_map=torch.ones(1, 4, 4))

    model = DummyModel()
    diagnostics.capture_phase_demo(model)

    model.weight.grad = torch.tensor([2.0])
    grad_norm = diagnostics.record_gradients(model, step=0)
    assert grad_norm == pytest.approx(2.0)

    diagnostics.record_loss(1, 0.5)
    diagnostics.record_mae(1, 0.25)
    diagnostics.record_noise_norm(1, 1.5)
    diagnostics.record_fft_feedback(
        1,
        {
            "amplitude_mae": 0.1,
            "phase_mae": 0.2,
            "real_mae": 0.3,
            "imag_mae": 0.4,
            "complex_mae": 0.5,
            "amplitude_low_mae": 0.15,
        },
    )
    diagnostics.record_coeff_stats(1, {"timestep_mean": 3.0, "snr_theory": 1.5})
    diagnostics.record_batch_stats(1, {"prediction_mean": 0.1, "target_std": 0.05})
    diagnostics.record_noise_stats(1, {"snr_theory": 1.2, "snr_emp": 1.1, "snr_rel": 0.95, "variance_sum": 1.0})

    diagnostics.finalise()

    loss_file = diagnostics.diagnostics_dir / "loss_gradients.png"
    assert loss_file.exists()
    aggregate_copy = aggregator.aggregate_dir / "demo_loss_gradients.png"
    assert aggregate_copy.exists()

    spectral_dir = aggregator.get_factor_dir("spectral_loss_weighting")
    sampler_dir = aggregator.get_factor_dir("sampler_type")
    phase_dir = aggregator.get_factor_dir("phase_attention_capacity")
    assert (spectral_dir / "demo_spatial_demo.png").exists()
    assert (spectral_dir / "demo_fft_demo.png").exists()
    assert (spectral_dir / "demo_noisy_spatial_demo.png").exists()
    assert (spectral_dir / "demo_noisy_fft_demo.png").exists()
    assert (spectral_dir / "demo_loss_grad_demo.png").exists()
    assert (spectral_dir / "demo_loss_tail_demo.png").exists()
    assert (sampler_dir / "demo_loss_grad_demo.png").exists()
    assert (sampler_dir / "demo_noise_norm_demo.png").exists()
    assert (phase_dir / "demo_phase_attention_demo.png").exists()

    fft_feedback_file = diagnostics.diagnostics_dir / "fft_feedback.json"
    assert fft_feedback_file.exists()
    assert (spectral_dir / "fft_feedback_demo.json").exists()
    payload = json.loads(fft_feedback_file.read_text())
    assert "amplitude_mae" in payload
    assert "amplitude_low_mae" in payload
    assert "amplitude_mae_mean" in payload

    coeff_file = diagnostics.diagnostics_dir / "diffusion_coefficients.json"
    batch_file = diagnostics.diagnostics_dir / "batch_signal_stats.json"
    assert coeff_file.exists()
    assert batch_file.exists()
    assert (spectral_dir / "diffusion_coefficients_demo.json").exists()
    assert (spectral_dir / "batch_signal_stats_demo.json").exists()
    noise_stats_file = diagnostics.diagnostics_dir / "noise_stats.json"
    assert noise_stats_file.exists()
    assert (spectral_dir / "noise_stats_demo.json").exists()

    assert plotter.loss_calls
    assert plotter.tail_calls
    assert plotter.noise_calls
    assert plotter.phase_calls


def _run_step_recorder(output_dir, *, steps=20):
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/debug/record_training_steps.py",
        "--config",
        "configs/test_synthetic_spectral.yaml",
        "--steps",
        str(steps),
        "--log-interval",
        "1",
        "--output-dir",
        str(output_dir),
        "--adaptive-snr",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, f"record_training_steps.py failed: {combined}"
    return combined


def test_step_recorder_diagnostics_stability(tmp_path):
    out_dir = tmp_path / "diagnostics_run"
    log = _run_step_recorder(out_dir, steps=20)

    assert "Traceback" not in log
    assert "Step recorder artefacts written" in log
    assert "[SNR-GOV]" in log

    metrics_path = out_dir / "step_metrics.jsonl"
    assert metrics_path.exists()
    records = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
    assert records, "No metrics captured"

    summary_path = out_dir / "summary.json"
    assert summary_path.exists(), "summary.json missing"
    summary_payload = json.loads(summary_path.read_text())
    diagnostics_block = summary_payload.get("diagnostic_events")
    assert isinstance(diagnostics_block, list), "summary.json missing diagnostic events"
    assert any(evt.get("tag") == "Normalization" for evt in diagnostics_block)
    assert summary_payload.get("lambda_var") == pytest.approx(7e-4)

    all_events = []
    for record in records:
        all_events.extend(record.get("events", []))

    assert all_events, "No diagnostics captured"

    loss_events = [evt for evt in all_events if evt.get("tag") == "Loss"]
    assert loss_events, "No loss events captured"
    assert all(
        0.0 < float(evt.get("loss", 0.0)) < 20.0
        for evt in loss_events
    )

    prediction_events = [
        evt
        for evt in all_events
        if evt.get("tag") == "TensorPreview" and evt.get("tensor") == "prediction"
    ]
    assert prediction_events, "No prediction std values captured"
    assert any("variance_penalty" in record for record in records)
    assert max(float(evt.get("std", 0.0)) for evt in prediction_events) < 50.0

    weight_events = [evt for evt in all_events if evt.get("tag") == "AdaptiveSNRWeight"]
    weight_means = [float(evt.get("mean_weight")) for evt in weight_events if evt.get("mean_weight") is not None]
    assert weight_means, "Adaptive weights not logged"
    for mean in weight_means:
        assert 0.7 <= mean <= 1.3

    overflow_events = [evt for evt in all_events if evt.get("tag") == "OverflowHandler"]
    assert overflow_events, "Deterministic overflow regime not observed"
    snr_values = [float(evt.get("snr", evt.get("snr_raw_max", 0.0))) for evt in overflow_events]
    assert max(snr_values) <= 300.0
    assert len(overflow_events) <= 5

    assert not (out_dir / "diagnostics.jsonl").exists()

    second_log = _run_step_recorder(out_dir, steps=5)
    assert "Step recorder artefacts written" in second_log

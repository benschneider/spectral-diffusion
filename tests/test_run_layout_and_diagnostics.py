from __future__ import annotations

from pathlib import Path

import torch

from src.cli.common import ensure_directories
from src.training.diagnostics import TaguchiAggregator, TrainingDiagnostics


def test_ensure_directories_supports_runs_output_root(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    results_root = tmp_path / "results"

    dirs_runs = ensure_directories(output_dir=runs_root, run_id="demo")
    assert dirs_runs["run_dir"] == runs_root / "demo"
    assert dirs_runs["logs_dir"].is_dir()
    assert dirs_runs["checkpoints_dir"].is_dir()
    assert dirs_runs["metrics_dir"].is_dir()
    assert dirs_runs["images_dir"].is_dir()

    dirs_results = ensure_directories(output_dir=results_root, run_id="demo")
    assert dirs_results["run_dir"] == results_root / "runs" / "demo"


def test_taguchi_aggregator_resolves_output_root() -> None:
    base = Path("tmp")  # purely for path arithmetic, no filesystem ops needed
    aggregator = TaguchiAggregator(base / "runs" / "demo", {})
    assert aggregator.work_dir.parent.name == "runs"


def test_taguchi_aggregator_detects_project_root(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir(parents=True, exist_ok=True)
    (project / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")

    run_dir = project / "runs" / "run123"
    run_dir.mkdir(parents=True, exist_ok=True)

    aggregator = TaguchiAggregator(run_dir, {})
    assert aggregator.aggregate_base == project / "runs"

    legacy_root = tmp_path / "results"
    legacy_run_dir = legacy_root / "runs" / "run123"
    legacy_run_dir.mkdir(parents=True, exist_ok=True)
    legacy = TaguchiAggregator(legacy_run_dir, {})
    assert legacy.aggregate_base == legacy_root


def test_capture_noisy_example_writes_eps_artifacts(tmp_path: Path) -> None:
    work_dir = tmp_path / "results" / "runs" / "run123"
    work_dir.mkdir(parents=True, exist_ok=True)
    factor_levels = {"spectral_operator_mode": {"level_label": "none"}}
    aggregator = TaguchiAggregator(work_dir, factor_levels)
    diagnostics = TrainingDiagnostics(
        run_id="run123",
        dataset_name="synthetic",
        work_dir=work_dir,
        aggregator=aggregator,
    )

    noisy = torch.randn(4, 3, 8, 8)
    eps = torch.randn(4, 3, 8, 8)
    diagnostics.capture_noisy_example(noisy, eps=eps)

    expected_json = aggregator.sanity_dir / "run123_eps_sanity_synthetic_eps.json"
    expected_fft = aggregator.sanity_dir / "run123_eps_sanity_synthetic_eps_fft_mag.png"
    assert expected_json.exists()
    assert expected_fft.exists()

    factor_dir = aggregator.get_factor_dir("spectral_operator_mode")
    assert factor_dir is not None
    assert (factor_dir / "demo_eps_fft_run123.png").exists()


def test_training_history_csv_written(tmp_path: Path) -> None:
    work_dir = tmp_path / "results" / "runs" / "run123"
    work_dir.mkdir(parents=True, exist_ok=True)
    aggregator = TaguchiAggregator(work_dir, {})
    diagnostics = TrainingDiagnostics(
        run_id="run123",
        dataset_name="synthetic",
        work_dir=work_dir,
        aggregator=aggregator,
    )
    diagnostics.record_noise_stats(
        1,
        {
            "snr_theory": 1.0,
            "snr_emp": 1.0,
            "snr_rel": 1.0,
            "variance_sum": 1.0,
            "noise_channel_std_min": 0.1,
            "noise_channel_std_max": 0.2,
        },
    )
    diagnostics.finalise()
    history_path = work_dir / "diagnostics" / "training_history.csv"
    assert history_path.exists()
    header = history_path.read_text(encoding="utf-8").splitlines()[0]
    assert "loss" in header

import json
from pathlib import Path

import pandas as pd

from src.visualization.analysis_utils import compute_fft_corrected, collect_loss_histories
from src.visualization.report import write_summary_markdown


def test_collect_loss_histories_round_trip(tmp_path):
    metrics_dir = tmp_path / "runs" / "example"
    metrics_dir.mkdir(parents=True)
    metrics_payload = {
        "loss_history": [1.0, 0.7, 0.5],
        "mae_history": [0.9, 0.6, 0.4],
    }
    metrics_path = metrics_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload))

    summary_df = pd.DataFrame(
        {
            "run_id": ["example"],
            "metrics_path": [str(metrics_path)],
            "loss_final": [0.5],
        }
    )

    histories = collect_loss_histories(summary_df)

    assert len(histories) == 1
    record = histories[0]
    assert record["label"] == "example"
    assert record["loss_history"] == metrics_payload["loss_history"]
    assert record["mae_history"] == metrics_payload["mae_history"]


def test_compute_fft_corrected_handles_missing_columns():
    df = pd.DataFrame({"run_id": ["a"], "loss_final": [0.25]})
    corrected = compute_fft_corrected(df)
    assert corrected.equals(df)

    df_with_runtime = pd.DataFrame(
        {"run_id": ["a"], "runtime": [12.0], "fft_runtime": [2.0], "efficiency": [5.0], "fft_efficiency": [1.0]}
    )
    corrected_full = compute_fft_corrected(df_with_runtime)
    assert corrected_full["runtime_corrected"].iloc[0] == 10.0
    assert corrected_full["efficiency_corrected"].iloc[0] == 4.0


def test_write_summary_markdown_includes_new_sections(tmp_path):
    synthetic_df = pd.DataFrame(
        {
            "run_id": ["synthetic"],
            "loss_final": [0.42],
            "images_per_second": [120.0],
            "runtime_seconds": [9.5],
            "loss_drop_per_second": [0.8],
            "high_freq_psnr": [32.7],
        }
    )
    cifar_df = pd.DataFrame(
        {
            "run_id": ["cifar"],
            "loss_final": [0.31],
            "images_per_second": [48.0],
            "runtime_seconds": [18.0],
            "loss_drop_per_second": [0.5],
            "high_freq_psnr": [29.2],
        }
    )

    taguchi_dir = tmp_path / "taguchi"
    demo_dir = taguchi_dir / "factors" / "spectral_adapter_placement" / "none"
    demo_dir.mkdir(parents=True)
    demo_img = demo_dir / "demo_spatial_run.png"
    demo_img.write_bytes(b"demo")

    sanity_dir = taguchi_dir / "sanity"
    sanity_dir.mkdir(parents=True, exist_ok=True)
    sanity_json = sanity_dir / "run_sanity_cifar10.json"
    sanity_json.write_text(json.dumps({"mean": 0.0, "std": 1.0, "is_complex": False, "fft_reconstruction_error": 0.0}))
    (sanity_dir / "run_sanity_cifar10_spatial.png").write_bytes(b"s")
    (sanity_dir / "run_sanity_cifar10_fft_mag.png").write_bytes(b"f")

    diag_dir = taguchi_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    (diag_dir / "example_loss_gradients.png").write_bytes(b"g")

    out_path = tmp_path / "summary.md"
    write_summary_markdown(
        synthetic_df,
        cifar_df,
        taguchi_report=None,
        out_path=out_path,
        descriptions={"taguchi_choices": {}},
        generated_at="2025-01-01T00:00:00",
        taguchi_dir=taguchi_dir,
    )
    text = out_path.read_text(encoding="utf-8")

    assert "## Synthetic Benchmark" in text
    assert "## CIFAR-10 Reconstruction Benchmark" in text
    assert "high_freq_psnr" in text or "Sharpest spectra" in text
    assert "Summary Table" in text
    assert "Factor Primer" in text
    assert "Factor Demos" in text
    assert "CIFAR Sanity Diagnostics" in text
    assert "CIFAR Spectral Diagnostics" in text

    html_path = out_path.with_suffix(".html")
    assert html_path.exists()


def test_write_summary_markdown_without_data(tmp_path):
    out_path = tmp_path / "summary.md"
    write_summary_markdown(
        synthetic_df=None,
        cifar_df=None,
        taguchi_report=None,
        out_path=out_path,
        descriptions={"taguchi_choices": {}},
    )
    text = out_path.read_text(encoding="utf-8")
    assert "_No benchmark data available._" in text

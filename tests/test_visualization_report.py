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

    out_path = tmp_path / "summary.md"
    write_summary_markdown(
        synthetic_df,
        cifar_df,
        taguchi_report=None,
        out_path=out_path,
        descriptions={"taguchi_choices": {}},
        generated_at="2025-01-01T00:00:00",
    )
    text = out_path.read_text(encoding="utf-8")

    assert "## Synthetic Benchmark" in text
    assert "## CIFAR-10 Reconstruction Benchmark" in text
    assert "high_freq_psnr" in text or "Sharpest spectra" in text
    assert "Summary Table" in text


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

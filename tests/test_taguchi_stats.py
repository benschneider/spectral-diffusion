from pathlib import Path

import pandas as pd
import yaml

from src.analysis.taguchi_stats import generate_taguchi_report


def test_generate_taguchi_report_single_run(tmp_path):
    """Test that single-run configs without factors return None gracefully."""
    summary_path = tmp_path / "summary.csv"
    config_dir = tmp_path / "configs"
    config_path = config_dir / "single.yaml"
    _write_config(config_path, {})  # No factors

    data = [{
        "run_id": "single_run",
        "config_path": str(config_path),
        "metrics_path": "metrics.json",
        "timestamp": "2024-01-01T00:00:00",
        "loss_mean": 0.5,
    }]
    pd.DataFrame(data).to_csv(summary_path, index=False)

    output_path = tmp_path / "report.csv"
    report = generate_taguchi_report(summary_path, "loss_mean", "larger", output_path)
    assert report is None  # Should return None for insufficient data
    assert not output_path.exists()  # Should not create file


def _write_config(path: Path, factor_row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"taguchi": {"row": factor_row}}, handle)


def test_generate_taguchi_report(tmp_path):
    summary_path = tmp_path / "summary.csv"
    config_dir = tmp_path / "configs"

    data = []
    for idx, (level_a, level_b, metric_val) in enumerate(
        [
            (1, 1, 0.1),
            (1, 2, 0.2),
            (2, 1, 0.5),
            (2, 2, 0.6),
        ]
    ):
        run_id = f"run{idx}"
        config_path = config_dir / f"{run_id}.yaml"
        _write_config(config_path, {"A": level_a, "B": level_b})
        data.append(
            {
                "run_id": run_id,
                "config_path": str(config_path),
                "metrics_path": "metrics.json",
                "timestamp": "2024-01-01T00:00:00",
                "loss_mean": metric_val,
            }
        )

    pd.DataFrame(data).to_csv(summary_path, index=False)

    output_path = tmp_path / "report.csv"
    report = generate_taguchi_report(summary_path, "loss_mean", "larger", output_path)
    assert output_path.exists()
    assert not report.empty
    factors = set(report["factor"].unique())
    assert factors == {"A", "B"}


def test_generate_taguchi_report_resolves_external_paths(tmp_path):
    report_root = tmp_path / "taguchi"
    runs_dir = report_root / "runs"
    run_id = "row01"
    config_path = runs_dir / run_id / "config.yaml"
    _write_config(config_path, {"A": 1})

    summary_path = report_root / "summary.csv"
    data = [
        {
            "run_id": run_id,
            "config_path": f"/external/machine/results/taguchi/runs/{run_id}/config.yaml",
            "metrics_path": "metrics.json",
            "timestamp": "2024-01-01T00:00:00",
            "loss_mean": 0.3,
        }
    ]
    pd.DataFrame(data).to_csv(summary_path, index=False)

    output_path = report_root / "report.csv"
    report = generate_taguchi_report(summary_path, "loss_mean", "larger", output_path)
    assert output_path.exists()
    assert not report.empty

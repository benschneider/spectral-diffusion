from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.analysis.taguchi_stats import generate_taguchi_report


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

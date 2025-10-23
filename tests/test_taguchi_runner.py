import csv
import json
from pathlib import Path

import pandas as pd

from src.experiments.run_experiment import TaguchiExperimentRunner


def _write_design_csv(path: Path) -> None:
    df = pd.DataFrame([
        {"A": 1, "B": 1, "C": 1},
        {"A": 2, "B": 2, "C": 2},
    ])
    df.to_csv(path, index=False)


def _base_config() -> dict:
    return {
        "model": {"type": "baseline"},
        "data": {
            "source": "synthetic",
            "channels": 3,
            "height": 8,
            "width": 8,
        },
        "training": {
            "batch_size": 2,
            "num_batches": 1,
            "epochs": 1,
            "log_every": 1,
        },
    }


def test_run_batch_persists_artifacts(tmp_path):
    design_path = tmp_path / "design.csv"
    _write_design_csv(design_path)
    runner = TaguchiExperimentRunner(design_matrix_path=design_path, base_config=_base_config())

    results = runner.run_batch(output_dir=tmp_path)

    assert len(results) == 2
    summary_path = tmp_path / "summary.csv"
    assert summary_path.exists()
    with summary_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows

    for result in results:
        run_id = result["run_id"]
        matching = [row for row in rows if row["run_id"] == run_id]
        assert matching
        row = matching[0]
        required_fields = [
            "loss_mean",
            "loss_initial",
            "loss_final",
            "loss_drop",
            "loss_drop_per_second",
            "runtime_seconds",
            "steps_per_second",
            "images_per_second",
        ]
        for field in required_fields:
            assert row[field] not in ("", None)
        metrics_path = Path(result["metrics_path"])
        assert metrics_path.exists()
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        assert metrics.get("status") == "ok"

        config_path = Path(result["config_path"])
        assert config_path.exists()


def test_build_config_does_not_mutate_base(tmp_path):
    design_path = tmp_path / "design.csv"
    _write_design_csv(design_path)
    base_cfg = _base_config()
    runner = TaguchiExperimentRunner(design_matrix_path=design_path, base_config=base_cfg)

    first_row = runner.design.iloc[0]
    cfg1 = runner._build_config_from_row(first_row)
    cfg1.setdefault("spectral", {})["freq_attention"] = True

    cfg2 = runner._build_config_from_row(first_row)
    assert cfg2.get("spectral", {}).get("freq_attention") is False

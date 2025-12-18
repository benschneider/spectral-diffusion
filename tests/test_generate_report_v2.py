import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_run(root: Path, run_id: str, loss_final: float, loss_drop: float, ips: float, factor_levels: dict | None = None) -> None:
    runs_dir = root / "runs" / run_id
    runs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = runs_dir / "metrics.json"
    metrics_path.write_text(json.dumps({"loss_history": [1.0, 0.8, 0.6, loss_final]}), encoding="utf-8")
    config_path = runs_dir / "config.yaml"
    taguchi_block = ""
    if factor_levels:
        taguchi_lines = ["taguchi:", "  factor_levels:"]
        for key, meta in factor_levels.items():
            taguchi_lines.append(f"    {key}:")
            taguchi_lines.append(f"      level_label: {meta.get('level_label', meta)}")
            taguchi_lines.append(f"      level_index: {meta.get('level_index', 1)}")
        taguchi_block = "\n" + "\n".join(taguchi_lines) + "\n"
    config_path.write_text("model:\n  type: test\n" + taguchi_block, encoding="utf-8")
    # Summary row closely matches run_full_report outputs
    summary = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "config_path": str(config_path),
                "metrics_path": str(metrics_path),
                "loss_final": loss_final,
                "loss_drop_per_second": loss_drop,
                "images_per_second": ips,
            }
        ]
    )
    summary.to_csv(root / "summary.csv", index=False)


def _write_taguchi(root: Path) -> None:
    taguchi_dir = root / "taguchi"
    taguchi_dir.mkdir(parents=True, exist_ok=True)
    _write_run(
        taguchi_dir,
        "taguchi_row01",
        0.15,
        0.02,
        70.0,
        factor_levels={
            "snr_ratio": {"level_label": "0.8", "level_index": 1},
            "spectral_operator_mode": {"level_label": "radial", "level_index": 2},
        },
    )

    report_rows = [
        {
            "factor": "snr_ratio",
            "level": "0.8",
            "mean_metric": 0.02,
            "snr": 1.0,
            "mean_runtime_seconds": 10.0,
            "mean_images_per_second": 70.0,
            "mean_loss_final": 0.15,
            "delta": 0.0,
            "rank": 1,
        },
        {
            "factor": "spectral_operator_mode",
            "level": "radial",
            "mean_metric": 0.03,
            "snr": 1.5,
            "mean_runtime_seconds": 9.0,
            "mean_images_per_second": 80.0,
            "mean_loss_final": 0.12,
            "delta": 0.0,
            "rank": 2,
        },
    ]
    pd.DataFrame(report_rows).to_csv(taguchi_dir / "taguchi_report.csv", index=False)
    factor_mapping = {
        "factors": {
            "snr_ratio": {"levels": [0.8, 1.0]},
            "spectral_operator_mode": {"levels": ["none", "radial"]},
        }
    }
    (taguchi_dir / "factor_mapping.json").write_text(json.dumps(factor_mapping), encoding="utf-8")


def test_generate_report_v2_creates_outputs(tmp_path):
    root = tmp_path / "report_root"
    synthetic_dir = root / "synthetic"
    cifar_dir = root / "cifar"
    synthetic_dir.mkdir(parents=True)
    cifar_dir.mkdir(parents=True)

    _write_run(synthetic_dir, "synthetic_run", loss_final=0.4, loss_drop=0.01, ips=120.0)
    _write_run(cifar_dir, "cifar_run", loss_final=0.05, loss_drop=0.03, ips=90.0)
    _write_taguchi(root)

    output_dir = root / "report_v2"
    cmd = [
        sys.executable,
        str(Path("scripts/generate_report_v2.py").resolve()),
        "--report-root",
        str(root),
        "--generated-at",
        "TEST-TIMESTAMP",
    ]
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    subprocess.run(cmd, check=True, env=env, cwd=Path(__file__).resolve().parents[1])

    summary_md = output_dir / "summary.md"
    assert summary_md.exists(), "summary.md should be generated"
    text = summary_md.read_text(encoding="utf-8")
    assert "Stability & Convergence" in text

    required_images = [
        "loss_curve_synthetic.png",
        "loss_curve_cifar.png",
        "tradeoff_loss_vs_speed_synthetic.png",
        "tradeoff_loss_vs_speed_cifar.png",
        "taguchi_main_effects_primary.png",
        "taguchi_contrib_primary.png",
        "samples_profile_comparison_1.png",
    ]
    for name in required_images:
        assert (output_dir / "images" / name).exists(), f"{name} should exist"

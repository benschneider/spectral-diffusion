#!/usr/bin/env python
"""Run a minimal ablation comparing baseline vs spectral toggles on/off."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualization.analysis_utils import compute_fft_corrected
from src.visualization.plots import plot_feature_toggle_ablation, save_figure


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TinyUNet baseline vs SpectralUNet with and without spectral toggles."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/benchmark_spectral_cifar.yaml",
        help="Base configuration for spectral runs (default: benchmark_spectral_cifar.yaml).",
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=ROOT / "configs/benchmark_spectral_cifar.yaml",
        help="Configuration for the baseline TinyUNet run (default: same as --config).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store ablation results (defaults to results/spectral_toggle_ablation_<timestamp>).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting the feature ablation figure.",
    )
    return parser.parse_args()


def _augment_config_for_uniform(source: Path, destination: Path) -> None:
    with source.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    diffusion_cfg = cfg.setdefault("diffusion", {})
    diffusion_cfg["uniform_corruption"] = True

    model_cfg = cfg.setdefault("model", {})
    model_cfg["enable_amp_residual"] = True
    model_cfg["enable_phase_attention"] = True
    model_cfg.setdefault("amp_hidden_dim", max(int(model_cfg.get("base_channels", 32)), 16))
    model_cfg.setdefault("phase_heads", 1)

    sampling_cfg = cfg.setdefault("sampling", {})
    sampling_cfg["sampler_type"] = "masf"
    sampling_cfg.setdefault("num_steps", 50)
    sampling_cfg.setdefault("num_samples", 8)

    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle)


def _run_training(config_path: Path, output_dir: Path, run_id: str, variant: Optional[str] = None) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--run-id",
        run_id,
    ]
    if variant:
        cmd.extend(["--variant", variant])
    subprocess.run(cmd, check=True)


def _load_metrics(run_dir: Path, run_id: str) -> Dict[str, float]:
    metrics_path = run_dir / "metrics" / f"{run_id}.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_summary_rows(base_output: Path, run_specs: List[Dict[str, str]]) -> pd.DataFrame:
    rows = []
    for spec in run_specs:
        run_id = spec["run_id"]
        run_dir = base_output / "runs" / run_id
        metrics = _load_metrics(run_dir, run_id)
        rows.append(
            {
                "run_id": run_id,
                "display_name": spec["label"],
                "variant": spec["variant"],
                "loss_final": metrics.get("loss_final"),
                "loss_drop_per_second": metrics.get("loss_drop_per_second"),
                "loss_drop": metrics.get("loss_drop"),
                "images_per_second": metrics.get("images_per_second"),
                "runtime_seconds": metrics.get("runtime_seconds"),
                "metrics_path": str(run_dir / "metrics" / f"{run_id}.json"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (ROOT / "results" / f"spectral_toggle_ablation_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_specs = [
        {"run_id": "baseline_tiny", "label": "TinyUNet Baseline", "variant": "baseline"},
        {"run_id": "spectral_plain", "label": "SpectralUNet Plain", "variant": "spectral"},
        {"run_id": "spectral_uniform", "label": "SpectralUNet Uniform+ARE/PCM+MASF", "variant": "spectral"},
    ]

    # Baseline run
    print("Running TinyUNet baseline...")
    _run_training(args.baseline_config, output_dir, run_specs[0]["run_id"], variant="baseline")

    # Spectral plain
    print("Running SpectralUNet (plain)...")
    _run_training(args.config, output_dir, run_specs[1]["run_id"], variant="spectral")

    # Spectral with toggles
    print("Running SpectralUNet with uniform corruption + ARE/PCM + MASF...")
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp_cfg:
        temp_path = Path(temp_cfg.name)
    _augment_config_for_uniform(args.config, temp_path)
    try:
        _run_training(temp_path, output_dir, run_specs[2]["run_id"], variant="spectral")
    finally:
        temp_path.unlink(missing_ok=True)

    print("Collecting metrics...")
    summary_df = _collect_summary_rows(output_dir, run_specs)
    summary_df = compute_fft_corrected(summary_df)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote summary to {summary_path}")

    if not args.no_plot:
        try:
            print("Generating spectral toggle ablation figure...")
            plot_feature_toggle_ablation(summary_df, output_dir / "spectral_toggle_ablation.png")
        except Exception as exc:  # pragma: no cover - plotting fallback
            print(f"[warn] Failed to generate ablation figure: {exc}")

    print(f"Done. Results stored in {output_dir}")


if __name__ == "__main__":
    main()

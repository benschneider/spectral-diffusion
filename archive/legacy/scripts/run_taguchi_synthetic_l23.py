#!/usr/bin/env python
"""Run the synthetic-only Taguchi L23 design using record_training_steps diagnostics.

This runner targets the synthetic diffusion pipeline and maps 13 control factors:

F1 Shock handler         -> --shock-handler {off,on}
F2 Normalization mode    -> --normalization-mode {off,batch,band}
F3 Variance penalty λ_var -> --lambda-var {7e-4,2e-3,4e-3}
F4 Gradient clipping      -> --clip-mode {none,global,ratio}
F5 EMA β                  -> --ema-beta {0,0.99,0.999}
F6 Optimizer              -> --optimizer {adamw,lion,adafactor}
F7 Learning rate η        -> --learning-rate {3e-4,1e-3,3e-3}
F8 Curriculum smoothing   -> --curriculum {none,light,strong}
F9 Synthetic profile      -> --input-profile {piecewise,texture,randomfield}
F10 Noise colour mix      -> --noise-color {white,multicolor,patterned}
F11 Spectral slope α      -> --spectral-slope {flat,mild,steep}
F12 SNR level             -> --snr-level {low,nominal,high}
F13 Seed mode             -> --seed-mode {42,123,314}

Each Taguchi row is executed via ``record_training_steps.py`` using the dedicated
baseline config ``configs/taguchi/L23_synthetic.yaml``. The resulting
``summary.json`` files are aggregated into ``results.csv`` alongside the original
factor levels and resolved CLI mappings.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECORD_SCRIPT = ROOT / "scripts" / "debug" / "record_training_steps.py"
DEFAULT_DESIGN = ROOT / "configs" / "taguchi" / "L23_synthetic.csv"
DEFAULT_BASE_CONFIG = ROOT / "configs" / "taguchi" / "L23_synthetic.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "taguchi_l23_synthetic"

FACTOR_NAMES = {
    "F1": "shock_handler",
    "F2": "normalization_mode",
    "F3": "lambda_var",
    "F4": "clip_mode",
    "F5": "ema_beta",
    "F6": "optimizer",
    "F7": "learning_rate",
    "F8": "curriculum",
    "F9": "input_profile",
    "F10": "noise_color",
    "F11": "spectral_slope",
    "F12": "snr_level",
    "F13": "seed_mode",
}

FACTOR_MAP: Dict[str, Dict[int, List[str]]] = {
    "F1": {0: ["--shock-handler", "off"], 1: ["--shock-handler", "on"]},
    "F2": {
        0: ["--normalization-mode", "off"],
        1: ["--normalization-mode", "batch"],
        2: ["--normalization-mode", "band"],
    },
    "F3": {
        0: ["--lambda-var", "0.0007"],
        1: ["--lambda-var", "0.002"],
        2: ["--lambda-var", "0.004"],
    },
    "F4": {
        0: ["--clip-mode", "none"],
        1: ["--clip-mode", "global"],
        2: ["--clip-mode", "ratio"],
    },
    "F5": {
        0: ["--ema-beta", "0.0"],
        1: ["--ema-beta", "0.99"],
        2: ["--ema-beta", "0.999"],
    },
    "F6": {
        0: ["--optimizer", "adamw"],
        1: ["--optimizer", "lion"],
        2: ["--optimizer", "adafactor"],
    },
    "F7": {
        0: ["--learning-rate", "0.0003"],
        1: ["--learning-rate", "0.001"],
        2: ["--learning-rate", "0.003"],
    },
    "F8": {
        0: ["--curriculum", "none"],
        1: ["--curriculum", "light"],
        2: ["--curriculum", "strong"],
    },
    "F9": {
        0: ["--input-profile", "piecewise"],
        1: ["--input-profile", "texture"],
        2: ["--input-profile", "randomfield"],
    },
    "F10": {
        0: ["--noise-color", "white"],
        1: ["--noise-color", "multicolor"],
        2: ["--noise-color", "patterned"],
    },
    "F11": {
        0: ["--spectral-slope", "flat"],
        1: ["--spectral-slope", "mild"],
        2: ["--spectral-slope", "steep"],
    },
    "F12": {
        0: ["--snr-level", "low"],
        1: ["--snr-level", "nominal"],
        2: ["--snr-level", "high"],
    },
    "F13": {
        0: ["--seed-mode", "42"],
        1: ["--seed-mode", "123"],
        2: ["--seed-mode", "314"],
    },
}


def _resolve_row_cli(row: pd.Series) -> Dict[str, List[str]]:
    resolved: Dict[str, List[str]] = {}
    for column in FACTOR_NAMES:
        level = int(row[column])
        mapping = FACTOR_MAP[column]
        if level not in mapping:
            raise KeyError(f"Level {level} not defined for {column}")
        resolved[column] = mapping[level]
    return resolved


def _command_tokens(base_tokens: Iterable[str], factor_tokens: Dict[str, List[str]]) -> List[str]:
    tokens: List[str] = list(base_tokens)
    for column in FACTOR_NAMES:
        tokens.extend(factor_tokens[column])
    return tokens


def _load_summary(summary_path: Path) -> Dict[str, object]:
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_design(
    design_path: Path,
    output_dir: Path,
    *,
    base_config: Path,
    steps: int,
    log_interval: int,
    dry_run: bool = False,
) -> None:
    design = pd.read_csv(design_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_rows: List[Dict[str, object]] = []
    results_rows: List[Dict[str, object]] = []

    base_cli = [
        sys.executable,
        str(RECORD_SCRIPT),
        "--config",
        str(base_config),
        "--steps",
        str(int(steps)),
        "--log-interval",
        str(int(log_interval)),
    ]

    for _, row in design.iterrows():
        run_id = int(row["RunID"])
        run_dir = output_dir / f"run_{run_id:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        factor_tokens = _resolve_row_cli(row)
        resolved_cli_strings = {
            FACTOR_NAMES[column]: " ".join(factor_tokens[column]) for column in FACTOR_NAMES
        }
        resolved_row = {"RunID": run_id}
        resolved_row.update({column: int(row[column]) for column in FACTOR_NAMES})
        resolved_row.update(resolved_cli_strings)
        resolved_rows.append(resolved_row)

        if dry_run:
            continue

        cmd = _command_tokens(base_cli + ["--output-dir", str(run_dir)], factor_tokens)
        subprocess.run(cmd, check=True)

        summary_path = run_dir / "summary.json"
        metrics = _load_summary(summary_path)

        result_record: Dict[str, object] = {
            "RunID": run_id,
        }
        result_record.update({column: int(row[column]) for column in FACTOR_NAMES})
        for key in (
            "final_loss",
            "final_mae",
            "mean_structure_corr",
            "mean_fft_corr",
            "variance_ratio_mean",
            "shock_trigger_count",
            "shock_active_steps",
            "optimizer",
            "base_learning_rate",
        ):
            if key in metrics:
                result_record[key] = metrics[key]
        result_record["summary_path"] = str(summary_path)
        results_rows.append(result_record)

    resolved_path = output_dir / "L23_synthetic_resolved.csv"
    with resolved_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(resolved_rows[0].keys()) if resolved_rows else [])
        writer.writeheader()
        writer.writerows(resolved_rows)

    if not dry_run and results_rows:
        results_path = output_dir / "results.csv"
        with results_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results_rows[0].keys()))
            writer.writeheader()
            writer.writerows(results_rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the synthetic Taguchi L23 design")
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN, help="CSV design matrix path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Destination for run artefacts")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG, help="Base YAML config")
    parser.add_argument("--steps", type=int, default=120, help="Number of optimisation steps per run")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval for record_training_steps")
    parser.add_argument("--dry-run", action="store_true", help="Only materialise resolved CLI table")
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_design(
        design_path=args.design,
        output_dir=args.output_dir,
        base_config=args.base_config,
        steps=int(args.steps),
        log_interval=int(args.log_interval),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()

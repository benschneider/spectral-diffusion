#!/usr/bin/env python
"""Execute the L23 Taguchi fine-tuning sweep on a pretrained checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECORD_SCRIPT = ROOT / "scripts" / "debug" / "record_training_steps.py"
DEFAULT_DESIGN = ROOT / "configs" / "taguchi" / "L23_synthetic.csv"
DEFAULT_CONFIG = ROOT / "configs" / "taguchi" / "fine_tune_l23.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "taguchi_fine_tune_l23"

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
        resolved[column] = mapping[level]
    return resolved


def _command_tokens(base_tokens: Iterable[str], factor_tokens: Dict[str, List[str]]) -> List[str]:
    tokens: List[str] = list(base_tokens)
    for column in FACTOR_NAMES:
        tokens.extend(factor_tokens[column])
    return tokens


def _load_summary(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_design(
    design_path: Path,
    output_dir: Path,
    *,
    base_config: Path,
    checkpoint: Optional[Path],
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
    if checkpoint is not None:
        base_cli.extend(["--checkpoint", str(checkpoint)])

    for _, row in design.iterrows():
        run_id = int(row["RunID"])
        run_dir = output_dir / f"run_{run_id:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        factor_tokens = _resolve_row_cli(row)
        resolved_cli_strings = {
            FACTOR_NAMES[column]: " ".join(factor_tokens[column]) for column in FACTOR_NAMES
        }
        resolved_row: Dict[str, object] = {"RunID": run_id}
        resolved_row.update({column: int(row[column]) for column in FACTOR_NAMES})
        resolved_row.update(resolved_cli_strings)
        resolved_rows.append(resolved_row)

        if dry_run:
            continue

        cmd = _command_tokens(base_cli + ["--output-dir", str(run_dir)], factor_tokens)
        subprocess.run(cmd, check=True)

        summary_path = run_dir / "summary.json"
        metrics = _load_summary(summary_path)

        result_record: Dict[str, object] = {"RunID": run_id}
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

    resolved_path = output_dir / "L23_finetune_resolved.csv"
    if resolved_rows:
        with resolved_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(resolved_rows[0].keys()))
            writer.writeheader()
            writer.writerows(resolved_rows)

    if not dry_run and results_rows:
        results_path = output_dir / "results.csv"
        with results_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results_rows[0].keys()))
            writer.writeheader()
            writer.writerows(results_rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Taguchi L23 fine-tuning design")
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN, help="Taguchi design CSV")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG, help="Baseline YAML")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pretrained checkpoint to load")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Run outputs")
    parser.add_argument("--steps", type=int, default=400, help="Number of fine-tuning steps each run")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval for diagnostics")
    parser.add_argument("--dry-run", action="store_true", help="Only materialise CLI plan without running")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_design(
        design_path=args.design,
        output_dir=args.output_dir,
        base_config=args.base_config,
        checkpoint=args.checkpoint,
        steps=int(args.steps),
        log_interval=int(args.log_interval),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()

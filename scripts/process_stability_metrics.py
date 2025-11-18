#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


def _to_float(value: str) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def _valid(values: Iterable[float]) -> List[float]:
    return [v for v in values if not math.isnan(v)]


def _load_stability_rows(csv_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {key: _to_float(value) if key != "step" else int(value) for key, value in row.items()}
            rows.append(parsed)
    return rows


def _load_noise_stats(json_path: Path) -> Mapping[str, Sequence[float]]:
    if not json_path.exists():
        return {}
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload


def _summarise(run_id: str, rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        raise ValueError(f"No stability rows for run '{run_id}'")

    def values(key: str) -> List[float]:
        return _valid(row.get(key, math.nan) for row in rows)

    def avg(key: str) -> float:
        vals = values(key)
        return mean(vals) if vals else math.nan

    def peak(key: str) -> float:
        vals = values(key)
        return max(vals) if vals else math.nan

    def total(key: str) -> float:
        vals = values(key)
        return sum(vals) if vals else math.nan

    summary = {
        "run_id": run_id,
        "snr_mean_avg": avg("snr_mean"),
        "snr_mean_std": pstdev(values("snr_mean")) if len(values("snr_mean")) > 1 else 0.0,
        "snr_measured_avg": avg("snr_measured"),
        "snr_raw_max_peak": peak("snr_raw_max"),
        "overflow_count_total": total("overflow_count"),
        "overflow_rate_peak": peak("overflow_rate_per_1k"),
        "variance_ratio_avg": avg("variance_ratio"),
        "variance_ratio_std": pstdev(values("variance_ratio")) if len(values("variance_ratio")) > 1 else 0.0,
        "prediction_std_ratio_avg": avg("prediction_std_ratio"),
        "spectral_pressure_avg": avg("spectral_pressure"),
    }
    return summary


def _write_summary(output_path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        raise ValueError("No runs to summarise")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "snr_mean_avg",
        "snr_mean_std",
        "snr_measured_avg",
        "snr_raw_max_peak",
        "overflow_count_total",
        "overflow_rate_peak",
        "variance_ratio_avg",
        "variance_ratio_std",
        "prediction_std_ratio_avg",
        "spectral_pressure_avg",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_run(run_id: str, rows: List[Dict[str, float]], noise_stats: Mapping[str, Sequence[float]], plot_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting optional
        raise RuntimeError("matplotlib is required for plotting. Install matplotlib and retry.") from exc

    plot_dir.mkdir(parents=True, exist_ok=True)
    steps = [row["step"] for row in rows]
    snr_mean = [row.get("snr_mean", math.nan) for row in rows]
    snr_measured = [row.get("snr_measured", math.nan) for row in rows]
    overflow_rate = [row.get("overflow_rate_per_1k", math.nan) for row in rows]
    variance_ratio = [row.get("variance_ratio", math.nan) for row in rows]
    prediction_std_ratio = [row.get("prediction_std_ratio", math.nan) for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(steps, snr_mean, label="snr_mean")
    axes[0].plot(steps, snr_measured, label="snr_measured", linestyle="--")
    if noise_stats:
        snr_ratio = noise_stats.get("snr_ratio")
        if isinstance(snr_ratio, Sequence):
            axes[0].plot(noise_stats.get("steps", steps), snr_ratio, label="snr_ratio (input)", alpha=0.7)
    axes[0].set_ylabel("SNR")
    axes[0].set_title(f"{run_id}: SNR statistics")
    axes[0].legend()

    axes[1].plot(steps, overflow_rate, color="tomato")
    axes[1].set_ylabel("Overflow / 1k steps")
    axes[1].set_title("Overflow incidence")

    axes[2].plot(steps, variance_ratio, label="variance_ratio")
    axes[2].plot(steps, prediction_std_ratio, label="prediction_std_ratio", linestyle="--")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Ratio")
    axes[2].set_title("Variance diagnostics")
    axes[2].legend()

    fig.tight_layout()
    target = plot_dir / f"{run_id}_stability.png"
    fig.savefig(target, dpi=200)
    plt.close(fig)


def _discover_runs(root: Path) -> List[Path]:
    runs = []
    for child in root.iterdir():
        if (child / "diagnostics" / "stability_metrics.csv").exists():
            runs.append(child)
    return sorted(runs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise and plot spectral diffusion stability metrics.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--runs", nargs="+", help="Explicit run directories to process.")
    group.add_argument("--root", type=str, help="Scan all run directories under this path.")
    parser.add_argument("--output", type=str, default="results/stability_summary.csv")
    parser.add_argument("--plot-dir", type=str, help="Optional directory for per-run diagnostic plots.")
    args = parser.parse_args()

    if args.runs:
        run_paths = [Path(p).resolve() for p in args.runs]
    else:
        run_paths = _discover_runs(Path(args.root).resolve())

    summaries: List[Dict[str, float]] = []
    for run_path in run_paths:
        csv_path = run_path / "diagnostics" / "stability_metrics.csv"
        if not csv_path.exists():
            continue
        rows = _load_stability_rows(csv_path)
        summaries.append(_summarise(run_path.name, rows))
        if args.plot_dir:
            noise_stats = _load_noise_stats(run_path / "diagnostics" / "noise_stats.json")
            _plot_run(run_path.name, rows, noise_stats, Path(args.plot_dir).resolve())

    if not summaries:
        raise SystemExit("No runs with stability metrics were found. Ensure training produced diagnostics/stability_metrics.csv.")

    _write_summary(Path(args.output).resolve(), summaries)


if __name__ == "__main__":
    main()

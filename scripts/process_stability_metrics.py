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

    summary = {
        "run_id": run_id,
        "snr_theory_avg": avg("snr_theory"),
        "snr_emp_avg": avg("snr_emp"),
        "snr_rel_avg": avg("snr_rel"),
        "snr_rel_std": pstdev(values("snr_rel")) if len(values("snr_rel")) > 1 else 0.0,
        "variance_sum_avg": avg("variance_sum"),
        "variance_sum_std": pstdev(values("variance_sum")) if len(values("variance_sum")) > 1 else 0.0,
        "grad_norm_avg": avg("grad_norm"),
        "noise_channel_std_min_avg": avg("noise_channel_std_min"),
        "noise_channel_std_max_avg": avg("noise_channel_std_max"),
    }
    return summary


def _write_summary(output_path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        raise ValueError("No runs to summarise")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "snr_theory_avg",
        "snr_emp_avg",
        "snr_rel_avg",
        "snr_rel_std",
        "variance_sum_avg",
        "variance_sum_std",
        "grad_norm_avg",
        "noise_channel_std_min_avg",
        "noise_channel_std_max_avg",
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
    snr_theory = [row.get("snr_theory", math.nan) for row in rows]
    snr_emp = [row.get("snr_emp", math.nan) for row in rows]
    snr_rel = [row.get("snr_rel", math.nan) for row in rows]
    variance_sum = [row.get("variance_sum", math.nan) for row in rows]
    grad_norm = [row.get("grad_norm", math.nan) for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(steps, snr_theory, label="snr_theory")
    axes[0].plot(steps, snr_emp, label="snr_emp", linestyle="--")
    axes[0].plot(steps, snr_rel, label="snr_rel", linestyle=":")
    axes[0].set_ylabel("SNR")
    axes[0].set_title(f"{run_id}: SNR statistics")
    axes[0].legend()

    axes[1].plot(steps, variance_sum, color="tomato")
    axes[1].set_ylabel("Variance sum")
    axes[1].set_title("Variance (signal+noise)")

    axes[2].plot(steps, grad_norm, label="grad_norm")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Grad norm")
    axes[2].set_title("Gradient diagnostics")
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

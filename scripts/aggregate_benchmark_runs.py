#!/usr/bin/env python3
"""Aggregate repeated benchmark runs and compute summary statistics.

The spectral benchmarks now execute multiple seeds per configuration. This
helper reads ``results/summary.csv`` (or another supplied path) and collapses
the per-run measurements into means/std/95% CI per variant so publication tables
can cite aggregate numbers instead of single runs.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _parse_variant(run_id: str, run_prefix: str) -> str | None:
    if not run_id.startswith(run_prefix):
        return None
    suffix = run_id[len(run_prefix) :]
    if suffix.startswith("_"):
        suffix = suffix[1:]
    if not suffix:
        return None
    # Expected format: <variant>_seed<seed>
    tokens = suffix.split("_seed")
    if not tokens or not tokens[0]:
        return None
    return tokens[0]


def _parse_seed(run_id: str) -> str | None:
    if "_seed" not in run_id:
        return None
    return run_id.split("_seed", 1)[-1]


def _load_rows(summary_path: Path, run_prefix: str) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with summary_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            run_id = row.get("run_id") or ""
            variant = _parse_variant(run_id, run_prefix)
            if not variant:
                continue
            grouped[variant].append(row)
    return grouped


def _compute_stats(values: Iterable[float]) -> Tuple[float, float, float]:
    samples = list(values)
    if not samples:
        return (math.nan, math.nan, math.nan)
    mean = statistics.fmean(samples)
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    if len(samples) > 0:
        sem = std / math.sqrt(len(samples))
    else:
        sem = math.nan
    ci95 = 1.96 * sem if not math.isnan(sem) else math.nan
    return mean, std, ci95


def aggregate(summary_path: Path, run_prefix: str, metrics: List[str]) -> None:
    grouped = _load_rows(summary_path, run_prefix)
    if not grouped:
        print(f"No runs found in {summary_path} matching prefix '{run_prefix}'.")
        return

    print(f"Aggregated metrics for runs with prefix '{run_prefix}':\n")
    for variant, rows in grouped.items():
        seeds = sorted(
            {seed for row in rows if (seed := _parse_seed(row.get("run_id", "")))}
        )
        print(f"Variant: {variant} (n={len(rows)}, seeds={', '.join(seeds) or 'n/a'})")
        for metric in metrics:
            samples: List[float] = []
            for row in rows:
                raw = row.get(metric)
                if raw is None or raw == "" or raw.lower() == "none":
                    continue
                try:
                    samples.append(float(raw))
                except ValueError:
                    continue
            if not samples:
                print(f"  {metric}: no data")
                continue
            mean, std, ci95 = _compute_stats(samples)
            print(
                f"  {metric}: mean={mean:.5f} std={std:.5f} 95%CI=±{ci95:.5f} (n={len(samples)})"
            )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate repeated benchmark runs.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/summary.csv"),
        help="Path to summary CSV file.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        required=True,
        help="Common prefix used for benchmark run IDs.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "loss_drop",
            "loss_drop_per_second",
            "runtime_seconds",
            "loss_threshold_steps",
            "loss_threshold_time",
        ],
        help="Metric columns to aggregate.",
    )
    args = parser.parse_args()
    aggregate(summary_path=args.summary, run_prefix=args.run_prefix, metrics=args.metrics)


if __name__ == "__main__":
    main()

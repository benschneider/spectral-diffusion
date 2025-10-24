#!/usr/bin/env python
"""Quick summary helper for Diffusion experiments.

Loads `results/summary.csv` (or a supplied path), prints available metrics,
shows the top-N runs for a chosen metric, and optionally merges Taguchi factor
metadata when present.
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

def load_summary(summary_path: Path) -> pd.DataFrame:
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    df = pd.read_csv(summary_path)
    if df.empty:
        raise ValueError("Summary CSV is empty")
    return df

def display_columns(df: pd.DataFrame) -> None:
    print("Available columns:")
    for col in df.columns:
        print(f"  - {col}")

def merge_taguchi(df: pd.DataFrame) -> pd.DataFrame:
    merged_rows = []
    for _, row in df.iterrows():
        cfg_path = Path(row.get("config_path", ""))
        factors = {}
        if cfg_path.exists():
            import yaml
            with cfg_path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            factors = (data.get("taguchi") or {}).get("row", {})
        new_row = row.to_dict()
        for key, value in factors.items():
            new_row[f"factor_{key}"] = value
        merged_rows.append(new_row)
    return pd.DataFrame(merged_rows)

def main(summary_file: Path, metric: Optional[str], top: int, include_factors: bool) -> None:
    df = load_summary(summary_file)

    # Try merging factor columns when requested
    if include_factors:
        try:
            df = merge_taguchi(df)
        except Exception as exc:
            print(f"Warning: unable to merge Taguchi factors ({exc})")

    display_columns(df)

    if metric is None or metric not in df.columns:
        if metric is not None:
            print(f"\nMetric '{metric}' not found. Showing first few rows instead.\n")
        print(df.head(top).to_string(index=False))
        return

    ascending = False
    if metric.startswith("eval_") or "fid" in metric.lower():
        ascending = True  # lower is better for fid/loss etc.

    print(f"\nTop {top} runs sorted by '{metric}' ({'ascending' if ascending else 'descending'})")
    subset = df.sort_values(metric, ascending=ascending).head(top)
    print(subset.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize results/summary.csv")
    parser.add_argument("--summary", type=Path, default=Path("results/summary.csv"))
    parser.add_argument("--metric", type=str, default=None, help="Metric column to rank")
    parser.add_argument("--top", type=int, default=5, help="Number of rows to display")
    parser.add_argument("--include-factors", action="store_true", help="Merge Taguchi factor columns")
    args = parser.parse_args()
    main(args.summary, args.metric, args.top, args.include_factors)

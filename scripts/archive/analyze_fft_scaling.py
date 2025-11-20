#!/usr/bin/env python
"""Analyze FFT scaling behavior across resolutions."""

import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_fft_scaling(results_dir: str = "results/fft_sweep"):
    """Analyze FFT scaling behavior and generate plots."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory {results_dir} not found. Run run_fft_benchmark_sweep.sh first.")
        return

    # Load all FFT benchmark results
    json_files = sorted(glob.glob(f"{results_dir}/fft_*.json"))
    if not json_files:
        print(f"No FFT benchmark files found in {results_dir}")
        return

    records = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                records.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    if not records:
        print("No valid FFT benchmark data found")
        return

    df = pd.DataFrame(records)

    # Compute scaling analysis
    df["log_size"] = np.log(df["size"])
    df["log_fft_time"] = np.log(df["fft_time"])

    # Fit power law: time ~ size^p
    if len(df) >= 3:
        p, _ = np.polyfit(df["log_size"], df["log_fft_time"], 1)
        print(".2f")
    else:
        p = None
        print("Insufficient data points for scaling analysis")

    # Create scaling plot
    plt.figure(figsize=(8, 6))
    plt.loglog(df["size"], df["fft_time"], "o-", linewidth=2, markersize=8)
    plt.xlabel("Image size (N)")
    plt.ylabel("FFT runtime (s)")
    if p is not None:
        plt.title(".2f")
    else:
        plt.title("FFT Runtime Scaling")
    plt.grid(True, which="both", alpha=0.3)

    # Add reference lines for common scaling laws
    size_range = np.logspace(np.log10(df["size"].min()), np.log10(df["size"].max()), 100)
    if p is not None:
        plt.loglog(size_range, df["fft_time"].iloc[0] * (size_range / df["size"].iloc[0])**p,
                  '--', alpha=0.7, label=".1f")

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{results_dir}/fft_scaling.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Save results
    df.to_csv(f"{results_dir}/fft_scaling.csv", index=False)
    print(f"FFT scaling analysis saved to {results_dir}/")

    return df

if __name__ == "__main__":
    analyze_fft_scaling()
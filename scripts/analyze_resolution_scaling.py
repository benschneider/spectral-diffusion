#!/usr/bin/env python
"""Analyze resolution scaling behavior for spectral vs baseline models."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_resolution_scaling(results_dir: str = "results/resolution_sweep"):
    """Analyze how runtime and efficiency scale with resolution."""
    results_path = Path(results_dir)
    summary_path = results_path / "summary.csv"

    if not summary_path.exists():
        print(f"Summary file {summary_path} not found. Run run_resolution_sweep.sh first.")
        return

    df = pd.read_csv(summary_path)

    # Filter for resolution sweep runs
    resolution_runs = df[df["run_id"].str.contains("baseline_|spectral_")].copy()

    if resolution_runs.empty:
        print("No resolution sweep runs found in summary")
        return

    # Extract resolution from run_id
    def extract_resolution(run_id):
        parts = run_id.split("_")
        if len(parts) >= 2:
            try:
                return int(parts[-1])
            except ValueError:
                pass
        return None

    resolution_runs["resolution"] = resolution_runs["run_id"].apply(extract_resolution)
    resolution_runs = resolution_runs.dropna(subset=["resolution"])

    if resolution_runs.empty:
        print("Could not extract resolution information from run IDs")
        return

    # Separate baseline and spectral runs
    baseline_runs = resolution_runs[resolution_runs["run_id"].str.contains("baseline_")]
    spectral_runs = resolution_runs[resolution_runs["run_id"].str.contains("spectral_")]

    # Compute scaling analysis
    for name, group_df in [("Baseline", baseline_runs), ("Spectral", spectral_runs)]:
        if len(group_df) < 3:
            print(f"Insufficient {name} runs for scaling analysis")
            continue

        group_df = group_df.copy()
        group_df["log_resolution"] = np.log(group_df["resolution"])
        group_df["log_runtime"] = np.log(group_df["runtime_seconds"])

        # Fit power law: runtime ~ resolution^p
        p, _ = np.polyfit(group_df["log_resolution"], group_df["log_runtime"], 1)
        print(".2f")

        # Compute efficiency if fit data available
        if "fit_k" in group_df.columns:
            group_df["efficiency"] = group_df["fit_k"] / group_df["runtime_seconds"]
            print(".4f")

    # Create scaling comparison plot
    plt.figure(figsize=(10, 6))

    if not baseline_runs.empty:
        plt.loglog(baseline_runs["resolution"], baseline_runs["runtime_seconds"],
                  "o-", label="Baseline", markersize=8)

    if not spectral_runs.empty:
        plt.loglog(spectral_runs["resolution"], spectral_runs["runtime_seconds"],
                  "s-", label="Spectral", markersize=8)

    plt.xlabel("Image Resolution (N)")
    plt.ylabel("Runtime (s)")
    plt.title("Runtime Scaling: Baseline vs Spectral Models")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/resolution_scaling.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Save detailed results
    resolution_runs.to_csv(f"{results_dir}/resolution_scaling_detailed.csv", index=False)
    print(f"Resolution scaling analysis saved to {results_dir}/")

    return resolution_runs

if __name__ == "__main__":
    analyze_resolution_scaling()
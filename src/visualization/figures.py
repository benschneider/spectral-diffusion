"""Figure generation utilities for Spectral Diffusion."""

# --- Imports for orchestration only ---
from pathlib import Path
from typing import Optional
import json
import pandas as pd
from datetime import datetime, timezone

from .plots import *
from .analysis_utils import *
from .report import write_summary_markdown

# Re-export functions that tests expect to find here
from .analysis_utils import collect_loss_histories
from .plots import plot_taguchi_metric_distribution, _setup_style, _color_palette

# Re-export additional internal functions for testing
def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path)

def _fft_benchmark_snapshot(path='results/fft_sweep/fft_scaling.csv'):
    """Load FFT scaling benchmark data from CSV if present."""
    import os
    if os.path.exists(path):
        df = pd.read_csv(path)
        return df
    else:
        return None


def generate_figures(
    synthetic_dir: Path,
    cifar_dir: Path,
    taguchi_dir: Path,
    output_dir: Path,
    descriptions_path: Optional[Path] = None,
    generated_at: Optional[str] = None,
) -> None:
    """Load benchmark data, render plots, and write markdown summary."""
    _setup_style()
    _ensure_output_dir(output_dir)

    # Load CSVs
    synthetic_df = _load_csv(synthetic_dir / "summary.csv") if synthetic_dir else None
    cifar_df = _load_csv(cifar_dir / "summary.csv") if cifar_dir else None
    taguchi_report = _load_csv(taguchi_dir / "taguchi_report.csv") if taguchi_dir else None

    # Load descriptions.json if present
    descriptions = {}
    if descriptions_path and descriptions_path.exists():
        data = json.loads(descriptions_path.read_text())
        descriptions = {
            "synthetic_title": data.get("synthetic_benchmark", {}).get("title", ""),
            "synthetic_text": data.get("synthetic_benchmark", {}).get("description", ""),
            "cifar_title": data.get("cifar_benchmark", {}).get("title", ""),
            "cifar_text": data.get("cifar_benchmark", {}).get("description", ""),
            "taguchi_title": data.get("taguchi_analysis", {}).get("title", ""),
            "taguchi_text": data.get("taguchi_analysis", {}).get("description", ""),
            "taguchi_choices": data.get("taguchi_choices", {}),
        }
    else:
        descriptions["taguchi_choices"] = {}

    # FFT benchmark snapshot
    fft_snapshot = _fft_benchmark_snapshot()

    # Compute FFT-corrected runtime columns if needed
    if synthetic_df is not None:
        synthetic_df = compute_fft_corrected(synthetic_df)
    if cifar_df is not None:
        cifar_df = compute_fft_corrected(cifar_df)

    # Plotting map: (function, args, out_filename)
    plot_map = [
        # Synthetic
        (plot_loss_metrics, [synthetic_df, "Synthetic Benchmark – Loss Metrics"], "loss_metrics_synthetic.png"),
        (plot_runtime_metrics, [synthetic_df, "Synthetic Benchmark – Runtime Metrics"], "runtime_metrics_synthetic.png"),
        (plot_tradeoff_scatter, [synthetic_df, "images_per_second", "loss_final",
                                 "Synthetic Benchmark – Loss vs Throughput",
                                 "Images per Second (Higher is Better)", "Final Loss (Lower is Better)"], "tradeoff_loss_vs_speed_synthetic.png"),
        (plot_metric_boxplot, [synthetic_df, "loss_final", "Synthetic Benchmark – Final Loss Distribution", "Final Loss"], "loss_final_distribution_synthetic.png"),
        (plot_metric_boxplot, [synthetic_df, "images_per_second", "Synthetic Benchmark – Throughput Distribution", "Images per Second"], "images_per_second_distribution_synthetic.png"),
        # CIFAR
        (plot_loss_metrics, [cifar_df, "CIFAR-10 Benchmark – Loss Metrics"], "loss_metrics_cifar.png"),
        (plot_runtime_metrics, [cifar_df, "CIFAR-10 Benchmark – Runtime Metrics"], "runtime_metrics_cifar.png"),
        (plot_tradeoff_scatter, [cifar_df, "images_per_second", "loss_final",
                                 "CIFAR-10 Benchmark – Loss vs Throughput",
                                 "Images per Second (Higher is Better)", "Final Loss (Lower is Better)"], "tradeoff_loss_vs_speed_cifar.png"),
        (plot_metric_boxplot, [cifar_df, "loss_final", "CIFAR-10 Benchmark – Final Loss Distribution", "Final Loss"], "loss_final_distribution_cifar.png"),
        (plot_metric_boxplot, [cifar_df, "images_per_second", "CIFAR-10 Benchmark – Throughput Distribution", "Images per Second"], "images_per_second_distribution_cifar.png"),
        # Taguchi
        (plot_taguchi_snr, [taguchi_report], "taguchi_snr.png"),
        (plot_taguchi_metric_distribution, [taguchi_report, "loss_drop_per_second"], "taguchi_loss_drop_per_second.png"),
    ]

    # Plot figures
    for func, args, fname in plot_map:
        # Only call if the first arg (df) is not None
        if args and args[0] is not None:
            # For taguchi plots, append descriptions if needed
            if func is plot_taguchi_snr:
                func(args[0], out_path=output_dir / fname, descriptions=descriptions)
                continue
            if func is plot_taguchi_metric_distribution:
                # Append descriptions as optional kwarg
                func(*args, out_path=output_dir / fname, descriptions=descriptions)
                continue
            func(*args, out_path=output_dir / fname)

    # Loss curves (special: need histories)
    for df, label, fname in [
        (synthetic_df, "Synthetic Benchmark – Loss Curves", "loss_curve_synthetic.png"),
        (cifar_df, "CIFAR-10 Benchmark – Loss Curves", "loss_curve_cifar.png"),
    ]:
        if df is not None:
            histories = collect_loss_histories(df)
            if histories:
                plot_loss_curves(histories, label, output_dir / fname)

    # Write summary markdown
    timestamp = generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_summary_markdown(
        synthetic_df,
        cifar_df,
        taguchi_report,
        output_dir / "summary.md",
        descriptions,
        generated_at=timestamp,
        fft_snapshot=fft_snapshot,
    )

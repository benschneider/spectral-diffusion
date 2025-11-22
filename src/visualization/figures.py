"""Figure generation utilities for Spectral Diffusion."""

# --- Imports for orchestration only ---
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis_utils import *
from .plots import *
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


def _load_image_array(path: Path) -> Optional[np.ndarray]:
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - optional dependency
        return None

    if not path.exists():
        return None

    try:
        with Image.open(path) as img:
            return np.array(img.convert("RGB"))
    except Exception:
        return None


def _render_triptych(
    images: Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]],
    titles: Tuple[str, str, str],
    suptitle: str,
    out_path: Path,
    footer: Optional[str] = None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.5))
    for ax, image, title in zip(axes, images, titles):
        ax.axis("off")
        if image is None:
            ax.set_facecolor("#f5f5f5")
            ax.text(0.5, 0.5, "Not available", ha="center", va="center", fontsize=7)
        else:
            ax.imshow(image)
        ax.set_title(title, fontsize=8)
    fig.suptitle(suptitle, fontsize=9)
    if footer:
        # Draw a simple footer bar with metadata text.
        fig.subplots_adjust(bottom=0.18, top=0.9)
        rect = plt.Rectangle((0, 0), 1, 0.12, transform=fig.transFigure, color="white", alpha=0.9)
        fig.patches.append(rect)
        fig.text(0.5, 0.06, footer, ha="center", va="center", fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _attach_run_dirs(df: Optional[pd.DataFrame], dataset_dir: Path) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return df

    df = df.copy()
    run_root = dataset_dir / "runs"
    if "run_id" not in df.columns:
        return df

    run_dirs: list[Optional[str]] = []
    for run_id in df["run_id"]:
        if not isinstance(run_id, str):
            run_dirs.append(None)
            continue
        candidate = run_root / run_id
        run_dirs.append(str(candidate) if candidate.exists() else None)

    df["run_dir"] = run_dirs
    return df


def _select_representative_run(
    df: Optional[pd.DataFrame],
    prefer_col: str,
    prefer_mode: str = "min",
) -> Optional[pd.Series]:
    if df is None or df.empty or prefer_col not in df.columns:
        return None

    try:
        numeric = pd.to_numeric(df[prefer_col], errors="coerce")
    except Exception:
        numeric = df[prefer_col]

    order = numeric
    if getattr(order, "isna", None):
        candidate = df.loc[~order.isna()]
    else:
        candidate = df

    if candidate.empty:
        candidate = df

    try:
        if prefer_mode == "max":
            idx = candidate[prefer_col].astype(float).idxmax()
        else:
            idx = candidate[prefer_col].astype(float).idxmin()
    except Exception:
        idx = candidate.index[0]

    return df.loc[idx]


def _run_mapping_for_row(row: pd.Series) -> Dict[str, str]:
    run_key = row.get("run_axis", row.get("run_id", "run"))
    if run_key is None or (isinstance(run_key, float) and np.isnan(run_key)):
        run_key = "run"
    context = row.get("run_axis_context", run_key)
    if context is None or (isinstance(context, float) and np.isnan(context)):
        context = run_key
    return {str(run_key): str(context)}


def _create_prediction_visual(
    df: Optional[pd.DataFrame],
    dataset_label: str,
    out_path: Path,
) -> Optional[Dict[str, Any]]:
    row = _select_representative_run(df, "loss_final", prefer_mode="min")
    if row is None:
        return None

    run_dir = row.get("run_dir")
    if run_dir is None:
        images = (None, None, None)
    else:
        run_path = Path(run_dir)
        sample_dir = next((p for p in run_path.glob("samples/*") if p.is_dir()), None)
        if sample_dir is None:
            images = (None, None, None)
        else:
            inputs = sorted(sample_dir.glob("*input*.png"))
            preds = sorted(sample_dir.glob("*sample*.png"))
            targets = sorted(sample_dir.glob("*target*.png"))
            images = (
                _load_image_array(inputs[0]) if inputs else None,
                _load_image_array(preds[0]) if preds else None,
                _load_image_array(targets[0]) if targets else None,
            )

    footer_parts = []
    run_name = str(row.get("run_id") or row.get("display_name") or "run")
    footer_parts.append(run_name)
    snr_val = row.get("snr_ratio")
    if snr_val is not None and not (isinstance(snr_val, float) and np.isnan(snr_val)):
        footer_parts.append(f"snr_ratio={snr_val}")
    adapter = row.get("spectral_adapter_placement")
    if adapter is None:
        adapter = row.get("adapter")
    if adapter:
        footer_parts.append(f"adapter={adapter}")
    footer = " | ".join(footer_parts)

    _render_triptych(
        images,
        ("Input", "Prediction", "Target"),
        f"Representative predictions ({dataset_label})",
        out_path,
        footer=footer,
    )

    if images[1] is None and images[2] is None:
        note = "predictions_missing"
    elif images[1] is None:
        note = "predictions_missing"
    elif images[2] is None:
        note = "targets_missing"
    else:
        note = ""

    return {
        "dataset": dataset_label,
        "run_mapping": _run_mapping_for_row(row),
        "notes": note,
    }


def _create_noising_visual(
    df: Optional[pd.DataFrame],
    dataset_label: str,
    out_path: Path,
) -> Optional[Dict[str, Any]]:
    row = _select_representative_run(df, "loss_drop_per_second", prefer_mode="max")
    if row is None:
        return None

    run_dir = row.get("run_dir")
    if run_dir is None:
        stages = (None, None, None)
    else:
        run_path = Path(run_dir)
        stage_files = {
            "clean": list(run_path.glob("**/*t0*.png")) + list(run_path.glob("**/*clean*.png")),
            "mid": list(run_path.glob("**/*t50*.png")) + list(run_path.glob("**/*mid*.png")),
            "noisy": list(run_path.glob("**/*t100*.png")) + list(run_path.glob("**/*noisy*.png")),
        }
        stages = (
            _load_image_array(stage_files["clean"][0]) if stage_files["clean"] else None,
            _load_image_array(stage_files["mid"][0]) if stage_files["mid"] else None,
            _load_image_array(stage_files["noisy"][0]) if stage_files["noisy"] else None,
        )

    _render_triptych(
        stages,
        ("Clean sample", "Intermediate", "Fully noised"),
        f"Noising trajectory ({dataset_label})",
        out_path,
    )

    missing = any(stage is None for stage in stages)
    return {
        "dataset": dataset_label,
        "run_mapping": _run_mapping_for_row(row),
        "notes": "noising_missing" if missing else "",
    }
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
    ablation_dir: Optional[Path] = None,
) -> None:
    """Load benchmark data, render plots, and write markdown summary."""
    _setup_style()
    _ensure_output_dir(output_dir)

    # Load CSVs
    synthetic_df = _load_csv(synthetic_dir / "summary.csv") if synthetic_dir else None
    cifar_df = _load_csv(cifar_dir / "summary.csv") if cifar_dir else None
    taguchi_report = _load_csv(taguchi_dir / "taguchi_report.csv") if taguchi_dir else None
    ablation_df = _load_csv(Path(ablation_dir) / "summary.csv") if ablation_dir else None

    metric_notes: Dict[str, Dict[str, bool]] = {}

    synthetic_df, synth_notes = sanitize_metric_frame(synthetic_df)
    if synth_notes:
        metric_notes["synthetic"] = synth_notes
    cifar_df, cifar_notes = sanitize_metric_frame(cifar_df)
    if cifar_notes:
        metric_notes["cifar"] = cifar_notes
    ablation_df, ablation_notes = sanitize_metric_frame(ablation_df)
    if ablation_notes:
        metric_notes["ablation"] = ablation_notes

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
        synthetic_df, _ = assign_run_axis(synthetic_df)
        synthetic_df = _attach_run_dirs(synthetic_df, synthetic_dir)
    if cifar_df is not None:
        cifar_df = compute_fft_corrected(cifar_df)
        cifar_df, _ = assign_run_axis(cifar_df)
        cifar_df = _attach_run_dirs(cifar_df, cifar_dir)
    if ablation_df is not None:
        ablation_df = compute_fft_corrected(ablation_df)
        ablation_df, _ = assign_run_axis(ablation_df)
        if ablation_dir:
            ablation_df = _attach_run_dirs(ablation_df, Path(ablation_dir))

    # Plotting map: (function, args, out_filename)
    plot_map = [
        # Synthetic
        (plot_loss_metrics, [synthetic_df, "Synthetic Benchmark – Loss Metrics"], "loss_metrics_synthetic.png", "synthetic"),
        (plot_runtime_metrics, [synthetic_df, "Synthetic Benchmark – Runtime Metrics"], "runtime_metrics_synthetic.png", "synthetic"),
        (
            plot_tradeoff_scatter,
            [
                synthetic_df,
                "images_per_second",
                "loss_final",
                "Synthetic Benchmark – Loss vs Throughput",
                "Images per Second (Higher is Better)",
                "Final Loss (Lower is Better)",
            ],
            "tradeoff_loss_vs_speed_synthetic.png",
            "synthetic",
        ),
        (
            plot_metric_boxplot,
            [synthetic_df, "loss_final", "Synthetic Benchmark – Final Loss Distribution", "Final Loss"],
            "loss_final_distribution_synthetic.png",
            "synthetic",
        ),
        (
            plot_metric_boxplot,
            [synthetic_df, "images_per_second", "Synthetic Benchmark – Throughput Distribution", "Images per Second"],
            "images_per_second_distribution_synthetic.png",
            "synthetic",
        ),
        # CIFAR
        (plot_loss_metrics, [cifar_df, "CIFAR-10 Benchmark – Loss Metrics"], "loss_metrics_cifar.png", "cifar"),
        (plot_runtime_metrics, [cifar_df, "CIFAR-10 Benchmark – Runtime Metrics"], "runtime_metrics_cifar.png", "cifar"),
        (
            plot_tradeoff_scatter,
            [
                cifar_df,
                "images_per_second",
                "loss_final",
                "CIFAR-10 Benchmark – Loss vs Throughput",
                "Images per Second (Higher is Better)",
                "Final Loss (Lower is Better)",
            ],
            "tradeoff_loss_vs_speed_cifar.png",
            "cifar",
        ),
        (
            plot_metric_boxplot,
            [cifar_df, "loss_final", "CIFAR-10 Benchmark – Final Loss Distribution", "Final Loss"],
            "loss_final_distribution_cifar.png",
            "cifar",
        ),
        (
            plot_metric_boxplot,
            [cifar_df, "images_per_second", "CIFAR-10 Benchmark – Throughput Distribution", "Images per Second"],
            "images_per_second_distribution_cifar.png",
            "cifar",
        ),
        # Taguchi
        (plot_taguchi_snr, [taguchi_report], "taguchi_snr.png", "taguchi"),
        (plot_taguchi_metric_distribution, [taguchi_report, "loss_drop_per_second"], "taguchi_loss_drop_per_second.png", "taguchi"),
    ]

    # Plot figures
    figure_metadata: Dict[str, Dict[str, Any]] = {}

    for func, args, fname, dataset_key in plot_map:
        # Only call if the first arg (df) is not None
        if args and args[0] is not None:
            out_path = output_dir / fname
            # For taguchi plots, append descriptions if needed
            if func is plot_taguchi_snr:
                func(args[0], out_path=out_path, descriptions=descriptions)
                figure_metadata.setdefault(fname, {})["dataset"] = dataset_key
                continue
            if func is plot_taguchi_metric_distribution:
                # Append descriptions as optional kwarg
                func(*args, out_path=out_path, descriptions=descriptions)
                figure_metadata.setdefault(fname, {})["dataset"] = dataset_key
                continue
            result = func(*args, out_path=out_path)
            entry = figure_metadata.setdefault(fname, {})
            entry.setdefault("dataset", dataset_key)
            if isinstance(result, dict) and result:
                entry["run_mapping"] = result

    if ablation_df is not None:
        plot_feature_toggle_ablation(ablation_df, output_dir / "spectral_feature_ablation.png")
        figure_metadata.setdefault("spectral_feature_ablation.png", {})["dataset"] = "ablation"

    # Loss curves (special: need histories)
    for df, label, fname in [
        (synthetic_df, "Synthetic Benchmark – Loss Curves", "loss_curve_synthetic.png"),
        (cifar_df, "CIFAR-10 Benchmark – Loss Curves", "loss_curve_cifar.png"),
    ]:
        if df is not None:
            histories = collect_loss_histories(df)
            if histories:
                plot_loss_curves(histories, label, output_dir / fname)
                figure_metadata.setdefault(fname, {})["dataset"] = "synthetic" if "synthetic" in fname else "cifar"

    # Prediction and noising triptychs
    dataset_labels = {"synthetic": "Synthetic 32×32", "cifar": "CIFAR-10"}

    for key, df in [
        ("synthetic", synthetic_df),
        ("cifar", cifar_df),
    ]:
        if df is None:
            continue
        pred_name = f"{key}_predictions.png"
        pred_meta = _create_prediction_visual(df, dataset_labels.get(key, key.capitalize()), output_dir / pred_name)
        if pred_meta:
            figure_metadata[pred_name] = pred_meta
        noise_name = f"{key}_noising_chain.png"
        noise_meta = _create_noising_visual(df, dataset_labels.get(key, key.capitalize()), output_dir / noise_name)
        if noise_meta:
            figure_metadata[noise_name] = noise_meta

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
        taguchi_dir=taguchi_dir,
        figure_metadata=figure_metadata,
        metric_notes=metric_notes,
    )

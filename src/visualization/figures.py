"""Figure generation utilities for Spectral Diffusion."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Please install matplotlib to generate figures.") from exc


def _setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.dpi": 300,
        }
    )


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        logging.warning("Skipping missing file: %s", path)
        return None
    return pd.read_csv(path)


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _color_palette(n: int) -> list[str]:
    cmap = plt.get_cmap("tab10")
    return [cmap(i % cmap.N) for i in range(n)]


def _bar_ann(ax, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _label_series(df: pd.DataFrame) -> pd.Series:
    if "display_name" in df.columns:
        return df["display_name"]
    return df["run_id"]


def plot_loss_metrics(df: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    colors = _color_palette(len(df))
    x = np.arange(len(df))

    bars1 = axes[0].bar(x, df["loss_drop"], color=colors, width=0.6)
    axes[0].set(title="Loss Drop (initial - final)", xlabel="Model", ylabel="Loss Drop")
    axes[0].set_xticks(x, _label_series(df), rotation=20)
    _bar_ann(axes[0], bars1)

    bars2 = axes[1].bar(x, df["loss_final"], color=colors, width=0.6)
    axes[1].set(title="Final Loss", xlabel="Model", ylabel="Loss")
    axes[1].set_xticks(x, _label_series(df), rotation=20)
    _bar_ann(axes[1], bars2)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_runtime_metrics(df: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    colors = _color_palette(len(df))
    x = np.arange(len(df))

    metrics = [
        ("runtime_seconds", "Runtime (s)"),
        ("steps_per_second", "Training Steps / s"),
        ("images_per_second", "Images / s"),
    ]

    for ax, (col, label) in zip(axes, metrics):
        bars = ax.bar(x, df[col], color=colors, width=0.6)
        ax.set(title=label, xlabel="Model")
        ax.set_xticks(x, _label_series(df), rotation=20)
        _bar_ann(ax, bars)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_taguchi_snr(df: pd.DataFrame, out_path: Path, descriptions: dict[str, str]) -> None:
    factors = df["factor"].unique()
    levels = sorted(df["level"].unique())
    width = 0.8 / max(len(levels), 1)
    x = np.arange(len(factors))
    colors = _color_palette(len(levels))

    fig, ax = plt.subplots(figsize=(7, 4))
    for idx, level in enumerate(levels):
        values = []
        for factor in factors:
            row = df[(df["factor"] == factor) & (df["level"] == level)]
            values.append(row["snr"].values[0] if not row.empty else np.nan)
        offsets = x + (idx - (len(levels) - 1) / 2) * width
        bars = ax.bar(offsets, values, width=width, color=colors[idx], label=f"Level {level}")
        _bar_ann(ax, bars)

    choices_map = descriptions.get("taguchi_choices", {})
    labels = [f"{factor}\n{choices_map.get(factor, '')}".strip() for factor in factors]

    ax.set(title="Taguchi Signal-to-Noise Ratios", xlabel="Factor", ylabel="S/N (dB)")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.legend(title="Level", loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_taguchi_metric_distribution(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    factor_cols = [c for c in df.columns if c.startswith("factor_")]
    if not factor_cols:
        logging.warning("No factor columns found for Taguchi distribution plot.")
        return

    fig, axes = plt.subplots(1, len(factor_cols), figsize=(4 * len(factor_cols), 4), sharey=True)
    if len(factor_cols) == 1:
        axes = [axes]

    for ax, factor in zip(axes, factor_cols):
        data = df[[metric, factor]].copy()
        levels = sorted(data[factor].unique())
        positions = np.arange(len(levels))
        box_data = [data.loc[data[factor] == level, metric].values for level in levels]
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor="#9ecae1", color="#3182bd"),
            medianprops=dict(color="#08519c"),
        )
        for patch in bp["boxes"]:
            patch.set_alpha(0.7)
        ax.set(title=factor.replace("factor_", "").title(), xlabel="Level")
        ax.set_xticks(positions, [str(lvl) for lvl in levels])
        if ax is axes[0]:
            ax.set_ylabel(metric)

    fig.suptitle(f"{metric} Distribution by Taguchi Factor", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_summary_markdown(
    synthetic_df: Optional[pd.DataFrame],
    cifar_df: Optional[pd.DataFrame],
    taguchi_report: Optional[pd.DataFrame],
    out_path: Path,
    descriptions: dict[str, str],
) -> None:
    lines = ["# Results Summary", ""]
    if synthetic_df is not None:
        title = descriptions.get("synthetic_title", "Synthetic Benchmark")
        desc = descriptions.get("synthetic_text", "")
        lines.append(f"## {title}")
        if desc:
            lines.extend([desc, ""])
        headers = ["Run", "Loss Drop", "Final Loss", "Images/s", "Runtime (s)"]
        if "eval_fid" in synthetic_df.columns:
            headers.append("FID")
        if "eval_lpips" in synthetic_df.columns:
            headers.append("LPIPS")
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        labels = _label_series(synthetic_df)
        for (_, row), label in zip(synthetic_df.iterrows(), labels):
            values = [
                label,
                f"{row['loss_drop']:.3f}",
                f"{row['loss_final']:.3f}",
                f"{row['images_per_second']:.1f}",
                f"{row['runtime_seconds']:.1f}",
            ]
            if "eval_fid" in synthetic_df.columns:
                values.append(
                    f"{row['eval_fid']:.3f}" if not pd.isna(row.get("eval_fid")) else "–"
                )
            if "eval_lpips" in synthetic_df.columns:
                values.append(
                    f"{row['eval_lpips']:.3f}" if not pd.isna(row.get("eval_lpips")) else "–"
                )
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    if cifar_df is not None:
        title = descriptions.get("cifar_title", "CIFAR-10 Benchmark")
        desc = descriptions.get("cifar_text", "")
        lines.append(f"## {title}")
        if desc:
            lines.extend([desc, ""])
        headers = ["Run", "Loss Drop", "Final Loss", "Images/s", "Runtime (s)"]
        if "eval_fid" in cifar_df.columns:
            headers.append("FID")
        if "eval_lpips" in cifar_df.columns:
            headers.append("LPIPS")
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        labels = _label_series(cifar_df)
        for (_, row), label in zip(cifar_df.iterrows(), labels):
            values = [
                label,
                f"{row['loss_drop']:.3f}",
                f"{row['loss_final']:.3f}",
                f"{row['images_per_second']:.1f}",
                f"{row['runtime_seconds']:.1f}",
            ]
            if "eval_fid" in cifar_df.columns:
                values.append(
                    f"{row['eval_fid']:.3f}" if not pd.isna(row.get("eval_fid")) else "–"
                )
            if "eval_lpips" in cifar_df.columns:
                values.append(
                    f"{row['eval_lpips']:.3f}" if not pd.isna(row.get("eval_lpips")) else "–"
                )
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")

    if taguchi_report is not None:
        title = descriptions.get("taguchi_title", "Taguchi Factors (Top 5)")
        desc = descriptions.get("taguchi_text", "")
        lines.append(f"## {title}")
        if desc:
            lines.extend([desc, ""])
        top = taguchi_report.sort_values("rank").head(5)
        lines.extend(
            [
                "| Rank | Factor | Level | S/N (dB) |",
                "| --- | --- | --- | --- |",
            ]
        )
        for _, row in top.iterrows():
            lines.append(
                f"| {int(row['rank'])} | {row['factor']} | {row['level']} | {row['snr']:.2f} |"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Wrote summary markdown to %s", out_path)


def generate_figures(
    synthetic_dir: Path,
    cifar_dir: Path,
    taguchi_dir: Path,
    output_dir: Path,
    descriptions_path: Optional[Path] = None,
) -> None:
    """Load benchmark data, render plots, and write markdown summary."""
    _setup_style()
    _ensure_output_dir(output_dir)

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
        }

    synthetic_df = _load_csv(synthetic_dir / "summary.csv") if synthetic_dir else None
    cifar_df = _load_csv(cifar_dir / "summary.csv") if cifar_dir else None
    taguchi_summary = _load_csv(taguchi_dir / "summary.csv") if taguchi_dir else None
    taguchi_report = _load_csv(taguchi_dir / "taguchi_report.csv") if taguchi_dir else None

    if synthetic_df is not None:
        plot_loss_metrics(
            synthetic_df,
            "Synthetic Benchmark – Loss Metrics",
            output_dir / "loss_metrics_synthetic.png",
        )
        plot_runtime_metrics(
            synthetic_df,
            "Synthetic Benchmark – Runtime Metrics",
            output_dir / "runtime_metrics_synthetic.png",
        )

    if cifar_df is not None:
        plot_loss_metrics(
            cifar_df,
            "CIFAR-10 Benchmark – Loss Metrics",
            output_dir / "loss_metrics_cifar.png",
        )
        plot_runtime_metrics(
            cifar_df,
            "CIFAR-10 Benchmark – Runtime Metrics",
            output_dir / "runtime_metrics_cifar.png",
        )

    if taguchi_report is not None:
        plot_taguchi_snr(taguchi_report, output_dir / "taguchi_snr.png", descriptions)
    if taguchi_summary is not None and taguchi_report is not None:
        plot_taguchi_metric_distribution(
            taguchi_summary,
            metric="loss_drop_per_second",
            out_path=output_dir / "taguchi_loss_drop_per_second.png",
        )

    write_summary_markdown(
        synthetic_df,
        cifar_df,
        taguchi_report,
        output_dir / "summary.md",
        descriptions,
    )

"""Figure generation utilities for Spectral Diffusion."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
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


DEFAULT_TAGUCHI_LABELS: dict[str, str] = {
    "A": "Freq-equalized noise",
    "B": "Spectral attention",
    "C": "Sampler",
    "D": "Spectral adapters",
}


def _extract_level_options(text: str) -> list[str]:
    if "(" not in text or ")" not in text:
        return []
    raw = text[text.find("(") + 1 : text.find(")")]
    return [opt.strip() for opt in raw.split("/") if opt.strip()]


def _taguchi_level_label(
    factor: str, level: int, choices_map: dict[str, str], default: str = ""
) -> str:
    choice_text = choices_map.get(factor, "")
    if not choice_text and factor.startswith("factor_"):
        choice_text = choices_map.get(factor.replace("factor_", ""), "")
    options = _extract_level_options(choice_text)
    idx = level - 1
    if options and 0 <= idx < len(options):
        return options[idx]
    if choice_text and "(" not in choice_text:
        return f"{choice_text} #{level}"
    return default or f"Level {level}"


def _safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _benchmark_takeaways(df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if df.empty:
        return lines

    labeled = df.copy()
    labeled["_label"] = _label_series(df).values

    def _fmt_row(row: pd.Series, col: str, precision: int = 3) -> str:
        return f"{row['_label']} ({row[col]:.{precision}f})"

    if "loss_final" in labeled.columns and labeled["loss_final"].notna().any():
        best_loss = labeled.loc[labeled["loss_final"].idxmin()]
        lines.append(f"- Lowest final loss: {_fmt_row(best_loss, 'loss_final')}")

    if "images_per_second" in labeled.columns and labeled["images_per_second"].notna().any():
        fastest = labeled.loc[labeled["images_per_second"].idxmax()]
        lines.append(f"- Fastest throughput: {_fmt_row(fastest, 'images_per_second', precision=1)} images/s")

    if (
        len(labeled) > 1
        and "loss_final" in labeled.columns
        and "images_per_second" in labeled.columns
        and labeled["loss_final"].notna().all()
        and labeled["images_per_second"].notna().all()
    ):
        best_loss = labeled.loc[labeled["loss_final"].idxmin()]
        fastest = labeled.loc[labeled["images_per_second"].idxmax()]
        if best_loss["_label"] != fastest["_label"]:
            speed_gap = _safe_ratio(fastest["images_per_second"], best_loss["images_per_second"])
            loss_gap = best_loss["loss_final"] - fastest["loss_final"]
            gap_bits = []
            if speed_gap is not None and speed_gap > 0:
                gap_bits.append(f"{speed_gap:.1f}× faster")
            gap_bits.append(
                f"Δ loss {loss_gap:+.3f}" if loss_gap != 0 else "similar loss"
            )
            lines.append(
                f"- Trade-off: {fastest['_label']} vs {best_loss['_label']} → "
                + ", ".join(gap_bits)
            )

    if (
        "loss_drop_per_second" in labeled.columns
        and labeled["loss_drop_per_second"].notna().any()
    ):
        best_drop = labeled.loc[labeled["loss_drop_per_second"].idxmax()]
        lines.append(
            "- Fastest convergence: "
            f"{_fmt_row(best_drop, 'loss_drop_per_second', precision=3)} loss drop/s"
        )
    return lines


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


def plot_tradeoff_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    out_path: Path,
) -> None:
    if x_col not in df.columns or y_col not in df.columns:
        logging.warning("Skipping trade-off plot; missing columns %s or %s", x_col, y_col)
        return

    labeled = df.copy()
    labeled["_label"] = _label_series(df).values
    groups = labeled.groupby("_label")
    colors = _color_palette(len(groups))

    fig, ax = plt.subplots(figsize=(6, 4))
    for color, (label, group) in zip(colors, groups):
        ax.scatter(
            group[x_col],
            group[y_col],
            label=label,
            color=color,
            s=64,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )
        if len(group) == 1 and len(df) <= 6:
            row = group.iloc[0]
            ax.annotate(
                label,
                xy=(row[x_col], row[y_col]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                ha="left",
                va="bottom",
            )

    ax.set(title=title, xlabel=x_label, ylabel=y_label)
    ax.grid(True, linestyle="--", alpha=0.4)
    if len(groups) > 1:
        ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_metric_boxplot(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    if metric not in df.columns:
        logging.warning("Skipping box plot; metric '%s' missing", metric)
        return

    labeled = df.copy()
    labeled["_label"] = _label_series(df).values
    groups = [(label, grp[metric].dropna()) for label, grp in labeled.groupby("_label")]
    groups = [(label, values) for label, values in groups if len(values) > 0]

    if not groups:
        logging.warning("Skipping box plot; metric '%s' has no valid values", metric)
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    positions = np.arange(len(groups))
    box_data = [values.values for _, values in groups]
    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="#9ecae1", color="#3182bd"),
        medianprops=dict(color="#08519c", linewidth=1.5),
        whiskerprops=dict(color="#3182bd"),
        capprops=dict(color="#3182bd"),
        flierprops=dict(markerfacecolor="#3182bd", markeredgecolor="#08519c", markersize=5),
    )
    for patch in bp["boxes"]:
        patch.set_alpha(0.75)

    # Jitter individual observations to show sample counts.
    for pos, (_, values) in zip(positions, groups):
        if len(values) == 1:
            jitter = np.array([0.0])
        else:
            jitter = np.linspace(-0.08, 0.08, len(values))
        ax.scatter(
            np.full(len(values), pos) + jitter,
            values,
            color="#08519c",
            alpha=0.6,
            s=26,
        )

    ax.set(title=title, ylabel=ylabel)
    ax.set_xticks(positions, [label for label, _ in groups], rotation=20, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_taguchi_snr(df: pd.DataFrame, out_path: Path, descriptions: dict[str, str]) -> None:
    factors = sorted(df["factor"].unique())
    if not factors:
        logging.warning("No Taguchi factors found for S/N plot.")
        return

    levels = sorted(df["level"].unique())
    colors = _color_palette(len(factors))
    fig_width = max(6, 3 * len(factors))
    fig, axes = plt.subplots(1, len(factors), figsize=(fig_width, 4), sharey=True)
    if len(factors) == 1:
        axes = [axes]

    choices_map = {
        **DEFAULT_TAGUCHI_LABELS,
        **(descriptions.get("taguchi_choices", {}) or {}),
    }

    any_plotted = False
    for idx, (factor, ax) in enumerate(zip(factors, axes)):
        factor_df = df[df["factor"] == factor].copy()
        factor_df = factor_df.sort_values("level")
        values = [factor_df.loc[factor_df["level"] == lvl, "snr"].mean() for lvl in levels]
        values_arr = np.array(values, dtype=float)
        if np.all(np.isnan(values_arr)):
            logging.warning("All NaN S/N values for factor '%s'; skipping plot.", factor)
            ax.set_visible(False)
            continue

        positions = np.arange(len(levels))
        any_plotted = True

        x_labels = [
            _taguchi_level_label(factor, int(level), choices_map, default=f"Level {level}")
            for level in levels
        ]

        ax.plot(
            positions,
            values_arr,
            color=colors[idx],
            marker="o",
            linewidth=2.0,
            markersize=6,
        )
        best_idx = int(np.nanargmax(values_arr))
        best_label = x_labels[best_idx]
        ax.scatter(
            [positions[best_idx]],
            [values_arr[best_idx]],
            color=colors[idx],
            s=70,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )
        ax.annotate(
            f"best→{best_label}",
            xy=(positions[best_idx], values_arr[best_idx]),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
        )

        desc = choices_map.get(factor, "")
        if desc and "(" in desc and ")" in desc:
            prefix = desc.split("(")[0].strip()
            suffix_raw = desc[desc.find("(") + 1 : desc.find(")")]
            suffix = suffix_raw.strip()
            base_title = prefix if prefix else factor
            full_title = f"{base_title}\n({suffix})" if suffix else base_title
        else:
            full_title = desc or factor
        ax.set(title=full_title, xlabel="Level")
        ax.set_xticks(positions, x_labels)
        ax.axhline(np.nanmean(values_arr), color="#9e9e9e", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    if any_plotted:
        visible_axes = [ax for ax in axes if ax.get_visible()]
        if visible_axes:
            visible_axes[0].set_ylabel("S/N (dB)")
    fig.suptitle("Taguchi Signal-to-Noise Ratios (Main Effects)", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.text(
        0.5,
        0.02,
        "Higher S/N (less negative) indicates a more robust (better-performing) configuration.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.savefig(out_path)
    plt.close(fig)


def plot_taguchi_metric_distribution(
    df: pd.DataFrame, metric: str, out_path: Path, descriptions: Optional[dict[str, str]] = None
) -> None:
    factor_cols = [c for c in df.columns if c.startswith("factor_")]
    if not factor_cols:
        logging.warning("No factor columns found for Taguchi distribution plot.")
        return
    label_map = {f"factor_{k}": v for k, v in DEFAULT_TAGUCHI_LABELS.items()}
    if descriptions and descriptions.get("taguchi_choices"):
        for key, value in descriptions["taguchi_choices"].items():
            label_map[f"factor_{key}"] = value
            label_map[key] = value

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
        letter = factor.replace("factor_", "")
        title = label_map.get(factor, DEFAULT_TAGUCHI_LABELS.get(letter, letter))
        ax.set(title=title, xlabel="Level")
        level_labels = [
            _taguchi_level_label(letter, int(level), label_map, default=f"Level {level}")
            for level in levels
        ]
        ax.set_xticks(positions, level_labels)
        if ax is axes[0]:
            ax.set_ylabel(metric)
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)

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
    generated_at: Optional[str] = None,
) -> None:
    base_dir = out_path.parent
    choices_map = {
        **DEFAULT_TAGUCHI_LABELS,
        **(descriptions.get("taguchi_choices", {}) or {}),
    }
    extended_choices: dict[str, str] = {}
    for key, value in choices_map.items():
        extended_choices[key] = value
        if len(key) == 1:
            extended_choices[f"factor_{key}"] = value
    choices_map = extended_choices

    def _embed_image(filename: str, alt: str) -> None:
        path = base_dir / filename
        if path.exists():
            lines.append(f"![{alt}]({filename})")
            lines.append("")

    def _factor_label(letter: str) -> str:
        text = choices_map.get(letter, choices_map.get(f"factor_{letter}", letter))
        if "(" in text:
            return text.split("(")[0].strip() or letter
        return text

    def _level_label(letter: str, level: int) -> str:
        return _taguchi_level_label(letter, level, choices_map, default=f"Level {level}")

    lines = ["# Results Summary", ""]
    if generated_at:
        lines.append(f"_Generated {generated_at}_")
        source_root = base_dir.parent
        if source_root != base_dir:
            lines.append(f"_Source: {source_root}_")
        lines.append("")
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
        takeaways = _benchmark_takeaways(synthetic_df)
        if takeaways:
            lines.append("**Quick takeaways**")
            lines.extend(takeaways)
            lines.append("")
        _embed_image("tradeoff_loss_vs_speed_synthetic.png", "Synthetic loss vs throughput")
        _embed_image("loss_final_distribution_synthetic.png", "Synthetic final loss distribution")

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
        takeaways = _benchmark_takeaways(cifar_df)
        if takeaways:
            lines.append("**Quick takeaways**")
            lines.extend(takeaways)
            lines.append("")
        _embed_image("tradeoff_loss_vs_speed_cifar.png", "CIFAR-10 loss vs throughput")
        _embed_image("loss_final_distribution_cifar.png", "CIFAR-10 final loss distribution")

    if taguchi_report is not None:
        title = descriptions.get("taguchi_title", "Taguchi Factors (Top 5)")
        desc = descriptions.get("taguchi_text", "")
        lines.append(f"## {title}")
        if desc:
            lines.extend([desc, ""])
        top = taguchi_report.sort_values("rank").head(5)
        has_runtime = "mean_runtime_seconds" in taguchi_report.columns
        has_throughput = "mean_images_per_second" in taguchi_report.columns
        has_loss_final = "mean_loss_final" in taguchi_report.columns
        headers = ["Rank", "Factor", "Level", "S/N (dB)"]
        if has_runtime:
            headers.append("Runtime (s)")
        if has_throughput:
            headers.append("Images/s")
        if has_loss_final:
            headers.append("Final Loss")
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in top.iterrows():
            factor = str(row["factor"])
            level = int(row["level"])
            cells = [
                str(int(row["rank"])),
                _factor_label(factor),
                _level_label(factor, level),
                f"{row['snr']:.2f}",
            ]
            if has_runtime:
                cells.append(f"{row['mean_runtime_seconds']:.3f}")
            if has_throughput:
                cells.append(f"{row['mean_images_per_second']:.2f}")
            if has_loss_final:
                cells.append(f"{row['mean_loss_final']:.3f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        _embed_image("taguchi_snr.png", "Taguchi S/N main effects")
        _embed_image("taguchi_loss_drop_per_second.png", "Taguchi loss_drop_per_second distributions")
        lines.append("_Higher S/N (less negative) indicates a more robust configuration. Secondary columns show per-level averages for runtime, throughput, and final loss when available._")
        lines.append("")
        lines.append("**Quick takeaways**")
        taguchi_lines: list[str] = []
        factors_seen: set[str] = set()
        report_sorted = taguchi_report.sort_values(["rank", "factor", "snr"], ascending=[True, True, False])
        for _, row in report_sorted.iterrows():
            factor = str(row["factor"])
            if factor in factors_seen:
                continue
            factor_df = taguchi_report[taguchi_report["factor"] == factor]
            best_idx = factor_df["snr"].idxmax()
            worst_idx = factor_df["snr"].idxmin()
            best_row = factor_df.loc[best_idx]
            worst_row = factor_df.loc[worst_idx]
            delta = best_row["snr"] - worst_row["snr"]
            runtime_text = ""
            throughput_text = ""
            loss_text = ""
            if "mean_runtime_seconds" in factor_df.columns:
                runtime_text = f", runtime {best_row['mean_runtime_seconds']:.3f}s vs {worst_row['mean_runtime_seconds']:.3f}s"
            if "mean_images_per_second" in factor_df.columns:
                throughput_text = (
                    f", images/s {best_row['mean_images_per_second']:.2f} vs {worst_row['mean_images_per_second']:.2f}"
                )
            if "mean_loss_final" in factor_df.columns:
                loss_text = (
                    f", final loss {best_row['mean_loss_final']:.3f} vs {worst_row['mean_loss_final']:.3f}"
                )
            taguchi_lines.append(
                "- "
                + f"{_factor_label(factor)} best at {_level_label(factor, int(best_row['level']))} "
                + f"({best_row['snr']:.2f} dB, Δ {delta:+.2f} dB vs. {_level_label(factor, int(worst_row['level']))}"
                + f"{runtime_text}{throughput_text}{loss_text})"
            )
            factors_seen.add(factor)
            if len(taguchi_lines) >= 3:
                break
        if taguchi_lines:
            lines.extend(taguchi_lines)
        else:
            lines.append("- No clear factor preference (insufficient S/N variation).")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Wrote summary markdown to %s", out_path)


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
        plot_tradeoff_scatter(
            synthetic_df,
            x_col="images_per_second",
            y_col="loss_final",
            title="Synthetic Benchmark – Loss vs Throughput",
            x_label="Images per Second (Higher is Better)",
            y_label="Final Loss (Lower is Better)",
            out_path=output_dir / "tradeoff_loss_vs_speed_synthetic.png",
        )
        plot_metric_boxplot(
            synthetic_df,
            metric="loss_final",
            title="Synthetic Benchmark – Final Loss Distribution",
            ylabel="Final Loss",
            out_path=output_dir / "loss_final_distribution_synthetic.png",
        )
        plot_metric_boxplot(
            synthetic_df,
            metric="images_per_second",
            title="Synthetic Benchmark – Throughput Distribution",
            ylabel="Images per Second",
            out_path=output_dir / "images_per_second_distribution_synthetic.png",
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
        plot_tradeoff_scatter(
            cifar_df,
            x_col="images_per_second",
            y_col="loss_final",
            title="CIFAR-10 Benchmark – Loss vs Throughput",
            x_label="Images per Second (Higher is Better)",
            y_label="Final Loss (Lower is Better)",
            out_path=output_dir / "tradeoff_loss_vs_speed_cifar.png",
        )
        plot_metric_boxplot(
            cifar_df,
            metric="loss_final",
            title="CIFAR-10 Benchmark – Final Loss Distribution",
            ylabel="Final Loss",
            out_path=output_dir / "loss_final_distribution_cifar.png",
        )
        plot_metric_boxplot(
            cifar_df,
            metric="images_per_second",
            title="CIFAR-10 Benchmark – Throughput Distribution",
            ylabel="Images per Second",
            out_path=output_dir / "images_per_second_distribution_cifar.png",
        )

    if taguchi_report is not None:
        plot_taguchi_snr(taguchi_report, output_dir / "taguchi_snr.png", descriptions)
    if taguchi_summary is not None and taguchi_report is not None:
        plot_taguchi_metric_distribution(
            taguchi_summary,
            metric="loss_drop_per_second",
            out_path=output_dir / "taguchi_loss_drop_per_second.png",
            descriptions=descriptions,
        )

    timestamp = generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")

    write_summary_markdown(
        synthetic_df,
        cifar_df,
        taguchi_report,
        output_dir / "summary.md",
        descriptions,
        generated_at=timestamp,
    )

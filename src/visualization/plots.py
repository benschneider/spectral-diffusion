from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from src.utils.plot_style import (
    autoscale_y,
    declutter_texts,
    reduce_tick_density,
    set_default_style,
    shorten_label,
    shorten_labels,
)


def _setup_style() -> None:
    set_default_style()


def _color_palette(n_colors: int | None = None):
    if n_colors is None:
        return sns.color_palette("tab20")
    return sns.color_palette("tab20", n_colors=n_colors)


def _dedupe_labels(labels: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    deduped: list[str] = []
    for label in labels:
        count = seen.get(label, 0)
        if count:
            deduped_label = f"{label} ({count + 1})"
        else:
            deduped_label = label
        seen[label] = count + 1
        deduped.append(deduped_label)
    return deduped


def _normalise_category(series: pd.Series, max_len: int = 25) -> pd.Series:
    values = []
    for value in series.astype(str).tolist():
        lowered = value.strip().lower()
        if lowered in {"on", "true"}:
            values.append("Enabled")
        elif lowered in {"off", "false"}:
            values.append("Disabled")
        else:
            values.append(value)

    shortened = shorten_labels(values, max_len=max_len)
    deduped = _dedupe_labels(shortened)
    return pd.Series(deduped, index=series.index, dtype="string")


def _normalise_list(labels: list[str], max_len: int = 25) -> list[str]:
    return _dedupe_labels(shorten_labels(labels, max_len=max_len))


def assign_run_axis(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Attach sequential run identifiers and return mapping."""

    if df is None or df.empty:
        return df, {}

    df = df.copy()
    axis_labels: list[str] = []
    contexts: list[str] = []
    mapping: dict[str, str] = {}

    for idx, (_, row) in enumerate(df.iterrows()):
        short = f"run_{idx:02d}"
        axis_labels.append(short)

        primary = row.get("display_name") or row.get("run_id") or short
        fallback = row.get("run_id")
        config_path = row.get("config_path")

        parts: list[str] = []
        if isinstance(primary, str) and primary:
            parts.append(primary)
        elif primary is not None:
            parts.append(str(primary))

        if fallback and fallback != primary:
            parts.append(f"id={fallback}")

        if isinstance(config_path, str) and config_path:
            parts.append(f"cfg={Path(config_path).name}")

        context = " – ".join(part for part in parts if part) or short
        mapping[short] = context
        contexts.append(context)

    df["run_axis"] = pd.Series(axis_labels, index=df.index, dtype="string")
    df["run_axis_context"] = pd.Series(contexts, index=df.index, dtype="string")
    return df, mapping


def _run_mapping_from_df(df: pd.DataFrame, axis_col: str = "run_axis") -> dict[str, str]:
    mapping: dict[str, str] = {}
    if df is None or axis_col not in df.columns:
        return mapping

    context_col = f"{axis_col}_context" if f"{axis_col}_context" in df.columns else "run_axis_context"
    if context_col not in df.columns:
        context_col = None

    for _, row in df.iterrows():
        key = row.get(axis_col)
        if key is None or (isinstance(key, float) and pd.isna(key)):
            continue
        text = str(key)
        if context_col:
            context_val = row.get(context_col)
            if context_val and not pd.isna(context_val):
                text = str(context_val)
        mapping[str(key)] = text
    return mapping


def _rotate_ticks(ax: plt.Axes, axis: str = "x", rotation: int = 45) -> None:
    if axis == "x":
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right")
    else:
        plt.setp(ax.get_yticklabels(), rotation=rotation, ha="right")
    reduce_tick_density(ax)


def save_figure(fig: plt.Figure, out_path) -> None:
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight", dpi=300)


def plot_loss_metrics(
    df: pd.DataFrame,
    title: str = "Loss Drop per Second by Model",
    out_path=None,
) -> None:
    """Plot loss metrics and optionally save to file."""
    if df is None or df.empty:
        return

    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 3.5))

    axis_col = "run_axis" if "run_axis" in df.columns else "display_name" if "display_name" in df.columns else "run_id"
    preferred_cols = ["loss_drop_per_second_corrected", "loss_drop_per_second"]
    y_col = next((col for col in preferred_cols if col in df.columns), None)
    if y_col is None:
        plt.close(fig)
        return

    plot_df = df.copy()
    if axis_col != "run_axis" and axis_col in plot_df.columns:
        plot_df[axis_col] = _normalise_category(plot_df[axis_col])

    unique = plot_df[axis_col].nunique(dropna=True)
    palette = _color_palette(unique)
    sns.barplot(
        data=plot_df,
        x=axis_col,
        y=y_col,
        palette=palette,
        hue=axis_col,
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Model")
    ylabel = "Loss Drop per Second"
    if y_col.endswith("_corrected"):
        ylabel += " (FFT-corrected)"
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
    _rotate_ticks(ax, axis="x")
    ax.tick_params(axis="y", labelrotation=0)
    autoscale_y(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    mapping = _run_mapping_from_df(plot_df, axis_col)

    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return mapping


def plot_metric_boxplot(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    out_path=None,
) -> None:
    """Plot boxplot for a metric."""
    if df is None or df.empty:
        return

    corrected_metric = f"{metric}_corrected"
    if metric in df.columns:
        metric_col = metric
    elif corrected_metric in df.columns:
        metric_col = corrected_metric
    else:
        return

    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    group_col = "run_axis" if "run_axis" in df.columns else "display_name" if "display_name" in df.columns else "run_id"

    data = []
    labels = []
    for name in df[group_col].unique():
        subset = df[df[group_col] == name][metric_col].dropna()
        if len(subset) > 0:
            data.append(subset.values)
            labels.append(str(name))

    if data:
        label_list = labels if group_col == "run_axis" else _normalise_list(labels)
        ax.boxplot(data, labels=label_list)
        ax.set_title(title)
        ylabel_adj = ylabel
        if metric_col.endswith("_corrected"):
            ylabel_adj += " (FFT-corrected)"
        ax.set_ylabel(ylabel_adj)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
        _rotate_ticks(ax, axis="x")
        ax.tick_params(axis="y", labelrotation=0)
        autoscale_y(ax)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    mapping = _run_mapping_from_df(df, group_col)

    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return mapping

    if not data:
        plt.close(fig)
        return None

    return fig


def plot_taguchi_snr(taguchi_report, out_path, descriptions=None):
    """Plot Taguchi S/N ratios."""
    if taguchi_report is None or taguchi_report.empty:
        return

    if "factor" not in taguchi_report.columns or "snr" not in taguchi_report.columns:
        return

    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    factors = taguchi_report["factor"].unique().tolist()
    palette = _color_palette(len(factors))

    x_pos = range(len(factors))
    snr_values = []
    factor_labels = []

    for factor in factors:
        factor_data = taguchi_report[taguchi_report["factor"] == factor]
        if not factor_data.empty and "snr" in factor_data.columns:
            snr_val = factor_data["snr"].iloc[0]
            snr_values.append(snr_val)
            factor_labels.append(str(factor))

    if snr_values:
        display_labels = _normalise_list(factor_labels)
        colors = [palette[idx % len(palette)] for idx in range(len(snr_values))]
        bars = ax.bar(x_pos, snr_values, color=colors)
        ax.set_xlabel("Factor")
        ax.set_ylabel("S/N Ratio (dB)")
        ax.set_title("Taguchi S/N Ratios by Factor")
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(display_labels)
        _rotate_ticks(ax, axis="x")
        ax.tick_params(axis="y", labelrotation=0)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
        autoscale_y(ax)

        for bar, val in zip(bars, snr_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(fig)


def plot_runtime_metrics(
    df: pd.DataFrame,
    title: str = "Images Processed per Second by Model",
    out_path=None,
) -> None:
    """Plot runtime metrics and optionally save to file."""
    if df is None or df.empty:
        return

    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    x_col = "run_axis" if "run_axis" in df.columns else "display_name" if "display_name" in df.columns else "run_id"
    preferred_cols = ["images_per_second_corrected", "images_per_second"]
    y_col = next((col for col in preferred_cols if col in df.columns), None)
    if y_col is None:
        plt.close(fig)
        return

    plot_df = df.copy()
    if x_col != "run_axis":
        plot_df[x_col] = _normalise_category(plot_df[x_col])

    unique = plot_df[x_col].nunique(dropna=True)
    palette = _color_palette(unique)
    sns.barplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        palette=palette,
        hue=x_col,
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Model")
    ylabel = "Images per Second"
    if y_col.endswith("_corrected"):
        ylabel += " (FFT-corrected)"
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5)
    _rotate_ticks(ax, axis="x")
    ax.tick_params(axis="y", labelrotation=0)
    autoscale_y(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    mapping = _run_mapping_from_df(plot_df, x_col)

    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return mapping
    else:
        return fig


def plot_tradeoff_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    out_path=None,
) -> None:
    """Plot tradeoff scatter plot and optionally save to file."""
    if df is None or df.empty:
        return

    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    group_col = "run_axis" if "run_axis" in df.columns else "display_name" if "display_name" in df.columns else "run_id"

    x_candidates = [x_col, f"{x_col}_corrected"]
    y_candidates = [y_col, f"{y_col}_corrected"]
    x_col_use = next((col for col in x_candidates if col in df.columns), x_col)
    y_col_use = next((col for col in y_candidates if col in df.columns), y_col)

    if x_col_use not in df.columns or y_col_use not in df.columns:
        plt.close(fig)
        return

    groups = df[group_col].unique()
    base_colors = _color_palette(len(groups))

    if group_col == "run_axis":
        display_map = {name: str(name) for name in groups}
    else:
        display_names = _normalise_list([str(name) for name in groups])
        display_map = {orig: disp for orig, disp in zip(groups, display_names)}
    palette = {display_map[name]: base_colors[idx % len(base_colors)] for idx, name in enumerate(groups)}

    for name in groups:
        subset = df[df[group_col] == name]
        if x_col_use in subset.columns and y_col_use in subset.columns:
            display = display_map[name]
            ax.scatter(
                subset[x_col_use],
                subset[y_col_use],
                label=display,
                color=palette[display],
                s=80,
            )
            for _, row in subset.iterrows():
                label = row.get("run_axis") or row.get(group_col, display)
                label = shorten_label(label, max_len=15)
                ax.text(
                    row[x_col_use],
                    row[y_col_use],
                    label,
                    fontsize=6,
                    ha="left",
                    va="center",
                )

    ax.set_title(title)
    x_label_adj = x_label + (" (FFT-corrected)" if x_col_use.endswith("_corrected") else "")
    y_label_adj = y_label + (" (FFT-corrected)" if y_col_use.endswith("_corrected") else "")
    ax.set_xlabel(x_label_adj)
    ax.set_ylabel(y_label_adj)
    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, legend_labels, title="Model", loc="best", fontsize=6)
    ax.grid(True, linestyle="--", linewidth=0.4)
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)
    declutter_texts(ax, min_dist=6)
    reduce_tick_density(ax)
    autoscale_y(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    mapping = _run_mapping_from_df(df, "run_axis" if "run_axis" in df.columns else group_col)

    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        return mapping
    else:
        return fig


def plot_loss_curves(histories, title, out_path):
    """Plot loss curves from history data."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    colors_list = _color_palette(max(len(histories), 1))
    colors = [colors_list[i % len(colors_list)] for i in range(len(histories))]

    for i, history in enumerate(histories):
        label = shorten_label(history.get("label", f"Run {i + 1}"))
        loss_history = history.get("loss_history", [])
        if loss_history:
            steps = list(range(len(loss_history)))
            loss_vals = np.clip(np.array(loss_history, dtype=float), 1e-8, None)
            ax.plot(steps, loss_vals, label=label, color=colors[i % len(colors)], linewidth=1.2)

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (log scale)")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.grid(True, linestyle="--", linewidth=0.4)
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)
    reduce_tick_density(ax)
    autoscale_y(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_taguchi_metric_distribution(taguchi_df, metric, out_path, descriptions=None):
    """Plot Taguchi metric distribution."""
    if taguchi_df is None or taguchi_df.empty:
        return

    factor_cols = [col for col in taguchi_df.columns if col.startswith("factor_")]
    if not factor_cols:
        return

    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    for factor in factor_cols:
        factor_data = taguchi_df[taguchi_df[factor].notna()]
        if factor_data.empty:
            continue

        levels = factor_data[factor].unique()
        palette = _color_palette()

        for i, level in enumerate(levels):
            level_data = factor_data[factor_data[factor] == level]
            if metric in level_data.columns:
                values = level_data[metric].dropna()
                if len(values) > 0:
                    legend_label = f"{shorten_label(factor.replace('factor_', ''))}={shorten_label(level)}"
                    ax.hist(
                        values,
                        alpha=0.65,
                        label=legend_label,
                        bins=min(10, len(values)),
                        color=palette[i % len(palette)],
                    )

    ax.set_title(f"Taguchi {metric.replace('_', ' ').title()} Distribution")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=7)
    ax.grid(True, linestyle="--", linewidth=0.4)
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)
    reduce_tick_density(ax)
    autoscale_y(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_taguchi_main_effects(main_df: pd.DataFrame, response_col: str) -> plt.Figure:
    """Plot mean response per level for each factor."""
    if main_df is None or main_df.empty:
        raise ValueError("main_df must be a non-empty DataFrame.")
    required_cols = {"factor", "level", "mean_response", "delta_from_global"}
    if not required_cols.issubset(main_df.columns):
        missing = required_cols - set(main_df.columns)
        raise KeyError(f"Missing columns for main effects plot: {sorted(missing)}")

    factors = main_df["factor"].unique()
    _setup_style()
    fig, axes = plt.subplots(1, len(factors), figsize=(3.5 * len(factors), 3.2), squeeze=False)

    for ax, factor in zip(axes[0], factors):
        subset = main_df[main_df["factor"] == factor].copy()
        subset.sort_values("mean_response", ascending=False, inplace=True)
        subset["level_display"] = _normalise_category(subset["level"], max_len=20)
        palette = _color_palette(len(subset))
        sns.barplot(
            data=subset,
            x="level_display",
            y="mean_response",
            palette=palette,
            hue="level_display",
            dodge=False,
            legend=False,
            ax=ax,
        )
        global_mean = subset["mean_response"] - subset["delta_from_global"]
        if not global_mean.empty:
            ax.axhline(global_mean.iloc[0], linestyle="--", color="gray", linewidth=1)
        ax.set_title(f"{shorten_label(factor)} main effect on {response_col}")
        ax.set_xlabel("Level")
        ax.set_ylabel("Mean response")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.4)
        _rotate_ticks(ax, axis="x")
        ax.tick_params(axis="y", labelrotation=0)
        reduce_tick_density(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_taguchi_contributions(contrib_df: pd.DataFrame, response_col: str) -> plt.Figure:
    """Plot the percentage contribution of each factor to variance in the response."""
    if contrib_df is None or contrib_df.empty:
        raise ValueError("contrib_df must be a non-empty DataFrame.")
    if "factor" not in contrib_df.columns or "contrib_pct" not in contrib_df.columns:
        raise KeyError("contrib_df must contain 'factor' and 'contrib_pct' columns.")

    df_sorted = contrib_df.sort_values("contrib_pct", ascending=True, na_position="last").copy()
    df_sorted["factor_display"] = _normalise_category(df_sorted["factor"], max_len=25)
    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    palette = _color_palette(len(df_sorted))
    sns.barplot(
        data=df_sorted,
        x="contrib_pct",
        y="factor_display",
        palette=palette,
        hue="factor_display",
        dodge=False,
        legend=False,
        ax=ax,
        orient="h",
    )
    ax.set_xlabel(f"Contribution to variance in {response_col} (%)")
    ax.set_ylabel("Factor")
    ax.set_title("Taguchi factor contributions")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.4)
    ax.tick_params(axis="y", labelrotation=0)
    reduce_tick_density(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_taguchi_interaction_heatmap(
    pivot: pd.DataFrame,
    factor_a: str,
    factor_b: str,
    response_col: str,
) -> plt.Figure:
    """Plot a heatmap of the interaction between factor_a and factor_b."""
    if pivot is None or pivot.empty:
        raise ValueError("pivot must be a non-empty DataFrame.")
    _setup_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    heatmap = sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": response_col},
        annot_kws={"size": 6},
        ax=ax,
    )
    ax.set_title(f"Interaction: {shorten_label(factor_a)} × {shorten_label(factor_b)} ({response_col})")
    ax.set_xlabel(shorten_label(factor_b))
    ax.set_ylabel(shorten_label(factor_a))

    x_labels = [tick.get_text() for tick in heatmap.get_xticklabels()]
    y_labels = [tick.get_text() for tick in heatmap.get_yticklabels()]
    heatmap.set_xticklabels(_normalise_list(x_labels, max_len=15))
    heatmap.set_yticklabels(_normalise_list(y_labels, max_len=15))
    _rotate_ticks(ax, axis="x")
    plt.setp(ax.get_yticklabels(), rotation=0)
    reduce_tick_density(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_feature_toggle_ablation(
    df: pd.DataFrame,
    out_path,
    title: str = "Spectral Feature Toggle Ablation",
) -> None:
    """Compare spectral feature toggles (on/off) across key metrics."""
    if df is None or df.empty:
        return

    label_col = "run_axis" if "run_axis" in df.columns else "display_name" if "display_name" in df.columns else "run_id"
    if label_col not in df.columns:
        return
    if "loss_final" not in df.columns:
        return

    metric_options = [
        ("loss_drop_per_second_corrected", "Loss Drop / Second (FFT-corrected, higher is better)"),
        ("loss_drop_per_second", "Loss Drop / Second (higher is better)"),
        ("images_per_second_corrected", "Images per Second (FFT-corrected, higher is better)"),
        ("images_per_second", "Images per Second (higher is better)"),
    ]
    secondary_metric = None
    secondary_label = ""
    for col, label in metric_options:
        if col in df.columns:
            secondary_metric = col
            secondary_label = label
            break
    if secondary_metric is None:
        return

    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    plot_df = df.copy()
    if label_col != "run_axis":
        plot_df[label_col] = _normalise_category(plot_df[label_col])
    palette = _color_palette(plot_df[label_col].nunique(dropna=True))

    sns.barplot(
        data=plot_df,
        x=label_col,
        y="loss_final",
        palette=palette,
        hue=label_col,
        dodge=False,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Final Loss (Lower is Better)")
    axes[0].set_xlabel("Configuration")
    axes[0].set_ylabel("Final Loss")
    axes[0].grid(True, axis="y", linestyle="--", linewidth=0.4)
    _rotate_ticks(axes[0], axis="x")
    axes[0].tick_params(axis="y", labelrotation=0)
    autoscale_y(axes[0])

    sns.barplot(
        data=plot_df,
        x=label_col,
        y=secondary_metric,
        palette=palette,
        hue=label_col,
        dodge=False,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title(secondary_label)
    axes[1].set_xlabel("Configuration")
    axes[1].set_ylabel(secondary_label)
    axes[1].grid(True, axis="y", linestyle="--", linewidth=0.4)
    _rotate_ticks(axes[1], axis="x")
    axes[1].tick_params(axis="y", labelrotation=0)
    autoscale_y(axes[1])

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

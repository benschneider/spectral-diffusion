import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
def _setup_style():
    sns.set(style="whitegrid")
    plt.rcParams.update({
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 12,
        "figure.figsize": (8, 6),
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

def _color_palette(n_colors: int | None = None):
    if n_colors is None:
        return sns.color_palette("tab20")
    return sns.color_palette("tab20", n_colors=n_colors)

def save_figure(fig, out_path):
    fig.savefig(out_path, bbox_inches='tight', dpi=300)

def plot_loss_metrics(df: pd.DataFrame, title="Loss Drop per Second by Model", out_path=None) -> None:
    """Plot loss metrics and optionally save to file."""
    if df is None or df.empty:
        return

    _setup_style()
    fig, ax = plt.subplots()

    # Use run_id or display_name for x-axis
    x_col = 'display_name' if 'display_name' in df.columns else 'run_id'
    y_col = 'loss_drop_per_second'

    if y_col not in df.columns:
        plt.close(fig)
        return

    unique = df[x_col].nunique()
    palette = _color_palette(unique)
    sns.barplot(data=df, x=x_col, y=y_col, palette=palette, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Model')
    ax.set_ylabel('Loss Drop per Second')
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

    if out_path:
        fig.savefig(out_path, bbox_inches='tight', dpi=300)
        plt.close(fig)


def plot_metric_boxplot(df: pd.DataFrame, metric: str, title: str, ylabel: str, out_path=None) -> None:
        """Plot boxplot for a metric."""
        if df is None or df.empty or metric not in df.columns:
            return
    
        _setup_style()
        fig, ax = plt.subplots()
    
        # Use run_id or display_name for grouping
        group_col = 'display_name' if 'display_name' in df.columns else 'run_id'
    
        # Prepare data for boxplot
        data = []
        labels = []
        for name in df[group_col].unique():
            subset = df[df[group_col] == name][metric].dropna()
            if len(subset) > 0:
                data.append(subset.values)
                labels.append(name)
    
        if data:
            ax.boxplot(data, labels=labels)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, axis='y', linestyle='--', linewidth=0.7)
    
            if out_path:
                fig.savefig(out_path, bbox_inches='tight', dpi=300)
                plt.close(fig)
            else:
                return fig
        else:
            plt.close(fig)


def plot_taguchi_snr(taguchi_report, out_path, descriptions=None):
    """Plot Taguchi S/N ratios."""
    if taguchi_report is None or taguchi_report.empty:
        return

    # Check if we have the required columns
    if 'factor' not in taguchi_report.columns or 'snr' not in taguchi_report.columns:
        return

    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by factor and plot S/N ratios
    factors = taguchi_report['factor'].unique()
    palette = _color_palette(len(factors))

    x_pos = range(len(factors))
    snr_values = []
    factor_labels = []

    for factor in factors:
        factor_data = taguchi_report[taguchi_report['factor'] == factor]
        if not factor_data.empty and 'snr' in factor_data.columns:
            snr_val = factor_data['snr'].iloc[0]  # Take first S/N value for the factor
            snr_values.append(snr_val)
            factor_labels.append(factor)

    if snr_values:
        colors = [palette[idx % len(palette)] for idx in range(len(snr_values))]
        bars = ax.bar(x_pos, snr_values, color=colors)
        ax.set_xlabel('Factor')
        ax.set_ylabel('S/N Ratio (dB)')
        ax.set_title('Taguchi S/N Ratios by Factor')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(factor_labels, rotation=45, ha='right')
        ax.grid(True, axis='y', linestyle='--', linewidth=0.7)

        # Add value labels on bars
        for bar, val in zip(bars, snr_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches='tight', dpi=300)
        plt.close(fig)


def plot_runtime_metrics(df: pd.DataFrame, title="Images Processed per Second by Model", out_path=None) -> None:
    """Plot runtime metrics and optionally save to file."""
    if df is None or df.empty:
        return

    _setup_style()
    fig, ax = plt.subplots()

    # Use run_id or display_name for x-axis
    x_col = 'display_name' if 'display_name' in df.columns else 'run_id'
    y_col = 'images_per_second'

    if y_col not in df.columns:
        plt.close(fig)
        return

    unique = df[x_col].nunique()
    palette = _color_palette(unique)
    sns.barplot(data=df, x=x_col, y=y_col, palette=palette, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Model')
    ax.set_ylabel('Images per Second')
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

    if out_path:
        fig.savefig(out_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        return fig

def plot_tradeoff_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, x_label: str, y_label: str, out_path=None) -> None:
    """Plot tradeoff scatter plot and optionally save to file."""
    if df is None or df.empty:
        return

    _setup_style()
    fig, ax = plt.subplots()

    # Use run_id or display_name for grouping
    group_col = 'display_name' if 'display_name' in df.columns else 'run_id'
    groups = df[group_col].unique()
    base_colors = _color_palette(len(groups))
    palette = {name: base_colors[idx % len(base_colors)] for idx, name in enumerate(groups)}

    for name in df[group_col].unique():
        subset = df[df[group_col] == name]
        if x_col in subset.columns and y_col in subset.columns:
            ax.scatter(subset[x_col], subset[y_col], label=name, color=palette[name], s=80)
            for _, row in subset.iterrows():
                ax.annotate(name, (row[x_col], row[y_col]),
                           textcoords="offset points", xytext=(5,5), ha='left', fontsize=9)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(title='Model')
    ax.grid(True, linestyle='--', linewidth=0.7)

    if out_path:
        fig.savefig(out_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        return fig

def plot_loss_curves(histories, title, out_path):
    """Plot loss curves from history data."""
    _setup_style()
    fig, ax = plt.subplots()

    colors_list = _color_palette(max(len(histories), 1))
    colors = [colors_list[i % len(colors_list)] for i in range(len(histories))]

    for i, history in enumerate(histories):
        label = history.get('label', f'Run {i+1}')
        loss_history = history.get('loss_history', [])
        if loss_history:
            steps = list(range(len(loss_history)))
            ax.plot(steps, loss_history, label=label, color=colors[i % len(colors)], linewidth=2)

    ax.set_title(title)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, linestyle='--', linewidth=0.7)

    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_taguchi_metric_distribution(taguchi_df, metric, out_path, descriptions=None):
    """Plot Taguchi metric distribution."""
    if taguchi_df is None or taguchi_df.empty:
        return

    # Check if we have factor columns
    factor_cols = [col for col in taguchi_df.columns if col.startswith("factor_")]
    if not factor_cols:
        return

    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by factor and level, plot distributions
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
                    ax.hist(values, alpha=0.7, label=f'{factor.replace("factor_", "")}={level}',
                           bins=min(10, len(values)), color=palette[i % len(palette)])

    ax.set_title(f'Taguchi {metric.replace("_", " ").title()} Distribution')
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, linestyle='--', linewidth=0.7)

    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_taguchi_main_effects(main_df: pd.DataFrame, response_col: str) -> plt.Figure:
    """
    Plot mean response per level for each factor.

    Expected columns: factor, level, mean_response, delta_from_global.
    Returns the Matplotlib Figure (caller is responsible for saving/closing).
    """
    if main_df is None or main_df.empty:
        raise ValueError("main_df must be a non-empty DataFrame.")
    required_cols = {"factor", "level", "mean_response", "delta_from_global"}
    if not required_cols.issubset(main_df.columns):
        missing = required_cols - set(main_df.columns)
        raise KeyError(f"Missing columns for main effects plot: {sorted(missing)}")

    factors = main_df["factor"].unique()
    _setup_style()
    fig, axes = plt.subplots(
        1, len(factors), figsize=(5 * len(factors), 4), squeeze=False
    )

    for ax, factor in zip(axes[0], factors):
        subset = main_df[main_df["factor"] == factor].copy()
        subset.sort_values("mean_response", ascending=False, inplace=True)
        palette = _color_palette(len(subset))
        sns.barplot(data=subset, x="level", y="mean_response", palette=palette, ax=ax)
        global_mean = subset["mean_response"] - subset["delta_from_global"]
        if not global_mean.empty:
            ax.axhline(global_mean.iloc[0], linestyle="--", color="gray", linewidth=1)
        ax.set_title(f"{factor} main effect on {response_col}")
        ax.set_xlabel("Level")
        ax.set_ylabel("Mean response")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.7)
    fig.tight_layout()
    return fig


def plot_taguchi_contributions(contrib_df: pd.DataFrame, response_col: str) -> plt.Figure:
    """
    Plot the percentage contribution of each factor to variance in the response.

    Expected columns: factor, contrib_pct.
    Returns the Matplotlib Figure (caller is responsible for saving/closing).
    """
    if contrib_df is None or contrib_df.empty:
        raise ValueError("contrib_df must be a non-empty DataFrame.")
    if "factor" not in contrib_df.columns or "contrib_pct" not in contrib_df.columns:
        raise KeyError("contrib_df must contain 'factor' and 'contrib_pct' columns.")

    df_sorted = contrib_df.sort_values("contrib_pct", ascending=True, na_position="last")
    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    palette = _color_palette(len(df_sorted))
    sns.barplot(
        data=df_sorted,
        x="contrib_pct",
        y="factor",
        palette=palette,
        ax=ax,
        orient="h",
    )
    ax.set_xlabel(f"Contribution to variance in {response_col} (%)")
    ax.set_ylabel("Factor")
    ax.set_title("Taguchi factor contributions")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.7)
    fig.tight_layout()
    return fig


def plot_taguchi_interaction_heatmap(
    pivot: pd.DataFrame, factor_a: str, factor_b: str, response_col: str
) -> plt.Figure:
    """
    Plot a heatmap of the interaction between factor_a and factor_b.

    Args:
        pivot: DataFrame indexed by factor_a with columns factor_b and values mean response.
    """
    if pivot is None or pivot.empty:
        raise ValueError("pivot must be a non-empty DataFrame.")
    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": response_col},
        ax=ax,
    )
    ax.set_title(f"Interaction: {factor_a} × {factor_b} ({response_col})")
    ax.set_xlabel(factor_b)
    ax.set_ylabel(factor_a)
    fig.tight_layout()
    return fig


def plot_feature_toggle_ablation(df: pd.DataFrame, out_path, title: str = "Spectral Feature Toggle Ablation") -> None:
    """Compare spectral feature toggles (on/off) across key metrics."""
    if df is None or df.empty:
        return

    label_col = "display_name" if "display_name" in df.columns else "run_id"
    if label_col not in df.columns:
        return

    if "loss_final" not in df.columns:
        return

    secondary_metric = None
    secondary_label = ""
    if "loss_drop_per_second" in df.columns:
        secondary_metric = "loss_drop_per_second"
        secondary_label = "Loss Drop / Second (Higher is Better)"
    elif "images_per_second" in df.columns:
        secondary_metric = "images_per_second"
        secondary_label = "Images per Second (Higher is Better)"
    else:
        return

    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    palette = _color_palette(df[label_col].nunique())

    sns.barplot(data=df, x=label_col, y="loss_final", palette=palette, ax=axes[0])
    axes[0].set_title("Final Loss (Lower is Better)")
    axes[0].set_xlabel("Configuration")
    axes[0].set_ylabel("Final Loss")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].grid(True, axis="y", linestyle="--", linewidth=0.7)

    sns.barplot(data=df, x=label_col, y=secondary_metric, palette=palette, ax=axes[1])
    axes[1].set_title(secondary_label)
    axes[1].set_xlabel("Configuration")
    axes[1].set_ylabel(secondary_label)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(True, axis="y", linestyle="--", linewidth=0.7)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close(fig)

#!/usr/bin/env python
"""CLI for Taguchi post-processing (main effects, contributions, interactions, insights)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.analyze_taguchi import (
    compute_factor_contributions,
    compute_main_effects,
    compute_pairwise_interactions,
    load_taguchi_csv,
    summarize_taguchi_insights,
)
from src.visualization.plots import (
    plot_taguchi_contributions,
    plot_taguchi_interaction_heatmap,
    plot_taguchi_main_effects,
    save_figure,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Taguchi sweep results.")
    parser.add_argument("--csv", required=True, type=Path, help="Path to taguchi_report.csv")
    parser.add_argument("--response-col", required=True, type=str, help="Metric column to optimise.")
    parser.add_argument(
        "--outdir",
        required=True,
        type=Path,
        help="Directory to write tables, figures, and markdown insights.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top factors/insights to highlight (default: 3).",
    )
    parser.add_argument(
        "--interaction-limit",
        type=int,
        default=None,
        help="Optional limit for number of interaction heatmaps to render (default: all).",
    )
    parser.add_argument(
        "--factor-prefix",
        action="append",
        default=["factor_"],
        help="Prefix hint for factor columns (can be supplied multiple times).",
    )
    return parser.parse_args()


def _resolve_response_column(df: pd.DataFrame, requested: str) -> str:
    """Return the dataframe column matching the requested response metric."""
    if requested in df.columns:
        return requested

    lowered_map = {col.lower(): col for col in df.columns}
    if requested.lower() in lowered_map:
        return lowered_map[requested.lower()]

    aliases = {
        "loss_drop_per_sec": ["loss_drop_per_second", "loss_drop_per_seconds"],
        "loss_drop_per_second": ["loss_drop_per_sec"],
        "images_per_second": ["imgs_per_sec", "throughput_images_per_second"],
    }
    for alt in aliases.get(requested.lower(), []):
        if alt in df.columns:
            return alt
        if alt.lower() in lowered_map:
            return lowered_map[alt.lower()]

    raise KeyError(
        f"Response column '{requested}' not found. Available columns: {list(df.columns)}"
    )


def _infer_factors(df: pd.DataFrame, response_col: str, prefixes: List[str]) -> List[str]:
    """Infer factor columns heuristically."""
    from pandas.api.types import is_bool_dtype, is_categorical_dtype, is_object_dtype

    exclusions = {"run_id", response_col}
    factors: List[str] = []
    for col in df.columns:
        if col in exclusions:
            continue
        if col == response_col:
            continue
        if any(col.startswith(prefix) for prefix in prefixes):
            factors.append(col)
            continue
        col_values = df[col]
        if is_object_dtype(col_values) or is_categorical_dtype(col_values) or is_bool_dtype(col_values):
            factors.append(col)
    if not factors:
        raise ValueError("Unable to infer factor columns. Provide prefixed factor names in the CSV.")
    return factors


def _best_levels(main_df: pd.DataFrame, response_col: str) -> List[Tuple[str, str, float]]:
    """Return tuples of (factor, best_level, mean_response) using main effects frame."""
    best = []
    for factor, group in main_df.groupby("factor"):
        sorted_group = group.sort_values("mean_response", ascending=False)
        best_level = sorted_group.iloc[0]["level"]
        best_val = sorted_group.iloc[0]["mean_response"]
        best.append((factor, str(best_level), float(best_val)))
    return best


def _write_markdown(
    outdir: Path,
    insights: List[str],
    contrib_df: pd.DataFrame,
    best_levels: List[Tuple[str, str, float]],
    response_col: str,
) -> None:
    """Write a markdown snippet with insights and recommendations."""
    md_lines: List[str] = ["## Taguchi Insights", ""]
    if insights:
        md_lines.append("### Key Findings")
        for bullet in insights:
            md_lines.append(f"- {bullet}")
        md_lines.append("")
    if not contrib_df.empty:
        md_lines.append("### Top Contributing Factors")
        md_lines.append("")
        md_lines.append("| Factor | Contribution (%) |")
        md_lines.append("| --- | ---: |")
        for _, row in contrib_df.sort_values("contrib_pct", ascending=False).iterrows():
            md_lines.append(f"| {row['factor']} | {row['contrib_pct']:.2f} |")
        md_lines.append("")
    if best_levels:
        md_lines.append("### Recommended Settings")
        md_lines.append("")
        md_lines.append("| Factor | Best Level | Mean Response |")
        md_lines.append("| --- | --- | ---: |")
        for factor, level, mean_val in best_levels:
            md_lines.append(f"| {factor} | {level} | {mean_val:.4f} |")
        md_lines.append("")
    md_path = outdir / "taguchi_insights.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.csv.exists():
        raise FileNotFoundError(f"Taguchi CSV not found: {args.csv}")

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_taguchi_csv(str(args.csv))
    response_col = _resolve_response_column(df, args.response_col)
    factors = _infer_factors(df, response_col, args.factor_prefix)

    logging.info("Using response column: %s", response_col)
    logging.info("Detected factors: %s", ", ".join(factors))

    main_effects = compute_main_effects(df, factors, response_col)
    contributions = compute_factor_contributions(df, factors, response_col)
    interactions = compute_pairwise_interactions(df, factors, response_col)
    insights = summarize_taguchi_insights(
        df,
        factors,
        response_col,
        top_k=args.top_k,
    )

    main_effects_path = outdir / "taguchi_main_effects.csv"
    contrib_path = outdir / "taguchi_contrib.csv"
    main_effects.to_csv(main_effects_path, index=False)
    contributions.to_csv(contrib_path, index=False)
    logging.info("Saved main effects to %s", main_effects_path)
    logging.info("Saved contributions to %s", contrib_path)

    # Interactions tables
    interaction_items = list(interactions.items())
    if args.interaction_limit is not None:
        interaction_items = interaction_items[: args.interaction_limit]

    for (factor_a, factor_b), pivot in interaction_items:
        csv_path = outdir / f"taguchi_interactions_{factor_a}_{factor_b}.csv"
        pivot.to_csv(csv_path)
        logging.info("Saved interaction table to %s", csv_path)

    # Figures
    figs: List[Tuple[plt.Figure, Path]] = []
    try:
        figs.append(
            (plot_taguchi_main_effects(main_effects, response_col), outdir / "taguchi_main_effects.png")
        )
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("Skipping main effects plot: %s", exc)

    try:
        figs.append(
            (
                plot_taguchi_contributions(contributions, response_col),
                outdir / "taguchi_contrib.png",
            )
        )
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("Skipping contributions plot: %s", exc)

    for (factor_a, factor_b), pivot in interaction_items:
        if pivot.empty:
            continue
        fig = plot_taguchi_interaction_heatmap(pivot, factor_a, factor_b, response_col)
        figs.append((fig, outdir / f"taguchi_interaction_{factor_a}_{factor_b}.png"))

    for fig, path in figs:
        save_figure(fig, path)
        plt.close(fig)
        logging.info("Saved figure %s", path)

    best_levels = _best_levels(main_effects, response_col) if not main_effects.empty else []
    _write_markdown(outdir, insights, contributions, best_levels, response_col)
    logging.info("Wrote Taguchi insights markdown to %s", outdir / "taguchi_insights.md")


if __name__ == "__main__":
    main()

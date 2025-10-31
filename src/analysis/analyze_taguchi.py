"""Taguchi analysis utilities for computing effects, contributions, and insights."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats  # noqa: F401  # retained for potential downstream use


FACTOR_PREFIXES = ("factor_",)


def load_taguchi_csv(path: str) -> pd.DataFrame:
    """Read the Taguchi sweep CSV into a DataFrame."""
    return pd.read_csv(path)


def _ensure_iterable_factors(df: pd.DataFrame, factors: List[str]) -> List[str]:
    """Return the subset of factor names that exist in the dataframe."""
    existing = [factor for factor in factors if factor in df.columns]
    if len(existing) != len(factors):
        missing = set(factors) - set(existing)
        if missing:
            raise KeyError(f"Missing factor columns: {sorted(missing)}")
    return existing


def compute_main_effects(df: pd.DataFrame, factors: List[str], response_col: str) -> pd.DataFrame:
    """
    For each factor and each level of that factor, compute the mean response.

    Returns a DataFrame with columns:
        factor, level, mean_response, delta_from_global
    """
    factors = _ensure_iterable_factors(df, factors)
    if response_col not in df.columns:
        raise KeyError(f"Response column '{response_col}' not found in dataframe.")

    out_rows: List[dict] = []
    global_mean = df[response_col].mean()
    for factor in factors:
        for level, group in df.groupby(factor):
            mean_val = group[response_col].mean()
            out_rows.append(
                {
                    "factor": factor,
                    "level": level,
                    "mean_response": mean_val,
                    "delta_from_global": mean_val - global_mean,
                }
            )
    return pd.DataFrame(out_rows)


def compute_factor_contributions(
    df: pd.DataFrame, factors: List[str], response_col: str
) -> pd.DataFrame:
    """
    Rough ANOVA-style variance attribution for each factor.

    For each factor:
        SS_factor = sum_over_levels( n_level * (mean_level - global_mean)^2 )

    Total_SS = sum( (y - global_mean)^2 )
    Contribution% = SS_factor / Total_SS * 100

    Returns a DataFrame sorted descending by contrib_pct with columns:
        factor, ss, contrib_pct
    """
    factors = _ensure_iterable_factors(df, factors)
    if response_col not in df.columns:
        raise KeyError(f"Response column '{response_col}' not found in dataframe.")

    y = df[response_col].to_numpy()
    global_mean = y.mean()
    total_ss = float(np.sum((y - global_mean) ** 2))

    rows: List[dict] = []
    for factor in factors:
        ss_factor = 0.0
        for level, group in df.groupby(factor):
            mean_level = group[response_col].mean()
            n_level = len(group)
            ss_factor += n_level * (mean_level - global_mean) ** 2
        contrib_pct = (ss_factor / total_ss * 100.0) if total_ss > 0 else np.nan
        rows.append({"factor": factor, "ss": ss_factor, "contrib_pct": contrib_pct})
    return pd.DataFrame(rows).sort_values("contrib_pct", ascending=False, na_position="last")


def compute_pairwise_interactions(
    df: pd.DataFrame, factors: List[str], response_col: str
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Estimate 2-way interaction surfaces.

    For each factor pair (A,B), compute the pivot table of mean response for A level x B level.

    Returns:
        dict mapping (factor_a, factor_b) -> pivot DataFrame.
    """
    interactions: Dict[Tuple[str, str], pd.DataFrame] = {}
    for factor_a, factor_b in combinations(_ensure_iterable_factors(df, factors), 2):
        pivot = df.pivot_table(
            index=factor_a,
            columns=factor_b,
            values=response_col,
            aggfunc="mean",
        )
        interactions[(factor_a, factor_b)] = pivot
    return interactions


def _determine_direction(response_col: str) -> str:
    """
    Decide whether higher or lower is better for the given response column.

    Returns "max" when larger values are preferable and "min" otherwise.
    """
    lowered = response_col.lower()
    lower_is_better_keywords = ("fid", "loss", "error", "mae", "mse", "rmse")
    for keyword in lower_is_better_keywords:
        if keyword in lowered:
            return "min"
    return "max"


def summarize_taguchi_insights(
    df: pd.DataFrame,
    factors: List[str],
    response_col: str,
    top_k: int = 3,
) -> List[str]:
    """
    Produce human-readable bullet insights for the Taguchi sweep.

    Example bullet:
        "sampler=masf beats global mean by +0.800 on loss_drop_per_sec"

    Returns:
        List of strings with at most `top_k` entries prioritised by absolute delta.
    """
    factors = _ensure_iterable_factors(df, factors)
    if response_col not in df.columns:
        raise KeyError(f"Response column '{response_col}' not found in dataframe.")

    bullets: List[str] = []
    direction = _determine_direction(response_col)
    global_mean = df[response_col].mean()

    deltas: List[Tuple[float, str]] = []
    for factor in factors:
        group_means = df.groupby(factor)[response_col].mean()
        if direction == "min":
            best_level = group_means.idxmin()
            best_val = group_means.loc[best_level]
            delta = global_mean - best_val
            sign = "-"
        else:
            best_level = group_means.idxmax()
            best_val = group_means.loc[best_level]
            delta = best_val - global_mean
            sign = "+"
        deltas.append((abs(delta), factor))
        direction_word = "lowers" if direction == "min" else "beats"
        diff = delta if direction == "max" else -delta
        bullets.append(
            f"{factor}={best_level} {direction_word} global mean by "
            f"{diff:+.3f} on {response_col}"
        )

    # Prioritise top_k by absolute delta
    deltas.sort(reverse=True)
    selected_factors = {factor for _, factor in deltas[:top_k]}
    prioritized = [
        bullet for bullet in bullets if any(bullet.startswith(f"{factor}=") for factor in selected_factors)
    ]
    return prioritized if prioritized else bullets[:top_k]

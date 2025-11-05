from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
import yaml


def _compute_snr(values: np.ndarray, mode: str) -> float:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan")
    eps = 1e-12
    if mode == "larger":
        positive = values[values > 0]
        if positive.size == 0:
            return float("nan")
        return -10.0 * np.log10(np.mean(1.0 / (positive**2 + eps)))
    if mode == "smaller":
        return -10.0 * np.log10(np.mean(values**2 + eps))
    raise ValueError(f"Unknown S/N mode '{mode}'")


def _resolve_config_path(config_path: Path, summary_path: Path, run_id: str) -> Path:
    """Best-effort resolution of the stored config path.

    Historical summary files may contain absolute paths referencing the
    environment where the sweep originally ran. When the artifacts are copied to
    another machine those paths no longer exist. We attempt to remap the
    reference to the local run directory using the captured run identifier.
    """

    if config_path.exists():
        return config_path

    candidates = []
    summary_root = summary_path.parent
    if run_id:
        candidates.append(summary_root / "runs" / run_id / "config.yaml")

    # Fall back to the filename relative to the runs directory if the layout
    # does not follow the expected run_id naming scheme.
    if config_path.name:
        candidates.append(summary_root / "runs" / config_path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Unable to resolve config for run '{run_id}'. Checked: "
        f"{', '.join(str(path) for path in [config_path, *candidates])}"
    )


def _load_taguchi_factors(config_path: Path) -> Dict[str, Dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    taguchi_cfg = cfg.get("taguchi", {}) or {}
    factor_levels = taguchi_cfg.get("factor_levels")
    if isinstance(factor_levels, dict):
        normalised: Dict[str, Dict[str, Any]] = {}
        for factor, data in factor_levels.items():
            if isinstance(data, dict):
                level_label = data.get("level_label", data.get("label", data.get("level")))
                level_index = data.get("level_index")
            else:
                level_label = data
                level_index = None
            normalised[factor] = {
                "level_label": level_label,
                "level_index": level_index,
            }
        return normalised

    design_row = taguchi_cfg.get("row", {}) or {}
    factor_mapping = taguchi_cfg.get("factor_mapping", {}) or {}
    factors: Dict[str, Dict[str, Any]] = {}
    if factor_mapping:
        for column, value in design_row.items():
            factor = factor_mapping.get(str(column))
            if factor:
                factors[factor] = {
                    "level_label": str(value),
                    "level_index": value,
                }
    else:
        for factor, value in design_row.items():
            factors[factor] = {
                "level_label": str(value),
                "level_index": value,
            }
    return factors


def generate_taguchi_report(
    summary_path: Path,
    metric: str,
    mode: str,
    output_path: Path,
) -> pd.DataFrame:
    summary_df = pd.read_csv(summary_path)
    if metric not in summary_df.columns:
        raise ValueError(f"Metric '{metric}' not found in {summary_path}")

    taguchi_rows = []
    for _, row in summary_df.iterrows():
        config_path = _resolve_config_path(Path(row["config_path"]), summary_path, row.get("run_id", ""))
        factors = _load_taguchi_factors(config_path)
        entry = row.to_dict()
        for key, meta in factors.items():
            label = meta.get("level_label") if isinstance(meta, dict) else meta
            entry[f"factor_{key}"] = label
        taguchi_rows.append(entry)

    df = pd.DataFrame(taguchi_rows)
    factor_cols = [col for col in df.columns if col.startswith("factor_")]
    if not factor_cols:
        print("DEBUG: No Taguchi factor information found in configs. Skipping report generation.")
        return None

    report_rows: list[dict[str, float]] = []
    factor_deltas: dict[str, float] = {}
    extra_metrics = [
        col
        for col in ("runtime_seconds", "images_per_second", "loss_final")
        if col in df.columns
    ]

    for factor in factor_cols:
        levels = df[factor].unique()
        level_stats = []
        for lvl in levels:
            values = df.loc[df[factor] == lvl, metric].astype(float).to_numpy()
            snr = _compute_snr(values, mode)
            mean_val = float(np.nanmean(values))
            extras = {
                f"mean_{extra}": float(
                    np.nanmean(
                        df.loc[df[factor] == lvl, extra].astype(float).to_numpy()
                    )
                )
                for extra in extra_metrics
            }
            level_stats.append((lvl, mean_val, snr))
            report_rows.append(
                {
                    "factor": factor.replace("factor_", ""),
                    "level": lvl,
                    "mean_metric": mean_val,
                    "snr": snr,
                    **extras,
                }
            )
        snr_values = [snr for (_, _, snr) in level_stats if not np.isnan(snr)]
        if snr_values:
            factor_deltas[factor] = max(snr_values) - min(snr_values)
        else:
            factor_deltas[factor] = float("nan")

    summary_df = pd.DataFrame(report_rows)
    if summary_df.empty:
        raise ValueError("No S/N data computed. Check metric values.")

    ranked_factors = (
        pd.Series(factor_deltas)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "factor", 0: "delta"})
    )
    ranked_factors["factor"] = ranked_factors["factor"].str.replace("factor_", "")
    ranked_factors["rank"] = np.arange(1, len(ranked_factors) + 1)

    summary_df = summary_df.merge(ranked_factors, on="factor", how="left")
    summary_df = summary_df.sort_values(["rank", "factor", "level"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    return summary_df


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compute Taguchi S/N ratios from summary.csv")
    parser.add_argument("--summary", type=Path, required=True, help="Path to summary CSV")
    parser.add_argument("--metric", type=str, required=True, help="Metric column to analyze")
    parser.add_argument("--mode", choices=["larger", "smaller"], default="larger")
    parser.add_argument("--output", type=Path, default=Path("results/taguchi_report.csv"))
    args = parser.parse_args(argv)

    report = generate_taguchi_report(
        summary_path=args.summary,
        metric=args.metric,
        mode=args.mode,
        output_path=args.output,
    )
    if report is not None:
        print(report)
    else:
        print("Taguchi report skipped due to insufficient data.")


if __name__ == "__main__":
    main()

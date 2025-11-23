#!/usr/bin/env python
"""Generate the cleaned report_v2 bundle (figures + summary.md)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.analysis.taguchi_stats import _load_taguchi_factors, _resolve_config_path
from src.analysis.analyze_taguchi import compute_factor_contributions
from src.reporting.generate_markdown import _export_report
from src.utils.plot_style import set_default_style
from src.visualization.analysis_utils import collect_loss_histories, compute_fft_corrected, sanitize_metric_frame
from src.visualization.collect import clean_summary
from src.visualization.figures import _attach_run_dirs, _create_noising_visual, _create_prediction_visual
from src.visualization.plots import (
    assign_run_axis,
    plot_loss_curves,
    plot_taguchi_contributions,
    plot_taguchi_main_effects,
    plot_tradeoff_scatter,
)


PRIMARY_FACTORS_BY_PROFILE = {
    "snr": ["snr_ratio", "spectral_noise_shaping_strength", "snr_weighting_mode", "spectral_adapter_placement"],
    "sampler": ["sampler_type", "sampling_steps", "snr_ratio"],
    "curriculum": ["curriculum_mode", "train_steps"],
}


@dataclass
class DatasetBundle:
    df: Optional[pd.DataFrame]
    notes: dict[str, bool]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report_v2 figures and summary.")
    parser.add_argument("--report-root", type=Path, default=None, help="Root with synthetic/, cifar/, taguchi/ folders.")
    parser.add_argument("--synthetic-dir", type=Path, default=None, help="Override synthetic summary directory.")
    parser.add_argument("--cifar-dir", type=Path, default=None, help="Override CIFAR summary directory.")
    parser.add_argument("--taguchi-dir", type=Path, default=None, help="Override Taguchi summary directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Target directory for report_v2 (default: <report-root>/report_v2).")
    parser.add_argument("--profile", choices=["snr", "sampler", "curriculum"], default="snr", help="Primary factor profile.")
    parser.add_argument("--primary-factor", action="append", default=None, help="Explicit primary factor (can repeat).")
    parser.add_argument("--max-loss-runs", type=int, default=4, help="Max runs to show on loss curves.")
    parser.add_argument("--max-tradeoff-points", type=int, default=12, help="Max scatter points per tradeoff plot.")
    parser.add_argument("--generated-at", type=str, default=None, help="Optional ISO timestamp to embed in summary.")
    return parser.parse_args()


def _latest_report_root() -> Optional[Path]:
    candidates: list[Path] = []
    for prefix in ("full_report", "smoke_report"):
        base = ROOT / "results"
        if not base.exists():
            continue
        for path in base.iterdir():
            if path.is_dir() and path.name.startswith(prefix):
                candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_dirs(args: argparse.Namespace) -> tuple[Optional[Path], Optional[Path], Optional[Path], Path]:
    root = args.report_root or _latest_report_root()
    synthetic = args.synthetic_dir or (root / "synthetic" if root else None)
    cifar = args.cifar_dir or (root / "cifar" if root else None)
    taguchi = args.taguchi_dir or (root / "taguchi" if root else None)
    output = args.output_dir or (root / "report_v2" if root else Path("report_v2"))
    return synthetic, cifar, taguchi, output


def _ensure_dirs(output_dir: Path) -> tuple[Path, Path, Path]:
    images_dir = output_dir / "images"
    appendix_dir = output_dir / "appendix"
    for sub in [
        images_dir,
        appendix_dir,
        appendix_dir / "noise_chains",
        appendix_dir / "taguchi_interactions",
        appendix_dir / "distributions",
    ]:
        sub.mkdir(parents=True, exist_ok=True)
    return images_dir, appendix_dir, appendix_dir / "taguchi_interactions"


def _humanise_name(name: str) -> str:
    if not name:
        return ""
    parts = name.replace("-", "_").split("_")
    tokens: list[str] = []
    for part in parts:
        if part.lower() == "snr":
            tokens.append("SNR")
        elif part.lower() in {"psnr", "fid"}:
            tokens.append(part.upper())
        else:
            tokens.append(part.capitalize())
    return " ".join(tokens)


def _humanise_level(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value)
    lowered = text.lower()
    if lowered in {"none", "off", "false"}:
        return "Off"
    if lowered in {"true", "on"}:
        return "On"
    return text


def _load_config(config_path: Optional[Path]) -> dict:
    if config_path is None or not config_path.exists():
        return {}
    try:
        return yaml.safe_load(config_path.read_text()) or {}
    except Exception:
        return {}


def _derive_weighting_mode(cfg: dict) -> Optional[str]:
    loss_cfg = cfg.get("loss", {}) if isinstance(cfg, dict) else {}
    adaptive = loss_cfg.get("adaptive_snr")
    use_weighting = loss_cfg.get("use_weighting", loss_cfg.get("snr_weighting", None))
    if adaptive:
        return "adaptive"
    if use_weighting is False:
        return "off"
    if use_weighting is True:
        return "static"
    return None


def _extract_run_metadata(row: pd.Series, factor_mapping: dict[str, Any]) -> dict[str, Any]:
    cfg_path = None
    if "config_path" in row and isinstance(row["config_path"], str):
        try:
            cfg_path = _resolve_config_path(Path(row["config_path"]), Path("."), row.get("run_id", ""))
        except Exception:
            cfg_path = Path(str(row["config_path"])) if isinstance(row["config_path"], str) else None
    cfg = _load_config(cfg_path)
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    diffusion_cfg = cfg.get("diffusion", {}) if isinstance(cfg, dict) else {}
    spectral_cfg = cfg.get("spectral", {}) if isinstance(cfg, dict) else {}
    loss_cfg = cfg.get("loss", {}) if isinstance(cfg, dict) else {}
    sampling_cfg = cfg.get("sampling", {}) if isinstance(cfg, dict) else {}
    curriculum_cfg = data_cfg.get("curriculum", {}) if isinstance(data_cfg, dict) else {}

    meta: Dict[str, Any] = {}
    meta["run"] = row.get("run_id") or row.get("display_name") or row.get("run_axis") or "run"
    meta["dataset"] = row.get("dataset") or data_cfg.get("source") or data_cfg.get("family") or ""
    meta["architecture"] = (cfg.get("model", {}) or {}).get("type") if isinstance(cfg, dict) else None
    meta["snr_ratio"] = row.get("snr_ratio", diffusion_cfg.get("snr_ratio", spectral_cfg.get("snr_ratio")))
    meta["spectral_noise_shaping_strength"] = (
        spectral_cfg.get("noise_shaping_strength")
        or diffusion_cfg.get("uniform_corruption_scale")
        or spectral_cfg.get("uniform_corruption")
    )
    meta["snr_weighting_mode"] = _derive_weighting_mode(cfg)
    meta["spectral_adapter_placement"] = spectral_cfg.get("apply_to")
    meta["snr_schedule_mean"] = row.get("snr_schedule_mean") or row.get("snr_mean")
    meta["snr_effective"] = row.get("snr_effective")
    meta["initial_loss"] = row.get("loss_initial")
    meta["final_loss"] = row.get("loss_final")
    meta["loss_drop_per_second"] = row.get("loss_drop_per_second")
    meta["images_per_second"] = row.get("images_per_second")
    meta["runtime_seconds"] = row.get("runtime_seconds")
    meta["sampler_type"] = sampling_cfg.get("sampler_type")
    meta["sampling_steps"] = sampling_cfg.get("num_steps")
    meta["curriculum_mode"] = curriculum_cfg.get("mode") if isinstance(curriculum_cfg, dict) else None

    # Attach factor levels when available (Taguchi).
    factors = factor_mapping.get("factors", {}) if isinstance(factor_mapping, dict) else {}
    for key in ("snr_ratio", "spectral_noise_shaping_strength", "snr_weighting_mode", "spectral_adapter_placement"):
        if key not in meta and key in factors:
            meta[key] = factors[key]
    return meta


def _format_metadata_block(metas: List[dict[str, Any]]) -> List[str]:
    if not metas:
        return ["_No run metadata available._", ""]
    lines: List[str] = ["### Runs shown in this figure:"]
    for meta in metas:
        lines.append(f"- **{meta.get('run','run')}**")
        details = []
        for key in [
            "dataset",
            "architecture",
            "snr_ratio",
            "snr_schedule_mean",
            "snr_effective",
            "spectral_noise_shaping_strength",
            "snr_weighting_mode",
            "spectral_adapter_placement",
            "final_loss",
            "loss_drop_per_second",
            "images_per_second",
            "sampler_type",
            "curriculum_mode",
        ]:
            val = meta.get(key)
            if val is None or val == "":
                continue
            if isinstance(val, float):
                details.append(f"{key}: {val:.4f}")
            else:
                details.append(f"{key}: {val}")
        if details:
            lines.append("  - " + "\n  - ".join(details))
    lines.append("")
    return lines


def _metadata_table(metas: List[dict[str, Any]]) -> List[str]:
    return []


def _load_dataset(summary_dir: Optional[Path]) -> DatasetBundle:
    if summary_dir is None:
        return DatasetBundle(df=None, notes={})
    summary_path = summary_dir / "summary.csv"
    if not summary_path.exists():
        return DatasetBundle(df=None, notes={})

    clean_summary(summary_path)
    df = pd.read_csv(summary_path)
    df, notes = sanitize_metric_frame(df)
    df = compute_fft_corrected(df)
    df = _attach_run_dirs(df, summary_dir)
    return DatasetBundle(df=df, notes=notes)


def _select_runs(df: Optional[pd.DataFrame], metric: str, max_count: int, mode: str = "min") -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    if metric in df.columns:
        subset = df.dropna(subset=[metric])
        if subset.empty:
            subset = df
    else:
        subset = df
    if subset.empty:
        return None

    sorted_df = subset.sort_values(metric, ascending=(mode == "min")) if metric in subset.columns else subset
    trimmed = sorted_df.head(max_count).copy()
    trimmed, _ = assign_run_axis(trimmed.reset_index(drop=True))
    return trimmed


def _filter_tradeoff(df: Optional[pd.DataFrame], metric: str, max_points: int) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    if metric in df.columns:
        subset = df.dropna(subset=[metric]).sort_values(metric, ascending=False)
    else:
        subset = df
    trimmed = subset.head(max_points).copy()
    trimmed, _ = assign_run_axis(trimmed.reset_index(drop=True))
    return trimmed


def _generate_loss_curves(df: Optional[pd.DataFrame], label: str, out_path: Path, max_runs: int) -> Optional[pd.DataFrame]:
    selected = _select_runs(df, "loss_final", max_runs, mode="min")
    if selected is None:
        return None

    histories = collect_loss_histories(selected)
    if not histories:
        return None
    plot_loss_curves(histories, f"{label} – Loss Curves", out_path)
    return selected


def _generate_tradeoff(df: Optional[pd.DataFrame], label: str, out_path: Path, max_points: int) -> Optional[pd.DataFrame]:
    filtered = _filter_tradeoff(df, "loss_drop_per_second", max_points)
    if filtered is None:
        return None
    plot_tradeoff_scatter(
        filtered,
        "images_per_second",
        "loss_drop_per_second",
        f"{label} – Loss vs Speed",
        "Images per Second",
        "Loss Drop per Second",
        out_path=out_path,
    )
    return filtered


def _load_factor_mapping(taguchi_dir: Optional[Path]) -> dict[str, object]:
    if taguchi_dir is None:
        return {}
    mapping_path = taguchi_dir / "factor_mapping.json"
    if not mapping_path.exists():
        return {}
    try:
        return json.loads(mapping_path.read_text())
    except Exception:
        return {}


def _resolve_primary_factors(profile: str, overrides: Optional[list[str]], factor_mapping: dict[str, object]) -> list[str]:
    if overrides:
        return overrides
    default = PRIMARY_FACTORS_BY_PROFILE.get(profile, [])
    available = set((factor_mapping.get("factors") or {}).keys())
    if available:
        return [factor for factor in default if factor in available] or list(available)
    return default


def _build_factor_frame(summary_path: Path, primary_factors: list[str]) -> Optional[pd.DataFrame]:
    if not summary_path.exists():
        return None
    summary_df = pd.read_csv(summary_path)
    rows: list[dict[str, object]] = []
    for _, row in summary_df.iterrows():
        try:
            cfg_path = _resolve_config_path(Path(row["config_path"]), summary_path, row.get("run_id", ""))
            factors = _load_taguchi_factors(cfg_path)
        except Exception:
            factors = {}
        entry = row.to_dict()
        for factor, meta in factors.items():
            if primary_factors and factor not in primary_factors:
                continue
            if isinstance(meta, dict):
                entry[factor] = meta.get("level_label", meta.get("level_index", ""))
            else:
                entry[factor] = meta
        rows.append(entry)
    df = pd.DataFrame(rows)
    return df if not df.empty else None


def _taguchi_response_column(df: pd.DataFrame) -> Optional[str]:
    priority = [
        "loss_drop_per_second",
        "loss_drop_per_second_corrected",
        "mean_metric",
        "loss_final",
    ]
    for col in priority:
        if col in df.columns:
            return col
    return None


def _render_taguchi_figures(
    taguchi_dir: Optional[Path],
    primary_factors: list[str],
    factor_labels: dict[str, str],
    images_dir: Path,
    interactions_dir: Path,
) -> dict[str, object]:
    outputs: dict[str, object] = {}
    if taguchi_dir is None:
        return outputs

    report_path = taguchi_dir / "taguchi_report.csv"
    summary_path = taguchi_dir / "summary.csv"
    report_df = pd.read_csv(report_path) if report_path.exists() else None
    factor_df = _build_factor_frame(summary_path, primary_factors)
    response_col = _taguchi_response_column(report_df) if report_df is not None else None

    if report_df is not None and response_col:
        main_df = report_df[report_df["factor"].isin(primary_factors)].copy()
        if not main_df.empty and response_col in main_df.columns:
            main_df = main_df.rename(columns={response_col: "mean_response"})
            global_mean = main_df["mean_response"].mean()
            main_df["delta_from_global"] = main_df["mean_response"] - global_mean
            main_df["factor"] = main_df["factor"].map(lambda x: factor_labels.get(x, x))
            main_df["level"] = main_df["level"].map(_humanise_level)
            fig = plot_taguchi_main_effects(main_df, response_col="mean_response")
            fig.savefig(images_dir / "taguchi_main_effects_primary.png", bbox_inches="tight", dpi=300)
            outputs["taguchi_main_effects_primary.png"] = True

    if factor_df is not None:
        response_col = _taguchi_response_column(factor_df) or response_col
        usable_factors = [f for f in primary_factors if f in factor_df.columns]
        if response_col and usable_factors:
            contrib_df = compute_factor_contributions(factor_df, usable_factors, response_col)
            if not contrib_df.empty:
                contrib_df = contrib_df.rename(columns={"factor": "factor_display"})
                contrib_df["factor"] = contrib_df["factor_display"].map(lambda x: factor_labels.get(x, x))
                fig = plot_taguchi_contributions(contrib_df, response_col=response_col)
                fig.savefig(images_dir / "taguchi_contrib_primary.png", bbox_inches="tight", dpi=300)
                outputs["taguchi_contrib_primary.png"] = True
    return outputs


def _dataset_stats(df: Optional[pd.DataFrame]) -> dict[str, object]:
    if df is None or df.empty:
        return {}
    stats: dict[str, object] = {}
    if "loss_final" in df.columns:
        best = df.loc[df["loss_final"].idxmin()]
        stats["best_loss"] = (best.get("display_name") or best.get("run_id"), float(best["loss_final"]))
    if "images_per_second" in df.columns:
        fast = df.loc[df["images_per_second"].idxmax()]
        stats["fastest"] = (fast.get("display_name") or fast.get("run_id"), float(fast["images_per_second"]))
    if "loss_drop_per_second" in df.columns:
        drop = df.loc[df["loss_drop_per_second"].idxmax()]
        stats["best_drop"] = (drop.get("display_name") or drop.get("run_id"), float(drop["loss_drop_per_second"]))
    return stats


def _write_summary(
    output_dir: Path,
    generated_at: Optional[str],
    figure_flags: set[str],
    stats: dict[str, dict[str, object]],
    taguchi_factors: list[str],
    figure_meta: dict[str, list[dict[str, Any]]],
    section_meta: dict[str, list[dict[str, Any]]],
    report_root: Optional[Path],
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    lines.append("# Spectral Diffusion Report v2")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    if generated_at:
        lines.append(f"**Timestamp:** {generated_at}")
    lines.append("")

    def _maybe_img(name: str) -> list[str]:
        if name in figure_flags:
            return [f"![](images/{name})", ""]
        return ["_Figure unavailable._", ""]

    def _stability_bullets(label: str) -> list[str]:
        summary = stats.get(label, {})
        bullets: list[str] = []
        if "best_loss" in summary:
            run, val = summary["best_loss"]
            bullets.append(f"- Lowest final loss: {val:.4f} ({run}).")
        if "best_drop" in summary:
            run, val = summary["best_drop"]
            bullets.append(f"- Best loss drop/sec: {val:.4f} ({run}).")
        if not bullets:
            bullets.append("- No runs found.")
        bullets.append("")
        return bullets

    def _efficiency_bullets(label: str) -> list[str]:
        summary = stats.get(label, {})
        bullets: list[str] = []
        if "fastest" in summary:
            run, val = summary["fastest"]
            bullets.append(f"- Fastest throughput: {val:.2f} img/s ({run}).")
        if "best_drop" in summary:
            run, val = summary["best_drop"]
            bullets.append(f"- Best loss drop/sec: {val:.4f} ({run}).")
        if not bullets:
            bullets.append("- No efficiency data available.")
        bullets.append("")
        return bullets

    # Experiment metadata
    lines.append("## Experiment Metadata")
    lines.append("")
    lines.append(f"- Report path: {report_root or 'unknown'}")
    datasets_used = {m.get('dataset') for metas in section_meta.values() for m in metas if m.get('dataset')}
    arch_used = {m.get('architecture') for metas in section_meta.values() for m in metas if m.get('architecture')}
    lines.append(f"- Datasets used: {', '.join(sorted(datasets_used)) if datasets_used else 'n/a'}")
    lines.append(f"- Architectures included: {', '.join(sorted(arch_used)) if arch_used else 'n/a'}")
    total_runs = sum(len(metas) for metas in section_meta.values())
    lines.append(f"- Total runs analyzed: {total_runs}")
    lines.append(f"- Primary factors for this profile: {', '.join(taguchi_factors) if taguchi_factors else 'n/a'}")
    lines.append(f"- Generated at: {now}")
    # SNR interpretation
    def _agg(key: str) -> str:
        vals = [m[key] for metas in section_meta.values() for m in metas if m.get(key) is not None]
        if not vals:
            return "n/a"
        mean = float(sum(vals) / len(vals))
        if len(vals) > 1:
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            std = var ** 0.5
            return f"{mean:.4f} ± {std:.4f}"
        return f"{mean:.4f}"
    lines.append(f"- Effective SNR (measured): {_agg('snr_effective')}")
    lines.append(f"- Schedule SNR (from noise schedule): {_agg('snr_schedule_mean')}")
    lines.append(f"- User SNR ratio multiplier: {_agg('snr_ratio')}")
    lines.append("")
    lines.append("SNR definitions: snr_schedule = schedule-implied SNR; snr_effective = post-spectral scaling SNR; snr_ratio = user multiplier.")
    lines.append("")

    lines.append("## Stability & Convergence")
    lines.append("")
    lines.append("### Synthetic")
    lines.extend(_maybe_img("loss_curve_synthetic.png"))
    lines.extend(_format_metadata_block(figure_meta.get("loss_curve_synthetic.png", [])))
    lines.extend(_stability_bullets("synthetic"))
    if figure_meta.get("loss_curve_synthetic.png"):
        appendix_runs = output_dir / "appendix" / "runs_synthetic.md"
    lines.append("- Lowest final loss and best drop/sec listed above.")
    lines.append("_Curves reflect schedule SNR; effective (spectral) SNR is listed in metadata._")
    lines.append("### CIFAR-10")
    lines.extend(_maybe_img("loss_curve_cifar.png"))
    lines.extend(_format_metadata_block(figure_meta.get("loss_curve_cifar.png", [])))
    lines.extend(_stability_bullets("cifar"))
    lines.append("- Lowest final loss and best drop/sec listed above. Full run table available in appendix.")
    lines.append("_Curves reflect schedule SNR; effective (spectral) SNR is listed in metadata._")

    lines.append("## Efficiency vs Runtime")
    lines.append("")
    lines.append("### Synthetic")
    lines.extend(_maybe_img("tradeoff_loss_vs_speed_synthetic.png"))
    lines.extend(_format_metadata_block(figure_meta.get("tradeoff_loss_vs_speed_synthetic.png", [])))
    lines.extend(_efficiency_bullets("synthetic"))
    lines.append("### CIFAR-10")
    lines.extend(_maybe_img("tradeoff_loss_vs_speed_cifar.png"))
    lines.extend(_format_metadata_block(figure_meta.get("tradeoff_loss_vs_speed_cifar.png", [])))
    lines.extend(_efficiency_bullets("cifar"))

    lines.append("## Taguchi Factor Effects")
    lines.append("")
    lines.append("### Main effects (primary factors)")
    lines.extend(_maybe_img("taguchi_main_effects_primary.png"))
    lines.append(f"- Factors: {', '.join(taguchi_factors) if taguchi_factors else 'none detected.'}")
    lines.append("")
    lines.append("### Factor contributions")
    lines.extend(_maybe_img("taguchi_contrib_primary.png"))
    lines.append("- Variance attribution across the selected primary factors.")
    lines.append("")

    lines.append("## Qualitative Samples")
    lines.append("")
    lines.append("### Comparison 1")
    lines.extend(_maybe_img("samples_profile_comparison_1.png"))
    lines.append("### Comparison 2 (optional)")
    lines.extend(_maybe_img("samples_profile_comparison_2.png"))

    lines.append("## Key Takeaways")
    lines.append("")
    takeaways: list[str] = []
    for label in ("synthetic", "cifar"):
        summary = stats.get(label, {})
        if "best_loss" in summary:
            run, val = summary["best_loss"]
            takeaways.append(f"- {label.capitalize()}: {run} reached loss {val:.4f}.")
    if taguchi_factors:
        takeaways.append(f"- Primary Taguchi factors evaluated: {', '.join(taguchi_factors)}.")
    if not takeaways:
        takeaways.append("- No quantitative takeaways captured.")
    lines.extend(takeaways)
    lines.append("")

    lines.append("## Appendix")
    lines.append("")
    lines.append("- Additional diagnostics live under `appendix/` (noise chains, Taguchi interactions, distributions).")
    lines.append("")

    _export_report(output_dir / "summary.md", output_dir, lines)


def main() -> None:
    set_default_style()
    args = _parse_args()
    synthetic_dir, cifar_dir, taguchi_dir, output_dir = _resolve_dirs(args)
    images_dir, appendix_dir, interactions_dir = _ensure_dirs(output_dir)

    synthetic = _load_dataset(synthetic_dir)
    cifar = _load_dataset(cifar_dir)
    factor_mapping = _load_factor_mapping(taguchi_dir)
    figure_flags: set[str] = set()
    figure_meta: dict[str, list[dict[str, Any]]] = {}

    synthetic_loss_sel = _generate_loss_curves(synthetic.df, "Synthetic", images_dir / "loss_curve_synthetic.png", args.max_loss_runs)
    if synthetic_loss_sel is not None:
        figure_flags.add("loss_curve_synthetic.png")
        figure_meta["loss_curve_synthetic.png"] = [
            _extract_run_metadata(row, factor_mapping) for _, row in synthetic_loss_sel.iterrows()
        ]
    cifar_loss_sel = _generate_loss_curves(cifar.df, "CIFAR-10", images_dir / "loss_curve_cifar.png", args.max_loss_runs)
    if cifar_loss_sel is not None:
        figure_flags.add("loss_curve_cifar.png")
        figure_meta["loss_curve_cifar.png"] = [
            _extract_run_metadata(row, factor_mapping) for _, row in cifar_loss_sel.iterrows()
        ]

    synthetic_tradeoff_sel = _generate_tradeoff(synthetic.df, "Synthetic", images_dir / "tradeoff_loss_vs_speed_synthetic.png", args.max_tradeoff_points)
    if synthetic_tradeoff_sel is not None:
        figure_flags.add("tradeoff_loss_vs_speed_synthetic.png")
        figure_meta["tradeoff_loss_vs_speed_synthetic.png"] = [
            _extract_run_metadata(row, factor_mapping) for _, row in synthetic_tradeoff_sel.iterrows()
        ]
    cifar_tradeoff_sel = _generate_tradeoff(cifar.df, "CIFAR-10", images_dir / "tradeoff_loss_vs_speed_cifar.png", args.max_tradeoff_points)
    if cifar_tradeoff_sel is not None:
        figure_flags.add("tradeoff_loss_vs_speed_cifar.png")
        figure_meta["tradeoff_loss_vs_speed_cifar.png"] = [
            _extract_run_metadata(row, factor_mapping) for _, row in cifar_tradeoff_sel.iterrows()
        ]

    primary_factors = _resolve_primary_factors(args.profile, args.primary_factor, factor_mapping)
    factor_labels = {name: _humanise_name(name) for name in primary_factors}
    taguchi_flags = _render_taguchi_figures(taguchi_dir, primary_factors, factor_labels, images_dir, interactions_dir)
    figure_flags.update(taguchi_flags.keys())

    # Qualitative samples (best effort)
    pred_meta = _create_prediction_visual(synthetic.df, "Synthetic 32×32", images_dir / "samples_profile_comparison_1.png")
    if pred_meta:
        figure_flags.add("samples_profile_comparison_1.png")
    pred_meta_cifar = _create_prediction_visual(cifar.df, "CIFAR-10", images_dir / "samples_profile_comparison_2.png")
    if pred_meta_cifar:
        figure_flags.add("samples_profile_comparison_2.png")

    stats = {
        "synthetic": _dataset_stats(synthetic.df),
        "cifar": _dataset_stats(cifar.df),
    }

    synthetic_meta_all = [
        _extract_run_metadata(row, factor_mapping) for _, row in (synthetic.df.iterrows() if synthetic.df is not None else [])
    ]
    cifar_meta_all = [
        _extract_run_metadata(row, factor_mapping) for _, row in (cifar.df.iterrows() if cifar.df is not None else [])
    ]

    _write_summary(
        output_dir=output_dir,
        generated_at=args.generated_at,
        figure_flags=figure_flags,
        stats=stats,
        taguchi_factors=[factor_labels.get(f, f) for f in primary_factors],
        figure_meta=figure_meta,
        section_meta={"synthetic": synthetic_meta_all, "cifar": cifar_meta_all},
        report_root=args.report_root or _latest_report_root(),
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd
import yaml

from src.utils.plot_style import is_duplicate
from src.utils.report_sanitizer import sanitize_markdown

try:  # pragma: no cover - optional dependency
    import pypandoc

    PYPANDOC_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PYPANDOC_AVAILABLE = False

ROOT_DIR = Path(__file__).resolve().parents[2]

FACTOR_YAML_KEYS = {
    "spectral_adapter_placement": "spectral.apply_to",
    "spectral_loss_weighting": "spectral.weighting",
    "spectral_noise_shaping_strength": "diffusion.uniform_corruption / spectral.freq_equalized_noise",
    "phase_attention_capacity": "model.enable_phase_attention / model.phase_heads",
    "sampler_type": "sampling.sampler_type",
    "sampling_steps": "sampling.num_steps",
    "curriculum_mode": "training.curriculum",
    "lr_schedule_mode": "optim.lr_schedule",
}

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".gif"}

_NOTE_MESSAGES = {
    "targets_missing": "*Note: Target samples were unavailable in the archived artifacts.*",
    "predictions_missing": "*Note: Prediction samples were unavailable; placeholder panels shown.*",
    "noising_missing": "*Note: Intermediate noising snapshots were missing; placeholders shown.*",
}


@dataclass(frozen=True)
class FigureMetadata:
    """Structured view over optional figure metadata supplied by callers."""

    dataset: Optional[str] = None
    run_mapping: Mapping[str, str] = field(default_factory=dict)
    note: Optional[str] = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> "FigureMetadata":
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, Mapping):
            return cls()

        dataset = payload.get("dataset")
        if not isinstance(dataset, str):
            dataset = None

        run_mapping_raw = payload.get("run_mapping")
        if isinstance(run_mapping_raw, Mapping):
            run_mapping = {str(key): str(value) for key, value in run_mapping_raw.items()}
        else:
            run_mapping = {}

        note = payload.get("notes")
        if not isinstance(note, str):
            note = None

        return cls(dataset=dataset, run_mapping=run_mapping, note=note, raw=dict(payload))


@dataclass(frozen=True)
class FigureEntry:
    """Renderable representation for a single figure in the gallery."""

    path: Path
    rel_path: str
    caption: str
    metadata: FigureMetadata

    def to_lines(self) -> list[str]:
        lines = [f"![]({self.rel_path})"]
        if self.caption:
            lines.append(self.caption)
        if self.metadata.run_mapping:
            mapping_pairs = [f"{key} → {value}" for key, value in self.metadata.run_mapping.items()]
            lines.append(f"*Run mapping:* {'; '.join(mapping_pairs)}")
        note = self.metadata.note
        if note:
            message = _NOTE_MESSAGES.get(note)
            if message:
                lines.append(message)
        lines.append("")
        return lines


def _normalise_metadata_map(
    figure_metadata: Optional[Dict[str, Dict[str, Any]]]
) -> dict[str, FigureMetadata]:
    if not figure_metadata:
        return {}
    normalised: dict[str, FigureMetadata] = {}
    for key, payload in figure_metadata.items():
        normalised[key] = FigureMetadata.from_payload(payload)
    return normalised


def _discover_figure_entries(
    output_dir: Path, figure_metadata: Optional[Dict[str, Dict[str, Any]]]
) -> list[FigureEntry]:
    metadata_lookup = _normalise_metadata_map(figure_metadata)
    seen_hashes: set[str] = set()
    entries: list[FigureEntry] = []
    figure_files = sorted(
        path
        for path in output_dir.glob("**/*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    for img in figure_files:
        if is_duplicate(img, seen_hashes):
            continue
        metadata = metadata_lookup.get(img.name, FigureMetadata())
        rel = os.path.relpath(img, output_dir)
        caption = infer_caption_with_metadata(img, metadata.raw)
        entries.append(
            FigureEntry(path=img, rel_path=rel, caption=caption, metadata=metadata)
        )
    return entries


def _note_lines(metric_notes: Mapping[str, Mapping[str, bool]]) -> list[str]:
    psnr_missing = [
        key for key, notes in metric_notes.items() if notes.get("high_freq_psnr_missing")
    ]
    if not psnr_missing:
        return []
    dataset_labels = {
        "synthetic": "synthetic 32×32",
        "cifar": "CIFAR-10",
        "ablation": "ablation",
    }
    friendly = ", ".join(dataset_labels.get(key, key) for key in psnr_missing)
    return [
        "*Note: Some PSNR values could not be computed due to missing data in the"
        f" {friendly} benchmark.*",
        "",
    ]


def _collect_summary_rows(
    dataset_label: str, frame: Optional[pd.DataFrame]
) -> list[dict[str, Any]]:
    if frame is None:
        return []
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "dataset": dataset_label,
                "run_id": row.get("run_id", row.get("display_name", "unknown")),
                "loss_final": row.get("loss_final", "N/A"),
                "images_per_second": row.get(
                    "images_per_second_corrected",
                    row.get("images_per_second", "N/A"),
                ),
                "runtime_seconds": row.get(
                    "runtime_corrected",
                    row.get("runtime_seconds", row.get("runtime")),
                ),
                "runtime": row.get("runtime", row.get("runtime_seconds")),
                "high_freq_psnr": row.get("high_freq_psnr"),
            }
        )
    return rows


def _format_summary_table(summary_data: Sequence[dict[str, Any]]) -> list[str]:
    if not summary_data:
        return ["_No benchmark data available._", ""]

    summary_df = pd.DataFrame(summary_data)
    display_cols = ["dataset", "run_id", "loss_final", "images_per_second", "runtime_seconds"]
    if "high_freq_psnr" in summary_df.columns:
        display_cols.insert(3, "high_freq_psnr")
    df_disp = summary_df[[c for c in display_cols if c in summary_df.columns]].copy()
    float_cols = [
        c
        for c in ["loss_final", "images_per_second", "runtime_seconds", "high_freq_psnr"]
        if c in df_disp.columns
    ]
    for col in float_cols:
        df_disp[col] = df_disp[col].apply(
            lambda x: f"{x:.3f}" if pd.notnull(x) and isinstance(x, (int, float)) else str(x)
        )
    lines: list[str] = []
    try:
        lines.append(df_disp.to_markdown(index=False))
    except ImportError:
        header = "| " + " | ".join(display_cols) + " |"
        lines.append(header)
        lines.append("| " + " | ".join(["---"] * len(display_cols)) + " |")
        for _, row in df_disp.iterrows():
            row_str = " | ".join(str(row[col]) for col in display_cols if col in df_disp.columns)
            lines.append(f"| {row_str} |")
    lines.append("")
    return lines


def _summary_section(
    synthetic_df: Optional[pd.DataFrame],
    cifar_df: Optional[pd.DataFrame],
    metric_notes: Optional[Dict[str, Dict[str, bool]]],
) -> list[str]:
    metric_notes = metric_notes or {}
    summary_data: list[dict[str, Any]] = []
    summary_data.extend(_collect_summary_rows("Synthetic", synthetic_df))
    summary_data.extend(_collect_summary_rows("CIFAR-10", cifar_df))
    lines = ["## Summary Table", ""]
    lines.extend(_format_summary_table(summary_data))
    lines.extend(_note_lines(metric_notes))
    return lines


def _fft_section(fft_snapshot: Optional[dict[str, Any] | pd.DataFrame | str], output_dir: Path) -> list[str]:
    if fft_snapshot is None:
        return []
    lines = ["## FFT Scaling Summary", ""]
    if isinstance(fft_snapshot, dict):
        lines.append("```json")
        lines.append(json.dumps(fft_snapshot, indent=2))
        lines.append("```")
    elif isinstance(fft_snapshot, str):
        fft_path = Path(fft_snapshot)
        rel_fft = os.path.relpath(fft_path, output_dir)
        lines.append(f"FFT scaling details: [{rel_fft}]({rel_fft})")
    else:
        lines.append("FFT scaling details:")
        lines.append("")
        try:
            lines.append(fft_snapshot.to_markdown(index=False))
        except ImportError:
            header = "| " + " | ".join(fft_snapshot.columns) + " |"
            lines.append(header)
            lines.append("| " + " | ".join(["---"] * len(fft_snapshot.columns)) + " |")
            for _, row in fft_snapshot.iterrows():
                row_str = " | ".join(str(row[col]) for col in fft_snapshot.columns)
                lines.append(f"| {row_str} |")
    lines.append("")
    return lines


def _benchmark_overview_sections(
    synthetic_df: Optional[pd.DataFrame], cifar_df: Optional[pd.DataFrame]
) -> list[str]:
    lines: list[str] = []
    if synthetic_df is not None:
        lines.append("## Synthetic Benchmark")
        lines.append(
            "Synthetic Benchmark performance summary including throughput and spectral fidelity metrics."
        )
        lines.append("")
    if cifar_df is not None:
        lines.append("## CIFAR-10 Reconstruction Benchmark")
        lines.append(
            "CIFAR-10 Reconstruction Benchmark highlights covering loss, throughput, and high-frequency PSNR."
        )
        lines.append("")
    return lines


def _key_metrics_section(
    synthetic_df: Optional[pd.DataFrame], cifar_df: Optional[pd.DataFrame]
) -> list[str]:
    lines = ["## Key Metrics Highlights", ""]
    takeaways = _benchmark_takeaways(synthetic_df, cifar_df)
    lines.append("```")
    lines.extend(takeaways)
    lines.append("```")
    lines.append("")
    return lines


def _noise_definitions_section(output_dir: Path) -> list[str]:
    noise_md = output_dir / "noise_definitions.md"
    if not noise_md.exists():
        return []
    lines = ["## Noise Definitions", ""]
    snippet = noise_md.read_text(encoding="utf-8").splitlines()
    if snippet and snippet[0].startswith("##"):
        snippet = snippet[1:]
    lines.extend(snippet)
    lines.append("")
    return lines


def _figure_gallery_section(
    output_dir: Path, figure_metadata: Optional[Dict[str, Dict[str, Any]]]
) -> list[str]:
    lines = ["## Figure Gallery", ""]
    entries = _discover_figure_entries(output_dir, figure_metadata)
    if not entries:
        lines.append("_No figures found in output directory._")
        lines.append("")
        return lines
    for entry in entries:
        lines.extend(entry.to_lines())
    return lines


def _taguchi_insights_section(output_dir: Path) -> list[str]:
    insights_path = output_dir / "taguchi_insights.md"
    if not insights_path.exists():
        return []
    lines = ["## Taguchi Insights", ""]
    insight_lines = insights_path.read_text(encoding="utf-8").splitlines()
    if insight_lines and insight_lines[0].startswith("#"):
        insight_lines = insight_lines[1:]
    lines.extend(insight_lines)
    if insight_lines and insight_lines[-1].strip():
        lines.append("")
    return lines


def _export_report(report_path: Path, output_dir: Path, lines: Iterable[str]) -> None:
    raw_text = "\n".join(lines)
    report_path.write_text(raw_text, encoding="utf-8")
    try:
        sanitize_markdown(report_path, output_dir)
    except Exception as exc:  # pragma: no cover - best effort
        print(f"Warning: failed to sanitize markdown: {exc}")

    pdf_path = report_path.with_suffix(".pdf")
    if PYPANDOC_AVAILABLE:
        try:
            _ensure_pandoc()
            extra_args = _resource_paths_for(output_dir)
            pypandoc.convert_file(
                str(report_path),
                "pdf",
                outputfile=str(pdf_path),
                extra_args=extra_args or None,
            )
            print(f"PDF report generated: {pdf_path}")
        except OSError as exc:
            print(f"PDF generation skipped (pandoc unavailable): {exc}")
        except RuntimeError as exc:
            print(f"PDF generation failed (likely missing LaTeX): {exc}")
        except Exception as exc:
            print(f"PDF generation failed: {exc}")
    else:
        print("pypandoc not available, skipping PDF generation")

    html_path = report_path.with_suffix(".html")
    html_generated = False
    if PYPANDOC_AVAILABLE:
        try:
            extra_args = _resource_paths_for(output_dir)
            pypandoc.convert_file(
                str(report_path),
                "html",
                outputfile=str(html_path),
                extra_args=extra_args or None,
            )
            html_generated = True
        except Exception as exc:  # pragma: no cover - fallback handled below
            print(f"HTML generation via pandoc failed: {exc}")
    if not html_generated:
        try:
            import markdown

            html_content = markdown.markdown(
                report_path.read_text(encoding="utf-8"),
                extensions=["tables", "fenced_code"],
            )
            html_path.write_text(html_content, encoding="utf-8")
        except Exception as exc:  # pragma: no cover - basic fallback
            html_path.write_text(
                "<pre>" + report_path.read_text(encoding="utf-8") + "</pre>",
                encoding="utf-8",
            )
            print(f"Simple HTML fallback used due to: {exc}")


def _ensure_pandoc() -> None:
    if not PYPANDOC_AVAILABLE:
        return
    try:
        pypandoc.get_pandoc_path()
    except (OSError, RuntimeError):  # pragma: no cover - download helper
        try:
            pypandoc.download_pandoc()
            print("Downloaded pandoc via pypandoc.")
        except Exception as exc:
            print(f"Unable to download pandoc automatically: {exc}")


def infer_caption(fig_path: Path) -> str:
    """Generate a short contextual caption for ``fig_path``."""

    return infer_caption_with_metadata(fig_path, None)


def infer_caption_with_metadata(fig_path: Path, metadata: Optional[Dict[str, Any]]) -> str:
    stem = fig_path.stem.lower()
    parts = [part.lower() for part in fig_path.parts]

    dataset_token = (metadata or {}).get("dataset")
    dataset_lookup = {
        "synthetic": "Synthetic 32×32",
        "cifar": "CIFAR-10",
        "taguchi": "Taguchi analysis",
        "ablation": "Ablation",
    }
    if isinstance(dataset_token, str):
        dataset = dataset_lookup.get(dataset_token.lower(), dataset_token)
    elif any("synthetic" in part for part in parts):
        dataset = "Synthetic 32×32"
    elif any("cifar" in part for part in parts):
        dataset = "CIFAR-10"
    elif any("ablation" in part for part in parts):
        dataset = "Ablation"
    elif "taguchi" in stem:
        dataset = "Taguchi analysis"
    else:
        dataset = "General"

    if "loss_gradients" in stem:
        return "*Figure: Training dynamics across 300 steps, showing convergence behavior and stability.*"

    if "predictions" in stem:
        suffix = f" for {dataset} dataset" if dataset != "General" else ""
        return f"*Figure: Model predictions compared with ground truth samples{suffix}.*"

    if "noising" in stem:
        suffix = f" for {dataset} dataset" if dataset != "General" else ""
        return f"*Figure: Progressive denoising trajectory across the diffusion schedule{suffix}.*"

    if "taguchi_main" in stem:
        desc = "Taguchi main effects"
    elif "interaction" in stem:
        desc = "Interaction matrix"
    elif "loss_vs" in stem or "tradeoff" in stem:
        desc = "Loss vs throughput trade-off"
    elif "loss_curve" in stem:
        desc = "Training loss curves"
    elif "runtime" in stem or "images_per_second" in stem:
        desc = "Throughput comparison"
    elif "loss_metrics" in stem or "loss_final" in stem:
        desc = "Loss metrics comparison"
    elif "snr" in stem:
        desc = "Signal-to-noise summary"
    elif "ablation" in stem:
        desc = "Feature toggle ablation"
    else:
        desc = stem.replace("_", " ")

    metric_token = None
    known_metrics = [
        "loss_drop_per_second",
        "images_per_second",
        "loss_final",
        "high_freq_psnr",
        "fid",
    ]
    for token in known_metrics:
        if token in stem:
            metric_token = token.replace("_", " ")
            break
    if metric_token is None and "taguchi" in stem:
        metric_token = stem.split("taguchi", 1)[-1].strip("_-").replace("_", " ") or None

    metric_clause = f" – {metric_token}" if metric_token else ""
    dataset_clause = f" for {dataset} dataset" if dataset != "General" else ""
    return f"*Figure: {desc}{dataset_clause}{metric_clause}.*"


def _resource_paths_for(output_dir: Path) -> list[str]:
    """Return pandoc ``--resource-path`` arguments for ``output_dir``.

    Pandoc resolves relative image references against the process working
    directory by default. The generated markdown uses paths that are relative to
    ``summary.md`` itself (which lives inside the figures directory). We provide
    pandoc with an explicit resource search path that covers the figures folder,
    the overall report root, and any immediate sibling directories (for example
    ``taguchi`` or ``synthetic``) that may contain linked assets. This keeps the
    paths stable even when the finalize script is executed from a different
    directory or when the report is copied to another machine.
    """

    candidates: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        path_str = str(path)
        if path_str and path_str not in seen:
            seen.add(path_str)
            candidates.append(path)

    _add(output_dir)
    parent = output_dir.parent
    if parent != output_dir:
        _add(parent)
        for child in parent.iterdir():
            if child.is_dir():
                _add(child)

    resource_paths = os.pathsep.join(str(path) for path in candidates)
    return ["--resource-path", resource_paths] if resource_paths else []


def write_summary_markdown(
    synthetic_df: Optional[pd.DataFrame],
    cifar_df: Optional[pd.DataFrame],
    taguchi_report: Optional[pd.DataFrame],
    out_path: Path,
    descriptions: dict[str, str],
    generated_at: Optional[str] = None,
    fft_snapshot: Optional[dict[str, Any]] = None,
    taguchi_dir: Optional[Path] = None,
    figure_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    metric_notes: Optional[Dict[str, Dict[str, bool]]] = None,
) -> None:
    import pathlib

    output_dir = pathlib.Path(out_path).parent
    report_path = out_path
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    lines.append("# Spectral Diffusion Benchmark Report")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    if generated_at:
        lines.append("")
        lines.append(f"**Timestamp:** {generated_at}")
    lines.append("")

    lines.extend(_summary_section(synthetic_df, cifar_df, metric_notes))

    if taguchi_dir is not None:
        lines.extend(_factor_primer_lines(taguchi_dir, output_dir))

    factor_demo_section: list[str] = _factor_demo_lines(taguchi_dir, output_dir) if taguchi_dir else []

    lines.extend(_benchmark_overview_sections(synthetic_df, cifar_df))
    lines.extend(_fft_section(fft_snapshot, output_dir))
    lines.extend(_key_metrics_section(synthetic_df, cifar_df))

    if taguchi_dir is not None:
        sanity_section = _cifar_sanity_lines(taguchi_dir, output_dir)
        if sanity_section:
            lines.extend(sanity_section)
        diag_section = _diagnostics_lines(taguchi_dir, output_dir)
        if diag_section:
            lines.extend(diag_section)

    lines.extend(_noise_definitions_section(output_dir))
    lines.extend(_figure_gallery_section(output_dir, figure_metadata))
    lines.extend(_taguchi_insights_section(output_dir))

    if factor_demo_section:
        lines.extend(factor_demo_section)

    _export_report(report_path, output_dir, lines)


def _factor_primer_lines(taguchi_dir: Path, output_dir: Path) -> list[str]:
    registry_path = ROOT_DIR / "configs" / "taguchi" / "factor_registry.yaml"
    if not registry_path.exists():
        return []
    try:
        with registry_path.open("r", encoding="utf-8") as handle:
            registry = yaml.safe_load(handle) or {}
    except Exception:
        return []

    factors = registry.get("factors", {}) or {}
    if not factors:
        return []

    rows = []
    for name, info in factors.items():
        levels = info.get("levels", [])
        description = info.get("description", "")
        yaml_key = FACTOR_YAML_KEYS.get(name, name)
        rows.append(
            {
                "Factor": name,
                "Levels": " / ".join(str(level) for level in levels),
                "Description": description,
                "YAML Key": yaml_key,
            }
        )

    df = pd.DataFrame(rows)
    lines = ["## Factor Primer", ""]
    try:
        lines.append(df.to_markdown(index=False))
    except Exception:
        header = "| " + " | ".join(df.columns) + " |"
        lines.append(header)
        lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in df.columns) + " |")
    lines.append("")
    return lines


def _factor_demo_lines(taguchi_dir: Path, output_dir: Path) -> list[str]:
    root = taguchi_dir / "factors"
    if not root.exists():
        return []
    lines: list[str] = ["## Factor Demos", ""]
    for factor_dir in sorted(root.iterdir()):
        if not factor_dir.is_dir():
            continue
        lines.append(f"### {factor_dir.name}")
        has_level = False
        for level_dir in sorted([p for p in factor_dir.iterdir() if p.is_dir()]):
            images = sorted(level_dir.glob("demo_*.png"))
            if not images:
                continue
            has_level = True
            rel_imgs = [f"![]({os.path.relpath(img, output_dir)}){{ width=200px }}" for img in images]
            lines.append(f"#### {level_dir.name}")
            lines.append(" ".join(rel_imgs))
            lines.append("")
        if not has_level:
            lines.append("_No demos captured._")
            lines.append("")
    if len(lines) <= 2:
        return []
    return lines


def _cifar_sanity_lines(taguchi_dir: Path, output_dir: Path) -> list[str]:
    sanity_dir = taguchi_dir / "sanity"
    if not sanity_dir.exists():
        return []
    entries = sorted(sanity_dir.glob("*sanity_*.json"))
    cifar_entries = [p for p in entries if "cifar" in p.stem.lower()]
    if not cifar_entries:
        return []
    lines = ["## CIFAR Sanity Diagnostics", ""]
    for stats_path in cifar_entries:
        try:
            with stats_path.open("r", encoding="utf-8") as handle:
                stats = json.load(handle)
        except Exception:
            continue
        dataset = stats_path.stem.split("sanity_")[-1].upper()
        lines.append(f"### {dataset} – {stats_path.stem}")
        mean_val = stats.get("mean")
        std_val = stats.get("std")
        lines.append(f"- mean: {float(mean_val):.4f}" if mean_val is not None else "- mean: n/a")
        lines.append(f"- std: {float(std_val):.4f}" if std_val is not None else "- std: n/a")
        lines.append(f"- is_complex: {stats.get('is_complex')}")
        fft_error = stats.get("fft_reconstruction_error")
        if fft_error is not None:
            lines.append(f"- fft_reconstruction_error: {fft_error:.3e}")
        warning_flags = []
        if fft_error is not None and fft_error > 1e-2:
            warning_flags.append("fft_reconstruction_error > 1e-2")
        if not stats.get("is_complex", True):
            warning_flags.append("input tensor flagged as non-complex")
        if warning_flags:
            lines.append("⚠️  " + "; ".join(warning_flags))
        spatial_img = stats_path.with_name(stats_path.stem + "_spatial.png")
        fft_img = stats_path.with_name(stats_path.stem + "_fft_mag.png")
        for img in [spatial_img, fft_img]:
            if img.exists():
                rel = os.path.relpath(img, output_dir)
                lines.append(f"![]({rel})")
        lines.append("")
    return lines


def _diagnostics_lines(taguchi_dir: Path, output_dir: Path) -> list[str]:
    diag_dir = taguchi_dir / "diagnostics"
    if not diag_dir.exists():
        return []
    images = sorted(diag_dir.glob("*.png"))
    if not images:
        return []
    lines = ["## CIFAR Spectral Diagnostics", ""]
    for img in images:
        rel = os.path.relpath(img, output_dir)
        lines.append(f"![]({rel})")
    lines.append("")
    return lines


def _benchmark_takeaways(
    synthetic_df: Optional[pd.DataFrame],
    cifar_df: Optional[pd.DataFrame],
) -> list[str]:
    bullets: list[str] = []
    if synthetic_df is not None and not synthetic_df.empty:
        best_synth = synthetic_df.sort_values("loss_drop_per_second", ascending=False).iloc[0]
        bullets.append(
            f"Synthetic: {best_synth.get('run_id', best_synth.get('display_name', 'unknown'))} led throughput"
            f" at {best_synth.get('images_per_second', best_synth.get('images_per_second_corrected', 'n/a'))} images/sec."
        )
    if cifar_df is not None and not cifar_df.empty:
        best_cifar = cifar_df.sort_values("loss_final", ascending=True).iloc[0]
        bullets.append(
            f"CIFAR-10: {best_cifar.get('run_id', best_cifar.get('display_name', 'unknown'))} achieved"
            f" loss {best_cifar.get('loss_final', 'n/a'):.3f}."
        )
    if not bullets:
        bullets.append("No benchmark takeaways available.")
    return bullets

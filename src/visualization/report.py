

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

try:
    import pypandoc
    PYPANDOC_AVAILABLE = True
except ImportError:
    PYPANDOC_AVAILABLE = False

"""
Markdown report writer for Spectral Diffusion benchmark results.
"""

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

def write_summary_markdown(
    synthetic_df: Optional[pd.DataFrame],
    cifar_df: Optional[pd.DataFrame],
    taguchi_report: Optional[pd.DataFrame],
    out_path: Path,
    descriptions: dict[str, str],
    generated_at: Optional[str] = None,
    fft_snapshot: Optional[dict[str, Any]] = None,
    taguchi_dir: Optional[Path] = None,
) -> None:
    """
    Generate a markdown report summarizing benchmark results.
    Args:
        output_dir: Directory (str or Path) to write the report and search for images.
        summary_df: DataFrame with columns: model, runtime, fit_k, efficiency_corrected, fft_fraction_runtime
        fft_snapshot: Optional path to FFT scaling snapshot or dict with FFT scaling summary.
        description: Optional string with a brief description.
    """
    import pathlib
    output_dir = pathlib.Path(out_path).parent
    report_path = out_path
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    # Header
    lines.append(f"# Spectral Diffusion Benchmark Report")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    if generated_at:
        lines.append("")
        lines.append(f"**Timestamp:** {generated_at}")
    lines.append("")
    # Summary Table - combine synthetic and cifar data
    lines.append("## Summary Table")
    lines.append("")

    # Combine synthetic and cifar data for summary
    summary_data = []
    if synthetic_df is not None:
        for _, row in synthetic_df.iterrows():
            summary_data.append({
                "dataset": "Synthetic",
                "run_id": row.get("run_id", row.get("display_name", "unknown")),
                "loss_final": row.get("loss_final", "N/A"),
                "images_per_second": row.get("images_per_second_corrected", row.get("images_per_second", "N/A")),
                "runtime_seconds": row.get("runtime_corrected", row.get("runtime_seconds", row.get("runtime"))),
                "runtime": row.get("runtime", row.get("runtime_seconds")),
                "high_freq_psnr": row.get("high_freq_psnr")
            })
    if cifar_df is not None:
        for _, row in cifar_df.iterrows():
            summary_data.append({
                "dataset": "CIFAR-10",
                "run_id": row.get("run_id", row.get("display_name", "unknown")),
                "loss_final": row.get("loss_final", "N/A"),
                "images_per_second": row.get("images_per_second_corrected", row.get("images_per_second", "N/A")),
                "runtime_seconds": row.get("runtime_corrected", row.get("runtime_seconds", row.get("runtime"))),
                "runtime": row.get("runtime", row.get("runtime_seconds")),
                "high_freq_psnr": row.get("high_freq_psnr")
            })

    if summary_data:
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
            df_disp[col] = df_disp[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) and isinstance(x, (int, float)) else str(x))
        try:
            lines.append(df_disp.to_markdown(index=False))
        except ImportError:
            # Fallback to simple text table if tabulate is not available
            lines.append("| " + " | ".join(display_cols) + " |")
            lines.append("| " + " | ".join(["---"] * len(display_cols)) + " |")
            for _, row in df_disp.iterrows():
                row_str = " | ".join(str(row[col]) for col in display_cols)
                lines.append(f"| {row_str} |")
    else:
        lines.append("_No benchmark data available._")
    lines.append("")
    if taguchi_dir is not None:
        lines.extend(_factor_primer_lines(taguchi_dir, output_dir))

    if synthetic_df is not None:
        lines.append("## Synthetic Benchmark")
        lines.append("Synthetic Benchmark performance summary including throughput and spectral fidelity metrics.")
        lines.append("")
    if cifar_df is not None:
        lines.append("## CIFAR-10 Reconstruction Benchmark")
        lines.append("CIFAR-10 Reconstruction Benchmark highlights covering loss, throughput, and high-frequency PSNR.")
        lines.append("")
    # FFT scaling summary
    if fft_snapshot is not None:
        lines.append("## FFT Scaling Summary")
        lines.append("")
        if isinstance(fft_snapshot, dict):
            fft_summary = fft_snapshot
            lines.append("```json")
            lines.append(json.dumps(fft_summary, indent=2))
            lines.append("```")
        elif isinstance(fft_snapshot, str):
            fft_path = pathlib.Path(fft_snapshot)
            rel_fft = os.path.relpath(fft_path, output_dir)
            lines.append(f"FFT scaling details: [{rel_fft}]({rel_fft})")
        else:
            # fft_snapshot is a DataFrame, convert to markdown
            lines.append("FFT scaling details:")
            lines.append("")
            try:
                lines.append(fft_snapshot.to_markdown(index=False))
            except ImportError:
                # Fallback if tabulate not available
                lines.append("| " + " | ".join(fft_snapshot.columns) + " |")
                lines.append("| " + " | ".join(["---"] * len(fft_snapshot.columns)) + " |")
                for _, row in fft_snapshot.iterrows():
                    row_str = " | ".join(str(row[col]) for col in fft_snapshot.columns)
                    lines.append(f"| {row_str} |")
        lines.append("")
    # Key metrics highlights
    lines.append("## Key Metrics Highlights")
    lines.append("")
    takeaways = _benchmark_takeaways(synthetic_df, cifar_df)
    lines.append("```")
    lines.extend(takeaways)
    lines.append("```")
    lines.append("")

    if taguchi_dir is not None:
        lines.extend(_factor_demo_lines(taguchi_dir, output_dir))

    sanity_section = _cifar_sanity_lines(taguchi_dir, output_dir) if taguchi_dir else []
    if sanity_section:
        lines.extend(sanity_section)

    diag_section = _diagnostics_lines(taguchi_dir, output_dir) if taguchi_dir else []
    if diag_section:
        lines.extend(diag_section)

    noise_md = output_dir / "noise_definitions.md"
    if noise_md.exists():
        lines.append("## Noise Definitions")
        lines.append("")
        snippet = noise_md.read_text(encoding="utf-8").splitlines()
        if snippet and snippet[0].startswith("##"):
            snippet = snippet[1:]
        lines.extend(snippet)
        if snippet and snippet[-1].strip():
            lines.append("")

    # Figure gallery
    lines.append("## Figure Gallery")
    lines.append("")
    img_exts = [".png", ".jpg", ".jpeg", ".svg", ".gif"]
    figure_files = [f for f in os.listdir(output_dir) if os.path.splitext(f)[1].lower() in img_exts]
    if figure_files:
        for img in sorted(figure_files):
            lines.append(f"![{img}]({img})")
            lines.append("")
    else:
        lines.append("_No figures found in output directory._")

    insights_path = output_dir / "taguchi_insights.md"
    if insights_path.exists():
        lines.append("## Taguchi Insights")
        lines.append("")
        insight_lines = insights_path.read_text(encoding="utf-8").splitlines()
        if insight_lines and insight_lines[0].startswith("#"):
            insight_lines = insight_lines[1:]
        lines.extend(insight_lines)
        if insight_lines and insight_lines[-1].strip():
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")

    # Generate PDF if pypandoc is available
    if PYPANDOC_AVAILABLE:
        try:
            # Ensure pandoc is available
            import subprocess
            result = subprocess.run(['pandoc', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                pdf_path = out_path.with_suffix(".pdf")
                pypandoc.convert_file(str(out_path), "pdf", outputfile=str(pdf_path))
                print(f"PDF report generated: {pdf_path}")
            else:
                print("pandoc not found in PATH, skipping PDF generation")
        except Exception as e:
            print(f"PDF generation failed: {e}")
    else:
        print("pypandoc not available, skipping PDF generation")

    html_path = out_path.with_suffix(".html")
    html_generated = False
    if PYPANDOC_AVAILABLE:
        try:
            pypandoc.convert_file(str(out_path), "html", outputfile=str(html_path))
            html_generated = True
        except Exception as exc:  # pragma: no cover - fallback handled below
            print(f"HTML generation via pandoc failed: {exc}")
    if not html_generated:
        try:
            import markdown

            html_content = markdown.markdown(
                out_path.read_text(encoding="utf-8"),
                extensions=["tables", "fenced_code"],
            )
            html_path.write_text(html_content, encoding="utf-8")
        except Exception as exc:  # pragma: no cover - basic fallback
            html_path.write_text(
                "<pre>" + out_path.read_text(encoding="utf-8") + "</pre>",
                encoding="utf-8",
            )
            print(f"Simple HTML fallback used due to: {exc}")


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
            rel_imgs = [
                f"![]({os.path.relpath(img, output_dir)}){{ width=200px }}"
                for img in images
            ]
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
        lines.append(
            f"- mean: {float(mean_val):.4f}" if mean_val is not None else "- mean: n/a"
        )
        lines.append(
            f"- std: {float(std_val):.4f}" if std_val is not None else "- std: n/a"
        )
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

    print(f"Markdown report written to {out_path}")

def _benchmark_takeaways(synthetic_df, cifar_df):
    """
    Generate key metrics highlights from synthetic and cifar DataFrames.
    Returns a list of strings (lines).
    """
    takeaways = []
    try:
        # Combine data for analysis
        all_data = []
        if synthetic_df is not None:
            synthetic_copy = synthetic_df.copy()
            synthetic_copy['dataset'] = 'Synthetic'
            all_data.append(synthetic_copy)
        if cifar_df is not None:
            cifar_copy = cifar_df.copy()
            cifar_copy['dataset'] = 'CIFAR-10'
            all_data.append(cifar_copy)

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)

            # Find best loss
            if "loss_final" in combined_df.columns:
                best_loss_idx = combined_df["loss_final"].astype(float).idxmin()
                best_loss_row = combined_df.loc[best_loss_idx]
                takeaways.append(f"Lowest final loss: {best_loss_row.get('run_id', best_loss_row.get('display_name', 'unknown'))} ({best_loss_row['loss_final']:.3f})")

            # Find fastest throughput
            throughput_col = None
            throughput_suffix = "images/s"
            for candidate in ["images_per_second_corrected", "images_per_second"]:
                if candidate in combined_df.columns:
                    throughput_col = candidate
                    if candidate.endswith("_corrected"):
                        throughput_suffix = "images/s (FFT-corrected)"
                    break
            if throughput_col is not None:
                fastest_idx = combined_df[throughput_col].astype(float).idxmax()
                fastest_row = combined_df.loc[fastest_idx]
                takeaways.append(
                    f"Fastest throughput: "
                    f"{fastest_row.get('run_id', fastest_row.get('display_name', 'unknown'))} "
                    f"({fastest_row[throughput_col]:.1f} {throughput_suffix})"
                )

            # Find fastest convergence if available
            convergence_col = None
            convergence_label = "loss drop/s"
            for candidate in ["loss_drop_per_second_corrected", "loss_drop_per_second"]:
                if candidate in combined_df.columns:
                    convergence_col = candidate
                    if candidate.endswith("_corrected"):
                        convergence_label = "loss drop/s (FFT-corrected)"
                    break
            if convergence_col is not None:
                fastest_conv_idx = combined_df[convergence_col].astype(float).idxmax()
                fastest_conv_row = combined_df.loc[fastest_conv_idx]
                takeaways.append(
                    f"Fastest convergence: "
                    f"{fastest_conv_row.get('run_id', fastest_conv_row.get('display_name', 'unknown'))} "
                    f"({fastest_conv_row[convergence_col]:.3f} {convergence_label})"
                )
            if "high_freq_psnr" in combined_df.columns:
                hf_values = pd.to_numeric(combined_df["high_freq_psnr"], errors="coerce")
                if hf_values.notna().any():
                    best_hf_idx = hf_values.idxmax()
                    best_hf_row = combined_df.loc[best_hf_idx]
                    best_value = hf_values.loc[best_hf_idx]
                    takeaways.append(
                        f"Sharpest spectra: {best_hf_row.get('run_id', best_hf_row.get('display_name', 'unknown'))} ({best_value:.2f} dB)"
                    )
        else:
            takeaways.append("No benchmark data available for analysis")
    except Exception as e:
        takeaways.append(f"Could not compute all metrics: {e}")
    return takeaways

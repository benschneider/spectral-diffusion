

import os
import json
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import pypandoc
    PYPANDOC_AVAILABLE = True
except ImportError:
    PYPANDOC_AVAILABLE = False

"""
Markdown report writer for Spectral Diffusion benchmark results.
"""

def write_summary_markdown(
    synthetic_df: Optional[pd.DataFrame],
    cifar_df: Optional[pd.DataFrame],
    taguchi_report: Optional[pd.DataFrame],
    out_path: Path,
    descriptions: dict[str, str],
    generated_at: Optional[str] = None,
    fft_snapshot: Optional[dict[str, Any]] = None,
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

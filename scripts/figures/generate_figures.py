#!/usr/bin/env python
"""CLI wrapper for figure generation."""
from __future__ import annotations

import sys
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import subprocess

from src.visualization import collect
from src.visualization.figures import generate_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark and Taguchi figures.")
    parser.add_argument(
        "--report-root",
        type=Path,
        default=None,
        help="Root directory containing synthetic/, cifar/, taguchi/ subfolders (e.g. from run_smoke_report). "
        "If omitted, the latest 'results/smoke_report_*' directory is used when available.",
    )
    parser.add_argument(
        "--synthetic-dir",
        type=Path,
        default=None,
        help="Directory containing synthetic benchmark summary.csv",
    )
    parser.add_argument(
        "--cifar-dir",
        type=Path,
        default=None,
        help="Directory containing CIFAR benchmark summary.csv",
    )
    parser.add_argument(
        "--taguchi-dir",
        type=Path,
        default=None,
        help="Directory containing Taguchi summary.csv and taguchi_report.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated figures",
    )
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=None,
        help="Optional directory containing ablation summary.csv for feature toggles.",
    )
    parser.add_argument(
        "--descriptions",
        type=Path,
        default=None,
        help="Path to JSON file with section descriptions",
    )
    parser.add_argument(
        "--include-taguchi-effects",
        action="store_true",
        help="Run Taguchi ANOVA/interaction analysis and embed outputs in the report.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def _latest_report_root(prefix: str = "smoke_report") -> Optional[Path]:
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    candidates = [
        path
        for path in results_dir.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_paths(
    report_root: Optional[Path],
    synthetic_dir: Optional[Path],
    cifar_dir: Optional[Path],
    taguchi_dir: Optional[Path],
    output_dir: Optional[Path],
    descriptions: Optional[Path],
    ablation_dir: Optional[Path],
) -> Tuple[Path, Path, Path, Path, Path, Path]:
    root = report_root
    if root is None:
        root = _latest_report_root()

    if root is None:
        logging.debug("No report root detected; falling back to documentation defaults.")

    syn = synthetic_dir or (root / "synthetic" if root else Path("results/spectral_benchmark"))
    cif = cifar_dir or (root / "cifar" if root else Path("results/spectral_benchmark_cifar"))
    tag = taguchi_dir or (root / "taguchi" if root else Path("results/taguchi_spectral_docs"))

    out = output_dir or (root / "figures" if root else Path("docs/figures"))
    desc = descriptions or (root / "descriptions.json" if root and (root / "descriptions.json").exists() else Path("docs/descriptions.json"))
    abl = ablation_dir or (root / "ablation" if root else None)

    return syn, cif, tag, out, desc, abl


def _select_response_column(csv_path: Path) -> Optional[str]:
    """Pick a sensible default response column from the Taguchi CSV."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, nrows=1)
    preferred = [
        "loss_drop_per_second",
        "loss_drop_per_sec",
        "loss_drop_per_s",
        "images_per_second",
        "fid",
        "snr_metric",
    ]
    lowered = {col.lower(): col for col in df.columns}
    for key in preferred:
        if key in df.columns:
            return key
        if key.lower() in lowered:
            return lowered[key.lower()]
    return None


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    synthetic_dir, cifar_dir, taguchi_dir, output_dir, descriptions, ablation_dir = _resolve_paths(
        report_root=args.report_root,
        synthetic_dir=args.synthetic_dir,
        cifar_dir=args.cifar_dir,
        taguchi_dir=args.taguchi_dir,
        output_dir=args.output_dir,
        descriptions=args.descriptions,
        ablation_dir=args.ablation_dir,
    )

    logging.info("Synthetic summary root: %s", synthetic_dir)
    logging.info("CIFAR summary root:     %s", cifar_dir)
    logging.info("Taguchi summary root:   %s", taguchi_dir)
    logging.info("Figures output:         %s", output_dir)
    logging.info("Descriptions:           %s", descriptions)
    if ablation_dir:
        logging.info("Ablation summary root:  %s", ablation_dir)

    for label, directory in [
        ("Synthetic", synthetic_dir),
        ("CIFAR", cifar_dir),
        ("Taguchi", taguchi_dir),
    ]:
        summary_path = directory / "summary.csv"
        if not summary_path.exists():
            logging.warning("%s summary missing at %s", label, summary_path)

    # Clean summaries so display_name is present
    collect.clean_summary(synthetic_dir / "summary.csv")
    collect.clean_summary(cifar_dir / "summary.csv")
    collect.clean_summary(taguchi_dir / "summary.csv")
    if ablation_dir:
        collect.clean_summary(Path(ablation_dir) / "summary.csv")
    if args.include_taguchi_effects:
        taguchi_csv = taguchi_dir / "taguchi_report.csv"
        response_col = _select_response_column(taguchi_csv)
        if response_col is None:
            logging.warning(
                "Unable to determine response column for Taguchi analysis; skipping."
            )
        else:
            logging.info(
                "Running Taguchi analysis with response column '%s'.", response_col
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/analyze_taguchi_cli.py"),
                    "--csv",
                    str(taguchi_csv),
                    "--response-col",
                    response_col,
                    "--outdir",
                    str(output_dir),
                ],
                check=True,
            )
    generate_figures(
        synthetic_dir=synthetic_dir,
        cifar_dir=cifar_dir,
        taguchi_dir=taguchi_dir,
        output_dir=output_dir,
        descriptions_path=descriptions,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ablation_dir=ablation_dir,
    )


if __name__ == "__main__":
    main()

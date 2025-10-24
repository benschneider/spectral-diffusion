#!/usr/bin/env python
"""CLI wrapper for figure generation."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualization import collect
from src.visualization.figures import generate_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark and Taguchi figures.")
    parser.add_argument(
        "--synthetic-dir",
        type=Path,
        default=Path("results/spectral_benchmark"),
        help="Directory containing synthetic benchmark summary.csv",
    )
    parser.add_argument(
        "--cifar-dir",
        type=Path,
        default=Path("results/spectral_benchmark_cifar"),
        help="Directory containing CIFAR benchmark summary.csv",
    )
    parser.add_argument(
        "--taguchi-dir",
        type=Path,
        default=Path("results/taguchi_spectral_docs"),
        help="Directory containing Taguchi summary.csv and taguchi_report.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/figures"),
        help="Output directory for generated figures",
    )
    parser.add_argument(
        "--descriptions",
        type=Path,
        default=Path("docs/descriptions.json"),
        help="Path to JSON file with section descriptions",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    # Clean summaries so display_name is present
    collect.clean_summary(args.synthetic_dir / "summary.csv")
    collect.clean_summary(args.cifar_dir / "summary.csv")
    generate_figures(
        synthetic_dir=args.synthetic_dir,
        cifar_dir=args.cifar_dir,
        taguchi_dir=args.taguchi_dir,
        output_dir=args.output_dir,
        descriptions_path=args.descriptions,
    )


if __name__ == "__main__":
    main()

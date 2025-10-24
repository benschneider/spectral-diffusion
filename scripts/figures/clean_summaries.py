#!/usr/bin/env python
"""CLI wrapper for summary cleaning."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.visualization.collect import clean_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean benchmark summaries.")
    parser.add_argument("paths", nargs="*", type=Path, help="Summary CSV paths to clean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in args.paths:
        clean_summary(path)


if __name__ == "__main__":
    main()

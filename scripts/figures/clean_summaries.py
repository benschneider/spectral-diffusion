#!/usr/bin/env python
"""Deduplicate and clean summary CSVs before plotting."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

NAME_MAP = {
    "SPECTRAL_BENCH_baseline": "TinyUNet (Synthetic)",
    "SPECTRAL_BENCH_spectral": "SpectralUNet (Synthetic)",
    "cifar_baseline": "TinyUNet (CIFAR)",
    "cifar_spectral": "SpectralUNet (CIFAR)",
}


def clean_summary(path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["run_id"], keep="last")
    df["display_name"] = df["run_id"].map(NAME_MAP).fillna(df["run_id"])
    df.to_csv(path, index=False)
    print(f"Cleaned {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    for p in args.paths:
        clean_summary(p)


if __name__ == "__main__":
    main()

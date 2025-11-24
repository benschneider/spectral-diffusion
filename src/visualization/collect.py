"""Utilities for cleaning benchmark summary tables."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

NAME_MAP = {
    "SPECTRAL_BENCH_baseline": "TinyUNet (Synthetic)",
    "synthetic_tiny": "TinyUNet (Synthetic)",
    "cifar_baseline": "TinyUNet (CIFAR)",
    "cifar_tiny": "TinyUNet (CIFAR)",
    "spectral_baseline": "TinyUNet (CIFAR)",
}


def clean_summary(path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df.drop_duplicates(subset=["run_id"], keep="last")
    df["display_name"] = df["run_id"].map(NAME_MAP).fillna(df["run_id"])
    df.to_csv(path, index=False)

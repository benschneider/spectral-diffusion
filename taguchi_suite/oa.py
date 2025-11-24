"""Design matrix helpers for Taguchi OA grids."""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1] / "configs" / "taguchi"


def _load_csv(name: str) -> pd.DataFrame:
    path = BASE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"OA design not found: {path}")
    return pd.read_csv(path)


_DESIGNS: Dict[str, Callable[[], pd.DataFrame]] = {
    "L27": lambda: _load_csv("L27_extended.csv"),
}


@lru_cache(maxsize=None)
def _load_design(name: str) -> pd.DataFrame:
    if name not in _DESIGNS:
        raise ValueError(f"Unknown OA design '{name}'")
    return _DESIGNS[name]().copy()


def select_oa(num_factors: int, levels: int = 3) -> pd.DataFrame:
    if num_factors > 6 or levels != 3:
        raise ValueError("Taguchi design supports six ternary factors via L27.")
    return _load_design("L27").copy()

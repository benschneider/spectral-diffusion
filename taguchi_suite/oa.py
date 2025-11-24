"""Design matrix helpers for Taguchi OA grids."""

from __future__ import annotations

import itertools
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


def _l9() -> pd.DataFrame:
    rows = []
    idx = 1
    for a in range(1, 4):
        for b in range(1, 4):
            rows.append({"RunID": idx, "A": a, "B": b, "C": ((idx - 1) % 3) + 1})
            idx += 1
    return pd.DataFrame(rows)


def _l36() -> pd.DataFrame:
    rows = []
    idx = 1
    for combo in itertools.product(range(1, 4), repeat=4):
        if idx > 36:
            break
        rows.append(
            {
                "RunID": idx,
                "A": combo[0],
                "B": combo[1],
                "C": combo[2],
                "D": combo[3],
                "E": (combo[0] + combo[1]) % 3 + 1,
            }
        )
        idx += 1
    return pd.DataFrame(rows)


_DESIGNS: Dict[str, Callable[[], pd.DataFrame]] = {
    "L9": _l9,
    "L18": lambda: _load_csv("L18_mixed.csv"),
    "L27": lambda: _load_csv("L27_extended.csv"),
    "L36": _l36,
}


@lru_cache(maxsize=None)
def _load_design(name: str) -> pd.DataFrame:
    if name not in _DESIGNS:
        raise ValueError(f"Unknown OA design '{name}'")
    return _DESIGNS[name]().copy()


def select_oa(num_factors: int, levels: int = 3) -> pd.DataFrame:
    matches = []
    for name in _DESIGNS:
        df = _load_design(name)
        factor_columns = len(df.columns) - 1
        matches.append((factor_columns, name, df))
    matches.sort(key=lambda item: (item[0] < num_factors, item[0]))
    for factor_columns, name, df in matches:
        if factor_columns >= num_factors:
            return df.copy()
    return matches[-1][2].copy()

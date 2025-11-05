"""Backwards-compatible shim for label shortening helpers."""

from __future__ import annotations

from typing import Iterable, List

from .plot_style import shorten_label as _shorten_label
from .plot_style import shorten_labels as _shorten_labels

__all__ = ["shorten_label", "shorten_labels"]


def shorten_label(label: str, max_len: int = 25) -> str:
    """Return a normalised, human-friendly label for plotting axes."""

    return _shorten_label(label, max_len=max_len)


def shorten_labels(labels: Iterable[str], max_len: int = 25) -> List[str]:
    """Vectorised helper around :func:`shorten_label`."""

    return _shorten_labels(labels, max_len=max_len)

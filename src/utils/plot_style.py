"""Utility helpers for consistent publication-grade plotting."""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterable, Set

import matplotlib.pyplot as plt
import seaborn as sns

__all__ = [
    "set_default_style",
    "shorten_label",
    "shorten_labels",
    "reduce_tick_density",
    "declutter_texts",
    "hash_image",
    "is_duplicate",
    "autoscale_y",
]

_PREFIX_RE = re.compile(r"(config_|metrics_|summary_|full_report_)", re.IGNORECASE)
_TIMESTAMP_RE = re.compile(r"\d{4}[-_]??\d{2}[-_]??\d{2}(?:[T_]?\d{2}\d{2}(?:\d{2})?)?")
_LONG_DIGIT_RE = re.compile(r"([0-9]{8,})")


def set_default_style() -> None:
    """Configure matplotlib/seaborn defaults for compact figures."""
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def shorten_label(label: object, max_len: int = 20) -> str:
    """Normalise a categorical label for compact figure axes."""
    if label is None:
        return ""

    text = str(label).strip()
    if not text:
        return ""

    if os.sep in text or "/" in text:
        candidate = text.replace("\\", "/").rstrip("/")
        text = os.path.basename(candidate) or text

    text = _PREFIX_RE.sub("", text)

    if text.lower().endswith((".json", ".csv")):
        text = Path(text).stem

    timestamp_match = _TIMESTAMP_RE.search(text)
    if timestamp_match:
        raw = timestamp_match.group(0)
        digits = re.sub(r"[^0-9]", "", raw)
        if len(digits) >= 8:
            formatted = f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
            if len(digits) >= 12:
                formatted += f" {digits[8:10]}:{digits[10:12]}"
            text = _TIMESTAMP_RE.sub(formatted, text)

    def _shrink_digits(match: re.Match[str]) -> str:
        token = match.group(1)
        return token[:6] + "…"

    text = _LONG_DIGIT_RE.sub(_shrink_digits, text)

    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"

    return text


def shorten_labels(labels: Iterable[object], max_len: int = 20) -> list[str]:
    """Vectorised :func:`shorten_label`."""
    return [shorten_label(label, max_len=max_len) for label in labels]


def reduce_tick_density(ax: plt.Axes, max_ticks: int = 12) -> None:
    """Hide excess tick labels for crowded plots."""
    if max_ticks <= 0:
        return

    for axis in (ax.xaxis, ax.yaxis):
        labels = axis.get_ticklabels()
        visible = [lbl for lbl in labels if lbl.get_text() or lbl.get_visible()]
        count = len(visible)
        if count > max_ticks:
            step = max(1, count // max_ticks + (1 if count % max_ticks else 0))
            for idx, label in enumerate(labels):
                label.set_visible(idx % step == 0)


def declutter_texts(ax: plt.Axes, min_dist: float = 8.0) -> None:
    """Hide text annotations that collide within ``min_dist`` pixels."""
    kept: list[tuple[float, float]] = []
    for text in list(ax.texts):
        x, y = text.get_position()
        if any(abs(x - ox) < min_dist and abs(y - oy) < min_dist for ox, oy in kept):
            text.set_visible(False)
        else:
            kept.append((x, y))


def autoscale_y(ax: plt.Axes, sci_limit: float = 0.01) -> None:
    """Switch to scientific notation for very small y ranges."""

    try:
        ymin, ymax = ax.get_ylim()
    except ValueError:
        return

    if ymax == 0:
        return

    if abs(ymax) < sci_limit and abs(ymin) < sci_limit:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.offsetText.set_fontsize(7)


def hash_image(path: os.PathLike[str] | str) -> str:
    """Return an md5 hash for the given image path."""
    file_path = Path(path)
    data = file_path.read_bytes()
    return hashlib.md5(data).hexdigest()


def is_duplicate(path: os.PathLike[str] | str, seen_hashes: Set[str]) -> bool:
    """Return ``True`` if the image matches a previously seen hash."""
    digest = hash_image(path)
    if digest in seen_hashes:
        return True
    seen_hashes.add(digest)
    return False

#!/usr/bin/env python
"""Legacy wrapper for the v1 figure generator.

The full implementation now lives under scripts/figures/archive/generate_figures.py.
Prefer running scripts/generate_report_v2.py for the cleaned report output.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    legacy_path = Path(__file__).with_name("archive") / "generate_figures.py"
    if not legacy_path.exists():
        raise FileNotFoundError(f"Legacy generator not found: {legacy_path}")

    # Ensure repository root is importable (runpy does not adjust sys.path automatically).
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    print("⚠️  scripts/figures/generate_figures.py is deprecated. Use scripts/generate_report_v2.py when possible.")
    runpy.run_path(str(legacy_path), run_name="__main__")


if __name__ == "__main__":
    main()

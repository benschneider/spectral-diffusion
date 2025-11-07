#!/usr/bin/env python
"""CLI wrapper around :mod:`src.reporting.hdf5_packager`."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reporting.hdf5_packager import main as pack_main


if __name__ == "__main__":
    # The packaging logic already lives inside ``src.reporting.hdf5_packager``.
    # Re-exporting the CLI here keeps automation entry points under ``scripts/``.
    pack_main(sys.argv[1:])

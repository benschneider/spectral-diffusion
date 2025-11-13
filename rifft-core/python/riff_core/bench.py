"""Entry point for `python -m riff_core.bench`."""

from __future__ import annotations

import sys

from . import bridge


def main() -> int:
    return bridge.main()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

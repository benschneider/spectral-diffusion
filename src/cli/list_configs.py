"""Utility CLI for discovering available configuration files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

SUPPORTED_EXTENSIONS = (".yaml", ".yml")
CSV_EXTENSIONS = (".csv",)


@dataclass(frozen=True)
class ConfigRecord:
    """Metadata describing a discovered configuration file."""

    path: Path

    @property
    def extension(self) -> str:
        return self.path.suffix.lower()

    @property
    def kind(self) -> str:
        if self.extension in CSV_EXTENSIONS:
            return "csv"
        return "yaml"

    @property
    def name(self) -> str:
        return self.path.stem


def discover_configs(
    root: Path,
    include_csv: bool = False,
    filters: Sequence[str] | None = None,
) -> List[ConfigRecord]:
    """Locate configuration files under ``root`` sorted by path."""

    if not root.exists():
        return []

    valid_exts = set(SUPPORTED_EXTENSIONS)
    if include_csv:
        valid_exts.update(CSV_EXTENSIONS)

    filters_lower = [item.lower() for item in filters or ()]

    records: List[ConfigRecord] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in valid_exts:
            continue
        relative = path.relative_to(root)
        haystack = f"{relative.as_posix()} {path.stem}".lower()
        if filters_lower and not all(token in haystack for token in filters_lower):
            continue
        records.append(ConfigRecord(path=path))
    return records


def _format_table(rows: Iterable[Sequence[str]]) -> str:
    table = list(rows)
    if not table:
        return "No configuration files found."

    widths = [0] * len(table[0])
    for row in table:
        for idx, column in enumerate(row):
            widths[idx] = max(widths[idx], len(column))

    lines = []
    for row in table:
        padded = [column.ljust(widths[idx]) for idx, column in enumerate(row)]
        lines.append("  ".join(padded).rstrip())
    return "\n".join(lines)


def format_config_records(
    records: Sequence[ConfigRecord],
    root: Path,
    absolute: bool = False,
) -> str:
    """Create a table summarising ``records``."""

    if not records:
        return "No configuration files found."

    header = ("TYPE", "NAME", "PATH")
    lines = [header]
    for record in records:
        if absolute:
            display_path = str(record.path.resolve())
        else:
            display_path = str(record.path.relative_to(root))
        lines.append((record.kind, record.name, display_path))
    return _format_table(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List training/evaluation configuration files bundled with the project."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("configs"),
        help="Directory tree to scan (defaults to 'configs').",
    )
    parser.add_argument(
        "--include-csv",
        action="store_true",
        help="Include Taguchi array CSV files in the output.",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Substring filter that must appear in the config name or relative path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Show absolute paths instead of paths relative to --root.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = discover_configs(
        root=args.root,
        include_csv=args.include_csv,
        filters=args.filter,
    )

    if args.json:
        payload = [
            {
                "type": record.kind,
                "name": record.name,
                "path": str(record.path if args.absolute else record.path.relative_to(args.root)),
            }
            for record in records
        ]
        print(json.dumps(payload, indent=2))
    else:
        print(format_config_records(records=records, root=args.root, absolute=args.absolute))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

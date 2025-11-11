#!/usr/bin/env python
"""Collate existing run artifacts into a single HDF5 archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np


def _string_dataset(group: h5py.Group, name: str, payload: str) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    data = np.array(payload, dtype=dtype)
    group.create_dataset(name, data=data, compression="gzip")


def _store_file(
    group: h5py.Group,
    name: str,
    path: Path,
    prune: bool = False,
    force_json: bool = False,
) -> bool:
    if not path.exists():
        return False
    content = path.read_text(encoding="utf-8")
    if force_json:
        try:
            json.loads(content)
        except json.JSONDecodeError:
            pass
    _string_dataset(group, name, content)
    if prune:
        path.unlink()
    return True


def _store_jsonl(group: h5py.Group, name: str, path: Path, prune: bool = False) -> bool:
    if not path.exists():
        return False
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return False
    dtype = h5py.string_dtype(encoding="utf-8")
    dataset = np.array(lines, dtype=dtype)
    group.create_dataset(name, data=dataset, compression="gzip")
    if prune:
        path.unlink()
    return True


def _collect_run_files(run_dir: Path, group: h5py.Group, prune: bool) -> None:
    run_id = run_dir.name
    group.attrs["run_dir"] = str(run_dir)

    config_path = run_dir / "config.yaml"
    _store_file(group, "config_yaml", config_path, prune=prune)

    metrics_path = run_dir / "metrics" / f"{run_id}.json"
    _store_file(group, "metrics_json", metrics_path, prune=prune, force_json=True)

    summary_json = run_dir / "summary.json"
    _store_file(group, "summary_json", summary_json, prune=prune, force_json=True)

    _store_jsonl(group, "step_metrics", run_dir / "step_metrics.jsonl", prune=prune)

    logs = run_dir / "logs"
    if logs.is_dir():
        for log_file in sorted(logs.iterdir()):
            if log_file.suffix == ".log":
                _store_file(group, f"log_{log_file.name}", log_file, prune=prune)


def _collect_root_files(root: Path, group: h5py.Group, pattern: Iterable[str], prune: bool) -> None:
    for suffix in pattern:
        for path in sorted(root.glob(f"*{suffix}")):
            dataset_name = path.stem.replace(".", "_")
            _store_file(group, dataset_name, path, prune=prune)


def collate(
    runs_root: Path,
    output_path: Path,
    taguchi_root: Optional[Path] = None,
    prune: bool = False,
) -> Path:
    runs_dir = runs_root / "runs"
    if not runs_dir.exists():
        raise SystemExit(f"No run outputs found under {runs_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as archive:
        runs_group = archive.require_group("runs")
        for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            run_group = runs_group.create_group(run_dir.name)
            _collect_run_files(run_dir, run_group, prune)

        root_group = archive.require_group("taguchi")
        _collect_root_files(root=taguchi_root or runs_root, group=root_group, pattern=(".csv", ".json"), prune=prune)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collate run artifacts into HDF5.")
    parser.add_argument("--runs-root", type=Path, required=True, help="Base directory containing run outputs")
    parser.add_argument("--output", type=Path, required=True, help="Target HDF5 archive path")
    parser.add_argument(
        "--taguchi-root",
        type=Path,
        default=None,
        help="Base directory to collect Taguchi CSV/JSON artifacts from (defaults to --runs-root).",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Delete original JSON/CSV logs after they're stored in HDF5.",
    )
    args = parser.parse_args()

    collate(
        runs_root=args.runs_root,
        output_path=args.output,
        taguchi_root=args.taguchi_root,
        prune=args.prune,
    )
    print(f"HDF5 archive written to {args.output}")


if __name__ == "__main__":
    main()

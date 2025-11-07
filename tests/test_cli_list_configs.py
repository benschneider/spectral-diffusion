from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cli.list_configs import (
    ConfigRecord,
    discover_configs,
    format_config_records,
    main,
)


def _write(root: Path, relative: str) -> Path:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("key: value\n", encoding="utf-8")
    return destination


def test_discover_configs_returns_sorted_records(tmp_path: Path) -> None:
    configs_root = tmp_path / "configs"
    _write(configs_root, "baseline.yaml")
    _write(configs_root, "benchmarks/synthetic.yaml")
    _write(configs_root, "notes.txt")

    records = discover_configs(configs_root)
    assert [record.path.name for record in records] == [
        "baseline.yaml",
        "synthetic.yaml",
    ]


def test_discover_configs_can_include_csv_and_filter(tmp_path: Path) -> None:
    configs_root = tmp_path / "configs"
    _write(configs_root, "taguchi/L16.csv")
    _write(configs_root, "taguchi/L27.csv")
    _write(configs_root, "baseline.yaml")

    records = discover_configs(configs_root, include_csv=True, filters=["L2"])
    assert [record.path.name for record in records] == ["L27.csv"]


def test_format_config_records_builds_table(tmp_path: Path) -> None:
    configs_root = tmp_path / "configs"
    baseline = _write(configs_root, "baseline.yaml")
    records = [ConfigRecord(path=baseline)]
    formatted = format_config_records(records, configs_root)
    assert "TYPE" in formatted
    assert "baseline" in formatted
    assert "baseline.yaml" in formatted


def test_cli_main_prints_json_when_requested(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    configs_root = tmp_path / "configs"
    config_path = _write(configs_root, "baseline.yaml")

    main([
        "--root",
        str(configs_root),
        "--json",
    ])

    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert payload == [
        {
            "type": "yaml",
            "name": "baseline",
            "path": str(config_path.relative_to(configs_root)),
        }
    ]


def test_cli_main_handles_no_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    configs_root = tmp_path / "configs"
    configs_root.mkdir()

    main(["--root", str(configs_root)])
    captured = capsys.readouterr().out
    assert "No configuration files" in captured

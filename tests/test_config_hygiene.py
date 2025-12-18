from __future__ import annotations

from pathlib import Path

import pytest

from src.cli.common import load_config


def test_load_config_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "dup.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  type: baseline",
                "training:",
                "  batch_size: 1",
                "  batch_size: 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate YAML key"):
        load_config(config_path)


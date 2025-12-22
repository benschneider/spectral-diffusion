from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from src.cli.train import train_from_config


def _write_minimal_config(destination: Path) -> Path:
    config: Dict[str, Any] = {
        "model": {"type": "baseline", "channels": 3},
        "data": {
            "source": "synthetic",
            "channels": 3,
            "height": 4,
            "width": 4,
        },
        "training": {"batch_size": 1, "epochs": 1, "train_steps": 1, "log_every": 1},
        "diffusion": {"num_timesteps": 2, "beta_schedule": "linear"},
        "optim": {"lr": 1e-3},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return destination


class _DummyPipeline:
    """Test double that records configuration and fabricates outputs."""

    last_config: Dict[str, Any] | None = None
    run_calls: int = 0

    def __init__(self, config: Dict[str, Any], work_dir: Path, logger: Any) -> None:  # noqa: D401 - simple shim
        type(self).last_config = config
        self.work_dir = work_dir
        self.logger = logger

    def run(self) -> Dict[str, Any]:
        type(self).run_calls += 1
        return {"status": "ok", "num_steps": 4, "loss_mean": 0.1}

    def save_checkpoint(self, step: int) -> Path:
        checkpoint = self.work_dir / "checkpoints" / f"checkpoint_{step}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.touch()
        return checkpoint


def test_train_from_config_applies_cli_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _DummyPipeline.last_config = None
    _DummyPipeline.run_calls = 0
    monkeypatch.setattr("src.cli.train.TrainingPipeline", _DummyPipeline)
    config_path = _write_minimal_config(tmp_path / "config.yaml")
    output_dir = tmp_path / "outputs"

    result = train_from_config(
        config_path=config_path,
        variant="unet_tiny",
        output_dir=output_dir,
        dry_run=True,
        snr_ratio=0.75,
        spectral_operator_mode="radial",
        train_steps=12,
        checkpoint_every=3,
        eval_every=4,
        eval_num_samples=5,
        eval_sampling_steps=6,
        eval_seed=123,
    )

    assert result["run_id"]
    assert _DummyPipeline.last_config is not None
    config = _DummyPipeline.last_config
    assert config["model"]["type"] == "unet_tiny"
    assert config["diffusion"]["snr_ratio"] == pytest.approx(0.75)
    assert config["diffusion"]["spectral_operator_mode"] == "radial"
    assert config["training"]["train_steps"] == 12
    assert config["training"]["checkpoint_every"] == 3
    assert config["training"]["eval_every"] == 4
    assert config["training"]["eval_num_samples"] == 5
    assert config["training"]["eval_sampling_steps"] == 6
    assert config["training"]["eval_seed"] == 123


def test_train_from_config_writes_json_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _DummyPipeline.last_config = None
    _DummyPipeline.run_calls = 0
    monkeypatch.setattr("src.cli.train.TrainingPipeline", _DummyPipeline)
    config_path = _write_minimal_config(tmp_path / "config.yaml")
    output_dir = tmp_path / "outputs"

    result = train_from_config(
        config_path=config_path,
        output_dir=output_dir,
        json_log=True,
        log_level="INFO",
    )

    assert _DummyPipeline.run_calls == 1
    run_logs = output_dir / "runs" / result["run_id"] / "logs"
    json_log_path = run_logs / "train.jsonl"
    assert json_log_path.exists()
    with json_log_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline()
    payload = json.loads(first_line)
    assert payload["event"] == "run_start"


def test_train_from_config_validates_log_level(tmp_path: Path) -> None:
    config_path = _write_minimal_config(tmp_path / "config.yaml")

    with pytest.raises(ValueError):
        train_from_config(config_path=config_path, log_level="NOT_A_LEVEL")

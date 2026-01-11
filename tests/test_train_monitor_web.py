from __future__ import annotations

from pathlib import Path

import yaml

import scripts.debug.train_monitor_web as monitor


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _write_csv(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _base_config() -> dict:
    return {
        "seed": 123,
        "model": {"type": "unet_tiny"},
        "data": {"dataset": "cifar10"},
        "diffusion": {
            "num_timesteps": 1000,
            "snr_ratio": 0.8,
            "spectral_operator_mode": "radial",
        },
        "training": {
            "train_steps": 2000,
            "eval_every": 200,
            "eval_num_samples": 16,
            "eval_sampling_steps": 50,
            "eval_seed": 10000,
            "checkpoint_every": 200,
        },
        "sampling": {"sampler_type": "ddim", "sampling_steps": 50},
    }


def test_config_summary_reads_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(monitor, "ROOT", tmp_path)
    config_path = tmp_path / "configs" / "test.yaml"
    _write_yaml(config_path, _base_config())

    summary = monitor._config_summary(config_path)

    assert summary["config"] == "configs/test.yaml"
    assert summary["seed"] == 123
    assert summary["model"] == "unet_tiny"
    assert summary["dataset"] == "cifar10"
    assert summary["num_timesteps"] == 1000
    assert summary["snr_ratio"] == 0.8
    assert summary["spectral_operator_mode"] == "radial"
    assert summary["train_steps"] == 2000
    assert summary["eval_every"] == 200
    assert summary["eval_num_samples"] == 16
    assert summary["eval_sampling_steps"] == 50
    assert summary["eval_seed"] == 10000
    assert summary["checkpoint_every"] == 200
    assert summary["sampler_type"] == "ddim"
    assert summary["sampling_steps"] == 50


def test_read_csv_tail_limits_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    _write_csv(
        csv_path,
        [
            "step,loss",
            "1,1.0",
            "2,0.9",
            "3,0.8",
        ],
    )
    rows = monitor._read_csv_tail(csv_path, limit=2)
    assert [row["step"] for row in rows] == ["2", "3"]


def test_coerce_row_casts_numbers() -> None:
    row = {"step": "10", "loss": "1.5", "name": "demo"}
    coerced = monitor._coerce_row(row)
    assert coerced["step"] == 10
    assert coerced["loss"] == 1.5
    assert coerced["name"] == "demo"


def test_run_info_uses_latest_metrics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(monitor, "ROOT", tmp_path)
    run_dir = tmp_path / "runs" / "demo_run"
    config_path = run_dir / "config.yaml"
    _write_yaml(config_path, _base_config())
    _write_csv(
        run_dir / "diagnostics" / "training_history.csv",
        [
            "step,loss,grad_norm",
            "1,1.0,2.0",
            "2,0.5,1.0",
        ],
    )

    info = monitor._run_info(run_dir)
    assert info["run_dir"] == "runs/demo_run"
    assert info["last_step"] == 2
    assert info["last_loss"] == 0.5
    assert info["last_grad_norm"] == 1.0

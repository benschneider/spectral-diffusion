import json
import shutil
from pathlib import Path

import torch
import yaml
from torchvision.utils import save_image

from src.cli.evaluate import evaluate_directory
from src.cli.sample import sample_from_run
from src.cli.train import train_from_config
from src.evaluation.metrics import LPIPS_AVAILABLE


def _write_config(tmp_path: Path, config: dict) -> Path:
    path = tmp_path / "config.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return path


def _synthetic_config() -> dict:
    return {
        "model": {"type": "baseline", "channels": 3},
        "data": {"source": "synthetic", "channels": 3, "height": 8, "width": 8},
        "training": {"batch_size": 2, "num_batches": 2, "epochs": 1},
        "diffusion": {"num_timesteps": 4, "beta_schedule": "linear"},
        "sampling": {"enabled": False},
    }


def test_train_cli_dry_run(tmp_path):
    config_path = _write_config(tmp_path, _synthetic_config())
    result = train_from_config(
        config_path=config_path,
        output_dir=tmp_path,
        dry_run=True,
        cleanup=True,
    )
    assert result["metrics"]["status"] == "dry_run"
    run_root = tmp_path / "runs"
    assert not run_root.exists() or not any(run_root.iterdir())
    summary_path = tmp_path / "summary.csv"
    if summary_path.exists():
        lines = [line for line in summary_path.read_text().splitlines() if line.strip()]
        assert len(lines) <= 1


def test_train_cli_variant_spectral_sets_model(tmp_path):
    config_path = _write_config(tmp_path, _synthetic_config())
    result = train_from_config(
        config_path=config_path,
        output_dir=tmp_path,
        variant="spectral",
        dry_run=False,
        cleanup=False,
    )
    saved_config = yaml.safe_load(Path(result["config_path"]).read_text())
    assert saved_config["model"]["type"] == "unet_spectral"
    system_path = Path(result["config_path"]).parent / "system.json"
    assert system_path.exists()
    data = json.loads(system_path.read_text())
    assert data["run_id"] == result["run_id"]
    shutil.rmtree(Path(result["config_path"]).parent, ignore_errors=True)


def test_train_cli_json_log_writes_jsonl(tmp_path):
    config_path = _write_config(tmp_path, _synthetic_config())
    result = train_from_config(
        config_path=config_path,
        output_dir=tmp_path,
        dry_run=False,
        cleanup=False,
        json_log=True,
    )
    run_dir = Path(result["config_path"]).parent
    json_log_path = run_dir / "logs" / "train.jsonl"
    assert json_log_path.exists()
    lines = [line for line in json_log_path.read_text().splitlines() if line.strip()]
    assert lines
    record = json.loads(lines[0])
    assert record["level"] in {"INFO", "WARNING", "ERROR"}
    shutil.rmtree(run_dir.parent, ignore_errors=True)
    summary_path = tmp_path / "summary.csv"
    if summary_path.exists():
        summary_path.unlink()


def test_sample_cli_generates_artifacts(tmp_path):
    config = _synthetic_config()
    config_path = _write_config(tmp_path, config)

    train_result = train_from_config(
        config_path=config_path,
        output_dir=tmp_path,
        dry_run=False,
        cleanup=False,
    )
    run_dir = Path(train_result["config_path"]).parent
    checkpoint_path = Path(train_result["checkpoint_path"])
    assert checkpoint_path.exists()
    system_path = run_dir / "system.json"
    assert system_path.exists()

    sample_result = sample_from_run(
        run_dir=run_dir,
        checkpoint=checkpoint_path,
        num_samples=2,
        num_steps=3,
        sampler_type="ddim",
        tag="unittest",
    )
    images_dir = Path(sample_result["images_dir"])
    assert images_dir.exists()
    assert (images_dir / "grid.png").exists()
    assert any(images_dir.glob("sample_*.png"))

    metadata_path = Path(sample_result["metadata_path"])
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["checkpoint_path"] == str(checkpoint_path)
    assert metadata["num_samples"] == 2
    assert metadata["num_steps"] == 3
    assert metadata["sampler_type"] == "ddim"

    # cleanup artifacts
    shutil.rmtree(run_dir.parent, ignore_errors=True)
    summary_path = tmp_path / "summary.csv"
    if summary_path.exists():
        summary_path.unlink()


def test_evaluate_cli_metrics(tmp_path):
    # Prepare run with samples
    config_path = _write_config(tmp_path, _synthetic_config())
    train_result = train_from_config(
        config_path=config_path,
        output_dir=tmp_path,
        dry_run=False,
        cleanup=False,
    )
    sample_result = sample_from_run(
        run_dir=Path(train_result["config_path"]).parent,
        checkpoint=Path(train_result["checkpoint_path"]),
        num_samples=1,
        num_steps=2,
    )
    generated_dir = Path(sample_result["images_dir"])

    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    for img_path in generated_dir.glob("sample_*.png"):
        shutil.copy(img_path, reference_dir / img_path.name)
    shutil.copy(generated_dir / "grid.png", reference_dir / "grid.png")

    result = evaluate_directory(
        generated_dir=generated_dir,
        reference_dir=reference_dir,
        image_size=None,
        use_fid=False,
        use_lpips=LPIPS_AVAILABLE,
        strict_filenames=False,
        update_metadata=True,
    )
    assert result["metrics"]["mse"] == 0.0
    assert result["metrics_path"].exists()
    if LPIPS_AVAILABLE:
        assert "lpips" in result["metrics"]

    metadata_path = Path(sample_result["metadata_path"])
    metadata = json.loads(metadata_path.read_text())
    assert "evaluation" in metadata
    evaluation_meta = metadata["evaluation"]
    assert evaluation_meta["metrics_path"] == str(result["metrics_path"])
    assert "metrics" in evaluation_meta
    if LPIPS_AVAILABLE:
        assert "lpips" in evaluation_meta["metrics"]

    shutil.rmtree((tmp_path / "runs"), ignore_errors=True)

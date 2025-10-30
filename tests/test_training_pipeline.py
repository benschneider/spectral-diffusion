import copy

import torch
import pytest

from src.training.pipeline import TrainingPipeline


def _build_base_config() -> dict:
    """Return a minimal config that trains quickly on synthetic data."""
    return {
        "model": {
            "type": "baseline",
            "channels": 3,
        },
        "data": {
            "source": "synthetic",
            "channels": 3,
            "height": 8,
            "width": 8,
        },
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "num_batches": 3,
            "log_every": 1,
        },
        "diffusion": {
            "num_timesteps": 8,
            "beta_schedule": "linear",
            "prediction_type": "eps",
            "snr_weighting": True,
            "snr_transform": "snr_clamped",
        },
        "loss": {
            "reduction": "mean",
        },
        "optim": {
            "lr": 1e-3,
            "weight_decay": 0.0,
        },
        "sampling": {
            "enabled": True,
            "sampler_type": "ddpm",
            "num_samples": 2,
            "num_steps": 4,
        },
    }


def test_training_pipeline_runs_end_to_end(tmp_path):
    torch.manual_seed(0)
    config = copy.deepcopy(_build_base_config())
    pipeline = TrainingPipeline(config=config, work_dir=tmp_path)

    metrics = pipeline.run()

    expected_steps = (
        config["training"]["epochs"] * config["training"]["num_batches"]
    )
    assert metrics["status"] == "ok"
    assert metrics["num_steps"] == expected_steps
    assert metrics["loss_mean"] is not None
    assert metrics["mae_mean"] is not None
    assert "sampling_images_dir" not in metrics


def test_training_pipeline_reports_spectral_stats(tmp_path):
    torch.manual_seed(0)
    config = copy.deepcopy(_build_base_config())
    config["sampling"]["enabled"] = False
    config["model"] = {
        "type": "unet_tiny",
        "channels": 3,
        "base_channels": 8,
        "depth": 1,
        "spectral": {
            "enabled": True,
            "weighting": "radial",
            "apply_to": ["input"],
            "normalize": True,
            "per_block": False,
        },
    }

    pipeline = TrainingPipeline(config=config, work_dir=tmp_path)
    metrics = pipeline.run()

    assert metrics["loss_mean"] is not None
    assert metrics["spectral_calls"] >= 1.0
    assert metrics["spectral_time_seconds"] >= 0.0


def test_generate_samples_falls_back_to_ddpm(tmp_path):
    torch.manual_seed(0)
    config = copy.deepcopy(_build_base_config())
    config["sampling"]["sampler_type"] = "not_a_sampler"
    pipeline = TrainingPipeline(config=config, work_dir=tmp_path)

    result = pipeline.generate_samples()
    assert result["sampler_type"] == "ddpm"
    assert result["images_dir"].exists()


def test_training_pipeline_regression_baseline(tmp_path):
    torch.manual_seed(1337)
    config = copy.deepcopy(_build_base_config())
    config["training"] = {
        "batch_size": 4,
        "epochs": 2,
        "num_batches": 6,
        "log_every": 10,
    }
    config["diffusion"]["num_timesteps"] = 16
    config["optim"]["lr"] = 5e-4

    pipeline = TrainingPipeline(config=config, work_dir=tmp_path)
    metrics = pipeline.run()

    expected_loss = 9.3600
    expected_loss_drop = -0.4518
    assert metrics["status"] == "ok"
    assert pytest.approx(metrics["loss_mean"], rel=0.05) == expected_loss
    assert pytest.approx(metrics["loss_drop"], rel=0.1) == expected_loss_drop


def test_training_pipeline_with_uniform_corruption(tmp_path):
    torch.manual_seed(42)
    config = copy.deepcopy(_build_base_config())
    config["diffusion"]["uniform_corruption"] = True
    pipeline = TrainingPipeline(config=config, work_dir=tmp_path)
    metrics = pipeline.run()
    assert metrics["status"] == "ok"


def test_generate_samples_with_masf_sampler(tmp_path):
    torch.manual_seed(7)
    config = copy.deepcopy(_build_base_config())
    pipeline = TrainingPipeline(config=config, work_dir=tmp_path)
    pipeline.run()
    result = pipeline.generate_samples(sampler_type="masf")
    assert result["sampler_type"] == "masf"
    assert result["num_samples"] == config["sampling"]["num_samples"]

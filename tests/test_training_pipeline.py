import copy

import torch

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

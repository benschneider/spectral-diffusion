import torch

from src.core import build_model
from src.training.builders import build_dataloader, build_optimizer


def _synthetic_config():
    return {
        "model": {"type": "baseline", "channels": 3},
        "data": {"source": "synthetic", "channels": 3, "height": 8, "width": 8},
        "training": {"batch_size": 4, "num_batches": 3},
        "optim": {"lr": 5e-4, "weight_decay": 1e-2},
    }


def test_build_dataloader_synthetic_shapes_and_length():
    config = _synthetic_config()
    loader = build_dataloader(config)
    batch = next(iter(loader))
    xb, yb = batch
    assert xb.shape == (config["training"]["batch_size"], 3, 8, 8)
    assert yb.shape == xb.shape
    assert len(loader) == config["training"]["num_batches"]


def test_build_optimizer_uses_config_hyperparams():
    config = _synthetic_config()
    model = build_model(config["model"])
    optimizer = build_optimizer(model, config)

    assert isinstance(optimizer, torch.optim.AdamW)
    group = optimizer.param_groups[0]
    assert group["lr"] == config["optim"]["lr"]
    assert group["weight_decay"] == config["optim"]["weight_decay"]


def test_build_dataloader_piecewise_family():
    config = _synthetic_config()
    config["data"]["family"] = "piecewise"
    config["data"]["piecewise"] = {
        "pattern_types": ["checkerboard", "stripes"],
        "edge_blur_sigma": 0.5,
    }
    loader = build_dataloader(config)
    xb, yb = next(iter(loader))
    assert xb.shape == yb.shape == (config["training"]["batch_size"], 3, 8, 8)
    assert xb.max() <= 1.0 and xb.min() >= -1.0


def test_build_dataloader_random_field_family():
    config = _synthetic_config()
    config["data"]["family"] = "random_field"
    config["data"]["random_field"] = {"alpha_range": [0.5, 1.5]}
    loader = build_dataloader(config)
    xb, _ = next(iter(loader))
    assert xb.shape == (config["training"]["batch_size"], 3, 8, 8)
    assert torch.isfinite(xb).all()

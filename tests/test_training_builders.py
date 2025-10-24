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

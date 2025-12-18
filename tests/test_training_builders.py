from pathlib import Path

import pytest
import torch
from PIL import Image

from src.core import build_model
from src.training.builders import build_dataloader, build_optimizer
from src.training.optimizers import Adafactor, Lion


def _synthetic_config():
    return {
        "model": {"type": "baseline", "channels": 3},
        "data": {"source": "synthetic", "channels": 3, "height": 8, "width": 8},
        "training": {"batch_size": 4},
        "optim": {"lr": 5e-4, "weight_decay": 1e-2},
    }


def test_build_dataloader_synthetic_shapes_and_length():
    config = _synthetic_config()
    loader = build_dataloader(config)
    batch = next(iter(loader))
    xb, yb = batch
    assert xb.shape == (config["training"]["batch_size"], 3, 8, 8)
    assert yb.shape == xb.shape
    assert len(loader) > 0


def test_build_optimizer_uses_config_hyperparams():
    config = _synthetic_config()
    model = build_model(config["model"])
    optimizer = build_optimizer(model, config)

    assert isinstance(optimizer, torch.optim.AdamW)
    group = optimizer.param_groups[0]
    assert group["lr"] == config["optim"]["lr"]
    assert group["weight_decay"] == config["optim"]["weight_decay"]


def test_build_optimizer_supports_lion():
    config = _synthetic_config()
    config["optim"]["type"] = "lion"
    model = build_model(config["model"])
    optimizer = build_optimizer(model, config)

    assert isinstance(optimizer, Lion)
    group = optimizer.param_groups[0]
    assert group["lr"] == config["optim"]["lr"]


def test_build_optimizer_supports_adafactor():
    config = _synthetic_config()
    config["optim"]["type"] = "adafactor"
    config["optim"]["relative_step"] = False
    config["optim"]["scale_parameter"] = False
    model = build_model(config["model"])
    optimizer = build_optimizer(model, config)

    assert isinstance(optimizer, Adafactor)
    group = optimizer.param_groups[0]
    assert group["lr"] == config["optim"]["lr"]


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


def test_build_cifar10_dataloader_auto_download(monkeypatch, tmp_path):
    calls = {}

    def _fake_cifar10(root, train, download, transform):
        calls["root"] = root
        calls["download"] = download

        class _Dummy(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                img = Image.new("RGB", (32, 32))
                tensor = transform(img) if transform else torch.zeros(3, 32, 32)
                return tensor, 0

        return _Dummy()

    monkeypatch.setattr("src.training.builders.datasets.CIFAR10", _fake_cifar10)

    config = {
        "data": {"source": "cifar10", "root": str(tmp_path)},
        "training": {"batch_size": 2},
    }
    loader = build_dataloader(config)
    xb, yb = next(iter(loader))
    assert xb.shape == (2, 3, 32, 32)
    assert yb.shape == xb.shape
    assert Path(calls["root"]).exists()
    assert calls["download"] is True


def test_build_cifar10_dataloader_manual_download_error(monkeypatch, tmp_path):
    def _fail_cifar10(**kwargs):
        raise RuntimeError("dataset unavailable")

    monkeypatch.setattr("src.training.builders.datasets.CIFAR10", _fail_cifar10)

    config = {
        "data": {"source": "cifar10", "root": str(tmp_path), "download": False},
        "training": {"batch_size": 2},
    }
    with pytest.raises(RuntimeError) as excinfo:
        build_dataloader(config)
    message = str(excinfo.value)
    assert "automatic download disabled" in message
